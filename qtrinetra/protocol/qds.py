"""M2 — Teleportation-based QDS protocol core.

KeyGen → Distribute (teleportation + Pauli-frame record) → Sign → Verify → Transfer.

The verifier never stores qubits: each arriving qubit is measured immediately
in a random basis ``c_i`` and the recorded bit is corrected classically once
the MAC-authenticated correction bits ``(m1, m2)`` arrive
(:func:`qtrinetra.core.pauli.pauli_frame_correct_vector`).
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field

import numpy as np

from qtrinetra.core import mac as wc
from qtrinetra.core.pauli import BASES, pauli_frame_correct_vector
from qtrinetra.engine.teleport import PauliChannel, TeleportBatch, teleport


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PrivateKey:
    """Alice's private key for one message bit: L pairs (a_i, v_i)."""
    key_block_id: str
    message_bit: int
    bases: np.ndarray      # a_i ∈ {0,1,2}
    values: np.ndarray     # v_i ∈ {0,1}

    @property
    def L(self) -> int:
        return int(self.bases.shape[0])

    def public_states(self) -> list[str]:
        from qtrinetra.core.pauli import STATE_NAMES
        return [STATE_NAMES[(int(a), int(v))] for a, v in zip(self.bases, self.values)]


@dataclass
class VerifierRecord:
    """What a verifier holds after distribution of one key block."""
    key_block_id: str
    message_bit: int
    meas_basis: np.ndarray           # c_i
    raw_bits: np.ndarray             # bits recorded on arrival
    corrected_bits: np.ndarray | None = None   # after Pauli-frame correction
    m1: np.ndarray | None = None
    m2: np.ndarray | None = None
    corrections_authentic: bool = True
    batch: TeleportBatch | None = None          # kept for detector statistics

    @property
    def L(self) -> int:
        return int(self.meas_basis.shape[0])


@dataclass
class Signature:
    """Alice's signature on message ``message`` with bit ``message_bit``:
    she reveals the private key for that bit."""
    key_block_id: str
    message: bytes
    message_bit: int
    bases: np.ndarray
    values: np.ndarray
    nonce: str = field(default_factory=lambda: secrets.token_hex(8))

    def message_hash(self) -> str:
        return hashlib.sha3_256(self.message).hexdigest()


@dataclass
class Verdict:
    decision: str                # ACCEPT | TRANSFERABLE_UNCERTAIN | REJECT
    checked: int                 # Lc
    mismatches: int              # M
    mismatch_rate: float
    per_basis_checked: np.ndarray    # [nX, nY, nZ]
    per_basis_mismatch: np.ndarray   # [mX, mY, mZ]
    s_a: float
    s_v: float

    @property
    def basis_error_rates(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.per_basis_checked > 0,
                            self.per_basis_mismatch / np.maximum(self.per_basis_checked, 1), 0.0)

    def as_dict(self) -> dict:
        e = self.basis_error_rates
        return {
            "decision": self.decision, "checked": self.checked, "mismatches": self.mismatches,
            "mismatch_rate": self.mismatch_rate,
            "per_basis_checked": self.per_basis_checked.tolist(),
            "per_basis_mismatch": self.per_basis_mismatch.tolist(),
            "eX": float(e[0]), "eY": float(e[1]), "eZ": float(e[2]),
            "s_a": self.s_a, "s_v": self.s_v,
        }


@dataclass
class ProtocolParams:
    L: int = 4800               # qubits per key block (≈1600 checked)
    s_a: float = 0.08           # acceptance threshold (fraction of Lc)
    s_v: float = 0.14           # verification / transferability threshold
    e0: float = 0.01            # expected honest channel error
    p_min: float = 1.0 / 6.0    # conservative worst-case forger mismatch rate
    backend: str = "fast"       # "fast" | "stim"


# ---------------------------------------------------------------------------
# Protocol steps
# ---------------------------------------------------------------------------
class Alice:
    """Signer."""

    def __init__(self, params: ProtocolParams, rng: np.random.Generator, mac_pool: wc.MacKeyPool):
        self.params = params
        self.rng = rng
        self.mac_pool = mac_pool
        self.keys: dict[tuple[str, int], PrivateKey] = {}

    def keygen(self, key_block_id: str | None = None) -> dict[int, PrivateKey]:
        """Generate private keys for both message bits for one key block."""
        key_block_id = key_block_id or secrets.token_hex(6)
        out = {}
        for b in (0, 1):
            k = PrivateKey(key_block_id, b,
                           self.rng.integers(0, 3, self.params.L).astype(np.uint8),
                           self.rng.integers(0, 2, self.params.L).astype(np.uint8))
            self.keys[(key_block_id, b)] = k
            out[b] = k
        return out

    def distribute(self, key: PrivateKey, verifier: "Verifier",
                   channel: PauliChannel | None = None,
                   bell_bias: np.ndarray | None = None) -> VerifierRecord:
        """Teleport the public-key eigenstates to ``verifier`` and send the
        MAC-authenticated correction bits over the classical channel."""
        c = self.rng.integers(0, 3, key.L).astype(np.uint8)   # verifier's random bases
        batch = teleport(self.rng, key.bases, key.values, c, channel,
                         backend=self.params.backend, bell_bias=bell_bias)
        rec = VerifierRecord(key.key_block_id, key.message_bit, c, batch.raw_bits, batch=batch)
        payload = np.packbits(np.concatenate([batch.m1, batch.m2])).tobytes()
        auth = wc.authenticate(self.mac_pool, key.key_block_id.encode() + payload)
        verifier.receive_corrections(rec, batch.m1, batch.m2, auth)
        return rec

    def sign(self, message: bytes, key_block_id: str, message_bit: int) -> Signature:
        k = self.keys[(key_block_id, message_bit)]
        return Signature(key_block_id, message, message_bit, k.bases.copy(), k.values.copy())


class Verifier:
    """Bob / Charlie."""

    def __init__(self, name: str, params: ProtocolParams, mac_pool: wc.MacKeyPool):
        self.name = name
        self.params = params
        self.mac_pool = mac_pool
        self.records: dict[tuple[str, int], VerifierRecord] = {}

    def receive_corrections(self, rec: VerifierRecord, m1: np.ndarray, m2: np.ndarray,
                            auth: wc.AuthenticatedMessage) -> None:
        rec.corrections_authentic = wc.check(self.mac_pool, auth)
        rec.m1, rec.m2 = m1, m2
        if rec.corrections_authentic:
            rec.corrected_bits = pauli_frame_correct_vector(rec.meas_basis, rec.raw_bits, m1, m2)
        else:
            # Teleportation lock: without trusted corrections the record is useless.
            rec.corrected_bits = rec.raw_bits.copy()
        self.records[(rec.key_block_id, rec.message_bit)] = rec

    def verify(self, sig: Signature, positions: np.ndarray | None = None) -> Verdict:
        rec = self.records[(sig.key_block_id, sig.message_bit)]
        return verify_record(rec, sig, self.params, positions)


def verify_record(rec: VerifierRecord, sig: Signature, params: ProtocolParams,
                  positions: np.ndarray | None = None) -> Verdict:
    """Core verification: compare revealed key with the corrected record on
    positions where c_i == a_i.  Optional ``positions`` restricts to the
    verifier's own half after symmetrization."""
    assert rec.corrected_bits is not None, "corrections not received"
    idx = np.arange(rec.L) if positions is None else positions
    same = rec.meas_basis[idx] == sig.bases[idx]
    chk = idx[same]
    mism = rec.corrected_bits[chk] != sig.values[chk]
    Lc, M = int(chk.size), int(mism.sum())
    per_chk = np.bincount(rec.meas_basis[chk], minlength=3)
    per_mis = np.bincount(rec.meas_basis[chk][mism], minlength=3)
    rate = M / Lc if Lc else 0.0
    if M <= params.s_a * Lc:
        d = "ACCEPT"
    elif M <= params.s_v * Lc:
        d = "TRANSFERABLE_UNCERTAIN"
    else:
        d = "REJECT"
    return Verdict(d, Lc, M, rate, per_chk, per_mis, params.s_a, params.s_v)


def mismatch_stream(rec: VerifierRecord, sig: Signature) -> np.ndarray:
    """Ordered 0/1 mismatch stream over checked positions (input to SPRT)."""
    same = rec.meas_basis == sig.bases
    return (rec.corrected_bits[same] != sig.values[same]).astype(np.uint8)


def symmetrize(rng: np.random.Generator, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Bob and Charlie secretly swap a random half of positions (Amiri et al.).
    Returns index arrays (bob_positions, charlie_positions) that each party
    uses for its own verification — a cheating Alice cannot target one of them."""
    perm = rng.permutation(L)
    half = L // 2
    return np.sort(perm[:half]), np.sort(perm[half:])


def basis_names(arr: np.ndarray) -> list[str]:
    return [BASES[int(a)] for a in arr]
