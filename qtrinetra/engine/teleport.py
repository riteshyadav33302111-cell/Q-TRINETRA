"""M1 — Entanglement & Teleportation Engine.

Two interchangeable back-ends that produce identical statistics:

``stim`` back-end
    Builds the genuine teleportation circuit for every qubit (prepare Pauli
    eigenstate, Bell pair, optional Pauli channel on the receiver's half,
    Bell measurement, receiver measurement in a chosen basis) and samples it
    with Stim's stabilizer simulator (Gottesman–Knill).  Because the whole QDS
    protocol is Clifford this is exact.

``fast`` back-end
    Closed-form of what the stabilizer simulation produces: the receiver's
    raw bit is ``v ^ flip(c, m1, m2) ^ pauli_flip`` if ``c == a`` and a fair
    coin otherwise; Bell outcomes are uniform.  Millions of qubits / second in
    NumPy.  Tests assert both back-ends agree.

Both return a :class:`TeleportBatch`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qtrinetra.core.pauli import (X, Y, Z, frame_flip_vector, pauli_flips_eigenvalue,
                                  sample_pauli_channel)

try:  # Stim is optional at import time so the fast path always works.
    import stim  # type: ignore
    HAS_STIM = True
except Exception:  # pragma: no cover
    stim = None
    HAS_STIM = False


@dataclass
class PauliChannel:
    """Pauli channel acting on the receiver's half of each Bell pair."""
    p_x: float = 0.0
    p_y: float = 0.0
    p_z: float = 0.0

    @property
    def p_i(self) -> float:
        return 1.0 - self.p_x - self.p_y - self.p_z

    @classmethod
    def depolarizing(cls, p: float) -> "PauliChannel":
        return cls(p / 3, p / 3, p / 3)

    @classmethod
    def dephasing(cls, p: float) -> "PauliChannel":
        return cls(0.0, 0.0, p)

    @classmethod
    def bitflip(cls, p: float) -> "PauliChannel":
        return cls(p, 0.0, 0.0)

    @classmethod
    def amplitude_damping_twirled(cls, gamma: float) -> "PauliChannel":
        """Pauli-twirled amplitude damping: pX=pY=gamma/4, pZ=(1-sqrt(1-gamma))/2 - gamma/4."""
        pz = (1 - np.sqrt(1 - gamma)) / 2 - gamma / 4
        return cls(gamma / 4, gamma / 4, max(pz, 0.0))

    def as_dict(self) -> dict[str, float]:
        return {"I": self.p_i, "X": self.p_x, "Y": self.p_y, "Z": self.p_z}


@dataclass
class TeleportBatch:
    """Result of teleporting ``n`` Pauli eigenstates to a verifier."""
    state_basis: np.ndarray     # a_i  ∈ {0,1,2}
    state_value: np.ndarray     # v_i  ∈ {0,1}
    m1: np.ndarray              # Bell outcome bit 1 (Z-type correction)
    m2: np.ndarray              # Bell outcome bit 2 (X-type correction)
    meas_basis: np.ndarray      # c_i ∈ {0,1,2}   verifier's random basis
    raw_bits: np.ndarray        # verifier's recorded bit BEFORE frame correction
    backend: str = "fast"

    @property
    def n(self) -> int:
        return int(self.state_basis.shape[0])

    def bell_outcome_counts(self) -> np.ndarray:
        idx = (self.m1.astype(int) << 1) | self.m2.astype(int)
        return np.bincount(idx, minlength=4)


# ---------------------------------------------------------------------------
# fast back-end
# ---------------------------------------------------------------------------
def teleport_fast(rng: np.random.Generator, state_basis: np.ndarray, state_value: np.ndarray,
                  meas_basis: np.ndarray, channel: PauliChannel | None = None,
                  bell_bias: np.ndarray | None = None) -> TeleportBatch:
    n = state_basis.shape[0]
    if bell_bias is None:
        m1 = rng.integers(0, 2, n, dtype=np.uint8)
        m2 = rng.integers(0, 2, n, dtype=np.uint8)
    else:  # biased source / tampered correction stream (for D3 testing)
        idx = rng.choice(4, size=n, p=np.asarray(bell_bias) / np.sum(bell_bias))
        m1 = (idx >> 1).astype(np.uint8)
        m2 = (idx & 1).astype(np.uint8)
    flip = frame_flip_vector(state_basis, m1, m2)
    if channel is not None and (channel.p_x or channel.p_y or channel.p_z):
        err = sample_pauli_channel(rng, n, channel.p_x, channel.p_y, channel.p_z)
        flip = flip ^ pauli_flips_eigenvalue(err, state_basis)
    same = state_basis == meas_basis
    coin = rng.integers(0, 2, n, dtype=np.uint8)
    raw = np.where(same, state_value.astype(np.uint8) ^ flip, coin).astype(np.uint8)
    return TeleportBatch(state_basis, state_value, m1, m2, meas_basis, raw, "fast")


# ---------------------------------------------------------------------------
# stim back-end
# ---------------------------------------------------------------------------
def _prep_ops(basis: int, value: int, q: int) -> list[str]:
    ops: list[str] = []
    if value:
        ops.append(f"X {q}")          # |1>
    if basis == X:
        ops.append(f"H {q}")          # |+> / |->
    elif basis == Y:
        ops.append(f"H {q}")
        ops.append(f"S {q}")          # |+i> / |-i>
    return ops


def _meas_ops(basis: int, q: int) -> str:
    return {X: f"MX {q}", Y: f"MY {q}", Z: f"M {q}"}[basis]


def _teleport_circuit(a: int, v: int, c: int, channel: PauliChannel | None) -> "stim.Circuit":
    """3-qubit teleportation circuit for eigenstate (a, v) measured by Bob in basis c.

    Qubits: 0 = message, 1 = Alice's half, 2 = Bob's half.
    Measurement record: [m2 (X-correction, from q1), m1 (Z-correction, from q0), Bob's bit].
    """
    lines = ["R 0 1 2"]
    lines += _prep_ops(a, v, 0)
    lines += ["H 1", "CX 1 2"]                                   # |Phi+>
    if channel is not None and (channel.p_x or channel.p_y or channel.p_z):
        lines.append(f"PAULI_CHANNEL_1({channel.p_x},{channel.p_y},{channel.p_z}) 2")
    lines += ["CX 0 1", "H 0", "M 1", "M 0", _meas_ops(c, 2)]   # Bell measurement + Bob
    return stim.Circuit("\n".join(lines))


def teleport_stim(rng: np.random.Generator, state_basis: np.ndarray, state_value: np.ndarray,
                  meas_basis: np.ndarray, channel: PauliChannel | None = None) -> TeleportBatch:
    """Simulate every teleportation with a genuine Stim circuit.

    Teleportations are independent, so the batch is grouped by the 18
    (a, v, c) combinations and each group is sampled with ``shots = count``
    from the same 3-qubit circuit — exact and fast.
    """
    if not HAS_STIM:  # pragma: no cover
        raise RuntimeError("stim is not installed")
    n = state_basis.shape[0]
    m1 = np.zeros(n, dtype=np.uint8)
    m2 = np.zeros(n, dtype=np.uint8)
    raw = np.zeros(n, dtype=np.uint8)
    combo = state_basis.astype(int) * 6 + state_value.astype(int) * 3 + meas_basis.astype(int)
    for key in np.unique(combo):
        idx = np.flatnonzero(combo == key)
        a, rem = divmod(int(key), 6)
        v, c = divmod(rem, 3)
        circ = _teleport_circuit(a, v, c, channel)
        sampler = circ.compile_sampler(seed=int(rng.integers(0, 2**31 - 1)))
        rec = sampler.sample(shots=len(idx)).astype(np.uint8)
        m2[idx], m1[idx], raw[idx] = rec[:, 0], rec[:, 1], rec[:, 2]
    return TeleportBatch(state_basis, state_value, m1, m2, meas_basis, raw, "stim")


def teleport(rng: np.random.Generator, state_basis: np.ndarray, state_value: np.ndarray,
             meas_basis: np.ndarray, channel: PauliChannel | None = None,
             backend: str = "fast", bell_bias: np.ndarray | None = None) -> TeleportBatch:
    if backend == "stim":
        if bell_bias is not None:
            raise ValueError("bell_bias is only supported by the fast back-end")
        return teleport_stim(rng, state_basis, state_value, meas_basis, channel)
    return teleport_fast(rng, state_basis, state_value, meas_basis, channel, bell_bias)


# ---------------------------------------------------------------------------
# CHSH / Bell-state fidelity sampler (D4 resource audit)
# ---------------------------------------------------------------------------
@dataclass
class BellSource:
    """Model of the entanglement source.

    ``fidelity`` is the fraction of ideal |Phi+> pairs; the remainder is the
    separable mixture ½(|00><00| + |11><11|) (a classical correlated source).
    This keeps Bell outcomes uniform (so D3 stays quiet) but CHSH <= 2 and
    <XX> = <YY> = 0 so fidelity F drops to (1 + <ZZ>)/4 = 0.5 for a fully fake
    source — exactly the D4 fingerprint of "fake entanglement".
    """
    fidelity: float = 1.0
    visibility: float = 1.0   # extra depolarising visibility factor on correlations

    def correlators(self) -> tuple[float, float, float]:
        f, v = self.fidelity, self.visibility
        xx = v * f
        yy = -v * f
        zz = v * (f + (1 - f))  # separable part still has perfect ZZ correlation
        return xx, yy, zz

    def sample_correlation(self, rng: np.random.Generator, n: int,
                           a: tuple[float, float, float], b: tuple[float, float, float]) -> np.ndarray:
        """Sample n products a·b ∈ {+1,-1} for measurement directions a, b on the Bloch sphere.
        For the model above E(a,b) = Σ_k T_kk a_k b_k with diagonal correlation matrix T."""
        xx, yy, zz = self.correlators()
        e = xx * a[0] * b[0] + yy * a[1] * b[1] + zz * a[2] * b[2]
        e = float(np.clip(e, -1, 1))
        p_plus = (1 + e) / 2
        return np.where(rng.random(n) < p_plus, 1, -1)


def chsh_sample(rng: np.random.Generator, source: BellSource, n_per_setting: int) -> dict:
    """Estimate CHSH S and Bell-state fidelity by sacrificing 4*n + 3*n pairs.

    Settings (optimal for |Phi+> whose correlator is +XX, -YY, +ZZ):
    a0 = Z, a1 = X ; b0 = (Z+X)/√2, b1 = (Z−X)/√2.
    S = E(a0,b0) + E(a0,b1) + E(a1,b0) − E(a1,b1) → 2√2 for ideal pairs.
    """
    s2 = 1 / np.sqrt(2)
    a0, a1 = (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
    b0, b1 = (s2, 0.0, s2), (-s2, 0.0, s2)
    E = {}
    for name, (a, b) in {"00": (a0, b0), "01": (a0, b1), "10": (a1, b0), "11": (a1, b1)}.items():
        E[name] = chsh_sample_mean(source.sample_correlation(rng, n_per_setting, a, b))
    S = E["00"] + E["01"] + E["10"] - E["11"]
    # Each E has variance <= 1/n; S is a sum of 4 independent means.
    se_S = 2.0 / np.sqrt(n_per_setting)
    xx = chsh_sample_mean(source.sample_correlation(rng, n_per_setting, (1, 0, 0), (1, 0, 0)))
    yy = chsh_sample_mean(source.sample_correlation(rng, n_per_setting, (0, 1, 0), (0, 1, 0)))
    zz = chsh_sample_mean(source.sample_correlation(rng, n_per_setting, (0, 0, 1), (0, 0, 1)))
    F = (1 + xx - yy + zz) / 4
    return {"S": float(S), "se_S": float(se_S), "E": {k: float(v) for k, v in E.items()},
            "XX": float(xx), "YY": float(yy), "ZZ": float(zz), "fidelity": float(F),
            "pairs_sacrificed": 7 * n_per_setting}


def chsh_sample_mean(x: np.ndarray) -> float:
    return float(np.mean(x))
