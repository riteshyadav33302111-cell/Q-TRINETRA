"""M4 — Attack Digital Twin: nine adversary models.

Each adversary is a *scenario* that perturbs one stage of the protocol:

    stage "source"      → BellSource with low fidelity / biased Bell outcomes
    stage "channel"     → PauliChannel applied to the receiver's qubit
    stage "classical"   → tampering with the correction-bit stream (MAC fails or bits flipped)
    stage "signature"   → the revealed key is replaced (forger / impersonator / replay)
    stage "verifier"    → verifier has no correction bits (unauthorized verification)

``apply_signature_attack`` returns the adversarial Signature; ``describe``
gives the expected fingerprint for the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qtrinetra.core.pauli import measure_eigenstate_in_basis
from qtrinetra.engine.teleport import BellSource, PauliChannel
from qtrinetra.protocol.qds import PrivateKey, Signature


@dataclass
class Scenario:
    name: str
    label: str
    stage: str
    description: str
    expected_fingerprint: str
    channel: PauliChannel = field(default_factory=PauliChannel)
    source: BellSource = field(default_factory=BellSource)
    bell_bias: np.ndarray | None = None
    tamper_corrections: bool = False
    no_corrections: bool = False
    signature_attack: str | None = None
    param: float = 0.0

    def as_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "stage": self.stage,
                "description": self.description, "expected_fingerprint": self.expected_fingerprint,
                "channel": self.channel.as_dict(), "source_fidelity": self.source.fidelity,
                "signature_attack": self.signature_attack, "param": self.param}


def honest(noise: PauliChannel | None = None) -> Scenario:
    return Scenario("honest", "Honest signature", "none",
                    "Alice signs with her genuine private key over a calibrated channel.",
                    "M = 0 (noiseless) → ACCEPT; Pauli radar flat.", channel=noise or PauliChannel())


def build_scenario(name: str, param: float | None = None) -> Scenario:
    """Factory for the 9 adversaries (+ benign noise + honest)."""
    if name == "honest":
        return honest()
    if name == "benign_noise":
        p = 0.02 if param is None else param
        s = honest(PauliChannel.depolarizing(p))
        s.name, s.label = "benign_noise", f"Benign depolarizing noise (p={p})"
        s.description = "Ordinary fibre noise: small isotropic error mass that matches the calibrated baseline."
        s.expected_fingerprint = "eX≈eY≈eZ≈2p/3, matches baseline → ACCEPT, no alarm."
        s.param = p
        return s
    if name == "random_forger":
        return Scenario(name, "Random forger", "signature",
                        "Adversary without any key reveals uniformly random (a,v) pairs.",
                        "mismatch ≈ ½ on checked positions → D1 alarm, class FORGERY.",
                        signature_attack="random")
    if name == "measure_guess_forger":
        return Scenario(name, "Measure-and-guess forger", "signature",
                        "Adversary intercepts the public-key qubits, measures each in a random basis and "
                        "reveals the measured (basis, outcome) as the key.",
                        "mismatch = ⅓ (wrong basis ⅔ of the time, then coin) → D1 alarm; D2 isotropic.",
                        signature_attack="measure_guess")
    if name == "cloning_forger":
        return Scenario(name, "Optimal universal cloning forger", "signature",
                        "Adversary 1→2 universally clones each qubit (fidelity 5/6), forwards one clone and "
                        "measures the other in a random basis to guess the key.",
                        "mismatch ≈ ⅙·(1/3)+... ≈ 0.28 overall; channel fingerprint ≈ ⅙ each (isotropic medium).",
                        channel=PauliChannel.depolarizing(0.25),   # clone fidelity 5/6 → depolarizing p=1/4
                        signature_attack="cloning")
    if name == "impersonator":
        return Scenario(name, "Impersonator (no key)", "signature",
                        "Someone claims to be Alice and signs a different message with a guessed key for the "
                        "same key block.",
                        "mismatch ≈ ⅓–½ → D1 alarm; MAC on corrections still valid.",
                        signature_attack="impersonate")
    if name == "replay_same_block":
        return Scenario(name, "Replay attacker (same block)", "signature",
                        "A previously accepted signature is re-submitted verbatim.",
                        "D5 Q-Ledger spent-key hit → deterministic REPLAY verdict.",
                        signature_attack="replay")
    if name == "replay_cross_block":
        return Scenario(name, "Replay attacker (cross block)", "signature",
                        "An old revealed key is replayed against a fresh key block.",
                        "≈ 50 % mismatch → D1 alarm (class forgery/cross-block replay).",
                        signature_attack="cross_block")
    if name == "unauthorized_verifier":
        return Scenario(name, "Unauthorized verifier (no correction bits)", "verifier",
                        "A party without the authenticated (m1,m2) stream tries to verify.",
                        "Teleportation lock: record uniformly random → ≈ 50 % mismatch, cannot verify.",
                        no_corrections=True)
    if name.startswith("intercept_resend"):
        basis = name.split("_")[-1].upper() if "_" in name[len("intercept_resend"):] else "Z"
        if basis == "Z":
            ch, fp = PauliChannel(0, 0, 0.5), "eX=eY=½, eZ=0 → pZ=½ (full dephasing)"
        elif basis == "X":
            ch, fp = PauliChannel(0.5, 0, 0), "eY=eZ=½, eX=0 → pX=½"
        elif basis == "Y":
            ch, fp = PauliChannel(0, 0.5, 0), "eX=eZ=½, eY=0 → pY=½"
        else:  # random basis: each basis measured w.p. 1/3 → average of the three
            ch, fp = PauliChannel(1 / 6, 1 / 6, 1 / 6), "eX=eY=eZ=⅓ → isotropic large"
            basis = "RANDOM"
        return Scenario(name, f"Intercept–resend in {basis} basis", "channel",
                        f"Eve measures every qubit in the {basis} basis and re-prepares the outcome.",
                        fp, channel=ch, param=0.5)
    if name == "bitflip_injection":
        q = 0.15 if param is None else param
        return Scenario(name, f"Bit-flip injection (q={q})", "channel",
                        "Eve applies X with probability q to each qubit in flight.",
                        f"eX=0, eY=eZ={q} → pX={q}", channel=PauliChannel.bitflip(q), param=q)
    if name == "fake_source":
        f = 0.0 if param is None else param
        return Scenario(name, "Fake Bell source (separable states)", "source",
                        "The entanglement source is replaced by classically correlated |00>/|11> pairs.",
                        "CHSH S ≤ 2, fidelity F ≈ 0.5 → D4 alarm, distribution blocked. Bell outcomes stay uniform (D3 quiet).",
                        source=BellSource(fidelity=f), channel=PauliChannel(0.25, 0.25, 0.0), param=f)
    if name == "biased_source":
        return Scenario(name, "Biased Bell-measurement device", "source",
                        "The Bell analyser is rigged so outcome 00 is over-reported.",
                        "D3 χ² non-uniform; teleportation still works → D1 quiet.",
                        bell_bias=np.array([0.55, 0.15, 0.15, 0.15]))
    if name == "correction_tamperer":
        return Scenario(name, "Correction-bit tamperer", "classical",
                        "Mallory flips bits in the (m1,m2) stream on the classical channel.",
                        "Wegman–Carter MAC fails → D3/R2 alarm; without MAC the record would be ≈ 50 % wrong.",
                        tamper_corrections=True)
    raise ValueError(f"unknown scenario {name!r}")


SCENARIO_NAMES = [
    "honest", "benign_noise", "random_forger", "measure_guess_forger", "cloning_forger",
    "impersonator", "replay_same_block", "replay_cross_block", "unauthorized_verifier",
    "intercept_resend_Z", "intercept_resend_X", "intercept_resend_Y", "intercept_resend_random",
    "bitflip_injection", "fake_source", "biased_source", "correction_tamperer",
]


def forge_signature(rng: np.random.Generator, kind: str, true_key: PrivateKey, message: bytes,
                    other_key: PrivateKey | None = None) -> Signature:
    """Produce the adversarial revealed key for signature-stage attacks."""
    L = true_key.L
    if kind in ("random", "impersonate"):
        bases = rng.integers(0, 3, L).astype(np.uint8)
        values = rng.integers(0, 2, L).astype(np.uint8)
    elif kind == "measure_guess":
        bases = rng.integers(0, 3, L).astype(np.uint8)
        values = measure_eigenstate_in_basis(rng, true_key.bases, true_key.values, bases)
    elif kind == "cloning":
        # Clone measured in random basis; clone has fidelity 5/6 → outcome correct w.p. 5/6 when
        # the basis matches, coin otherwise.
        bases = rng.integers(0, 3, L).astype(np.uint8)
        values = measure_eigenstate_in_basis(rng, true_key.bases, true_key.values, bases)
        same = bases == true_key.bases
        flip = (rng.random(L) < 1 / 6) & same
        values = (values ^ flip.astype(np.uint8)).astype(np.uint8)
    elif kind == "cross_block":
        assert other_key is not None
        bases, values = other_key.bases.copy(), other_key.values.copy()
    else:
        raise ValueError(kind)
    return Signature(true_key.key_block_id, message, true_key.message_bit, bases, values)
