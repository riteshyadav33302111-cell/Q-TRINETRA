"""Pauli algebra helpers and Pauli-frame correction (M0).

Conventions
-----------
* Bases are encoded as integers: X=0, Y=1, Z=2 (``BASES = ("X", "Y", "Z")``).
* A Pauli eigenstate is a pair ``(basis, value)`` with ``value ∈ {0,1}``:
  (X,0)=|+>, (X,1)=|->, (Y,0)=|+i>, (Y,1)=|-i>, (Z,0)=|0>, (Z,1)=|1>.
* Teleportation with Bell outcome ``(m1, m2)`` leaves the receiver with
  ``X^m2 Z^m1 |psi>``. A Pauli eigenstate stays a Pauli eigenstate under this
  correction; only its eigenvalue may flip. The flip rule is:

    X-basis: flip iff m1 == 1      (Z anticommutes with X)
    Z-basis: flip iff m2 == 1      (X anticommutes with Z)
    Y-basis: flip iff m1 ^ m2 == 1 (both X and Z anticommute with Y)

  This is the *teleportation lock*: without (m1, m2) the record is uniform.
"""
from __future__ import annotations

import numpy as np

BASES: tuple[str, str, str] = ("X", "Y", "Z")
X, Y, Z = 0, 1, 2

STATE_NAMES = {
    (X, 0): "|+>", (X, 1): "|->",
    (Y, 0): "|+i>", (Y, 1): "|-i>",
    (Z, 0): "|0>", (Z, 1): "|1>",
}

# Pauli-frame flip rules as in the idea document's code sketch.
FLIP = {
    "X": lambda m1, m2: m1,
    "Z": lambda m1, m2: m2,
    "Y": lambda m1, m2: m1 ^ m2,
}


def pauli_frame_correct(basis: str | int, raw_bit: int, m1: int, m2: int) -> int:
    """O(1) classical correction of a recorded measurement bit.

    ``basis`` may be "X"/"Y"/"Z" or 0/1/2.
    """
    if isinstance(basis, (int, np.integer)):
        basis = BASES[int(basis)]
    return int(raw_bit) ^ int(FLIP[basis](int(m1), int(m2)))


def frame_flip_vector(bases: np.ndarray, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Vectorised flip bits for arrays of bases (0/1/2) and Bell outcomes."""
    bases = np.asarray(bases)
    m1 = np.asarray(m1, dtype=np.uint8)
    m2 = np.asarray(m2, dtype=np.uint8)
    flip = np.where(bases == X, m1, np.where(bases == Z, m2, m1 ^ m2))
    return flip.astype(np.uint8)


def pauli_frame_correct_vector(bases: np.ndarray, raw_bits: np.ndarray,
                               m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Vectorised version of :func:`pauli_frame_correct`."""
    return (np.asarray(raw_bits, dtype=np.uint8) ^ frame_flip_vector(bases, m1, m2)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pauli channel action on eigenstates
# ---------------------------------------------------------------------------
# Pauli P applied to eigenstate of basis a flips the eigenvalue iff P anticommutes
# with the basis Pauli, i.e. iff P != I and P != a.
PAULI_I, PAULI_X, PAULI_Y, PAULI_Z = 0, 1, 2, 3
PAULI_NAMES = ("I", "X", "Y", "Z")


def pauli_flips_eigenvalue(pauli: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return 1 where applying ``pauli`` (0..3) to an eigenstate of ``basis`` (0..2)
    flips the eigenvalue. Pauli index p∈{1,2,3} corresponds to basis index p-1."""
    pauli = np.asarray(pauli)
    basis = np.asarray(basis)
    return ((pauli != PAULI_I) & (pauli != basis + 1)).astype(np.uint8)


def sample_pauli_channel(rng: np.random.Generator, n: int,
                         p_x: float, p_y: float, p_z: float) -> np.ndarray:
    """Sample n Pauli errors from channel (pI, pX, pY, pZ)."""
    p_i = 1.0 - p_x - p_y - p_z
    if p_i < -1e-12:
        raise ValueError("Pauli probabilities exceed 1")
    probs = np.clip(np.array([p_i, p_x, p_y, p_z]), 0.0, 1.0)
    probs /= probs.sum()
    return rng.choice(4, size=n, p=probs).astype(np.uint8)


def measure_eigenstate_in_basis(rng: np.random.Generator,
                                state_basis: np.ndarray, state_value: np.ndarray,
                                meas_basis: np.ndarray) -> np.ndarray:
    """Projective measurement of Pauli eigenstates in Pauli bases.

    If the measurement basis equals the state basis, the outcome is the
    eigenvalue deterministically; otherwise (mutually unbiased bases) it is a
    fair coin. This is exactly what a stabilizer simulator produces.
    """
    state_basis = np.asarray(state_basis)
    same = state_basis == np.asarray(meas_basis)
    coin = rng.integers(0, 2, size=state_basis.shape, dtype=np.uint8)
    return np.where(same, np.asarray(state_value, dtype=np.uint8), coin).astype(np.uint8)


def eigenstate_bloch(basis: int, value: int) -> tuple[float, float, float]:
    """Bloch vector of eigenstate for visualisation."""
    sign = 1.0 if value == 0 else -1.0
    vec = [0.0, 0.0, 0.0]
    vec[basis] = sign
    return tuple(vec)  # type: ignore[return-value]
