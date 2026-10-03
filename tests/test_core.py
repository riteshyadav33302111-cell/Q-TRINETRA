"""Property + unit tests for the Pauli core, MAC, teleportation engine and protocol."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qtrinetra.core import mac as wc
from qtrinetra.core.pauli import (BASES, pauli_frame_correct, pauli_frame_correct_vector,
                                  pauli_flips_eigenvalue)
from qtrinetra.engine.teleport import BellSource, PauliChannel, chsh_sample, teleport
from qtrinetra.protocol.qds import Alice, ProtocolParams, Signature, Verifier


# --- Pauli frame -----------------------------------------------------------
@given(st.sampled_from(BASES), st.integers(0, 1), st.integers(0, 1), st.integers(0, 1))
def test_frame_correction_is_involution(basis, bit, m1, m2):
    once = pauli_frame_correct(basis, bit, m1, m2)
    assert pauli_frame_correct(basis, once, m1, m2) == bit


def test_frame_rules_match_document():
    assert pauli_frame_correct("X", 0, 1, 0) == 1 and pauli_frame_correct("X", 0, 0, 1) == 0
    assert pauli_frame_correct("Z", 0, 0, 1) == 1 and pauli_frame_correct("Z", 0, 1, 0) == 0
    assert pauli_frame_correct("Y", 0, 1, 1) == 0 and pauli_frame_correct("Y", 0, 1, 0) == 1


def test_pauli_flip_table():
    # X on Z-eigenstate flips; Z on Z-eigenstate does not; Y flips everything but Y-basis.
    assert pauli_flips_eigenvalue(np.array([1]), np.array([2]))[0] == 1
    assert pauli_flips_eigenvalue(np.array([3]), np.array([2]))[0] == 0
    assert pauli_flips_eigenvalue(np.array([2]), np.array([1]))[0] == 0
    assert pauli_flips_eigenvalue(np.array([0]), np.array([0]))[0] == 0


# --- MAC -------------------------------------------------------------------
def test_wegman_carter_roundtrip_and_tamper():
    pool = wc.MacKeyPool.generate(4, seed=1)
    msg = wc.authenticate(pool, b"m1m2 stream")
    assert wc.check(pool, msg)
    bad = wc.AuthenticatedMessage(b"m1m2 strea_", msg.key_index, msg.tag)
    assert not wc.check(pool, bad)


# --- Teleportation engine ----------------------------------------------------
@pytest.mark.parametrize("backend", ["fast", "stim"])
def test_honest_teleportation_zero_mismatch(backend):
    """Deterministic acceptance: honest noiseless signature → M = 0 (both back-ends)."""
    rng = np.random.default_rng(0)
    n = 6000
    a = rng.integers(0, 3, n).astype(np.uint8)
    v = rng.integers(0, 2, n).astype(np.uint8)
    c = rng.integers(0, 3, n).astype(np.uint8)
    b = teleport(rng, a, v, c, backend=backend)
    corr = pauli_frame_correct_vector(c, b.raw_bits, b.m1, b.m2)
    same = a == c
    assert int((corr[same] != v[same]).sum()) == 0
    # different-basis positions are coins
    assert abs((corr[~same] == v[~same]).mean() - 0.5) < 0.03
    # Bell outcomes uniform
    counts = b.bell_outcome_counts()
    assert counts.min() > n / 4 * 0.85


@pytest.mark.parametrize("backend", ["fast", "stim"])
def test_dephasing_fingerprint(backend):
    rng = np.random.default_rng(1)
    n = 30000
    a = rng.integers(0, 3, n).astype(np.uint8)
    v = rng.integers(0, 2, n).astype(np.uint8)
    c = a.copy()  # check every position
    b = teleport(rng, a, v, c, PauliChannel.dephasing(0.5), backend=backend)
    corr = pauli_frame_correct_vector(c, b.raw_bits, b.m1, b.m2)
    e = [float((corr[a == k] != v[a == k]).mean()) for k in range(3)]
    assert abs(e[0] - 0.5) < 0.03 and abs(e[1] - 0.5) < 0.03 and e[2] < 0.01


def test_chsh_ideal_vs_fake():
    rng = np.random.default_rng(2)
    good = chsh_sample(rng, BellSource(1.0), 2000)
    fake = chsh_sample(rng, BellSource(0.0), 2000)
    assert good["S"] > 2.6 and good["fidelity"] > 0.95
    assert fake["S"] <= 2.1 and fake["fidelity"] < 0.6


# --- Protocol --------------------------------------------------------------
@settings(deadline=None, max_examples=15)
@given(st.integers(0, 10_000), st.integers(300, 3000))
def test_property_honest_signature_always_accepted_at_zero_noise(seed, L):
    rng = np.random.default_rng(seed)
    pool = wc.MacKeyPool.generate(4, seed=seed)
    p = ProtocolParams(L=L)
    alice, bob = Alice(p, rng, pool), Verifier("Bob", p, pool)
    keys = alice.keygen("kb")
    alice.distribute(keys[0], bob)
    v = bob.verify(alice.sign(b"msg", "kb", 0))
    assert v.decision == "ACCEPT" and v.mismatches == 0


def test_random_forger_rejected():
    rng = np.random.default_rng(5)
    pool = wc.MacKeyPool.generate(4, seed=5)
    p = ProtocolParams(L=3000)
    alice, bob = Alice(p, rng, pool), Verifier("Bob", p, pool)
    keys = alice.keygen("kb")
    alice.distribute(keys[1], bob)
    forged = Signature("kb", b"evil", 1, rng.integers(0, 3, 3000).astype(np.uint8),
                       rng.integers(0, 2, 3000).astype(np.uint8))
    v = bob.verify(forged)
    assert v.decision == "REJECT" and 0.4 < v.mismatch_rate < 0.6
