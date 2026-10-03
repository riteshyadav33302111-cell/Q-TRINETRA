"""Detector, SPRT, fusion, ledger and end-to-end scenario regression tests."""
import numpy as np
import pytest

from qtrinetra.detect.d1_mismatch import hoeffding_forgery_bound, hoeffding_honest_reject_bound, required_Lc
from qtrinetra.detect.d2_pauli_inversion import expected_error_rates, pauli_channel_inversion
from qtrinetra.detect.d3_bell_uniformity import bell_uniformity_pvalue
from qtrinetra.detect.fusion import decision_table_export, fisher_combine
from qtrinetra.detect.sprt import sprt_verify
from qtrinetra.engine.teleport import PauliChannel
from qtrinetra.ledger.qledger import QLedger, merkle_proof, merkle_root, verify_merkle_proof
from qtrinetra.pipeline import run_session


def test_pauli_inversion_is_exact_inverse_of_forward_model():
    ch = PauliChannel(0.1, 0.05, 0.2)
    eX, eY, eZ = expected_error_rates(ch)
    rec = pauli_channel_inversion(eX, eY, eZ)
    assert rec["X"] == pytest.approx(0.1) and rec["Y"] == pytest.approx(0.05)
    assert rec["Z"] == pytest.approx(0.2) and rec["I"] == pytest.approx(0.65)


def test_document_fingerprints():
    assert pauli_channel_inversion(0.5, 0.5, 0.0)["Z"] == pytest.approx(0.5)      # Z intercept-resend
    assert pauli_channel_inversion(0.0, 0.2, 0.2)["X"] == pytest.approx(0.2)      # bit-flip q
    iso = pauli_channel_inversion(1 / 3, 1 / 3, 1 / 3)
    assert iso["X"] == pytest.approx(iso["Y"]) == pytest.approx(1 / 6)


def test_hoeffding_worked_example():
    """Document: e0=0.01, s_a=0.08, p_min=1/6, Lc≈1600 → forgery ≤ 1e-10, honest reject ≈ 1e-5."""
    Lc = 1600
    assert hoeffding_forgery_bound(Lc, 1 / 6, 0.08) < 1e-10 * 1.5
    assert hoeffding_honest_reject_bound(Lc, 0.01, 0.08) < 2e-5
    assert 1500 < required_Lc(1 / 6, 0.08, 1e-10) < 1700


def test_sprt_decisions_and_early_stop():
    rng = np.random.default_rng(0)
    honest = sprt_verify(rng.random(1600) < 0.01, 0.01, 1 / 6)
    forger = sprt_verify(rng.random(1600) < 1 / 3, 0.01, 1 / 6)
    assert honest.decision == "ACCEPT" and honest.n_used < 600
    assert forger.decision == "REJECT" and forger.n_used < 150


def test_bell_uniformity():
    assert bell_uniformity_pvalue([250, 250, 250, 250])[1] == pytest.approx(1.0)
    assert bell_uniformity_pvalue([700, 100, 100, 100])[1] < 1e-10


def test_fisher():
    stat, p = fisher_combine([1.0, 1.0, 1.0])
    assert stat == pytest.approx(0.0) and p == pytest.approx(1.0)
    _, p2 = fisher_combine([1e-10, 0.5, 0.5])
    assert p2 < 1e-6


def test_decision_table_exported_and_has_default():
    rows = decision_table_export()
    assert rows[-1]["id"] == "R0" and len(rows) > 10


# --- Ledger --------------------------------------------------------------------
def test_ledger_chain_merkle_and_replay():
    led = QLedger(batch_size=4)
    for i in range(8):
        led.append(f"n{i}", f"kb{i}", 0, "h", "ACCEPT")
    assert led.verify_chain() and len(led.batches()) == 2
    proof = led.merkle_proof(5)
    assert proof["anchored"] and proof["valid"]
    assert led.is_spent("kb3", 0) and not led.is_spent("kb3", 1)
    # tamper detection
    led.conn.execute("UPDATE records SET verdict='REJECT' WHERE idx=2")
    assert not led.verify_chain()


def test_merkle_helpers():
    leaves = [f"{i:064x}" for i in range(5)]
    root = merkle_root(leaves)
    for i in range(5):
        assert verify_merkle_proof(leaves[i], merkle_proof(leaves, i), root)


# --- End-to-end scenario regression ---------------------------------------------
EXPECTED = {
    "honest": "NONE (HONEST)",
    "benign_noise": "NONE (HONEST)",
    "random_forger": "FORGERY",
    "measure_guess_forger": "MEASURE-AND-GUESS",
    "impersonator": "FORGERY",
    "replay_same_block": "REPLAY ATTACK",
    "replay_cross_block": "FORGERY",
    "unauthorized_verifier": "UNAUTHORIZED VERIFICATION",
    "intercept_resend_Z": "Z-BASIS",
    "intercept_resend_X": "X-BASIS",
    "intercept_resend_Y": "Y-BASIS",
    "intercept_resend_random": "RANDOM-BASIS",
    "bitflip_injection": "BIT-FLIP",
    "fake_source": "FAKE / DEGRADED ENTANGLEMENT",
    "biased_source": "BIASED SOURCE",
    "correction_tamperer": "CORRECTION-BIT TAMPERING",
}


@pytest.mark.parametrize("scenario,expected", list(EXPECTED.items()))
@pytest.mark.parametrize("seed", [1, 2])
def test_scenario_classification(scenario, expected, seed):
    r = run_session(scenario, seed=seed)
    assert expected in r.report.attack_class, r.report.attack_class
    if scenario in ("honest", "benign_noise"):
        assert r.verdict_bob.decision == "ACCEPT" and r.report.threat_index < 0.999
    elif scenario != "fake_source":
        assert r.report.threat_index > 0.99


def test_fake_source_blocks_distribution():
    r = run_session("fake_source", seed=1)
    assert r.distribution_blocked and r.signature is None


def test_stim_backend_end_to_end():
    from qtrinetra.protocol.qds import ProtocolParams
    r = run_session("intercept_resend_Z", seed=1, params=ProtocolParams(L=3000, backend="stim"))
    assert "Z-BASIS" in r.report.attack_class
    r = run_session("honest", seed=1, params=ProtocolParams(L=3000, backend="stim"))
    assert r.report.attack_class == "NONE (HONEST)"
