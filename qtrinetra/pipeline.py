"""End-to-end session: source audit → distribution → signing → verification →
five detectors → SPRT → fusion → Q-Ledger.

    from qtrinetra.pipeline import run_session
    result = run_session("intercept_resend_Z", seed=1)
    print(result.report.attack_class)
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field

import numpy as np

from qtrinetra.attacks.adversaries import Scenario, build_scenario, forge_signature
from qtrinetra.core import mac as wc
from qtrinetra.core.pauli import pauli_frame_correct_vector
from qtrinetra.detect.base import DetectorResult
from qtrinetra.detect.d1_mismatch import detect_d1
from qtrinetra.detect.d2_pauli_inversion import detect_d2
from qtrinetra.detect.d3_bell_uniformity import detect_d3
from qtrinetra.detect.d4_chsh import detect_d4
from qtrinetra.detect.d5_replay import detect_d5
from qtrinetra.detect.fusion import ThreatReport, fuse
from qtrinetra.detect.sprt import SPRTResult, sprt_verify
from qtrinetra.engine.teleport import PauliChannel, chsh_sample, teleport
from qtrinetra.ledger.qledger import QLedger
from qtrinetra.protocol.qds import (Alice, ProtocolParams, Signature, Verdict, Verifier,
                                    VerifierRecord, mismatch_stream, symmetrize, verify_record)


@dataclass
class SessionResult:
    scenario: Scenario
    params: ProtocolParams
    session_nonce: str
    report: ThreatReport
    verdict_bob: Verdict | None
    verdict_charlie: Verdict | None
    sprt: SPRTResult | None
    signature: Signature | None
    distribution_blocked: bool
    ledger_record: dict | None
    timings_ms: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "scenario": self.scenario.as_dict(),
            "params": vars(self.params),
            "session_nonce": self.session_nonce,
            "report": self.report.as_dict(),
            "verdict_bob": self.verdict_bob.as_dict() if self.verdict_bob else None,
            "verdict_charlie": self.verdict_charlie.as_dict() if self.verdict_charlie else None,
            "sprt": self.sprt.as_dict() if self.sprt else None,
            "signature": ({"key_block_id": self.signature.key_block_id,
                           "message": self.signature.message.decode(errors="replace"),
                           "message_bit": self.signature.message_bit,
                           "message_hash": self.signature.message_hash(),
                           "nonce": self.signature.nonce,
                           "revealed_key_preview": self.signature.bases[:24].tolist(),
                           "revealed_values_preview": self.signature.values[:24].tolist()}
                          if self.signature else None),
            "distribution_blocked": self.distribution_blocked,
            "ledger_record": self.ledger_record,
            "timings_ms": self.timings_ms,
        }


def _distribute_with_hooks(alice: Alice, key, verifier: Verifier, sc: Scenario, rng) -> VerifierRecord:
    """Alice.distribute() with adversarial hooks on the channel / classical stream."""
    c = rng.integers(0, 3, key.L).astype(np.uint8)
    batch = teleport(rng, key.bases, key.values, c, sc.channel, backend=alice.params.backend,
                     bell_bias=sc.bell_bias if alice.params.backend == "fast" else None)
    rec = VerifierRecord(key.key_block_id, key.message_bit, c, batch.raw_bits, batch=batch)
    payload = np.packbits(np.concatenate([batch.m1, batch.m2])).tobytes()
    auth = wc.authenticate(alice.mac_pool, key.key_block_id.encode() + payload)
    m1, m2 = batch.m1, batch.m2
    if sc.tamper_corrections:   # Mallory flips 30 % of the correction bits in transit
        m1 = m1 ^ (rng.random(key.L) < 0.3).astype(np.uint8)
        m2 = m2 ^ (rng.random(key.L) < 0.3).astype(np.uint8)
        tampered_payload = np.packbits(np.concatenate([m1, m2])).tobytes()
        # The receiver sees the tampered stream with the original tag -> MAC must fail.
        auth = wc.AuthenticatedMessage(key.key_block_id.encode() + tampered_payload, auth.key_index, auth.tag)
    if sc.no_corrections:       # unauthorized verifier never receives (m1, m2)
        rec.m1 = rec.m2 = None
        rec.corrections_authentic = False
        rec.corrected_bits = rec.raw_bits.copy()
        verifier.records[(rec.key_block_id, rec.message_bit)] = rec
        return rec
    verifier.receive_corrections(rec, m1, m2, auth)
    return rec


def run_session(scenario: str | Scenario, params: ProtocolParams | None = None, seed: int | None = None,
                ledger: QLedger | None = None, message: bytes = b"TRANSFER 1,00,000 INR -> ACC 4471",
                baseline: PauliChannel | None = None, chsh_pairs: int = 400,
                block_on_d4: bool = True, sprt_alpha: float = 1e-6, sprt_beta: float = 1e-10) -> SessionResult:
    import time
    t0 = time.perf_counter()
    sc = build_scenario(scenario) if isinstance(scenario, str) else scenario
    params = params or ProtocolParams()
    rng = np.random.default_rng(seed)
    ledger = ledger if ledger is not None else QLedger()
    # Calibrated honest channel: the benign noise the operator measured during calibration.
    baseline = baseline or PauliChannel.depolarizing(params.e0)
    if sc.name == "honest":
        sc.channel = baseline
    nonce = secrets.token_hex(8)
    timings: dict[str, float] = {}

    pool = wc.MacKeyPool.generate(16, seed=None if seed is None else seed + 1)
    alice = Alice(params, rng, pool)
    bob = Verifier("Bob", params, pool)
    charlie = Verifier("Charlie", params, pool)

    # ---- D4: audit the entanglement resource BEFORE distributing any key ----
    chsh = chsh_sample(rng, sc.source, chsh_pairs)
    d4 = detect_d4(chsh)
    timings["chsh_ms"] = (time.perf_counter() - t0) * 1e3
    detectors: list[DetectorResult] = []

    if d4.flag and block_on_d4:
        d1 = DetectorResult("D1", "Pauli Mismatch Test", 1.0, False, "ok", {"skipped": True},
                            "Not run — distribution blocked by D4.")
        d2 = DetectorResult("D2", "Pauli Channel Inversion", 1.0, False, "ok", {"skipped": True,
                            "fingerprint_class": "benign"}, "Not run — distribution blocked by D4.")
        d3 = DetectorResult("D3", "Bell-Outcome Uniformity Audit", 1.0, False, "ok",
                            {"skipped": True, "mac_ok": True}, "Not run — distribution blocked by D4.")
        d5 = DetectorResult("D5", "Freshness & Replay Guard", 1.0, False, "ok", {"skipped": True},
                            "Not run — distribution blocked by D4.")
        detectors = [d1, d2, d3, d4, d5]
        report = fuse(detectors, extra={"chsh": chsh})
        rec = ledger.append(nonce, "BLOCKED-" + secrets.token_hex(3), 0, "-", "BLOCKED",
                            report.attack_class, report.threat_index, mark_spent=False)
        timings["total_ms"] = (time.perf_counter() - t0) * 1e3
        return SessionResult(sc, params, nonce, report, None, None, None, None, True, rec.as_dict(), timings)

    # ---- KeyGen + distribution (teleportation) to Bob and Charlie ----
    t1 = time.perf_counter()
    kb_id = "KB-" + secrets.token_hex(3)
    keys = alice.keygen(kb_id)
    bit = 1
    key = keys[bit]
    rec_bob = _distribute_with_hooks(alice, key, bob, sc, rng)
    rec_cha = _distribute_with_hooks(alice, key, charlie, sc, rng)
    timings["distribution_ms"] = (time.perf_counter() - t1) * 1e3

    # ---- Signing (honest or adversarial) ----
    other_key = None
    if sc.signature_attack == "replay":
        # First an honest, accepted signature is recorded in the ledger…
        honest_sig = alice.sign(message, kb_id, bit)
        v0 = bob.verify(honest_sig)
        ledger.append(honest_sig.nonce, kb_id, bit, honest_sig.message_hash(), v0.decision)
        # …then the adversary re-submits it verbatim.
        sig = Signature(kb_id, message, bit, honest_sig.bases, honest_sig.values, honest_sig.nonce)
    elif sc.signature_attack == "cross_block":
        old_keys = alice.keygen("KB-OLD-" + secrets.token_hex(2))
        other_key = old_keys[bit]
        sig = forge_signature(rng, "cross_block", key, message, other_key)
    elif sc.signature_attack:
        sig = forge_signature(rng, sc.signature_attack, key,
                              b"TRANSFER 99,00,000 INR -> ACC 0666 (forged)" if sc.signature_attack
                              in ("random", "impersonate") else message)
    else:
        sig = alice.sign(message, kb_id, bit)

    # ---- Verification (Bob; Charlie on symmetrized half for transferability) ----
    t2 = time.perf_counter()
    verdict = verify_record(rec_bob, sig, params)
    _, charlie_pos = symmetrize(rng, params.L)
    verdict_cha = verify_record(rec_cha, sig, params, charlie_pos)
    stream = mismatch_stream(rec_bob, sig)
    sprt = sprt_verify(stream, params.e0, params.p_min, sprt_alpha, sprt_beta)
    timings["verification_ms"] = (time.perf_counter() - t2) * 1e3

    # ---- The five eyes ----
    t3 = time.perf_counter()
    d1 = detect_d1(verdict, params)
    d2 = detect_d2(verdict, baseline)
    d3 = detect_d3(rec_bob.batch.bell_outcome_counts(), mac_ok=rec_bob.corrections_authentic
                   or sc.no_corrections)
    d5 = detect_d5(sig, ledger, nonce)
    if sc.no_corrections:
        d3.explanation += " (Verifier has no correction bits → teleportation lock engaged.)"
    detectors = [d1, d2, d3, d4, d5]
    report = fuse(detectors, extra={
        "chsh": chsh, "sprt": sprt.as_dict(), "charlie": verdict_cha.as_dict(),
        "teleportation_lock": sc.no_corrections, "mac_ok": rec_bob.corrections_authentic,
        "transferable": verdict.decision != "REJECT" and verdict_cha.decision != "REJECT",
        "backend": params.backend,
    })
    timings["detectors_ms"] = (time.perf_counter() - t3) * 1e3

    final_verdict = verdict.decision if not report.attack_class.startswith(("REPLAY",)) else "REJECT"
    if report.attack_class != "NONE (HONEST)" and not report.attack_class.startswith("ELEVATED"):
        final_verdict = "REJECT"
    lrec = ledger.append(nonce, kb_id, bit, sig.message_hash(), final_verdict,
                         report.attack_class, report.threat_index)
    timings["total_ms"] = (time.perf_counter() - t0) * 1e3
    return SessionResult(sc, params, nonce, report, verdict, verdict_cha, sprt, sig, False,
                         lrec.as_dict(), timings)
