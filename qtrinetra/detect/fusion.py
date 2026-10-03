"""Fusion — deterministic threat verdict (NO machine learning).

1. Fisher's method combines the detector p-values:  X = -2 Σ ln p_i ~ χ²_{2k}
   → Quantum Threat Index QTI = 1 - P(χ²_{2k} >= X)  ∈ [0, 1].
2. A fixed, auditable DECISION TABLE (ordered list of boolean rules over the
   detector flags / fingerprint class) maps the pattern to an attack class.
   The table is exported verbatim to the dashboard so judges can audit it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

from qtrinetra.detect.base import DetectorResult, _jsonable


@dataclass
class ThreatReport:
    attack_class: str
    rule_id: str
    rule_text: str
    threat_index: float
    fisher_stat: float
    fisher_p: float
    confidence: str
    detectors: list[DetectorResult]
    recommended_action: str
    where: str
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "attack_class": self.attack_class, "rule_id": self.rule_id, "rule_text": self.rule_text,
            "threat_index": self.threat_index, "fisher_stat": self.fisher_stat,
            "fisher_p": self.fisher_p, "confidence": self.confidence,
            "detectors": [d.as_dict() for d in self.detectors],
            "recommended_action": self.recommended_action, "where": self.where,
            "extra": _jsonable(self.extra),
        }


def fisher_combine(p_values: list[float]) -> tuple[float, float]:
    p = np.clip(np.asarray(p_values, dtype=float), 1e-300, 1.0)
    stat = float(-2 * np.log(p).sum())
    return stat, float(chi2.sf(stat, df=2 * len(p)))


# ---------------------------------------------------------------------------
# Decision table — evaluated top to bottom, first match wins.
# Each rule: (id, predicate(ctx) -> bool, attack_class, human text, where, action)
# ctx keys: d1..d5 (flags), fp (D2 fingerprint class), unphys (bool), mac_ok,
#           d1_rate (mismatch rate), d4_S
# ---------------------------------------------------------------------------
DECISION_TABLE = [
    ("R1", lambda c: c["d5"],
     "REPLAY ATTACK", "D5 spent-key hit → signature re-submitted for an already spent key block",
     "classical submission channel", "Drop signature; alert operator; key block already consumed."),
    ("R2", lambda c: not c["mac_ok"],
     "CORRECTION-BIT TAMPERING", "Wegman–Carter MAC failure on correction stream",
     "classical correction channel", "Discard distribution; re-key MAC; investigate classical link."),
    ("R3", lambda c: c["d4"] and c["d3"],
     "COMPROMISED ENTANGLEMENT SOURCE (biased + non-local failure)",
     "D4 low AND D3 non-uniform → source replaced by a biased classical device",
     "entanglement source", "Block distribution; quarantine source; switch to backup pair source."),
    ("R4", lambda c: c["d4"],
     "FAKE / DEGRADED ENTANGLEMENT SOURCE", "D4 CHSH ≤ threshold with D3 normal → separable substitute",
     "entanglement source", "Block key distribution before any key is sent over these pairs."),
    ("R5", lambda c: c["d3"],
     "BIASED SOURCE / TAMPERED CORRECTION STREAM", "D3 Bell outcomes non-uniform, D4 normal",
     "source or classical channel", "Audit Bell-measurement device and correction channel."),
    ("R5b", lambda c: c["lock"],
     "UNAUTHORIZED VERIFICATION ATTEMPT (teleportation lock)",
     "Verifier holds no authenticated correction bits → record is uniformly random (≈½ mismatch)",
     "verifier without (m1,m2)", "Verification impossible without MAC-authenticated corrections; deny."),
    ("R6", lambda c: c["unphys"],
     "ADAPTIVE BASIS-AWARE ADVERSARY", "D2 recovered unphysical (negative) Pauli probabilities",
     "quantum channel (adaptive)", "Reject; adversary has basis information → rotate bases & re-key."),
    ("R7", lambda c: c["d2"] and c["fp"].startswith("Z-dominant"),
     "CHANNEL MANIPULATION: Z-BASIS INTERCEPT–RESEND", "D2 fingerprint eX≈eY≈½, eZ≈0 → pZ dominant",
     "quantum channel", "Abort block; eavesdropper measuring in Z; re-route channel."),
    ("R8", lambda c: c["d2"] and c["fp"].startswith("X-dominant"),
     "CHANNEL MANIPULATION: X-BASIS INTERCEPT–RESEND / BIT-FLIP INJECTION",
     "D2 fingerprint eY≈eZ, eX≈0 → pX dominant", "quantum channel", "Abort block; re-route channel."),
    ("R9", lambda c: c["d2"] and c["fp"].startswith("Y-dominant"),
     "CHANNEL MANIPULATION: Y-BASIS INTERCEPT–RESEND", "D2 fingerprint eX≈eZ, eY≈0 → pY dominant",
     "quantum channel", "Abort block; re-route channel."),
    ("R10", lambda c: c["d1"] and c["d1_rate"] > 0.42 and c["fp"] != "isotropic-large",
     "FORGERY (random key) / CROSS-BLOCK REPLAY", "D1 mismatch ≈ ½ → revealed key is unrelated to block",
     "signer / submission", "Reject; key holder has no knowledge of the block."),
    ("R11", lambda c: c["d2"] and c["fp"] == "isotropic-large" and c["d1"] and c["d1_rate"] < 0.42,
     "MEASURE-AND-GUESS FORGER / RANDOM-BASIS INTERCEPT–RESEND",
     "D2 isotropic ≈⅓ each + D1 ≈ ⅓ → adversary measured every qubit in a random Pauli basis (forger or Eve; physically identical signature)",
     "quantum channel / signer identity", "Reject block; adversary measured the public-key qubits in random bases."),
    ("R10b", lambda c: c["d1"] and c["d1_rate"] > 0.42,
     "FORGERY (random key) / CROSS-BLOCK REPLAY", "D1 mismatch ≈ ½ → revealed key is unrelated to block",
     "signer / submission", "Reject; key holder has no knowledge of the block."),
    ("R12", lambda c: c["d1"] and c["d2"] and c["fp"] == "isotropic-medium",
     "FORGERY / IMPERSONATION (cloning-class adversary)",
     "D1 high + D2 isotropic medium (≈⅙) + D4 normal → optimal-cloning or measure-and-guess forger",
     "signer identity", "Reject; impersonation attempt without private key."),
    ("R13", lambda c: c["d1"],
     "FORGERY / IMPERSONATION", "D1 mismatch above Hoeffding threshold, D4 normal",
     "signer identity", "Reject signature."),
    ("R14", lambda c: c["d2"],
     "CHANNEL MANIPULATION (unclassified Pauli fingerprint)", "D2 deviates from baseline but D1 passes",
     "quantum channel", "Raise channel alert; continue monitoring."),
    ("R15", lambda c: c["d1_decision"] == "TRANSFERABLE_UNCERTAIN",
     "ELEVATED NOISE / POSSIBLE REPUDIATION ATTEMPT", "Mismatch in (s_a, s_v] band, no detector alarm",
     "channel or signer", "Accept as transferable-uncertain; cross-check with second verifier."),
    ("R0", lambda c: True,
     "NONE (HONEST)", "All five eyes quiet", "—", "Accept signature; record in Q-Ledger."),
]


def decision_table_export() -> list[dict]:
    return [{"id": rid, "attack_class": cls, "condition": txt, "where": where, "action": act}
            for rid, _, cls, txt, where, act in DECISION_TABLE]


def fuse(detectors: list[DetectorResult], extra: dict | None = None) -> ThreatReport:
    by = {d.name: d for d in detectors}
    d1, d2, d3, d4, d5 = (by.get(k) for k in ("D1", "D2", "D3", "D4", "D5"))
    ctx = {
        "d1": bool(d1 and d1.flag), "d2": bool(d2 and d2.flag), "d3": bool(d3 and d3.flag),
        "d4": bool(d4 and d4.flag), "d5": bool(d5 and d5.flag),
        "fp": (d2.stats.get("fingerprint_class", "benign") if d2 else "benign"),
        "unphys": bool(d2 and d2.stats.get("unphysical")),
        "mac_ok": (d3.stats.get("mac_ok", True) if d3 else True),
        "d1_rate": float(d1.stats.get("rate", 0.0)) if d1 else 0.0,
        "d1_decision": d1.stats.get("decision", "ACCEPT") if d1 else "ACCEPT",
        "lock": bool((extra or {}).get("teleportation_lock", False)),
    }
    stat, p = fisher_combine([d.p_value for d in detectors])
    qti = 1.0 - p
    for rid, pred, cls, txt, where, act in DECISION_TABLE:
        if pred(ctx):
            break
    n_alarms = sum(ctx[k] for k in ("d1", "d2", "d3", "d4", "d5"))
    if cls.startswith("NONE"):
        conf = "high" if qti < 0.5 else "medium"
    else:
        conf = "high" if (qti > 0.999 or n_alarms >= 2 or ctx["d5"]) else ("medium" if qti > 0.95 else "low")
    return ThreatReport(cls, rid, txt, float(qti), stat, p, conf, detectors, act, where, extra or {})
