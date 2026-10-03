"""D1 — Pauli Mismatch Test (forgery & impersonation).

H0 (honest): each checked position mismatches independently w.p. e0.
H1 (forger): mismatch rate >= p_min (measure-and-guess gives 1/3; the
conservative bound used is p_min = 1/6, verified by the attack simulator).

Statistics
----------
* exact binomial tail  P(M >= m | Lc, e0)         -> p-value
* Hoeffding bounds for the thresholds:
      P(forgery accepted) <= exp(-2 Lc (p_min - s_a)^2)
      P(honest rejected)  <= exp(-2 Lc (s_a - e0)^2)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from qtrinetra.detect.base import DetectorResult, clamp_p
from qtrinetra.protocol.qds import ProtocolParams, Verdict


def hoeffding_forgery_bound(Lc: int, p_min: float, s_a: float) -> float:
    return float(np.exp(-2 * Lc * max(p_min - s_a, 0.0) ** 2))


def hoeffding_honest_reject_bound(Lc: int, e0: float, s_a: float) -> float:
    return float(np.exp(-2 * Lc * max(s_a - e0, 0.0) ** 2))


def required_Lc(p_min: float, s_a: float, target: float) -> int:
    """Smallest Lc with Hoeffding forgery bound <= target."""
    return int(np.ceil(np.log(1 / target) / (2 * (p_min - s_a) ** 2)))


def detect_d1(verdict: Verdict, params: ProtocolParams) -> DetectorResult:
    Lc, M = verdict.checked, verdict.mismatches
    e0 = max(params.e0, 1e-3)
    p_honest = binom.sf(M - 1, Lc, e0) if Lc > 0 else 1.0     # P(M' >= M | honest)
    p_forger = binom.cdf(M, Lc, params.p_min) if Lc > 0 else 1.0  # P(M' <= M | forger)
    flag = verdict.decision != "ACCEPT"
    if verdict.decision == "REJECT":
        sev = "alarm"
    elif verdict.decision == "TRANSFERABLE_UNCERTAIN":
        sev = "warn"
    else:
        sev = "ok"
    expl = (f"M={M} mismatches on Lc={Lc} checked positions (rate {verdict.mismatch_rate:.4f}); "
            f"thresholds s_a={params.s_a}, s_v={params.s_v}. ")
    if flag:
        expl += ("Mismatch rate is far above the honest channel error → key holder does NOT know "
                 "the private key (forgery / impersonation / cross-block replay).")
    else:
        expl += "Consistent with honest signer."
    return DetectorResult(
        "D1", "Pauli Mismatch Test", clamp_p(p_honest), flag, sev,
        stats={
            "Lc": Lc, "M": M, "rate": verdict.mismatch_rate, "decision": verdict.decision,
            "p_value_honest": float(p_honest), "p_value_forger": float(p_forger),
            "hoeffding_forgery_bound": hoeffding_forgery_bound(Lc, params.p_min, params.s_a),
            "hoeffding_honest_reject_bound": hoeffding_honest_reject_bound(Lc, e0, params.s_a),
        },
        explanation=expl,
    )
