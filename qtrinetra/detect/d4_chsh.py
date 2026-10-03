"""D4 — CHSH Entanglement Witness.

A random sample of Bell pairs is sacrificed to estimate the CHSH value S and
the Bell-state fidelity F = (1 + <XX> - <YY> + <ZZ>)/4.

* classical / separable resources:  S <= 2
* ideal |Phi+> pairs:               S = 2*sqrt(2) ≈ 2.83,  F = 1

p-value: one-sided z-test of H0 "S >= S_threshold" (i.e. small p means S is
significantly BELOW the trusted threshold).  If S or F is below threshold
the pairs are distrusted and key distribution is blocked.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from qtrinetra.detect.base import DetectorResult, clamp_p

S_CLASSICAL = 2.0
S_IDEAL = 2 * np.sqrt(2)


def detect_d4(chsh: dict, s_threshold: float = 2.4, f_threshold: float = 0.85) -> DetectorResult:
    S, se = chsh["S"], max(chsh["se_S"], 1e-9)
    F = chsh["fidelity"]
    z = (S - s_threshold) / se
    p = float(norm.cdf(z))            # P(observing S this low | true S == threshold)
    flag = S < s_threshold or F < f_threshold
    sev = "alarm" if flag else ("warn" if S < 2.6 else "ok")
    if flag:
        expl = (f"CHSH S={S:.3f}±{se:.3f} (threshold {s_threshold}), fidelity F={F:.3f} "
                f"(threshold {f_threshold}) → entanglement resource NOT trusted; "
                f"{'classical bound violated → genuinely non-local' if S > 2 else 'no Bell violation → separable / fake source'}. "
                "Key distribution over these pairs is BLOCKED.")
    else:
        expl = f"CHSH S={S:.3f}±{se:.3f} (ideal 2.83), F={F:.3f} → entanglement verified."
    return DetectorResult("D4", "CHSH Entanglement Witness", clamp_p(p), flag, sev,
                          stats={**chsh, "s_threshold": s_threshold, "f_threshold": f_threshold,
                                 "S_classical": S_CLASSICAL, "S_ideal": float(S_IDEAL), "z": float(z)},
                          explanation=expl)
