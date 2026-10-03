"""D3 — Bell-Outcome Uniformity Audit (source / classical-channel tampering).

Honest teleportation yields Bell outcomes uniform over {00, 01, 10, 11}
regardless of the teleported state.  A chi-square test with 3 dof flags a
biased / fake source or a tampered correction stream.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from qtrinetra.detect.base import DetectorResult, clamp_p


def bell_uniformity_pvalue(counts) -> tuple[float, float]:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0, 1.0
    exp = np.full(4, total / 4)
    stat = float((((counts - exp) ** 2) / exp).sum())
    return stat, float(chi2.sf(stat, df=3))


def detect_d3(counts, alpha: float = 1e-3, mac_ok: bool = True) -> DetectorResult:
    stat, p = bell_uniformity_pvalue(counts)
    counts = np.asarray(counts)
    flag = p < alpha or not mac_ok
    sev = "alarm" if flag else ("warn" if p < 0.05 else "ok")
    freqs = (counts / max(counts.sum(), 1)).tolist()
    if not mac_ok:
        expl = "Wegman–Carter MAC on the correction-bit stream FAILED → tampered classical channel."
    elif flag:
        expl = (f"Bell outcomes are non-uniform (χ²={stat:.1f}, p={p:.2e}); frequencies "
                f"{[round(f, 3) for f in freqs]} → biased/fake source or tampered correction stream.")
    else:
        expl = f"Bell outcomes uniform (χ²={stat:.2f}, p={p:.3f})."
    return DetectorResult("D3", "Bell-Outcome Uniformity Audit", clamp_p(p), flag, sev,
                          stats={"counts": counts.tolist(), "freqs": freqs, "chi2": stat,
                                 "mac_ok": mac_ok}, explanation=expl)
