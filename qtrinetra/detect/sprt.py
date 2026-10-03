"""Wald Sequential Probability Ratio Test for early-stop verification.

H0: mismatch prob = e0 (honest)      H1: mismatch prob = p_min (forger)
Stop when LLR >= A = ln((1-β)/α)  → REJECT (forger)
          LLR <= B = ln(β/(1-α))  → ACCEPT (honest)
α = P(reject honest), β = P(accept forger).  e0 must be > 0 (floor 1e-3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SPRTResult:
    decision: str              # ACCEPT | REJECT | ESCALATE
    n_used: int
    n_available: int
    A: float
    B: float
    llr_trace: list[float] = field(default_factory=list)
    expected_n_honest: float = 0.0
    expected_n_forger: float = 0.0

    @property
    def saving(self) -> float:
        return 1.0 - self.n_used / max(self.n_available, 1)

    def as_dict(self) -> dict:
        return {"decision": self.decision, "n_used": self.n_used, "n_available": self.n_available,
                "A": self.A, "B": self.B, "saving": self.saving,
                "llr_trace": self.llr_trace[:2000],
                "expected_n_honest": self.expected_n_honest,
                "expected_n_forger": self.expected_n_forger}


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    return float(np.log((1 - beta) / alpha)), float(np.log(beta / (1 - alpha)))


def expected_sample_sizes(e0: float, p_min: float, alpha: float, beta: float) -> tuple[float, float]:
    """Wald's approximations for E[N] under H0 and H1."""
    A, B = sprt_bounds(alpha, beta)
    l1, l0 = np.log(p_min / e0), np.log((1 - p_min) / (1 - e0))
    mu0 = e0 * l1 + (1 - e0) * l0
    mu1 = p_min * l1 + (1 - p_min) * l0
    en0 = ((1 - alpha) * B + alpha * A) / mu0
    en1 = (beta * B + (1 - beta) * A) / mu1
    return float(abs(en0)), float(abs(en1))


def sprt_verify(mismatch_stream, e0: float, p_min: float,
                alpha: float = 1e-6, beta: float = 1e-10, keep_trace: bool = True) -> SPRTResult:
    e0 = max(e0, 1e-3)
    A, B = sprt_bounds(alpha, beta)
    l1, l0 = float(np.log(p_min / e0)), float(np.log((1 - p_min) / (1 - e0)))
    llr, n = 0.0, 0
    trace: list[float] = []
    stream = np.asarray(mismatch_stream)
    decision = "ESCALATE"
    for n, mis in enumerate(stream, 1):
        llr += l1 if mis else l0
        if keep_trace:
            trace.append(llr)
        if llr >= A:
            decision = "REJECT"
            break
        if llr <= B:
            decision = "ACCEPT"
            break
    en0, en1 = expected_sample_sizes(e0, p_min, alpha, beta)
    return SPRTResult(decision, int(n), int(stream.size), A, B, trace, en0, en1)
