"""Common detector result container."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectorResult:
    name: str                       # "D1" .. "D5"
    title: str
    p_value: float                  # probability of seeing stats this extreme under H0 (honest)
    flag: bool                      # True = alarm
    severity: str                   # "ok" | "warn" | "alarm"
    stats: dict = field(default_factory=dict)
    explanation: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name, "title": self.title, "p_value": float(self.p_value),
            "flag": bool(self.flag), "severity": self.severity, "stats": _jsonable(self.stats),
            "explanation": self.explanation,
        }


def _jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def clamp_p(p: float) -> float:
    """Keep p-values inside (1e-300, 1] so Fisher's method stays finite."""
    return float(min(max(p, 1e-300), 1.0))
