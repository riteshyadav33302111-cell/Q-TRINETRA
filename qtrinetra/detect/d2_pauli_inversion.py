"""D2 — Pauli Channel Inversion (channel-manipulation fingerprinting).

Basis-resolved error rates eX, eY, eZ measured on checked positions.  For a
Pauli (twirled) channel

    eZ = pX + pY,   eX = pY + pZ,   eY = pX + pZ

so the adversary's operation is recovered exactly by

    pX = (eZ + eY - eX)/2,  pY = (eZ + eX - eY)/2,  pZ = (eX + eY - eZ)/2,
    pI = 1 - (eX + eY + eZ)/2.

Extra checks:
* chi-square test of (eX, eY, eZ) against the calibrated baseline channel;
* negative p_i beyond statistical error  =>  non-Pauli / basis-aware adaptive
  adversary (an alarm on its own);
* anisotropy classification of the fingerprint.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm

from qtrinetra.detect.base import DetectorResult, clamp_p
from qtrinetra.engine.teleport import PauliChannel
from qtrinetra.protocol.qds import Verdict


def pauli_channel_inversion(eX: float, eY: float, eZ: float) -> dict[str, float]:
    pX = (eZ + eY - eX) / 2
    pY = (eZ + eX - eY) / 2
    pZ = (eX + eY - eZ) / 2
    pI = 1 - (eX + eY + eZ) / 2
    return {"I": pI, "X": pX, "Y": pY, "Z": pZ}


def expected_error_rates(ch: PauliChannel) -> np.ndarray:
    """Forward model: (eX, eY, eZ) produced by a Pauli channel."""
    return np.array([ch.p_y + ch.p_z, ch.p_x + ch.p_z, ch.p_x + ch.p_y])


def classify_fingerprint(p: dict[str, float], total: float) -> str:
    """Deterministic fingerprint class for the recovered channel."""
    if total < 0.03:
        return "benign"
    px, py, pz = p["X"], p["Y"], p["Z"]
    arr = np.array([px, py, pz])
    dom = int(np.argmax(arr))
    share = arr[dom] / max(arr.sum(), 1e-12)
    if share > 0.7:
        return ["X-dominant (bit-flip)", "Y-dominant", "Z-dominant (dephasing)"][dom]
    spread = arr.max() - arr.min()
    if spread < 0.08:
        # total = pX+pY+pZ : random-basis intercept-resend gives 3*(1/6)=0.5,
        # optimal cloning gives 3*(1/12)=0.25.
        return "isotropic-large" if total > 0.4 else "isotropic-medium"
    return "anisotropic-mixed"


def detect_d2(verdict: Verdict, baseline: PauliChannel | None = None,
              sigma_unphysical: float = 3.0) -> DetectorResult:
    baseline = baseline or PauliChannel()
    n = verdict.per_basis_checked.astype(float)
    k = verdict.per_basis_mismatch.astype(float)
    e = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    eX, eY, eZ = (float(v) for v in e)
    rec = pauli_channel_inversion(eX, eY, eZ)

    # Standard error of each p_i: each is ±½ a sum of three binomial rates.
    var_e = np.where(n > 0, e * (1 - e) / np.maximum(n, 1), 0.0)
    # Floor the variance with the rule-of-three to avoid zero-variance when e==0.
    var_e = np.maximum(var_e, (3.0 / np.maximum(n, 1)) ** 2 / 9)
    se_p = 0.5 * np.sqrt(var_e.sum())

    # chi-square against calibrated baseline (expected counts per basis).
    e_exp = np.clip(expected_error_rates(baseline), 1e-4, 1 - 1e-4)
    chi_stat = 0.0
    for ni, ki, ei in zip(n, k, e_exp):
        if ni > 0:
            exp_k = ni * ei
            chi_stat += (ki - exp_k) ** 2 / exp_k + ((ni - ki) - ni * (1 - ei)) ** 2 / (ni * (1 - ei))
    p_chi = float(chi2.sf(chi_stat, df=3))

    unphysical = [name for name in ("X", "Y", "Z", "I")
                  if rec[name] < -sigma_unphysical * se_p]
    total = rec["X"] + rec["Y"] + rec["Z"]
    cls = classify_fingerprint(rec, total)
    base_total = baseline.p_x + baseline.p_y + baseline.p_z
    excess = total - base_total

    flag = (p_chi < 1e-3 and excess > 0.02) or bool(unphysical)
    sev = "alarm" if flag else ("warn" if p_chi < 0.05 else "ok")
    if unphysical:
        expl = (f"Recovered channel has unphysical negative probabilities {unphysical} "
                f"(> {sigma_unphysical}σ) → basis-aware ADAPTIVE adversary.")
    elif flag:
        expl = (f"Live fingerprint (eX={eX:.3f}, eY={eY:.3f}, eZ={eZ:.3f}) deviates from calibration "
                f"(χ²={chi_stat:.1f}, p={p_chi:.2e}). Recovered channel pX={rec['X']:.3f}, "
                f"pY={rec['Y']:.3f}, pZ={rec['Z']:.3f} → class '{cls}'.")
    else:
        expl = f"Fingerprint matches calibrated baseline (χ² p={p_chi:.3f}); class '{cls}'."
    return DetectorResult(
        "D2", "Pauli Channel Inversion", clamp_p(p_chi), flag, sev,
        stats={"eX": eX, "eY": eY, "eZ": eZ, "recovered": rec, "se_p": float(se_p),
               "chi2": float(chi_stat), "unphysical": unphysical, "fingerprint_class": cls,
               "total_error_mass": float(total), "excess_over_baseline": float(excess),
               "baseline": baseline.as_dict(), "per_basis_checked": n.astype(int).tolist(),
               "per_basis_mismatch": k.astype(int).tolist()},
        explanation=expl,
    )
