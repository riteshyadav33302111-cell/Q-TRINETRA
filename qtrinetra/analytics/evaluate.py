"""M6 — Analytics & benchmarks (evaluation plan of the idea document).

    forgery_probability_vs_L()  simulated forgery acceptance vs Hoeffding bound
    honest_rejection_vs_noise() FRR vs channel noise (deterministic acceptance at 0)
    sprt_savings()              average qubits checked: SPRT vs fixed length
    confusion_matrix()          attack-fingerprint confusion matrix from the decision table
    chsh_distribution()         honest vs fake source
    throughput()                qubits/s fast vs stim; signatures/s
"""
from __future__ import annotations

import time

import numpy as np

from qtrinetra.attacks.adversaries import SCENARIO_NAMES
from qtrinetra.core.pauli import pauli_frame_correct_vector
from qtrinetra.detect.d1_mismatch import hoeffding_forgery_bound, hoeffding_honest_reject_bound
from qtrinetra.detect.sprt import sprt_verify
from qtrinetra.engine.teleport import BellSource, PauliChannel, chsh_sample, teleport
from qtrinetra.ledger.qledger import QLedger
from qtrinetra.pipeline import run_session
from qtrinetra.protocol.qds import ProtocolParams


def forgery_probability_vs_L(Ls=(60, 120, 240, 480, 960, 1920), trials=4000, p_min=1 / 6, s_a=0.08,
                             seed=0) -> list[dict]:
    """Measure-and-guess forger: mismatch ~ Binomial(Lc, 1/3). Hoeffding bound uses p_min."""
    rng = np.random.default_rng(seed)
    out = []
    for L in Ls:
        Lc = L // 3
        M = rng.binomial(Lc, 1 / 3, size=trials)
        acc = float(np.mean(M <= s_a * Lc))
        out.append({"L": L, "Lc": Lc, "simulated": acc, "hoeffding_bound": hoeffding_forgery_bound(Lc, p_min, s_a)})
    return out


def honest_rejection_vs_noise(noise=(0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08), L=4800, trials=300,
                              s_a=0.08, seed=0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for p in noise:
        ch = PauliChannel.depolarizing(p)
        rej = 0
        rates = []
        for _ in range(trials):
            a = rng.integers(0, 3, L).astype(np.uint8)
            v = rng.integers(0, 2, L).astype(np.uint8)
            c = rng.integers(0, 3, L).astype(np.uint8)
            b = teleport(rng, a, v, c, ch)
            corr = pauli_frame_correct_vector(c, b.raw_bits, b.m1, b.m2)
            same = a == c
            M = int((corr[same] != v[same]).sum())
            Lc = int(same.sum())
            rates.append(M / Lc)
            rej += M > s_a * Lc
        out.append({"noise": p, "honest_rejection_rate": rej / trials, "mean_mismatch_rate": float(np.mean(rates)),
                    "hoeffding_bound": hoeffding_honest_reject_bound(L // 3, 2 * p / 3, s_a)})
    return out


def sprt_savings(Lc=1600, trials=400, e0=0.01, p_min=1 / 6, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    n_h, n_f, wrong = [], [], 0
    for _ in range(trials):
        s = sprt_verify(rng.random(Lc) < e0, e0, p_min)
        n_h.append(s.n_used)
        wrong += s.decision == "REJECT"
        s = sprt_verify(rng.random(Lc) < 1 / 3, e0, p_min)
        n_f.append(s.n_used)
        wrong += s.decision == "ACCEPT"
    return {"fixed_length": Lc, "sprt_mean_honest": float(np.mean(n_h)), "sprt_mean_forger": float(np.mean(n_f)),
            "saving_honest": 1 - float(np.mean(n_h)) / Lc, "saving_forger": 1 - float(np.mean(n_f)) / Lc,
            "wrong_decisions": wrong, "trials": trials}


def confusion_matrix(seeds=range(5), scenarios=None) -> dict:
    scenarios = scenarios or SCENARIO_NAMES
    led = QLedger()
    table: dict[str, dict[str, int]] = {}
    for sc in scenarios:
        row: dict[str, int] = {}
        for sd in seeds:
            r = run_session(sc, seed=int(sd), ledger=led)
            row[r.report.attack_class] = row.get(r.report.attack_class, 0) + 1
        table[sc] = row
    return table


def chsh_distribution(trials=200, n_per_setting=400, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    honest = [chsh_sample(rng, BellSource(1.0), n_per_setting)["S"] for _ in range(trials)]
    degraded = [chsh_sample(rng, BellSource(0.75), n_per_setting)["S"] for _ in range(trials)]
    fake = [chsh_sample(rng, BellSource(0.0), n_per_setting)["S"] for _ in range(trials)]
    return {"honest": honest, "degraded_f0.75": degraded, "fake": fake,
            "honest_mean": float(np.mean(honest)), "fake_mean": float(np.mean(fake))}


def throughput(n=200_000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, n).astype(np.uint8)
    v = rng.integers(0, 2, n).astype(np.uint8)
    c = rng.integers(0, 3, n).astype(np.uint8)
    out = {}
    for be in ("fast", "stim"):
        t = time.perf_counter()
        teleport(rng, a, v, c, PauliChannel.depolarizing(0.01), backend=be)
        out[f"qubits_per_s_{be}"] = n / (time.perf_counter() - t)
    t = time.perf_counter()
    led = QLedger()
    k = 20
    for i in range(k):
        run_session("honest", seed=i, ledger=led, chsh_pairs=100)
    out["signatures_per_s"] = k / (time.perf_counter() - t)
    return out


def full_report() -> dict:
    return {
        "forgery_probability_vs_L": forgery_probability_vs_L(),
        "honest_rejection_vs_noise": honest_rejection_vs_noise(trials=100),
        "sprt_savings": sprt_savings(trials=200),
        "confusion_matrix": confusion_matrix(seeds=range(3)),
        "chsh": {k: v for k, v in chsh_distribution(trials=60).items() if not isinstance(v, list)},
        "throughput": throughput(n=100_000),
        "complexity": {"verification": "O(L)", "correction_per_qubit": "O(1)", "ledger_proof": "O(log N)"},
    }
