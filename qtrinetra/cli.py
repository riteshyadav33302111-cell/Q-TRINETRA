"""Command-line entry point.

    python -m qtrinetra.cli demo                 # 5-step SIH demo script
    python -m qtrinetra.cli run <scenario>       # one scenario, full JSON report
    python -m qtrinetra.cli list                 # list scenarios
    python -m qtrinetra.cli bench                # evaluation metrics
    python -m qtrinetra.cli table                # print the decision table
"""
from __future__ import annotations

import json
import sys

from qtrinetra.attacks.adversaries import SCENARIO_NAMES, build_scenario
from qtrinetra.detect.fusion import decision_table_export
from qtrinetra.ledger.qledger import QLedger
from qtrinetra.pipeline import run_session

BAR = "─" * 78


def _print_session(r) -> None:
    rep = r.report
    print(f"Scenario : {r.scenario.label}  [{r.scenario.stage}]")
    print(f"Verdict  : {rep.attack_class}   (rule {rep.rule_id}, confidence {rep.confidence})")
    print(f"QTI      : {rep.threat_index:.6f}   Fisher χ²={rep.fisher_stat:.1f}  p={rep.fisher_p:.2e}")
    print(f"Where    : {rep.where}")
    print(f"Action   : {rep.recommended_action}")
    if r.verdict_bob:
        v = r.verdict_bob
        print(f"Bob      : {v.decision}  M={v.mismatches}/Lc={v.checked}  eX={v.basis_error_rates[0]:.3f} "
              f"eY={v.basis_error_rates[1]:.3f} eZ={v.basis_error_rates[2]:.3f}")
        print(f"Charlie  : {r.verdict_charlie.decision}  M={r.verdict_charlie.mismatches}/Lc={r.verdict_charlie.checked}")
        print(f"SPRT     : {r.sprt.decision} after {r.sprt.n_used} of {r.sprt.n_available} qubits "
              f"(saving {r.sprt.saving:.0%})")
    for d in rep.detectors:
        mark = "🔴" if d.severity == "alarm" else ("🟡" if d.severity == "warn" else "🟢")
        print(f"  {mark} {d.name} {d.title:30s} p={d.p_value:.2e}  {d.explanation}")
    if rep.detectors[1].stats.get("recovered"):
        rec = rep.detectors[1].stats["recovered"]
        print(f"  Recovered Pauli channel: pI={rec['I']:.3f} pX={rec['X']:.3f} pY={rec['Y']:.3f} pZ={rec['Z']:.3f}")
    print(f"Ledger   : idx={r.ledger_record['index']} chain={r.ledger_record['chain_hash'][:16]}…  "
          f"time={r.timings_ms['total_ms']:.1f} ms")


def demo() -> None:
    led = QLedger()
    steps = [
        ("1. Honest signature arrives", "honest"),
        ("2. A forger attacks (measure-and-guess)", "measure_guess_forger"),
        ("3. Intercept–resend in Z basis", "intercept_resend_Z"),
        ("4. Entanglement source swapped for a fake one", "fake_source"),
        ("5. Old signature replayed", "replay_same_block"),
    ]
    for title, sc in steps:
        print(BAR)
        print(title)
        print(BAR)
        _print_session(run_session(sc, seed=42, ledger=led))
        print()
    print(f"Q-Ledger: {len(led)} records, chain valid = {led.verify_chain()}, "
          f"Merkle batches = {len(led.batches())}")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "demo":
        demo()
    elif cmd == "list":
        for n in SCENARIO_NAMES:
            s = build_scenario(n)
            print(f"{n:26s} {s.label:45s} {s.stage}")
    elif cmd == "run":
        name = argv[1] if len(argv) > 1 else "honest"
        r = run_session(name, seed=int(argv[2]) if len(argv) > 2 else None)
        if "--json" in argv:
            print(json.dumps(r.as_dict(), indent=1, default=str))
        else:
            _print_session(r)
    elif cmd == "table":
        for row in decision_table_export():
            print(f"{row['id']:5s} {row['attack_class']:60s} ← {row['condition']}")
    elif cmd == "bench":
        from qtrinetra.analytics.evaluate import full_report
        print(json.dumps(full_report(), indent=1, default=str))
    else:
        print(f"unknown command {cmd!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
