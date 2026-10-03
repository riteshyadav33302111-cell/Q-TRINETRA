#!/usr/bin/env python3
"""No-ML CI gate: fails the build if any ML / AI library appears in the
dependency files or is imported anywhere in the source tree.
This is visible proof that Q-Trinetra meets the sponsor's "no AI/ML" requirement."""
import pathlib
import re
import sys

FORBIDDEN = ["torch", "tensorflow", "keras", "sklearn", "scikit-learn", "scikit_learn", "jax",
             "xgboost", "lightgbm", "catboost", "transformers", "pytorch", "theano", "mxnet",
             "paddle", "onnx", "openai", "langchain", "huggingface"]
ROOT = pathlib.Path(__file__).resolve().parents[1]
bad = []
for f in [ROOT / "requirements.txt", ROOT / "pyproject.toml"]:
    txt = f.read_text().lower()
    bad += [f"{f.name}: {w}" for w in FORBIDDEN if re.search(rf"\b{re.escape(w)}\b", txt)]
for py in list((ROOT / "qtrinetra").rglob("*.py")) + list((ROOT / "backend").rglob("*.py")):
    for line in py.read_text().splitlines():
        s = line.strip()
        if s.startswith(("import ", "from ")):
            mod = s.split()[1].split(".")[0].lower()
            if mod in FORBIDDEN:
                bad.append(f"{py.relative_to(ROOT)}: {s}")
if bad:
    print("NO-ML GATE FAILED:\n  " + "\n  ".join(bad))
    sys.exit(1)
print("NO-ML GATE PASSED: no machine-learning libraries in dependencies or imports.")
