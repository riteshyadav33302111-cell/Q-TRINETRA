# Q-TRINETRA — The Three-Eyed Quantum Sentinel

**Pauli-Spectrum Threat Forensics for Teleportation-Based Quantum Digital Signatures**

SIH 2026 | PS ID 26141 | Egreen Quanta | Theme: Blockchain & Cybersecurity | Category: Software

> **Trinetra = three eyes = the three Pauli bases X, Y, Z.**
> Most detectors only say "attack or not". Q-Trinetra watches every signature through the three Pauli
> eyes, mathematically reconstructs the attacker's quantum operation from measurement statistics,
> and tells you **what kind of attack happened, where, and how confident the verdict is** —
> with pure quantum physics + statistics. **No AI / ML.**

---

## 0. HANDOFF NOTE (read this first if you are an AI/dev resuming this project)

This repository is being built incrementally. If the previous session ran out of tokens, **resume
from the "Implementation Status" table in §3** — every module has a status (`DONE` / `WIP` / `TODO`)
and the file it lives in. The design is fully specified in §2–§6 so you can implement any `TODO`
module without re-deriving the idea. Follow the conventions in §7 (Dev Workflow).

Quick start for a resuming agent:

```bash
cd /home/user/webapp            # repo root
pip install -r requirements.txt  # numpy, scipy, stim, fastapi, uvicorn, pydantic, pytest, hypothesis
pytest -q                        # all tests must stay green
python -m qtrinetra.cli demo     # run the 5-step demo script in the terminal
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload   # dashboard at http://localhost:8000
```

Git conventions: work on branch `genspark_ai_developer`, commit after every change, squash, push,
open/update PR to `main`.

---

## 1. Problem Statement (from SIH 26141)

| Field | Details |
|---|---|
| PS ID | 26141 |
| Title | Quantum-Inspired Cyber Threat Detection for Digital Signature Security |
| Organisation | Egreen Quanta |
| Theme | Blockchain & Cybersecurity |
| Category | Software |

Shor's algorithm breaks RSA/ECC. Quantum Digital Signatures (QDS) offer information-theoretic
security; teleportation-based QDS is among the most practical. But a QDS protocol alone only outputs
*accept/reject* — it cannot say whether a rejection came from a forger, impersonator, replay, a
tampered entanglement source, or ordinary noise. Critical infrastructure needs **threat
intelligence**, not a single bit.

**Required**: a software framework that simulates quantum public-key distribution via Bell-state
entanglement + teleportation, applies Pauli corrections + projective measurements for verification,
and detects forgery / impersonation / replay / channel-manipulation via statistical evaluation and
threshold rules — **without AI/ML** — including mathematical modelling, attack simulation, security
analysis, and performance evaluation.

---

## 2. The Idea — Design Specification

### 2.1 Protocol model (the QDS being protected)

Parties: **Alice** (signer), **Bob** and **Charlie** (verifiers; Charlie models transferability /
non-repudiation).

1. **KeyGen** — for each message bit `b ∈ {0,1}` Alice draws `L` random pairs `(a_i, v_i)`,
   `a_i ∈ {X,Y,Z}` (basis), `v_i ∈ {0,1}` (eigenvalue). This is the private key. The public key is
   the list of Pauli eigenstates `|a_i, v_i⟩` ∈ {|0⟩,|1⟩,|+⟩,|−⟩,|+i⟩,|−i⟩}.
2. **Distribute (teleportation)** — a source distributes Bell pairs `|Φ+⟩`. Alice does a Bell
   measurement on each public-key qubit and her half of the pair → 2-bit outcome `(m1, m2)`. The
   verifier's qubit becomes `X^m2 Z^m1 |ψ⟩`. Correction bits go over a classical channel
   authenticated with a **Wegman–Carter MAC** (information-theoretically secure).
3. **Pauli-frame verification (efficiency contribution)** — Pauli eigenstates stay Pauli eigenstates
   under Pauli corrections; only the eigenvalue may flip. So the verifier measures **immediately on
   arrival** in a random basis `c_i ∈ {X,Y,Z}` and later corrects the bit *classically*:
   - X-basis outcome: flip iff `m1 = 1` (Z anticommutes with X)
   - Z-basis outcome: flip iff `m2 = 1` (X anticommutes with Z)
   - Y-basis outcome: flip iff `m1 ⊕ m2 = 1`

   → **no quantum memory, O(1) per qubit**. Also a **teleportation lock**: without the authenticated
   correction bits the verifier's record is uniformly random (defeats unauthorized verification).
4. **Sign** — to sign bit `b` Alice reveals the private key for `b` (plus message hash, nonce,
   key-block id).
5. **Verify** — look only at positions where `c_i = a_i` (≈ `L/3`, call it `Lc`); count mismatches
   `M`. Two thresholds `s_a < s_v`:
   `ACCEPT` if `M ≤ s_a·Lc`; `TRANSFERABLE-UNCERTAIN` if `s_a·Lc < M ≤ s_v·Lc`; `REJECT` otherwise.
   *Deterministic acceptance*: noiseless honest signature → `M = 0` → accepted w.p. 1.
6. **Symmetrization** — Bob and Charlie secretly swap a random half of their positions so a cheating
   Alice cannot make Bob accept while Charlie rejects (non-repudiation).

### 2.2 The Five Eyes of detection (core innovation)

| Eye | Name | What it catches | Mechanism |
|---|---|---|---|
| **D1** | Pauli Mismatch Test | forgery, impersonation | Mismatch rate on checked positions; a key-less adversary guessing (a,v) gets ≥ `p_min` (≈1/3 for measure-and-guess; conservative 1/6). Hoeffding: `P(forge accepted) ≤ exp(−2·Lc·(p_min − s_a)²)`, `P(honest rejected) ≤ exp(−2·Lc·(s_a − e0)²)`. |
| **D2** | Pauli Channel Inversion | channel manipulation fingerprinting | Basis-resolved error rates `eX,eY,eZ`. For a Pauli channel `eZ = pX+pY`, `eX = pY+pZ`, `eY = pX+pZ` ⇒ `pX=(eZ+eY−eX)/2`, `pY=(eZ+eX−eY)/2`, `pZ=(eX+eY−eZ)/2`, `pI=1−(eX+eY+eZ)/2`. χ² test vs calibrated baseline; **negative p_i beyond error ⇒ basis-aware adaptive adversary alarm**. |
| **D3** | Bell-Outcome Uniformity Audit | fake/biased source, tampered correction stream | Honest teleportation gives uniform Bell outcomes over {00,01,10,11}; χ² with 3 dof. |
| **D4** | CHSH Entanglement Witness | fake / degraded entanglement | Sacrifice a sample of pairs, estimate CHSH `S` (classical ≤ 2, ideal 2√2) and Bell fidelity `F = (1+⟨XX⟩−⟨YY⟩+⟨ZZ⟩)/4`. Block distribution if below threshold. |
| **D5** | Freshness & Replay Guard (Q-Ledger) | replay | Each key block used once. Record `H(nonce ‖ key-block-ID ‖ msg-hash ‖ verdict)`, SHA3-256 hash chain, Merkle-root batches. Spent-block replay caught deterministically; cross-block replay gives ~50% mismatch → D1. |

**Fingerprint table (D2):**

| Attack | eX, eY, eZ | Recovered channel |
|---|---|---|
| Z-basis intercept–resend | ½, ½, 0 | pZ = ½ (full dephasing) |
| Bit-flip injection (rate q) | 0, q, q | pX = q |
| Random-basis intercept–resend | ⅓, ⅓, ⅓ | isotropic, large |
| Optimal universal cloning | ≈⅙ each | isotropic, medium |
| Benign fibre noise | small, ≈ calibration | matches baseline |

### 2.3 Fusion — deterministic threat verdict (no ML)

- Detector p-values combined with **Fisher's method** → `χ²_{2k}` → one **Quantum Threat Index**
  (QTI ∈ [0,1], = 1 − p_combined).
- A **fixed, auditable decision table** maps the detector pattern to an attack class, e.g.
  `D1 high + D2 isotropic + D4 normal → FORGERY/IMPERSONATION`,
  `D4 low + D3 non-uniform → COMPROMISED ENTANGLEMENT SOURCE`,
  `D2 anisotropic (pZ≫) → CHANNEL MANIPULATION: Z-basis intercept-resend`, etc.
- **Wald SPRT early stop**: verification ends when the log-likelihood ratio crosses
  `A = ln((1−β)/α)` or `B = ln(β/(1−α))`; far fewer qubits on average than fixed-length.
  (`e0` must be > 0 → floor at 1e-3 for ideal channel.)

### 2.4 Threat coverage matrix

| Threat | Primary detector | Mechanism |
|---|---|---|
| Forgery | D1 (+D2) | mismatch count > Hoeffding threshold |
| Impersonation | D1 + MAC | no private key → ~1/3 mismatch; authenticated corrections |
| Replay | D5 (+D1) | spent-key ledger, nonce binding |
| Unauthorized verification | teleportation lock | no correction bits → uniform random record |
| Quantum channel manipulation | D2 | Pauli-channel fingerprint |
| Fake / degraded entanglement | D3 + D4 | χ² uniformity, CHSH / fidelity |
| Repudiation by signer | symmetrization + s_a/s_v gap | cross-verifier consistency |

### 2.5 Attack digital twin (9 adversaries + noise)

1. random forger · 2. measure-and-guess forger · 3. optimal-cloning forger · 4. impersonator without key ·
5. replay attacker (same-block / cross-block) · 6. unauthorized verifier (no correction bits) ·
7. intercept–resend in Z / X / Y / random basis · 8. fake Bell source (separable states) ·
9. correction-bit tamperer. Plus **benign** profile: depolarizing / dephasing / amplitude-damping-like
(Pauli-twirled) noise for false-alarm measurement.

---

## 3. Implementation Status  ← resume here

| # | Module | File(s) | Status | Notes |
|---|---|---|---|---|
| M0 | Pauli algebra + Pauli-frame correction + eigenstate helpers | `qtrinetra/core/pauli.py` | TODO | `pauli_frame_correct`, `FLIP`, `measure_eigenstate_in_basis` |
| M0 | Wegman–Carter MAC (polynomial universal hashing over GF(2^64)) | `qtrinetra/core/mac.py` | TODO | one-time-key, info-theoretic |
| M1 | Entanglement & teleportation engine | `qtrinetra/engine/teleport.py` | TODO | Stim circuit (Bell pair + Bell measurement + verifier measurement in chosen basis) and a fast exact classical "stabilizer-table" path; CHSH sampler |
| M2 | QDS protocol core (KeyGen/Distribute/Sign/Verify/Transfer/Symmetrize) | `qtrinetra/protocol/qds.py` | TODO | dataclasses: `PrivateKey`, `Signature`, `VerifierRecord`, `Verdict` |
| M3 | Detectors D1–D5 | `qtrinetra/detect/d1_mismatch.py` … `d5_replay.py` | TODO | each returns `DetectorResult(p_value, stats, flag)` |
| M3 | SPRT, Fisher fusion, decision table | `qtrinetra/detect/sprt.py`, `fusion.py` | TODO | `ThreatReport` with class, QTI, confidence |
| M4 | Attack digital twin | `qtrinetra/attacks/adversaries.py`, `noise.py` | TODO | 9 adversaries + benign noise |
| M5 | Q-Ledger | `qtrinetra/ledger/qledger.py` | TODO | SHA3-256 chain + Merkle root, SQLite |
| M6 | Analytics / benchmarks | `qtrinetra/analytics/evaluate.py`, `python -m qtrinetra.cli bench` | TODO | forgery-prob vs L, SPRT qubit savings, FAR/FRR, confusion matrix |
| API | FastAPI orchestrator (REST + WebSocket) | `backend/app.py` | TODO | `/api/run`, `/api/demo`, `/api/ledger`, `/ws/live` |
| UI | Dashboard (Pauli radar, QTI gauge, SPRT curve, CHSH meter, ledger explorer) | `frontend/index.html` (+ Plotly CDN) | TODO | single-page, no build step |
| QA | Tests (pytest + Hypothesis), No-ML CI gate | `tests/`, `.github/workflows/ci.yml`, `scripts/no_ml_gate.py` | TODO | property test: honest signature always accepted at zero noise |
| Ops | Docker / compose | `Dockerfile`, `docker-compose.yml` | TODO | |
| Docs | Math model | `docs/MATH_MODEL.md` | TODO | Hoeffding/SPRT bounds, Pauli inversion derivation |

(Status is updated by the agent as modules land. If a row says `WIP`, check `git log` for the last
commit touching that file.)

---

## 4. Repository Layout

```
Q-TRINETRA/
├── README.md                   ← this file (idea + plan + status)
├── requirements.txt
├── pyproject.toml
├── Dockerfile / docker-compose.yml
├── qtrinetra/                  ← Python package (the framework)
│   ├── core/      pauli.py, mac.py
│   ├── engine/    teleport.py          (M1: Stim + CHSH)
│   ├── protocol/  qds.py               (M2)
│   ├── detect/    d1_mismatch.py d2_pauli_inversion.py d3_bell_uniformity.py
│   │              d4_chsh.py d5_replay.py sprt.py fusion.py   (M3)
│   ├── attacks/   adversaries.py noise.py                     (M4)
│   ├── ledger/    qledger.py                                  (M5)
│   ├── analytics/ evaluate.py                                 (M6)
│   ├── pipeline.py   ← one call: run_session(scenario) → full ThreatReport
│   └── cli.py        ← `python -m qtrinetra.cli demo|bench|run <attack>`
├── backend/app.py              ← FastAPI (REST + WS) serving frontend/
├── frontend/index.html         ← dashboard (Plotly via CDN)
├── tests/                      ← pytest + hypothesis
├── scripts/no_ml_gate.py       ← fails CI if any ML lib is in deps
├── docs/MATH_MODEL.md
└── .github/workflows/ci.yml
```

---

## 5. How the Prototype Was Built (implementation approach)

1. **Everything in the protocol is Clifford** (Bell pairs, Bell measurement, Pauli corrections,
   Pauli-basis measurements). So the engine has two interchangeable back-ends:
   - `stim` — builds the real teleportation circuit per qubit batch and samples it (proof that the
     physics is simulated, not faked), and
   - `fast` — an exact closed-form stabilizer-table path: for eigenstate `(a,v)` teleported with
     Bell outcome `(m1,m2)` and measured in basis `c`, the outcome is `v ⊕ flip(c,m1,m2)` if `c == a`,
     else a fair coin. This is exactly what Stim produces, but at millions of qubits/sec in NumPy.
   Tests assert the two back-ends agree statistically.
2. **Attacks are modelled as operations on the qubit stream**: a Pauli channel `(pI,pX,pY,pZ)` for
   channel attacks / noise; intercept–resend = measure in basis `β`, re-prepare eigenstate; forgers
   replace the revealed key; replay re-submits a spent `(key_block_id, signature)`; fake source
   replaces `|Φ+⟩` by a separable mixture `½(|00⟩⟨00|+|11⟩⟨11|)` (for which CHSH ≤ 2 and Bell
   outcomes are *still* uniform, so D4, not D3, catches it — D3 catches a biased/tampered stream).
3. **Detectors are pure statistics** (`scipy.stats`): binomial tail / Hoeffding (D1), channel
   inversion + χ² vs baseline (D2), χ² 3-dof (D3), CHSH with standard error → one-sided z-test (D4),
   ledger lookup (D5, p = 0 or 1).
4. **Fusion**: Fisher `−2Σ ln p_i ~ χ²_{2k}` → QTI; decision table in `fusion.py` is a plain
   ordered list of boolean rules (auditable, printed in the UI).
5. **Dashboard** is a static page + Plotly fed by FastAPI; a WebSocket streams the SPRT LLR so the
   curve animates live.

---

## 6. Evaluation plan & metrics (implemented in `qtrinetra/analytics/evaluate.py`)

- Forgery acceptance probability vs `L` (simulated vs Hoeffding bound, log scale)
- Honest rejection rate vs channel noise (deterministic acceptance at zero noise)
- Attack-fingerprint confusion matrix from the decision table
- Average qubits checked: SPRT vs fixed-length
- CHSH `S` distribution: honest vs fake source
- Throughput: signatures/s, qubits/s (fast vs Stim)
- Complexity: O(L) verification, O(1) correction/qubit, O(log N) Merkle proofs

Worked example to confirm: `e0=0.01, s_a=0.08, p_min=1/6, Lc≈1600` ⇒ forgery ≤ 1e-10, honest
rejection ≈ 1e-5.

---

## 7. Dev Workflow / Conventions

- Python 3.11+, type-hinted, dataclasses, no ML libraries (CI gate enforces: torch, tensorflow,
  sklearn, keras, jax, xgboost, lightgbm, transformers are forbidden).
- `pytest -q` must pass before every commit.
- Branch `genspark_ai_developer` → PR → `main`. Conventional commits (`feat(detect): …`).
- Keep this README's §3 status table current.

---

## 8. Demo script (2 minutes) — `python -m qtrinetra.cli demo`

1. Honest signature → `M = 0` → ACCEPT; Pauli radar flat & green.
2. Forger → SPRT rejects within ~100 qubits; Threat Index red.
3. Intercept–resend in Z → radar shows Z-dephasing fingerprint → "Channel manipulation: Z-basis".
4. Fake entanglement source → CHSH ≤ 2 → distribution blocked before any key is sent.
5. Replayed old signature → Q-Ledger flags spent key block instantly.

---

## 9. References

Gottesman & Chuang (2001) QDS · Bennett et al. PRL 70 (1993) teleportation · Clarke et al. Nat.
Commun. (2012) · Wallden et al. PRA (2015) · Amiri et al. PRA (2016) symmetrization · CHSH (1969) ·
Wald SPRT (1945) · Hoeffding (1963) · Wegman & Carter (1981) · Gidney, Stim (Quantum 2021) ·
Zhao et al. Results in Physics (2023) · Entropy 28(9):1030.
