# Q-Trinetra — Mathematical Model

All quantities here are implemented and checked in `tests/` (see `test_hoeffding_worked_example`,
`test_pauli_inversion_is_exact_inverse_of_forward_model`, `test_sprt_decisions_and_early_stop`).

## 1. States, teleportation and the Pauli frame

Public-key qubits are the six Pauli eigenstates |a, v⟩, a ∈ {X, Y, Z}, v ∈ {0, 1}.
Teleportation through |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 with Bell outcome (m₁, m₂) delivers

  X^{m₂} Z^{m₁} |a, v⟩.

Because every Pauli either commutes or anticommutes with the basis Pauli σ_a, the output is still an
eigenstate of σ_a; its eigenvalue flips iff the applied correction anticommutes with σ_a:

| basis a | flip bit |
|---|---|
| X | m₁ (Z anticommutes with X) |
| Z | m₂ (X anticommutes with Z) |
| Y | m₁ ⊕ m₂ |

Hence the verifier may measure immediately in a random basis c and later XOR the recorded bit with the
flip bit — O(1) classical work, no quantum memory. Without (m₁, m₂) the record is uniformly random
(**teleportation lock**), since m₁, m₂ are uniform and independent of |ψ⟩.

The Stim circuit used per qubit (`qtrinetra/engine/teleport.py::_teleport_circuit`):

```
R 0 1 2          # q0 message, q1 Alice half, q2 Bob half
<prep a,v on 0>  # X / H / H S
H 1 ; CX 1 2     # |Φ+⟩
[PAULI_CHANNEL_1(pX,pY,pZ) 2]   # adversary / noise on the flying qubit
CX 0 1 ; H 0 ; M 1 ; M 0        # Bell measurement → (m2, m1)
M / MX / MY 2                   # Bob measures in basis c
```

## 2. Verification statistics (D1)

Only positions with c_i = a_i are checked; Lc ≈ L/3. M = number of mismatches.
Decision: ACCEPT if M ≤ s_a·Lc; TRANSFERABLE-UNCERTAIN if s_a·Lc < M ≤ s_v·Lc; REJECT otherwise.

- Honest, noiseless: each checked position agrees deterministically ⇒ M = 0 ⇒ P(accept) = 1.
- Honest with Pauli channel error rate e₀ per checked position: M ~ Bin(Lc, e₀).
- Key-less adversary: must guess (a, v). Measure-and-guess in a random basis is right w.p. 1/3 on the
  basis and then outcome-correct; otherwise the outcome is a coin ⇒ mismatch 1/3. We use the
  conservative worst-case p_min = 1/6.

Hoeffding:
  P(forgery accepted) ≤ exp(−2·Lc·(p_min − s_a)²),  P(honest rejected) ≤ exp(−2·Lc·(s_a − e₀)²).

Worked example (verified in tests): e₀ = 0.01, s_a = 0.08, p_min = 1/6, Lc = 1600 ⇒
forgery ≤ 1.0·10⁻¹⁰ (exact: exp(−24.0) = 3.8·10⁻¹¹), honest rejection ≤ 1.5·10⁻⁵.
Required Lc for 10⁻¹⁰: ⌈ln(10¹⁰)/(2·(1/6 − 0.08)²)⌉ = 1534.

## 3. Pauli channel inversion (D2)

For a Pauli channel 𝓔(ρ) = Σ_P p_P P ρ P, the eigenvalue of an eigenstate of σ_a flips iff P ∉ {I, σ_a}:

  e_X = p_Y + p_Z,  e_Y = p_X + p_Z,  e_Z = p_X + p_Y.

Linear system with determinant −2 ⇒ unique inverse

  p_X = (e_Z + e_Y − e_X)/2,  p_Y = (e_Z + e_X − e_Y)/2,  p_Z = (e_X + e_Y − e_Z)/2,  p_I = 1 − (e_X + e_Y + e_Z)/2.

Standard error of each p̂ ≈ ½·√(Σ_a ê_a(1−ê_a)/n_a). A recovered p̂ < −3σ is impossible for a Pauli
channel ⇒ adversary is basis-aware (not twirled) ⇒ alarm. Random basis choice on both ends
Pauli-twirls any non-adaptive channel, which justifies the Pauli model.

Fingerprints: Z-intercept-resend measures in Z and re-prepares ⇒ complete dephasing ⇒ p_Z = ½ ⇒
(e_X, e_Y, e_Z) = (½, ½, 0). Bit-flip rate q ⇒ (0, q, q). Random-basis intercept-resend ⇒ p_X = p_Y = p_Z
= 1/6 ⇒ e = 1/3 each. Optimal 1→2 universal cloning (F = 5/6) ⇒ depolarizing p = 1/4 ⇒ e = 1/6 each.

Baseline test: χ² with 3 dof of observed per-basis mismatch counts against expected counts under
the calibrated channel.

## 4. Bell-outcome uniformity (D3)

For any input state, P(m₁, m₂) = ¼. χ² = Σ (n_k − N/4)²/(N/4) ~ χ²₃ under H₀.

## 5. CHSH witness (D4)

Settings a₀ = Z, a₁ = X, b₀ = (Z+X)/√2, b₁ = (Z−X)/√2. For |Φ⁺⟩ the correlation matrix is
T = diag(+1, −1, +1), so S = E(a₀b₀) + E(a₀b₁) + E(a₁b₀) − E(a₁b₁) = 2√2. A separable
½(|00⟩⟨00| + |11⟩⟨11|) source has T = diag(0, 0, 1) ⇒ S = √2 ≤ 2 and F = (1 + ⟨XX⟩ − ⟨YY⟩ + ⟨ZZ⟩)/4 = ½.
se(S) ≈ 2/√n per setting; one-sided z-test against the trust threshold S* = 2.4.

## 6. SPRT (Wald)

Per checked qubit, LLR increment = ln(p_min/e₀) on mismatch, ln((1−p_min)/(1−e₀)) on match.
Stop at A = ln((1−β)/α) → REJECT, B = ln(β/(1−α)) → ACCEPT. With α = 10⁻⁶, β = 10⁻¹⁰, e₀ = 0.01:
E[N | honest] ≈ 160, E[N | forger at 1/3] ≈ 25 — vs 1600 for the fixed-length test (≈ 90–98 % saving).

## 7. Fusion

Fisher: X = −2 Σᵢ ln pᵢ ~ χ²_{2k} under H₀ (k = 5). QTI = 1 − P(χ²₁₀ ≥ X). The decision table is an
ordered list of boolean predicates over the detector flags and fingerprint class (`fusion.py`), first
match wins; it is exported through `/api/decision-table` for audit.

## 8. Authentication and ledger

Wegman–Carter with polynomial hashing over GF(2⁶¹ − 1): forgery probability ≤ n_blocks/p ≈ 4·10⁻¹⁹ per
tag regardless of adversary power. Ledger records h = SHA3-256(nonce‖block‖msg-hash‖verdict),
chain_i = SHA3-256(chain_{i−1}‖h_i), Merkle roots every 8 records; inclusion proofs are O(log N).

## 9. Complexity

KeyGen O(L); distribution O(L) teleportations (Clifford, stabilizer-simulable); correction O(1)/qubit;
verification O(L) (expected O(E[N]) with SPRT); ledger lookup O(1), proof O(log N).
