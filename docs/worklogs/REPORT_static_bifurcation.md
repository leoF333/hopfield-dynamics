# Static bifurcation of the delayed mixed Hopfield network
### Full methodology, numerical techniques, choices, and results

**Author:** numerical agent (Claude Code) · **Date:** 2026-07-09 (rev. 6 — adds §5.9 large-scale scaling campaign E30: σ ~ N^{−1/2} CLT law at fixed α confirmed up to N=14000, gap→0 verified, mechanism identified as a central limit theorem of the quenched cross-talk; rev. 5 added §5.8 finite-size statistics; rev. 4 §5.7 memory→front unification; rev. 3 §5.6 large-τ Hopf crossover, §9 figure inventory)
**Code:** `numerics/src/` · **Figures/data:** `numerics/results/`
**Companions:** `CYCLE_worklog.md` (limit-cycle/Floquet study, self-sufficient) · `FLOQUET_methodology.md` (early methodology draft).

> **Scope.** Self-contained for a scientist comfortable with linear algebra and
> ODEs but not assumed expert in bifurcation theory or delay-equation functional
> analysis. §1–2 build the theory from scratch; §3 documents every numerical
> method, technique and choice (including two exact solvers exploiting the
> problem's low-rank structure, and a hard-won float64 lesson); §4 validates; §5
> gives **all** results — the λ_c(α) curve, the finite-size behaviour, the
> τ-independence, and the near-capacity breakdown; §6–7 give the verdict and its
> limits; §8 points to the dynamic (cycle) continuation. Nothing is simplified
> away: the infinite DDE spectrum, the exact fold↔zero-root coincidence, the
> saddle-node/SNIC ambiguity, spurious pseudospectral modes, the GMRES
> near-capacity failure, and the intrinsic ill-definedness of the fold near the
> Hopfield capacity are all treated explicitly.

---

## 1. Model and the static problem

The state $u(t)\in\mathbb R^N$ obeys a first-order delay differential equation:
$$
t_0\,\dot{u}(t) = -u(t) + (1-\lambda)\,J\,\tanh\!\big(\beta u(t)\big)
                       + \lambda\,K\,\tanh\!\big(\beta u(t-\tau)\big),
\tag{1}
$$
with $t_0$ the time constant, $\beta$ the gain, $\tau>0$ the delay, and
$\lambda\in[0,1]$ mixing symmetric and asymmetric couplings built from $P=\alpha N$
random patterns $\xi^\mu\in\{\pm1\}^N$:
$$
J=\frac1N\sum_{\mu=1}^P \xi^\mu(\xi^\mu)^{\mathsf T},\qquad
K=\frac1N\sum_{\mu=1}^P \xi^{\mu+1}(\xi^\mu)^{\mathsf T},\qquad
C(\lambda)=(1-\lambda)J+\lambda K .
\tag{2}
$$
$J$ stabilizes static memories; $K$ pushes the state along the stored cycle
$\xi^1\!\to\!\xi^2\!\to\!\cdots$. $\lambda$ tunes **standing memory** vs **moving
sequence**. Reference parameters: $\beta=20$ ($T=0.05$), $t_0=1$, $\tau=10$.

### 1.1 Fixed points do not see the delay

A steady state $u(t)\equiv u^\*$ has $u(t-\tau)=u(t)=u^\*$, so the delay drops out:
$$
\boxed{\,F(u^\*,\lambda) = -u^\* + C(\lambda)\,\tanh(\beta u^\*) = 0\, }.
\tag{3}
$$
Hence the fixed-point set is **τ-independent**: the delay affects only *how*, not
*where*, a state loses stability. The **memory branch** $u^\*(\lambda)$ starts from
$\xi^1$ at $\lambda=0$ and is continued in $\lambda$.

### 1.2 The question

As $\lambda$ grows, $K$ destabilizes the standing memory. **By what mechanism —
Hopf (birth of oscillation) or saddle-node/SNIC (the state ceases to exist)?** And
**at which $\lambda_c(\alpha,\tau)$?** These require the theory below.

---

## 2. Theoretical background

### 2.1 Linear stability

For $\dot x=f(x)$ with $f(x^\*)=0$, perturbations $\delta x(t)=e^{zt}v$ obey
$\mathcal J v=zv$ with $\mathcal J=Df(x^\*)$; stability ⟺ every eigenvalue has
$\mathrm{Re}(z)<0$. A **bifurcation** occurs when the rightmost eigenvalue crosses
$\mathrm{Re}(z)=0$; *how* it crosses names the type.

### 2.2 The three relevant bifurcations

Real-operator eigenvalues are real or conjugate pairs, giving two generic crossings
plus one global variant:

- **(a) Saddle-node (fold).** A single *real* eigenvalue through 0. Normal form
  $\dot x=\mu-x^2$: two equilibria for $\mu>0$ collide at $\mu=0$ and vanish — the
  Jacobian is **singular** at the fold; the state **disappears**.
- **(b) Hopf.** A conjugate *pair* $\sigma\pm i\omega$, $\omega\neq0$, crosses; a
  small **limit cycle** of frequency $\approx\omega$ is born.
- **(c) SNIC** (saddle-node on an invariant circle). A saddle-node *on a closed
  invariant curve*: above threshold a **limit cycle** with **diverging period**
  $\sim(\lambda-\lambda_c)^{-1/2}$ is freed ($\omega\to0$ at onset).

> **Why (a) and (c) are indistinguishable by the linear spectrum.** Both show one
> real eigenvalue through 0; the difference — whether the colliding equilibria lie
> on an invariant circle — is *global* and invisible to the fixed-point
> linearization. Hence a real crossing is honestly labelled **"saddle-node /
> SNIC"**: the linear spectrum *rules out Hopf* and *locates the crossing*, but SN
> vs SNIC needs the global/dynamic analysis of §8.

### 2.3 Why the delay changes the problem

Linearizing (1) about $u^\*$ with the diagonal gain
$D=\mathrm{diag}(\beta(1-\tanh^2\beta u^\*))$ and ansatz $\delta u=e^{zt}v$ gives the
**characteristic matrix**
$$
\boxed{\;\Delta(z)=(t_0z+1)I-(1-\lambda)JD-\lambda KD\,e^{-z\tau}=0\;}
\tag{4}
$$
(meaning $\det\Delta(z)=0$). The $e^{-z\tau}$ makes this **transcendental**: unlike
an ODE ($N$ eigenvalues), the DDE has a **countably infinite** set of characteristic
roots, discrete, with real parts bounded above, accumulating at
$\mathrm{Re}(z)\to-\infty$. Stability ⟺ every root has $\mathrm{Re}(z)<0$; a rightmost
root always exists, so stability is decidable.

### 2.4 The DDE as an operator problem

The genuine state is the history $u_t\in X=C([-\tau,0],\mathbb R^N)$. The flow's
**infinitesimal generator** $\mathcal A$ acts as $(\mathcal A\phi)(\theta)=\phi'(\theta)$
with a boundary condition at $\theta=0$ encoding (1). Its eigenvalues are **exactly**
the roots of (4) (eigenfunctions $\phi(\theta)=e^{z\theta}v$). So "rightmost
characteristic roots" = "rightmost eigenvalues of $\mathcal A$" — the object our
pseudospectral method (§3.4) discretizes.

---

## 3. Numerical methods, techniques, and choices

### 3.1 Matrix-free couplings (low rank)

$J,K$ are $N\times N$ but rank $\le P=\alpha N$. We never form them: with
$m=\tfrac1N\,\xi x\in\mathbb R^P$,
$$
C(\lambda)x=(1-\lambda)\,\xi^{\mathsf T}m+\lambda\,\xi_{\text{shift}}^{\mathsf T}m,
$$
cost $O(NP)$, memory $O(NP)$. Implemented in `couplings.py`. **Choice:** for the
production static analysis we use the **NumPy float64** path (see §3.7); the MLX
float32 GPU path exists but is unsafe for the continuation.

### 3.2 Two exact solvers from the low-rank structure

The Jacobian of (3) is $M(\lambda)=-I+C(\lambda)D$. Because $C\!D$ is **rank $\le P$**,
two operations that would otherwise be expensive/iterative become **exact and cheap**.
Write $C(\lambda)D=\tfrac1N UV$ with
$$
U=\big((1-\lambda)\xi+\lambda\,\xi_{\text{shift}}\big)^{\mathsf T}\ (N\times P),\qquad
V=\xi\,\mathrm{diag}(\text{gain})\ (P\times N).
$$

**(i) Woodbury — exact Newton solve.** $M=-I+\tfrac1N UV$, so
$$
\boxed{\;M^{-1}r=-r-U'\big(I_P-VU'\big)^{-1}(Vr),\qquad U'=U/N\;}
\tag{5}
$$
a single $P\times P$ solve, cost $O(NP^2+P^3)$, **no iterative convergence to fail**.
This is the key to robustness near the Hopfield capacity (§3.3). Validated to match a
dense $M^{-1}$ to machine precision.

**(ii) Sylvester — exact rightmost eigenvalue of $M$.** The nonzero eigenvalues of
$C\!D=\tfrac1N UV$ equal those of $\tfrac1N VU$ ($P\times P$), so
$$
\boxed{\;\mathrm{eig\_max}(M)=-1+\max\mathrm{Re}\,\mathrm{eig}\!\big(\tfrac1N VU\big)\;}
\tag{6}
$$
— exact and cheap at any $N$. The **static fold** is exactly where $\mathrm{eig\_max}(M)=0$
(see §3.5). Verified equal to dense `eigvals(M)` at every tested point. Both in
`robust_branch.py`.

### 3.3 The fixed-point branch: from pseudo-arclength to the robust Woodbury tracer

**Original method (pseudo-arclength continuation, `fixed_point.py`).** Newton–Krylov
(JFNK) with GMRES on $M$, Keller bordering to traverse folds, adaptive arclength.
This works well **below the Hopfield capacity** (small $\alpha$).

**Failure near capacity.** For $\alpha=0.10$ ($\alpha_c\approx0.138$, a spin-glassy
regime), $M$ is ill-conditioned; GMRES stalls, Newton leaves the memory basin, the
step collapses, and the fold-detector fires a **false fold** (observed
$\lambda_{\text{fold}}\approx0.05$–$0.11$ instead of the true $\approx0.18$ at $N=1000$).
Diagnosed against a dense-Newton + dense-`eigvals(M)` gold standard.

**Robust replacement (`robust_branch.py`).** Warm-start (previous solution) + **direct
Newton with the exact Woodbury step (5)** + backtracking line search + small adaptive
$ds$. The fold is detected two equivalent ways (identity $\Delta(0)=-M$, §3.5): either
$\mathrm{eig\_max}(M)$ (6) crosses 0, or Newton stalls because $M$ goes singular *at*
the fold. A tangent predictor was tried and **removed** — near capacity it drifts onto
spurious nearby $m_1\!\approx\!1$ solutions.

**β-annealing (a technique, not a hack).** At $\beta=20$ the $\tanh$ is nearly a step
function and direct Newton from the raw pattern **stalls** (residual stuck at
$\sim10^{-3}$, e.g. $\alpha=0.10,N=10^4$). Warm-starting up a schedule
$\beta=5\!\to\!8\!\to\!12\!\to\!16\!\to\!20$ — where the fixed-point map is smooth —
converges to $\sim10^{-13}$. Applied to the initial solve everywhere.

### 3.4 The DDE spectrum: pseudospectral infinitesimal-generator operator

To find the rightmost roots of (4) (needed to *rule out Hopf*), we discretize $\mathcal A$
by **infinitesimal-generator collocation** (Breda–Maset–Vermiglio 2005, `dde_stability.py`):
Chebyshev–Gauss–Lobatto nodes on $[-\tau,0]$; interior rows apply $(2/\tau)D_{\rm cheb}$
(differentiation); the boundary row at $\theta=0$ imposes the linearized DDE
$-\delta u+(1-\lambda)JD\,\delta u(t)+\lambda KD\,\delta u(t-\tau)$. Eigenfunctions
$e^{z\theta}v$ are analytic ⇒ **spectral convergence**; a small $M=16$–$24$ resolves the
dominant roots to machine level. The leading eigenvalues (largest real part) are
extracted matrix-free by **Arnoldi** (`scipy.sparse.linalg.eigs`, `which='LR'`), with
**partial eigenvalues used on ARPACK non-convergence** (the rightmost converge first).

**Spurious modes (a known artefact, handled).** The finite operator has $N(M{+}1)$
eigenvalues; only $O(1)$ approximate true roots. A $\lambda$-independent cluster at
$\mathrm{Re}\approx-0.43,\ |\mathrm{Im}|\approx3.85$ (for $\tau=10$) is a discretization
artefact — recognizable because it does not move with $\lambda$ and does not converge
under $M$-refinement. §4 verifies these are *not* roots of (4) while the physical modes
*are*.

### 3.4b The exact reduced spectrum (artefact-free)

The same rank-$P$ structure of §3.2 lets us **remove the spurious modes entirely** for the
*full* $z$-dependent spectrum, not just at $z=0$. Since $C(\lambda)D=\tfrac1N U(z)V$ with
$U(z)=((1-\lambda)\xi+\lambda\,\xi_{\rm shift}e^{-z\tau})^{\mathsf T}$, $V=\xi D$, the
Weinstein–Aronszajn identity factorizes the characteristic determinant exactly:
$$
\det\Delta(z)=(t_0z+1)^{N-P}\,\det T_P(z),\qquad
T_P(z)=(t_0z+1)I_P-(1-\lambda)G_J-\lambda e^{-z\tau}G_K,
\tag{4b}
$$
with $G_J=\tfrac1N\xi D\xi^{\mathsf T}$, $G_K=\tfrac1N\xi D\xi_{\rm shift}^{\mathsf T}$ ($P\times P$).
The $(t_0z+1)^{N-P}$ factor is the trivial bulk root $z=-1/t_0$; **all physical characteristic
roots solve the $P\times P$ transcendental problem $\det T_P(z)=0$, which — having no
discretization — carries no spurious modes.** This factorization is *exact at any $\alpha<1$*
(it is not a mean-field/condensed reduction); its only cost saving is $N\!\to\!P$, i.e.
$\sim(1/\alpha)^3$ in the determinant, still large at $\alpha=0.05$ ($P{=}500$ vs $N{=}10^4$).

Finding the *rightmost* roots of (4b) is still a genuine transcendental root-hunt (infinitely
many roots). We use a **hybrid**: a small reduced-DDE IG operator (dimension $P(M{+}1)$, using
$G_J,G_K$) supplies rightmost-first *candidates*; each is kept iff $\sigma_{\min}(T_P(z))$ is a
genuine root (small vs the scale $|t_0z+1|$; spurious give $O(1)$) and then **polished** on
$T_P$. Kept roots are verified against the *dense* $\Delta$ to $\sigma_{\min}\approx10^{-17}$
(§4). This is `reduced_spectrum.py`; it produces the artefact-free **Fig. 2** (physical roots
near $\lambda_c$).

### 3.5 The discriminator — an exact identity

At a fold, $M(\lambda)$ is singular. Evaluate (4) at $z=0$:
$$
\Delta(0)=I-C(\lambda)D=-\,M(\lambda)\quad\Rightarrow\quad
\boxed{\;\det\Delta(0)=0\iff\det M(\lambda)=0\iff\text{fold}\;}
\tag{7}
$$
and, since $e^{-0\cdot\tau}=1$, **τ-independent**. Consequences: (i) "$\max\mathrm{Re}(z)\to0$
at the fold" is *guaranteed* and is **no evidence** of a saddle-node — it holds for any
fold, SNIC included; (ii) the only non-fold linear signature is a **complex pair**
crossing at $\lambda_{\rm Hopf}<\lambda_{\rm fold}$ (a Hopf). The analysis therefore tracks
the rightmost **real** and **complex** modes separately (`|Im|` split at $10^{-3}$):
Hopf iff the complex class crosses first, else saddle-node/SNIC at the fold.
The two classes are visualised in `results/1_bifurcation_statique/figures/fig3_maxre_N2000_a0.050_tau10_b20.0_s42.png`
(margin climbing to 0 at the fold) and, root-by-root in the complex plane, in
`results/1_bifurcation_statique/figures/fig2_roots_panel_N2000_a0.050_tau10_b20.0_s42.png` (IG spectrum, N=2000)
and `results/1_bifurcation_statique/figures/fig2_physical_roots_near_lc.png` (exact reduced spectrum, N=10⁴).

### 3.6 Finite-size scaling and multi-seed averaging

The random couplings give $O(1/\sqrt N)$ sample fluctuations, so a single realization
can mislead. Two orthogonal sweeps: **finite-size scaling** (fixed seed, several $N$;
`finite_size_scaling.py`) and **multi-seed** (fixed $N$, several seeds, mean ± std;
`multiseed_fold.py`, seeds run concurrently across CPU cores via `ProcessPoolExecutor`).
Multi-seed is *essential* near capacity, where the fold fluctuates violently (§5.4).

### 3.7 Numerical-precision choice: float64, not GPU float32 (a hard lesson)

The MLX (Apple-Silicon) path uses **float32** (matvec floor $\sim10^{-5}$). For the
*eigenvalue* spectrum that is tolerable, but for the **continuation** it is fatal at
large $N$: GMRES/Newton tolerances ($10^{-6}/10^{-8}$) are unreachable, so the corrector
stalls and reports **false folds** — an initial $N=10^4$ run was thereby *invalid*
($\alpha=0.05$ folded low and scattered; $\alpha=0.10$ failed entirely). Switching the
couplings to **NumPy float64** fixed the branch tracing (and is fast — the low-rank
matvecs are cheap; the Woodbury solve is $\sim13$ s at $N=10^4$). **Rule adopted:** the
fixed-point continuation runs in float64 on CPU; GPU float32 is reserved for
precision-tolerant, matvec-heavy *dynamics* (the cycle simulations of §8).

---

## 4. Validation

| Check | Result | Status |
|---|---|---|
| Chebyshev differentiation, $M=16$ | $\max\lvert D\cos-(-\sin)\rvert=2.6\times10^{-14}$ | spectral ✓ |
| Matrix-free $M$ vs dense ($N{=}100$) | rel. error $1.8\times10^{-16}$ | ✓ |
| Newton at $\lambda=0$ | $\lVert F\rVert/\sqrt N=1.2\times10^{-18}$, $m_1=1.0000$ | ✓ |
| **IG eigenvalues are true roots** | $\sigma_{\min}(\Delta(z_{\rm IG}))\approx10^{-14}$ (dense, $\tau=10$) | ✓✓ |
| Spurious modes are *not* roots | $\sigma_{\min}\approx1.26$ (control $z{=}0.5$: $1.17$) | ✓ |
| Woodbury $M^{-1}$ vs dense | fold reproduced to $10^{-4}$ ($\alpha{=}0.10,N{=}1000$: 0.1839 vs 0.184) | ✓✓ |
| Sylvester $\mathrm{eig\_max}(M)$ vs dense `eigvals` | equal at every point | ✓✓ |
| $\tau\to0$ reduction (IG vs dense ODE) | $\max\mathrm{Re}$ agree to $0.30$ on $O(2.5)$ | ✓ |
| M-convergence of leading root | $z_1=-0.215911$, $M$-independent to 6 digits ($M\ge8$) | ✓ |

The **σ_min test is decisive**: forming $\Delta(z)$ densely at the IG eigenvalues and
checking it is singular *proves* the operator computes genuine characteristic roots at
$\tau>0$, and simultaneously certifies the spurious cluster as artefacts.

---

## 5. Results

All at $\tau=10,\ \beta=20,\ t_0=1$ unless noted.

### 5.1 The memory branch and its fold ($N=2000$, α=0.05)

The retrieval branch keeps $m_1\approx1$, $m_{2,3}\approx0$ and **ends in a fold at
$\lambda\approx0.284$** (the $m_1$-vs-$\lambda$ curl is shown per seed in
`results/1_bifurcation_statique/figures/fold_N10000_a0.05.png`, right panel). The stability margin
$\max_k\mathrm{Re}(z_k)$ (`results/1_bifurcation_statique/figures/fig3_maxre_N2000_a0.050_tau10_b20.0_s42.png`)
shows the $\lambda$-independent spurious plateau at
$\approx-0.45$, then the **physical real mode climbing to 0 at the fold**,
$\lambda_c\approx0.282$ — a *real* crossing, hence **saddle-node / SNIC** by (7).

**Fig. 2 = `results/1_bifurcation_statique/figures/fig2_physical_roots_near_lc.png` (exact physical roots
near $\lambda_c$, $N=10^4$).** Using the artefact-free
reduced spectrum of §3.4b, the 10 rightmost *physical* roots are shown at six $\lambda$
clustered below the fold ($\lambda/\lambda_c$ from 0.75 to 1.00). The **real fold-mode
(red) climbs monotonically to the axis** ($-0.175\to-0.059\to-0.035\to0$) while the
**complex modes (blue) stay left** ($\mathrm{Re}\le-0.11$) and never overtake — the
saddle-node read directly off certified exact roots
($\sigma_{\min}(\Delta)\approx10^{-17}$), with no spurious cluster in sight. (One panel,
$\lambda{=}0.25$, misses its real mode $\approx-0.13$ due to a root-finder gap; the
monotone trend across the other five panels is unambiguous.)

### 5.2 Nature and τ-independence: the α–τ grid (corrected, artefact-free)

The definitive grid (`lambda_c_grid_v2.py`, $N=2000$, robust Woodbury branch + exact
reduced spectrum of §3.4b, $\tau=0$ included exactly as the ODE limit; fig.
`lambda_c_grid_v2_N2000.png`): over $\alpha\in\{0.01,0.03,0.05,0.07,0.10\}\times
\tau\in\{0,0.5,1,2,4,6,8,10,12,15\}$ — **all 50 points are saddle-node/SNIC; no Hopf
anywhere; λ_c is exactly τ-independent per α**:

| $\alpha$ | 0.01 | 0.03 | 0.05 | 0.07 | 0.10 (near-cap.) |
|---|---|---|---|---|---|
| $\lambda_c$ (all $\tau\in[0,15]$) | 0.3896 | 0.3253 | 0.2822 | 0.2017 | 0.0808 (indicative) |

This confirms (7) *empirically over the whole plane*: the static loss is the fold, and
$e^{-0\cdot\tau}=1$ makes it delay-blind. Unlike the v1 grid ($N{=}1000$, old GMRES
continuation, Hopf check truncated by the α=0.10 false fold), here the **full** memory
branch is examined at every point with certified physical roots.

**How close does the delay come to a Hopf?** The grid also records the *margin* — the
rightmost complex root's real part near the fold. It shrinks **monotonically in τ** (the
delay pushes oscillatory modes toward the axis) and faster at higher α: e.g.
$\max\mathrm{Re}_{\rm cplx}$ = −0.79 (τ=0) → −0.105 (τ=15) at α=0.01, and −0.91 (τ=4) →
**−0.055** (τ=15) at α=0.07. Within the studied window τ ≤ 15 the complex modes never
cross below the fold; §5.6 extends the scan to τ=100 and finds the **crossover**. (This
also explains the legacy table's τ-variation as a methodological artefact of its
discrete-time criterion, not real delay physics in this window.)
Grid figure: `results/1_bifurcation_statique/figures/lambda_c_grid_v2_N2000.png` (five flat λ_c(τ) lines,
all fold markers, no Hopf marker anywhere).

### 5.3 The λ_c(α) curve at N = 10⁴ (low α — the clean, reference result)

Robust Woodbury tracer, **5 seeds per point** (`multiseed_fold.py`; summary figure
`lambda_c_vs_alpha.png`):

| $\alpha$ | $\lambda_c$ (mean ± std) | legacy analytic $\lambda_c^{(1)}$ | gap |
|---:|:---:|:---:|:---:|
| 0.01 | **0.379 ± 0.010** | 0.403 | −6% |
| 0.03 | **0.314 ± 0.007** | — | on curve |
| 0.05 | **0.263 ± 0.002** | 0.277 | −5% |

$\lambda_c(\alpha)$ decreases monotonically (more stored patterns ⇒ earlier
destabilization), with **tiny scatter (≤2.6%)** and a **consistent ≈−5% finite-size
bias** below the analytic curve. Every point is a real crossing (saddle-node/SNIC),
τ-independent, no Hopf. **This low-α curve is the solid deliverable of the static
study.**

### 5.4 Near the Hopfield capacity (α ≈ 0.10): breakdown of single-realization numerics

At $\alpha=0.10$ (near $\alpha_c\approx0.138$) the memory-branch fold is **not a
well-defined single-sample quantity**:

- **Strong finite-size decrease** (robust tracer, dense-verified): $\lambda_{\rm fold}\approx$
  **0.18** ($N{=}1000$) → **0.081** ($N{=}2000$) → **0.053** ($N{=}10^4$).
- **Huge sample scatter** (8 seeds): $N{=}2000\!:\ 0.099\pm0.052$ (52%); $N{=}10^4\!:\
  0.062\pm0.040$ (64%). Folds span 0.03–0.20.
- **Spurious states:** some seeds' annealed initial state is already *unstable*
  ($\mathrm{eig\_max}(M)>0$ at $\lambda=0$, lower $m_1$) — the fingerprint of the
  rough, many-metastable-state landscape near capacity.

On the summary figure the α=0.10 point sits far below the analytic (0.062 vs 0.246, a
75% gap) with a large error bar — a **visible breakdown** of single-realization
continuation near a capacity transition. The $m_1\!\approx\!0.99$ states other solvers
find at higher $\lambda$ are *different* solutions, not the $\xi^1$-connected branch.
**Verdict:** near capacity the reliable tool is the analytic/DMFT ensemble average, not
sample-wise numerics.

### 5.5 Finite-size scaling (α=0.05)

Single-seed scaling over $N\in\{2000,4000,8000\}$ (`finite_size_scaling.py`) gives
$\lambda_{\rm fold}$ = 0.284 / 0.255 / 0.285 — **non-monotonic**, i.e. dominated by
single-seed fluctuations ($\pm0.015$); the *nature* (real crossing, no Hopf) is robust,
the *value* is not, motivating the multi-seed protocol of §5.3. (The clean thermodynamic
extrapolation is the natural next static step: multi-seed × several $N$ → $\lambda_c(\alpha,N\to\infty)$.)
Figure: `results/1_bifurcation_statique/figures/finite_size_scaling_a0.050_tau10_b20.0.png`.

### 5.6 Large τ: the delay-induced Hopf crossover at τ*(α) — and its limits

Extending the margin scan of §5.2 to $\tau\in[15,130]$ (exact reduced spectrum §3.4b,
$M{=}40$–$64$, $N{=}2000$, seed 42) yields the study's last static discovery
(figure `results/1_bifurcation_statique/figures/hopf_margin_vs_tau.png`: margin $|\mathrm{Re}\,z_{\rm cplx}|$
at $\lambda_c$ vs τ, linear + log panels):

- **α = 0.07: crossover at τ\* ≈ 52 ± 3.** The margin crosses zero: +0.055 (τ=15) →
  +0.001 (τ=50) → −0.001 (τ=55) → −0.005 (τ=100). For τ ≥ 55 an unstable complex pair
  exists on the memory branch *just below* the fold: the **delay-induced Hopf preempts
  the fold** — a codimension-2 fold–Hopf point at (τ\*≈52, λ≈0.2017).
- **α = 0.05: no crossing up to τ = 100** (margin +0.005 at τ=100, slow decay);
  τ\*(α) grows as α decreases (extrapolated ≳150 at α=0.05).
- **The emergent oscillation is SLOW**: refining α=0.07 at τ∈{55,70,100,130}
  (`results/1_bifurcation_statique/figures/hopf_crossover_a0.07.png`) gives $T_{\rm Hopf}=2\pi/\omega_c$ =
  210→238→299→364, i.e. $T_{\rm Hopf}/\tau$ = 3.81→2.80 — trending to the classic
  delayed-feedback half-wave law $T\to2\tau$ (not the pacemaker clock $T_1=\tau+t_{\rm esc}$;
  see `CYCLE_worklog.md` §B.3/E15 for the unified two-clock picture).

**Width caveat (E17, 2026-07-03).** A nonlinear probe of the α=0.07, τ=100 window
(perturbed exact states integrated with the exact P-dim DDE; `CYCLE_worklog.md` §6-E17,
figure `results/1_bifurcation_statique/figures/figH_hopf_birth_v2.png`) found **no Hopf growth**: direct
$T_P$ roots at the probed state λ=0.20167 give Re z = −0.0018 ± 0.0239i (weakly stable,
$T_{\rm osc}=263\approx2.6\tau$) although the swept placement put λ_Hopf=0.20162 below it.
The Hopf **window (λ_Hopf, λ_fold) is narrower than the placement resolution (≲5×10⁻⁵)**
and pinched against the fold; sample-to-sample fluctuations of λ_c (±0.01) dwarf it.
The probe instead revealed **bistability of twin memory states** 10⁻² apart near the
fold (second exact fixed point, residual 2.5×10⁻¹⁵, margin 0.038) — the fingerprint of
landscape roughness at α=0.07, echoing §5.4. **Robust statements:** the existence of the
margin crossover, τ\*(α)≈52 at α=0.07, and the $T\to2\tau$ family; **not robust:** the
in-window Hopf dynamics at any single sample.

---

### 5.7 The memory→front unification: one branch family, per-pattern thresholds (rev. 4, E21/E21d, 2026-07-05)

The cycle study had introduced a second static object, the **"pinned front"** — the
two-consecutive-pattern mixture (a_μ, a_{μ+1}) ≈ (0.68, 0.32) on which the recall cycle
dies at λ\*≈0.328 (`CYCLE_worklog.md` E9/E12/E18). Experiments **E21a/a2** (pseudo-arclength
continuation + state identification) settle its relation to the memory branch studied here:

1. **The pinned front IS the memory branch of its leading pattern.** The bond-91 front,
   continued DOWNWARD in λ by pseudo-arclength, shows **no turning point and no eigenvalue
   crossing**: it connects smoothly to the pure Hopfield memory ξ⁹¹ as λ→0 (a₉₁→1, a₉₂→0).
   At matched λ, Newton from pure ξ⁹¹ and the front branch give the **same fixed point**
   to ‖Δa‖≈5×10⁻⁴ with identical spectra (npz `results/3_unification_seuils_scaling/data/E21_dissolution.npz`;
   figure `results/3_unification_seuils_scaling/figures/figS_front_birth.png`).
2. **Every memory pattern ξ^μ therefore lives on one branch** that starts at the pure
   pattern at λ=0, **tilts continuously** toward its cyclic successor ξ^{μ+1} as λ grows
   (the λK drive feeds a_{μ+1}), reaches the quasi-universal mixture shape ~(0.70, 0.30)
   near its top, and dies there by a **saddle-node at a per-pattern quenched threshold
   λ_c(μ)**. The λ_c=0.2822 of §5.1 is the threshold **of pattern 1**; the depinning
   λ\*≈0.328 is the threshold **of pattern 91** = **max_μ λ_c(μ)** (extreme-value statistics
   over the P quenched thresholds, cf. E9/E13). What looked like two distinct transitions
   (memory fold at λ_c, cycle SNIC at λ\*) is **one mechanism** — the per-pattern
   saddle-node — read at two different quantiles of the same threshold distribution.
3. **Honest re-verification of the §5.1 fold (E21d)** — the unification prompted a
   re-audit of the memory-fold claim with the strongest available tools
   (npz `results/3_unification_seuils_scaling/data/E21d_memory_fold_check.npz`):
   - **Arclength through the fold**: the ξ¹ branch **turns** at λ≈0.2835 with a single
     *real* eigenvalue crossing 0 (−0.0016 at the top; +0.04…+0.21 on the returned,
     unstable flank). The saddle-node verdict of §5.1 is confirmed by an independent
     continuation method that rounds the fold instead of stalling at it.
   - **Independent dense check of the spectral pipeline**: the rightmost eigenvalue of
     the explicit dense N×N Jacobian (scipy `eigvals`, N=2000 — a fully independent code
     path) agrees with the P×P Sylvester value (`eigmax_M`) to **1.8×10⁻¹⁵** (λ=0.20) and
     **2.2×10⁻¹⁶** (λ=0.2404).
   - **Fold-region micro-structure**: stepwise Newton finds the main stable segment up to
     0.2820 (eig −0.071), a stall at 0.2825–0.2830 (eig→0: the fold, consistent with
     λ_c=0.2822), then a **re-convergent twin micro-segment at 0.2835** (residual
     1.3×10⁻¹¹) before final death; the unstable flank re-folds near 0.269. The fold
     position is thus intrinsically defined only to ±1.5×10⁻³ — **twin-memory roughness**,
     the α=0.05 echo of the E17 twin bistability at α=0.07 (§5.6) and of §5.4.
   - **Shape at the fold**: ξ¹ dies as the tilted mixture **(a₁,a₂)=(0.699, 0.300)**
     (ratio 0.43), matching the bond-91 front at *its* fold (0.664, 0.324, ratio 0.49) —
     the same universal death shape.
4. **Delay-robustness of the branch (E21b/c).** The z=0 structure (all fold positions)
   is **τ-independent to exactly 0.0** (max|T_P(0;τ)−T_P(0;τ′)| = 0 over τ∈{5..100},
   the Δ(0)=−M identity realized numerically). Along the branch the rightmost delayed
   root stays strictly negative for all τ∈{5,10,20,50} (margin eroding ~1/τ, leading mode
   real→complex), so **no Hopf ever preempts the per-pattern saddle-node** on these
   branches at α=0.05 (figure `results/3_unification_seuils_scaling/figures/figT_front_stability_vs_tau.png`).

**Consequence for the phase map.** "End of the static-recall phase" is not a single
number: pattern branches disappear one by one over [min_μ λ_c(μ), λ\*], while their
**basins** collapse earlier (E20: the memory basin fraction dies around λ≈0.30–0.31).
The full statistics of λ_c(μ) is the object of §5.8.

### 5.8 Finite-size statistics of the per-pattern thresholds λ_c(μ) (rev. 5, E24, 2026-07-05)

All P thresholds were traced for every (N, seed) by warm-started Newton continuation
with adaptive step and final bisection (±2×10⁻⁴, cross-checked by the spectral
extrapolation eig²(λ) linear; validation at N=2000/seed 42: pattern 1 → 0.2821 ≡ §5.1,
pattern 91 → 0.3276 ≡ λ\* **and argmax of all 100** — λ\*=max_μ λ_c(μ) verified
directly). Full protocol and tables: `CYCLE_worklog.md` §6-E24; codes `e24_*.py`.

**(a) The tilt is smooth — no knee.** Along every branch the tilt a_{μ+1}/a_μ grows
steadily from λ=0.05 on; half of the final tilt is acquired at **x₅₀ = 0.64·λ_c(μ)**
(IQR [0.61, 0.66] over the 100 branches — a sudden departure would give x₅₀≈1), and the
100 curves **collapse onto a universal shape** once λ is rescaled by λ_c(μ): the quenched
disorder sets *only* the threshold, not the trajectory. The only fast regime is the
universal fold singularity a_μ = a_c + k√(λ_c−λ) (R²=0.93–0.98). Hence the honest
definition of the *complete*-recall phase is **λ < min_μ λ_c(μ)** — at N=2000/seed 42
this is **0.2195** (pattern 28), well below the ξ¹ probe value 0.2822. Delayed-stability
spot-checks (30 T_P probes on 5 percentile branches, τ=10): **0 unstable points** —
existence implies DDE stability up to the fold across the whole distribution.
Figure: `3_unification_seuils_scaling/figures/figU_branch_geometry.png`.

**(b) The λ_c/λ\* gap is a finite-size effect.** At fixed α=0.05 (5 seeds; 3 at N=8000):

| N | P | median | σ | gap=max−min | λ\*=max (obs) | EVT-predicted max |
|---|---|---|---|---|---|---|
| 500 | 25 | 0.3080 | 0.0341 | 0.131±0.029 | 0.3589 | 0.3647 |
| 1000 | 50 | 0.2803 | 0.0319 | 0.159±0.028 | 0.3471 | 0.3476 |
| 2000 | 100 | 0.2742 | 0.0229 | 0.114±0.014 | 0.3254 | 0.3276 |
| 4000 | 200 | 0.2691 | 0.0175 | 0.095±0.005 | 0.3048 | 0.3128 |
| 8000 | 400 | 0.2674 | 0.0124 | 0.079±0.009 | 0.2956 | 0.3015 |

The gap decreases monotonically from N=1000 on (effective slope ≈ N^{−0.22}) and → 0 as
N→∞: in the thermodynamic limit all thresholds merge into a single λ_c^∞ and "memory
death" and "cycle birth" become **one transition**; at finite N the wide band (0.11 at
N=2000) is real and non-self-averaging. At fixed P=100 the per-threshold dispersion
scales as σ ~ N^{−0.61} (≈CLT −0.5, slightly faster since α=100/N also decreases); at
fixed α, σ ~ N^{−0.38} with the growing extreme-value factor √(2 ln αN) slowing the gap.

**(c) The gap is quantitatively the Gaussian extreme-value gap.** The λ_c(μ)
distribution is near-Gaussian with a stable moderate left skew (−0.5; standardized CDFs
collapse, figV-b). The iid-Gaussian prediction **max ≈ mean + σ(N)·a_P**, with
a_P = √(2lnP) − (lnlnP + ln4π)/(2√(2lnP)), reproduces the observed λ\* to 0.002–0.008 at
every N (slight over-prediction, consistent with the negative skew). The theoretical
recall-death→cycle-birth gap is thus **λ\*−median ≈ σ(N)·√(2 ln αN)**.
Figure: `3_unification_seuils_scaling/figures/figV_threshold_distribution.png`.

### 5.9 Large-scale scaling campaign: the threshold law is a CLT of the quenched cross-talk (rev. 6, E30, 2026-07-08/09)

An overnight campaign (303 runs, `results/3_unification_seuils_scaling/data/E30_scaling/`, driver
`e30_scaling_overnight.py`, seeds 1000+, bisection λ_c as the primary estimator)
re-measured the scaling of §5.8b with far more statistics and larger sizes:
**fixed P=100** with N = 2000…32000 (8–24 seeds per N, α=100/N ≤ 0.05), and
**fixed α=0.05** with N = 1000…14000 (full P = αN traced at every size — 500/500
branches at N=10⁴, 700/700 at N=14000; runs with N ≥ 10⁴ in `--no-eig` mode,
bisection only). Analysis `e30_analysis.py`, figure
`3_unification_seuils_scaling/figures/figAE_threshold_scaling.png`.

| series | σ(λ_c) scaling | gap = max−min scaling |
|---|---|---|
| fixed α = 0.05 (physical) | **N^{−0.508}** | **N^{−0.42}** |
| fixed P = 100 | N^{−0.597} | N^{−0.619} |

**(a) The CLT is confirmed.** At fixed α the dispersion follows σ ~ N^{−1/2} to
within fit error — the earlier −0.38 of §5.8b was small-N contamination (the
N ≤ 8000, few-seed fit), settled by the reach points: σ = 0.0308 → 0.0079 from
N=1000 → 14000. Mechanism: λ_c(μ) is a smooth functional of the quenched
cross-talk field felt by pattern μ, an aggregate of the P−1 cross-overlaps
o_ν = (1/N)ξ^μ·ξ^ν, each Gaussian of variance 1/N — a sum of ~αN weakly
dependent contributions, hence a central limit theorem: near-Gaussian shape
(pooled standardized λ_c over N ≥ 4000, n = 8764: skew −0.45, excess kurtosis
+1.02, KS = 0.026 vs normal) and σ ~ N^{−1/2}. The fixed-P exponent −0.60 is
steeper because α = 100/N also decreases along that series (cleaner landscape).

**(b) gap→0 verified to N=14000.** The recall-death/cycle-birth band shrinks
monotonically, 0.149 (N=1000) → 0.078 (8000) → 0.055 (10⁴) → **0.0498 (N=14000)**,
effective slope N^{−0.42} = the CLT σ-decay slowed by the growing extreme-value
factor √(2 ln αN), exactly the §5.8c prediction. The fixed-α median is
N-converged at **λ_c^∞(0.05) ≈ 0.268** from N=4000 on.

**(c) Practical ceiling.** Full-P tracing at α=0.05 costs ~N^{3.4}
(Woodbury O(P²N) per solve × P branches × O(N^{0.4}) steps): N=10⁴ ≈ 90 min/seed,
N=14000 ≈ 4.8 h/seed (8 cores) — the realistic overnight ceiling is N ≈ 14000;
N=20000 (~6 h/seed) was dropped.

---

## 6. Conclusion (static verdict)

For the delayed mixed Hopfield network at $\tau=10,\beta=20$:

1. **Mechanism:** the standing memory loses stability through a **fold (saddle-node /
   SNIC)** — a single *real* characteristic root reaches $z=0$, guaranteed by the exact
   identity $\Delta(0)=-M$ and **τ-independent**. **Hopf is excluded** (no complex pair
   crosses below the fold, verified via the IG spectrum).
2. **Location (low α, clean):** $\lambda_c(\alpha)=$ 0.379 / 0.314 / 0.263 ± (0.010 /
   0.007 / 0.002) for $\alpha=$ 0.01 / 0.03 / 0.05 at $N=10^4$ — a monotone curve ≈5%
   below the analytic mean field (finite-size bias).
3. **Near capacity (α≳0.10):** the single-realization fold is intrinsically
   ill-defined (strong finite-size drift + ~60% sample scatter + spurious states);
   numerics break down and the analytic/DMFT treatment is required.
4. **Validity boundary in τ (rev. 3):** the fold verdict holds for τ ≲ τ\*(α); at
   α=0.07 a delay-induced Hopf preempts the fold beyond τ\*≈52 (§5.6) — but its λ-window
   is ultra-narrow (≲10⁻⁴) and below single-sample resolution. For α ≤ 0.05 no crossover
   was found up to τ=100.
5. **SN vs SNIC** remains open by design (a global, not linear, distinction) — settled
   by the dynamic analysis: see `CYCLE_worklog.md` (verdict: the cycle dies by a SNIC
   *not* on the memory but at the weakest bond, λ\*≈0.327 > λ_c; the static fold at λ_c
   is a plain saddle-node with bistability memory/pinned-front below it).
6. **Unification (rev. 4, §5.7):** memory patterns and "pinned fronts" are **one branch
   family** — each ξ^μ tilts continuously toward ξ^{μ+1} with λ and dies at its own
   quenched saddle-node λ_c(μ); λ_c = threshold of ξ¹, λ\* = max_μ λ_c(μ). The §5.1 fold
   was re-verified by arclength (true turning point, real eigenvalue crossing) and the
   spectral pipeline validated against a dense N×N diagonalization to 2×10⁻¹⁶; the fold
   region carries ±1.5×10⁻³ twin-memory micro-structure.

---

## 7. Caveats and limitations

1. **Saddle-node vs SNIC** is not decidable from the fixed-point spectrum (§2.2c);
   needs the cycle (§8).
2. **Near-capacity numerics are unreliable** (§5.4): single-seed meaningless, strong
   finite-size drift; use analytic/DMFT.
3. **Spurious pseudospectral modes** are present by construction and must always be
   filtered (identified by λ-independence and $M$-non-convergence).
4. **float32 is unsafe for the continuation** (§3.7): all static branch results use
   float64.
5. **Finite-size bias:** the reported $\lambda_c$ sit ~5% below the thermodynamic value;
   a multi-seed × multi-$N$ extrapolation would remove it.

---

## 8. Toward the dynamic branch (limit cycle)

For $\lambda>\lambda_c$ the memory is gone and the expected attractor is a
**sequence-recall limit cycle**. Its stability — and the definitive **SN-vs-SNIC**
answer — follow from **Floquet theory for periodic DDEs**: construct the periodic orbit
$u_p(t)$, linearize (periodic-coefficient variational DDE), and compute the **monodromy
operator's Floquet multipliers** by a pseudospectral discretization (the periodic
generalization of §3.4), exploiting the cyclic $\mathbb Z_P$ symmetry (Bloch reduction).
The SNIC signature is a cycle whose **period diverges** as $\lambda\downarrow\lambda_c$
with a multiplier $\to1$. Full derivation in **`FLOQUET_methodology.md`**.

---

## 9. Figure inventory (exact paths, all under `numerics/results/`)

| Path | Content | Section |
|---|---|---|
| `1_bifurcation_statique/figures/fig2_roots_panel_N2000_a0.050_tau10_b20.0_s42.png` | IG characteristic roots in the complex plane at 6 λ up to the fold (N=2000; last panel: the real root reaching 0 at λ=0.284) | §5.1 |
| `1_bifurcation_statique/figures/fig3_maxre_N2000_a0.050_tau10_b20.0_s42.png` | stability margin max Re z(λ): spurious plateau −0.45 then the physical real mode climbing to 0 at λ_c≈0.282 | §5.1 |
| `1_bifurcation_statique/figures/fig2_physical_roots_near_lc.png` | **the exact artefact-free spectrum**: 10 rightmost physical roots (σ_min(Δ)≈10⁻¹⁷) at 6 λ/λ_c ∈ [0.75, 1.00], N=10⁴ — real fold-mode climbs to 0, complex modes stay at Re ≤ −0.11 | §3.4b, §5.1 |
| `1_bifurcation_statique/figures/lambda_c_grid_v2_N2000.png` | λ_c(α,τ) grid, 5 α × 10 τ ∈ [0,15]: all folds, flat in τ, no Hopf | §5.2 |
| `1_bifurcation_statique/figures/fold_N10000_a0.05.png` | memory branch at N=10⁴, 5 seeds: eig_max(M)→0 and m₁(λ) (the branch "curl"), λ_c=0.263±0.002 | §5.3 |
| `1_bifurcation_statique/figures/fold_N10000_a0.01.png`, `..._a0.03.png`, `..._a0.1.png` | same per-α (α=0.10: the scattered near-capacity case) | §5.3–5.4 |
| `1_bifurcation_statique/figures/lambda_c_vs_alpha.png` | **the deliverable**: λ_c(α) at N=10⁴ vs legacy analytic; α=0.10 breakdown point with huge error bar; capacity line α_c≈0.138 | §5.3–5.4 |
| `1_bifurcation_statique/figures/finite_size_scaling_a0.050_tau10_b20.0.png` | single-seed λ_fold(N) non-monotonicity (sample-fluctuation dominated) | §5.5 |
| `1_bifurcation_statique/figures/hopf_margin_vs_tau.png` | Hopf margin at λ_c vs τ ∈ [0,100], 5 α, lin+log: the erosion and the α=0.07 crossing | §5.6 |
| `1_bifurcation_statique/figures/hopf_crossover_a0.07.png` | left: rightmost complex pair vs fold mode near the fold (τ=55–130); right: T_Hopf=2π/ω_c vs τ between the 2τ and 3τ lines | §5.6 |
| `1_bifurcation_statique/figures/figH_hopf_birth_v2.png` | nonlinear probe of the τ=100 window: control decay staircase + basin hop to the twin state | §5.6 (E17) |
| `3_unification_seuils_scaling/figures/figS_front_birth.png` | memory→front unification: the front branch continues smoothly to pure ξ⁹¹ at λ→0 (no fold) + front ≡ tilted-memory overlay | §5.7 (E21) |
| `3_unification_seuils_scaling/figures/figT_front_stability_vs_tau.png` | rightmost delayed root of the branch vs λ for τ∈{5,10,20,50}: strictly negative (delay-robust, no Hopf), margin ~1/τ | §5.7 (E21c) |
| `3_unification_seuils_scaling/figures/figJ_front_branch.png` | the bond-91 branch over [0.13, 0.328] with instantaneous + delayed stability (context for §5.7) | §5.7 (E18) |
| `3_unification_seuils_scaling/figures/figU_branch_geometry.png` | all 100 tilt curves + universal collapse in λ/λ_c(μ) + knee detector (x₅₀=0.64) + terminal √ law | §5.8 (E24a) |
| `3_unification_seuils_scaling/figures/figV_threshold_distribution.png` | λ_c(μ) distribution vs N, standardized CDFs vs Gaussian, σ(N) both scalings, observed gap vs EVT prediction | §5.8 (E24b-d) |
| `3_unification_seuils_scaling/figures/figAE_threshold_scaling.png` | large-scale scaling (303 runs): σ~N^{−0.51} (CLT) & gap~N^{−0.42} at fixed α up to N=14000, standardized histogram vs normal | §5.9 (E30) |

Caveat: `9_obsolete_ne_pas_citer/N10000_float32_invalide/A1_branch_overlay.png`, `A2_stability_overlay.png`, `A3_roots_panel.png`
are **from the invalidated float32 run** (§3.7) — kept for the record, do not use.
Early N=1000 exploratory figures live in `results/9_obsolete_ne_pas_citer/Stabilité/`, `results/9_obsolete_ne_pas_citer/critical lambda/`,
`results/9_obsolete_ne_pas_citer/MCMC phase diagrams/` (legacy discrete-time/MCMC study, 2026-06-03 notes).

---

## Appendix — reproduction and symbols

```bash
# static branch + IG spectrum + figures (N=2000 reference)
conda run -n mcmc_env python run_validation.py --tau 10 --beta 20
# alpha-tau grid (tau-independence, no-Hopf)
conda run -n mcmc_env python lambda_c_grid.py --alphas 0.05 0.10 --taus 1 2 5 10 15 20 --Ngrid 1000
# multi-seed lambda_c(alpha) at N=10000 (robust Woodbury tracer)
conda run -n mcmc_env python multiseed_fold.py --N 10000 --alpha 0.05 --seeds 42 43 44 45 46 --outdir results/N=10000
conda run -n mcmc_env python summary_lambdac_alpha.py           # -> lambda_c_vs_alpha.png
```

Modules: `couplings.py` (matrix-free ops), `fixed_point.py` (pseudo-arclength — below
capacity), **`robust_branch.py`** (Woodbury solve + Sylvester eig_max + robust tracer +
β-annealing), `dde_stability.py` (IG operator + Arnoldi), `finite_size_scaling.py`,
`lambda_c_grid.py`, `multiseed_fold.py`, `summary_lambdac_alpha.py`.

| symbol | meaning |
|---|---|
| $u^\*(\lambda)$ | memory fixed point | 
| $C(\lambda)=(1-\lambda)J+\lambda K$ | mixed coupling |
| $D=\mathrm{diag}(\beta(1-\tanh^2\beta u^\*))$ | gain |
| $M=-I+C(\lambda)D$ | Jacobian $\partial F/\partial u$ |
| $\Delta(z)$ | characteristic matrix (4) |
| $\mathrm{eig\_max}(M)$ | rightmost eigenvalue of $M$ (fold at 0) |
| $U,V$ | low-rank factors $C D=\tfrac1N UV$ |
| $\lambda_c(\alpha)$ | static fold / critical mixing |
| $\alpha_c\approx0.138$ | Hopfield capacity |
