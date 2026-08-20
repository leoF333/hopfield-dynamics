> **⚠ SUPERSEDED IN PART (2026-07-01).** A critical review found this plan misses
> the **exact P-dimensional reduction** (the pattern span is exactly invariant and
> attracting; all nontrivial Floquet multipliers come from an exact P-dim
> variational problem — not mean-field), overstates the ℤ_P/Bloch reduction
> (approximate at finite α), and omits the fold-of-cycles / boundary-crisis
> scenarios for the slow↔pacemaker transition. The authoritative working plan is
> **`CYCLE_worklog.md`** (resumable log). This file remains valid for the general
> Floquet-DDE theory (§4, §7) and the rotating-wave ansatz (§2, mean-field).

# Limit-cycle (sequence-recall) stability at high λ — full methodology

### Floquet analysis of the periodic orbit of the delayed mixed Hopfield network

**Author:** numerical agent (Claude Code) · **Date:** 2026-07-01
Companion to `REPORT_static_bifurcation.md` (which treats the λ<λ_c fixed point).

> **Goal.** For λ above the static fold λ_c, the standing memory is gone and the
> network is expected to *cycle* through the stored sequence ξ¹→ξ²→⋯→ξ¹ — a
> **limit cycle** (periodic orbit). This document derives, from first principles,
> (i) what that periodic orbit is and how to construct it, (ii) the linear
> variational problem around it, (iii) **Floquet theory for delay equations**
> (monodromy operator, multipliers, stability criterion), (iv) the pseudospectral
> numerical method to compute the multipliers, (v) the reduction afforded by the
> cyclic symmetry, and (vi) how the multipliers classify the bifurcation and,
> together with the fixed-point analysis, finally decide **saddle-node vs SNIC vs
> Hopf**.

---

## 1. Model, regime, and the object of study

The state $u(t)\in\mathbb R^N$ obeys the retarded functional differential equation
$$
t_0\,\dot u(t) = -u(t) + (1-\lambda)\,J\,g\!\big(u(t)\big) + \lambda\,K\,g\!\big(u(t-\tau)\big),
\qquad g(x)=\tanh(\beta x)\ \text{(elementwise)},
\tag{1}
$$
with the symmetric Hebbian and asymmetric cyclic couplings
$$
J=\frac1N\sum_{\mu=1}^P \xi^\mu\xi^{\mu\mathsf T},\qquad
K=\frac1N\sum_{\mu=1}^P \xi^{\mu+1}\xi^{\mu\mathsf T}\quad(\text{indices mod }P).
\tag{2}
$$
$K$ maps a state aligned with $\xi^\mu$ toward $\xi^{\mu+1}$; with the delay $\tau$
it is a **time-lagged push along the stored cycle**. For $\lambda$ large enough the
attractor is a periodic **sequence-recall** orbit $u_p(t)$, $u_p(t+T)=u_p(t)$, of
some period $T$. **We want the linear stability of $u_p$.**

The natural state space of (1) is the space of history segments
$$
X = C\big([-\tau,0],\mathbb R^N\big),\qquad u_t(\theta):=u(t+\theta),\ \theta\in[-\tau,0],
$$
on which (1) generates a (nonlinear) semiflow $\Phi^t:X\to X$. A limit cycle is a
closed orbit of $\Phi^t$; its stability is governed by the **monodromy operator**
$\Phi^T$ linearized about $u_p$ — this is Floquet theory (§4).

---

## 2. The periodic orbit: existence and construction

### 2.1 Condensed (overlap) reduction and the cyclic symmetry

Define the **activity overlaps** $m_\mu(t)=\tfrac1N\,\xi^\mu\!\cdot g(u(t))$. Using (2),
the field in (1) is a superposition of patterns,
$$
(1-\lambda)Jg(u(t))+\lambda Kg(u(t-\tau)) = \sum_{\nu} \xi^\nu\,b_\nu(t),\qquad
\boxed{\,b_\nu(t) = (1-\lambda)\,m_\nu(t) + \lambda\,m_{\nu-1}(t-\tau)\,}.
\tag{3}
$$
The $b_\nu$ are the **effective fields along each pattern**; the delayed term feeds
pattern $\nu{-}1$'s past overlap into pattern $\nu$'s present field — the engine of
sequential advance. In the condensed regime ($\alpha=P/N\to0$) the state stays in the
pattern span, $u(t)\approx\sum_\nu \xi^\nu a_\nu(t)$, and (1) closes on the
$P$ amplitudes:
$$
t_0\,\dot a_\nu(t) = -a_\nu(t) + (1-\lambda)\,\mathcal M_\nu[a(t)] + \lambda\,\mathcal M_{\nu-1}[a(t-\tau)],
\tag{4}
$$
$$
\mathcal M_\nu[a] = \big\langle\, \xi^\nu\,\tanh\!\big(\beta\,\textstyle\sum_\rho \xi^\rho a_\rho\big)\big\rangle_\xi
\qquad(\xi^\rho\ \text{i.i.d. }\pm1),
$$
the mean-field overlap function. **(4) is exactly invariant under the cyclic shift
$\nu\mapsto\nu+1$** — relabelling patterns leaves $J,K$ and hence (4) unchanged. (This
symmetry is exact for the reduced/mean-field dynamics; microscopically it is only
statistical, because $g$ acts coordinate-wise and the $\xi^\mu$ are random — see the
caveat in §6.3.)

### 2.2 Discrete rotating-wave ansatz

Cyclic symmetry + periodicity ⇒ the sequence-recall cycle is a **discrete rotating
wave**: advancing the pattern label by one is the same as advancing time by one
"step period" $T_1=T/P$,
$$
\boxed{\,a_\nu(t) = A\big(t-\nu T_1\big),\qquad T = P\,T_1\,}
\tag{5}
$$
for a single waveform $A(\cdot)$. Indeed $a_{\nu+1}(t)=A(t-(\nu{+}1)T_1)=a_\nu(t-T_1)$,
and $a_\nu(t+T)=a_\nu(t)$. Substituting (5) into (4) and passing to the co-moving
coordinate $s=t-\nu T_1$ gives a **single self-consistent delay boundary-value
problem** for the pair $(A,\,T_1)$:
$$
t_0\,A'(s) = -A(s) + (1-\lambda)\,\widehat{\mathcal M}\big[A;s\big]
                    + \lambda\,\widehat{\mathcal M}\big[A;s-\tau+T_1\big],
\tag{6}
$$
where $\widehat{\mathcal M}[A;s]=\big\langle\xi^0\tanh(\beta\sum_k \xi^{-k}A(s+kT_1))\big\rangle_\xi$
collects the (localized) contributions of neighbouring patterns $k=0,\pm1,\dots$.
Equation (6) is the co-moving-frame profile equation: solve for the waveform $A$ on
one step and the step period $T_1$ (equivalently the wave speed $1/T_1$).

### 2.3 Practical construction of $u_p$ (three options)

1. **Direct simulation + phase lock.** Integrate (1) at high λ from a memory
   initial condition until it settles on the recall cycle; detect the period $T$
   from a Poincaré section (e.g. $m_1(t)$ maxima); extract one clean period as
   $u_p$. Cheapest; float32/GPU-friendly. Used to *seed* the accurate methods below.
2. **Periodic collocation BVP (full N).** Solve, over one period with the delay,
   $$
   t_0\dot u(t)=F(u_t),\quad u(0)=u(T),\quad \langle u(0)-u^{\rm seed}(0),\,\dot u^{\rm seed}(0)\rangle=0,
   $$
   the last being the **phase condition** that removes time-translation degeneracy.
   Chebyshev/piecewise-polynomial collocation in $t$ on $[-\tau,T]$; Newton on the
   coefficients and $T$. This is the DDE-BIFTOOL / `knut` style; gives $u_p$ and $T$
   to spectral accuracy.
3. **Reduced BVP (6)** in the condensed limit — cheapest analytic route, yields the
   waveform, $T_1$, and the scaling $T=P\,T_1$. Good for insight and initial guesses;
   the pacemaker heuristic $T_1\sim\tau+O(t_0)$ (hence $T\sim P(\tau+\!\cdot)$) follows
   from balancing the relaxation time $t_0$ against the delayed advance $\tau$.

---

## 3. Linearization about the cycle (the variational equation)

Write $u(t)=u_p(t)+\delta u(t)$ in (1) and keep first order. With the **$T$-periodic
gain**
$$
D(t)=\operatorname{diag}\,g'\!\big(u_p(t)\big)=\beta\,\operatorname{diag}\!\big(1-g(u_p(t))^2\big),
\qquad D(t+T)=D(t),
\tag{7}
$$
the perturbation obeys the **linear, $T$-periodic delay equation**
$$
\boxed{\;t_0\,\dot{\delta u}(t) = -\delta u(t) + (1-\lambda)\,J\,D(t)\,\delta u(t)
                                   + \lambda\,K\,D(t-\tau)\,\delta u(t-\tau)\;}
\tag{8}
$$
i.e. $\dot{\delta u}=A(t)\,\delta u(t)+B(t)\,\delta u(t-\tau)$ with
$A(t)=\tfrac1{t_0}(-I+(1-\lambda)JD(t))$, $B(t)=\tfrac1{t_0}\lambda KD(t-\tau)$, both
$T$-periodic. This is the exact analogue of the fixed-point linearization of the
static report, except the coefficients now **oscillate in time with the cycle** —
which is precisely what Floquet theory handles.

---

## 4. Floquet theory for periodic delay equations

### 4.1 Monodromy operator and multipliers

The solution of (8) defines the **monodromy (period) operator**
$$
U:\;X\to X,\qquad U\,\delta u_0 = \delta u_T ,
$$
mapping an initial history segment $\delta u_0=\delta u|_{[-\tau,0]}$ to the segment one
period later, $\delta u_T=\delta u|_{[T-\tau,T]}$ (re-based to $[-\tau,0]$). For $\tau>0$,
$U$ is **compact** on $X$; hence its spectrum is a countable set of eigenvalues
$$
\{\mu_k\}\subset\mathbb C,\qquad \mu_k\to0,
$$
the **Floquet multipliers**. A Floquet eigensolution satisfies $\delta u(t+T)=\mu\,\delta u(t)$,
and by Floquet–Lyapunov it has the form $\delta u(t)=e^{zt}\,p(t)$ with $p$ $T$-periodic
and $\mu=e^{zT}$; $z$ is a **Floquet exponent** (defined mod $2\pi i/T$).

### 4.2 Stability criterion and the trivial multiplier

$$
\boxed{\;u_p\ \text{is (orbitally, asymptotically) stable}\iff |\mu_k|<1\ \ \forall k,\ \text{except the trivial}\ \mu_0=1.\;}
$$
The value $\mu_0=1$ is **always** present for an autonomous system's limit cycle: its
eigenfunction is $\dot u_p$ (time-translation / Goldstone phase mode). Its presence
and accurate reproduction ($\mu_0\approx1$) is the primary numerical check. Stability
is decided by the **largest nontrivial** $|\mu|$; equivalently the largest nontrivial
$\mathrm{Re}(z)=\tfrac1T\ln|\mu|$.

### 4.3 Why an operator (and not a matrix)

For an ODE limit cycle the monodromy is a finite $N\times N$ matrix (integrate the
variational equation over one period). The **delay makes the "initial condition" a
function** (a whole history segment), so $U$ acts on the infinite-dimensional $X$ and
has infinitely many multipliers accumulating at $0$ — exactly as the fixed-point case
had infinitely many characteristic roots. Only finitely many multipliers lie outside
any circle $|\mu|>r>0$, so stability is decidable from the **leading** ones.

---

## 5. Numerical method: pseudospectral discretization of the monodromy operator

We compute the leading multipliers by discretizing $U$ — the periodic generalization
of the infinitesimal-generator method used for the fixed point.

### 5.1 Collocation construction of $U$

1. **Mesh the period.** Represent $\delta u$ on $[-\tau,T]$ by piecewise polynomials on
   a mesh, or globally by a Chebyshev grid of $M{+}1$ nodes per delay-interval; let the
   discrete unknown be the nodal values.
2. **Impose the variational DDE (8)** at the interior collocation points, using the
   differentiation matrix for $\dot{\delta u}$ and interpolation of the delayed term
   $\delta u(t-\tau)$ from the mesh (the $T$-periodic $D(t)$, $D(t-\tau)$ are evaluated
   from the stored cycle $u_p$).
3. **Read off the period map.** The linear relations express the "final" segment
   $\delta u|_{[T-\tau,T]}$ as a matrix acting on the "initial" segment
   $\delta u|_{[-\tau,0]}$. That matrix $\mathsf U_M\in\mathbb C^{N(M+1)\times N(M+1)}$
   (or $N\times$mesh) **is the discretized monodromy**.
4. **Eigenvalues.** The dominant eigenvalues of $\mathsf U_M$ (largest $|\mu|$, via
   Arnoldi/`eigs`, matrix-free using the same low-rank $J,K$ matvecs) approximate the
   Floquet multipliers. Convergence is **spectral** in $M$ for the smooth (dominant)
   modes (Breda–Maset–Vermiglio, *pseudospectral differencing for periodic RFDEs*).

Two equivalent implementations:
- **Solution-operator (time-stepping):** build $\mathsf U_M$ column-by-column by
  integrating (8) over $[0,T]$ for each basis history — conceptually the monodromy
  "matrix" assembled from trajectories.
- **Full-period collocation (BVP eigenproblem):** assemble the periodicity condition
  $\delta u(T)=\mu\,\delta u(0)$ as a generalized eigenproblem
  $\mathsf A\,v=\mu\,\mathsf B\,v$ over the whole period; more accurate, DDE-BIFTOOL style.

### 5.2 Cost and matrix-free structure

$\mathsf U_M$ has dimension $N(M{+}1)$; its action needs only $J,K$ **applied to
vectors** (the low-rank $\tfrac1N\xi^{\mathsf T}(\xi\,\cdot)$ trick), so no $N\times N$
matrix is formed. The extra cost over the fixed-point case is one full-period
integration of the variational equation (the coefficients are now time-dependent).
Because it is *dynamics* (not a near-singular Newton solve), this stage is
**float32/GPU-tolerant**, unlike the fixed-point continuation.

---

## 6. Exploiting the cyclic symmetry (Bloch reduction)

### 6.1 Discrete rotating wave ⇒ block-diagonal monodromy

By §2.2 the cycle satisfies $u_p(t+T_1)=Q\,u_p(t)$, where $Q$ is the pattern-shift
($Q\xi^\mu=\xi^{\mu+1}$) and $T_1=T/P$. Hence the coefficients of (8) obey
$A(t+T_1)=Q\,A(t)\,Q^{-1}$, $B(t+T_1)=Q\,B(t)\,Q^{-1}$, and the **one-step propagator**
$\Phi_1$ over $[0,T_1]$ intertwines with $Q$. The full monodromy factorizes,
$$
U = \big(R_Q\,\Phi_1\big)^{P},
$$
with $R_Q$ the shift action on $X$. By the representation theory of the cyclic group
$\mathbb Z_P$, the eigenproblem **block-diagonalizes** over its irreps: multipliers
organize into $P$ **Bloch families** indexed by $j=0,\dots,P{-}1$,
$$
\boxed{\;\mu = \rho\,e^{\,i2\pi j/P},\qquad \rho\in\operatorname{spec}\big(e^{-i2\pi j/P}R_Q\Phi_1\big),\;}
\tag{9}
$$
so one only diagonalizes the **one-step** operator $\Phi_1$ within each Bloch sector —
a factor-$P$ reduction, and a clean bookkeeping of the spectrum.

### 6.2 Reading the families

- The $j{=}0$ sector contains the **trivial multiplier** $\mu_0=1$ (phase mode) and the
  amplitude/stability modes of the wave.
- $j\neq0$ sectors are the **"phason"** modes — long-wavelength distortions of the
  pattern-ordering; a family crossing $|\mu|=1$ signals an instability of the *ordering*
  of the recall (e.g. skipping/reversing) rather than of a single step.

### 6.3 Caveat (exactness of the symmetry)

The reduction (9) is **exact in the condensed/mean-field description** (§2.1), where
$\mathbb Z_P$ symmetry is exact. Microscopically (finite $N$, fixed realization) the
coordinate-wise $g$ and random patterns break it at $O(1/\sqrt N)$; then (9) is an
approximate organizing principle and the *rigorous* multipliers come from the full
monodromy $\mathsf U_M$ of §5. Use the Bloch structure to label and to initialize; use
the full $\mathsf U_M$ to certify.

---

## 7. Bifurcation classification — and the SN-vs-SNIC verdict

How the **leading nontrivial multiplier** $\mu^\star(\lambda)$ crosses the unit circle
names the bifurcation of the cycle:
$$
\begin{array}{lll}
\mu^\star\to +1 & \text{fold / cyclic-fold, or SNIC connection} & \text{(period }\to\infty\text{ at SNIC)}\\
\mu^\star\to -1 & \text{period-doubling} & \\
\mu^\star=e^{\pm i\phi}\ (\phi\neq0,\pi) & \text{Neimark–Sacker (torus)} &
\end{array}
$$
**The decisive experiment for our transition at $\lambda_c$.** The fixed-point analysis
proved a *real eigenvalue through zero* at $\lambda_c$ (a fold), ruling out Hopf, but
could not separate **plain saddle-node** from **SNIC**. Floquet + cycle construction
settles it:

- **SNIC.** Just above $\lambda_c$ a stable limit cycle exists whose **period diverges**,
  $T(\lambda)\sim(\lambda-\lambda_c)^{-1/2}$, and which **limits onto the two colliding
  fixed points** as $\lambda\downarrow\lambda_c$; correspondingly a Floquet multiplier
  $\to1$ and the cycle's average dwells near the fold "ghost". The frequency $\to0$.
- **Plain saddle-node.** No such limit cycle is created at $\lambda_c$; trajectories
  fall to another attractor. No diverging-period cycle appears.
- **(For contrast) Hopf**, already excluded here, would give a cycle of **finite**
  frequency and **vanishing** amplitude born at the crossing.

Thus the program is: (i) fixed-point fold at $\lambda_c$ (done, real crossing → SN/SNIC,
no Hopf); (ii) construct the cycle for $\lambda>\lambda_c$ (§2.3); (iii) measure
$T(\lambda)$ and its onset scaling, and compute $\mu^\star(\lambda)$ by §5–6. A
diverging-period, marginally-emerging ($\mu\to1$) stable cycle ⇒ **SNIC**; its absence
⇒ **plain saddle-node**. This is the first computation that can *prove* the type.

---

## 8. Validation checklist and practical recipe

**Recipe.**
1. Simulate (1) at $\lambda>\lambda_c$ → seed cycle $(u_p,T)$; refine by the periodic
   BVP with phase condition (§2.3).
2. Store $D(t),D(t-\tau)$ on the period mesh from $u_p$.
3. Assemble the discretized monodromy $\mathsf U_M$ (§5), matrix-free.
4. Leading eigenvalues → multipliers $\mu_k$; exponents $z_k=\tfrac1T\ln\mu_k$.
5. Optionally Bloch-reduce (§6) to label/accelerate.
6. Verdict: stable iff all nontrivial $|\mu_k|<1$; classify onset by §7.

**Validation.**
- **Trivial multiplier**: $\mu_0=1$ to discretization accuracy (certifies $u_p$ and $\mathsf U_M$). Its eigenvector must match $\dot u_p$.
- **Spectral convergence** of the leading $\mu_k$ under $M$-refinement (as in the fixed-point M-convergence check).
- **Lyapunov cross-check**: leading $\mathrm{Re}(z)=\tfrac1T\ln|\mu^\star|$ vs the largest Lyapunov exponent from two nearby direct simulations.
- **Limits**: $\tau\to0$ reduces the monodromy to the ODE limit-cycle case; the static
  fold $\lambda_c$ must appear as the endpoint where a cycle multiplier $\to1$ and
  $T\to\infty$ (SNIC) or where the cycle ceases to exist.

**Symbols.** $u_p$ periodic orbit, $T=PT_1$ period, $D(t)$ periodic gain,
$U/\mathsf U_M$ monodromy operator/matrix, $\mu_k$ Floquet multipliers,
$z_k=\tfrac1T\ln\mu_k$ exponents, $Q$ pattern-shift, $j$ Bloch index.
