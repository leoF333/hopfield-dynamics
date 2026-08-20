# Correlated patterns in Hopfield / sequence / continuous-attractor networks — literature review

**Compiled 2026-07-08 (orchestrator, web-verified citations).** Companion to the correlated-pattern
experiments E27 (Markov video) / E28 (GP-circle video) in `CYCLE_worklog.md`. Every reference below was
checked against a live source; where a detail (page/volume) was not confirmed it is flagged.

## Our findings this review must contextualize
Delayed mixed Hopfield: `t0 u' = -u + (1-λ)J tanh(βu(t)) + λK tanh(βu(t-τ))`, J=Hebb (memories),
K=cyclic shift (sequence xi^μ→xi^{μ+1}), delay τ on K. For **iid** patterns: each memory tilts toward its
successor and dies at a quenched saddle-node λ_c(μ); the sequential-recall limit cycle is born by a SNIC at
the weakest bond (depinning). For **correlated "video" patterns** we found a dichotomy: **(1) local Markov
correlation** collapses per-pattern thresholds and, at strong correlation, destroys sequential recall
(memories melt into ~5-frame packets); **(2) smooth global (GP-on-circle) correlation** melts the individual
memories just the same, yet makes the recall cycle *more* robust (no depinning down to λ=0.20) — the network
becomes a **quasi-continuous ring attractor** (frames sample a smooth closed manifold with ~2K latent coords).

---

## 1. Static classical Hopfield with correlated patterns
- **Amit, Gutfreund, Sompolinsky (1985–87)** — the AGS mean-field theory of the Hopfield model: capacity
  α_c≈0.138 for iid patterns, the spin-glass/retrieval phase diagram, mixture ("spurious") states.
  *Phys. Rev. A 32, 1007 (1985); Ann. Phys. 173, 30 (1987).* Baseline against which all correlated-pattern
  work is measured; our λ_c(α) and the near-capacity breakdown (E26) are the delayed-model face of this.
- **Löwe, M. (1998), "On the storage capacity of Hopfield models with correlated patterns,"
  *Ann. Appl. Probab. 8(4), 1216–1250.*** Rigorous result: the standard Hopfield model *can* store a growing
  number of correlated patterns, provided the correlation comes from a homogeneous Markov chain; treats both
  **semantically** correlated (correlated in the pattern index, independent across neurons) and **spatially**
  correlated (correlated across neurons) patterns, relating capacity to a moderate-deviation bound on the
  empirical pattern correlation. **Directly relevant**: our Markov-flip "video" (E27) is exactly a
  homogeneous-Markov *spatial* correlation along the pattern index; Löwe's framework is the rigorous home of
  "how much correlation can still be stored," and his semantic/spatial distinction maps onto our
  along-sequence vs across-pixel correlation.
- **Personnaz, Guyon, Dreyfus (1985–86)** — the **projection / pseudo-inverse rule**
  J = ξ(ξᵀξ)⁻¹ξᵀ, which stores *linearly independent* (hence arbitrarily correlated) patterns as exact
  fixed points by inverting the pattern Gram matrix. *"Collective computational properties… new learning
  mechanisms," Phys. Rev. A 34, 4217 (1986); J. Physique Lett. 46, L359 (1985).* Also **Kanter & Sompolinsky
  (1987), Phys. Rev. A 35, 380** on the pseudo-inverse near saturation. **Relevant methodologically**: our
  exact reduction already carries the pattern Gram matrix `Gram=(1/N)ξξᵀ`; the reduced amplitudes are the
  Gram-corrected (projection-like) coordinates — the natural language for correlated patterns (see the E25
  cross-talk analysis, task 2).
- **Tsodyks & Feigelman (1988), "The enhanced storage capacity in neural networks with low level of
  activity," *Europhys. Lett. 6, 101–105.*** Low-activity/biased patterns; capacity α_c(p)∼1/(p|ln p|).
  Correlated/biased patterns require a modified (covariance) Hebb rule to avoid a condensed "ferromagnetic"
  background. **Relevant**: strong correlation in our videos produces exactly such a condensed background
  (the delocalized packet), the failure mode Tsodyks–Feigelman's covariance rule was designed to remove.
- **Fontanari (1990)** on categorization / hierarchically correlated patterns, and **Engel; Tarkowski &
  Lewenstein** on structured-pattern storage — the ultrametric/hierarchical-correlation branch. [volumes not
  re-verified] Peripheral to our uniform-along-sequence correlation but the reference frame for "structured
  correlations change the capacity and the basin geometry."

## 2. Temporal sequence retrieval and correlations between successive patterns
- **Sompolinsky & Kanter (1986), "Temporal association in asymmetric neural networks,"
  *Phys. Rev. Lett. 57, 2861–2864.*** The foundational delayed-synapse sequence model: a symmetric term
  stabilizes patterns, an **asymmetric shifted term with slow/delayed response** drives transitions
  xi^μ→xi^{μ+1}; mean-field condition on the transition gain. **This is the direct ancestor of our model**
  (our J + delayed K is a continuous-time, exactly-reducible instance); our T₁=τ+t_esc pacemaker law and the
  SNIC depinning are the fine structure they did not resolve.
- **Kleinfeld (1986), *PNAS 83, 9469*** and **Kleinfeld & Sompolinsky (1988)** — sequence generation via
  delayed asymmetric connections; identified transition-timing limitations. Contemporaneous with S–K.
- **Herz, Sulzer, Kühn, van Hemmen (1988–89); Herz, Li & van Hemmen (1991), "Statistical mechanics of
  temporal association…"** and **van Hemmen's "Temporal Association" chapter (in *Models of Neural Networks*,
  Springer, 1995).** Hebbian learning with a **broad distribution of delays** storing both static patterns
  and sequences; persistence-time estimates. **Relevant**: shows the delay is the generic clock for sequence
  recall — our single-τ model is the sharp, exactly-solvable limit.
- **Kühn & van Hemmen (1991)** — asynchronous dynamics dilutes sequential recall; temporal activity traces
  keep metastable states alive long enough for clean transitions. **Relevant to E27**: when correlation
  destroys the discrete metastable states, the transition mechanism itself fails — our "melting" is their
  dilution taken to completion.
- **Düring, Coolen & Sherrington (1998), "Phase diagram and storage capacity of sequence-processing neural
  networks," *J. Phys. A 31, 8607*** (arXiv:cond-mat/9805073). **Generating-functional (path-integral)
  solution** of Hopfield-type networks storing *sequences* near saturation: exact closed equations for the
  sequence overlap plus correlation and response functions. **The closest statistical-mechanics treatment to
  our regime**; it is the natural analytic counterpart to our finite-α exact reduction, and the place to look
  for a mean-field prediction of λ_c(α) and of the sequence-retrieval capacity. Coolen's two-part review
  *"Statistical mechanics of recurrent neural networks I–II"* (arXiv:cond-mat/0006010/0006011) is the
  textbook entry point.
- Recent: **"A Dynamical Theory of Sequential Retrieval in Input-Driven Hopfield Networks"**
  (arXiv:2603.03201, 2026) — current work on sequential retrieval dynamics; worth reading for overlap with
  our SNIC/pacemaker picture. [recent preprint, claims not independently re-verified]

## 3. Bridge to continuous / ring attractors
- **Amari (1977), "Dynamics of pattern formation in lateral-inhibition type neural fields,"
  *Biol. Cybern. 27, 77–87.*** Neural-field theory: a localized activity bump, once evoked, persists and
  **drifts to the maximum of the input** — the prototype of a continuous line/ring of marginally-stable
  states. Our GP-circle result (a bump sliding on a circular valley pushed by the delayed drive) is a
  disordered, delay-driven Amari bump.
- **Ben-Yishai, Bar-Or, Sompolinsky (1995), "Theory of orientation tuning in visual cortex,"
  *PNAS 92(9), 3844–3848.*** The **ring model**: Mexican-hat interaction on the ring of preferred
  orientations sustains a fixed-width bump whose *position* is free — a continuous attractor by construction.
  **Relevant**: our smooth-correlation network *self-organizes* into exactly this ring, but from *stored
  correlated patterns* rather than a hand-built Mexican hat.
- **Zhang (1996), "Representation of spatial orientation by the intrinsic dynamics of the head-direction cell
  ensemble," *J. Neurosci. 16, 2112–2126.*** Ring attractor for head direction; crucially, moving the bump at
  gain 1 requires an **asymmetric connectivity ∝ derivative of the symmetric one** — driven by an
  angular-velocity input. **Very relevant**: our K (cyclic shift) is precisely a discrete asymmetric
  "derivative" term, and the delay τ plays the role of the velocity drive that rotates the bump. Our moving
  recall wave = Zhang's moving HD bump, with the rotation clocked by the delay instead of a vestibular input.
- **Wu, Hamaguchi, Amari; Wu & Amari — Continuous Attractor Neural Networks (CANN)** (e.g. *Neural Comput.
  2008; arXiv:0808.2341, "A moving bump in a continuous manifold"*). The tracking dynamics of a bump on a
  continuous manifold — the dynamical-systems language for our sliding recall wave.

## 4. Is our specific dichotomy documented? (and the depinning link)
- **Local-vs-global correlation crossover (discrete memory → continuous attractor).** The *equivalence*
  "smoothly parameterized / strongly correlated stored patterns → continuous attractor" is folklore-to-
  established in the CANN literature (Amari; Ben-Yishai; Wu–Amari): a continuum of highly-overlapping patterns
  *is* a continuous attractor. What we did not find stated as such is the **sharp controlled contrast** — that
  *local* (Markov) correlation with no global structure **destroys** sequence retrieval while *smooth global*
  correlation **converts** it to a robust ring — obtained at fixed architecture by tuning only the pattern
  correlation *structure* (not length alone). This looks like a **novel, clean numerical statement**; the
  closest framing is Löwe's semantic-vs-spatial capacity distinction (§1) and the CANN "fine-tuning"
  literature (below), but neither poses the recall-cycle robustness dichotomy.
- **Pinning / depinning of a bump in a ring attractor with quenched disorder — the direct analogue of our
  weakest-bond SNIC.** This is an active and *highly relevant* line:
  - **"Information content in continuous attractor neural networks is preserved in the presence of moderate
    disordered background connectivity"** (arXiv:2304.13334, 2023) — replica/DMFT analysis: **random
    modulation of the interactions pins the bump** ("the bump can get stuck in the absence of neural noise");
    information is preserved up to a **threshold disorder strength**. This is *exactly* our physics: quenched
    disorder pins the moving wave; below a disorder threshold the wave still runs. Our SNIC-at-the-weakest-bond
    is the microscopic mechanism of this pinning, and the GP-circle (near-homogeneous bonds, sd(q)≈0.01) is
    the "below-threshold" clean-ring limit.
  - **"Symmetries and continuous attractors in disordered neural circuits"** (bioRxiv 2025.01.26.634933) —
    when/how a continuous attractor survives quenched disorder; the symmetry-breaking that pins it.
  - Classic background: **CANN require fine-tuning** (Seung; Tsodyks; Renart–Song–Wang 2003 on homeostatic
    tuning) — generic disorder destroys the marginal manifold and pins the bump. Our result quantifies the
    pinning as an *extreme-value depinning transition over quenched bond thresholds*, which is a fresh angle.

## Synthesis — known vs novel, and what to cite
**Known / has direct antecedents:** (i) storing correlated patterns and its capacity cost — AGS, Löwe,
Personnaz et al., Tsodyks–Feigelman; (ii) delayed-synapse sequence retrieval — Sompolinsky–Kanter, Kleinfeld,
Herz/van Hemmen, Düring–Coolen–Sherrington; (iii) smoothly-correlated patterns ⇒ continuous/ring attractor —
Amari, Ben-Yishai–Bar-Or–Sompolinsky, Zhang, Wu–Amari; (iv) quenched disorder pins a ring-attractor bump —
the 2023–2025 CANN-with-disorder papers.

**Appears novel in our study:** (a) the **exact finite-α reduction** of the *continuous-time delayed*
non-reciprocal model (no mean-field step) and the resulting per-pattern **quenched-saddle-node / SNIC-depinning
at the weakest bond** as the death mechanism of sequential recall — a sharpening of both the sequence-network
statics (Düring–Coolen–Sherrington) and the CANN-pinning story into an extreme-value-statistics law
(λ*=max_μ λ_c(μ)); (b) the **controlled Markov-vs-GP dichotomy** (local correlation kills recall / smooth
global correlation builds a robust ring) at fixed architecture; (c) the **delay-clocked** rotation (τ as the
angular-velocity drive of Zhang's bump) with the pacemaker law T₁=τ+t_esc.

**Recommended core citations for a paper (≤8):** Amit–Gutfreund–Sompolinsky 1987; Sompolinsky–Kanter 1986;
Löwe 1998; Düring–Coolen–Sherrington 1998; Amari 1977; Ben-Yishai–Bar-Or–Sompolinsky 1995; Zhang 1996; the
CANN-with-quenched-disorder paper (arXiv:2304.13334, 2023). Add Personnaz–Guyon–Dreyfus 1986 and
Tsodyks–Feigelman 1988 when discussing the reduced (Gram-corrected) coordinates and the correlated-pattern
melting, respectively.
