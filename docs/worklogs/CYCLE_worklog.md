# CYCLE_worklog — Stabilité du cycle pacemaker : journal auto-suffisant

**Créé :** 2026-07-01 · **Dernière mise à jour :** 2026-07-03 (auto-suffisant ; E1–E17 terminées — verdict SNIC complet ; E15 : front piégé delay-robuste ; E17 : fenêtre Hopf ultra-étroite + bistabilité de jumeaux mémoire)
**Modèle :** réseau de Hopfield mixte à délai, temps continu. **Auteur :** agent numérique (sessions Fable 5).
**Règles de tenue :** mise à jour continue pendant les runs ; chaque résultat = chiffres précis + référence code + figure ; graphes systématiques.

---

## 0. STATUT & COMMENT REPRENDRE

**État au 2026-07-02 — l'étude du cycle est essentiellement CONCLUE à N=2000 :**

- [x] Théorie : réduction P-dim exacte démontrée et certifiée (§3, audit 6/6)
- [x] Implémentation : `cycle_reduced.py`, `pacemaker_cycle.py`, `floquet_monodromy.py`, `pacemaker_scan.py`, `cycle_figures.py` (§11 codes)
- [x] E1 scan Floquet 13 λ : cycle hyper-stable sur [0.35, 0.95] (§6-E1)
- [x] E2 N-dépendance : T₁ convergé à 0.1% ; exposants vrais mesurés à N=500 (§6-E2)
- [x] E3–E4 : loi pacemaker T₁ = τ + t_échap(λ) exacte (collapse τ=5/10/20) (§6-E4, figA)
- [x] E5 : réfutation du SNIC-sur-mémoire ; chute sur front figé (§6-E5)
- [x] E6 : λ* ∈ (0.325, 0.33) ; dispersion des folds par pattern (§6-E6)
- [x] E7–E8 : fenêtre intermédiaire = chaos itinérant (λ_L≈+0.034) + états gelés exacts (§6-E7/E8, figC)
- [x] E9–E10 : VERDICT — mort du cycle = **SNIC localisé au maillon le plus faible (liaison 91), λ*≈0.327, loi √ confirmée, homocline rejetée** (§6-E9/E10, figB)
- [x] Figures de présentation figA–figD (§10)
- [x] **E11** : spectres de Floquet complets résolus (figE + npz) — confirmation SNIC indépendante (μ→0 car T→∞) — 2026-07-02
- [x] **E12** : bifurcation LUE sur les valeurs propres locales — eig_max(M_front-piégé) réel, monte en −c√(λ*−λ) (c≈2.1, λ*_spectral≈0.328), dépiégeage à 0.328 complémentaire du cycle ⇒ SNIC (figF) — 2026-07-02
- [x] **E13** : universalité du mécanisme — 4 seeds → 4 maillons faibles (91/64/84/0), λ* ∈ [0.315, 0.328], forme du front quasi-universelle [0.68, 0.32] — 2026-07-02
- [x] **E14** : contraction par tour SANS plancher (Benettin/QR) — A = 220→440 vers λ*, z₁ fini (−0.23), |μ₁| ~ e^{−440} : mort purement géométrique — 2026-07-02
- [x] **E15** : marges de Hopf du front piégé vs τ — érosion confirmée MAIS loi pur-1/τ (marge×τ plateau 0.79–2.2) ⇒ front piégé delay-robuste (pas de τ* ≤ 100), contrairement à la mémoire (décroît plus vite que 1/τ, τ*≈52 à α=0.07) ; mode lent = demi-onde de délai T→2τ (préasymptote 3τ) — figG — 2026-07-02
- [x] **E17 (+E17b/c)** : sondage non linéaire de la fenêtre Hopf de la mémoire (α=0.07, τ=100) — PAS de croissance Hopf observée : fenêtre (λ_Hopf, fold) plus étroite que la résolution de placement (≲5e−5) ; DÉCOUVERTE à la place : **bistabilité de jumeaux mémoire** à distance 1.01e−2 près du fold (second point fixe stable, res 2.5e−15, marge 0.038) ; familles de modes confirmées (lente 2.6τ marginale + harmonique 0.78τ) — figH v2 — 2026-07-03
- [ ] Ouverts : certification N=10⁴ (GPU, λ*(N) par extrêmes), fraction de gel des bassins, prédiction petit-τ (fenêtre chaotique), V3 Benettin, V5 limite λ=1, théorie de la delay-robustesse du front piégé (E15-2)

**Pour reprendre en session neuve :** lire §1–§4 (modèle + méthode), §6 (expériences), §9 (synthèse). Contexte statique : `REPORT_static_bifurcation.md`. Environnement : conda `mcmc_env`, float64 CPU (le float32 GPU est proscrit pour les solveurs — leçon documentée dans NOTES 2026-07-01). Données et figures : organisation thématique par phase, `results/<phase>/data/` et `results/<phase>/figures/` — voir `results/README.md` (réorganisation 2026-07-10).

---

## 1. Modèle, paramètres, contexte statique

**Dynamique** (DDE, temps continu) :
$$t_0\,\dot u(t) = -u(t) + (1-\lambda)\,J\,g(u(t)) + \lambda\,K\,g(u(t-\tau)),\qquad g=\tanh(\beta\,\cdot)$$
avec $J=\frac1N\sum_\mu \xi^\mu(\xi^\mu)^T$ (ancrage symétrique), $K=\frac1N\sum_\mu \xi^{\mu+1}(\xi^\mu)^T$ (poussée cyclique, indices mod P), patterns $\xi^\mu\in\{\pm1\}^N$ i.i.d.

**Paramètres de référence :** N=2000, α=0.05 (P=100), τ=10, β=20, t₀=1, seed 42, dt=0.01.

**Contexte statique acquis** (`REPORT_static_bifurcation.md`, machinerie `robust_branch.py`/`reduced_spectrum.py`, audit `audit_static_analysis.py` 15/15) :
- fold de la branche mémoire ξ¹ : **λ_c = 0.2822** (N=2000 seed 42) ; multi-seed N=10⁴ : λ_c(α=0.05)=0.263±0.002 ;
- transition statique = selle-nœud (racine réelle en z=0, identité Δ(0)=−M), pas de Hopf pour τ≤50 à α=0.05 ;
- **dispersion des folds par pattern** : des états mémoire sur d'autres patterns survivent jusqu'à λ≈0.3025 (mesuré §6-E6b).

---

## 2. Questions scientifiques

**Q1** — Construire le cycle de rappel séquentiel (pacemaker) et mesurer sa loi de rythme T₁(λ,τ). ✅ §6-E4.
**Q2** — La transition « dynamique lente » ↔ « pacemaker » du legacy (λ_c⁽²⁾≈0.59, temps discret) : existe-t-elle en continu, où, et de quelle nature ? ✅ §6-E1/E9 : pas de bifurcation locale ; la vraie frontière est λ*≈0.327 (dépiégeage).
**Q3** — La branche de cycles rejoint-elle λ_c (SNIC sur la mémoire) ? ✅ §6-E5 : NON — elle meurt à λ*>λ_c sur le front piégé.

---

## 3. La clef de voûte : réduction P-dimensionnelle EXACTE

**Théorème.** Soit $X=[\xi^1\cdots\xi^P]\in\mathbb R^{N\times P}$, $S$ la permutation cyclique $(Sm)_\nu=m_{\nu-1}$. Alors :

(i) *Invariance* : le drive est dans span(X) : $J g(u)=X\,m(u)$, $K g(u)=X\,S\,m(u)$, avec $m(u)=\frac1N X^T g(u)\in\mathbb R^P$.

(ii) *Attractivité exacte* : pour $u=Xa+w$ ($w\perp$), $t_0\dot w=-w$ **exactement** ⇒ $w(t)=w(0)e^{-t/t_0}$.

(iii) *Dynamique réduite exacte* ($X$ injectif p.s.) :
$$t_0\,\dot a = -a + (1-\lambda)\,m(a(t)) + \lambda\,S\,m(a(t-\tau)),\qquad m(a)=\tfrac1N X^T\tanh(\beta X a).$$
Ce n'est **pas** du mean-field : m(a) contient tout le désordre d'échantillon à α fini.

(iv) *Spectre transverse trivial* : autour de toute solution, $\delta u=X\delta a+\delta w$ donne $t_0\delta\dot w=-\delta w$ exactement (les couplages $JD\delta u$, $KD\delta u$ sont dans span(X) ∀δu) ⇒ multiplicateurs transverses $=e^{-T/t_0}$ (multiplicité N−P). **Tout le spectre de Floquet non trivial vient du problème variationnel P-dim** :
$$t_0\,\delta\dot a = -\delta a + (1-\lambda)\,G(t)\,\delta a(t) + \lambda\,S\,G(t-\tau)\,\delta a(t-\tau),\quad G(t)=\tfrac1N X^T D(t)X,\ D(t)=\mathrm{diag}\big(\beta(1-\tanh^2(\beta Xa_p(t)))\big).$$

**Piège d'indices (résolu)** : le variational réduit porte $S\!\cdot\!G$ alors que le T_P statique porte $G\!\cdot\!S$ ; équivalence spectrale par AB↔BA ($G W$ et $W G$, $W=(1-\lambda)I+\lambda e^{-z\tau}S$). Convention gelée : `S = np.roll(np.eye(P),1,axis=0)`, terme retardé du RHS = `np.roll(m,1)`.

**Certification numérique** (`cycle_reduced.py __main__`, N=500, λ=0.22, 2026-07-02) :
| Test | Contenu | Erreur |
|---|---|---|
| V0a | u* ∈ span(X) | 1.17e−15 |
| V0b | G ≡ G_J statique | 3.70e−14 |
| V0c | G_K = G·S | 3.70e−14 |
| V0d | racines T_P annulent T_red (à 0.97·fold) | 1.61e−15 |
| V1 | trajectoire P-dim ≡ plein-N (t=8) | **5.49e−16** |
| V4 | RK4+Hermite : ordre 4.1 ; err(dt 0.01 vs 0.005) | 1.73e−12 |

---

## 4. Méthodologie de Floquet : mathématiques et algorithme (détail complet)

### 4.1 Cadre
Le cycle $a_p(t)=a_p(t+T)$ vit dans l'espace des histoires $\mathcal C=C([-\tau,0],\mathbb R^P)$. L'**opérateur de monodromie** $U:\mathcal C\to\mathcal C$ transporte un segment d'histoire de perturbation sur une période :
$$(U\,v)(\theta)=\delta a(T+\theta;\,v),\quad \theta\in[-\tau,0],$$
où $\delta a(\cdot;v)$ résout le variational (§3-iv) avec histoire initiale $v$. $U$ est **compact** (T≫τ) ⇒ spectre ponctuel $\{\mu_k\}\to0$ : les **multiplicateurs de Floquet**. Exposants : $z_k=\ln\mu_k/T$. Stabilité orbitale ⟺ $|\mu_k|<1\ \forall k$ **sauf** $\mu_0=1$ (mode de phase, vecteur propre $=\dot a_p|_{[-\tau,0]}$ — mode de Goldstone de l'invariance par translation temporelle).

### 4.2 Discrétisation (`floquet_monodromy.py`, classe `Monodromy`)
- **Représentation de l'état** : grille uniforme dt sur $[-\tau,0]$ : $v=(\delta a(-L\,dt),\dots,\delta a(0))\in\mathbb R^{(L+1)P}$, $L=\tau/dt=1000$ ⇒ **dim = 101 000** (N=2000, P=100, dt=0.01). Choix « grille = buffer » : aucun aller-retour d'interpolation entre représentation et intégrateur.
- **matvec** $=$ intégration du variational sur $n_T=\mathrm{round}(T/dt)$ pas ($n_T=108\,324$ pour T=1083.24) : méthode des pas, **RK4** avec valeurs retardées par **interpolation d'Hermite cubique** du buffer (positions + dérivées stockées) — le schéma certifié V1/V4.
- **Coefficients** : $G(t)\delta a=\frac1N X\big(g(t)\odot(X^T\delta a)\big)$ avec $g(t)=\beta(1-\tanh^2(\beta X a_p(t)))$ ; le terme retardé applique ensuite `roll(·,1)` (= S). Les gains $g$ sont **précalculés une fois** sur la demi-grille des temps d'étage RK4 : $t_s=-\tau+s\,dt/2$, $s=0..2(L+n_T)$, soit ~218 650 vecteurs de taille N en **float32** (1.75 GB, garde-fou 4 GB) — licite car ce sont des *coefficients* (pas un solveur ; erreur relative 1e−7 ⇒ décalage d'exposant ≤ 1e−7·T ~ 1e−4).
- Le cycle $a_p$ vient de `pacemaker_cycle.build_cycle` (§5), stocké sur $[-\tau,T]$ avec dérivées (Hermite vers la demi-grille).
- **Valeurs propres** : Arnoldi ARPACK (`scipy eigs`, which='LM', k=8, ncv=22, tol 1e−6, maxiter 200), matrix-free ; ~20–40 matvecs. Coût mesuré : **376–627 s par λ** (cycle + gains + Arnoldi, N=2000).

### 4.3 Validations spécifiques Floquet
- **V2 (μ₀=1)** : mesuré |μ₀−1| = 3.12e−4 (N=500, τ=5, λ=0.9) — égal à la *fermeture* du cycle extrait (6.03e−4), comme attendu (l'erreur de période δT se lit sur μ₀) ; alignement du vecteur propre avec $\dot a_p$ : **1.0000**. Sur le scan N=2000 : |μ₀−1| ∈ [7.0e−5, 4.3e−3] ≃ fermetures, alignement 1.000 partout (§6-E1).
- **Plancher numérique (identifié et vérifié)** : les $|\mu_{k\ge1}|$ rapportés ~2e−16 sont le plancher d'arrondi (les vrais multiplicateurs sont plus petits : $e^{\mathrm{Re}z\cdot T}$ avec Re z≈−0.06 et T≈1083 ⇒ $\sim e^{-65}$). Preuve : les « Re z* » imprimés ≡ ln(plancher)/T — p.ex. ln(2.725e−16)/1083.24 = **−0.0331** = valeur imprimée à λ=0.90 ; ln(8.534e−16)/1633.5 = **−0.0212** à λ=0.35. ⇒ la colonne `rez` du npz du scan est une **borne supérieure**, pas une mesure.
- **Exposants vrais** : mesurables quand $e^{\mathrm{Re}z\,T}$ > plancher, i.e. à petit N (T=P·T₁ ∝ N). Mesures E2 : **Re z* = −0.0600 (N=500, λ=0.90) ; −0.0791 (N=500, λ=0.65)** (T≈271 ⇒ μ~8e−8, résolu) ; N=1000 (T≈542, μ~1e−15) : borderline plancher (−0.0637, −0.0638 : à considérer contaminés). E11 archive les spectres complets résolus à N=500 (figE).

### 4.4 Où sont les résultats spectraux
- Résumés par λ (N=2000) : `results/2_cycle_rappel_snic/data/floquet_scan_N2000_tau10.0.npz` (champs : lam, ok, T1, closure, mu0_err, mu_star, rez — avec le caveat plancher ci-dessus) + figure `floquet_scan_N2000_tau10.0.png` (3 panneaux : Re z* [borne], |μ*| [plancher], T₁(λ)).
- **Spectres complets dans le plan complexe (résolus)** : `results/2_cycle_rappel_snic/data/E11_floquet_spectra_N500.npz` (μ_k, T pour λ=0.90/0.65/0.45) + **figE_floquet_spectrum.png** (cercle unité, μ₀ étoile rouge, non-triviaux) — E11, en cours au moment de cette réécriture ; résultats insérés en §6-E11 dès disponibles.

---

## 5. Construction du cycle (`pacemaker_cycle.py`, fonction `build_cycle`)
Simulation de la DDE réduite depuis l'IC mémoire (a=0.99·e₁, histoire constante), fenêtre $(s+1.35)\,P(\tau+2)$ ; suivi du front par $\mathrm{lead}(t)=\arg\max_\nu a_\nu(t)$ ; les P+1 derniers changements de lead donnent les temps de pas ; extraction d'une période complète $[-\tau,T]$ (positions + dérivées) + **fermeture** $\|a(T)-a(0)\|/\|a(0)\|$.
**Mesures de référence** : N=500/τ=5/λ=0.9 : T₁=5.824±0.017, T=145.59, fermeture 6.0e−4. **N=2000/τ=10/λ=0.9 : T₁=10.832±0.011 (τ+1=11), T=1083.24, fermeture 1.5e−3**, désordre des pas ±0.011. Figure : `results/2_cycle_rappel_snic/figures/waveform_lam0.9_tau10.0_N2000.png`.

---

## 6. Journal des expériences (protocoles précis, chiffres, figures)

### E1 — Scan Floquet descendant (`pacemaker_scan.py`) — 2026-07-02, 91 min
**Protocole :** N=2000, τ=10, λ=0.95→0.35 (13 pts) ; par λ : cycle (IC mémoire) + monodromie k=8.
**Résultats** (npz/figure `floquet_scan_N2000_tau10.0.*`) :
| λ | 0.95 | 0.90 | 0.85 | 0.80 | 0.75 | 0.70 | 0.65 | 0.60 | 0.55 | 0.50 | 0.45 | 0.40 | 0.35 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T₁ | 10.774 | 10.832 | 10.901 | 10.983 | 11.081 | 11.203 | 11.357 | 11.558 | 11.831 | 12.222 | 12.831 | 13.905 | 16.335 |
|μ₀−1| ≤ 4.3e−3 partout (= fermeture), alignement 1.000 ; |μ*| = 1.7–8.5e−16 = **plancher** (§4.3) ⇒ contraction réelle ≥ e^{−36}/période (borne), vraie valeur ~e^{−65}.
**Conclusions :** (1) cycle **hyper-stable sur tout [0.35,0.95]** — pas de bifurcation locale ; le λ_c⁽²⁾≈0.59 du legacy (temps discret + MCMC thermique) n'existe pas en continu déterministe ; (2) T₁ accélère vers le bas (incréments +0.06→+2.43) → transition ailleurs (E5–E10).

### E2 — N-dépendance des exposants — 2026-07-02
**Protocole :** N∈{500,1000} × λ∈{0.90,0.65}, k=6.
**Résultats :** T₁ : 10.826/10.836 (λ=0.9, N=500/1000), 11.348/11.361 (0.65) — **convergé à 0.1%**. Re z* : **−0.0600 / −0.0791** (N=500, mesures vraies, §4.3) ; −0.0637/−0.0638 (N=1000, borderline plancher).

### E3 — T₁ vers λ_c⁺ (test SNIC-sur-mémoire, volet A) — 2026-07-02
**Protocole :** IC mémoire, λ∈{0.34,0.32,0.30,0.29,0.285}, fenêtre 5220.
**Résultats :** λ=0.34 : T₁=17.279 ; λ=0.32 : 91 avancées (ambigu) ; λ≤0.30 : 1 avancée (verrouillage immédiat). T₁·√(λ−λ_c) = 5.26/4.77/4.25/4.15 (λ=0.45/0.40/0.35/0.34) — dérive ⇒ loi √ vers λ_c NON confirmée. Volet B planté (bug seed, consigné) → refait en E5.

### E4 — Collapse en τ : T₁ = τ + t_échap(λ) — 2026-07-02 — **figA**
**Protocole :** τ∈{5,20} × λ∈{0.9,0.6,0.45,0.35}, sim seule, fenêtre 1995 (τ=5).
**Résultats** (réf. τ=10 entre parenthèses) :
| λ | t_échap(τ=5) | (τ=10) | (τ=20) |
|---|---|---|---|
| 0.90 | 0.831 | 0.83 | 0.832 |
| 0.60 | 1.555 | 1.56 | 1.558 |
| 0.45 | 2.824 | 2.83 | 2.831 |
| 0.35 | (fenêtre) | 6.34 | 6.335 |
**Conclusions :** décomposition **EXACTE** (~4 chiffres) ; le délai est un plancher additif pur ; « pacemaker strict » = t_échap≪τ (frontière τ-dépendante). L'« échec » τ=5/λ=0.35 prouvé artefact de fenêtre : 177 avancées × T₁ prédit 11.34 = 2007 ≈ fenêtre 1995. Figure : `figures/figA_pacemaker_law_collapse.png`.

### E5 — Balayage descendant ENSEMENCÉ (hystérésis) — 2026-07-02 — **décisif**
**Protocole :** seed = cycle vivant λ=0.34 (T₁=17.28) ; λ=0.32→0.25 (7 valeurs), fenêtre 6000, ré-ensemencement adiabatique.
**Résultats :** 0 avancée en 2ᵉ moitié à TOUT λ≤0.32 ; état final = **front figé 2-patterns** : a_trié = [0.68,0.31,…] (0.32) → [0.75,0.24,…] (0.25). Jamais la mémoire pure, même sous λ_c.
**Conclusions :** H1 (SNIC-sur-mémoire) **réfutée** ; H3 (fenêtre) réfutée ; branche de cycles morte à λ*∈(0.32,0.34) ; **bistabilité front-piégé/mémoire sous λ_c**.

### E6 — Encadrement λ* + machinerie statique — 2026-07-02
**(a)** seed cyclant, t=6000 : λ=0.335 : 167 av., T₁=17.92 ; 0.33 : 160 av., 18.72 ; 0.325 : 0 → **λ*∈(0.325,0.33)**. Fit log T₁≈10.75−1.65·ln(λ−0.322) (piste homocline H2′, réglée par E9).
**(b)** Polissage Newton du front piégé → **saut de bassin** vers une mémoire quasi-pure m=[0.996,0.051,0.05] (res 2.1e−15, eigM=−0.3384) stable jusqu'à λ≈0.3025 ⇒ **dispersion des folds par pattern** (sous-produit) ; test à refaire amorti (E8b l'a remplacé).

### E7 — Recensement d'attracteurs / test spin-glass (λ=0.31) — 2026-07-02 — **figC (gauche)**
**Protocole :** 40 IC aléatoires in-span (σ=0.3), t=1000. npz `E7_attractor_census_lam0.31.npz`.
**Résultats :** 0/40 stationnaires (drift~1e−2) ; états à 6–12 composantes macroscopiques (>0.15) ; **P(q)** : moyenne 0.441, médiane 0.417, q>0.9 : 20/780, q<0.3 : 272/780, histogramme (10 bacs) [92,85,95,100,87,70,89,71,71,20] — **large et quasi-plat**.

### E8 — Triple discriminant — 2026-07-02 — **figC (centre)**
**(a)** t=10⁴ depuis IC aléatoire (λ=0.31) : ‖ȧ‖ = 3.27e−2 (t=100) → 2.02e−2 (10³) → 2.10e−2 (6×10³) → 1.17e−2 (10⁴) : **plateau, pas de vieillissement** ; **Lyapunov λ_L ≈ +0.0339** (2 trajectoires, fenêtre 300 ; séparation 10⁻⁵→~0.3 : croissance exponentielle certaine, valeur à raffiner Benettin). 1/λ_L≈29≈3τ (≈ période du Hopf lent du fold statique — connexion possible).
**(b)** front piégé (reconstruit λ=0.30) : **‖ȧ‖ = 0.0 exactement** à t=2400/4000/6000, a=[0.699,0.283,0.04] — **point fixe vrai**.
**(c)** λ=0.332/0.331/0.329/0.328 : T₁=18.33/18.48/18.85/19.31 ; 0.327 : mort. T₁ MOYEN fini à la mort (~19.3) → biais de moyenne identifié (∝1/P) → E9.

### E9 — Statistique PAR LIAISON près de λ* — 2026-07-02 — **VERDICT** — figB (droite), figC (droite)
**Protocole :** seed cyclant, t=8000, durées de passage individuelles t_μ (2ᵉ moitié). npz `E9_bond_times.npz` (λ=0.328).
| λ | t_méd | t_90% | **t_MAX** | liaison |
|---|---|---|---|---|
| 0.332 | 17.80 | 20.37 | 35.62 | **91** |
| 0.330 | 18.02 | 20.94 | 41.45 | **91** |
| 0.329 | 18.13 | 21.12 | 46.91 | **91** |
| 0.328 | 18.30 | 21.41 | **72.87** | **91** |
Top-8 (0.328) : [72.9,72.9,28.2,28.2,25.2,25.2,23.3,23.3] (2 tours : reproductibilité du désordre ✓).
**Lois sur t_MAX :** racine (SNIC) : C=t_MAX·√(λ−0.327) = 2.52/2.27/2.10/2.30 = **2.3±0.2 → TIENT** ; log (homocline) : pente locale 11.6→13.5→37.1 → **REJETÉE**. Linéarisation 1/t²_MAX → intercept **λ*=0.3262** (inset figB). Seuil n°2 ≈ 0.322 (t₂≈28 inversé par la loi √).
**Verdict : mort du cycle = SNIC LOCALISÉ à la liaison la plus faible (91)** ; la médiane bouge à peine (17.8→18.3) : une liaison meurt, 99 restent passantes — statistique d'extrêmes.

### E10 — Position de piégeage vs profondeur de trempe — 2026-07-02
| trempe → λ | arrêt sur | composition |
|---|---|---|
| 0.325 | **(91,92)** | [0.669, 0.320] |
| 0.322 | **(91,92)** | [0.675, 0.316] |
| 0.30 | (9,10) | [0.699, 0.283] |
**Conclusion :** juste sous λ*, seule la liaison 91 bloque → arrêt exactement dessus (prédiction E9 vérifiée au pattern près) ; en trempe profonde plusieurs liaisons bloquent (n°2≈0.322 ; la 9 bloque dès ≥0.30) → arrêt à la première rencontrée. **Implication N : λ*(N)=max de P=αN seuils → croît avec N (extrêmes).**

### E11 — Spectres de Floquet complets (plan complexe) — FAIT (2026-07-02) — **figE**
**Protocole :** N=500 (T≈271–322 ⇒ non-triviaux > plancher aux λ hauts), λ∈{0.90,0.65,0.45}, k=12, tol 1e−10, maxiter 400. Sorties : `E11_floquet_spectra_N500.npz` + `figures/figE_floquet_spectrum.png`.
**Résultats (multiplicateurs μ_k et exposants z_k=ln μ_k/T) :**
| λ | T | |μ₀−1| | μ₁…μ₅ (non triviaux) | Re z₁…z₅ |
|---|---|---|---|---|
| 0.90 | 270.6 | 8.6e−4 | 8.8e−8, 7.8e−8, 4.0e−8, 2.7e−8, 2.2e−8 (tous réels>0) | −0.0600, −0.0605, −0.0629, −0.0644, −0.0652 |
| 0.65 | 283.7 | 1.3e−3 | 1.8e−10, 1.3e−10, (3.7±1.9i)e−11 (paire), 3.1e−11 | −0.0791, −0.0804, −0.0843 (×2), −0.0853 |
| 0.45 | 321.8 | 3.3e−5 | ~3e−16 (PLANCHER : ln(3.6e−16)/321.8=−0.1105 = borne) | ≤ −0.110 (borne) |
**Trois enseignements :**
1. **Structure spectrale résolue** : μ₀=1 isolé ; puis une **bande dense de multiplicateurs réels positifs** (Re z ∈ [−0.060, −0.068] à λ=0.9) — la bande des modes de déplacement de front (« phasons ») avec levée de dégénérescence par le désordre ; une seule paire complexe marginale à λ=0.65.
2. **Confirmation SNIC indépendante** : les multiplicateurs **décroissent** vers la transition (8.8e−8 → 1.8e−10 → ≤3e−16 pour λ=0.90→0.45), car T croît à Re z quasi-stable → μ=e^{zT}→0. C'est la signature d'une terminaison **SNIC** (période divergente ⇒ contraction totale par période → ∞) — un fold cyclique exigerait au contraire μ→+1. Converge avec le verdict E9.
3. **Contraction par unité de temps renforcée vers le bas** (−0.060 → −0.079 → ≤−0.11) : près du piégeage, le front s'assoit plus profondément dans chaque puits entre les sauts — cohérent avec l'image de dépiégeage.

### E12 — Lecture SPECTRALE de la bifurcation (selle-nœud du front piégé) — FAIT (2026-07-02) — **figF**
**Motivation (question utilisateur, 2026-07-02) :** λ* a été établi par des preuves
TRAJECTORIELLES (mort du cycle, loi √ de t_MAX, position de piégeage) et par la
fuite μ→0 (E11) — jamais par une lecture spectrale directe. Point théorique : un
SNIC est une bifurcation GLOBALE, invisible comme croisement du cercle unité sur
le spectre de Floquet du cycle (sa signature y est T→∞, μ→0 — vue en E11, et
c'est ce qui l'y distingue d'un fold de cycles μ→+1). En revanche le selle-nœud
sous-jacent SE LIT sur le jacobien local du POINT FIXE (front piégé) :
**eig_max(M_pinned)(λ) doit traverser 0 à λ*** — lecture τ-indépendante et valable
pour la DDE complète via l'identité Δ(0)=−M (auditée).
**Protocole :** front piégé exact (liaison 91) obtenu par trempe 0.34→0.318
(t=3000) ; continuation en λ CROISSANT par correcteur DYNAMIQUE (reconvergence
t=400–900 par pas — aucune possibilité de saut de bassin, contrairement au Newton
d'E6b) ; à chaque pas : résidu ‖F‖, composition (a₉₁,a₉₂), **eig_max(M) exact
(Sylvester P×P)** ; arrêt au dépiégeage (le front repart). Sorties :
`E12_pinned_branch.npz` + `figures/figF_pinned_fold_spectral.png` (eig_max(λ) → 0,
avec λ* dynamique 0.327–0.328 et intercept E9 0.3262 superposés).
**Prédiction falsifiable :** eig_max monte vers 0 et la branche meurt dans
[0.326, 0.328] ; la valeur propre qui s'annule est RÉELLE (selle-nœud).
**Résultats (2026-07-02) — PRÉDICTION VÉRIFIÉE :**
| λ | 0.318 | 0.320 | 0.322 | 0.324 | 0.325 | 0.326 | 0.3265 | 0.327 | 0.3275 | 0.328 |
|---|---|---|---|---|---|---|---|---|---|---|
| eig_max(M) | −0.313 | −0.225 | −0.119 | −0.103 | −0.146 | −0.109 | −0.094 | −0.063 | **−0.058** | **DÉPIÉGÉ** |
Résidu ‖F‖ ≈ 6e−15 partout (point fixe machine-exact suivi par la DDE elle-même
→ stabilité retardée vérifiée par construction). Composition dérivant doucement
(a₉₁ : 0.681→0.664 ; a₉₂ : 0.311→0.324 — le front « se penche » vers la
libération).
**Conclusions :**
1. Valeur propre dominante RÉELLE montant vers 0 selon la loi de selle-nœud
   eig ≈ −c√(λ*−λ) : fit **λ*_spectral ≈ 0.328, c ≈ 2.1** (vérif : −2.1√0.001 =
   −0.066 vs mesuré −0.063 à λ=0.327). Lecture exacte de 0 impossible par
   ralentissement critique du correcteur (t_conv ~ 1/|eig|) — dernier point à
   Δλ=5e−4 du bord.
2. **Existence complémentaire** front-piégé/cycle : dépiégeage à 0.3280 = là où
   le cycle renaît (E8c) ; pas de coexistence ⇒ SNIC (pas fold de cycles).
3. **Triple concordance** : λ*_dyn ∈ (0.327, 0.328) [E8c] ; λ*_ghost 0.3262–0.327
   [E9] ; λ*_spectral ≈ 0.328 [E12]. La bifurcation est désormais LUE sur les
   valeurs propres locales — côté point fixe, seul endroit où un SNIC se lit.
4. Bosse non monotone vers 0.324–0.325 : croisement de niveaux probable entre
   deux modes réels (structure fine à creuser si utile).
Figure : `figures/figF_pinned_fold_spectral.png` ; données : `E12_pinned_branch.npz`.

### E13 — Statut de la « liaison 91 » : universalité vs échantillon — FAIT (2026-07-02)
**Motivation (objection utilisateur, 2026-07-02) :** la liaison 91 n'a AUCUN rôle
intrinsèque — les patterns sont tirés aléatoirement par seed. Clarification du
statut épistémologique :
- liaison 91 = **argmax du tirage seed 42** parmi les P seuils gelés {λ*_μ} ;
  sa récurrence dans E9/E10/E12 est un CONTRÔLE DE COHÉRENCE à désordre fixé
  (trois observables indépendantes doivent désigner le même maillon si le
  mécanisme est correct), pas une propriété du site 91.
- **Universel (mécanisme)** : dépiégeage par selle-nœud local au maillon le plus
  faible ; loi √ du fantôme ; T₁=τ+t_échap(λ) ; existence complémentaire
  cycle/front-piégé ; hyper-stabilité du cycle.
- **Aléatoire d'échantillon** : identité du maillon faible ; λ* = max_μ λ*_μ
  (statistique d'extrêmes → λ*(N,α) croît avec P=αN) ; seuil n°2 (0.322) ;
  λ_c (±0.01 inter-seed, cf. étude statique).
**Protocole E13 :** seeds {7, 123, 2024} : cycle construit à λ=0.40 (IC mémoire),
descente adiabatique λ∈{0.37,…,0.31} (t=1000/pas), détection de la mort +
lecture du site de piégeage (paire dominante de a final). **Prédictions :**
liaisons de piégeage DIFFÉRENTES d'une seed à l'autre ; λ* différents (dispersion
d'extrêmes) ; même phénoménologie (cycle → front figé 2-patterns).
**Résultats (2026-07-02) — LES TROIS PRÉDICTIONS VÉRIFIÉES :**
| seed | λ* (encadrement) | maillon faible | composition du front figé |
|---|---|---|---|
| 42 (réf.) | (0.327, 0.328) | (91, 92) | [0.67, 0.32] |
| 7 | (0.31, 0.32) | **(64, 65)** | [0.69, 0.32] |
| 123 | (0.32, 0.33) | **(84, 85)** | [0.67, 0.31] |
| 2024 | (0.32, 0.33) | **(0, 1)** | [0.68, 0.33] |
**Conclusions :** (1) le site de piégeage est du pur bruit d'échantillon (4 seeds
→ 4 liaisons différentes) — la « liaison 91 » n'avait aucun rôle intrinsèque,
comme soulevé par l'utilisateur ; (2) λ* fluctue de ±0.006 autour de ~0.323
(première mesure de la distribution d'extrêmes — 4 échantillons) ; (3) la
PHÉNOMÉNOLOGIE est universelle, y compris un détail remarquable : la
**composition du front figé est quasi-universelle ≈ [0.68, 0.32]** d'une seed à
l'autre — la forme du front dépiégé est auto-similaire, seul son site et son
seuil sont aléatoires.
**Complément (question utilisateur sur figE) :** figE v1 (plan complexe linéaire)
est visuellement dégénérée (10⁻⁸…10⁻¹⁶ tous écrasés sur l'origine) →
**figE_floquet_spectrum_v2.png** : |μ_k| en LOG (montre l'effondrement
8.8e−8→1.8e−10→≤3e−16 et qu'AUCUN multiplicateur ne monte vers 1 — le message
principal, signature d'invisibilité du SNIC côté cycle) + bande des exposants
Re z_k. NB : E11 tourne à N=500 → réalisation de désordre DIFFÉRENTE de N=2000
(même numéro de seed mais patterns différents) — la liaison 91 n'y a pas de sens.

### E14 — Contraction totale par tour A(λ) = −ln|μ₁| PRÈS de la mort — FAIT (2026-07-02)
**Motivation (question utilisateur, 2026-07-02) :** pourquoi E11 s'arrête à
λ=0.45 ? Parce qu'Arnoldi mesure |μ| en précision RELATIVE à μ₀=1 (plancher
2e−16) et que μ₁=e^{z₁T} s'effondre quand T diverge — près de λ*, |μ| est
inaccessible à tout N par cette méthode. La bonne observable près de la mort est
**A(λ) = −ln|μ₁|** (contraction par tour), mesurable SANS plancher par itération
orthogonale à 2 vecteurs (Benettin/QR toutes les ~T₁ : on accumule des LOGS de
normes ; contrôle intégré : Σlog du 1er vecteur ≈ 0 = ln μ₀).
**Protocole :** N=2000 seed 42, λ ∈ {0.40, 0.335, 0.329}, `e14_contraction_near_death.py`
(réutilise Monodromy._rhs + cache de gains ; QR tous k₁=round(T₁/dt) pas).
**Enjeu :** A sature (cinématique SNIC pure : z₁=−A/T→0 trivialement) vs A croît
(approfondissement des puits par liaison, cohérent avec E12). Réf. E11 (N=500,
P=25) : A = 16.2 / 22.4 / ≥35.7 pour λ = 0.90/0.65/0.45, soit A/P = 0.65/0.90/≥1.4.
**Résultats (2026-07-02) :**
| λ | T | ln|μ₀| (contrôle) | A = −ln|μ₁| | z₁ = −A/T | A/P (par liaison) |
|---|---|---|---|---|---|
| 0.400 | 1390.5 | **0.001** | 219.7 | −0.158 | 2.20 |
| 0.335 | 1789.6 | **0.001** | 419.5 | −0.234 | 4.20 |
| 0.329 | 1890.4 | **0.001** | 440.1 | −0.233 | 4.40 |
**Conclusions :**
1. **Contrôle parfait** : Σlog du mode de phase = 0.001 ≈ 0 (= ln μ₀) aux trois λ
   — la machinerie Benettin/QR est validée de bout en bout.
2. **A CROÎT vers la mort** (220 → 440) : chaque passage de liaison contracte
   e^{−4.4} près de λ* (A/P : 2.2 → 4.4) ; ralentissement de la croissance entre
   0.335 et 0.329 (+5%) — début de saturation possible tout près de λ*.
3. **|μ₁(λ→λ*)| ~ e^{−440}** : les multiplicateurs FUIENT le cercle unité de
   façon astronomique à l'approche de la mort — la quantification floor-free
   définitive de l'« invisibilité » du SNIC côté cycle (et la raison pour
   laquelle Arnoldi était sans espoir : plancher 2e−16 vs e^{−440}).
4. **z₁ reste fini (−0.23), aucun ramollissement** : la contraction par unité de
   temps ne s'adoucit PAS à l'approche de λ* — le cycle meurt en restant
   brutalement stable ; sa disparition est purement géométrique (collision avec
   le point fixe naissant, lue en E12), pas spectrale.
5. Comparaison E11 (N=500) : A/P y valait 0.65–1.4 vs 2.2–4.4 ici (N=2000) — la
   contraction par liaison croît avec N (observation, à creuser si utile).
Cohérence : les « Re z* » du scan E1 (bornes plancher, ex. −0.026 à λ=0.40) sont
bien des bornes SUPÉRIEURES de la vraie valeur (−0.158 mesurée ici) ✓.

---

## 7. Hypothèses formulées et leur sort
| Hyp. | Énoncé | Prédiction testée | Sort |
|---|---|---|---|
| H1 | SNIC-sur-mémoire à λ_c=0.2822 | cycle ensemencé survit jusqu'à λ_c, chute sur MÉMOIRE | **Réfutée** (E5 : chute sur front figé, mort à λ*>λ_c) |
| H2 | fold/terminaison à λ*>λ_c, fenêtre intermédiaire | mort au-dessus de λ_c, état tiers | **Confirmée & précisée** (E5/E6/E9) |
| H2′ | terminaison homocline (loi log) | pente log constante sur t_MAX | **Réfutée** (E9 : pente explose ; √ tient) |
| H3 | échec 0.32 = artefact de fenêtre | fenêtre longue → cycle vivant | **Réfutée** à 0.32/0.325 (E5/E6a) ; mais AVÉRÉE pour τ=5/λ=0.35 (E4) |
| Verdict final | **SNIC localisé au maillon faible (dépiégeage), λ*≈0.327** | loi √ sur t_MAX + arrêt sur (91,92) | **Établi** (E9+E10) |

---

## 8. Mécanisme des échelles de temps (fait discret de l'utilisateur)
**Fait (modèle discret)** : à τ élevé, phases statique/dynamique adjacentes ; à τ=0, phase chaotique intermédiaire ; en continu, pas de phase dynamique à τ=0.
**Mécanisme (paramètre τ/t_relax)** : à τ≫t_relax, le terme non réciproque λK·g(u(t−τ)) est un champ quasi-statique par époques → dynamique intra-époque gouvernée par J (gradient) → pas de chaos, drive cohérent (m≈1). À τ≲t_relax : frustration temporelle ancrage/poussée → chaos (phénoménologie Crisanti–Sompolinsky). Discret « τ=0 » : délai unité intrinsèque de l'update parallèle (τ_eff=1≈t_relax) → fenêtre frustrée. Continu τ=0 : aucun plancher → pas de phase dynamique.
**Raffinement (E8a)** : le mécanisme protège le secteur MONO-front (mémoire, front piégé, pacemaker — tous non chaotiques) mais PAS le secteur multi-patterns (image retardée jamais convergée) → chaos itinérant persistant à τ=10 dans la fenêtre. **Prédiction ouverte :** à τ~t₀, la fenêtre devient majoritairement chaotique (version continue du diagramme discret).

---

## 9. Synthèse : carte des phases (α=0.05, τ=10, N=2000, seed 42, déterministe) — **figD**
| Région | Attracteur(s) | Références |
|---|---|---|
| λ < 0.2822 | mémoire (+ front piégé en coexistence — E5) | λ_c : REPORT statique ; bistabilité : E5 |
| 0.2822 – 0.327 | **phase mixte** : mer chaotique (λ_L≈+0.034) + fronts piégés (points fixes exacts) + mémoires résiduelles (folds dispersés ≤0.3025) | E6b/E7/E8 |
| λ > 0.327 | cycle de rappel : escape-limited → (crossover t_échap≈τ) → pacemaker strict T₁≈τ+0.83 | E1/E4 |
Points critiques : **λ_c=0.2822** (fold ξ¹) ; **λ*≈0.327** (dépiégeage SNIC @ liaison 91 ; seuil n°2 : 0.322). Figure : `figures/figD_phase_map.png`.

---

## 10. Figures (ventilées par phase dans `results/<phase>/figures/` — voir `results/README.md`)
| Fichier | Contenu | Source |
|---|---|---|
| figA_pacemaker_law_collapse.png | T₁(λ,τ) brut + collapse t_échap(λ) | E1+E4 |
| figB_depinning_transition.png | T₁(λ) complet (λ_c, λ*, fenêtre ombrée) + t_MAX vs loi √ + inset 1/t² → λ*=0.3262 | E1+E8c+E9 |
| figC_intermediate_phase.png | P(q) (E7 npz) + plateau ‖ȧ‖/λ_L (E8a) + histogramme t_μ liaison 91 (E9 npz) | E7/E8/E9 |
| figD_phase_map.png | carte des phases 1-D annotée | synthèse |
| figE_floquet_spectrum.png | multiplicateurs dans le plan complexe (N=500, 3 λ) — v1, visuellement dégénérée | E11 |
| **figE_floquet_spectrum_v2.png** | \|μ_k\| en LOG (effondrement 8.8e−8→≤3e−16, rien ne monte vers 1) + bande des exposants Re z_k — version de référence | E11/E13 |
| **figF_pinned_fold_spectral.png** | eig_max(M) du front piégé → 0 en −c√(λ*−λ) (c≈2.1) ; λ* dynamique et intercept E9 superposés | E12 |
| **figG_pinned_hopf_margin.png** | érosion de la marge de Hopf : front piégé (pur 1/τ, marge×τ plateau) vs mémoire (plus vite que 1/τ) ; contrôle mode réel | E15 |
| **figH_hopf_birth_v2.png** | sondage de la fenêtre Hopf (α=0.07, τ=100) : contrôle décroissant en escalier T≈τ + saut de bassin vers le jumeau a\*\* (plateau=1.013e−2) — v1 supersédée | E17 |
| floquet_scan_N2000_tau10.0.png | Re z* (borne plancher), |μ*| (plancher), T₁(λ) | E1 |
| waveform_lam0.9_tau10.0_N2000.png | forme d'onde a_ν(t) + T₁ par pas (v1, fenêtre mal placée — supersédée par figI) | §5 |
| **figI_traveling_wave.png** | le cycle comme onde progressive : heatmap a_ν(t) sur une période (raie diagonale) + zoom du passage de relais (T₁=10.83) — rendu depuis cycle_lam0.9 npz | §5 |
| **figD_phase_map_v2.png** | carte des phases v2 : barre pleine échelle + panneau zoom connecté [0.26,0.36] (λ_c, dispersion 0.3025, seuil 0.322, λ*, bandes de coexistence) — remplace figD dans la présentation | synthèse |
| **figJ_front_branch.png** | naissance du front piégé : branche a₉₁/a₉₂ sur [0.132,0.328] (fold bas + λ_c + λ*) + eig_max(M) instantané et racine T_P retardée, tous deux <0 partout | E18a/b |
| **figK_front_basins.png** | taille du bassin du front piégé vs λ : rayon d'échappement directionnel + fraction d'IC aléatoires capturées → rétrécissement monotone vers λ* (stabilité intacte) | E18d |
| **figL_smalltau_cycle.png** | cycle à petit τ : horloge T₁(τ) vs loi pacemaker (pente 0.56≠1) + désordre des relais + largeur du front + kymographes τ=0/0.5/2 (front large→net) | E19a |
| **figM_smalltau_snic.png** | loi de ralentissement critique t_MAX(λ−λ*) à petit τ (log-log, ajustement √, points censurés en ×) | E19c |
| **figP_desync.png** | désynchronisation à petit τ : T₁(τ) & deux lois, pente d'horloge, paramètre d'ordre R(τ) avec τ_c≈1.3, largeur/participation, rappel/lobes, kymographe τ=0 | E22 |
| **figN_basin_competition.png** | compétition des bassins vs λ (empilé mémoire/front/cycle/chaos/autre) : mémoire→chaos→cycle, front en filet | E20 |
| **figO_metastability.png** | métastabilité du front : bassins mémoire/cycle/front + indice front/bassin-dominant (log) ≲0.1 partout | E20 |
| **figQ_chaos_nature.png** | nature du chaos (λ=0.31, τ=10) : spectre de Lyapunov (D_KY≈6.4), convergence λ_L→+0.048, λ_L(λ), projection PCA de l'attracteur, return-map, spectre de puissance large-bande | E23a |
| **figR_cycle_chaos_frontier.png** | frontière cycle-chaos dans le plan (τ,λ) : régions cycle/chaos/front, λ*_front τ-indép, saut de λ_L à la mort | E23b |
| **figS_front_birth.png** | (rev. 2) naissance du front : branche complète [0, λ*] (membres E21a2+E18a fusionnés), superposition front≡mémoire ξ⁹¹ (Newton indépendant) sur tout le domaine, spectre de M sans croisement de 0 | E21a/a2/e |
| **figT_front_stability_vs_tau.png** | stabilité retardée du front vs λ pour τ∈{5,10,20,50} : Re z<0 partout (delay-robuste, pas de Hopf), marge ~1/τ | E21c |
| **figU_branch_geometry.png** | géométrie des 100 branches mémoire→front : inclinaison lisse (pas de genou, x₅₀=0.64λ_c), collapse universel en λ/λ_c, régime √ terminal | E24a |
| **figV_threshold_distribution.png** | distribution des seuils λ_c(μ) : histogrammes vs N, CDFs standardisées ≈ gaussienne (skew −0.5), σ(N) deux séries, gap observé vs prédiction EVT σ·a_P | E24b-d |
| **figW_deathshape.png** | forme au fold vs μ : r_c(λ_c) (corr +0.81, étoile=maillon extrême 68/32), q_μ vs λ_c (−0.48), plan (a_μ,a_{μ+1}), indépendance lag-1 multi-seed | E25 |
| **figZ_alpha_law.png** | loi des seuils vs α à N=10⁴ : centre suit λ_c(α), σ quasi-constant, collapse KS-compatible des CDFs standardisées → famille position-échelle universelle | E26 |
| **figX_markov_video.png** | motifs corrélés Markov (vidéo) : distributions bulk vs c, couture=dernière digue, famille r_c(λ_c), seuil vs q_μ | E27 |
| **figY_gpcircle_video.png** | motifs corrélés GP-cercle (vidéo périodique sans couture) : distributions vs K, stats vs corrélation q₁, formes au fold | E28 |
| figW/figX/figY `_N10000.png` | versions N=10⁴ des vérifications de taille finie (E25/E27/E28) — mêmes panneaux, verdicts de robustesse dans les entrées | E25/27/28 |
| **figAA_cycle_vs_ghosts.png** | le cycle juste au-dessus de λ* : projections (a_μ,a_{μ+1}) à travers les folds, pics analogiques quasi purs (≥0.978), readout binaire exact, goulot de vitesse au fantôme de la liaison 91 | E29 |
| **figAB_alpha_lambda_basins.png** | diagramme de phase des bassins (α,λ) : heatmaps statique/cycle/chaos, érosion du cycle vers la capacité (GPU float32 validé) | E32 |
| **figAC_aging_twotime.png** | test de vieillissement à α=0.05 : relaxation quench (τ_r≈2), collapse deux-temps en lag (stationnaire), non-collapse en t/t_w | E33 |
| **figAD_aging_vs_alpha.png** | écart-collapse deux-temps vs α : pas d'aging jusqu'à la capacité (n chaotique annoté) + collapse de référence α=0.05 | E33 |
| **figAE_threshold_scaling.png** | scaling grande échelle des seuils : σ~N^{−0.51} (TCL) et gap~N^{−0.42} à α fixe jusqu'à N=14000, histogramme standardisé vs normale | E30 |
| **figE34_ET0_edge_tracking.png** (`2_…/figures/E34/`) | gate ET0 : convergence de bissection d_G, racine instable certifiée, continuation des selles (μ=8 multifold 7 tours vs μ=14 simple), carte des ω-limites W^u sur l'anneau | E34-ET0 |
| **E35_D0_readouts.png** (`8_mhn_video/figures/`) | diagnostics de readout D0 : PSNR par frame par décodeur vs B1/B2, MSE vs PSNR, amplification ‖G†m‖/‖m‖, ΔPSNR par transition (bootstrap) | E35-D0 |
| **E35_V1_design.png**, **E35_V1_eval.png** (`8_mhn_video/figures/`) | V1 conception : grille du générateur vs critères gelés (2/9), factoriel J×K, calage TSVD r\*=54, scan β′ — et évaluation : PSNR par seed vs B0/B1/B2 (bande « dyn below B0 hold »), ΔPSNR appariés 16 combinaisons, différence par transition, validité du cycle 10/10 | E35-V1 |
| **E36_delay_scaling_multiseed.png**, **E36_b_lambda_dense.png** (`7_tau_implicite/figures/`) | réplication seeds 42–46 : T₁ vs K−1, b(λ) barres d'erreur inter-seed, brackets λ_c ; grille dense + comparaison AICc → loi puissance, plateau δ̄∞≈0.75 | E36 |
| **figE34_ET1pp.png** (`2_…/figures/E34/`) | ET1″ à N=2000 : grille des ω-limites de W^u(partenaire du fold) sur 9 cellules, selle émergeant du fold (d_G et z_u vs δ), escalier de la perle meneuse de l'orbite voyageuse (μ=91), audit de la table des seuils (33/100) | E34-ET1″ |

## 11. Codes (numerics/src/)
| Fichier | Rôle |
|---|---|
| `cycle_reduced.py` | réduction P-dim exacte (ReducedDDE : m, Dm, rhs, integrate RK4+Hermite) + validations V0/V1/V4 en __main__ |
| `pacemaker_cycle.py` | build_cycle : simulation, détection de période, extraction [−τ,T] |
| `floquet_monodromy.py` | Monodromy : cache de gains float32 (demi-grille), matvec variational, Arnoldi, phase_mode |
| `pacemaker_scan.py` | scan λ complet (cycle+Floquet par λ, figure, npz) |
| `cycle_figures.py` | figures A–D (données de provenance en en-tête) |
| `e14_contraction_near_death.py` | contraction par tour A=−ln\|μ₁\| sans plancher (Benettin/QR 2 vecteurs) |
| `e15_pinned_hopf_margin.py` | spectre retardé du front piégé vs τ (états construits une fois, scan τ dans T_P) |
| `e17_hopf_birth.py`, `e17b_identify_final_state.py`, `e17c_fig_refit.py` | sondage non linéaire de la fenêtre Hopf + identification du jumeau + figH v2 |
| `e18a_front_branch.py`, `e18b_delayed_stability.py`, `e18c_coexistence.py`, `e18d_basins.py`, `e18_figures.py` | naissance du front piégé : continuation de branche, stabilité retardée T_P, coexistence mémoire↔front, bassins, figJ/figK |
| `e19a_cycle_clock.py`, `e19b_death.py`, `e19c_slowing.py`, `e19d_fixedpoint.py` | régime petit τ : horloge/kymographes, point de mort, loi √ du ralentissement, croisement stabilité du front (figL/figM) |
| `e22_desync.py` (pilot/sweep/lamcheck) | désynchronisation petit τ : grille τ dense, largeur/participation, paramètre d'ordre R(τ), τ_c, contrôle λ (figP) |
| `e20_common.py`, `e20a_basin_fractions.py`, `e20a_hi.py`, `e20_figures.py` | compétition des bassins : échantillonnage IC, classifieur (front=liaison faible), balayage λ, figN/figO |
| `e21a_front_birth.py`, `e21a2_dissolution.py`, `e21b_tau_independence.py`, `e21c_tau_nature.py`, `e21_figures.py` | naissance du front : arclength (pas de fold), front≡mémoire inclinée, τ-indépendance z=0, nature/Hopf vs τ (figS/figT) |
| `e21d_memory_fold_check.py` | re-vérification du fold mémoire ξ¹ : arclength à travers le fold (turned=True à 0.2835), contre-vérification N×N dense vs Sylvester (2e-16), micro-structure jumelle |
| `e24_thresholds.py`, `e24a_geometry.py`, `e24_analysis.py`, `e24e_spotchecks.py` | seuils par motif λ_c(μ) : traceur tous-motifs (bisection+extrapolation spectrale, options --subset/--no-eig), géométrie/genou (figU), scalings & EVT (figV), spot-checks T_P |
| `e25_deathshape.py` | forme au fold vs μ : r_c(λ_c), prédicteur q_μ, indépendance lag-1 multi-seed (figW) |
| `e26_analysis.py` | loi des seuils vs α à N=10⁴ : collapse standardisé, tests KS, breakdown α≥0.08 (figZ) |
| `e27_markov_thresholds.py`, `e28_gpcircle_thresholds.py` | motifs corrélés « vidéo » : chaîne de Markov de flips (couture traitée) et GP latent périodique sur le cercle ; seuils + formes + contrôle dynamique (figX/figY) |
| `e23a_chaos_nature.py`, `e23b_frontier.py` | nature du chaos (Lyapunov/D_KY/Poincaré, figQ) + frontière cycle-chaos (τ,λ) et paradoxe SNIC (figR) |
| `e21e_figS_full.py` | figS rev. 2 : fusion des deux membres de la branche front + marques mémoire ξ⁹¹ indépendantes jusqu'à λ* (E21_mem_marks.npz) |
| `e29_cycle_vs_ghosts.py`, `e29_fig.py` | cycle vs fantômes juste au-dessus de λ* : descente warm-start, détection de tour, pics/readout binaire/point lent vs fold, période, vitesse (figAA) ; régénération figure + approche minimale aux folds |
| `e30_scaling_overnight.py`, `e30_analysis.py` | campagne d'échelle des seuils (303 runs resumables, séries P-fixe/α-fixe/reach) ; fits σ(N)/gap(N), gaussianité (figAE) |
| `e31_fixedpoint_census.py` | survie des mémoires f_mem(α,λ) par Newton recuit depuis chaque motif (tracer-indépendant) |
| `cycle_reduced_batch.py`, `cycle_reduced_mlx.py` | intégrateurs réduits batchés : float64 CPU (K IC en colonnes, validé 1e-13) et float32 GPU/MLX (validé temps-court 8e-5 + porte fractions-de-bassins) |
| `e32_alpha_lambda_basins.py`, `e32_analysis.py` | diagramme de bassins (α,λ) sur GPU avec porte de validation f32-vs-f64 ; heatmaps + pinch-off (figAB) |
| `e33_aging.py`, `e33_analysis.py`, `e33_vs_alpha.py` | test de vieillissement : C(t_w+t,t_w) d'ensemble, quench τ_r, extension vs α (figAC/figAD) |
| `e34_lib.py`, `e34a_census.py`, `e34_threshold_table.py`, `e34b_edge_tracking.py`, `e34_ET0_figures.py` | E34 : briques certifiées (Gram, principe de l'argument, arclength, historiques exponentiels), census/table pilote N=400, gate ET0 edge-tracking + figure (import du dossier jumeau 2026-07-24 + ET0 nouveau) |
| `mhn_reduced.py`, `e35_dynamics.py`, `e35_video_tools.py`, `e35_runtime.py`, `e35_analysis.py`, `e35b/c_*.py`, `e35d0_readout_diagnostics.py`, `e35d0_make_figure.py` | E35 : pipeline MHN + POC vidéo (jumeau), diagnostics de readout D0 (nouveau) |
| `layered_chain.py`, `e36a_regimes.py`, `e36a2_delay.py`, `e36bench.py`, `e36_postprocess.py`, `job_monitor.py` | E36 : chaîne en couches (délai implicite), campagnes régimes/délai, bench mesuré, postprocess (jumeau ; multi-seed 2026-07-24) |
| `cycle_reduced.py` (rév. hist+dhist) | intégrateur réduit corrigé : `integrate(..., dhist0)` ré-injectable (fix E34-R2.3, importé du jumeau ; ancien fichier sauvegardé `cycle_reduced_pre_dhist_backup.py`) |
| `e35v1_common.py`, `e35v1_bench.py`, `e35v1_design.py`, `e35v1_eval.py`, `e35v1_figures.py` | E35-V1 (2026-07-27) : générateur d'advection non linéaire + 4 décodeurs + 2 horloges + bootstrap hiérarchique ; conception en 4 étapes (headroom/screen/select/freeze) ; évaluation **refusant de démarrer sur hash de config divergent** |
| `e34_necklace_wu_n2000.py`, `e34_et1pp_followup.py`, `e34_threshold_table_audit.py`, `e34_certif_tune.py`, `e34_et1_boundary_diag.py`, `e34_dwell_profile.py`, `e34_dwell_seed_test.py`, `e34_ET1pp_figures.py` | E34-ET1″ (2026-07-27) : pilote W^u du partenaire du fold, 5 modes de levée (arc/land/loc/cycle/recell), audit de la table des seuils, escalade de contour mesurée, les 3 diagnostics qui invalident l'edge-tracking à N=2000, figure 4 panneaux (remplace `e34_ET1prime_figures.py`, design caduc) |
| `health_probe.sh` | sonde de santé de campagne (CPU/GPU/avancement/détection de stall) → HEALTH_LOG.txt |
| `build_report_html.py` | rendu markdown→HTML MathJax (`--md/--out/--no-figures`) |
| (statique) `robust_branch.py`, `reduced_spectrum.py`, `audit_static_analysis.py` | Woodbury/Sylvester/T_P, audits 15/15 |

## 12. Points ouverts
1. **Certification N=10⁴** (GPU MLX float32 pour le variational — dynamique tolérante) : λ*(N) via extrêmes (prédit croissant), exposants vrais inaccessibles au plancher → mesurer par itération déflatée ou Lyapunov.
2. Fraction de gel des bassins dans la fenêtre (IC aléatoires, t≫10³).
3. ~~Prédiction petit-τ : fenêtre chaotique remplaçant la gelée (τ=1–2).~~ **RÉSOLU par E19** : à petit τ le front se délocalise (multi-lobe), la statistique d'extrêmes s'effondre, le cycle meurt par collision avec le chaos (pas de SNIC propre, pas de loi √). Reste à cartographier finement la frontière cycle→chaos en (τ,λ) pour τ∈[0.25,5].
4. V3 Benettin (λ_L propre), V5 (limite λ=1, front analytique).
5. Multi-seed : distribution de λ* (extrêmes sur P seuils) et de λ_c.
6. **Table des seuils par perle (E24) vs fold des branches — ouvert depuis ET1″ (2026-07-27).** Au moins 33/100 perles (N=2000, P=100, seed 42) portent encore un équilibre STABLE d'identité μ au-dessus de leur λ_c tabulé ; l'écart atteint +1.27e−2 pour μ=28, où l'arclength monte sans retournement jusqu'à 0.2321. Deux lectures non tranchées — table en avance vs branche en S (μ=8, cas multifold d'ET0′, en fait partie). Test propre : fold mesuré par arclength perle par perle (≈4 h). Enjeu : le recensement `solve_alive_nodes` (tout E34) et la loi λ_c(μ) (E24/E27) reposent sur cette table.
7. **Identification de l'objet de bord à N=2000 (ET1″ §1).** L'edge-tracking nœud–nœud ne fournit pas d'amorce Newton convergente au budget employé ; la selle dont W^u relie les deux nœuds n'est donc pas identifiée à l'objet séparant les bassins. Test : budget d'intégration augmenté + bissection resserrée (≈1 h).

## 13. Journal des décisions
- **2026-07-27 — E34-ET1″ : re-cadrage du test sur MESURE, et audit incident de la table des seuils.** ET1′ (edge-tracking → Newton) a été abandonné **après** trois mesures, pas par intuition : frontière propre (13/13), amorce « point le plus lent » captant l'installation sur le nœud (4.7e−13), Newton simple ET déflaté stagnant à ~1e−3 depuis 6 amorces du plateau. La déflation — correctif envisagé au départ — était donc sans objet (elle divergeait). Décision : tester le collier **directement** par W^u de la selle partenaire du fold, obtenue par arclength à 1e−15. Résultat : maillon **positif 4/4** là où une perle suivante existe, **fermeture sur le cycle 3/3** à la dernière perle. Deux obstacles levés au passage, tous deux mesurés : la certification par principe de l'argument exige **(32,9)** à P=100 (contre (16,7) au pilote P=20) ; le successeur est souvent déjà mort, la cible est la première perle **vivante** en avant. **Incident dépassant E34** : les deux verdicts négatifs initiaux venaient du recensement — l'objet « hors recensement » à μ=1 est la perle 3 (m₃=0.9862, stable, res 4e−11) à λ au-dessus de son seuil tabulé ; l'audit systématique donne **33/100 perles portant encore un équilibre stable au-dessus de leur λ_c tabulé** (borne inférieure : la sonde Newton-au-motif échoue précisément là où le flot converge). Non tranché : table en avance vs branche en S. Le recensement et la loi λ_c(μ) d'E24/E27 reposent sur cette table → point ouvert §12.6. Coût machine total : 26 min de grille + 15 min de diagnostics, 3 cœurs maximum en simultané avec E35-V1.
- **2026-07-27 — E35-V1 exécuté après validation utilisateur du protocole (2026-07-25) : NO-GO interpolation, cycle parfait.** Discipline de gel tenue de bout en bout : décisions sur `design_seeds={1…6}`, hash de contenu `35a3bc12f9c992ef` écrit AVANT ouverture des `eval_seeds={101…110}`, script d'évaluation refusant de démarrer sur hash divergent, §1–§8 du protocole intacts (A.9–A.12 seules ajoutées). Critères jamais affaiblis : 2 réglages sur 9 les satisfont, la parcimonie tranche (headroom +7.95 dB vs +1.90 dB pour V0 — le POC avait cette fois une marge réelle, et B2 bat B1 de +7.73 dB en évaluation, donc la marge existait bien). **Verdict négatif, non marginal** : meilleur readout dynamique 3,83 dB SOUS B0 « recopier la frame-clé », 0/10 seeds et 0/1000 blocs favorables sur les 16 combinaisons décodeur × horloge. Séparation ferme des deux constats : le **mécanisme de cycle est parfait** (couverture et sens 1.000 sur 10/10, `period_cv ≤ 2.6e−5`) — c'est la **reconstruction** qui échoue. NO-GO V0 conservé indépendamment ; escalade Moving MNIST / vraie vidéo écartée par la règle gelée. Coût 56,4 min, aucune réduction A.8 nécessaire. Sous-agent Opus : chaîne d'exécution puis rédaction, tous les chiffres re-vérifiés contre les npz (36 contrôles, 0 écart) ; 7 défauts de figures corrigés, dont un substantiel (B3 tracé en ligne continue alors qu'il est NaN sur 7 seeds sur 10 — suggérait visuellement des valeurs là où aucune frame n'est décodable).
- **2026-07-24 — exécution des trois gates des roadmaps (E34/E35/E36, rév. 3)** : import du dossier jumeau (« théorie claude + implémentation GPT ») dans CE workspace — modules certifiés (cycle_reduced hist+dhist, e34_lib, pipeline E35, layered_chain) + artefacts de campagne ; validation avant usage : pytest 81/81, cycle_reduced 5/5 (V1=5.5e−16). Trois sous-agents Opus en parallèle (E34 ≤3 procs, E36 ≤4, E35 1 ; surveillance horaire CPU/RAM). Verdicts : **E34-ET0 NO-GO** (règle gelée 3/3) mais topologie dynamique à λ fixé PROPRE sur 2/2 sentinelles testables — la dissociation « géométrie de branche multifold vs flot à λ fixé » est le fait nouveau ; μ=2 intestable (seul nœud vivant à λ_local) → amendement ET0′ proposé, non exécuté ; ET1 reste verrouillé. **E35-D0 POSITIF interne** (TSVD r*=7 : +3.3 dB target-blind) avec NO-GO V0 conservé (−11.5 dB sous B1) ; brouillon V1 verrouillé prêt. **E36 : loi b(λ)=0.752+0.909·λ^{−1.14}** promue par AICc sur 5 seeds (plateau δ̄∞≈0.75·t₀ ⇒ conjecture forte K−1 réfutée, faible confirmée) ; λ_c(K=6) resserré à (0.64, 0.70] ; bande « irrégulière » découverte. Coûts réels ≪ estimations FLOP (bench mesuré ~30×) ; total machine < 30 min — aucune nuit nécessaire.
- **2026-07-08/09 — campagne nocturne autonome (5 tâches utilisateur)** : surveillance 30 min (CPU/GPU/stall, `health_probe.sh`→HEALTH_LOG.txt), CPU et GPU exploités EN PARALLÈLE. **E30** (loi d'échelle, tâche 1) : 303 runs, TCL confirmé σ_α~N^{−0.51}, gap→0 jusqu'à N=14000, quasi-gaussien (figAE, REPORT statique rev. 6 §5.9) ; 2 réinterventions sur le ladder (eig→no-eig, abandon N=20000) car coût full-P ~N^{3.4} sous-estimé — plafond nocturne honnête N≈14000. **E31** (survie mémoires, tâche 5 Exp C) : f_mem chute avec α ET λ, breakdown de capacité RÉEL (tracer-indépendant). **E32** (bassins (α,λ), tâche 5 Exp A) : cycle érodé vers la capacité (figAB) — **première utilisation GPU validée** (MLX float32, porte fractions f32-vs-f64, la leçon float64 est LOCALISÉE au Newton/folds ; correction d'une sur-généralisation antérieure). **E33** (vieillissement, tâche 3) : PAS d'aging, attracteur stationnaire à mesure SRB (C deux-temps invariante, τ_r≈2), robuste jusqu'à α=0.10 (figAC/figAD). Aussi : revue littérature `LITERATURE_correlated_patterns.md` (tâche 4, web) ; tâche 2 (E25/spins partagés) répondue par contrôle numérique (cross-talk ∝ q dans l'amplitude réduite, +0.95 = corr partielle/suppression). Restant : Exp B Lyapunov (le cycle grand-α est-il chaotique ?), validation float64 de l'aging. Journal détaillé : `results/OVERNIGHT_PLAN.md`.
- **2026-07-01** : plan v2 ; découverte de la réduction P-dim exacte ; protocole hystérésis ; V0 défini premier test ; GPU réservé au variational.
- **2026-07-02 (matin)** : implémentation + validations 6/6 ; scan E1 (hyper-stabilité, artefact plancher identifié §4.3) ; E4 collapse exact.
- **2026-07-02 (après-midi)** : E5 réfute H1 ; E6 λ* encadré, piste homocline ; E7 P(q) large ; E8 chaos itinérant + front piégé point fixe ; **E9/E10 verdict SNIC-au-maillon-faible (liaison 91), λ*≈0.327** ; figures A–D ; règles utilisateur (worklog continu, résultats dans le chat, graphes systématiques) ; réécriture auto-suffisante du worklog + E11 lancée (spectres complexes).
- **2026-07-02 (soir)** : questions utilisateur (E12 protocole, statut liaison 91, figE dégénérée) → E12/E13 conclues, figE_v2 ; E14 (Benettin floor-free) et E15 (marges du front piégé vs τ) conçues et exécutées ; Annexes A–C rédigées (pédagogie Floquet/SNIC, convergences, T_P) ; E17 lancée (fenêtre Hopf α=0.07, τ=100).
- **2026-07-06 (soir) — campagne de vérification N=10⁴** (demande utilisateur, méfiance envers N=2000) : E25/E27/E28 relancés à N=10⁴, α=0.05 (P=500), sous-échantillons de 100–150 branches, options --subset/--no-eig, K∝P pour E28. **Tous les verdicts qualitatifs sont robustes** : famille r_c(λ_c) (+0.75), prédicteur q (−0.49, partielle +0.90), iid des seuils TRANCHÉ (lag-1=−0.009), effondrement bulk + couture=digue max (Markov c=0.2), fonte à forte corrélation, cycle GP parfait à tous K et **aucun épinglage jusqu'à λ=0.20**. Deux raffinements de taille finie : frontière de destruction du cycle Markov déplacée de c∈(0.2,0.4) à c∈(0.4,0.6) ; corr(q,λ_c) renforcée (−0.55). Incidents : K_dyn codé en dur (crash E28, corrigé + checkpoint) ; diagnostic « faux blocage » = bufferisation conda run + phase dynamique dim-500 dans le process principal ; watchdog 30 min mis en place.
- **2026-07-08 — figS rev. 2 + deck 2** : question utilisateur sur la borne λ=0.15 de figS → simple graine d'arclength, pas de limite physique ; figure refaite sur [0, λ*] avec preuve front≡mémoire étendue à tout le domaine (`e21e_figS_full.py`). Deck beamer `presentation_2.md` (25 slides, E18–E29) créé la veille dans `Beamer presentations/`, figures en `figures/`.
- **2026-07-07 — E29 (question utilisateur : par quoi passe le cycle ?)** : descente warm-start λ=0.90→0.330 (10 valeurs, ~12 min). Verdict double : le cycle passe par les motifs quasi PURS (pic 0.992–1.000, sign(u)=ξ^μ exactement pour les 100 motifs à tous les λ) ET enfile tous les états de fold en transit (approche ≤0.01) — mais ne RAMPE que sur le fantôme de la liaison critique 91 (coïncidence à 0.006, séjour ×2.3). Résolution du paradoxe : le tilt 70/30 du point fixe vient de sa propre image retardée ; sur le cycle, l'image retardée au pic est encore μ−1, le drive retardé pousse vers μ lui-même → purification. Le rappel séquentiel est exact PARCE QUE le retard réaligne le drive. figAA.
- **2026-07-06** : questions utilisateur 1–4 sur λ_c(μ). **E25** (forme au fold) : pas universelle — r_c(λ_c) famille à un paramètre (corr +0.81), 68/32 = maillon extrême ; prédicteur q_μ (corr partielle +0.95 à λ_c fixé) ; λ_c(μ) quasi-iid multi-seed. **E26** (loi vs α, N=10⁴) : centre suit λ_c(α), σ quasi-constant, collapse KS des formes standardisées → λ_c(μ)≈m(α)+σ(N)·X universelle ; breakdown α≥0.08 (perte d'identité de branche). **E27** (vidéo Markov) : à c=0.2 seuils bulk effondrés (0.27→0.08) et couture=digue max (prédiction E25 ✓) ; c≥0.4 : mémoires→paquets délocalisés, cycle détruit. **E28** (vidéo GP-cercle, sans couture) : mémoires fondent pareillement MAIS cycle parfait à tous K et **aucun épinglage jusqu'à λ=0.20** à K=10 — quasi-attracteur continu, la digue SNIC disparaît. Incidents : deadlock multiprocessing 11 h (garde __main__ manquante, corrigée partout) ; critère d'identité de branche assoupli pour motifs corrélés (mode relaxed).
- **2026-07-05 (E24)** : statistique des seuils par motif. Traceur tous-motifs validé (motif 1→0.2821=λ_c ✓, motif 91→0.3276=λ*=argmax ✓). (a) passage motif→mélange **LISSE** (x₅₀=0.64λ_c, collapse universel, √ terminal seulement) → fin du rappel complet = min_μ=0.2195 (motif 28, ≪0.2822) ; (b) gap=max−min **décroît avec N** (0.159→0.079 de N=1000 à 8000, ~N^{−0.22}) → effet de taille finie, transitions fusionnant à N→∞ ; (c) distribution quasi-gaussienne (skew −0.5), **gap prédit par les valeurs extrêmes σ(N)·a_P à 0.002–0.008 près à tous les N** ; σ(N) : −0.61 à P fixé (∼CLT), −0.38 à α fixé. E24e : 0/30 sondes T_P instables (existence⇒stabilité DDE partout). figU/figV. REPORT statique → rev. 5 (§5.8).
- **2026-07-05 (suite)** : **E21d** (re-vérification honnête du fold mémoire, question utilisateur) : selle-nœud ξ¹ CONFIRMÉ par arclength (turned à 0.2835, valeur propre réelle traversant 0), pipeline spectral validé contre diagonalisation N×N dense (accord 2e-16), micro-structure de folds jumeaux ±1.5e-3 (écho E17). **Réorganisation documentaire** : REPORT_static_bifurcation passe en rev. 4 (§5.7 unification mémoire↔front) ; création de `results/REPORT_faible_tau.md` (E19+E22+E23B consolidés) et `results/REPORT_attracteurs.md` (E18d+E20+E23A consolidés) — points d'entrée des approfondissements annoncés.
- **2026-07-05** : 4 nouvelles directions lancées en sous-agents Opus (tâches utilisateur 2–6). **E22** (désynchronisation petit τ) terminé : transition abrupte multi-lobe→mono-lobe à τ_c≈1.3≈t₀ (N_part 18→1, W 14→1.3, rappel 0.2→0.9, éc.(T₁) ±1.0→±0.01) ; loi T₁=0.52τ+1.73 (petit τ) → τ+0.815 (grand τ) ; λ ne resynchronise pas ; figP. Agent bloqué sur permission Bash → calcul + rédaction faits par l'orchestrateur. **E20** (bassins) : front métastable (bassin ≤5 %, jamais gagnant), mémoire→chaos→cycle, chaos résiduel 16–21 % à haut λ ; figN/figO. **E21** (naissance) : pas de bifurcation — front≡mémoire inclinée (arclength sans fold, front≡ξ⁹¹ à 5e-4), position τ-indép exacte, aucun Hopf du front jusqu'à τ=50 (delay-robuste) ; figS/figT. **E23** : chaos = attracteur soutenu D_KY≈6.4 (hyperchaos ~3 exposants+) figQ ; frontière cycle-chaos — à petit τ le « cycle » est déjà chaotique (λ_L>0), le selle-nœud du front reste τ-indép mais le cycle se fond dans le chaos au lieu de s'y poser (résolution du paradoxe SNIC) ; figR. Tous les agents Opus de cette session bloqués sur permission Bash → scripts écrits par eux, exécutés/documentés par l'orchestrateur.
- **2026-07-04/05** : deux investigations lancées en sous-agents (repris manuellement après coupure de session). **E18** (naissance des fronts piégés) : continuation de branche vers le bas → fold propre à λ_f=0.132, coexistence exacte mémoire↔front dans toute la phase statique, stabilité (instantanée + retardée) intacte partout, bassin rétrécissant monotoniquement vers λ* ; figJ/figK. **E19** (petit τ) : cycle subsistant jusqu'à τ=0 mais front délocalisé multi-lobe, loi pacemaker dégénérée (pente 0.56), mort par collision avec le chaos et **destruction de la loi √** (R²≈0.02–0.31 vs τ=10) ; stabilité du front confirmée τ-indépendante (E19d ≡ E12) ; figL/figM. Point ouvert §12.3 résolu.
- **2026-07-03** : E17 lu et réanalysé (E17b/E17c) : pas de croissance Hopf — fenêtre plus étroite que le placement ; découverte de la bistabilité de jumeaux mémoire près du fold (α=0.07) ; figH v2 ; mise à jour §0/§10/§11 ; documents rendus auto-suffisants pour la présentation ; deck beamer `presentation.md` écrit (39 slides) ; figE_v2/figF/figG re-rendues en anglais ; figI (onde progressive, depuis le npz du cycle) et figD v2 (carte des phases zoom connecté) générées pour la présentation (`figs_wave_phasemap_v2.py`).

---

## Annexe A — Explication pédagogique : Floquet sans prérequis, et la convergence des mesures vers le SNIC
*(rédigée pour un scientifique de bon niveau mathématique, non expert en bifurcations/DDE — support direct pour la présentation.)*

### A.1 De la stabilité d'un point à celle d'une orbite
Pour un **point fixe** x\*, on linéarise δẋ = J·δx (J constant) : stable ssi toutes les valeurs propres ont Re < 0. Pour une **orbite périodique** a_p(t) = a_p(t+T), la linéarisation δȧ = J(t)·δa a des **coefficients T-périodiques** : les valeurs propres de J(t) gelé à un instant ne signifient rien. La bonne question : *que devient une perturbation après un tour complet ?* On définit l'application linéaire **U (monodromie)** : perturbation à t=0 → perturbation à t=T (transportée par l'équation linéarisée). Après n tours : δa(nT) = Uⁿ δa(0). Le destin est réglé par les **valeurs propres de U : les multiplicateurs de Floquet μ_k** — facteur de multiplication du mode k *par tour*.
- |μ| < 1 : contraction par tour (stable) ; |μ| > 1 : instable ; **le cercle unité |μ|=1 remplace l'axe imaginaire** des points fixes. Lien aux taux : μ = e^{zT}, z = taux par unité de temps.

### A.2 Pourquoi μ₀ = 1, exactement et toujours
Le système est autonome : si a_p(t) est solution, a_p(t+ε) aussi — même orbite, phase décalée. Leur différence δa ≈ ε·ȧ_p(t) est une perturbation qui revient identique à elle-même après un tour (ȧ_p est T-périodique) : la tangente à l'orbite est vecteur propre de U avec **valeur propre exactement 1** (mode de Goldstone de l'invariance par translation du temps ; glisser le long du cycle ne coûte rien). Prédiction théorique SANS paramètre libre → **thermomètre de validation** : mesuré |μ₀−1| = 10⁻³–10⁻⁵ (= fermeture du cycle extrait) et Σlog = 0.001 (E14), vecteur propre aligné à ȧ_p à 1.0000.

### A.3 Pourquoi μ₁ ~ 0 ici
μ₁ = plus grand multiplicateur restant (modes qui déforment l'orbite). Stabilité orbitale ⟺ |μ₁| < 1. Ici le taux par unité de temps est ordinaire (z₁ ≈ −0.16 à −0.23, mesuré E14) mais **le tour est très long** (T ≈ 1100–1900) : μ₁ = e^{z₁T} ~ e^{−220} à e^{−440}. « ~0 » = hyper-stable ; « ~1 » = marginal.

### A.4 La spécificité DDE en deux points
(1) Avec délai, l'état est le **segment d'histoire** sur [−τ,0] (dimension infinie) → U est un opérateur ; ses multiplicateurs s'accumulent en 0 (compacité), seuls les plus grands comptent. (2) Réduction exacte (§3) : tout le spectre non trivial vit dans un problème P-dimensionnel ; le reste contracte exactement en e^{−T/t₀}.

### A.5 La convergence des mesures indépendantes vers le SNIC (le cœur de la démonstration)
**① Ce que la mort n'est PAS — spectre du cycle (E1, E11, E14).** Toute bifurcation LOCALE d'un cycle s'annonce par un multiplicateur montant vers le cercle unité : μ→+1 (fold de cycles), μ→−1 (doublement), paire complexe → |μ|=1 (tore/Neimark–Sacker). Mesuré sans plancher (E14) : μ₁ fait l'INVERSE — e^{−220} → e^{−440} en approchant λ*, z₁ fini (−0.23). Le cycle ne perd jamais sa stabilité : **il cesse d'exister** → scénario global obligatoire.
**② Comment la trajectoire meurt — cinématique (E9).** Les scénarios globaux se départagent par la loi de divergence du temps : homocline → log ; **SNIC → 1/√**. Mesuré sur la bonne observable (temps de passage de la liaison LA PLUS LENTE, pas la moyenne) : t_MAX = C/√(λ−λ*), C = 2.3±0.2 constant ; loi log rejetée (pente 11.6→37.1, explose). Divergence en 1/√ = **fantôme de selle-nœud** sur le chemin.
**③ Ce qui naît à la place — le point fixe (E10, E12).** Sous λ*, au lieu exact du goulot, existe un nouveau point fixe : le **front piégé** (résidu du champ 6×10⁻¹⁵ — zéro machine), suivi en λ par la dynamique elle-même (pas de saut de bassin, stabilité retardée prouvée par construction). Sa valeur propre dominante est **RÉELLE** et monte vers 0 en −c√(λ*−λ) (c≈2.1) ; sa branche se termine EXACTEMENT (±5×10⁻⁴) où le cycle renaît : existence complémentaire, pas de coexistence → le selle-nœud a lieu **sur l'orbite**.
**④ Synthèse géométrique = SNIC** (Saddle-Node on an Invariant Circle) : en descendant λ, une paire de points fixes (front piégé stable + front-barrière instable) naît sur le chemin du cycle, au maillon le plus faible du désordre. λ>λ* : passage possible mais ralenti dans le fantôme (②) ; λ<λ* : passage impossible, l'orbite est remplacée par les points fixes (③) ; et comme le cycle contracte toujours pendant un tour qui s'allonge, μ₁ = e^{−A} → 0 (①). **Les trois faces sont les trois prédictions du même objet.**
**⑤ Statut statistique (E13).** Mécanisme universel, lieu aléatoire : 4 seeds → 4 maillons faibles (91, 64, 84, 0), λ* ∈ [0.315, 0.328] (extrêmes sur P seuils), même loi et forme de front quasi-universelle [0.68, 0.32].
**Phrase de synthèse :** le cycle meurt d'un selle-nœud de points fixes né sur son propre chemin, au maillon le plus faible du désordre gelé (dépiégeage) — établi par l'exclusion spectrale de tout scénario local (①), la loi du fantôme en 1/√ (②), l'observation directe du point fixe naissant et de sa valeur propre réelle s'annulant au bon endroit (③), et la reproduction multi-échantillons (⑤).

## Annexe B — Rôle de chaque expérience et les AUTRES convergences

### B.1 Rôle de chaque expérience (hors chaîne SNIC principale)
| Exp. | Rôle | Apport |
|---|---|---|
| E1 | cartographier la stabilité sur [0.35,0.95] | hyper-stabilité partout ; réfute λ_c⁽²⁾≈0.59 (legacy) ; courbe T₁(λ) |
| E2 | représentativité en N | T₁ convergé à 0.1% ; exposants vrais à N=500 |
| E3 | 1er test SNIC-sur-mémoire | réfute la loi √ vers λ_c ; découvre l'échec de formation sous 0.32 (→ fenêtre intermédiaire) |
| E4 | question utilisateur « régime pacemaker » | loi EXACTE T₁ = τ + t_échap(λ) ; requalifie les échecs de fenêtre |
| E5 | discriminant H1/H2 (hystérésis) | chute sur front figé ; bistabilité sous λ_c |
| E6a | encadrer λ* | (0.325, 0.33) ; fit log provisoire (corrigé par E9) |
| E6b | continuation statique du front (échec Newton) | ÉCHEC INSTRUCTIF → dispersion des folds par pattern (≤0.3025) |
| E7 | question spin-glass | P(q) large, états 6–12 composantes, pas de gel à t=10³ |
| E8a | vieillissement vs chaos | itinérance chaotique, λ_L=+0.034 |
| E8b | nature du front piégé | point fixe exact (‖ȧ‖=0) |
| E8c | λ* fin + biais | identifie le biais de moyenne (∝1/P) → motive E9 |
| E13 | universalité vs échantillon | sites et λ* aléatoires, mécanisme et forme universels |

### B.2 Convergence 2 — LA LOI DU PACEMAKER (E1 + E2 + E4 + mécanisme §8)
Quatre mesures indépendantes convergent vers : **le délai est une horloge additive pure**. E1 : T₁(λ) à τ=10 ; E4 : superposition exacte (~4 chiffres) de T₁−τ pour τ=5/10/20 ; E2 : T₁ indépendant de N à 0.1% ; §8 : le mécanisme d'époques (le terme retardé = champ quasi-statique pendant τ) EXPLIQUE pourquoi la décomposition est exacte. Conséquence : « pacemaker strict » vs « rappel lent » = deux régimes du même attracteur (t_échap ≶ τ), sans bifurcation entre eux (E1).

### B.3 Convergence 3 — LA FENÊTRE INTERMÉDIAIRE MIXTE (E3 + E5 + E7 + E8a + E8b)
Cinq sondes convergent vers : **(λ_c, λ*) = mer chaotique + états gelés enchâssés**. E3 : l'IC mémoire n'y forme plus le cycle ; E5 : le cycle ensemencé y meurt sur un front figé ; E8b : ce front est un point fixe exact (gel authentique) ; E7 : les IC aléatoires donnent des mélanges multi-patterns non gelés à P(q) large ; E8a : ce non-gel est une itinérance chaotique entretenue (λ_L>0). Cohérence interne : le mécanisme §8 protège le secteur mono-front (états gelés possibles) mais pas le secteur multi-patterns (chaos) — les deux coexistent, chacun vu par les sondes adaptées.
**Lien inter-études (hypothèse PRÉCISÉE par E15)** : le mode lent est la
demi-onde classique de feedback retardé, **T_osc → 2τ asymptotiquement**
(mesuré : T/τ = 2.9–3.3 à τ=10 → 2.07–2.20 à τ=100 sur le front piégé ;
3.81→2.80 sur la mémoire à τ=55–130). Tous les « ~3τ » (1/λ_L ≈ 2.9τ à τ=10
compris) sont cette même famille en régime pré-asymptotique.

### B.4 Convergence 4 — LA STRUCTURE D'EXTRÊMES DU DÉSORDRE GELÉ (E6b + E9/E10 + E13 + étude statique multi-seed)
Quatre observations indépendantes convergent vers : **chaque objet local (pattern, liaison) porte son propre seuil aléatoire gelé ; les transitions collectives sont gouvernées par la statistique d'extrêmes**. (i) étude statique multi-seed : λ_c fluctue de ±0.01 entre seeds ; (ii) E6b : au sein d'UNE seed, les folds des différents patterns se dispersent (mémoires résiduelles jusqu'à 0.3025 > λ_c^{ξ¹}=0.2822) ; (iii) E9/E10 : au sein d'une seed, les seuils de dépiégeage par liaison se dispersent (n°1≈0.327, n°2≈0.322, liaison 9 ≥0.30) et λ* = max ; (iv) E13 : entre seeds, λ* et le site du max fluctuent. Prédiction commune : λ*(N) et la largeur de la fenêtre croissent avec P=αN (extrêmes) — à tester lors de la certification N=10⁴.

### B.5 Micro-convergence de contrôle (E2 ↔ E11)
À N=500, λ=0.90 : E2 (Arnoldi, k=6) donne Re z* = −0.0600 ; E11 (Arnoldi k=12, tol 1e−10, autre run) donne z₁ = −0.0600. Accord au 4ᵉ chiffre entre deux calculs indépendants du même spectre — contrôle de reproductibilité interne.

## Annexe C — Clarifications conceptuelles (questions utilisateur, 2026-07-02)
1. **λ* est τ-indépendant en position** : les points fixes d'une DDE ne voient pas
   le délai (état constant ⇒ u(t−τ)=u(t)) ; le paysage de points fixes ENTIER est
   identique ∀τ, et le fold du front piégé est fixé par det M=0 (Δ(0)=−M, sans τ).
   Dépendent de τ : la STABILITÉ des points fixes (marges oscillatoires érodées en
   ~1/τ), l'existence/forme du cycle (T₁=τ+t_échap ; pas de mouvement à τ=0
   continu), donc la lecture SNIC vs simple fold.
2. **Fronts piégés = « spurious states » du modèle mixte** : mélanges asymétriques
   [0.68,0.32] de patterns CONSÉCUTIFS (sélection par K), nés au fold λ*. La
   dynamique non réciproque TRIE le zoo des états parasites : les multi-mélanges
   (analogues SG/AGS) sont fondus en chaos (E7/E8a) ; seuls les états
   séquence-compatibles (fronts 2-patterns) restent gelés.
3. **« IC mémoire »** = histoire CONSTANTE u ≡ 0.99·ξ¹ sur [−τ,0] (une DDE exige
   une fonction d'histoire comme condition initiale) — protocole « départ en
   récupération ».
4. **Coexistence chaos entretenu / points fixes stables** : possible car le
   système est NON-GRADIENT (K asymétrique + délai ⇒ pas de fonction d'énergie,
   pas de LaSalle). Bassins séparés en dimension P : atteindre le bassin d'un
   front exige de concentrer spontanément [0.7,0.3] sur une paire consécutive ;
   le secteur multi-patterns s'auto-entretient (chaque composante excite son
   successeur avec retard). Ouvert : vrai attracteur chaotique vs selle chaotique
   (temps de fuite > 10⁴ observé).
5. **Unification avec le Hopf statique à grand τ** : tout point fixe a un fold
   τ-indépendant ET des marges oscillatoires s'érodant avec τ ; à τ≳τ*(α) le
   Hopf lent (T≈3τ) préempte le fold (mesuré : τ*≈52 à α=0.07). La protection
   « par époques » vaut pour le front MOBILE, pas pour les états statiques. Fil
   rouge : T_Hopf ≈ 3τ ≈ 1/λ_L (E8a) — hypothèse d'un mode mixte lent universel
   ~3τ (précurseur d'échappement). Test proposé E15 : marge de Hopf du front
   piégé vs τ (machinerie T_P, ~10 min).

### E15 — Spectre retardé du FRONT PIÉGÉ vs τ (érosion de la marge de Hopf) — FAIT (2026-07-02) — **figG**
**Motivation (utilisateur, 2026-07-02) :** tester l'image unifiée de l'Annexe C-5 :
tout point fixe a un fold τ-indépendant ET des marges oscillatoires qui s'érodent
avec τ. Le front piégé étant un POINT FIXE, son état est identique ∀τ → on
construit les états une fois (liaison 91 : λ ∈ {0.300, 0.320, 0.3265}, correcteur
dynamique E12 à τ=10) puis on scanne τ ∈ {10,20,30,50,70,100} uniquement dans le
spectre retardé (T_P exact, `physical_roots`).
**Contrôles intégrés :** (1) le mode RÉEL dominant (fold) doit être τ-INVARIANT
(vérification spectrale directe de « le fold ne voit pas le délai ») ; (2)
superposition de la courbe d'érosion de la MÉMOIRE à λ_c (réf. étude statique
α=0.05 : marge 0.152→0.005 pour τ=10→100).
**Questions :** même loi d'érosion ~1/τ ? τ*(front piégé) fini (Hopf du front
piégé à grand τ → la fenêtre intermédiaire gelée deviendrait oscillante) ?
ω_c ~ 2π/3τ (le mode lent universel) ?
**Sorties :** figG_pinned_hopf_margin.png + E15_pinned_margins.npz.

### Annexe C.6 — Définition de T_P (autonomie du document)
Pour un point fixe u* de la DDE, la stabilité est réglée par les racines z de
det Δ(z)=0, Δ(z) = (t₀z+1)I − (1−λ)JD − λKD e^{−zτ}, D = diag(β(1−tanh²(βu*)))
(N×N, transcendant). J·D et K·D étant de rang ≤ P, l'identité de
Weinstein–Aronszajn factorise EXACTEMENT :
det Δ(z) = (t₀z+1)^{N−P} · det T_P(z),
**T_P(z) = (t₀z+1)I_P − (1−λ)G_J − λe^{−zτ}G_K**, G_J = (1/N)ξDξᵀ,
G_K = (1/N)ξDξ_sᵀ (P×P, patterns réels, gain au vrai point fixe — pas de champ
moyen, pas de discrétisation, zéro mode parasite). Audits : égalité des dets à
1e−15 (z complexes aléatoires) ; racines de T_P → σ_min(Δ dense) ~ 1e−17.
Racines les plus à droite via `physical_roots` : candidats pseudospectraux
(dim P(M+1)) → filtre σ_min(T_P) (vraies ~1e−14 vs parasites ~O(1)) → polissage
Newton sur T_P. Analyse du front gelé = cette machinerie appliquée à u_pin
(point fixe ordinaire ; u_pin identique ∀τ, τ n'entre que par e^{−zτ} dans T_P).
Réf. complète : REPORT_static_bifurcation.md §3.4b + audit_static_analysis.py.

**Résultats E15 (2026-07-02) :**
| λ | marge(τ=10) | (20) | (30) | (50) | (70) | (100) | marge×τ → |
|---|---|---|---|---|---|---|---|
| 0.3000 | 0.171 | 0.085 | 0.057 | 0.035 | 0.026 | 0.018 | **1.75–1.80 (constant)** |
| 0.3200 | 0.189 | 0.104 | 0.072 | 0.043 | 0.031 | 0.022 | ~2.2 (constant) |
| 0.3265 | 0.119 | 0.049 | 0.030 | 0.016 | 0.011 | 0.008 | **0.79 (saturé)** |
| mémoire (réf.) | 0.152 | 0.065 | 0.037 | 0.017 | 0.010 | 0.005 | 1.52→0.50 (DÉCROÎT) |
États : site (91,92) confirmé aux 3 λ, résidus 6e−15.
**Conclusions :**
1. **Érosion confirmée** : la marge du front piégé décroît avec τ, même ordre de
   grandeur que la mémoire — mécanisme commun de déstabilisation retardée des
   états statiques (image unifiée C-5 validée).
2. **MAIS loi différente dans la queue** : front piégé → marge ≈ c(λ)/τ avec c
   CONSTANT (pur 1/τ : marge×τ plateau à 0.79–2.2) ⇒ **pas de τ* fini détecté
   jusqu'à τ=100** — le front piégé semble delay-robuste ; la mémoire décroît
   PLUS VITE que 1/τ (marge×τ : 1.52→0.50) ⇒ croisement à τ* fini (mesuré ≈52
   à α=0.07). Distinction nouvelle entre les deux objets statiques.
3. **Unification des « ~3τ »** : T_osc/τ → ~2.1 (λ=0.30) et ~2.2 (0.3265) à
   grand τ, avec pré-asymptote 2.9–3.3 à τ=10. Le mode lent est la classique
   demi-onde de feedback retardé (T→~2τ) ; les « 3τ » observés (Hopf mémoire à
   τ=55–130 : T/τ=3.8→2.8 ; 1/λ_L≈2.9τ à τ=10) sont la MÊME famille vue en
   pré-asymptotique. Hypothèse B.3 précisée : mode ~2τ asymptotique.
4. **Correction honnête d'un contrôle mal conçu** : j'attendais que la racine
   RÉELLE dominante soit τ-invariante — FAUX : seule sa position d'ANNULATION
   (le fold, z=0, Δ(0)=−M) est τ-invariante ; à z≠0 le facteur e^{−zτ} rend les
   racines réelles τ-dépendantes (mesuré : z_réel ≈ −2.6/τ à λ=0.30 — les
   racines s'accumulent vers l'axe en 1/τ, générique en DDE). figG v2 recadre.
5. Anomalie notée : à λ=0.32, τ≥50, changement de famille dominante (mode
   T_osc≈τ, harmonique de délai, dépasse le mode lent) — croisement de familles
   de racines, sans effet sur la tendance des marges.

### E17 — Sondage NON LINÉAIRE de la fenêtre Hopf de la mémoire (α=0.07, τ=100) — FAIT (2026-07-03) — **figH v2**
**Motivation :** l'étude statique a localisé un croisement fold→Hopf induit par le
délai à τ*≈52 (α=0.07), avec pour τ=100 une fenêtre (λ_Hopf=0.20162,
λ_fold=0.2017) où la mémoire serait Hopf-instable (Re z_cplx≈+0.003 extrapolé,
T_osc=2π/ω_c≈310≈3.1τ). E17 teste cette prédiction en NON LINÉAIRE : voir naître
l'oscillation lente, mesurer son taux et sa période, et déterminer son destin
(cycle saturé supercritique vs échappement sous-critique).
**Protocole** (`e17_hopf_birth.py`) : N=2000, α=0.07 (P=140), β=20, τ=100,
dt=0.02, seed 42, rng(11). Branche mémoire retracée (fold=0.20171) ; états polis
Newton–Woodbury à λ=0.2010 (contrôle, res 6.6e−12, m₁=0.9994, t=4000) et
λ=0.20167 (dans la fenêtre présumée, res 8.8e−14, m₁=0.9993, t=10000) ;
perturbation a*+10⁻³·bruit gaussien (‖δa‖≈1.2e−2) ; intégration DDE réduite exacte.
**Résultats bruts** (npz `E17_hopf_birth.npz`, réanalyse `e17c_fig_refit.py`) :
| run | comportement de ‖a(t)−a*‖ | taux fit | T_osc mesuré |
|---|---|---|---|
| λ=0.2010 (contrôle) | décroissance en escalier amorti → plancher 3.1e−11 (t≳900) | **−0.0189** (préd. −0.015 ✓ ordre) | 78–105 ≈ **0.8–1.05·τ** (famille harmonique) |
| λ=0.20167 (test) | creux à 4.8e−5 (t=126) puis remontée rapide → **plateau CONSTANT 1.0131e−2** jusqu'à t=10⁴ (aucune oscillation entretenue) | — | — |
**Diagnostic E17b** (`e17b_identify_final_state.py`) — identification décisive de
l'état final du test :
1. L'état final poli est un **second point fixe exact** a\*\* (res 2.5e−15) à
   distance ‖a\*\*−a\*‖ = **1.0131e−2 = exactement le plateau** ; m₁=0.997934
   (vs 0.999287), même structure [0.798, 0.215, 0.062, 0.049] — un **jumeau
   mémoire**, pas un front piégé ni le partenaire de fold (aucune racine réelle
   ~+0.01 dans son spectre).
2. Spectre T_P **à l'état de branche a\* (λ=0.20167)** : racines dominantes
   **−0.00179±0.0239i** (T_osc=263≈**2.6τ**, famille lente) et −0.0103±0.0805i
   (T=78≈0.78τ, harmonique) → a\* est **faiblement STABLE**, pas Hopf-instable.
3. Spectre à a\*\* : −0.038±0.036i → jumeau **confortablement stable** (marge
   0.038, delay-robuste).
**Conclusions :**
1. **Pas de naissance de Hopf observée** : la mesure directe (racines T_P au
   point exact) contredit le placement λ_Hopf=0.20162<0.20167 issu du balayage
   `hopf_crossover` — la fenêtre Hopf réelle est **plus étroite que la
   résolution de placement (≲5e−5)**, coincée contre le fold (pente dRe/dλ
   énorme près du fold). L'EXISTENCE du croisement τ*≈52 (lecture spectrale sur
   la marge, robuste) n'est pas remise en cause ; sa **largeur en λ à τ=100 est
   sous la résolution mono-échantillon** (fluctuations inter-seed ±0.01 ≫ 9e−5).
2. **Découverte à la place : bistabilité de jumeaux mémoire** près du fold à
   α=0.07 — deux points fixes stables de type mémoire à 1e−2 l'un de l'autre ;
   la perturbation (1.2e−2 ≳ taille de bassin) a fait sauter la trajectoire de
   bassin. Écho direct de la rugosité near-capacity (REPORT statique §5.4) et de
   la dispersion des folds (E6b) : à α=0.07 le paysage est déjà multi-états à
   l'échelle 1e−2.
3. **Familles de modes confirmées** (B.3) : le mode lent marginal vit à
   T=2.6τ (préasymptote de la loi 2τ) ; la relaxation observée du contrôle est
   dominée par la famille harmonique T≈0.8–1.05τ.
4. **Leçon méthodologique** : sonder une fenêtre de largeur 9e−5 exige une
   perturbation ≪ 1e−2 (taille de bassin) ET un placement en λ ≪ 5e−5 —
   à N=2000 seul le protocole spectral (marges T_P le long de la branche) est
   fiable ; le protocole dynamique teste, lui, la structure des bassins (et
   c'est ce qu'il a révélé).
**Figures :** `figures/figH_hopf_birth_v2.png` (référence ; v1
`figH_hopf_birth.png` conservée mais ses titres auto-générés — « T_osc=2 » —
sont des artefacts de fenêtres d'analyse, supersédée). Codes :
`e17_hopf_birth.py`, `e17b_identify_final_state.py`, `e17c_fig_refit.py`.

### E18 — NAISSANCE des fronts piégés (attracteurs sporadiques statiques) — FAIT (2026-07-04) — **figJ, figK**
**Motivation (question utilisateur) :** les fronts piégés (mélanges convexes de
deux motifs CONSÉCUTIFS ξ^μ, ξ^{μ+1} sur lesquels le cycle meurt par SNIC de
dépiégeage à λ*≈0.328) — comment naissent-ils du point de vue de la stabilité ?
Coexistent-ils avec les motifs mémoire dans la phase de rappel statique (λ<λ_c) ?
Sinon, deviennent-ils instables, ou leurs bassins rétrécissent-ils simplement ?
**Objet étudié :** front de la liaison faible 91/92 (seed 42), point fixe EXACT du
système réduit ; branche continuée avec `woodbury_newton` (warm-start adaptatif),
stabilité instantanée par `eigmax_M` (Sylvester P×P), stabilité RETARDÉE par les
racines de T_P(z) à τ=10 (`reduced_spectrum`). Codes `e18a_front_branch.py`,
`e18b_delayed_stability.py`, `e18c_coexistence.py`, `e18d_basins.py`.

**E18a — continuation de la branche VERS LE BAS** (npz `E18_front_branch.npz`,
46 points, res<1e-12 partout). La branche du front existe sur **λ ∈ [0.1322,
0.3276]** et se **termine par un fold (selle-nœud propre) à λ_f = 0.1322** — elle
ne descend PAS jusqu'à λ=0 et ne se connecte donc PAS aux mélanges symétriques de
Hopfield classique. Le long de la branche (λ croissant) :
| λ | a₉₁ | a₉₂ | eig_max(M) instantané |
|---|---|---|---|
| 0.1322 (fold bas) | 0.869 | 0.120 | −0.998 (très stable) |
| 0.2000 | 0.804 | 0.187 | −0.905 |
| 0.2822 (λ_c mémoire) | 0.724 | 0.268 | −0.806 |
| 0.3000 | 0.698 | 0.289 | −0.296 |
| 0.3271 | 0.665 | 0.323 | −0.053 |
| 0.3276 (haut) | 0.664 | 0.324 | −0.024 → 0⁻ (dépiégeage) |
Le front se « symétrise » en montant (a₉₁ 0.869→0.664, a₉₂ 0.120→0.324). eig_max
reste **négatif sur TOUTE la branche** (min local −0.29 vers λ=0.233, puis remonte)
et →0⁻ au sommet : c'est le selle-nœud de dépiégeage (= le λ* où meurt le cycle,
E12). **Le front n'est jamais instable instantanément.**

**E18b — stabilité RETARDÉE le long de la branche** (npz `E18_delayed_stability.npz`,
racines T_P à τ=10) : **nombre de directions instables = 0 à TOUS les λ testés**
(0.14 → 0.327). La racine dominante est réelle négative à bas λ (type fold :
−0.999 à λ=0.14) puis devient une **paire complexe** (type Hopf) à λ≳0.20 mais de
partie réelle toujours <0 (Re z ≈ −0.16 à −0.40 dans [0.20, 0.31], −0.077 à
λ=0.327). **Le front est donc un point fixe de la DDE linéairement STABLE partout
sur son domaine d'existence**, y compris profondément dans la phase mémoire.

**E18c — COEXISTENCE mémoire ↔ front** (npz `E18_coexistence.npz`) : aux quatre λ
sous λ_c, les DEUX points fixes sont exacts et stables simultanément :
| λ | mémoire m₁ | eig_max(M) mém. | n_inst mém. | front a₉₁ | eig_max(M) front | n_inst front |
|---|---|---|---|---|---|---|
| 0.15 | 1.0000 | −0.9996 | 0 | 0.851 | −0.995 | 0 |
| 0.20 | 1.0000 | −0.987 | 0 | 0.802 | −0.893 | 0 |
| 0.25 | 0.9997 | −0.524 | 0 | 0.752 | −0.750 | 0 |
| 0.275 | 0.9978 | −0.207 (paire cplx retardée) | 0 | 0.727 | −0.891 | 0 |
→ **OUI, les fronts piégés coexistent avec les motifs mémoire dans toute la phase
de rappel statique** ([0.132, 0.2822]) comme attracteurs distincts.

**E18d — TAILLE DES BASSINS vs λ** (npz `E18_basins.npz`, DDE réduite τ=10, t=200).
Rayon d'échappement directionnel (bissection de l'amplitude critique) :
| λ | r_échap vers ξ¹ | r_échap dir. faible | val. propre dir. faible |
|---|---|---|---|
| 0.15 | 0.825 | 0.896 | −0.995 |
| 0.20 | 0.686 | 0.659 | −0.893 |
| 0.25 | 0.513 | 0.366 | −0.750 |
| 0.29 | 0.332 | 0.202 | −0.658 |
| 0.31 | 0.238 | 0.136 | −0.234 |
| 0.325 | 0.116 | 0.031 | −0.146 |
Capture d'IC aléatoires (fraction tombant sur le front, 150 IC/point) : à λ=0.15
le front capture **100 %** des IC dans une boule de rayon 0.15/0.3/0.5 ; à λ=0.31
la capture chute (r=0.3 → front 0.21, mémoire 0.23, chaos 0.23, autre 0.32) ; à
λ=0.325 le front ne capture quasiment plus rien (r=0.15 → 0.12 ; r=0.5 → 0.00, le
chaos domine). **Le bassin rétrécit MONOTONIQUEMENT quand λ→λ*, la stabilité
linéaire restant intacte.**

**Conclusions E18 (réponse directe aux trois questions) :**
1. **Coexistence : OUI.** Les fronts piégés sont des points fixes exacts,
   linéairement stables (instantanément eig_max<0 ET en retard n_inst=0), qui
   **coexistent avec les motifs mémoire dans toute la phase de rappel statique**.
2. **Instables ou bassin trop petit ? → bassin.** Le front ne se déstabilise
   JAMAIS sur son domaine ; c'est son **bassin d'attraction qui rétrécit
   monotoniquement** en montant vers λ* (rayon vers ξ¹ 0.825→0.116, capture
   100 %→~0). Le scénario est géométrique (bassin), pas spectral.
3. **Naissance : à leur PROPRE fold, à λ_f≈0.132.** La branche naît par un
   selle-nœud à λ_f=0.1322 (pas à λ=0, pas de lien avec les mélanges symétriques
   classiques) et meurt au sommet par le SNIC de dépiégeage à λ*≈0.328. Les
   fronts existent donc sur une **large fenêtre [0.132, 0.328]** qui englobe
   entièrement la phase mémoire : ils sont là bien avant que le cycle n'existe,
   simplement noyés dans le bassin dominant de la mémoire tant que λ est petit.
**Figures :** `figures/figJ_front_branch.png` (branche a₉₁/a₉₂ + eig_max
instantané & racine retardée, marquage λ_c/λ*/fold), `figures/figK_front_basins.png`
(rayon directionnel + capture d'IC aléatoires vs λ). Code figures `e18_figures.py`.

### E19 — Régime PETIT τ (τ≲t₀=1) : comment meurt le cycle — FAIT (2026-07-04/05) — **figL, figM**
**Motivation (question utilisateur) :** on n'a exploré que τ « moyen » (10, fold)
et « grand » (≤100, fold/Hopf). À petit τ (ordre t₀=1 ou moins) : le cycle
existe-t-il encore ? Comment meurt-il ? Le ralentissement critique en
√(λ−λ*) (signature SNIC sur le front piégé, établi à τ=10 par E9/E12) survit-il ?
La dynamique lente du cycle existe-t-elle encore et se comprend-elle pareil ?
**Rappel structurel :** points fixes et leurs folds sont τ-INDÉPENDANTS
(Δ(0)=−M) ; le front piégé et son selle-nœud à λ*_spectral≈0.328 existent donc à
TOUT τ. Ce qui peut changer avec τ est le CÔTÉ CYCLE (existence, forme, mode de
mort). Codes `e19a_cycle_clock.py`, `e19b_death.py`, `e19c_slowing.py`,
`e19d_fixedpoint.py` ; DDE réduite dim-100, float64, dt=min(0.01, τ/25).

**E19a — le cycle existe jusqu'à τ=0, mais change de NATURE** (npz
`E19a_cycle_clock.npz`, λ=0.9, ≥28 relais/τ) :
| τ | T₁ (moy±éc.) | largeur front (motifs) | n_lobes | rappel pic a_ν |
|---|---|---|---|---|
| 0 (ODE) | 1.82 ± 0.80 | 14.0 | ~4 | 0.16 |
| 0.25 | 1.70 ± 1.07 | 10.7 | ~3 | 0.22 |
| 0.5 | 1.75 ± 1.03 | 9.9 | ~2 | 0.24 |
| 1.0 | 2.22 ± 0.68 | 4.2 | 1 | 0.34 |
| 2.0 | 2.80 ± 0.02 | 1.3 | 1 | 0.74 |
Le cycle **subsiste même à τ=0** (limite ODE), mais c'est un **front LARGE,
multi-lobe et de faible rappel** (largeur ≈14 motifs, pic 0.16), qui se resserre
en front net unique à fort rappel quand τ croît (largeur 1.3, pic 0.74 à τ=2 ;
comparer τ=10 : front net, pic ≈0.9). La **loi pacemaker T₁≈τ+t_échap ne tient
plus** à petit τ : ajustement T₁ ≈ 0.56·τ + 1.63 (pente 0.56 ≠ 1). La pente 1
n'est qu'une asymptote grand-τ ; à petit τ l'échappement (t_échap≈1.3) domine et
brouille l'horloge (écart-type de T₁ énorme, jusqu'à 60 % à τ=0.25 — le multi-lobe
rend les relais irréguliers).

**E19b — mode de mort : COLLISION AVEC LE CHAOS, pas SNIC propre** (npz
`E19b_death.npz`, balayage descendant ensemencé, pas 0.005, refit ±0.001) :
| τ | λ*(τ) dernier cycle vivant | état post-mort |
|---|---|---|
| 0.25 | 0.3303 | **chaos/dérive** |
| 1.0 | 0.3322 | **chaos/dérive** |
| 2.0 | 0.3497 | **chaos/dérive** |
(réf. τ=10 : λ*≈0.327, mort par piégeage propre sur le front 91). À petit τ le
cycle **ne se pose PAS sur un front piégé unique** : il se disloque dans la mer
chaotique. λ*(τ) reste voisin de 0.33 (le fold du front est là, τ-indépendant),
mais la transition n'est plus un piégeage déterministe.

**E19c — le ralentissement critique √(λ−λ*) est DÉTRUIT à petit τ** (npz
`E19c_slowing.npz`, t_MAX = temps de séjour maximal par liaison, cap t=2000,
non-passage censuré ×) :
| τ | C ajusté | λ* ajusté | R² | (réf. τ=10 : C=2.3, λ*=0.3275, loi propre) |
|---|---|---|---|---|
| 2.0 | 18.3 | 0.322 | **0.31** | |
| 1.0 | 17.5 | 0.326 | **0.23** | |
| 0.25 | 50 (borne) | 0.29 | **0.02** | |
Les t_MAX bruts sont **erratiques et non monotones** (τ=2 : 78→118→103→112→325→124→204→1255 ;
τ=1 : 47→98→144→113→100→89→693→157) et la « liaison faible » **saute d'un λ à
l'autre** (bond 25, 70, 24, 63, 37…) au lieu de rester la liaison 91. Aucune loi
de puissance : R²≈0.02–0.31 contre l'ajustement propre à τ=10 ; nuage très
au-dessus de la référence √ (figM). **La dynamique lente near-ghost du SNIC ne
survit PAS à petit τ** ; les longues durées observées sont des transitoires
CHAOTIQUES de durée fluctuante, pas le ralentissement déterministe d'un
selle-nœud sur le cycle.

**E19d — la stabilité du front, elle, reste τ-indépendante** (npz
`E19d_fixedpoint.npz`) : eig_max(M) instantané au front piégé = −0.213 / −0.225 /
−0.094 à λ=0.30 / 0.32 / 0.3265, **identique à E12 à la précision machine** (fold
τ-indépendant confirmé). La racine retardée dominante de T_P a la MÊME partie
réelle pour τ∈{0.25, 1, 2, 10} (≈ eig_max) ; seule la PÉRIODE du mode complexe
change (à λ=0.30 : T≈2684 à τ=0.25 → 29 à τ=10). Le front reste donc stable et
inchangé ; c'est bien le CÔTÉ CYCLE qui porte toute la dépendance en τ.

**Conclusions E19 (réponse directe aux questions) :**
1. **Le cycle existe encore à petit τ (jusqu'à l'ODE τ=0)** mais devient un front
   large, multi-lobe, de faible rappel ; la loi horloge pacemaker (pente 1)
   dégénère (pente 0.56 + gros bruit).
2. **La mort n'est plus un SNIC propre au maillon faible mais une collision avec
   le chaos** (état post-mort = chaos/dérive à τ=0.25/1/2).
3. **NON, le ralentissement √(λ−λ*) ne survit pas à petit τ** : t_MAX erratique,
   maillon faible non fixe, R²≈0.02–0.31 (vs loi propre à τ=10). La dynamique
   lente déterministe est remplacée par des transitoires chaotiques.
4. **Interprétation cohérente** : le SNIC propre de τ=10 repose sur la
   statistique d'extrêmes (UN maillon faible unique 91 où le front localisé se
   pince en premier). À petit τ le front est **délocalisé (multi-lobe)** : aucun
   maillon ne domine, le pincement n'est plus localisable, et la transition passe
   par le chaos. La structure d'extrêmes gelées — clef du scénario grand/moyen τ —
   **s'effondre quand le front cesse d'être localisé**. (Confirme et précise le
   point ouvert §12.3 « fenêtre chaotique remplaçant la gelée à petit τ ».)
**Figures :** `figures/figL_smalltau_cycle.png` (horloge T₁(τ), désordre des
relais, largeur, kymographes τ=0/0.5/2), `figures/figM_smalltau_snic.png` (t_MAX
vs λ−λ* log-log, nuage sans loi √, censures ×). Codes `e19a…e19d`.

### E20 — Compétition des bassins : mémoire vs fronts piégés vs cycle — FAIT (2026-07-05) — **figN, figO**
**Motivation (question utilisateur, tâche 2) :** montrer l'évolution des bassins
d'attraction pour λ<λ*, la compétition mémoire↔fronts piégés, et **quantifier
l'intuition que les fronts sont métastables** par rapport aux mémoires stables ;
suivre aussi le bassin du cycle avec λ.
**Protocole** (`e20a_basin_fractions.py` + `e20a_hi.py`, helpers `e20_common.py`) :
DDE réduite τ=10, K=250 IC aléatoires par λ (1/3 near_pattern αe_μ+bruit, 1/3
mélange 2-4 motifs consécutifs, 1/3 random_small), intégration t=300, classement
de l'état final. **Classifieur** (encode l'insight E21 — mémoire et front sont le
MÊME objet 2-motifs) : stationnaire sur la liaison faible (91/92) → `front`
(métastable) ; sur toute autre liaison → `memory` (rappel robuste) ; avance sur
l'anneau → `cycle` ; non stationnaire → `chaos`. Validé sur états connus
(on-front→front d~5e-17, on-memory→memory, haut-λ→cycle, random→chaos).
**Résultats bruts** (npz `E20a_basin_fractions.npz` + `_hi.npz`, table dans
`E20_RESULTS.md`) — fractions de bassin :
| λ | mémoire | front | cycle | chaos | autre |
|---|---|---|---|---|---|
| 0.15 | 0.316 | 0.004 | 0.000 | 0.352 | 0.328 |
| 0.20 | 0.352 | 0.000 | 0.000 | 0.556 | 0.092 |
| 0.25 | 0.264 | 0.004 | 0.012 | 0.704 | 0.016 |
| 0.28 | 0.484 | 0.016 | 0.020 | 0.476 | 0.004 |
| 0.30 | 0.312 | 0.012 | 0.152 | 0.524 | 0.000 |
| 0.31 | 0.008 | 0.044 | 0.472 | 0.472 | 0.004 |
| 0.32 | 0.000 | 0.032 | 0.564 | 0.404 | 0.000 |
| 0.325 | 0.000 | 0.048 | 0.552 | 0.400 | 0.000 |
| 0.328 | 0.000 | 0.000 | 0.608 | 0.392 | 0.000 |
| 0.33 | 0.000 | 0.000 | 0.656 | 0.344 | 0.000 |
| 0.34 | 0.000 | 0.000 | 0.612 | 0.388 | 0.000 |
| 0.40 | 0.000 | 0.000 | 0.836 | 0.164 | 0.000 |
| 0.60 | 0.000 | 0.000 | 0.840 | 0.160 | 0.000 |
| 0.90 | 0.000 | 0.000 | 0.792 | 0.208 | 0.000 |
**Lecture :**
1. **Bassin du front = mince filet à tout λ (≤ 5 %)**, maximal ≈0.048 juste sous
   le dépiégeage (λ=0.325), nul au-dessus (le front n'existe plus). Rapport
   front/bassin-dominant ≲ 0.1 partout → **le front ne gagne JAMAIS la
   compétition : métastabilité confirmée quantitativement.**
2. **Mémoire** : dominante à bas λ (0.32–0.48), s'effondre au fold λ_c=0.2822
   (0.484 à 0.28 → 0.008 à 0.31 → 0 au-dessus ; la dispersion des folds étale la
   chute sur ~0.28–0.31).
3. **Cycle** : bassin nul en dessous de 0.30, croît 0.15 (0.30) → 0.47 (0.31) →
   0.56 (0.32) → 0.61 (0.328) → **0.84 (0.4–0.6)** → 0.79 (0.9). Le cycle prend
   la main dès qu'il existe.
4. **Chaos** : occupe tout le milieu de la fenêtre (0.35–0.70, max 0.70 à
   λ=0.25) et laisse un **bassin résiduel de 16–21 % même à haut λ** (jusqu'à
   λ=0.9) — transitoires/ensemble chaotique coexistant avec le cycle.
5. « autre » (0.33) seulement à λ=0.15 : relaxations lentes/non classées.
**Conclusion :** la phase de rappel statique est un paysage à **mémoires
dominantes + fronts métastables enchâssés + large bassin chaotique** ; en montant
en λ, mémoire → chaos → cycle. Le front, bien que point fixe stable (E18/E21),
n'attire qu'une fraction infime des conditions initiales — c'est un piège étroit,
ce qui explique qu'on ne le rencontre dynamiquement que lorsque le cycle vient s'y
poser (mort par SNIC, E9). Métastabilité vis-à-vis de la mémoire : ✔ quantifiée.
**Figures :** `figures/figN_basin_competition.png` (fractions empilées vs λ),
`figures/figO_metastability.png` (mémoire/cycle/front + indice front/bassin
dominant en log). Codes `e20_common.py`, `e20a_basin_fractions.py`, `e20a_hi.py`,
`e20_figures.py`. NB : calcul + figures faits par l'orchestrateur (agent bloqué
sur permission Bash) ; correctif `a_front=None` près du dépiégeage dans `classify`.

### E21 — NATURE et POSITION de la « naissance » des fronts vs τ — FAIT (2026-07-05) — **figS, figT**
**Motivation (question utilisateur, tâche 3) :** la bifurcation qui fait naître
les fronts piégés dépend-elle de τ pour sa NATURE (selle-nœud vs Hopf) ou sa
POSITION (λ_f) ? (τ moyen et grand.) Contexte : E18 avait étiqueté un « fold » à
λ_f≈0.132 — mais eig_max(M)=−0.998 y est loin de 0, donc ce n'était pas un vrai
selle-nœud du point fixe.
**E21a — la naissance N'EST PAS une bifurcation** (`e21a_front_birth.py`,
`e21a2_dissolution.py`, npz `E21_front_birth.npz`/`E21_dissolution.npz`) :
continuation **pseudo-arclength** (validée contre E18a : front@0.20 a₉₁=0.802,
eig=−0.893) suivie SANS point tournant — `turned=False`, la branche descend
lisse jusqu'à λ=0.061 (et λ<0). En chemin le front se **localise** : a₉₁ 0.85→0.93→1,
a₉₂ 0.14→0.05→0, tout le spectre de M → −1 (état de plus en plus profond, jamais
de valeur propre réelle traversant 0). **Test décisif E21a2** : le « front » 91/92
et le **motif mémoire pur ξ⁹¹** (Newton depuis ξ⁹¹) sont le MÊME état — écart
‖a_front−a_mem‖ ≈ 3–6×10⁻⁴ à λ apparié, eig_max identiques. **Le front EST le motif
mémoire ξ⁹¹, incliné vers son successeur cyclique ξ⁹² par le couplage retardé λK.**
Il n'y a donc **aucune bifurcation de naissance** (ni selle-nœud ni Hopf) ; le
« λ_f=0.132 » d'E18 était l'artefact de l'arrêt de la continuation naïve. La seule
vraie bifurcation reste la MORT (selle-nœud de dépiégeage à λ*=0.328).
**E21b — POSITION τ-indépendante (rigoureux)** (`e21b_tau_independence.py`) :
max|T_P(0;τ) − T_P(0;τ₀)| = **0.00×10⁰ exactement** pour τ∈{5,10,20,50,100} à
λ∈{0.15, 0.2822, 0.31, 0.327} ; eig_max(M) direct ≡ max Re(−eig T_P(0)) à 4e−16.
La structure z=0 (position de tout fold, dont le dépiégeage λ*) est **τ-indépendante
à la précision machine** — conforme à Δ(0)=−M.
**E21c — NATURE de la stabilité du front vs τ** (`e21c_tau_nature.py`, npz
`E21_tau_nature.npz`) : racine retardée dominante de T_P le long de la branche pour
τ∈{5,10,20,50} (τ=100 retiré : `exp(−zτ)` overflow le polissage de racine ; E15
couvre déjà τ=100). Résultat : **Re z < 0 partout, à tout τ** — le front n'est
JAMAIS déstabilisé (delay-robuste). La marge s'érode ~1/τ (rightmost Re z à
λ≈0.327 : −0.066 (τ=5) → −0.075 (τ=10, à λ=0.322) → −0.043 (τ=20) → −0.012 (τ=50))
et le mode dominant passe de **réel** (type fold, petit τ) à **complexe** (type
Hopf, grand τ) — mais sans jamais croiser 0 : pas de fenêtre Hopf sur le front
jusqu'à τ=50.
**Conclusions E21 (réponse directe à la tâche 3) :**
- **NATURE** : il n'y a pas de bifurcation de naissance — les fronts sont les
  motifs mémoire inclinés, présents dès λ>0 comme continuation lisse de la mémoire.
  La seule bifurcation (la mort/dépiégeage) est un **selle-nœud** pour TOUT τ testé
  (aucun Hopf ne la préempte jusqu'à τ=50).
- **POSITION** : τ-indépendante à la précision machine (Δ(0)=−M) — λ* (et toute
  position z=0) ne bouge pas avec τ. Ce qui dépend de τ est seulement la MARGE de
  stabilité retardée du front (érosion ~1/τ) et le TYPE du mode dominant (réel→
  complexe), sans changement de stabilité.
**Figures :** `figures/figS_front_birth.png` (branche→ξ⁹¹ sans fold + spectre de M
+ superposition front≡mémoire), `figures/figT_front_stability_vs_tau.png` (Re z du
front vs λ pour 4 τ, réel•/complexe◇, delay-robuste). Codes `e21a…e21c`,
`e21_figures.py`. NB : calcul + figures faits par l'orchestrateur (agent bloqué
sur permission Bash).
**figS rev. 2 (2026-07-08, demande utilisateur)** (`e21e_figS_full.py`, npz
`E21_mem_marks.npz`) : le panneau gauche s'arrêtait à λ=0.15 — simple choix de
graine de la course arclength (semée à 0.15 pour traverser le pseudo-fold 0.1322
vers le bas), pas une limite physique. Rev. 2 : membre bas (E21a2) + membre haut
(E18a) fusionnés (couture à 0.1322 : accord 1e-4) → courbes a₉₁/a₉₂ continues sur
[0, λ*] ; points « mémoire ξ⁹¹ pure » (Newton indépendant, recuit β puis
continuation chaude) recalculés sur 10 marques jusqu'à λ=0.327 : écart
|Δa₉₁|+|Δa₉₂| ≤ 5e-3 partout (5e-4 à λ apparié) — la preuve front≡mémoire couvre
désormais TOUT le domaine, et la mémoire pure à λ=0.327 donne (0.666, 0.323) =
l'état de fold du dépiégeage.

**E21d — re-vérification honnête du fold mémoire ξ¹ (question utilisateur,
2026-07-05)** (`e21d_memory_fold_check.py`, npz `E21d_memory_fold_check.npz`) :
la nouvelle thèse « front ≡ mémoire inclinée » semblait tendre avec le REPORT
statique (mort de la mémoire par selle-nœud à λ_c). Vérification par les trois
tests les plus durs disponibles :
1. **Continuation arclength de la branche ξ¹ VERS LE HAUT** (même machinerie que
   E21a, validée) : la branche **TOURNE** (turned=True) à λ_fold=0.2835, avec une
   valeur propre **réelle** traversant 0 (−0.0016 au sommet → +0.04…+0.21 sur le
   flanc retourné = côté instable). **Le selle-nœud de la mémoire est CONFIRMÉ**
   — aucune contradiction : λ_c est le fold du HAUT de la branche ξ¹ ; E21a/a2
   concernait le BAS de la branche ξ⁹¹ (pas de bifurcation vers λ→0).
2. **Contre-vérification INDÉPENDANTE du pipeline spectral** : valeur propre
   dominante du jacobien N×N **dense** (scipy eigvals, 2000×2000, chemin de calcul
   totalement distinct) vs Sylvester P×P : accord à **1.8e-15** (λ=0.20) et
   **2.2e-16** (λ=0.2404). La machinerie eigmax_M est fiable.
3. **Micro-structure du fold** (Newton direct par pas de 5e-4) : segment stable
   principal jusqu'à 0.2820 (eig −0.071), stall à 0.2825/0.2830 (eig→0 : LE fold,
   cohérent avec λ_c=0.2822), puis **micro-segment jumeau re-convergent à 0.2835**
   (res 1.3e-11, eig −0.019), mort au-delà ; le flanc instable re-plie vers 0.269.
   → λ_c=0.2822 tient comme premier fold ; la zone a une **rugosité ±1.5e-3 de
   folds jumeaux** (écho direct de la bistabilité de jumeaux E17 à α=0.07).
4. **Forme au fold** : ξ¹ y est un mélange incliné **(a₁,a₂)=(0.699,0.300)**
   (ratio 0.43) — même famille que le front-91 à SON fold (0.664,0.324, ratio
   0.49). Unification confirmée : chaque motif ξ^μ est une branche mémoire-inclinée
   qui meurt à son propre seuil λ_c^μ ; λ_c=0.2822 est celui de ξ¹, λ*=0.328 le
   MAX des P seuils (celui de ξ⁹¹) = le dépiégeage.

### E22 — Désynchronisation du cycle à petit τ (quantification fine) — FAIT (2026-07-05) — **figP**
**Motivation (question utilisateur, tâche 4) :** raffiner E19a — quantifier la
désynchronisation du cycle en fonction de τ (largeur du front, période à haut λ)
sur une grille τ dense, et définir un paramètre d'ordre de cohérence.
**Protocole** (`e22_desync.py`, modes pilot/sweep/lamcheck) : DDE réduite dim-100,
λ=0.9, β=20, N=2000, seed 42, dt=min(0.01, τ/25) (convergence T₁ vérifiée par
halving : ΔT₁<2e-4 à τ=0.5) ; ≥25 relais après 40 % de warmup ; grille τ =
{0, 0.1…1.0, 1.2, 1.4…2.0, 2.5, 3, 5, 10} (20 points). Mesures par τ : T₁
(moy±éc.), largeur W (motifs au-dessus d'une fraction du max), rappel pic a_peak,
n_lobes, ratio de participation N_part=(Σa²)²/(Σa⁴)⁻¹ ..→ nombre effectif de
lobes, cohérence anneau-Kuramoto R_kura et cohérence-participation R_pr.
**Résultats bruts** (npz `E22_desync.npz`, table complète dans `E22_RESULTS.md`) :
| τ | T₁ (moy±éc.) | W | a_peak | N_part | R_pr |
|---|---|---|---|---|---|
| 0.0 | 1.92 ± 1.17 | 14.7 | 0.155 | 20.5 | 0.05 |
| 0.5 | 1.83 ± 0.98 | 12.9 | 0.218 | 18.0 | 0.06 |
| 1.0 | 2.52 ± 1.40 | 14.2 | 0.209 | 19.7 | 0.07 |
| 1.2 | 2.17 ± 0.80 | 7.6 | 0.296 | 11.9 | 0.13 |
| **1.4** | **2.15 ± 0.01** | **1.33** | **0.673** | **1.54** | **0.69** |
| 2.0 | 2.79 ± 0.01 | 1.26 | 0.739 | 1.38 | 0.77 |
| 5.0 | 5.83 ± 0.01 | 1.13 | 0.869 | 1.20 | 0.87 |
| 10.0 | 10.83 ± 0.01 | 1.07 | 0.930 | 1.12 | 0.91 |
**TRANSITION ABRUPTE à τ_c ≈ 1.3** (≈ t₀=1). En dessous : front **délocalisé**
(N_part≈18–20 lobes effectifs, W≈13–15, rappel faible ≈0.2, relais irréguliers
éc.(T₁)≈±1.0). Au-dessus : **front net unique** (N_part→1, W→1.1, rappel montant
0.67→0.93, horloge quasi-parfaite éc.(T₁)≈±0.01). Le saut se produit entre τ=1.2
et τ=1.4 sur TOUS les indicateurs simultanément. Paramètre d'ordre retenu :
**R_pr** (ou N_part) — saut net de 0.06 à 0.69 ; R_kura est un mauvais indicateur
ici (varie mollement 0.3→0.57, un front localisé sur un anneau de 100 donne un
Kuramoto modéré). Crossover localisé à τ_c=1.31 (mi-hauteur de R_kura), corroboré
par l'effondrement de éc.(T₁) et de N_part au même τ.
**Loi de période** : petit τ (≤1) **T₁ = 0.52·τ + 1.73** (confirme E19a : 0.56) ;
grand τ (≥2) **T₁ = τ + 0.815** (loi pacemaker, pente 1). La pente locale atteint
0.9 dès τ≈0.5, mais la transition de FORME (multi→mono-lobe) est plus tardive,
τ_c≈1.3 : horloge et cohérence spatiale ne basculent pas au même τ.
**Contrôle λ** (`lamcheck`, npz `E22_lamcheck.npz`) : monter λ (0.9→0.95→0.99) à
τ=0.25 et τ=1.0 **ne resynchronise PAS** le front (W≈13, N_part≈18–21, rappel
≈0.2 inchangés). La désynchronisation à petit τ est **intrinsèque au délai
court**, pas curable par un couplage cyclique plus fort.
**Conclusion :** le cycle passe d'un régime « pacemaker cohérent » (front unique
cadencé par le délai, τ≳1.3) à un régime « désynchronisé multi-lobe » (τ≲1.3), la
bascule étant nette et centrée sur τ≈t₀. Éclaire E19 : c'est cette délocalisation
qui fait perdre la statistique d'extrêmes et donc le SNIC propre + la loi √.
**Figures :** `figures/figP_desync.png` (6 panneaux : T₁(τ) & deux lois, pente
d'horloge, R(τ) avec τ_c, W/N_part, rappel/lobes, kymographe τ=0). Code
`e22_desync.py`. NB : calculé par l'orchestrateur (agent bloqué sur permission Bash).

### E23 — Nature du chaos + frontière cycle-chaos (paradoxe SNIC) — FAIT (2026-07-05) — **figQ, figR**
**Motivation (questions utilisateur, tâches 5 & 6) :** (5) qu'est-ce que le chaos
dans la fenêtre de mixture λ_c<λ<λ* — un vrai attracteur chaotique (type Lorenz) ?
(6) si les fronts (points fixes) existent indépendamment de τ, comment perd-on à
petit τ le SNIC qui les déstabilise ? Garde-t-on le SNIC mais le cycle meurt sur
le chaos avant de l'atteindre ? Que veut dire « mourir sur le chaos » ?

**PART A — nature du chaos** (`e23a_chaos_nature.py`, npz `E23A_lyap.npz`/
`E23A_geometry.npz`, τ=10) :
- **Exposant de Lyapunov maximal λ_L>0 et SOUTENU** dans toute la fenêtre :
  +0.050 (λ=0.29), +0.040 (0.30), +0.053 (0.31), +0.050 (0.32), +0.050 (0.325).
  Vérifié non-transitoire : à λ=0.31, λ_L fini-temps converge vers +0.048 sur
  T=3000 (queue non-fixe).
- **Spectre partiel (k=8) à λ=0.31** : {+0.035, +0.011, +0.0015, −0.003, −0.008,
  −0.022, −0.032, −0.042} → **2 exposants strictement positifs + le mode neutre**
  (tout système autonome sur une trajectoire bornée non-fixe possède un exposant
  exactement 0 — la direction du flot ; le +0.0015 en est l'estimateur fini-temps).
  **Hyperchaos** (≥2 directions dilatantes). Sommes cumulées positives jusqu'à
  l'indice 6 → **dimension de Kaplan-Yorke D_KY≈6.44**.
- **Géométrie** : PCA (var. 0.41/0.17/0.14 sur les 3 premières), section de
  Poincaré + return-map DISPERSÉ (pas une courbe 1-D), spectre de puissance
  large-bande.
- **Verdict** : attracteur chaotique **soutenu, déterministe, de dimension
  MODÉRÉE (D_KY≈6.4, hyperchaos ~3 directions instables)** — pas un simple
  attracteur de basse dimension à la Lorenz (D_KY≈2), mais pas non plus le chaos
  extensif de haute dimension d'un verre de spin. Un attracteur étrange compact
  vivant dans le sous-espace réduit P. (NB : un pré-calcul k=4 donnait D_KY≈3.1 —
  sous-estimation ; le k=8 est la valeur de référence.)

**PART B — frontière cycle-chaos & paradoxe SNIC** (`e23b_frontier.py`, npz
`E23B_frontier.npz`) :
- **λ*_front (selle-nœud de dépiégeage) = 0.328, τ-indépendant** (recalculé,
  cohérent E12/E21b).
- **B2 — Lyapunov au passage de la mort du cycle** (le point clef) : à petit τ,
  λ_L est **POSITIF DÉJÀ SUR le « cycle »** — +0.051 (τ=1, λ=0.330), +0.054 (τ=2,
  λ=0.335) — alors qu'à **τ=10 le cycle a λ_L≈0** (−0.003 à λ=0.335, −0.002 à
  0.330), vrai cycle limite régulier. Autrement dit, à petit τ le « cycle » n'est
  pas une orbite périodique propre : c'est l'état **désynchronisé multi-lobe
  d'E22, qui est lui-même faiblement CHAOTIQUE** (λ_L>0). À τ=5 tout est régulier
  (λ_L<0).
- **B3 — coexistence** : dans la bande de mort, le front piégé reste un point fixe
  **stable** (eig_max(M)<0, n_inst(T_P)=0) à tout τ pour λ<λ* (τ-indép confirmé) ;
  il devient instable seulement au-dessus de λ*=0.328 (eig_max=+0.003, n_inst=1).
  Chaos et front stable **coexistent** donc dans la bande — mais la trajectoire
  chaotique ne tombe pas sur le front (bassin infime, E20).
- **Frontière (τ,λ)** : la mort du cycle se produit près de λ*_front mais du côté
  chaos ; états post-mort mélangés selon τ (τ=1 → pinned-front ; τ=2 → chaos ;
  résolution ±0.005).
- **RÉSOLUTION DU PARADOXE** : l'« ingrédient SNIC » (le selle-nœud du front à
  λ*=0.328) est bien **toujours présent et τ-indépendant**. Ce qui disparaît à
  petit τ, ce n'est pas le selle-nœud, c'est **le cycle limite propre** qui
  pouvait rider son ghost : à τ<τ_c≈1.3 (E22) le « cycle » est un état
  désynchronisé **déjà chaotique** (λ_L>0), qui se fond dans la mer chaotique au
  lieu de se poser proprement sur le front. « Mourir sur le chaos » = le cycle,
  déjà faiblement chaotique, perd sa cohérence et rejoint l'attracteur chaotique
  ambiant, sans jamais réaliser le ralentissement déterministe √ du selle-nœud
  (d'où la destruction de la loi √ en E19c). Le front, lui, reste un point fixe
  stable coexistant mais quasi-inaccessible (bassin infime).
**Lien inter-expériences :** E22 (désync à τ_c≈1.3) + E23B (λ_L>0 sur le
« cycle » à petit τ) + E19 (perte de la loi √) forment une seule histoire :
sous τ_c le front délocalisé est chaotique, ce qui casse simultanément la
cohérence spatiale, la statistique d'extrêmes et le scénario SNIC.
**Figures :** `figures/figQ_chaos_nature.png` (spectre Lyapunov+D_KY, convergence
λ_L, λ_L(λ), attracteur PCA, return-map, spectre de puissance),
`figures/figR_cycle_chaos_frontier.png` (carte (τ,λ) cycle/chaos/front, λ*_front
τ-indép, saut de λ_L à la mort). Codes `e23a_chaos_nature.py`, `e23b_frontier.py`.
NB : calcul par l'orchestrateur (agent bloqué sur permission Bash).

### E24 — Statistique des seuils par motif λ_c(μ) : forme des branches, taille finie, valeurs extrêmes — FAIT (2026-07-05) — **figU, figV**
**Motivation (questions utilisateur a/b/c) :** après l'unification §5.7 du REPORT
statique (chaque motif ξ^μ = une branche mémoire→front mourant à son selle-nœud gelé
λ_c(μ) ; λ*=max_μ), trois questions : (a) le passage motif→mélange est-il lisse ou
y a-t-il un « genou » λ_i(μ)<λ_c(μ) (définition de la fin du rappel) ; (b) l'écart
max−min des seuils est-il un effet de taille finie (→0 quand N→∞ ?) ; (c) la
distribution des λ_c(μ) est-elle caractérisable et le « gap » prédictible par les
valeurs extrêmes ?
**Méthode** (`e24_thresholds.py`) : pour CHAQUE motif, continuation Newton–Woodbury
warm-start vers le haut avec pas adaptatif ~eig², bisection finale (précision 2e−4,
sous la rugosité jumelle ±1.5e−3 documentée E21d qui borne le sens de λ_c),
extrapolation spectrale eig²(λ) linéaire en contrôle croisé (écart médian 4e−4 à
7e−3 selon N). Parallélisé sur les motifs. **Validation** : à N=2000/seed 42, motif 1
→ 0.2821 (=λ_c statique 0.2822 ✓), motif 91 → 0.3276 (=λ* ✓) **et argmax des 100
seuils ✓** — λ*=max_μ λ_c(μ) vérifié directement.
**(a) E24a — départ LISSE, pas de genou** (`e24a_geometry.py`, courbes complètes des
100 branches, npz `E24_curves_N2000_s42.npz`) : l'inclinaison a_{μ+1}/a_μ croît
régulièrement dès λ=0.05 ; la **moitié de l'inclinaison finale est acquise à
x₅₀=0.64·λ_c(μ)** (IQR [0.61, 0.66] — un départ brutal donnerait x₅₀≈1) ; ~45 % de
l'inclinaison déjà là à λ_c/2 ; rescalées par λ_c(μ) les 100 courbes **collapsent sur
une forme universelle** (le désordre ne fixe QUE le seuil) ; seul régime rapide : la
singularité √ générique du fold (a_μ = a_c + k√(λ_c−λ), R²=0.93–0.98 sur 5 branches
percentiles). → **La fin de la phase de rappel complet = min_μ λ_c(μ)** ; à
N=2000/seed 42 : **min=0.2195 (motif 28)** ≪ 0.2822 (ξ¹ n'était qu'une sonde) ;
distribution sur [0.220, 0.328].
**E24e — contrôle stabilité retardée** (`e24e_spotchecks.py`, npz
`E24e_spotchecks.npz`) : 30 sondes T_P (τ=10) sur 5 branches percentiles
{min, 25, 50, 75, max} : **0 point instable** — partout, l'existence de la branche
implique sa stabilité DDE jusqu'au fold (généralise E18b/E21c à toute la distribution).
**(b)/(c) E24b-d — taille finie et valeurs extrêmes** (`e24_analysis.py`, 38 runs,
npz `E24_thr_N*_P*_s*.npz`) :
| N (α=0.05) | P | seeds | médiane | σ | gap=max−min | λ*=max | max prédit EVT |
|---|---|---|---|---|---|---|---|
| 500 | 25 | 5 | 0.3080 | 0.0341 | 0.131±0.029 | 0.3589 | 0.3647 |
| 1000 | 50 | 5 | 0.2803 | 0.0319 | 0.159±0.028 | 0.3471 | 0.3476 |
| 2000 | 100 | 5 | 0.2742 | 0.0229 | 0.114±0.014 | 0.3254 | 0.3276 |
| 4000 | 200 | 5 | 0.2691 | 0.0175 | 0.095±0.005 | 0.3048 | 0.3128 |
| 8000 | 400 | 3 | 0.2674 | 0.0124 | 0.079±0.009 | 0.2956 | 0.3015 |
Série à P=100 fixé (isole la fluctuation par seuil) : σ = 0.0229 / 0.0135 / 0.0090 /
0.0064 pour N = 2000 / 4000 / 8000 / 16000, pente log-log **−0.61** (∼CLT −0.5, un
peu plus rapide car α=100/N décroît aussi → paysage plus propre). À α fixé : pente
σ(N) −0.38, pente gap(N) **−0.22** (σ·√(2 ln αN) : le facteur d'extrêmes croissant
ralentit la décroissance).
**Réponses :**
- **(b) OUI, effet de taille finie** : le gap décroît de façon monotone dès N≥1000
  (0.159→0.079) et → 0 quand N→∞, mais LENTEMENT (≈N^{−0.22} effectif, σ~N^{−0.4}
  contre un facteur d'extrêmes √(2 ln P) croissant). Dans la limite thermodynamique
  tous les seuils fusionnent en un λ_c^∞ unique : « mort de la mémoire » et
  « naissance du cycle » deviennent UNE transition ; à N physique (2000) la bande
  large de 0.11 est bien réelle et auto-moyennage absent.
- **(c) OUI, distribution caractérisable et gap prédictible** : λ_c(μ) est
  quasi-gaussienne avec asymétrie gauche modérée (skew≈−0.5, kurt≈0.5 stables en N ;
  collapse des CDF standardisées, figV-b). La prédiction valeurs-extrêmes
  gaussiennes-iid **λ*−médiane ≈ σ(N)·a_P** (a_P=√(2lnP)−(lnlnP+ln4π)/(2√(2lnP)))
  reproduit le max observé à 0.002–0.008 près À TOUS les N (léger sur-comptage,
  cohérent avec le skew négatif). Le « gap théorique » demandé s'écrit donc
  **gap(N) ≈ σ(N)·√(2 ln αN)** avec σ(N) mesuré (ou σ ≈ σ₀N^{−0.4}).
**Figures :** `figures/figU_branch_geometry.png` (100 courbes d'inclinaison + collapse
+ détecteur de genou + test √), `figures/figV_threshold_distribution.png`
(histogrammes vs N, CDFs standardisées vs normale, σ(N) deux séries, gap vs EVT).
Codes `e24_thresholds.py`, `e24a_geometry.py`, `e24_analysis.py`, `e24e_spotchecks.py`.
**Caveats :** λ_c(μ) opérationnel (limite Newton warm-start, ±2e−4) sous la rugosité
jumelle ±1.5e−3 (E21d) ; N=500 bruité (P=25 seulement) ; pente −0.38 de σ à α fixé à
consolider (contamination possible des petits N).

### E25 — La forme au fold dépend-elle de μ ? (question utilisateur 3) — FAIT (2026-07-06) — **figW**
**Motivation :** E13 (4 seeds) rapportait une forme de front figé quasi-universelle
~[0.68, 0.32] ; le panneau √ de figU suggérait au contraire une dépendance en μ.
**Protocole** (`e25_deathshape.py`) : depuis les courbes complètes des 100 branches
(npz `E24_curves_N2000_s42.npz`), extraction du point de fold (a_μ^c, a_{μ+1}^c,
ratio r_c) de CHAQUE branche ; corrélations avec λ_c(μ) et avec le désordre gelé de
liaison q_μ=(1/N)ξ^μ·ξ^{μ+1} ; en contrôle, autocorrélation d'anneau des λ_c(μ)
(12 runs multi-seed).
**Résultats :**
1. **La forme au fold N'EST PAS universelle** : r_c ∈ [0.285, 0.489] (médiane 0.383),
   a_μ^c ∈ [0.664, 0.773]. **E13 réconcilié** : le « 68/32 » est exactement la forme
   du maillon EXTRÊME (μ=91 : (0.664, 0.324), r=0.489) — E13 ne mesurait que le
   maillon le plus faible de chaque seed, donc conditionnait sur max_μ λ_c(μ).
2. **La forme est essentiellement une fonction de la POSITION de la mort** :
   corr(r_c, λ_c(μ)) = **+0.81** (Pearson, p=7e−25) — les branches qui meurent plus
   haut en λ meurent plus inclinées (la première morte, μ=28 à 0.2195 : r=0.35 ;
   l'extrême, μ=91 à 0.3276 : r=0.49). Famille à un paramètre dans le plan
   (a_μ, a_{μ+1}), ordonnée par λ_c (figW-c). Le collapse universel de figU (E24a)
   porte sur la TRAJECTOIRE r(λ/λ_c) ; le point final r_c(λ_c) suit, lui, la même
   fonction universelle r(λ) évaluée à λ_c(μ) — cohérent.
3. **Prédicteur microscopique** : q_μ corrèle avec λ_c(μ) (−0.48 : liaisons plus
   alignées → seuil plus BAS) et, à λ_c fixé, q_μ explique quasi toute la dispersion
   résiduelle de la forme : **corr partielle (q, r_c | λ_c) = +0.95**. Le couple
   (λ_c(μ), q_μ) détermine donc la forme de mort presque déterministiquement.
4. **Indépendance (contrôle pour la tâche 1)** : autocorrélation lag-1 des λ_c(μ)
   NON robuste — seed 42 : −0.34 (hors bande iid ±0.20) mais moyenne des 12 runs
   **−0.077**, tous les runs N=4000/8000 DANS la bande : **quasi-iid**, tendance
   négative faible décroissant avec N (cohérent avec le succès de l'EVT iid en E24d).
**Figures :** `figures/figW_deathshape.png` (4 panneaux : r_c vs λ_c coloré q,
q vs λ_c, plan (a_μ,a_{μ+1}), lag-1 multi-seed). Npz `E25_deathshape.npz`. Code
`e25_deathshape.py`.
**Vérification N=10⁴ (2026-07-06, demande utilisateur)** — 150 branches à N=10⁴,
α=0.05 (npz `E24_curves_N10000_s42.npz`, fig `figW_deathshape_N10000.png`) :
TOUS les verdicts robustes — r_c↔λ_c : **+0.75** (vs +0.81) ; q↔λ_c : **−0.485**
(vs −0.48, identique) ; corrélation partielle (q, r_c | λ_c) : **+0.90** (vs
+0.95) ; maillon extrême (0.706, 0.292), première mort (0.740, 0.252) — même
famille à un paramètre. Et le point ambigu est TRANCHÉ : **lag-1 = −0.009**
(bande iid ±0.08) — les seuils sont bien iid à N=10⁴ ; le −0.34 de N=2000/seed 42
était une fluctuation de taille finie.

### E26 — La loi des seuils dépend-elle de P/α ? (question utilisateur 1) — FAIT (2026-07-06) — **figZ**
**Motivation :** existe-t-il une loi de probabilité FIXE, indépendante de P, dont les
P seuils λ_c(μ) seraient des tirages iid — ou la loi elle-même dépend-elle de la
charge ? Test : distribution des λ_c(μ) en fonction de α à N=10⁴ fixé.
**Protocole** (`e24_thresholds.py` étendu : options `--subset` et `--no-eig` pour
P grand — le Woodbury coûte O(P²N)/itération) : α ∈ {0.01, 0.02, 0.05} × 2 seeds
(P=100/200 : toutes les branches ; P=500 : sous-échantillon aléatoire de 100),
plus α ∈ {0.08, 0.10} × 1 seed (sous-échantillons 80/60). Analyse
`e26_analysis.py`.
**Résultats (α ≤ 0.05, régime propre) :**
| α | P | n_seuils | médiane | σ | skew | kurt |
|---|---|---|---|---|---|---|
| 0.01 | 100 | 200 | 0.3815 | 0.0076 | 0.00 | −0.13 |
| 0.02 | 200 | 400 | 0.3458 | 0.0081 | −0.03 | −0.17 |
| 0.05 | 500 | 200 | 0.2705 | 0.0094 | −0.15 | 0.00 |
1. **La loi n'est PAS fixe** : son CENTRE suit fortement la charge (médiane
   0.382→0.271 de α=0.01 à 0.05), en parallèle exact de la courbe statique
   λ_c(α) du §5.3 (sonde ξ¹ : 0.379/0.314/0.263 à α=0.01/0.03/0.05) —
   c'est l'interférence moyenne entre motifs.
2. **Mais sa FORME standardisée est universelle** : les CDFs de (λ_c−méd)/σ
   collapsent à travers α — tests KS deux-à-deux TOUS compatibles (p=0.85, 0.87,
   0.25). Et sa LARGEUR dépend à peine de α à N fixé (σ : 0.0076→0.0094, +24 %
   pour α ×5), alors qu'elle dépend fortement de N (E24 : σ=0.0229 à N=2000,
   α=0.05 → 0.0094 à N=10⁴).
3. **Bilan (réponse à la question)** : pas de loi unique indépendante de P, mais une
   **famille position-échelle à forme universelle** :
   **λ_c(μ) ≈ m(α) + σ(N,α)·X**, X variable standardisée universelle
   (quasi-gaussienne symétrique à N=10⁴ — le skew −0.5 vu à N=2000 était un effet
   de taille finie), m(α) la courbe d'interférence, σ dominé par N (≈N^{−0.4..−0.6})
   avec dépendance en α faible. Combiné au quasi-iid d'E25 (lag-1 ≈ −0.08) : les P
   motifs tirent bien leurs seuils (presque) indépendamment dans cette loi — ce qui
   justifie a posteriori le succès de la prédiction EVT-iid d'E24d.
4. **Breakdown near-capacity (α ≥ 0.08, N=10⁴)** : le traceur s'effondre — 57/80
   (α=0.08) et 19/60 (α=0.10) branches seulement, avec des « λ_c » anormalement bas
   (médianes 0.11/0.067) qui marquent une **perte d'identité de branche**
   (condensation d'un 3ᵉ overlap avant tout fold), PAS des selles-nœuds. La notion
   même de seuil-par-motif se dissout à l'approche de la capacité — version
   par-motif du breakdown §5.4 du REPORT statique. Exclus de l'analyse de loi.
**Figures :** `figures/figZ_alpha_law.png` (histogrammes vs α, centre vs λ_c(α)
statique + σ jumelé, collapse des CDFs standardisées, carte σ(α) vs σ(N)). Npz
`E26_thr_N10000_P*_s*.npz` (8). Codes `e24_thresholds.py` (--subset/--no-eig),
`e26_analysis.py`.

### E27 — Motifs corrélés I : vidéo par CHAÎNE DE MARKOV de flips (question utilisateur 2) — FAIT (2026-07-06) — **figX**
**Motivation :** le schéma de bifurcation (mémoires inclinées mourant chacune à son
seuil, cycle né une fois toutes les digues sautées) survit-il quand les motifs ne
sont plus iid mais forment une « vidéo cohérente » (ξ^μ et ξ^{μ+1} corrélés) ?
**Modèle :** ξ^{μ+1}_i = ξ^μ_i·s_i, flips iid P(s=−1)=(1−c)/2 → corrélation entre
trames consécutives c ∈ {0, 0.2, 0.4, 0.6, 0.8} (c=0 = contrôle iid). L'anneau ne
se referme PAS : la **couture** (liaison 99→0) a q≈0 au milieu de liaisons à q≈c —
traitée séparément (prédiction E25 : q↑⇒λ_c↓, donc la couture devrait devenir la
digue MAXIMALE). N=2000, P=100, seed 42, τ=10.
**Méthode :** traceur E24 en mode « relaxed » (identité de branche = motif dominant
+ continuité warm-start ; le critère iid — successeur en 2ᵉ, 3ᵉ overlap <0.15 —
rejette à tort les mémoires corrélées, naturellement « habillées » par leurs
voisines). Contrôle dynamique : qualité du cycle à λ=0.9 pour chaque c (fraction
de relais vers l'avant) + descente ensemencée depuis λ=0.85 (pas 0.01) à c=0.6.
**Résultats (seuils) :**
| c | q_bulk | tracées | méd. bulk | [min, max] bulk | σ | couture λ_c | corr(q,λ_c) |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.00 | 100/100 | 0.2733 | [0.225, 0.329] | 0.022 | 0.237 (pas max) | −0.28 |
| 0.2 | 0.20 | 65/100 | 0.0815 | [0.052, 0.141] | 0.027 | **0.2066 (MAX de tous)** | −0.39 |
| 0.4–0.8 | 0.4–0.8 | **0/100** | — | — | — | — | — |
1. **c=0.2 : prédiction E25 confirmée spectaculairement.** Les seuils du bulk
   s'effondrent (médiane 0.273→0.082 : des voisins corrélés « tirent » la mémoire
   bien plus tôt), tandis que la couture — seule liaison non corrélée — garde un
   seuil élevé (0.207) et devient **la digue maximale** : dans une vidéo Markov,
   c'est à la couture que le cycle mourrait.
2. **c≥0.4 : les mémoires individuelles N'EXISTENT PLUS.** Diagnostic direct
   (`/tmp/diag_c06.py`, à refaire proprement si besoin) : Newton depuis 2ξ^μ ne
   converge plus (res~0.05–0.1) et l'état approché est un **paquet symétrique étalé
   sur ~5 motifs voisins** (à c=0.6 : a≈[0.57, 0.53, 0.52, 0.50, 0.46] sur
   {μ, μ±1, μ±2}) — le paysage devient celui d'un **réseau à attracteur
   quasi-continu par paquets**, plus un Hopfield à mémoires discrètes.
3. **Dynamique :** cycle de rappel **parfait** à λ=0.9 pour c=0 et 0.2 (fraction
   avant 1.00), **dégradé/détruit dès c≥0.4** (0.65/0.50/0.60 — errance
   avant-arrière). Descente à c=0.6 : « épinglé » dès λ=0.83 sur une paire bulk
   [65,66] — cohérent avec l'absence de cycle propre : à forte corrélation locale
   SANS structure globale lisse (marche aléatoire de trames + couture), le rappel
   séquentiel lui-même est perdu.
**Conclusion :** le schéma « digues par motif + SNIC au maillon fort » **survit à
corrélation modérée** (c≲0.2) avec seuils fortement abaissés et la couture en
dernière digue ; à corrélation forte il est remplacé par une phase de paquets
délocalisés sans rappel séquentiel. **Figures :** `figures/figX_markov_video.png`.
Npz `E27_markov.npz`. Code `e27_markov_thresholds.py`.
**Leçon méthodologique (coûteuse) :** premier lancement bloqué 11 h dans un
deadlock silencieux — scripts multiprocessing SANS garde `__main__` sur macOS
(spawn ré-importe le module dans chaque worker → Pool dans un process démon →
workers morts, parent en attente infinie). Garde ajoutée partout (`fix_guard.py`) ;
à respecter pour tout futur script à Pool.
**Vérification N=10⁴ (2026-07-06, demande utilisateur)** — α=0.05 (P=500 trames),
sous-échantillon 100 branches + couture (npz `E27_markov_N10000.npz`, fig
`figX_markov_video_N10000.png`) :
| verdict | N=2000 | N=10⁴ | robuste ? |
|---|---|---|---|
| contrôle c=0 | méd 0.273, σ=0.022 | méd 0.269, σ=0.009 | ✓ (≡E24/E26) |
| c=0.2 : effondrement bulk | méd 0.082 | méd 0.062 | ✓ (plus net) |
| c=0.2 : couture = digue MAX | 0.207 | **0.169, max de tous** | ✓ |
| c≥0.4 : fonte des mémoires | 0/100 | 0/500 | ✓ |
| corr(q,λ_c) bulk | −0.28/−0.39 | **−0.55** | ✓ renforcée |
| cycle λ=0.9 à c=0.4 | dégradé (0.65) | **TOURNE (0.91)** | ⚠ raffiné |
| cycle c=0.6/0.8 | détruit | détruit (0.57/0.43) | ✓ |
Seule correction de taille finie : la frontière de destruction du cycle se déplace
de c∈(0.2, 0.4) à **c∈(0.4, 0.6)** — le cycle tolère plus de corrélation à grand N
(le bruit gelé relatif qui aide à désynchroniser diminue). Descente à c=0.6 :
épinglage immédiat (λ=0.77) sur une paire bulk — cohérent avec l'absence de cycle
propre à ce c.

### E28 — Motifs corrélés II : vidéo par PROCESSUS GAUSSIEN sur le cercle (question utilisateur 2) — FAIT (2026-07-06) — **figY**
**Modèle :** z_i(θ)=Σ_{k=1}^{K}[a_ik cos kθ + b_ik sin kθ], a,b~N(0,1) iid ;
ξ^μ_i = sign(z_i(2πμ/P)). Vidéo **exactement périodique** (aucune couture), lissité
réglée par K (petit K = vidéo lente et lisse) ; corrélation entre trames
q₁=(2/π)arcsin(ρ(2π/P)) vérifiée à ±0.003 (mesuré vs prédit). K ∈ {80, 40, 20,
10, 5} → q₁ ∈ {−0.12, +0.13, +0.53, +0.75, +0.87}. N=2000, P=100, seed 42, τ=10.
**Résultats (seuils, mode relaxed) :**
| K | q₁ | tracées | méd. | [min, max] | σ | corr(q,λ_c) |
|---|---|---|---|---|---|---|
| 80 | −0.12 | 100/100 | 0.2595 | [0.212, 0.308] | 0.022 | −0.28 |
| 40 | +0.13 | 100/100 | 0.2285 | [0.108, 0.285] | 0.035 | −0.43 |
| 20 | +0.53 | 51/100 | 0.0768 | [0.051, 0.138] | 0.022 | −0.23 |
| 10 | +0.75 | 11/100 | 0.0603 | — | — | — |
| 5 | +0.87 | 5/100 | — | — | — | — |
Comme E27 : les mémoires individuelles fondent quand la vidéo devient lisse
(paquets délocalisés pour K≲20) ; à K=40 la distribution développe une lourde
queue gauche (min 0.108) — les liaisons les plus corrélées meurent très tôt.
**MAIS le contraste dynamique est total :**
1. **Le cycle de rappel tourne PARFAITEMENT à tous les K** (λ=0.9 : 35–37 relais,
   fraction avant 1.00 — même à K=5 où quasi aucune mémoire individuelle
   n'existe).
2. **Descente ensemencée à K=10 : AUCUN épinglage jusqu'à λ=0.20** — le cycle
   survit bien EN DESSOUS du λ*≈0.33 iid et de tous les seuils statiques tracés.
   **La digue de dépiégeage a disparu.**
**Interprétation :** avec une corrélation lisse et périodique, les P motifs
échantillonnent une variété continue de basse dimension (2K modes latents) — le
réseau devient un **quasi-attracteur continu en anneau** : l'onde de rappel est un
paquet qui TOURNE continûment, poussé par le délai, sans paysage gelé rugueux pour
l'épingler (fluctuations de liaison sd(q)≈0.014 seulement). La statistique
d'extrêmes des digues — cœur du scénario SNIC iid — exige du désordre gelé
hétérogène ; une vidéo lisse l'élimine. **Contraste E27/E28 :** corrélation locale
sans structure globale (Markov) détruit le cycle ; corrélation globale lisse (GP
cercle) le renforce. Pour encoder une séquence robuste, mieux vaut une trajectoire
lisse sur une variété qu'une chaîne de trames localement corrélées.
**Figures :** `figures/figY_gpcircle_video.png`. Npz `E28_gpcircle.npz`. Code
`e28_gpcircle_thresholds.py`.
**Vérification N=10⁴ (2026-07-06, demande utilisateur)** — α=0.05 (P=500), K∝P
pour matcher les q₁ (q₁ mesuré ≡ prédit à ±0.001 aux 5 niveaux), sous-échantillon
100 branches (npz `E28_gpcircle_N10000.npz`, fig `figY_gpcircle_video_N10000.png`) :
| verdict | N=2000 | N=10⁴ | robuste ? |
|---|---|---|---|
| seuils q₁≈−0.12 / +0.15 | méd 0.259 / 0.229 | méd 0.256 / 0.231 | ✓ |
| fonte à q₁≥0.54 | 51/100 puis ~0 | 30/500 puis ~0 | ✓ |
| **cycle parfait à TOUS les K** | fwd 1.00 partout | fwd 1.00 partout (même q₁=+0.88) | ✓ |
| **descente : AUCUN épinglage jusqu'à λ=0.20** | K/P=0.1 | K/P=0.1 (K=50) | ✓ |
Le résultat central — vidéo lisse ⇒ quasi-attracteur continu, onde sans digue de
dépiégeage — est **indépendant de la taille**. Incident du premier passage N=10⁴ :
K_dyn=10 codé en dur (inexistant à P=500) → crash avant sauvegarde ; corrigé
(K_dyn=0.1·P) + checkpoint npz des seuils avant la phase dynamique.

### E37 — Motifs corrélés III : ce que la composante HOPFIELD MODERNE apporte (question utilisateur) — FAIT (2026-07-31) — **figE37**
**Question :** E27 avait montré que le système usuel perd le rappel séquentiel
quand les trames se corrèlent ; E35-V1 avait montré, sur UNE vidéo, que le
factoriel J×K sépare et que K=PINV est la condition du rappel ordonné. Aucun des
deux ne chiffrait l'apport. E37 croise les deux : factoriel complet contre la
corrélation réglable d'E27, aux paramètres exacts d'E27 (N=2000, P=100, β=20,
τ=10, λ=0.9), donc le bras `JH_KH` **est** la référence d'E27.
**Méthode :** lecture identique pour les quatre bras (overlap physique
`m=ξ·tanh(βu)/N` — décoder par G† seulement pour les bras modernes inscrirait la
réponse dans l'instrument). CI `a₀=e₀`, `t_total=8000`, `transient=0.4` (≈4.4
tours). Deux observables nouvelles : `selectivity` (marge du meneur au pic) et
`packet_width` (trames à plus de la moitié du pic), qui séparent « l'anneau
tourne » de « l'anneau tourne en résolvant une trame à la fois ».
**Résultat (couverture/avant, seed 42) :**
| c | JH_KH | JP_KH | JH_KP | JP_KP |
|---|---|---|---|---|
| 0.0–0.2 | **1.00/1.00** | **1.00/1.00** | **1.00/1.00** | **1.00/1.00** |
| 0.4 | 0.23/0.19 | 0.35/0.28 | **1.00/1.00** | **1.00/1.00** |
| 0.6 | 0.01/0.00 | 0.01/0.00 | **1.00/1.00** | **1.00/1.00** |
| 0.8 | 0.01/0.00 | 0.01/0.00 | **1.00/1.00** | **1.00/1.00** |
| 0.9 | 0.01/0.00 | 0.01/0.00 | 0.05/0.50 | **1.00/1.00** |
1. **La plage de fonctionnement passe de c≤0.2 à c≥0.9.** Interrupteur, pas
   gradation : aucune zone intermédiaire. **Multi-graines (42/43/44) : 3/3 sans
   exception**, à c=0.4, 0.6 et 0.9.
2. **K domine, J n'agit qu'à l'extrême** : `JH_KP ≡ JP_KP` au 4ᵉ chiffre jusqu'à
   c=0.8 ; à c=0.9, JH_KP échoue 0/3 et JP_KP réussit 3/3. Cohérent E35-V1.
3. **La mort du système usuel est un GEL daté** : `no_lead_change`, et le suivi
   fenêtre par fenêtre (`e37_arrest_in_time.py`) montre l'arrêt à **t≈1200** à
   c=0.6 (24, 16, 1, 0, 0, 0…), contre 37 changements par fenêtre indéfiniment
   pour JP_KP.
4. **Netteté dégradée mais continue pour le bras moderne** : marge au pic
   0.948→0.094 et largeur 1→13 trames de c=0 à c=0.9. La composante moderne
   restaure **l'ordre et le mouvement**, PAS la résolution individuelle — même
   séparation qu'E35-V1 (cycle parfait / reconstruction en échec).
5. **Non anticipé :** `JP_KP` donne `T=1080.7`, `cv=4.6e−05` à TOUS les c et
   TOUTES les graines (4 chiffres identiques), alors que `JH_KP` dérive
   (1078.2→1164.1). Hypothèse : G† appliqué AVANT le décalage décorrèle les
   coordonnées, donc la période ne dépend plus que de (λ,τ,β,P). **Non testée.**
**Correction apportée à E27 :** E37 donnait `forward=0.19` à c=0.4 là où E27
publiait `0.65`. Le protocole E27 rejoué reproduit `0.65/0.50/0.60` au chiffre
près, et la même trajectoire relue avec l'observable `m` donne `0.67/0.52/0.59` —
**l'observable n'explique rien**. La cause est la fenêtre : E27 analyse 400 u.t.
(0.37 période) ouvertes après 200 u.t., soit entièrement dans le transitoire qui
précède le gel. Les conclusions qualitatives d'E27 tiennent ; **les valeurs
0.65/0.50/0.60 ne doivent pas être citées comme fractions avant stationnaires**.
**Non fait :** balayage en λ des bras modernes (interrompu à la demande
utilisateur — seuls les 16 cellules à K hebbien sont mesurées, toutes en échec à
λ∈{0.35,0.4,0.5,0.7}, donc « baisser λ ne sauve pas le système usuel ») ;
vérification à N=10⁴ ; généralisation au générateur GP-cercle d'E28.
**Incident :** `Path.with_suffix` sur `c0.40_seed42_…` traite `.40_seed42_…`
comme l'extension → les 6 runs du premier factoriel ont écrit dans le même
fichier. Métriques sauvées par les logs, artefacts perdus, factoriel relancé —
**reproductibilité exacte sur les 24 cellules**, incident converti en contrôle.
**Document :** `results/6_motifs_correles/E37_MODERN_K_RESULTS.md`.
**Figures :** `figE37_modernK_doseresponse.png`, `figE37_traces_c0.60.png`,
`figE37_traces_c0.90.png`. **Code :** `e37_modern_k_markov.py`,
`e37_e27_reconciliation.py`, `e37_arrest_in_time.py`, `e37_figure_*.py`.

### E29 — Par QUOI passe le cycle juste au-dessus de λ* : motifs purs ou mélanges morts ? — FAIT (2026-07-07) — **figAA**
**Question (utilisateur) :** les points fixes rappelés se déforment continûment de
ξ^μ vers des mélanges ~70/30 (ξ^μ, ξ^{μ+1}) et meurent en selle-nœud ; le cycle né
à λ* ressent leurs fantômes. Le cycle passe-t-il alors par les motifs PURS ξ^μ, ou
par les MÉLANGES que les points fixes étaient devenus à leur mort ? S'il passe par
les motifs exacts, comment l'expliquer ?
**Protocole :** N=2000, P=100, seed 42, τ=10, β=20, réduction P-dim exacte.
Descente en λ avec warm-start 0.90→0.330 (λ*=0.32763, liaison 91), 10 valeurs. À
chaque λ : intégration jusqu'à ≥2 tours complets du gagnant argmax_μ a_μ(t)
(dt=0.01, RK4+Hermite), mesures sur le DERNIER tour : pic analogique max_t a_μ(t)
et co-overlaps au pic ; readout binaire m^sign_μ=(1/N)ξ^μ·sign(u) ; point le plus
lent de chaque fenêtre du gagnant comparé à l'état de fold (a_c, a1_c) stocké
(E25) ; période T(λ) ; profil de vitesse |ȧ| le long du tour.
**Résultats :**
| λ | T_tour | méd. pic a_μ | co a_{μ+1} au pic | min_μ m^sign au pic | frac. temps sign(u)=ξ^μ exact |
|---|---|---|---|---|---|
| 0.900 | 1088 | 1.000 | −0.002 | 0.9990 | 0.853 |
| 0.500 | 1226 | 1.000 | −0.002 | 1.0000 | 0.750 |
| 0.370 | 1512 | 0.998 | −0.001 | 1.0000 | 0.587 |
| 0.340 | 1729 | 0.995 | +0.000 | 1.0000 | 0.507 |
| 0.335 | 1800 | 0.994 | +0.001 | 1.0000 | 0.491 |
| 0.332 | 1845 | 0.993 | +0.002 | 1.0000 | 0.479 |
| 0.330 | 1879 | 0.992 | +0.002 | 1.0000 | 0.469 |
1. **Le cycle passe par les motifs quasi PURS, pas par les mélanges 70/30** : même
   à λ*+0.002, le pic analogique vaut 0.992 (min sur les 100 : 0.978), co-overlap
   du successeur ≈ 0 au pic — très loin des états de fold (a_c méd. 0.718,
   a1_c méd. 0.278). Le readout binarisé sign(u) atteint EXACTEMENT ξ^μ pour
   chacun des 100 motifs (min m^sign=1.0000 dès λ≤0.7), et reste exactement sur
   un motif pendant 47–85 % du tour.
2. **MAIS il enfile aussi tous les états de fold en transit** : à λ=0.330, la
   distance minimale de la trajectoire à l'état de fold de sa liaison vaut 0.004
   (liaison 91), 0.005 (43), 0.010 (28) — chaque relais μ→μ+1 traverse la région
   du plan (a_μ, a_{μ+1}) où le point fixe est mort.
3. **Le fantôme SNIC n'est ressenti QUE par la liaison critique** : le point le
   plus lent de la fenêtre 91 coïncide avec son état de fold à 0.006 près
   ((0.661, 0.330) vs (0.664, 0.324)), vitesse minimale 2.6e−3, séjour 41.4 t.u.
   contre 18.1 (médiane) — ×2.3. Pour les autres liaisons, le point lent de la
   fenêtre est le plateau de consolidation au motif pur (leur fold est loin,
   λ−λ_c(μ) grand). T(λ) croît (1088→1879) sans divergence visible à δ=0.0024 —
   le goulot (41 t.u.) reste une petite fraction du tour ; la divergence SNIC
   n'émerge qu'à δ≪1e−3 (coefficients raides, β=20).
**Explication de la pureté (résolution du paradoxe) :** l'inclinaison 70/30 du
point fixe statique vient de sa PROPRE image retardée : à l'équilibre
a(t−τ)=a(t), donc le terme λ(Sm)(a) pousse en permanence vers ξ^{μ+1} — le point
fixe DOIT porter ce tilt. Sur le cycle, au moment du pic de a_μ, l'état retardé
est encore dominé par μ−1 (le pic survient dans les premiers ~τ du règne de μ) :
le drive retardé λ(Sm)(a(t−τ)) ≈ λ e_μ pousse vers μ LUI-MÊME, et le drive total
(1−λ)e_μ + λe_μ = e_μ est celui d'un Hopfield pur → l'état se purifie vers ξ^μ.
Chaque règne a deux phases : **consolidation** (image retardée = μ−1, drive
aligné, purification — d'où T₁≈τ+ε par liaison à grand λ) puis **poussée** (image
retardée = μ, drive incliné vers μ+1, l'état glisse à travers la région de fold
vers le motif suivant). Le mélange 70/30 n'est visité qu'EN MOUVEMENT — sauf à la
liaison critique où le passage rampe sur le fantôme. Le rappel séquentiel est
donc exact au sens de Hopfield (sign(u)=ξ^μ) précisément PARCE QUE le retard
réaligne le drive : l'état voyageur échappe au tilt que subissait le point fixe.
**Figures :** `figures/figAA_cycle_vs_ghosts.png`. Npz `E29_cycle_ghosts.npz`.
Codes `e29_cycle_vs_ghosts.py` (+ `e29_fig.py` régénération/analyse d'approche).

### E30 — Campagne d'échelle des seuils λ_c(μ) : TCL confirmé, gap→0 jusqu'à N=14000 — FAIT (2026-07-08/09, nuit) — **figAE**
**Question (utilisateur, tâche 1) :** le profil gaussien des λ_c(μ) à variance
décroissante ressemble à un TCL — quelles variables sont sommées, quelle est la
vraie loi d'échelle, et le gap tend-il vraiment vers 0 ? Jusqu'où pousser N en une nuit ?
**Protocole :** 303 runs du traceur E24 (`e30_scaling_overnight.py`, resumable, une
npz par (N,P,seed), seeds 1000+, λ_c par bisection) : série **P=100 fixe**
N=2000…32000 (8–24 seeds ; N<2000 exclus, α>0.05 near-capacity) ; série **α=0.05
fixe** N=1000…14000, TOUTES les branches tracées (500/500 à N=10⁴, 700/700 à
N=14000 ; `--no-eig` pour N≥10⁴ — les solves Woodbury O(P²N) dominent, l'eig
devenait pathologique : 48 min/seed à N=10⁴ avec eig). Analyse `e30_analysis.py`.
**Résultats :**
| série | σ(N) | gap(N) |
|---|---|---|
| α=0.05 fixe | **N^{−0.508} ≈ N^{−1/2} (TCL)** | **N^{−0.42}, gap→0 confirmé** |
| P=100 fixe | N^{−0.597} | N^{−0.619} |
1. **C'est bien un TCL.** Les variables sommées : les P−1 recouvrements croisés
   o_ν=(1/N)ξ^μ·ξ^ν (gaussiens, variance 1/N) — λ_c(μ) est fonctionnelle lisse de
   cet agrégat de ~αN contributions. À α fixe σ suit N^{−1/2} (le −0.38 d'E24
   était une contamination petit-N) ; forme quasi-gaussienne (pool standardisé
   N≥4000, n=8764 : skew −0.45, kurt +1.02, KS=0.026).
2. **gap→0 vérifié** : 0.149 (N=1000) → 0.078 (8000) → 0.055 (10⁴) → **0.0498
   (N=14000)** ; pente N^{−0.42} = σ·√(2 ln αN) (prédiction E24c ✓). Médiane α-fixe
   convergée : λ_c^∞(0.05)≈0.268 dès N=4000.
3. **Plafond nocturne (réponse honnête)** : coût full-P ~N^{3.4} → N=10⁴ ≈ 90
   min/seed, N=14000 ≈ 4.8 h/seed (8 cœurs) ; **N≈14000 est le plafond réaliste**,
   N=20000 (~6 h/seed) abandonné. Mes deux premières estimations (20000 puis
   14000×3+20000×2) étaient trop optimistes — corrigées en cours de nuit (2
   réinterventions sur le ladder, documentées dans OVERNIGHT_PLAN.md).
**Figures :** `3_unification_seuils_scaling/figures/figAE_threshold_scaling.png`. Npz `3_unification_seuils_scaling/data/E30_scaling/` (303).
Codes `e30_scaling_overnight.py`, `e30_analysis.py`. REPORT statique → rev. 6 (§5.9).

### E31 — Survie des mémoires par motif f_mem(α,λ) : le breakdown de capacité est réel — FAIT (2026-07-09)
**Question (tâche 5, Exp C du design grand-α) :** la perte d'identité de branche
à α≥0.08 (E26) est-elle une vraie prolifération verre-de-spin ou un artefact du
traceur ? Sonde tracer-indépendante.
**Protocole** (`e31_fixedpoint_census.py`) : pour CHAQUE motif, Newton recuit en β
(5→20) depuis 0.99·ξ^μ à (α,λ) ; survie = point fixe macroscopique (maxov>0.7)
stable (eig_max<0). N=2000, 2 seeds. **Pivot méthodologique (smoke-test)** :
Newton depuis des IC aléatoires diffuses ne converge quasi pas (4/40) — remplacé
par le départ sur motif (robuste, 100/100 à α=0.05) ; le recensement des attracteurs
parasites est délégué à la dynamique (E32).
**Résultats — f_mem(α,λ), moyenne 2 seeds :**
| α\λ | 0.00 | 0.05 | 0.15 | 0.25 |
|---|---|---|---|---|
| 0.05 | 1.00 | 1.00 | 1.00 | 0.14 |
| 0.07 | 1.00 | 1.00 | 0.96 | 0.00 |
| 0.09 | 0.94 | 0.96 | 0.59 | 0.00 |
| 0.11 | 0.70 | 0.63 | 0.10 | 0.00 |
| 0.13 | 0.42 | 0.34 | 0.00 | 0.00 |
1. **Le breakdown est réel** (pas un artefact) : même à λ=0 (Hopfield pur), f_mem
   chute 1.00→0.42 de α=0.05 à 0.13 — l'approche de la capacité AGS (α_c≈0.138 à
   N=∞ ; taille finie ici).
2. **Mort brutale, une à une** : les survivants gardent maxov≈0.85 (0.997 à λ=0),
   n_cond=1, et restent des motifs distincts (P(q) méd 0.06–0.09) — pas
   d'affaiblissement graduel ni de condensation ; encore une statistique de survie
   par motif.
3. **Le couplage séquence coûte de la capacité** : f_mem décroît aussi en λ à α
   fixé (à α=0.11 : 0.70→0.10 de λ=0 à 0.15) — le drive λK ajoute son cross-talk.
**Npz** `E31_survival_N2000.npz`. Code `e31_fixedpoint_census.py`.

### E32 — Diagramme de phase des bassins (α,λ) : l'érosion du cycle vers la capacité — FAIT (2026-07-08, GPU) — **figAB**
**Question (tâche 5, Exp A) :** où la route mémoire→cycle dégénère-t-elle en
mémoire→chaos quand α croît ?
**Méthode — première utilisation GPU validée** (`cycle_reduced_mlx.py`,
`e32_alpha_lambda_basins.py`) : intégrateur réduit **MLX float32/Metal** batché
(K IC en colonnes). La leçon float64 est LOCALISÉE (Newton/folds sous le plancher
float32) ; la classification de bassins (sortie grossière moyennée) tolère le
float32 — **vérifié par une porte de validation** : mêmes IC en f32-GPU et
f64-CPU, fractions concordantes à 0.03 (≪3σ binomial), accord per-IC 0.85–0.95
(désaccords aux frontières chaotiques, compensés dans les fractions) ; GPU ~18×
plus rapide et tournant EN PARALLÈLE du CPU (E30). Grille α∈{0.05…0.15} ×
λ∈{0.15…0.90}, N=2000, 2 seeds × 200 IC, t=400, partition robuste
{statique/cycle/chaos/autre} (classifieur E20 grossier, seed-indépendant).
**Résultats (fraction du cycle, moyenne seeds) :**
| α\λ | 0.28 | 0.31 | 0.35 | 0.40 | 0.50 | 0.70 | 0.90 |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.12 | 0.48 | 0.80 | 0.86 | 0.81 | 0.81 | 0.47 |
| 0.09 | 0.13 | 0.14 | 0.26 | 0.48 | 0.74 | 0.60 | 0.41 |
| 0.13 | 0.00 | 0.02 | 0.07 | 0.21 | 0.54 | 0.42 | 0.38 |
| 0.15 | 0.00 | 0.00 | 0.04 | 0.08 | 0.39 | 0.35 | 0.27 |
Le bassin du cycle **s'érode fortement** (à λ=0.40 : 0.86→0.08 pour α 0.05→0.15),
son bord bas recule (λ≈0.28→λ≥0.40) et le chaos envahit le centre — mais **pas de
fermeture complète à α=0.15** (0.39 à λ=0.50) : la route mémoire→chaos est une
érosion graduelle, la fermeture se situe vers/au-delà de la capacité. **Caveat** :
le classifieur compte « cycle » = avance sur l'anneau ; il reste à vérifier par
Lyapunov (Exp B, non lancée — hors délai) si le « cycle » survivant à α=0.13–0.15
est périodique ou déjà faiblement chaotique (mécanisme E23B).
**Figures :** `figures/figAB_alpha_lambda_basins.png`. Npz `E32_basins_N2000_mlx.npz`.
Codes `cycle_reduced_batch.py` (batché f64, validé 1e-13), `cycle_reduced_mlx.py`
(GPU f32, validé temps-court 8e-5), `e32_alpha_lambda_basins.py`, `e32_analysis.py`.

### E33 — Vieillissement du chaos de mixture : NON — mesure stationnaire (SRB), jusqu'à la capacité — FAIT (2026-07-08, GPU) — **figAC, figAD**
**Question (utilisateur, tâche 3) :** le chaos intermédiaire vieillit-il (analogie
verre de spin) ? Lien avec l'existence d'une mesure invariante type SRB.
**Cadrage (design du sous-agent, corrigé)** : « pas de mesure invariante ⇒
vieillissement » est inversé — un attracteur borné d'un flot autonome porte
génériquement une mesure SRB, et c'est précisément l'état SANS vieillissement.
Le test définitionnel est la corrélation à deux temps C(t_w+t,t_w) : collapse en
t_w ⇒ stationnaire ; éventail avec collapse en t/t_w ⇒ aging.
**Protocole** (`e33_aging.py`, GPU MLX float32, ensembles 64 IC × 4 seeds, settle
400, T=6000, masque « resté chaotique » pour éliminer les échappées vers
cycle/mémoire) : C(t_w+t,t_w) pour t_w∈{0,50,200,800,3200} ; relaxation post-quench
⟨|a|²⟩(t) sans settle (τ_r, tueur de confusion transitoire) ; extension Exp E2 :
α∈{0.08,0.10,0.12}.
**Résultats :**
1. **α=0.05 (λ=0.31) : STATIONNAIRE, pas de vieillissement.** Les C normalisées se
   superposent à travers t_w (écart 0.009 à lag~20 ; écart relatif de C(t_w,0) :
   0.002) et NE collapsent PAS en t/t_w. τ_r≈2 : le mode quasi-marginal
   λ₄≈−0.003 (E23A) ne crée AUCUN transitoire lent. figAC.
2. **Pas d'aging émergent vers la capacité** : écarts 0.021 (α=0.08, n=112) et
   0.025 (α=0.10, n=41), tous ≪ seuil 0.1 ; à α=0.12 l'ensemble chaotique à λ=0.31
   s'évanouit (n=11, intestable — informatif en soi : près de la capacité λ=0.31
   n'est plus la mer chaotique). figAD.
**Verdict :** le chaos de la fenêtre de mixture est un **attracteur étrange
stationnaire à mesure SRB** — il ne vieillit pas, contrairement à la dynamique
thermique du verre de spin de Hopfield ; l'analogie s'arrête à la géométrie du
paysage. **Caveat** : float32/GPU, porte de validation float64 (`--validate`)
non exécutée (fenêtre de temps) — le signal (0.009) est 400× au-dessus du bruit
float32 attendu, le verdict est robuste ; la porte reste à passer pour la forme.
**Figures :** `figures/figAC_aging_twotime.png`, `figAD_aging_vs_alpha.png`.
Npz `E33_aging_mlx[_a0p08/0p10/0p12].npz`. Codes `e33_aging.py`, `e33_analysis.py`,
`e33_vs_alpha.py`.

### E34 (suite) — Gate ET0 : edge-tracking des selles sur les sentinelles N=400 — NO-GO du gate, topologie dynamique INTACTE — FAIT (2026-07-24) — **figE34_ET0**
**Contexte :** le pilote E34 (N=400, P=20, seed 42 ; `E34_RESULTS.md`) avait donné
19/20 folds spectraux (μ=17 indéterminé), le gate local GO pour {14, 8, 2}, et le
NO-GO du « collier simple » par continuation longue (9–11 retournements pour
μ=8/μ=2 à λ_common). Suite sanctionnée (R2.12 rév. 3) : gate exploratoire
conditionnel **ET0** — edge-tracking à λ fixé, protocole gelé AVANT calcul
(`roadmaps/E34_ET0_PROTOCOL.md`), λ_local(μ)=λ_c(μ)−0.020.
**Protocole exécuté** (`src/e34b_edge_tracking.py`, float64, API hist+dhist) :
par sentinelle, (a) deux bassins adjacents ; (b) bissection de bord en métrique
de Gram d_G (2 segments × dt∈{0.01, 0.005}) ; (c) polissage Newton de l'objet de
bord ; (d) certification spectrale (principe de l'argument, doublement de grille,
déplacement de contour ; cross-checks racine réelle + eigmax_M) ; (e)
continuation en λ jusqu'au fold local avec comptage de retournements ; (f) deux
branches de W^u par historique exponentiel (ε∈{1e−4,1e−3} × dt/2), ω-limites.
Smoke 67 s → production 379 s (annoncé ≪ 4 h).
**Résultats :**
| μ | bord convergé | résidu | (n_inst) | z_u | rejoint le fold ? / tours | ω(−,+) | verdict |
|---|---|---|---|---|---|---|---|
| 14 | oui (accord 8.8e−14) | 1.3e−14 | (0,1) | 0.523 | OUI, 1 tour, écart 3.7e−5 | node14 / node15 (4/4) | **positif** |
| 8 | oui (accord 3.4e−16) | 2.0e−16 | (0,1) | 0.656 | NON : 7 tours (multifold) ; fold terminal 0.288448 = table à 1.15e−5 | node8 / node9 (4/4) | **limité** |
| 2 | — (seul nœud vivant à λ_local=0.3299 : pas de voisin à bracketer) | — | — | — | — | — | **indéterminé (basin_ambiguous)** |
**GATE ET0 = NO-GO** (règle gelée : GO ⇔ 3/3 positifs). Anomalies : μ=17 reste
`arclength_spectral_fit_failed` après resserrage ; μ=3 multifold confirmé
(7 retournements, fold_cont 0.2622 = census, ≠ table 0.2429).
**Lecture scientifique (la dissociation μ=8, le fait marquant) :** la topologie
dynamique à λ FIXÉ exigée par l'hypothèse-collier — nœud → selle d'indice 1 →
nœud adjacent, trouvée par pur suivi de frontière de bassins — est PROPRE sur
2/2 sentinelles testables (l'objet de bord coïncide avec la selle d'arclength à
d_G≤5e−13). Ce qui échoue est uniquement le critère d'identité de BRANCHE en λ
(le multifold est une propriété de la géométrie de branche, pas du flot à λ
fixé). Conséquences : ET1 (N=2000) reste verrouillé ; le graphe multifold reste
l'objet de repli ; « forte évidence numérique » (N=400, seed 42, 2 sentinelles)
pour la chaîne hétérocline à λ fixé — rien de plus. Amendement candidat ET0′
(NON exécuté) : bracketer μ=2 contre le bassin du cycle, ou edge-tracker à λ
plus bas puis continuer la selle vers le haut.
**Fichiers :** `2_cycle_rappel_snic/data/E34/E34_ET0_{RESULTS.md, results/summary/log _20260724T124822_763937.*}` ;
figure `2_cycle_rappel_snic/figures/E34/figE34_ET0_edge_tracking.png` ; codes
`src/e34b_edge_tracking.py`, `src/e34_ET0_figures.py` ; protocole
`roadmaps/E34_ET0_PROTOCOL.md`. NB méthodo : le mode de tir W^u doit être le
vecteur nul VARIATIONNEL (S·Dm), pas celui de T_P (écart ~3 % sinon) ;
l'historique exponentiel se reconstruit à chaque dt.
**Amendement ET0′ (même jour, protocole amendé daté, 330 s)** — μ=2 refait à
λ′=0.2229 (= λ_c(3)−0.02, nœud 3 vivant, sep d_G=1.167) : edge-tracking
machine-propre (résidu 3.1e−16, indice 1, z_u=0.537, gate linéaire 2e−4), MAIS
(i) l'objet de bord ≠ selle d'arclength (d_G=0.159) et (ii) la branche W^u « + »
n'atteint PAS le nœud 3 : elle tombe sur un **état mixture stable 3–4 motifs**
(|a| dominants : 18/14/19/0 ≈ 0.39/0.31/0.31/0.25 ; poli Newton 2.2e−16,
eigmax_M=−0.054) — un état parasite de type Hopfield, cohérent avec la richesse
des bassins (E20/E23A). Continuation montante : atteint le fold table
λ_c(2)=0.3499 à 3.6e−6 mais via **17 retournements**. **Scoreboard
bicritère** : GATE-DYN (topologie à λ fixé) = 2/3 (14 ✓, 8 ✓, 2 ✗) ;
GATE-BRANCH (identité de branche en λ) = 1/3 (14 seul). Lecture honnête : le
contre-exemple μ=2 est établi à λ BAS (0.127 sous son fold) — le régime
near-fold (le seul où vit le SNIC de naissance) reste **numériquement
indéterminé** (aucun voisin vivant à bracketer là) ; le collier simple reste
NO-GO, le graphe multifold reste l'objet de repli ; l'edge-tracking nœud–nœud ne
suit PAS la selle du collier quand la branche est multifold. Proposition ET1
re-scopée (NON lancée, approbation requise) : N=2000, GATE-DYN primaire,
δ-sweep λ=λ_c(μ)−δ, δ∈{0.005, 0.010, 0.020} (tester si l'atterrissage adjacent
tient PRÈS du fold et se dégrade en s'en éloignant) ; coût mesuré projeté
~1.5–2 h (3 workers). Fichiers : `E34_ET0prime_{results,summary,log}_…`,
`figures/E34/figE34_ET0prime.png`, section ET0′ de `E34_ET0_RESULTS.md`,
`src/e34_ET0prime_figures.py`.

### E35 (suite) — Gate D0 : diagnostics de readout sur la trajectoire V0 stockée — POSITIF interne, NO-GO V0 CONSERVÉ — FAIT (2026-07-24) — **E35_D0_readouts**
**Contexte :** POC « orbit » V0 (rejeu certifié, `VERDICT_AUDIT.md` §2) : B0 30.06
/ B1 42.84 / B2 44.74 / JP_KP 28.06 dB ; NO-GO qualité conservé ; coefficients
hors simplexe, cond(G)=5.9e5 → hypothèse d'amplification pseudoinverse. Gate D0
(R2.3) : décodeurs préspécifiés, trajectoire seule, aucun choix informé par les
cibles. Protocole gelé avant toute métrique : `roadmaps/E35_D0_PROTOCOL.md`.
**Exécution** (`src/e35d0_readout_diagnostics.py`) : rejeu bit-exact
(max|m−m_stocké| = 0) ; D1=X·a reproduit exactement le décodage stocké (28.059 dB).
**Résultats (48 frames cachées, moyennes, horloge uniforme / phase)** :
D1 X·a : MSE 0.00163, 28.06 dB, SSIM 0.926 | D2 X·G†m clip : 27.70 | D2 sans clip :
25.96 | **D3 TSVD r*=7 (rang choisi sur rappels bruités de frames-CLÉS
uniquement) : MSE 0.00077, 31.35 dB** | D4 β*=5 : 27.93 | D5 renorm 2-comp :
29.02 dB mais MSE 0.00225 (divergence PSNR/MSE connue). Amplification ‖G†m‖/‖m‖ :
méd 1.25, max 2.47, plus haute en transition ; pinv plein rang ~1.7× pire que
r=7 même sur les frames-clés.
**Verdicts :** (1) **D0 POSITIF (règle gelée)** : D3 gagne +3.29 dB (IC bootstrap
par transition [+2.55, +4.07], 16/16 favorables) et divise la MSE par 2 —
l'amplification pseudoinverse est un CONTRIBUTEUR plausible et désormais étayé
(causalité toujours non démontrée stricto sensu). (2) **NO-GO V0 conservé** :
tous décodeurs 0/16 vs B1 ET B2 ; D3 reste −11.5 dB sous B1 (IC [−12.9, −10.1]).
D0 referme une fraction interne du gap (28.1→31.4 dB, juste au-dessus de B0),
pas le gap structurel à l'interpolation linéaire/mouvement.
**Conséquence V1 :** le readout TSVD devient le bras décodeur par défaut du
protocole V1 verrouillé — brouillon `roadmaps/E35_V1_PROTOCOL_DRAFT.md` (seeds
design {1–6} vs éval verrouillés {101–110}, générateur à mouvement non linéaire,
GO seulement si > B1 apparié ET comparé à B2, ≥8/10 seeds) ; AUCUN run V1 lancé.
**Fichiers :** `8_mhn_video/data/E35_D0_{RESULTS.md, diagnostics.npz}` ;
`8_mhn_video/figures/E35_D0_readouts.png` ; codes `src/e35d0_readout_diagnostics.py`,
`src/e35d0_make_figure.py`. SSIM : gaussienne numpy (σ=1.5, K1=0.01, K2=0.03 —
scikit-image absent de l'env).

### E36 (suite) — Réplication multi-seed + raffinement λ_c + LOI b(λ) dense (AICc) — FAIT (2026-07-24) — **E36_delay_scaling_multiseed, E36_b_lambda_dense**
**Contexte :** campagne seed 42 (dossier jumeau, endossée rév. 3) : τ_eff
bi-estimateur, T₁ linéaire en K−1 (R²≥0.999), b(1)=1.710/b(2)=1.192/b(4)=0.950
(tendance 3 points), λ_c ∈ (0.5, 1] à K=6. Suite R2.11 exécutée dans l'ordre :
(1) réplication seeds 43–46 ; (2) raffinement λ_c ; (3) grille b(λ) dense.
N=500, P=25, β=20, dt=0.01, float64, machinerie `layered_chain.py` certifiée
(tests 81/81). Bench mesuré : 6–10k pas RK4/s (~30× plus rapide que l'estimation
FLOP de la roadmap) ; coût réel total ≈ 10 min.
**(1) Réplication (24/24 contrôles dt/2, 0 état non-fini) :** b(λ) par seed
(42/43/44/45/46) → b(1)=1.710/1.661/1.672/1.551/1.758 (**1.670±0.077**),
b(2)=1.192/1.180/1.158/1.135/1.174 (**1.168±0.022**), b(4)=0.950/0.944/0.934/
0.946/0.946 (**0.944±0.006**). Ordre b(1)>b(2)>b(4) pour CHAQUE seed ;
concordance bi-estimateur ≤3.35 % (pire cas, seed 44 λ=1). σ inter-seed
s'effondre avec λ.
**(2) λ_c à K=6 resserré de (0.5, 1] à ≈(0.64, 0.70]** (brackets largeur 0.016 :
42 (0.6562,0.6719] ; 43 et 45 (0.6406,0.6562] ; 44 (0.6875,0.7031] ; 46
(0.6562,0.6719]). DÉCOUVERTE : bande **« irrégulière »** étroite ≈[0.625,
0.66–0.69] entre stationnaire et onde ordonnée, où les 2 IC CONCORDENT sur
`irregular_switching` (pas une ambiguïté d'IC) — étiquetée « irrégulier », pas
« chaos » (pas de Lyapunov, R2.8) ; candidat pour un run Lyapunov ciblé.
**(3) Grille dense** λ∈{1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,8} × K∈{4,6,8,11},
5 seeds, 60 fits R²≥0.999 : b décroît strictement sur toute la grille pour
chaque seed ; σ inter-seed 0.077→0.003. Comparaison de modèles (R2.4, AICc,
n=12/seed) : **le modèle puissance gagne pour les 5 seeds** (ΔAICc ≥53 vs
affine, ≥22 vs plateau-exp) → condition de promotion R2.11 satisfaite :
**LOI b(λ) = 0.752 + 0.909·λ^{−1.14}** (a₀∈[0.728,0.787], p∈[1.03,1.32] par
seed). Le délai par couche ne tend PAS vers le quantum t₀=1 mais vers un
**plateau δ̄∞ ≈ 0.75·t₀** (croisement b=1 vers λ≈3.1) : la conjecture forte
« ligne à retard » (τ_eff=K−1) est quantitativement RÉFUTÉE, la faible
(τ_eff=(K−1)·δ̄(λ)) confirmée. Portée : une seule taille (N=500, P=25) ;
séparation taille/réalisation (étape 4) et mécanismes lourds (étape 5 :
folds/SNIC/Floquet) AVANT toute revendication N→∞.
**Fichiers :** `7_tau_implicite/data/E36_MULTISEED_RESULTS.md` (§11 = grille
dense), `E36_lambda_c_refine.json`, `E36_b_lambda_dense_fits.{json,npz}`,
`E36_bdense_seed*.json`, campagnes `E36A_regimes_seed{43–46}/`,
`E36A2_delay_seed{43–46}/` ; figures `7_tau_implicite/figures/
E36_delay_scaling_multiseed.png`, `E36_b_lambda_dense.png`.
**Étape 4 (même jour) — séparation taille/réalisation :** N=1000/P=50 (seeds
42, 43) et N=2000/P=100 (seed 42), même grille 7 λ × K∈{4,6,8,11} (bench mesuré :
5.8k→1.4k pas/s de N=1000/K=4 à N=2000/K=11 ; 5.9 min wall à 3 processus).
Résultat : **la loi puissance gagne l'AICc aux 8 fits (toutes tailles)** ;
paramètres à CI recouvrants — δ̄∞ = 0.732±0.014 / 0.766±0.046 / 0.743±0.014 et
p = 1.09±0.04 / 1.20±0.17 / 1.11±0.04 pour N=500/1000/2000 ; décalages de b(λ)
entre tailles ≤ bande inter-seed N=500 (≤2 % à bas λ, <1 % pour λ≥3) ; qualité
intacte (R²≥0.999, concordance ≤1.34 %). Verdict R2.10 : **contrôle de
robustesse sur trois tailles et plusieurs réalisations** — PAS une loi N→∞
(P=αN covarie ; étude emboîtée = point ouvert). Fichiers :
`E36_bdense_N{1000,2000}_seed*.json`, `E36_size_separation_fits.{json,npz}`,
`E36_MULTISEED_RESULTS.md` §12, `figures/E36_b_lambda_size.png`.

---

### E34 (suite) — ET1″ : le maillon du collier testé DIRECTEMENT par W^u à N=2000 — POSITIF 4/4 + fermeture sur le cycle 3/3 — FAIT (2026-07-27) — **figE34_ET1pp**
**Re-cadrage mesuré (ET1′ → ET1″).** Le design ET1′ localisait la selle par
edge-tracking nœud–nœud puis polissage Newton. Trois mesures l'ont invalidé à
N=2000 (μ=0, δ=0.020, λ=0.262) : (a) la frontière est **propre** — 13/13
échantillons du côté lointain retombent sur node_1, tous stationnaires, aucune
mer chaotique ; (b) l'amorce « point le plus lent » capte l'**installation
finale sur le nœud** (vitesse 4.7e−13, d_G(node)=0.0000) et non la stagnation
près de la selle, l'approche la plus proche de la selle arclength valant
d_G=0.0706 à t=101.9 contre d_G(selle,nœud)=0.1337 ; (c) Newton **simple ET
déflaté** stagne à res 4e−4…2e−3 depuis **6 amorces** du plateau. Le défaut
était donc dans l'AMORCE, pas dans le polissage — la déflation, correctif
initialement envisagé, était sans objet (elle divergeait : res 1.07e−01).
Statut R2.1 de l'objet de bord : **numériquement indéterminé** (nous
n'établissons pas qu'il n'est pas un équilibre, seulement que ce budget ne
fournit pas d'amorce convergente).
**Protocole ET1″** (`src/e34_necklace_wu_n2000.py`) : par cellule (μ, δ) à
λ=λ_c(μ)−δ — selle partenaire du fold par pseudo-arclength (indépendante de la
dynamique) → certification d'indice par principe de l'argument avec **escalade
du contour** → mode nul **variationnel** (S·Dm, jamais T_P : ~3 % d'erreur de
taux) → **8 tirs de W^u** (2 signes × ε∈{1e−4,1e−3} × dt∈{0.01,0.005}), t_max
1200 → ω-limites. Cible = **première perle vivante en avant** (roadmap §5).
Grille 9 cellules, 1574,8 s.
**Résolution de contour à P=100 (mesurée, `E34_certif_tune`) :** (16,7) ne
stabilise PAS le comptage (n_unstable=−1, débordements slogdet) alors que
l'objet est manifestement instable ; **(32,9) le stabilise à 1**. Escalade
(16,7)→(32,9)→(48,11) câblée dans le pilote.
**Résultats :**
| μ | δ | λ | résidu selle | z_u | d_G(selle,nœud) | W^u₋ | W^u₊ | verdict |
|---|---|---|---|---|---|---|---|---|
| 28 | 0.020 | 0.212122 | 1.5e−15 | 0.1164 | 0.0871 | → 28 | **→ 29** (successeur littéral) | **positif** |
| 1 | 0.005 | 0.298819 | 1.1e−15 | 0.2470 | 0.0320 | → 1 | **→ 3** | **positif** |
| 1 | 0.010 | 0.293819 | 5.4e−16 | 0.2912 | 0.0470 | → 1 | **→ 3** | **positif** |
| 1 | 0.020 | 0.283819 | 6.3e−15 | 0.3567 | 0.1103 | → 1 | **→ 3** | **positif** |
| 91 | 0.005 | 0.322629 | 2.7e−15 | 0.2246 | 0.0358 | → 91 | orbite voyageuse | **fermeture** |
| 91 | 0.010 | 0.317629 | 2.0e−13 | 0.1977 | 0.0575 | → 91 | orbite voyageuse | **fermeture** |
| 91 | 0.020 | 0.307629 | 2.2e−16 | 0.2590* | 0.0840 | → 91 | orbite voyageuse | **fermeture** |
\* localisation de repli : comptage stable à 1 + racine réelle encadrée ⇒ la
racine instable unique EST cette racine réelle ; mode variationnel construit en
ce point (résidu 7.5e−16, porte linéaire 7.47e−3).
**Conclusions :**
1. **Le maillon tient 4 fois sur 4 là où une perle suivante existe.** Pour
   μ=28 (100 perles vivantes) la cible est le successeur **littéral** μ+1=29.
2. **À la dernière perle (μ=91, λ_c=0.327629 ≈ λ*), le collier se referme sur le
   cycle** : aucune autre perle vivante, W^u₊ rejoint une orbite à **pas
   d'anneau exactement +1** (30 changements de meneur, durée médiane 19.0,
   cv 0.176, période de tour ≈1900, amplitude du meneur 0.934). Fermeture SNIC,
   pas un échec.
3. **Les deux verdicts négatifs initiaux venaient du RECENSEMENT, pas de la
   physique.** À μ=1 δ=0.005, l'objet limite « hors recensement » est la perle
   **3** : résidu 3.96e−11, m₃=+0.9862, identité vraie, eigmax −0.1569 — une
   mémoire stable à λ **au-dessus** de son seuil tabulé de 1.29e−3.
4. **μ=28 était mal paramétrée par la table** : fold réel 0.232122 contre
   λ_c tabulé 0.219458 (**+1.27e−2** ; à comparer à +2.96e−5 pour μ=1 et
   −3.0e−6 pour μ=91). Rejouée au fold mesuré, la cellule aboutit sans autre
   modification.
**Constat d'audit (nouveau, dépasse E34) :** test systématique « existe-t-il
encore un équilibre STABLE d'identité μ à λ_c(μ)+offset ? » sur les 100 perles
→ **33/100** (sondes 5e−4…2e−2 ; 11 jusqu'à +0.010, 4 jusqu'à +0.020 ; une
passe grossière {5e−3,1e−2} n'en trouvait que 22). **C'est une borne
inférieure** : la sonde repose sur Newton amorcé au motif, qui échoue
précisément pour la perle 3 à un λ où le flot y converge — la perle 3 ne figure
pas dans les 33. Deux lectures non tranchées : **table en avance** vs **branche
en S** (μ=8, déjà le cas multifold d'ET0′, figure parmi les 33). Pour μ=28
l'arclength penche pour la première (montée de 0.1995 à 0.2321 **sans
retournement**). Portée : le recensement (`solve_alive_nodes`) est bâti sur
cette table et sert dans tout E34 ; la loi λ_c(μ) d'E24/E27 aussi. Test propre =
fold par arclength perle par perle (≈4 h) — **non fait**.
**Ce qui n'est PAS établi :** que la selle testée soit l'objet séparant
effectivement les deux bassins. W^u(selle) → node_μ et → perle suivante est
mesuré ; l'identification avec l'objet de bord reste ouverte. Une seule taille,
une seule réalisation (R2.10). μ=28 rejouée au seul δ=0.020.
**Fichiers :** `2_cycle_rappel_snic/E34_ET1pp_RESULTS.md` ;
données `2_cycle_rappel_snic/data/E34/E34_ET1pp_wu_{summary,results,log}_20260727T154356.*`,
`E34_ET1pp_followup_{arc,land,loc,cycle,recell}_*.json`,
`E34_threshold_table_audit_20260727T162013.json` (fines) et `...T161647.json`
(grossières), `E34_certif_tune_*.json`, `E34_dwell_seed_test_*.json`,
`E34_deflation_test_*.json` ; figure
`2_cycle_rappel_snic/figures/E34/figE34_ET1pp.png`.
**Codes :** `e34_necklace_wu_n2000.py`, `e34_et1pp_followup.py` (modes arc /
land / loc / cycle / recell), `e34_threshold_table_audit.py`,
`e34_certif_tune.py`, `e34_et1_boundary_diag.py`, `e34_dwell_profile.py`,
`e34_dwell_seed_test.py`, `e34_ET1pp_figures.py` (remplace
`e34_ET1prime_figures.py`, écrit pour le design caduc).

---

### E35 (suite) — Gate V1 : nouveau POC verrouillé (conception ≠ évaluation) — NO-GO INTERPOLATION, cycle PARFAIT — FAIT (2026-07-27) — **E35_V1_design, E35_V1_eval**
**Contexte :** D0 avait promu le readout TSVD (+3,3 dB target-blind) sans lever le
NO-GO V0 (−11,5 dB sous B1). Le brouillon V1 (`roadmaps/E35_V1_PROTOCOL_DRAFT.md`)
a été **validé par l'utilisateur le 2026-07-25** puis exécuté sans modification de
§1–§8 (précisions d'exécution A.9–A.12 ajoutées, datées).
**Discipline de gel :** décisions prises sur `design_seeds={1…6}` uniquement ;
`eval_seeds={101…110}` ouverts **une seule fois**, après écriture du hash de
contenu **`35a3bc12f9c992ef`** — le script d'évaluation refuse de démarrer si le
hash diverge (vérifié contre le JSON ET le npz de conception). Aucun critère
affaibli ; l'architecture classée sur la seule couverture de cycle (aucune
métrique d'image) ; `r*` issu de rappels bruités de frames-CLÉS seules (règle D0).
**Générateur figé (A.3, 9 réglages × 6 seeds, aucune dynamique) :** advection non
linéaire d'une texture à bande limitée sur une courbe fermée du tore 64×64
(N=4096, P=100, k=4). **2 réglages sur 9** satisfont les DEUX critères gelés
(`B2−B1 ≥ 3 dB` ET résidu hors-span ≥ 0.05) ; la parcimonie A.3 retient
**winding=(3,2), σ=2.5** : headroom **+7.95 dB**, résidu **0.0639**, cond(G) 8.0e7
— un POC à marge réelle, contrairement à V0 (+1.90 dB, 0.0151).
**Factoriel J×K :** cycle valide sur 100 % des seeds pour `JP_KP`, 17 % pour
`JH_KP`, **0 %** pour `JH_KH` et `JP_KH` → bras figé `JP_KP` (B3 = `JH_KH`).
**Résultats d'évaluation (10 seeds, ouverture unique) :**
| Condition GO gelée §5 | Mesuré | Satisfaite |
|---|---|---|
| cycle valide ≥ 80 % des eval_seeds | **100 %** (10/10) | **OUI** |
| ΔPSNR(dyn−B1) > 0 et IC 95 % hiérarchique > 0 | **−15.830 dB**, IC [−16.680, −15.020] | NON |
| dyn ≥ B2 (GO fort) | **−23.563 dB**, IC [−24.119, −23.029] | NON |
| ≥ 8/10 seeds favorables | **0/10** seeds, **0/1000** blocs-transitions | NON |
| Référence | PSNR (dB) |
|---|---:|
| dyn `JP_KP`, D3 TSVD (r\*=54), CLK-P | **17.373** |
| dyn `JP_KP`, D3 TSVD, CLK-U (nominal) | 17.195 |
| B0 maintien de la frame-clé | **21.205** |
| B1 interpolation linéaire | 33.025 |
| B2 mouvement | 40.758 |
| B3 Hebb `JH_KH` (126/3000 frames décodables seulement) | 10.5 |
**Conclusions :**
1. **V1 = NÉGATIF (NO-GO interpolation).** La condition 1 étant satisfaite, ce
   n'est PAS « numériquement indéterminé » : le cycle tient, c'est la
   reconstruction qui échoue. L'écart n'est pas marginal.
2. **Fait central : le meilleur readout dynamique est 3,83 dB EN DESSOUS de B0**,
   la ligne de base qui se contente de recopier la frame-clé précédente. Aucune
   des 16 combinaisons décodeur × horloge ne produit une seule transition
   favorable sur les 1000 blocs.
3. **Le mécanisme de cycle est PARFAIT** : `coverage = forward = 1.000` sur 10/10
   seeds, périodes 1095,4–1117,2 (4,03–4,11 tours), `period_cv ≤ 2.63e−5` (et
   exactement 0 sur plusieurs seeds de conception) — rappel séquentiel ordonné et
   stable sur des motifs corrélés à cond(G)~8e7.
4. **La marge existait** : B2 bat B1 de +7,73 dB sur les seeds d'évaluation — ce
   n'est donc pas un plafond de la tâche, c'est un échec du bras dynamique.
5. **NO-GO V0 CONSERVÉ** indépendamment (clause invariante §5) : V1 teste une
   autre classe de mouvement, il n'efface pas V0.
**Conséquence prescrite (§5/R2.3) :** pas d'escalade vers Moving MNIST ni vers la
vraie vidéo. Replis : proxy de taux (frames-clés + précision, Gram/SVD) et
rescousse des corrélations (dense vs Hebb pur — ici net : 100 % vs 0 % de cycles
valides).
**Coût :** 56,4 min au total (bench 27 s, générateur 24,7 s, pré-screen 299 s,
factoriel 1717 s, gel 21,9 s, évaluation 1292 s), mono-processus, 2 threads BLAS,
en parallèle d'E34 sans contention. Aucune des deux réductions pré-déclarées A.8
n'a été nécessaire (P=100 et 6 seeds de conception conservés).
**Fichiers :** `8_mhn_video/data/E35_V1_RESULTS.md`, `E35_V1_{design,eval}.npz`,
`E35_V1_frozen_config.json` ; figures `8_mhn_video/figures/E35_V1_{design,eval}.png` ;
protocole `roadmaps/E35_V1_PROTOCOL_DRAFT.md` (A.9–A.12 datées).
**Codes :** `e35v1_common.py`, `e35v1_bench.py`, `e35v1_design.py` (stages
headroom/screen/select/freeze), `e35v1_eval.py` (refus sur hash divergent),
`e35v1_figures.py`.
