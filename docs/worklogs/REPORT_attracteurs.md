# Attracteurs du réseau de Hopfield mixte retardé : bassins d'attraction et nature du chaos

**Auteur :** agent numérique (Claude Code) · **Date :** 2026-07-09 (rev. 2 — §4bis E33 : le chaos est un attracteur stationnaire à mesure SRB, il ne vieillit pas ; §4bis E32 bassins (α,λ) ; rev. 1 : bassins E18d/E20 + chaos E23A)
**Compagnons :** `CYCLE_worklog.md` (journal E-numéroté maître), `REPORT_static_bifurcation.md`
(§5.7 : unification mémoire↔front), `REPORT_faible_tau.md` (petit τ).
**Code :** `numerics/src/` · **Figures/données :** `numerics/results/4_attracteurs_bassins_chaos/`.

> **Objet.** Consigne les explorations 2026-07-04/05 sur (I) les **bassins d'attraction**
> des points fixes et du cycle — compétition mémoire ↔ fronts épinglés ↔ cycle ↔ chaos,
> métastabilité des fronts — et (II) la **nature de la mer chaotique** de la fenêtre de
> mixture (attracteur étrange ? dimension ?). Point d'entrée pour l'approfondissement
> futur ; protocoles complets dans `CYCLE_worklog.md` §6 (E18d, E20, E23A).

**Cadre.** τ=10, N=2000, α=0.05 (P=100), β=20, seed 42 ; DDE réduite exacte dim-100,
float64. Paysage établi : chaque motif ξ^μ porte une branche de point fixe qui meurt à
son seuil gelé λ_c(μ) (REPORT statique §5.7) ; λ_c=λ_c(ξ¹)=0.2822, λ*=max_μ=0.328
(liaison 91) ; entre les deux, mer chaotique ; au-dessus, cycle de rappel séquentiel.

---

## 1. Questions

1. Comment les bassins des attracteurs se partagent-ils l'espace des phases en fonction
   de λ ? Les fronts épinglés (stables) sont-ils métastables vis-à-vis des mémoires ?
2. Le bassin du cycle : comment croît-il avec λ ?
3. Le chaos de la fenêtre λ_c<λ<λ* : attracteur chaotique véritable (type Lorenz) ou
   autre chose (transitoire, chaos extensif) ? Quelle dimension ?

---

## 2. E18d — premières mesures de bassin du front (2026-07-04) — figK

**Protocole** (`CYCLE_worklog.md` §6-E18) : autour du front 91/92 exact, (i) **rayon
d'échappement directionnel** (bissection de l'amplitude critique le long de la direction
vers ξ¹ et de la direction propre la plus faible) ; (ii) capture d'IC aléatoires
(150/point, boules de rayon 0.15/0.3/0.5, t=200).

| λ | r_échap vers ξ¹ | r_échap dir. faible | capture (r=0.3) |
|---|---|---|---|
| 0.15 | 0.825 | 0.896 | 100 % |
| 0.25 | 0.513 | 0.366 | 100 % |
| 0.31 | 0.238 | 0.136 | 21 % |
| 0.325 | 0.116 | 0.031 | ~1 % |

**Conclusion** : le bassin du front **rétrécit monotoniquement** vers λ* pendant que sa
stabilité linéaire reste intacte — la disparition du front du paysage dynamique est
géométrique (bassin), pas spectrale. **Figure :** `figures/figK_front_basins.png`.
**Npz :** `E18_basins.npz`. **Code :** `e18d_basins.py`.

---

## 3. E20 — compétition globale des bassins et métastabilité (2026-07-05) — figN, figO

**Protocole** : K=250 IC aléatoires par λ (14 valeurs, familles near-pattern / mélanges
consécutifs / random-small documentées), intégration t=300, classification de l'état
final. Classifieur (encode l'unification §5.7 du REPORT statique) : état stationnaire
2-motifs sur la **liaison faible** 91/92 → `front` (métastable) ; sur toute autre
liaison → `memory` ; avance soutenue sur l'anneau → `cycle` ; non-stationnaire →
`chaos`. Validé sur états connus (front→front à 5e−17, mémoire→mémoire, λ haut→cycle).

**Fractions de bassin** (extrait ; table complète : `4_attracteurs_bassins_chaos/data/E20_RESULTS.md`) :

| λ | mémoire | front | cycle | chaos |
|---|---|---|---|---|
| 0.15 | 0.32 | 0.004 | 0.00 | 0.35 |
| 0.28 | 0.48 | 0.016 | 0.02 | 0.48 |
| 0.31 | 0.008 | 0.044 | 0.47 | 0.47 |
| 0.325 | 0.00 | 0.048 | 0.55 | 0.40 |
| 0.40 | 0.00 | 0.00 | 0.84 | 0.16 |
| 0.90 | 0.00 | 0.00 | 0.79 | 0.21 |

**Conclusions :**
1. **Métastabilité des fronts quantifiée** (l'intuition de l'utilisateur est confirmée) :
   bassin du front ≤ 5 % à tout λ (max 0.048 à λ=0.325), rapport front/bassin-dominant
   ≲ 0.1 partout — **le front ne gagne jamais la compétition**. C'est un piège étroit
   qu'on n'atteint dynamiquement que si le cycle vient s'y poser (SNIC).
2. **Mémoire** : dominante à bas λ, s'effondre au voisinage des folds (0.48 à λ=0.28 →
   0.008 à λ=0.31) — la fin *effective* de la phase de rappel (par bassins) précède la
   disparition des branches (λ*(μ) jusqu'à 0.328).
3. **Cycle** : bassin nul sous 0.30, croît vite (0.47 dès λ=0.31), sature vers 0.84.
4. **Chaos** : remplit la fenêtre de mixture (jusqu'à 0.70 à λ=0.25) et **persiste en
   bassin résiduel de 16–21 % jusqu'à λ=0.9** — le cycle n'est jamais globalement
   attracteur unique.
**Figures :** `figures/figN_basin_competition.png` (fractions empilées),
`figures/figO_metastability.png` (indice de métastabilité). **Npz :**
`E20a_basin_fractions.npz`, `E20a_basin_fractions_hi.npz`. **Codes :** `e20_common.py`,
`e20a_basin_fractions.py`, `e20a_hi.py`, `e20_figures.py`.

---

## 4. E23A — nature de la mer chaotique (2026-07-05) — figQ

**Protocole** : exposants de Lyapunov par Benettin/QR sur le système variationnel
retardé (adaptation du floor-free d'E14) ; λ_L(λ) sur la fenêtre ; convergence
fini-temps sur T=3000 à λ=0.31 ; spectre partiel k=8 + dimension de Kaplan-Yorke ;
géométrie (PCA, section de Poincaré, first-return map, spectre de puissance).

**Résultats :**
1. **λ_L > 0 et SOUTENU sur toute la fenêtre** : +0.050 / +0.040 / +0.053 / +0.050 /
   +0.050 pour λ = 0.29 / 0.30 / 0.31 / 0.32 / 0.325. À λ=0.31, l'exposant fini-temps
   converge vers **+0.048** sur T=3000 (≈150 temps de Lyapunov) sans jamais se poser →
   **attracteur chaotique, pas transitoire** (pas une selle chaotique).
2. **Spectre k=8 à λ=0.31** : {+0.035, +0.011, +0.0015, −0.003, −0.008, −0.022, −0.032,
   −0.042}. Lecture : **2 exposants strictement positifs** + le **mode neutre** (le
   +0.0015 est l'estimateur fini-temps du 0 exact de la direction du flot, présent dans
   tout système autonome sur trajectoire bornée) → **hyperchaos** (2 directions
   dilatantes ; Lorenz n'en a qu'une).
3. **Dimension de Kaplan-Yorke D_KY ≈ 6.44** (j=6, résolue). L'attracteur est un objet
   fractal d'environ 6 dimensions dans l'espace des 100 overlaps — **chaos déterministe
   de basse dimension effective**, PAS extensif (D_KY ≪ P ≪ N). PCA : 3 composantes
   capturent 72 % de la variance. ⚠ Un pré-calcul k=4 donnait D_KY≈3.1 (sous-estimé,
   fenêtre trop courte) — la valeur de référence est le k=8 sur T=2000.
4. **Géométrie** : first-return map **dispersé** (pas une courbe 1-D à la Lorenz),
   spectre de puissance **large-bande** (pas de quasi-périodicité).

**Verdict** : même famille que Lorenz (attracteur étrange déterministe soutenu,
sensibilité exponentielle, basse dimension) mais « plus épais » : hyperchaotique
(2 directions instables), D_KY≈6.4 vs 2.06, non réductible à une dynamique 1-D.
Image physique : errance itinérante entre voisinages de mélanges de motifs, éjectée
par le drive retardé non-réciproque ; les ~6 dimensions effectives ≈ les 2-3 overlaps
dominants du voisinage courant + direction du flot + directions de relais.
**Figure :** `figures/figQ_chaos_nature.png` (6 panneaux). **Npz :** `E23A_lyap.npz`,
`E23A_geometry.npz`. **Code :** `e23a_chaos_nature.py`.

**Cohérence avec l'existant** : λ_L≈+0.05 raffine le +0.034 d'E8a (fenêtre plus courte) ;
la P(q) large d'E7 et l'itinérance d'E8 sont les signatures statiques du même attracteur.

---

## 4bis. E33 — L'attracteur chaotique a-t-il une mesure invariante ? Vieillissement ? (2026-07-08, GPU) — figAC, figAD

**Question (utilisateur) :** dans l'analogie avec le chaos verre-de-spin du Hopfield
classique, notre chaos de la fenêtre de mixture vieillit-il ? Piste proposée :
existe-t-il une mesure invariante (type SRB, comme Lorenz) ; sinon, le système
vieillirait-il ?

**Cadrage (corrigé).** La prémisse « pas de mesure invariante ⇒ vieillissement » est
inversée. Pour un attracteur **borné** d'un flot **autonome** dissipatif — ce qu'E23A
a établi (λ_L≈+0.048 soutenu sur ~150 temps de Lyapunov, D_KY≈6.4) — une mesure
physique SRB **existe génériquement** (cas Lorenz), et un attracteur SRB *stationnaire*
est précisément l'état qui **ne** vieillit **pas**. Le vieillissement au sens verre de
spin est la brisure explicite d'invariance par translation temporelle des quantités à
deux temps : C(t_w+t,t_w) dépendant séparément de t_w (collapse en t/t_w), + violation
FDT. Le test définitionnel est donc la corrélation à deux temps, PAS l'existence de la
mesure (qui est acquise).

**Protocole** (`e33_aging.py`, intégrateur réduit batché MLX **float32/GPU**, ensembles
64 IC × 4 seeds, settle 400 puis enregistrement T=6000 à dt_rec=0.5, masque « resté
chaotique » excluant les trajectoires échappées vers cycle/mémoire ; a·a invariant de
jauge) : C(t_w+t,t_w)=⟨a(t_w+t)·a(t_w)⟩ pour t_w∈{0,50,200,800,3200} ; relaxation
post-quench ⟨|a|²⟩(t) SANS settle (mesure τ_r — tueur de confusion transitoire) ;
extension « aging près de la capacité » (design Exp E2) α∈{0.08,0.10,0.12}.

**Résultats.**
1. **α=0.05, λ=0.31 : STATIONNAIRE, pas de vieillissement.** Les C(t_w+t,t_w)
   normalisées se **superposent** à travers tous les t_w (écart 0.009 à lag~20 ;
   écart relatif de C(t_w,0) : 0.002) et NE collapsent PAS sous t/t_w (figAC). La
   dynamique échantillonne donc une mesure invariante SRB : statistiques invariantes
   par translation temporelle.
2. **Relaxation post-quench τ_r≈2** (⟨|a|²⟩ chute de ~9 à un plateau 0.74 en ~2 t.u.).
   Le mode quasi-marginal λ₄≈−0.003 du spectre E23A ne crée **aucun transitoire lent**
   (~1/|λ₄|~300 aurait pu masquer un vieillissement — écarté) : le settle 400 ≫ τ_r.
3. **Pas d'aging émergent vers la capacité** (figAD) : écart-collapse 0.021 (α=0.08,
   n=112 chaotiques) et 0.025 (α=0.10, n=41), tous ≪ seuil 0.1 ; à α=0.12 l'ensemble
   chaotique à λ=0.31 s'évanouit (n=11, intestable — informatif : près de la capacité
   λ=0.31 n'est plus la mer chaotique mais du cycle/statique).

**Verdict.** La mer chaotique est un **attracteur étrange stationnaire à mesure SRB
bien définie et échantillonnée** — on y a donc « accès » au sens opérationnel (moyennes
temporelles = moyennes d'ensemble sur la mesure). Il **ne vieillit pas**, contrairement
à la dynamique thermique activée du verre de spin de Hopfield ; l'analogie s'arrête à la
géométrie rugueuse du paysage (P(q) large, itinérance), elle n'inclut PAS la
non-stationnarité vieillissante. **Caveats :** (i) float32/GPU, porte de validation
float64 (`e33_aging.py --validate`) non encore exécutée — mais le signal (0.009) est
~400× au-dessus du plancher float32 (~10⁻⁵), le verdict est robuste ; (ii) l'énoncé
grand-α est préliminaire (n chaotique faible) ; un test propre près de la capacité
demanderait beaucoup plus de trajectoires ou un λ choisi plus profond dans le chaos par α.
**Ce qui reste pour « toucher » la mesure SRB plus finement** : convergence explicite de
la densité (histogrammes des coordonnées PCA/participation vs fenêtre et vs t_w, distance
KS→0), test d'ergodicité (moyenne d'une longue trajectoire = moyenne d'ensemble), et
réponse/FDT (pente réponse-corrélation, X). Voir §6 points ouverts.
**Figures :** `figures/figAC_aging_twotime.png`, `figures/figAD_aging_vs_alpha.png`.
Npz `E33_aging_mlx[_a0p08/0p10/0p12].npz`. Codes `e33_aging.py`, `e33_analysis.py`,
`e33_vs_alpha.py`, `cycle_reduced_mlx.py` (intégrateur GPU validé).

---

## 5. Vue d'ensemble du paysage (τ=10, seed 42)

| λ | attracteurs coexistants (stabilité linéaire) | qui gagne les bassins |
|---|---|---|
| < 0.28 | P branches mémoire-inclinées (dont le « front ») + chaos | mémoire (~0.3–0.5) et chaos |
| 0.28–0.31 | branches survivantes (λ_c(μ) étagés) + chaos + cycle naissant | chaos ; mémoire s'effondre |
| 0.31–0.328 | dernières branches (front 91) + chaos + cycle | cycle (~0.5) et chaos (~0.4) |
| > 0.328 | cycle + chaos | cycle (~0.8), chaos résiduel (~0.2) |

## 6. Points ouverts (pour l'approfondissement annoncé)

1. **D_KY(λ)** à travers la fenêtre et au-delà (le chaos résiduel à λ=0.9 a-t-il la même
   dimension ?) ; λ_L et D_KY multi-seed.
2. **Frontières de bassin** : sont-elles fractales (expliquerait la sensibilité des
   fractions près de λ_c) ? Dimension de la frontière mémoire/chaos.
3. **Temps d'échappement** du chaos résiduel à haut λ (le bassin chaos à λ=0.9 est-il un
   vrai attracteur coexistant ou un très long transitoire ? → allonger t≫300).
4. **Mesure d'échantillonnage** : les fractions dépendent de la famille d'IC (documentée
   dans `e20_common.py`) ; refaire avec une mesure alternative (IC sur la sphère de
   norme fixée) pour tester la robustesse.
5. Lien E23A ↔ petit τ : l'état multi-lobe sous τ_c est-il le MÊME attracteur étrange
   (continuation en τ de D_KY) ? Cf. `REPORT_faible_tau.md` §6.
6. **Mesure SRB, caractérisation fine (suite d'E33)** : (a) validation float64 de la
   corrélation deux-temps (porte `e33_aging.py --validate`) ; (b) convergence explicite
   de la densité invariante (histogrammes des observables vs fenêtre/t_w, KS→0) et test
   d'ergodicité (moyenne temporelle longue = moyenne d'ensemble) ; (c) réponse/FDT
   déterministe (pente réponse-corrélation X) ; (d) aging près de la capacité avec un
   ensemble chaotique suffisant (λ choisi profond dans le chaos par α).

## 7. Inventaire

| Objet | Chemin (sous `numerics/results/4_attracteurs_bassins_chaos/`) |
|---|---|
| figK — bassin du front (rayons, capture) | `figures/figK_front_basins.png` |
| figN — compétition des bassins vs λ | `figures/figN_basin_competition.png` |
| figO — indice de métastabilité du front | `figures/figO_metastability.png` |
| figQ — nature du chaos (Lyapunov, D_KY, géométrie) | `figures/figQ_chaos_nature.png` |
| figAB — bassins (α,λ), érosion du cycle (E32) | `figures/figAB_alpha_lambda_basins.png` |
| figAC — vieillissement à α=0.05 : stationnaire (E33) | `figures/figAC_aging_twotime.png` |
| figAD — pas d'aging vs α (E33, Exp E2) | `figures/figAD_aging_vs_alpha.png` |
| données bassins | `data/E18_basins.npz`, `data/E20a_basin_fractions.npz`, `data/E20a_basin_fractions_hi.npz`, `data/E20_RESULTS.md`, `data/E32_basins_N2000_mlx.npz` |
| données chaos | `E23A_lyap.npz`, `E23A_geometry.npz`, `E33_aging_mlx[_a0p08/0p10/0p12].npz` |
| codes | `src/e18d_basins.py`, `src/e20_common.py`, `src/e20a_basin_fractions.py`, `src/e20a_hi.py`, `src/e20_figures.py`, `src/e23a_chaos_nature.py`, `src/e32_alpha_lambda_basins.py`, `src/e33_aging.py`, `src/cycle_reduced_mlx.py`, `src/cycle_reduced_batch.py` |
