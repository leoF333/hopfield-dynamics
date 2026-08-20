# Le régime à petit délai (τ ≲ t₀) : désynchronisation du cycle et mort dans le chaos

**Auteur :** agent numérique (Claude Code) · **Date :** 2026-07-05 (rev. 1)
**Compagnons :** `CYCLE_worklog.md` (journal E-numéroté maître, protocoles complets),
`REPORT_static_bifurcation.md` (statique), `REPORT_attracteurs.md` (bassins & chaos).
**Code :** `numerics/src/` · **Figures/données :** `numerics/results/5_petit_tau/`.

> **Objet.** Consigne les explorations 2026-07-04/05 du régime **τ petit (ordre t₀=1
> ou moins)**, jusque-là inexploré : le cycle de rappel séquentiel existe-t-il encore,
> comment meurt-il, et le ralentissement critique en √(λ−λ*) (signature SNIC établie à
> τ=10) survit-il ? Ce rapport est le point d'entrée pour l'approfondissement futur ;
> les protocoles détaillés vivent dans `CYCLE_worklog.md` §6 (E19, E22, E23).

**Modèle et référence.** t₀u̇ = −u + (1−λ)J·tanh(βu(t)) + λK·tanh(βu(t−τ)) ;
N=2000, α=0.05 (P=100), β=20, t₀=1, seed 42 ; toutes les intégrations sur la
**réduction exacte en dimension P** (`cycle_reduced.py`, certifiée à 5.5e−16), float64.
Acquis à τ=10 (rappel) : cycle pacemaker T₁≈τ+t_esc(λ), mort par **SNIC de dépiégeage**
au maillon faible (liaison 91/92) à λ*≈0.327–0.328, lois √ des deux côtés
(t_MAX=C/√(λ−λ*), C=2.3 ; eig_max=−c√(λ*−λ), c≈2.1).

---

## 1. Questions

1. Le cycle existe-t-il à τ∈{0…2} (τ=0 = limite ODE) ? Que devient l'horloge T₁(τ) ?
2. Comment le cycle meurt-il quand λ décroît à petit τ — même SNIC, même λ* ?
3. La loi √(λ−λ*) du temps de passage survit-elle ?
4. La dynamique lente near-ghost existe-t-elle encore et se comprend-elle pareil ?

---

## 2. E19 — première exploration (2026-07-04) — figL, figM

**Protocoles** : `CYCLE_worklog.md` §6-E19. Quatre volets : (a) construction du cycle à
λ=0.9 pour τ∈{0, 0.25, 0.5, 1, 2} et mesure des relais ; (b) descente ensemencée en λ
(pas 0.005) jusqu'à la mort, classification de l'état post-mort ; (c) loi de
ralentissement t_MAX(λ) pour τ∈{2, 1, 0.25}, cap t=2000, non-passages censurés ;
(d) contrôle côté point fixe (eig_max(M) ≡ E12, racines T_P au front vs τ).

**Résultats clefs :**

| τ | T₁ (moy±éc.) | largeur front | rappel pic | λ*(τ) mort | état post-mort |
|---|---|---|---|---|---|
| 0 (ODE) | 1.82 ± 0.80 | 14.0 motifs | 0.16 | — | — |
| 0.25 | 1.70 ± 1.07 | 10.7 | 0.22 | 0.3303 | chaos/dérive |
| 0.5 | 1.75 ± 1.03 | 9.9 | 0.24 | — | — |
| 1.0 | 2.22 ± 0.68 | 4.2 | 0.34 | 0.3322 | chaos/dérive |
| 2.0 | 2.80 ± 0.02 | 1.3 | 0.74 | 0.3497 | chaos/dérive |

- **Le cycle survit jusqu'à τ=0** mais sous forme de **front large multi-lobe de faible
  rappel** ; la loi pacemaker dégénère (fit E19a : T₁≈0.56τ+1.63, pente ≠ 1).
- **La mort n'est pas un piégeage propre** : état post-mort = chaos/dérive (jamais le
  front épinglé seul), λ*(τ) voisin mais au-dessus du seuil spectral 0.328.
- **La loi √ est DÉTRUITE** : fits C/√(λ−λ*) de qualité nulle (R²=0.31/0.23/0.02 pour
  τ=2/1/0.25 contre l'ajustement propre à τ=10), t_MAX erratique, « maillon faible »
  sautant de liaison en liaison (25, 70, 24, 63, 37…) au lieu de rester la 91.
- **Côté point fixe rien ne bouge** (E19d) : eig_max(M) au front ≡ E12 à la précision
  machine (τ-indépendant) ; seule la période du mode complexe retardé varie avec τ.

**Figures :** `5_petit_tau/figures/figL_smalltau_cycle.png` (horloge, désordre des relais,
largeur, kymographes τ=0/0.5/2) ; `5_petit_tau/figures/figM_smalltau_snic.png` (t_MAX vs λ−λ*
log-log : nuage sans loi, censures ×). **Npz :** `E19a_cycle_clock.npz`,
`E19b_death.npz`, `E19c_slowing.npz`, `E19d_fixedpoint.npz`. **Codes :** `e19a…e19d`.

---

## 3. E22 — quantification de la désynchronisation (2026-07-05) — figP

**Protocole** : grille τ dense {0, 0.1, …, 1.0, 1.2, …, 2.0, 2.5, 3, 5, 10} à λ=0.9 ;
dt=min(0.01, τ/25), convergence T₁ vérifiée par dt-halving (ΔT₁<2e−4) ; ≥25 relais.
Observables : T₁±éc., largeur W, rappel pic, nombre de lobes, **ratio de participation
N_part=(Σa²)²/Σa⁴** (nombre effectif de lobes), cohérences R_kura et R_pr.

**Résultat central : transition de désynchronisation ABRUPTE à τ_c ≈ 1.3 ≈ t₀,**
simultanée sur tous les indicateurs (bascule entre τ=1.2 et 1.4) :

| | τ ≲ 1.3 | τ ≳ 1.3 |
|---|---|---|
| N_part (lobes effectifs) | 18–20 | → 1 |
| largeur W | 13–15 motifs | 1.3 → 1.07 |
| rappel pic | ≈ 0.2 | 0.67 → 0.93 |
| éc.(T₁) | ± 1.0 | ± 0.01 |
| R_pr | 0.05 | 0.69 → 0.91 |

- **Lois de période** : T₁=0.518τ+1.727 (τ≤1) → **T₁=τ+0.815** (τ≥2, pacemaker pente 1).
  L'horloge bascule plus tôt (pente 0.9 dès τ≈0.5) que la forme spatiale (τ_c≈1.3) :
  deux crossovers distincts.
- **λ ne resynchronise pas** (λ=0.9→0.99 à τ=0.25/1 : W, N_part, rappel inchangés) —
  la désynchronisation est intrinsèque au délai court.
- Bon paramètre d'ordre : **R_pr/N_part** (saut 0.06→0.69) ; R_kura inadapté (mou).

**Figure :** `5_petit_tau/figures/figP_desync.png` (6 panneaux). **Npz :** `E22_desync.npz`,
`E22_lamcheck.npz` (+ table `5_petit_tau/data/E22_RESULTS.md`). **Code :** `e22_desync.py`.

---

## 4. E23B — frontière cycle-chaos et résolution du paradoxe SNIC (2026-07-05) — figR

**Le paradoxe** : les fronts épinglés sont des points fixes **τ-indépendants** (Δ(0)=−M),
leur selle-nœud reste à λ*=0.328 à tout τ — comment le SNIC peut-il disparaître à petit τ ?

**Protocole** : carte (τ,λ) par descente ensemencée (τ∈{0.25, 0.5, 1, 2, 5} × λ fin
autour de [0.325, 0.40]) avec classification de l'attracteur ; exposant de Lyapunov λ_L
mesuré SUR le cycle juste avant la mort et sur l'état juste après ; stabilité du front
recalculée dans la bande de mort.

**Résultats décisifs :**

| τ | λ_L sur le « cycle » (juste au-dessus de la mort) | λ_L après la mort |
|---|---|---|
| 1.0 | **+0.051** (déjà chaotique !) | +0.028 |
| 2.0 | **+0.054** (déjà chaotique !) | −0.026 (front) |
| 5.0 | −0.102 (cycle régulier) | −0.176 |
| 10.0 | −0.002…−0.003 (cycle régulier) | — |

- **λ*_front = 0.3280, τ-indépendant, recalculé et confirmé.** Dans la bande de mort le
  front reste un point fixe stable (eig_max<0, 0 direction instable T_P) à tout τ pour
  λ<λ* ; chaos et front **coexistent** — mais le bassin du front est infime (E20).
- **Résolution** : ce qui disparaît à petit τ n'est pas le selle-nœud, c'est **le cycle
  limite propre**. Sous τ_c≈1.3 (E22), le « cycle » est l'état désynchronisé multi-lobe,
  qui est **lui-même faiblement chaotique** (λ_L>0 mesuré sur l'attracteur « cyclant »).
  À la mort il se **fond dans la mer chaotique** au lieu de se poser sur le front —
  « mourir sur le chaos ». Sans orbite périodique cohérente, pas de passage near-ghost
  déterministe par le selle-nœud → **pas de loi √** (E19c), et le λ*(τ) mesuré
  dynamiquement (0.330–0.350) est le point de perte de cohérence, pas le fold.
- Gap mort-du-cycle − λ*_front : −0.003 à +0.007 selon τ (résolution ±0.005) — la mort
  se produit au voisinage du fold mais par un mécanisme de crise/fusion avec le chaos.

**Figure :** `5_petit_tau/figures/figR_cycle_chaos_frontier.png` (carte (τ,λ), largeur de la
bande chaos, saut de λ_L à la mort). **Npz :** `E23B_frontier.npz`. **Code :**
`e23b_frontier.py`.

---

## 5. Synthèse : une seule histoire

Le scénario SNIC de τ moyen repose sur **trois ingrédients** : (i) un front localisé
(un seul lobe), (ii) la statistique d'extrêmes des seuils gelés (UN maillon faible où le
front pince en premier), (iii) une orbite périodique cohérente qui ride le ghost du
selle-nœud. À **τ < τ_c ≈ 1.3 ≈ t₀**, la délocalisation du front (E22) détruit (i),
donc (ii) — aucun maillon ne domine — et rend l'orbite elle-même chaotique (E23B),
détruisant (iii). Les trois signatures disparaissent d'un coup : loi pacemaker, loi √,
piégeage propre. **Le selle-nœud du front, lui, ne bouge jamais** (τ-indépendant exact).

Interprétation physique : à τ ≳ t₀ le délai agit comme une **horloge de rafraîchissement**
qui re-synchronise le front à chaque relais (le drive λK·g(u(t−τ)) pointe encore vers le
successeur du motif que le front vient de quitter). À τ ≲ t₀ le feedback revient trop
vite — avant que le relais local (durée t_esc≈1.3) ne soit résolu — et n'impose plus de
phase globale : plusieurs fronts co-existent et interfèrent.

## 6. Points ouverts (pour l'approfondissement annoncé)

1. **Frontière (τ,λ) à haute résolution** autour de τ_c (grille τ∈[1.0, 1.6], λ±0.001) :
   la transition de désynchronisation E22 (à λ=0.9) et la mort-dans-le-chaos E23B
   (à λ≈0.33) se rejoignent-elles en un point critique (τ_c, λ_c') du plan ?
2. **τ_c vs t₀ et t_esc** : τ_c≈1.3 ≈ t_esc(0.9) — coïncidence ou loi τ_c=t_esc(λ) ?
   Tester en variant β (qui change t_esc) et λ.
3. **Nature de l'état multi-lobe** : λ_L(τ) le long de λ=0.9 (est-il chaotique dès
   τ<τ_c même loin de la mort ?) ; distribution du nombre de lobes ; corrélations
   front-front.
4. **N-dépendance** : la largeur du front à petit τ (≈14 motifs) scale-t-elle avec P ?
5. Lien avec la prédiction initiale « fenêtre chaotique remplaçant la gelée à τ=1–2 »
   (`CYCLE_worklog.md` §12.3, résolue qualitativement — reste la cartographie fine).

## 7. Inventaire

| Objet | Chemin (sous `numerics/results/5_petit_tau/`) |
|---|---|
| figL — cycle petit-τ (horloge, kymographes) | `figures/figL_smalltau_cycle.png` |
| figM — destruction de la loi √ | `figures/figM_smalltau_snic.png` |
| figP — désynchronisation, τ_c≈1.3 | `figures/figP_desync.png` |
| figR — frontière cycle-chaos, paradoxe SNIC | `figures/figR_cycle_chaos_frontier.png` |
| données E19 | `E19a_cycle_clock.npz`, `E19b_death.npz`, `E19c_slowing.npz`, `E19d_fixedpoint.npz` |
| données E22 | `data/E22_desync.npz`, `data/E22_lamcheck.npz`, `data/E22_RESULTS.md` |
| données E23B | `data/E23B_frontier.npz` |
| codes | `src/e19a…e19d`, `src/e22_desync.py`, `src/e23b_frontier.py` |
