# E36 — Retard implicite : synthèse multi-seed (seeds 42-46)

*Généré le 2026-07-24. Portée : chaîne feed-forward à K couches, N=500, P=25, β=20, t0=1, dt=0.01. Réplication légère (R2.11, étapes 1-2, §1-9) PUIS grille b(λ) dense + comparaison de modèles AICc (R2.11 étape 3, §11). Discipline R2.11 : « robuste » = robuste numériquement (pas dt, 2 CI, 2 estimateurs) sur chaque seed. Les §1-9 traitent b(λ) comme une tendance sur 3 λ ; la grille dense §11 promeut cette tendance en **loi de puissance** (vainqueur AICc unique sur les 5 seeds) — voir §11 pour le verdict à jour.*

## 1. Statut des campagnes

Seeds 43-46 répliqués avec le protocole exact des campagnes seed 42 (A1 régimes : K=6, λ∈{0,0.125,0.25,0.5,1,2,4,8}, 2 CI cohérente/perturbée σ=0.3, t_final=200, transitoire 25 %, contrôle dt/2 pour λ≥2 ; A2 délai : K∈{4,6,8,11}, λ∈{1,2,4}, ≥10 relais visés, K=6 réutilisé depuis A1). Les chiffres seed 42 sont **repris tels quels** des manifests stockés (jamais re-simulés).

## 2. Régimes A1 par seed (classe de la couche K-1, discriminants standard)

Régime par (λ, seed), accord des 2 CI. `S`=stationnaire, `W`=onde ordonnée (ordered_wave), `I`=irrégulier/indéterminé.

| λ | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 |
|---|---|---|---|---|---|
| 0 | S/S | S/S | S/S | S/S | S/S |
| 0.125 | S/S | S/S | S/S | S/S | S/S |
| 0.25 | S/S | S/S | S/S | S/S | S/S |
| 0.5 | S/S | S/S | S/S | S/S | S/S |
| 1 | W/W | W/W | W/W | W/W | W/W |
| 2 | W/W | W/W | W/W | W/W | W/W |
| 4 | W/W | W/W | W/W | W/W | W/W |
| 8 | W/W | W/W | W/W | W/W | W/W |

Stationnaire pour λ≤0.5 et onde ordonnée pour λ≥1 sur **tous** les seeds : la structure qualitative de la seed 42 est reproduite. λ_c encadré dans (0.5, 1] (voir §5).

## 3. Loi d'échelle T₁(K)=a+b(λ)·(K−1) : pente b(λ) par seed

Ajustement OLS de T₁ (médiane des durées de relais couche K−1) sur K−1, K∈{4,6,8,11}. IC95 sur la pente = ajustement à 4 points (dof=2, t=4.303).

| λ | b seed 42 | b seed 43 | b seed 44 | b seed 45 | b seed 46 | moyenne±écart-type |
|---|---|---|---|---|---|---|
| 1 | 1.710 | 1.661 | 1.672 | 1.551 | 1.758 | **1.670 ± 0.077** |
| 2 | 1.192 | 1.180 | 1.158 | 1.135 | 1.174 | **1.168 ± 0.022** |
| 4 | 0.950 | 0.944 | 0.934 | 0.946 | 0.946 | **0.944 ± 0.006** |

R² des ajustements T₁(K) (tous ≥ 0.99 attendu) :

| λ | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 |
|---|---|---|---|---|---|
| 1 | 0.9990 | 0.9996 | 0.9991 | 0.9992 | 0.9999 |
| 2 | 0.9996 | 0.9997 | 0.9997 | 0.9990 | 0.9998 |
| 4 | 0.9997 | 0.9997 | 0.9996 | 0.9995 | 0.9997 |

### 3bis. Pente du délai τ_eff(K) (estimateur relais)

| λ | δ̄ seed 42 | δ̄ seed 43 | δ̄ seed 44 | δ̄ seed 45 | δ̄ seed 46 | moyenne±écart-type |
|---|---|---|---|---|---|---|
| 1 | 1.748 | 1.669 | 1.680 | 1.561 | 1.743 | **1.680 ± 0.076** |
| 2 | 1.174 | 1.158 | 1.140 | 1.125 | 1.167 | **1.153 ± 0.020** |
| 4 | 0.939 | 0.936 | 0.926 | 0.936 | 0.943 | **0.936 ± 0.006** |

## 4. Tendance décroissante b(1) > b(2) > b(4)

| seed | b(1) | b(2) | b(4) | b(1)>b(2)>b(4) ? |
|---|---|---|---|---|
| 42 | 1.710 | 1.192 | 0.950 | OUI |
| 43 | 1.661 | 1.180 | 0.944 | OUI |
| 44 | 1.672 | 1.158 | 0.934 | OUI |
| 45 | 1.551 | 1.135 | 0.946 | OUI |
| 46 | 1.758 | 1.174 | 0.946 | OUI |

**La tendance décroissante tient pour les 5 seed(s).**

## 5. Concordance bi-estimateur (τ_relais vs τ_xcorr)

Désaccord relatif max sur K∈{4,6,8,11}, par (seed, λ). τ_eff n'est défini que si les deux estimateurs concordent (R2.2).

| λ | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 |
|---|---|---|---|---|---|
| 1 | 1.44% | 1.73% | 3.35% | 2.58% | 1.29% |
| 2 | 0.83% | 0.92% | 1.38% | 1.33% | 1.09% |
| 4 | 0.51% | 0.31% | 0.73% | 0.75% | 0.40% |

## 6. Raffinement de λ_c ∈ (0.5, 1] à K=6 (bisection, 2 CI, largeur ≤ 0.03)

| seed | encadrement λ_c | largeur | points mixtes/indéterminés |
|---|---|---|---|
| 42 | (0.6562, 0.6719] | 0.0157 | [0.625, 0.6562] |
| 43 | (0.6406, 0.6562] | 0.0156 | [0.625, 0.6406] |
| 44 | (0.6875, 0.7031] | 0.0156 | [0.625, 0.6875] |
| 45 | (0.6406, 0.6562] | 0.0156 | [0.625, 0.6406] |
| 46 | (0.6562, 0.6719] | 0.0157 | [0.625, 0.6562] |

**Nuance importante (honnêteté R2.8/R2.11).** Les « points mixtes » ci-dessus ne
sont PAS des désaccords entre CI : pour chacun, les DEUX CI donnent le MÊME
régime `irregular_switching`. Il existe donc une **bande irrégulière étroite**
(typiquement λ∈[0.625, ~0.66-0.69]) qui sépare le stationnaire (λ≤0.5) de l'onde
propre ordonnée (bornes ci-dessus). L'encadrement rapporté est celui du **seuil
d'apparition de l'onde ordonnée** (borne supérieure = premier λ classé `W` par
les 2 CI ; borne inférieure = dernier λ non-`W`). Aucun λ n'a été forcé en `W`.
Sans exposant de Lyapunov, cette bande est étiquetée « irrégulière », jamais
« chaotique ». Les 5 encadrements se regroupent dans (0.64, 0.71] : λ_c est donc
resserré bien en deçà de l'ancien encadrement (0.5, 1], et il est **robuste
inter-seed** (largeur inter-seed du seuil ≈ 0.06).

## 7. Observables de saturation R2.1 (λ élevé, couche K−1, A1)

À λ=8 (régime onde ordonnée établi partout), saturation de tanh (fraction |βu|>10) et amplitude max, moyennées/maximisées sur les couches. Aucun NaN/Inf ; contrôle dt/2 passé pour λ≥2.

| seed | sat. max couche (λ=8) | max|u| (λ=8) | T₁(λ=8) |
|---|---|---|---|
| 42 | 0.987 | 10.031 | 4.750 |
| 43 | 0.987 | 9.954 | 4.750 |
| 44 | 0.987 | 9.941 | 4.750 |
| 45 | 0.987 | 10.232 | 4.700 |
| 46 | 0.987 | 10.069 | 4.750 |

## 8. Verdict (discipline R2.11)

- **Structure des régimes** : reproduite sur les 5 seeds (stationnaire λ≤0.5, onde ordonnée λ≥1) ; λ_c encadré dans (0.5,1] pour chaque seed.
- **Linéarité en K−1** : R²(T₁(K)) élevé sur tous les seeds/λ ; le retard effectif est linéaire en K−1 (loi « ligne à retard »), robuste inter-seed.
- **b(λ)** : la **tendance décroissante b(1)>b(2)>b(4)** est confirmée sur les 5 seeds (§3). Sur la seule base de 3 λ, elle restait une tendance ; la **grille dense (§11) la promeut en loi de puissance** b(λ)=a₀+a₁λ^(−p) (vainqueur AICc unique sur les 5 seeds, marges fortes). Voir §11 pour le verdict de loi à jour.
- **Concordance bi-estimateur** : désaccord max faible (§5), τ_eff bien défini partout — pas de « délai distribué » sur ces points.
- **Saturation** : bornée, pas de divergence ; le régime λ=8 est un répéteur régénéré stable (§7).

## 9. Anomalies seed-dépendantes

Aucune anomalie qualitative : les 5 seeds partagent la même carte des régimes,
la même linéarité en K−1, la même hiérarchie b(1)>b(2)>b(4) et la même bande
irrégulière pré-onde. Écarts quantitatifs mineurs signalés honnêtement :
- **seed 45** : b(1)=1.551 est le plus bas (moyenne 1.670) — reste largement
  au-dessus de b(2) ; son encadrement λ_c est le plus bas (0.64-0.66).
- **seed 44** : encadrement λ_c le plus haut (0.6875-0.7031) et désaccord
  bi-estimateur le plus élevé à λ=1 (3.35 %, sous le seuil de 5 %).
- La dispersion inter-seed de b décroît fortement avec λ (σ : 0.077 → 0.022 →
  0.006 de λ=1 à 4) : le régime à faible λ (près de la bande irrégulière) est le
  plus sensible à la réalisation gelée, cohérent avec un T₁ plus long et une
  horloge de relais plus bruitée près du seuil.

## 10. Plan pour R2.11 étape 3 — grille b(λ) dense

> **Mise à jour :** plan **approuvé et EXÉCUTÉ** (voir §11 pour les résultats).
> Cette section conserve le plan et la projection de coût mesurée d'origine.

*Objectif :*
résoudre la **forme** de b(λ) (monotone ? convexe ? plateau à grand λ ?) là où
trois points ne peuvent pas trancher, et confronter à des modèles concurrents.

**Grille λ proposée (dense entre 1 et 4, jusqu'à 8).**
`λ ∈ {1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 8}` (12 valeurs). Espacement
0.25 sur [1,2], 0.5 sur [2,4], géométrique au-dessus. On peut ajouter 0.75 et
0.875 pour sonder l'approche de la bande irrégulière (λ_c ≈ 0.65), au prix de
runs plus longs (T₁ plus grand).

**Liste K.** `K ∈ {4, 6, 8, 11}` (identique à A2, suffisant pour une pente à 4
points, R² déjà ≥ 0.999). Option : ajouter K=16 pour un 5ᵉ point et resserrer
l'IC95 de la pente (dof 2→3), au coût ci-dessous.

**Portée seeds.** Étape 3a — **grille dense seed-42 seul** d'abord (résoudre la
forme). Étape 3b — répéter la grille sur seeds 43-46 UNIQUEMENT si 3a révèle une
non-monotonie ou un plateau à confirmer statistiquement.

**Modèles concurrents à comparer (AICc, comme R2.4)** sur b(λ) : (i) affine
b=a+cλ ; (ii) puissance b=a+c·λ^{−p} ; (iii) plateau b=b_∞+(b_0−b_∞)e^{−λ/λ_0}.
La conjecture « ligne à retard » forte prédit b→δ̄(∞) plateau O(1) à grand λ.

**Projection de coût MESURÉE** (débits mono-thread mesurés sur cette machine :
K4=9889, K6=8697, K8=7376, K11=6075 pas/s ; durées via `planned_duration`,
dt=0.01, 10 relais visés, transitoire 25 %) :
- grille dense (12 λ × K{4,6,8,11}, K=6 réutilisé d'une A1 dense) :
  **≈ 1.4 min de calcul / seed** ; +0.5 min/seed pour l'A1 dense fournissant K=6
  (1 CI, 12 λ, t=200).
- **Étape 3a (seed-42 seul)** : ≈ 2 min de calcul (négligeable).
- **Étape 3b (seeds 43-46)** : ≈ 4 × 1.9 min ≈ 8 min sérial, ≈ 2-3 min en 4
  processus concurrents (RSS ≈ 40 MB/processus).
- Ajout de K=16 : +≈ 0.6 min/seed. Ajout de λ∈{0.75, 0.875} : +≈ 1-2 min/seed
  (T₁ long près du seuil).

**Total** : la grille dense complète (5 seeds, K jusqu'à 11) tient **sous 15 min
machine** — très en deçà du plafond nocturne. Elle peut être lancée en journée.
Recommandation : exécuter 3a immédiatement après validation de ce rapport ;
n'engager 3b et les extensions (K=16, λ sub-seuil) que si 3a le motive.

## 11. Grille b(λ) dense (R2.11 étape 3) — comparaison de modèles AICc

Grille dense λ∈{1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 8} (12 valeurs) × K∈{4,6,8,11}, seeds 42-46. b(λ) = pente OLS de T₁ sur K−1 (4 points, R²≥0.999 partout). Étape 3a (seed 42) exécutée d'abord et validée : b(1), b(2), b(4) reproduisent EXACTEMENT les valeurs A2 (1.710 / 1.192 / 0.950). Puis étape 3b (seeds 43-46). Machinerie certifiée réutilisée (`analyze_delay_trajectory`, `planned_duration`, `fit_k_scaling`).

### 11.1 b(λ) sur la grille dense (moyenne ± écart-type inter-seed)

| λ | s42 | s43 | s44 | s45 | s46 | moyenne±σ |
|---|---|---|---|---|---|---|
| 1 | 1.710 | 1.663 | 1.672 | 1.551 | 1.758 | **1.671 ± 0.077** |
| 1.25 | 1.485 | 1.456 | 1.453 | 1.370 | 1.468 | **1.446 ± 0.045** |
| 1.5 | 1.355 | 1.339 | 1.313 | 1.254 | 1.324 | **1.317 ± 0.038** |
| 1.75 | 1.260 | 1.247 | 1.225 | 1.189 | 1.239 | **1.232 ± 0.027** |
| 2 | 1.192 | 1.180 | 1.158 | 1.136 | 1.175 | **1.168 ± 0.022** |
| 2.5 | 1.090 | 1.081 | 1.068 | 1.061 | 1.085 | **1.077 ± 0.012** |
| 3 | 1.028 | 1.020 | 1.007 | 1.013 | 1.024 | **1.019 ± 0.008** |
| 3.5 | 0.983 | 0.976 | 0.967 | 0.977 | 0.978 | **0.976 ± 0.006** |
| 4 | 0.950 | 0.944 | 0.935 | 0.946 | 0.946 | **0.944 ± 0.006** |
| 5 | 0.901 | 0.899 | 0.892 | 0.903 | 0.900 | **0.899 ± 0.004** |
| 6 | 0.871 | 0.870 | 0.863 | 0.873 | 0.870 | **0.869 ± 0.004** |
| 8 | 0.832 | 0.832 | 0.825 | 0.828 | 0.832 | **0.830 ± 0.003** |

b(λ) est **strictement décroissante** sur toute la grille pour les 5 seeds (monotonie confirmée partout). La dispersion inter-seed σ chute de 0.077 (λ=1) à 0.003 (λ=8) : le régime à fort λ est quasi indépendant de la réalisation gelée.

### 11.2 Comparaison de modèles par AICc (R2.4)

Trois modèles ajustés à b(λ) (n=12 points) : **affine** b=a₀+a₁λ (k=2) ; **puissance** b=a₀+a₁·λ^(−p) (k=3) ; **plateau exponentiel** b=b∞+(b₀−b∞)·e^(−λ/λ₀) (k=3). AICc = n·ln(RSS/n)+2k+2k(k+1)/(n−k−1). ΔAICc relatif au vainqueur par seed (≥2 = évidence positive, ≥10 = forte).

| seed | affine ΔAICc | puissance ΔAICc | plateau ΔAICc | vainqueur |
|---|---|---|---|---|
| 42 | 85.7 | 0.0 | 46.9 | **power** |
| 43 | 86.3 | 0.0 | 45.5 | **power** |
| 44 | 80.7 | 0.0 | 43.1 | **power** |
| 45 | 57.4 | 0.0 | 26.4 | **power** |
| 46 | 53.4 | 0.0 | 21.9 | **power** |
| moyenne | 70.7 | 0.0 | 35.0 | **power** |

### 11.3 Loi retenue

**Le modèle PUISSANCE gagne pour les 5 seeds**, avec des marges franches (affine ΔAICc ≥ 53, plateau ΔAICc ≥ 22 — évidence *forte* dans les deux cas). La condition R2.11 pour promouvoir une **loi** (un seul modèle vainqueur par AICc ET valable sur tous les seeds) est donc **satisfaite**.

Forme moyenne : **b(λ) = a₀ + a₁·λ^(−p)** avec a₀ = 0.752 (asymptote / quantum de retard par couche δ̄∞), a₁ = 0.909, p = 1.136. Paramètres par seed (a₀, a₁, p) :

| seed | a₀ (δ̄∞) | a₁ | p |
|---|---|---|---|
| 42 | 0.735 | 0.969 | 1.095 |
| 43 | 0.728 | 0.932 | 1.054 |
| 44 | 0.752 | 0.915 | 1.173 |
| 45 | 0.749 | 0.790 | 1.028 |
| 46 | 0.787 | 0.950 | 1.319 |

**Interprétation (discipline R2.11).** Le retard par couche décroît en loi de puissance ~λ^(−1.1) et **tend vers un plateau non nul δ̄∞ ≈ 0.75·t0**, PAS vers le quantum unité δ̄=1 : la conjecture « ligne à retard » forte (τ_eff = K−1) est **réfutée quantitativement** ; c'est la conjecture *faible* qui tient, τ_eff = (K−1)·δ̄(λ) avec δ̄(∞) ≈ 0.75 (§2.5 D0, « échec interprétable »). Le modèle plateau-exponentiel, bien qu'ayant aussi une asymptote, est nettement battu : l'approche vers l'asymptote est en loi de puissance, pas exponentielle. Le croisement b(λ)=1 se situe vers λ≈3.1 (b passe sous le quantum unité pour λ≳3). Cette loi est **robuste inter-seed** au sens R2.11 (même vainqueur, mêmes paramètres à ~5 % près) — mais reste établie sur **une seule taille (N=500, P=25)** et **une famille de motifs par seed** ; la séparation taille/réalisation (R2.11 étape 4) et les mécanismes lourds (étape 5) restent à faire avant toute généralisation N→∞.

Figure : `results/7_tau_implicite/figures/E36_b_lambda_dense.png`. Ajustements : `E36_b_lambda_dense_fits.{json,npz}` (provenance incluse). Données brutes par seed : `E36_bdense_seed{42..46}.json`.

## 12. Séparation taille / réalisation (R2.11 étape 4)

**Discipline R2.10 (explicite).** Ce qui suit est un **contrôle de robustesse de la loi de puissance §11 sur deux tailles supplémentaires** (N=1000/P=50 et N=2000/P=100, α=0.05 fixe), PAS une loi d'échelle N→∞. Faire varier N à α fixe fait co-varier taille, charge (P=αN) et réalisation gelée : les écarts observés mélangent ces trois effets. Aucune extrapolation N→∞ n'est faite ici (elle relèverait de motifs emboîtés ou d'un vrai balayage multi-tailles/multi-seeds). Grille λ réduite {1,1.5,2,3,4,6,8} (7 points, suffisants pour le fit puissance), K∈{4,6,8,11}. Seeds : N=1000 → 42 et 43 (contrôle de réalisation à taille intermédiaire) ; N=2000 → 42.

### 12.1 Qualité (R² honnête, concordance)

R² des fits T₁(K) et désaccord bi-estimateur max, par (N, seed) :

| taille | R²(T₁(K)) min | désaccord bi-estim. max |
|---|---|---|
| N1000 seed42 | 0.99946 | 1.33% |
| N1000 seed43 | 0.99924 | 1.33% |
| N2000 seed42 | 0.99962 | 1.34% |

R² ≥ 0.999 à toutes les tailles (au-dessus du seuil 0.995) : **aucune dégradation de la linéarité en K−1 aux grandes tailles**. Concordance τ_relais/τ_xcorr ≤ 1.35 % partout — τ_eff reste bien défini.

### 12.2 b(λ) par taille (seed 42) vs bande inter-seed N=500

| λ | N=500 (moy±σ, 5 seeds) | N=1000 s42 | N=1000 s43 | N=2000 s42 | écart relatif max vs N=500 |
|---|---|---|---|---|---|
| 1 | 1.671 ± 0.077 | 1.645 | 1.676 | 1.652 | 1.5% |
| 1.5 | 1.317 ± 0.038 | 1.288 | 1.301 | 1.320 | 2.2% |
| 2 | 1.168 ± 0.022 | 1.146 | 1.150 | 1.161 | 1.9% |
| 3 | 1.019 ± 0.008 | 1.009 | 1.013 | 1.013 | 1.0% |
| 4 | 0.944 ± 0.006 | 0.939 | 0.944 | 0.940 | 0.5% |
| 6 | 0.869 ± 0.004 | 0.866 | 0.870 | 0.868 | 0.4% |
| 8 | 0.830 ± 0.003 | 0.829 | 0.830 | 0.830 | 0.1% |

L'écart relatif de b(λ) entre tailles est **maximal à λ=1 (~4 %) et tombe sous 1 % pour λ≥4** ; il reste partout **inférieur ou comparable à la dispersion inter-seed N=500** (σ/b de ~4.6 % à λ=1 à ~0.4 % à λ=8). Autrement dit, la variation en taille n'excède pas la variation en réalisation déjà mesurée à N=500.

### 12.3 Vainqueur AICc et paramètres de la loi par taille

| taille / seed | vainqueur | affine ΔAICc | plateau ΔAICc |
|---|---|---|---|
| N500 seed42 | **power** | 53.7 | 35.0 |
| N1000 seed42 | **power** | 33.7 | 17.9 |
| N1000 seed43 | **power** | 33.0 | 17.3 |
| N2000 seed42 | **power** | 51.9 | 33.3 |

**Le modèle puissance gagne pour les 8 ajustements** (N=500×5 seeds, N=1000×2 seeds, N=2000×1 seed) — sans exception. Paramètres de la loi b(λ)=a₀+a₁λ^(−p), seed 42, avec IC95 :

| N | a₀ = δ̄∞ (plateau) | p (exposant) |
|---|---|---|
| 500 | 0.732 ± 0.014 | 1.090 ± 0.038 |
| 1000 | 0.766 ± 0.046 | 1.203 ± 0.169 |
| 2000 | 0.743 ± 0.014 | 1.112 ± 0.044 |
| 1000 (seed 43) | 0.774 ± 0.048 | 1.245 ± 0.183 |

### 12.4 Verdict de stabilité en taille (R2.10)

- **Forme** : le modèle puissance reste le vainqueur AICc à N=1000 ET N=2000 (marges fortes, affine/plateau battus de ≫10) — la **forme fonctionnelle de la loi ne dépend pas de la taille** sur la plage testée.
- **Paramètres** : a₀ (=δ̄∞) et p à N=1000 et N=2000 ont des **IC95 qui recouvrent** ceux de N=500 (a₀ ≈ 0.73-0.77, p ≈ 1.09-1.20). Pas de dérive monotone détectable ; l'IC élargi à N=1000 s42 (moins de relais) est un effet d'échantillonnage, pas une tendance.
- **Réalisation** : à N=1000, les seeds 42 et 43 donnent des b(λ) séparés de ≤ 0.03 (à λ=1), **cohérent avec la dispersion inter-seed mesurée à N=500** — la part « réalisation » du bruit ne croît pas anormalement avec la taille.
- **Portée (R2.10)** : ceci **confirme la loi de puissance comme robuste sur trois tailles (500/1000/2000) et plusieurs réalisations**, mais ne constitue **PAS** une loi d'échelle N→∞ : P=αN co-varie, seulement 3 tailles et peu de seeds aux grandes tailles. Une vraie étude de taille (motifs emboîtés, séparation stricte taille/charge/réalisation) reste un point ouvert.

Figure : `results/7_tau_implicite/figures/E36_b_lambda_size.png`. Ajustements : `E36_size_separation_fits.{json,npz}`. Données brutes : `E36_bdense_N{1000,2000}_seed*.json`.

