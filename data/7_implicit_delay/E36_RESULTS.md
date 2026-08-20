# E36 — Retard effectif dans la chaîne de Hopfield non réciproque

## Résultat en bref

Les quatre campagnes seed 42 sont complètes et leurs flux numériques ont été validés jusqu’aux chunks, checkpoints et états finaux. Sur la grille échantillonnée, la chaîne K=6 est stationnaire pour $\lambda\leq0.5$ et devient une onde ordonnée pour $\lambda\geq1$ : le seuil est donc seulement encadré par $0.5<\lambda_c\leq1$. Les deux conditions initiales donnent la même classification à chaque valeur de λ.

Dans le régime ondulatoire, la période de retour et les deux estimateurs indépendants du retard croissent presque linéairement avec K−1. Cette observation **étaye l’interprétation d’un retard effectif**, mais pas l’identité avec une DDE à retard discret.

![Transition de régime et convergence temporelle](figures/E36_regimes_dt.png)

![Échelle avec K, retards et amplitudes](figures/E36_delay_scaling.png)

![Comparaison pilote et campagne de référence](figures/E36_reference_comparison.png)

## Périmètre numérique

- Pilote : N=500, P=25, β=20, t₀=1, seed=42 ; intégrateur RK4.
- A1 : K=6, λ={0, 0.125, 0.25, 0.5, 1, 2, 4, 8}, conditions initiales cohérente et perturbée, t=200.
- A2 : K={4, 6, 8, 11}, λ={1, 2, 4} ; les K=6 sont réutilisés depuis A1.
- Pas principal Δt=0.01 ; contrôles à Δt=0.005 pour λ={1,2,4,8}.
- Référence : N=2000, P=100, K=11, λ={1,2,4}, mêmes deux initialisations et même rapport P/N=0.05.
- Le post-traitement n’a lancé aucune simulation.

Le modèle intégré utilise $g(x)=\tanh(\beta x)$. Pour $k=1,\ldots,K-1$, $t_0\dot U_k=-U_k+Jg(U_k)+\lambda g(U_{k-1})$, tandis que la fermeture est $t_0\dot U_0=-U_0+K_{\rm seq}g(U_{K-1})$. $J$ est la matrice hebbienne et $K_{\rm seq}$ réalise le décalage cyclique des motifs.

## A1 — transition de régime et robustesse à l’initialisation

| λ | cohérente | perturbée | T₁ coh. | T₁ pert. | retard intercouche coh. | saturation moyenne coh. | max \|U\| coh. |
|---:|:---|:---|---:|---:|---:|---:|---:|
| 0 | stationary | stationary | — | — | — | 0.9977 | 1.5800 |
| 0.125 | stationary | stationary | — | — | — | 0.9983 | 1.6810 |
| 0.25 | stationary | stationary | — | — | — | 0.9907 | 1.8060 |
| 0.5 | stationary | stationary | — | — | — | 0.9683 | 2.0968 |
| 1 | ordered_wave | ordered_wave | 10.850 | 10.975 | 2.000 | 0.9417 | 3.9131 |
| 2 | ordered_wave | ordered_wave | 6.950 | 6.950 | 1.250 | 0.9508 | 4.4048 |
| 4 | ordered_wave | ordered_wave | 5.400 | 5.400 | 0.950 | 0.9604 | 6.2259 |
| 8 | ordered_wave | ordered_wave | 4.750 | 4.750 | 0.800 | 0.9670 | 10.0313 |

Le passage observé entre 0.5 et 1 n’est pas une estimation précise de λc : aucune valeur intermédiaire n’a été simulée. La coïncidence des deux initialisations exclut ici une forte dépendance au bassin initial, mais une seule réalisation des motifs est disponible.

La fraction dite « saturée » mesure |βU|>10. Elle reste élevée dans les états stationnaires et ne constitue donc pas, seule, un diagnostic d’oscillation. L’amplitude maximale croît de 1.58 à λ=0 jusqu’à 10.03 à λ=8 ; aucune divergence numérique n’est observée.

## Convergence temporelle

Les 12 comparaisons Δt=0.01/0.005 passent : classification identique, erreur relative maximale sur T₁ 0, sur l’état final 5.126e-05, et écart maximal de saturation 4.665e-06. Cela valide le pas temporel pour les observables rapportées aux λ testés. Parmi elles, quatre contrôles appartiennent à la campagne N=2000/P=100.

## A2 — période et retard en fonction de K

| λ | K | T₁ | CV(T₁) | τ xcorr | τ relais | désaccord | saturation | max \|U\| |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 7.2654 | 0.0640 | 6.4907 | 6.3979 | 1.44% | 0.9089 | 3.7971 |
| 1 | 6 | 10.8914 | 0.0675 | 10.1846 | 10.0808 | 1.02% | 0.9417 | 3.9131 |
| 1 | 8 | 14.4307 | 0.0703 | 13.7486 | 13.6058 | 1.04% | 0.9563 | 3.9616 |
| 1 | 11 | 19.2359 | 0.0723 | 18.8895 | 18.6562 | 1.24% | 0.9684 | 3.9999 |
| 2 | 4 | 4.4240 | 0.0246 | 3.7506 | 3.7196 | 0.83% | 0.9185 | 4.1813 |
| 2 | 6 | 6.9381 | 0.0290 | 6.1927 | 6.1770 | 0.25% | 0.9508 | 4.4048 |
| 2 | 8 | 9.3160 | 0.0274 | 8.5799 | 8.5424 | 0.44% | 0.9651 | 4.7011 |
| 2 | 11 | 12.7859 | 0.0287 | 11.9970 | 11.9519 | 0.38% | 0.9755 | 4.8043 |
| 4 | 4 | 3.4171 | 0.0151 | 2.7447 | 2.7490 | 0.16% | 0.9252 | 6.0844 |
| 4 | 6 | 5.4028 | 0.0126 | 4.7206 | 4.7011 | 0.42% | 0.9604 | 6.2259 |
| 4 | 8 | 7.3179 | 0.0119 | 6.6005 | 6.5670 | 0.51% | 0.9737 | 6.3902 |
| 4 | 11 | 10.0736 | 0.0131 | 9.3628 | 9.3355 | 0.29% | 0.9821 | 6.5238 |

| λ | a | b par couche | R² | pic xcorr unique min. | verdict |
|---:|---:|---:|---:|---:|:---|
| 1 | 2.2710 | 1.7096 | 0.999009 | 1.00 | GO-limited |
| 2 | 0.9166 | 1.1919 | 0.999628 | 1.00 | GO-strong |
| 4 | 0.6170 | 0.9497 | 0.999659 | 1.00 | GO-strong |

Les pics de corrélation sont uniques pour toutes les couches et le désaccord entre estimateurs reste ≤1.439%. La linéarité avec K est très forte (R²≥0.9990), mais le quantum effectif dépend de λ : b=1.710 à λ=1, 1.192 à λ=2 et 0.950 à λ=4. Le cas λ=1 est donc GO-limited : il soutient la mémoire distribuée linéaire en profondeur, pas un retard unitaire universel.

À λ fixé, saturation et amplitude augmentent modérément avec K. À K=6, l’augmentation de λ raccourcit T₁ (10.891, 6.938, 5.403) et le retard effectif, tandis que l’amplitude maximale croît (3.913, 4.405, 6.226).

## Campagne de référence — N=2000, P=100, K=11

| λ | initialisation | régime | GO automatique | transitions couche finale | T₁ | CV(T₁) | retard intercouche | saturation | max \|U\| |
|---:|:---|:---|:---:|---:|---:|---:|---:|---:|---:|
| 1 | coherent | ordered_wave | false | 8 | 18.4000 | 0.0367 | 1.7000 | 0.9664 | 3.9083 |
| 1 | perturbed | ordered_wave | false | 8 | 18.4000 | 0.0367 | 1.7000 | 0.9664 | 3.9083 |
| 2 | coherent | ordered_wave | true | 12 | 12.5500 | 0.0139 | 1.2000 | 0.9746 | 4.7568 |
| 2 | perturbed | ordered_wave | true | 12 | 12.5000 | 0.0144 | 1.2000 | 0.9746 | 4.7571 |
| 4 | coherent | ordered_wave | true | 15 | 9.9750 | 0.0076 | 0.9500 | 0.9815 | 6.5344 |
| 4 | perturbed | ordered_wave | true | 15 | 9.9750 | 0.0072 | 0.9500 | 0.9816 | 6.5341 |

### Réanalyse homogène avec les estimateurs A2

Cette réanalyse utilise uniquement les overlaps enregistrés et applique au pilote et à la référence les mêmes croisements interpolés et la même corrélation vectorielle ; elle ne réintègre pas le modèle.

| λ | initialisation | T₁ A2 | τ xcorr | τ relais | désaccord | pics uniques | blocs |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | coherent | 18.4044 | 17.6791 | 17.6369 | 0.24% | 1.00 | 2 |
| 1 | perturbed | 18.4043 | 17.6793 | 17.6368 | 0.24% | 1.00 | 2 |
| 2 | coherent | 12.5189 | 11.8150 | 11.7325 | 0.70% | 1.00 | 2 |
| 2 | perturbed | 12.5198 | 11.8158 | 11.7297 | 0.73% | 1.00 | 2 |
| 4 | coherent | 9.9916 | 9.2975 | 9.2708 | 0.29% | 1.00 | 3 |
| 4 | perturbed | 9.9925 | 9.2977 | 9.2706 | 0.29% | 1.00 | 3 |

À λ=1, `regime=ordered_wave`, la fraction d’ordre vaut 1 et les deux initialisations donnent T₁=18.4. Le drapeau `go_A1_ordered_wave=false` n’est **pas un échec physique** : la fenêtre contient 8 transitions de la dernière couche, sous le seuil automatique fixe de 10. À λ=2 et 4, 12 et 15 transitions rendent ce même drapeau vrai.

Les quatre contrôles temporels de référence passent ; erreur maximale sur T₁ 0, état final 4.390e-06, saturation 8.331e-07.

### Comparaison au pilote N=500/P=25/K=11

| λ | T₁ pilote | T₁ réf. coh. | T₁ réf. pert. | écart réf. coh. | τ relais/couche pilote | τ relais/couche réf. coh. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.2359 | 18.4044 | 18.4043 | -4.32% | 1.8656 | 1.7637 |
| 2 | 12.7859 | 12.5189 | 12.5198 | -2.09% | 1.1952 | 1.1733 |
| 4 | 10.0736 | 9.9916 | 9.9925 | -0.81% | 0.9335 | 0.9271 |

À estimateur A2 identique, les périodes de référence cohérentes diffèrent du pilote de -4.32%, -2.09%, -0.81% pour λ=1, 2 et 4. Les pics xcorr sont tous uniques et le désaccord entre les deux estimateurs de référence reste inférieur à 0.74%. Pour λ=1 et 2, la fenêtre ne fournit que deux blocs longs : aucune prétention d’intervalle de confiance par blocs n’est faite.

Cette comparaison ne constitue pas une étude de taille à paramètre aléatoire fixé. N et P augmentent ensemble d’un facteur quatre à P/N=0.05 constant, mais la forme du tableau aléatoire change aussi avec ses dimensions malgré la seed identique. Les écarts mélangent donc effets de taille, de charge finie et de réalisation des motifs.

## Vérification des artefacts

- 45 flux validés, 749 chunks NPZ vérifiés.
- 937 fichiers, 1000537045 octets couverts par SHA-256.
- Empreinte agrégée : `0f34589563d3e0a058340b77773264bdb937aaf3c300d164413d6fa4ff1b54e0`.
- États finaux et checkpoints identiques bit à bit ; métadonnées, dimensions, finitude et chronologie contrôlées.
- Une seule grille comporte un dernier intervalle raccourci : λ=1, K=11 (0.01 après une cadence 0.05). Ce dernier callback est exclu des estimateurs ; aucune irrégularité intérieure n’est présente.
- Empreintes détaillées : `E36_CHECKSUMS.sha256`.

## Limites et conclusion

La chaîne à K couches est markovienne dans son espace d’état étendu et contient des dynamiques intermédiaires non linéaires. Leur élimination formelle produit une dépendance à l’histoire généralement distribuée et dépendante de l’état, pas une DDE à retard ponctuel exacte. Une forme Erlang/gamma ne serait justifiée qu’après des hypothèses supplémentaires de linéarisation et de filtres identiques. Les résultats démontrent donc : (i) une transition stationnaire–onde ordonnée encadrée sur la grille testée, (ii) un retard effectif robuste à deux estimateurs, (iii) une croissance quasi linéaire de période et retard avec K, et (iv) une dépendance substantielle du quantum de retard à λ.

Les intervalles par blocs sont descriptifs et corrélés au sein d’une même trajectoire. Avant une revendication statistique générale, les résultats déterminants doivent être reproduits sur plusieurs seeds.

Données structurées : `E36_RESULTS.json`.
