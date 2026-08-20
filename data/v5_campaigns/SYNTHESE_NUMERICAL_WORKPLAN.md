# Synthèse du numerical workplan

**Périmètre.** Ce document résume exclusivement les calculs, audits et analyses
nouveaux réalisés dans le cadre de
[`NUMERICAL_WORKPLAN.md`](../paper_v5/NUMERICAL_WORKPLAN.md). Il ne résume pas
les résultats historiques du projet. Les résultats réutilisés dans l'article
mais obtenus avant ce workplan ne sont donc pas décrits ici.

**État de la synthèse : 30 juillet 2026, 09:20 CDT.** Les conclusions sont
classées ci-dessous en résultats validés, résultats provisoires et gates non
franchis. Les contrôles de front épinglé N5B sont encore en cours; ils sont
explicitement séparés des résultats terminés.

## 1. Résumé exécutif

| Tâche | Travail nouveau | Statut scientifique | Résultat principal |
|---|---|---|---|
| **N1** | Comparaison statique–dynamique sur 20 réalisations de désordre | **Validé** | Le motif qui meurt dynamiquement est le motif statique extrême dans 20/20 réalisations. Pour les 19 réalisations où l'extrême est identifiable sans ambiguïté, le seuil statique et l'onset dynamique coïncident à \(10^{-3}\). |
| **N2** | Lois locales près du pli sur cinq réalisations | **Partiel; gate final non franchi** | Quatre réalisations donnent un exposant dynamique effectif \(0.410\!-\!0.539\), compatible à ces tailles avec une loi en racine carrée et très défavorable à une loi logarithmique. La cinquième ne sonde pas correctement l'asymptotique locale. Le premier balayage statique suivait la mauvaise branche et ne doit pas être utilisé. |
| **N3** | Réanalyse seed-aware de 303 fichiers E30 | **Analyse terminée** | Les intervalles d'incertitude tiennent désormais compte des réalisations de désordre. À \(\alpha=0.05\), l'exposant effectif de largeur vaut \(0.4308\), IC95 \([0.3964,0.4652]\); cette fenêtre finie ne permet ni de mesurer exactement \(1/2\), ni de rejeter proprement une limite asymptotique en \(N^{-1/2}\) sans corrections de taille finie. |
| **N5A** | Robustesse multi-seed à délai court | **Crossover validé; frontière cycle–chaos non résolue** | Les cinq réalisations montrent une perte de cohérence dans une fenêtre étroite, avec milieux \(\tau\simeq1.233\!-\!1.338\). La grille de Lyapunov est trop clairsemée pour établir une frontière cycle–chaos robuste. |
| **N5B** | Spectre à délai long près du pli | **Branche mémoire terminée; contrôles supplémentaires actifs** | Sur 200/200 points spectraux et cinq réalisations, aucune paire complexe ne traverse l'axe imaginaire. Le résultat soutenu est un ramollissement spectral sans préemption de Hopf résolue. |
| **N6** | Spectres de Lyapunov float64 multi-seed | **Validé à \(\lambda=0.31\)** | Les cinq réalisations possèdent au moins trois exposants de Lyapunov positifs robustes au contrôle \(dt=0.01\to0.005\). L'attracteur est donc hyperchaotique à ce point de paramètres. |
| **N8** | Audit des états mobiles à forte charge | **Résultat de trajectoire robuste; classification dynamique provisoire** | 96/100 trajectoires satisfont un critère strict de récupération séquentielle, dans 19/20 cellules pour les cinq seeds. Les données de Lyapunov ne couvrent encore qu'une seed; elles ne résolvent aucun exposant positif. |

Les deux nouveaux résultats les plus directement utilisables dans le corps de
l'article sont donc **N1**, qui relie l'extrême de l'ensemble de bifurcations à
l'onset dynamique réalisation par réalisation, et **N6**, qui établit la nature
hyperchaotique de l'attracteur pour plusieurs réalisations de désordre.

## 2. Protocole commun, traçabilité et contrôles

Les campagnes ont été conduites en float64 sur CPU, avec au plus quatre
processus monothreads et sans superposition de deux familles de simulations
lourdes. MLX/Metal n'a pas été retenu pour la production : les calculs de plis,
de racines et de Lyapunov demandent ici une précision double, et le backend
Metal n'était pas disponible de manière fiable dans l'environnement
d'exécution.

Chaque campagne lourde a été précédée d'un smoketest et d'un benchmark. Les
sorties sont checkpointées point par point ou seed par seed afin que les calculs
terminés ne soient pas répétés après une interruption. Le dossier historique
sur le Desktop est resté en lecture seule; toutes les nouvelles données sont
dans [`runs/`](runs/), les analyses dans [`reports/`](reports/) et les figures
dans [`figures/`](figures/).

Les seeds n'ont pas été écartées parce que leur valeur était atypique. Une
exclusion n'a été admise que sur un critère de qualité indépendant du résultat
testé : non-identifiabilité de l'extrême, mauvaise branche, non-convergence ou
attracteur différent. Les valeurs exclues et leur raison ont été conservées.

## 3. N1 — identité entre extrême statique et onset dynamique

### Question et calcul réalisé

N1 devait tester directement le mécanisme central du papier : dans une
réalisation finie du désordre, le seuil collectif est-il sélectionné par
l'extrême de l'ensemble des bifurcations associées aux motifs?

Pour les seeds 42–61 à \(N=2000\) et \(P=100\), les plis statiques des branches
ont été confrontés à un bracket dynamique indépendant de disparition du cycle.
Un audit complémentaire a contrôlé le pas de temps, la largeur du bracket, les
histoires initiales et la possibilité d'atteindre une branche stationnaire
différente après l'arrêt.

### Résultat

- L'indice du dernier motif statiquement survivant est identique à celui du
  motif dont la disparition arrête le cycle dans **20/20 seeds**.
- Pour les **19/19 seeds admissibles**, la valeur critique de l'extrême statique
  et le milieu du bracket d'onset dynamique diffèrent de moins de \(10^{-3}\).
  Le plus grand écart retenu est \(7.32\times10^{-4}\).
- La seed 48 n'est pas retirée parce qu'elle serait un « outlier » : les
  brackets des deux premiers extrêmes statiques se recouvrent, donc l'extrême
  unique requis par le test n'est pas identifiable. Ses données et la raison
  d'exclusion restent présentes dans le rapport.
- Lorsqu'on compare la dynamique à la branche stationnaire effectivement
  pertinente après l'arrêt, les 20 seeds sont compatibles à \(10^{-3}\).

La conclusion à employer est une **égalité à la résolution \(10^{-3}\)**, et
non une identité mathématique exacte. Les écarts observés à \(10^{-4}\) sont du
même ordre que la résolution dynamique, la sélection de branche et les effets
de bassin; ils ne portent pas de contenu physique établi dans les données
actuelles.

### Apport à l'article

N1 fournit la démonstration multi-seed la plus directe du mécanisme proposé :
le seuil dynamique global et l'identité du motif terminal sont prédits par une
statistique extrême de l'ensemble statique de bifurcations, réalisation par
réalisation.

### Données et figures

- Figure article : [`Figure3_finite_size_selection.pdf`](../paper_v5/figures_ready/Figure3_finite_size_selection.pdf)
- Figure directe statique–dynamique :
  [`N1_static_extreme_vs_dynamic_onset.svg`](figures/N1_static_extreme_vs_dynamic_onset.svg)
- Figure d'identité des motifs :
  [`N1_multiseed_motif_identity.pdf`](figures/N1_multiseed_motif_identity.pdf)
- Figure des seuils à \(10^{-3}\) :
  [`N1_multiseed_lambda_identity_1e-3.pdf`](figures/N1_multiseed_lambda_identity_1e-3.pdf)
- Tableau réalisation par réalisation :
  [`N1_static_extreme_vs_dynamic_onset.csv`](reports/N1_static_extreme_vs_dynamic_onset.csv)
- Résumé machine-readable et règle d'admissibilité :
  [`N1_static_extreme_vs_dynamic_onset_summary.json`](reports/N1_static_extreme_vs_dynamic_onset_summary.json)
- Résumé incluant les branches stationnaires :
  [`N1_multiseed_identity_summary.json`](reports/N1_multiseed_identity_summary.json)

## 4. N2 — lois locales de ralentissement près du pli

### Question et calcul réalisé

N2 devait distinguer quantitativement la loi locale dynamique au-dessus du pli,
la loi de ramollissement statique sous le pli, et la géométrie de forme normale.
Cinq seeds admissibles de N1 ont été simulées : 42, 47, 51, 52 et 60. Les
continuations pseudo-arclength passent bien à travers le pli, les résidus de
l'équation augmentée sont conformes et les 80 mesures dynamiques prévues sont
présentes sans censure.

### Résultat actuellement soutenu

Les exposants libres obtenus du côté dynamique sont :

| seed | exposant effectif |
|---:|---:|
| 42 | 0.539 |
| 47 | 0.520 |
| 52 | 0.410 |
| 60 | 0.439 |

Pour ces quatre réalisations, la loi de puissance est très fortement préférée
à l'alternative logarithmique. À la taille finie et sur la fenêtre accessible,
la plage \(0.410\!-\!0.539\) est compatible avec le ralentissement en racine
carrée attendu près d'un pli. Ces ajustements ne mesurent cependant pas avec
précision un exposant universel exactement égal à \(1/2\).

La seed 51 ne montre pas la divergence attendue autour du pli statique utilisé
pour le scan local. Son bracket dynamique est décalé d'environ
\(2.5\times10^{-4}\), quantité négligeable pour le test N1 à \(10^{-3}\), mais
plus grande que les plus petits offsets \(10^{-5}\) de N2. Elle ne doit donc pas
être utilisée pour inférer un exposant local.

### Gate non franchi et correction

Le premier calcul N2B du côté statique a sélectionné la branche selle instable
pour quatre seeds, parce que Newton était amorcé exactement au pli. Les racines
réelles positives obtenues étaient incompatibles avec la branche mémoire stable
requise. Ces résultats sont **invalides pour la physique** et ont été conservés,
sans être mélangés aux données acceptées, dans
[`N2_INVALID_STATIC_BRANCH_20260729/`](runs/N2_INVALID_STATIC_BRANCH_20260729/).

Un audit sans écriture, initialisé depuis l'état terminal stable de N1, retrouve
une racine réelle négative à \(\delta=10^{-5}\) pour les cinq seeds :
\(-0.02076,-0.01223,-0.01896,-0.02501,-0.02035\). Le diagnostic de branche est
donc établi et le code a été corrigé. Le recalcul statique dense, l'analyse de
sensibilité aux fenêtres et le gate final restent à effectuer. En conséquence,
N2 ne remplace pas encore le panneau historique de ralentissement local dans
l'article.

### Données

- Résumés dynamiques et géométriques :
  [`seed 42`](runs/N2/summary_seeds_42.json),
  [`seed 47`](runs/N2/summary_seeds_47.json),
  [`seed 51`](runs/N2/summary_seeds_51.json),
  [`seed 52`](runs/N2/summary_seeds_52.json) et
  [`seed 60`](runs/N2/summary_seeds_60.json)
- Sorties statiques invalidées, conservées pour audit :
  [`N2_INVALID_STATIC_BRANCH_20260729/`](runs/N2_INVALID_STATIC_BRANCH_20260729/)
- Diagnostic détaillé et chronologie de la correction :
  [`RUN_STATUS.md`](RUN_STATUS.md)

## 5. N3 — statistiques de seuils avec incertitudes seed-aware

### Question et calcul réalisé

N3 est une réanalyse nouvelle des 303 fichiers de la campagne E30 historique.
Les motifs d'une même matrice de désordre n'ont plus été traités comme des
réalisations indépendantes. Les intervalles ont été obtenus par 5000
rééchantillonnages des seeds, avec diagnostics de dépendance, tests de modèles
et analyses leave-one-size-out.

### Résultats à charge fixée, \(\alpha=0.05\)

- Largeur de la distribution des seuils :
  \(N^{-b}\) avec \(b=0.43078\), médiane bootstrap \(0.43009\) et IC95
  \([0.39638,0.46520]\).
- Intervalle entre extrêmes :
  exposant \(d=0.27582\), IC95 \([0.22538,0.33040]\).
- Le modèle de puissance pure n'est préféré au modèle avec plancher additif que
  de \(\Delta\mathrm{AICc}=2.11\) : la sélection reste faible.
- La prédiction d'extrême gaussien indépendant reproduit raisonnablement les
  maxima, mais n'est pas exacte. L'erreur signée moyenne sur le maximum vaut
  \(+0.00734\), l'erreur absolue maximale environ \(0.02745\), et sa forme
  imposée pour l'écart des extrêmes est défavorisée de
  \(\Delta\mathrm{AICc}=8.37\) par rapport à une puissance libre.

L'exposant effectif \(0.43\) est quantitativement proche de \(1/2\), mais son
intervalle d'ajustement sur les tailles simulées n'inclut pas \(1/2\).
Scientifiquement, cela signifie que les données actuelles ne mesurent pas une
loi exactement en racine de \(N\). Cela ne suffit pas non plus à exclure une
limite asymptotique \(N^{-1/2}\), car les corrections de taille finie n'ont pas
été paramétrées ni contrôlées sur des tailles beaucoup plus grandes.

### Résultats à \(P=100\) fixé

- Largeur : exposant effectif \(b=0.90312\), IC95
  \([0.88596,0.92102]\).
- Un modèle puissance plus plancher est préféré à la puissance pure de
  \(\Delta\mathrm{AICc}=10.64\).
- Intervalle entre extrêmes : exposant \(0.75496\), IC95
  \([0.73520,0.77489]\).

Ces résultats montrent que les deux protocoles thermodynamiques — charge fixée
et nombre de motifs fixé — ne doivent pas être confondus.

### Données et figure

- Analyse complète :
  [`E30_seed_aware_analysis.json`](runs/N3/E30_seed_aware_analysis.json)
- Tableau synthétique :
  [`N3_E30_seed_statistics.csv`](reports/N3_E30_seed_statistics.csv)
- Figure :
  [`N3_E30_seed_aware.pdf`](figures/N3_E30_seed_aware.pdf)

## 6. N5A — perte de cohérence à délai court

### Calcul réalisé

Les cinq seeds 42–46 ont été analysées sur le scan de cohérence et sur une
grille locale destinée à séparer cycles cohérents, mouvement désynchronisé et
chaos. Les cinq productions sont complètes, avec 279 cellules de frontière par
seed.

### Résultat

Les milieux interpolés de la perte de cohérence sont :

| seed | \(\tau\) au milieu du crossover |
|---:|---:|
| 42 | 1.28225 |
| 43 | 1.23475 |
| 44 | 1.33786 |
| 45 | 1.23384 |
| 46 | 1.23338 |

La fenêtre \(\tau\simeq1.23\!-\!1.34\) est donc robuste au désordre au sens où
les cinq réalisations présentent le même crossover qualitatif dans une plage
étroite.

En revanche, le gate complet échoue : la plupart des lignes de frontière ne
possèdent que deux points résolus par Lyapunov au lieu des quatre prescrits, et
les classifications de trajectoires ne déterminent pas une frontière
périodique–chaotique robuste entre seeds. La figure N5A est donc un diagnostic;
elle ne remplace pas la figure historique validée de l'article.

### Données et figure

- Agrégation complète : [`analysis.json`](runs/N5A/analysis.json)
- Figure diagnostique :
  [`N5A_delay_boundary.pdf`](figures/N5A_delay_boundary.pdf)

## 7. N5B — stabilité à délai long

### Calcul terminé sur la branche mémoire

Pour les seeds 42–46, les spectres ont été calculés à huit délais et cinq
distances du pli, avec deux résolutions spectrales \(M=48\) et \(M=64\), soit
**200/200 points**.

Les plis trouvés sont :

| seed | \(\lambda_{\mathrm{fold}}\) | maximum de \(\Re z_{\mathrm{complexe}}\) à \(M=64\) |
|---:|---:|---:|
| 42 | 0.23400074 | \(-0.0047675\) |
| 43 | 0.21919712 | \(-0.0139634\) |
| 44 | 0.22506768 | \(-0.0078914\) |
| 45 | 0.24499230 | \(-0.0123338\) |
| 46 | 0.22983400 | \(-0.0151893\) |

Toutes les parties réelles complexes sont négatives aux deux résolutions. Aucun
croisement candidat ou raffiné n'a été détecté; aucun test non linéaire de Hopf
n'a donc été déclenché. Les 200 racines sélectionnées passent le test de résidu.
La sélection de famille est cohérente modulo conjugaison dans 198/200 cellules.
Dans deux cellules de la seed 45, \(M=48\) et \(M=64\) choisissent des familles
différentes, mais les deux familles restent stables.

Le résultat soutenu est : **ramollissement du spectre complexe sans préemption
de Hopf résolue sur les délais, offsets et seeds testés**. On ne peut pas
conclure à l'absence mathématique de toute bifurcation de Hopf hors de cette
grille, ni revendiquer un attracteur né d'un Hopf.

### Contrôles encore actifs

Les contrôles du front épinglé sur les 20 seeds N1, six délais et trois
distances sous le pli sont encore en cours. Ils sont supplémentaires et ne
modifient pas la conclusion déjà acquise sur les cinq branches mémoire. Leur
état de reprise autoritatif se trouve dans
[`recovery_state.json`](runs/N5B/recovery_state.json).

### Données

- Branches mémoire :
  [`seed 42`](runs/N5B/memory_scan_N2000_s42_production.json),
  [`seed 43`](runs/N5B/memory_scan_N2000_s43_production.json),
  [`seed 44`](runs/N5B/memory_scan_N2000_s44_production.json),
  [`seed 45`](runs/N5B/memory_scan_N2000_s45_production.json) et
  [`seed 46`](runs/N5B/memory_scan_N2000_s46_production.json)
- Racines point par point : [`runs/N5B/`](runs/N5B/)
- État restart-safe : [`recovery_state.json`](runs/N5B/recovery_state.json)

## 8. N6 — nature hyperchaotique de l'attracteur

### Calcul réalisé

Les 34 spectres demandés ont été calculés à \(N=2000\), \(P=100\), avec les
huit premiers exposants, des fenêtres cumulées jusqu'à \(T=4000\), des
estimations par blocs et un contrôle \(dt=0.01\to0.005\) à
\(\lambda=0.31\) pour les cinq seeds 42–46.

### Résultat validé

À \(\lambda=0.31\), les trois premiers exposants restent positifs pour toutes
les seeds et pour les deux pas de temps :

| seed | \((\Lambda_1,\Lambda_2,\Lambda_3)\), \(dt=0.01\) | \((\Lambda_1,\Lambda_2,\Lambda_3)\), \(dt=0.005\) |
|---:|---:|---:|
| 42 | (0.04620, 0.02891, 0.01957) | (0.03807, 0.02958, 0.01145) |
| 43 | (0.05369, 0.03562, 0.02219) | (0.05154, 0.03389, 0.02261) |
| 44 | (0.05112, 0.02927, 0.01786) | (0.05532, 0.03133, 0.02056) |
| 45 | (0.04667, 0.02894, 0.01620) | (0.05237, 0.03628, 0.02657) |
| 46 | (0.04956, 0.03587, 0.02138) | (0.05026, 0.03512, 0.02081) |

Le troisième exposant le plus petit de tous ces contrôles vaut encore
\(0.01145>0\). Le gate conservateur est donc franchi : **l'attracteur à
\(\lambda=0.31\) possède au moins trois directions instables robustes dans les
cinq réalisations testées**, ce qui établit l'hyperchaos au-delà d'une seed de
référence.

Les points seed 42 à \(\lambda=0.29\) et \(0.32\) sont sensibles au bassin ou au
pas de temps et ne servent pas à revendiquer un intervalle continu
hyperchaotique. Huit exposants ne suffisent pas non plus à fermer la somme
partielle : la dimension de Kaplan–Yorke complète et l'exposant neutre du flot
restent non résolus.

### Données et figures

- Rapport de gate : [`N6_GATE_STATUS.md`](reports/N6_GATE_STATUS.md)
- Tableau complet : [`N6_multiseed_lyapunov.csv`](reports/N6_multiseed_lyapunov.csv)
- Résumé machine-readable :
  [`N6_multiseed_lyapunov_summary.json`](reports/N6_multiseed_lyapunov_summary.json)
- Figure dédiée :
  [`N6_multiseed_lyapunov.pdf`](figures/N6_multiseed_lyapunov.pdf)
- Figure article :
  [`Figure6a_multiseed_chaos.pdf`](../paper_v5/figures_ready/Figure6a_multiseed_chaos.pdf)
- Spectres bruts : [`runs/N6/`](runs/N6/)

## 9. N8 — récupération séquentielle à forte charge

### Calcul et audit réalisés

La grille contient quatre charges
\(\alpha=0.05,0.09,0.13,0.15\), cinq valeurs de
\(\lambda=0.35,0.40,0.50,0.70,0.90\) et cinq seeds, soit 100
trajectoires à \(N=2000\). Chaque cellule contient une histoire initiale de type
récupération par réalisation : les fractions entre seeds ne sont donc pas des
volumes de bassin.

Le champ de classification produit initialement ne pouvait pas être utilisé :
il exigeait deux tours complets, impossibles dans de nombreux enregistrements à
grande charge, et l'ordre de fusion des résumés pouvait écraser le label. Un
audit indépendant des trajectoires a donc défini avant inspection des résultats
un critère strict : au moins 50 changements de gagnant, fraction de pas vers
l'avant \(>0.99\), fraction de retours \(<0.01\), et résidu incompatible avec
un point fixe.

### Résultat de trajectoire

- **96/100 trajectoires** satisfont le critère séquentiel strict.
- Dans **19/20 cellules**, les cinq seeds sont strictement séquentielles.
- La seule cellule hétérogène est
  \((\alpha,\lambda)=(0.15,0.35)\) : seed 44 séquentielle, seed 42
  dégradée avec sauts, seeds 43 et 46 irrégulières, seed 45 fixe.
- Pour toutes les séquences strictes, le nombre d'événements reste stationnaire
  entre les quatre quarts de la fenêtre; le plus petit rapport de comptages vaut
  0.96.
- Conditionnellement au régime séquentiel, la vitesse locale dépend surtout de
  \(\lambda\) et peu de la charge. La qualité de récupération diminue avec
  \(\alpha\) et s'améliore avec \(\lambda\).

### Nature dynamique : conclusion provisoire

Les exposants de Lyapunov ne sont disponibles que pour 19 des 96 séquences
strictes, toutes pour la seed 46. Les estimations cumulées finales vont de
\(-0.00827\) à \(-0.00294\); les moyennes sur les 250 dernières unités vont de
\(-0.00134\) à \(+0.00120\), sans dépasser le seuil positif prédéfini
\(0.002\). Aucun chaos n'est donc résolu pour cette seed; les données sont
compatibles avec la direction neutre d'un cycle limite séquentiel stable.

Ce n'est pas encore une classification multi-seed. Il manque les exposants des
77 autres trajectoires mobiles et des trajectoires assez longues pour vérifier
la fermeture sur plusieurs tours complets. N8 ne doit donc pas être présenté
comme une carte de phase définitive ni comme une mesure de volumes de bassin.

### Données et figure

- Rapport détaillé :
  [`N8_PRELIMINARY_ANALYSIS.md`](reports/N8_PRELIMINARY_ANALYSIS.md)
- Audit trajectoire par trajectoire :
  [`N8_trajectory_audit.csv`](reports/N8_trajectory_audit.csv)
- Résumé par cellule :
  [`N8_cell_summary.csv`](reports/N8_cell_summary.csv)
- Résumé machine-readable :
  [`N8_preliminary_summary.json`](reports/N8_preliminary_summary.json)
- Figure d'audit :
  [`N8_preliminary_audit.pdf`](figures/N8_preliminary_audit.pdf)
- Données brutes : [`runs/N8/`](runs/N8/)

## 10. Incidents numériques détectés et corrigés

Ces audits constituent une partie importante du workplan, car ils empêchent de
transformer des artefacts de code ou de branche en conclusions physiques.

| Campagne | Problème détecté | Traitement |
|---|---|---|
| N1 | À \(10^{-4}\), l'écart dépend de la résolution du bracket, de l'histoire et parfois de la branche stationnaire atteinte. | Affirmation limitée à \(10^{-3}\); branches et contrôles d'histoire conservés. |
| N2 | Newton amorcé au pli pouvait sélectionner la selle instable. | Sorties invalides archivées; continuation corrigée depuis la branche mémoire stable. |
| N5B | Collision entre checkpoint de smoketest et fichier de production. | Espaces de noms `_smoke` et production séparés; fichiers antérieurs archivés. |
| N5B | Comparaison de deux membres opposés d'une paire complexe conjuguée créait de faux désaccords \(M=48/64\). | Appariement rendu invariant par conjugaison; racines brutes inchangées. |
| N8 | Le label historique exigeait deux tours et pouvait être écrasé lors de la fusion. | Classification refaite à partir des trajectoires avec un critère explicite indépendant. |

## 11. Ce qui peut être affirmé dans l'article

### Résultats prêts pour le corps principal

1. **Sélection extrême multi-seed (N1).** Le motif terminal et l'onset dynamique
   sont prédits par l'extrême statique, avec accord à \(10^{-3}\) sur toutes les
   réalisations admissibles.
2. **Hyperchaos multi-seed (N6).** À \(\lambda=0.31\), les cinq réalisations
   possèdent au moins trois exposants de Lyapunov positifs robustes au pas de
   temps.
3. **Incertitudes finies correctement structurées (N3).** Les erreurs et
   intervalles doivent être calculés sur les seeds, et non en traitant les
   motifs corrélés comme des échantillons indépendants.

### Résultats utilisables avec qualification

1. **N2 :** ralentissement dynamique compatible avec une loi en racine carrée
   pour quatre seeds correctement centrées. L'exposant moyen vaut \(0.477\)
   avec intervalle bootstrap \([0.424,0.529]\), et \(0.470\)
   \([0.435,0.504]\) sur la fenêtre proche. Le logarithme est rejeté, mais le
   gate strict d'un exposant universel mesuré exactement à \(1/2\) échoue à
   cause des dépendances à la seed et à la fenêtre. Seed 51 reste archivée et
   visible, mais son bracket dynamique ne contient pas le pli statique exact.
2. **N5A :** crossover de cohérence robuste autour de
   \(\tau\simeq1.23\!-\!1.34\), sans frontière cycle–chaos multi-seed établie.
3. **N5B :** exploration archivée et retirée du périmètre du manuscrit par
   décision explicite le 30 juillet 2026; aucune conclusion sur la marge avec
   Hopf n'est retenue dans l'article.
4. **N8 :** récupération séquentielle robuste dans 19/20 cellules pour
   l'initialisation prescrite; nature périodique seulement compatible avec les
   données disponibles, pas encore certifiée sur plusieurs seeds.

## 12. Travail restant dans le périmètre retenu

1. Le rafraîchissement statique N2, les ajustements de fenêtres et le gate
   statique–dynamique sont terminés. Les sorties sont référencées dans
   `reports/N2_GATE_STATUS.md` et `reports/N2_summary.json`.
2. Les calculs de Lyapunov et multi-tours N8 ont été retirés du périmètre par
   décision explicite le 30 juillet 2026. Les trajectoires et l'audit
   préliminaire existants sont conservés, mais aucune nouvelle simulation N8 ne
   doit être lancée.
3. La figure N2 finale a été intégrée avec la qualification issue du gate dans
   `paper_v5/figures_ready/Figure4b_N2_critical_laws.pdf`.

Les calculs et audits N5B déjà terminés sont conservés sous `runs/N5B/` pour la
traçabilité, mais la piste a été arrêtée par l'utilisateur le 30 juillet 2026.
Elle ne doit plus déclencher de simulation ni de modification du manuscrit.

Les tâches N7, N9 et N10 ont été retirées du périmètre par décision explicite et
ne font pas partie des résultats présentés ici. N4 a été une tâche d'assemblage
et de traçabilité des figures, sans nouveau résultat physique; sa provenance est
documentée dans
[`N4_figure_provenance.json`](manifests/N4_figure_provenance.json).
