# E37 — Ce que la composante Hopfield moderne apporte sur une séquence corrélée

> Convention : prose FR, chiffres et labels EN.
> Code : `src/e37_modern_k_markov.py`, `src/e37_e27_reconciliation.py`,
> `src/e37_arrest_in_time.py`, `src/e37_figure_doseresponse.py`,
> `src/e37_figure_traces.py`.
> Données : `results/6_motifs_correles/data/E37_*.json`, `E37_*.npz`.
> Figures : `figures/figE37_modernK_doseresponse.png`,
> `figures/figE37_traces_c0.60.png`, `figures/figE37_traces_c0.90.png`.
> Date : 2026-07-31.

---

## 1. Verdict

**Moderniser K fait passer la plage de fonctionnement en corrélation de `c ≤ 0.2`
à `c ≤ 0.8` ; moderniser J en plus la porte à `c ≥ 0.9`.** L'effet est un
interrupteur, pas une amélioration graduelle : aux corrélations où le système
usuel échoue, les bras à K moderne donnent `coverage = 1.000` et
`forward_fraction = 1.000`, sans régime intermédiaire.

Vérifié sur **3 réalisations de motifs** (seeds 42, 43, 44) aux quatre
corrélations décisives, sans une seule exception.

Ce que le résultat **n'établit pas** : que le rappel redevienne net. À forte
corrélation le bras moderne lit un paquet de plusieurs trames ; ce paquet avance
simplement dans le bon ordre et à période constante. La composante moderne
restaure **l'ordre et le mouvement**, pas la résolution individuelle des trames.
C'est la même séparation qu'E35-V1 avait isolée (mécanisme de cycle parfait,
qualité de reconstruction en échec) et ce résultat ne la contredit pas.

---

## 2. La question, et ce qui était déjà acquis

Deux résultats du dépôt encadraient la question sans la chiffrer.

**E27 (2026-07-06)** — vidéo de Markov `ξ^{μ+1}_i = ξ^μ_i·s_i`, flips iid
`P(s=−1) = (1−c)/2`, donc corrélation `c` entre trames consécutives. Le système
usuel perd le rappel séquentiel quand `c` monte, et à `c ≥ 0.4` les mémoires
individuelles fondent en paquets délocalisés. Corrélation réglable, mais une
seule architecture.

**E35-V1 (2026-07-27)** — factoriel J×K complet sur UNE vidéo (texture advectée,
`cond(G) ~ 8e7`) : `JH_KH` 0.137 de couverture, `JP_KP` 1.000. Message net
(« K = PINV est la condition du rappel ordonné ») mais point unique : pas de
courbe dose-réponse, donc **pas de quantification de l'apport**.

E37 croise les deux : le factoriel complet contre la corrélation réglable d'E27,
aux paramètres exacts d'E27, de sorte que le bras `JH_KH` **est** la référence
d'E27.

---

## 3. Protocole

Générateur d'E27 reproduit verbatim (`make_markov`), `N = 2000`, `P = 100`,
`β = 20`, `τ = 10`, `t0 = 1`, `λ = 0.9`, `dt = 0.01`, échantillonnage 0.1.
`c ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 0.9}` ; corrélation de bulk mesurée conforme à la
consigne à ±0.002. Condition initiale `a₀ = e₀` (l'état est exactement sur la
trame 0), historique constant. `t_total = 8000`, `transient = 0.4` — soit 4800
u.t. d'analyse, ≈ 4.4 tours à la période mesurée.

Quatre bras : `{J ∈ hebb, pinv} × {K ∈ hebb, pinv}` via
`mhn_reduced.factorial_architectures`, intégrés par `MhnReducedDDE`
(sous-classe de l'intégrateur RK4/Hermite partagé, champs remplacés, aucun
couplage `N×N` formé).

**Point de méthode.** La lecture est **la même pour les quatre bras** :
l'overlap physique `m = ξ·tanh(βu)/N`. Décoder par `G†` seulement pour les bras
modernes reviendrait à inscrire la réponse dans l'instrument. Le meneur décodé
est enregistré comme diagnostic secondaire, pour les quatre bras également.

Observables. Les deux premières viennent du critère de cycle gelé en E35
(`classify_cycle`) ; les deux dernières sont propres à E37 et servent à séparer
« l'anneau tourne » de « l'anneau tourne en résolvant une trame à la fois » —
le mode de défaillance qu'E27 avait effectivement observé.

| observable | définition |
|---|---|
| `coverage_fraction` | fraction des 100 trames qui prennent la tête au moins une fois |
| `forward_fraction` | fraction des changements de meneur qui vont vers la trame suivante |
| `selectivity` | `m_meneur − max_{ν≠} m_ν`, au pic propre de chaque trame |
| `packet_width` | nombre de trames à plus de la moitié du pic |

---

## 4. Résultat principal : la courbe dose-réponse

Seed 42, λ = 0.9. `couverture / fraction avant` ; **gras** = cycle valide au
critère E35 (couverture 1.0, avant ≥ 0.95, ≥ 3 tours, `period_cv ≤ 0.05`).

| c | JH_KH *(usuel)* | JP_KH | JH_KP | JP_KP |
|---|---|---|---|---|
| 0.0 | **1.00 / 1.00** | **1.00 / 1.00** | **1.00 / 1.00** | **1.00 / 1.00** |
| 0.2 | **1.00 / 1.00** | **1.00 / 1.00** | **1.00 / 1.00** | **1.00 / 1.00** |
| 0.4 | 0.23 / 0.19 | 0.35 / 0.28 | **1.00 / 1.00** | **1.00 / 1.00** |
| 0.6 | 0.01 / 0.00 | 0.01 / 0.00 | **1.00 / 1.00** | **1.00 / 1.00** |
| 0.8 | 0.01 / 0.00 | 0.01 / 0.00 | **1.00 / 1.00** | **1.00 / 1.00** |
| 0.9 | 0.01 / 0.00 | 0.01 / 0.00 | 0.05 / 0.50 | **1.00 / 1.00** |

Trois lectures.

1. **K moderne est un interrupteur.** À `c = 0.4, 0.6, 0.8` les bras à K hebbien
   sont morts et les bras à K moderne sont parfaits. Aucune zone intermédiaire.
2. **La mort du système usuel est un gel, pas une divagation.** À `c ≥ 0.6` le
   diagnostic est `no_lead_change`, couverture 0.01 : l'état s'immobilise sur une
   trame. Voir §6.
3. **J moderne n'agit qu'à l'extrême.** `JH_KP` et `JP_KP` coïncident au 4ᵉ
   chiffre jusqu'à `c = 0.8`. À `c = 0.9`, `JH_KP` s'effondre et seul `JP_KP`
   tient. Cohérent avec E35-V1 (`JP_KP` 1.000 vs `JH_KP` 0.418), et le classe
   comme effet **second mais réel**.

La netteté du rappel, elle, se dégrade continûment — c'est ce qui rend le
résultat crédible plutôt que magique :

| c | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| marge au pic, `JP_KP` | 0.948 | 0.785 | 0.592 | 0.392 | 0.191 | 0.094 |
| largeur de paquet, `JP_KP` | 1 | 1 | 1 | 3 | 7 | 13 |
| marge au pic, `JH_KH` | 0.948 | 0.781 | 0.003 | 0.012 | 0.013 | 0.001 |
| largeur de paquet, `JH_KH` | 1 | 1 | 19 | 16 | 19 | 51 |

À `c = 0.9` le bras moderne tient un paquet de 13 trames — mais ce paquet avance
dans le bon ordre à période constante. Le système usuel tient un paquet de 51
trames, immobile.

**Figure : `figE37_modernK_doseresponse.png`.** La couleur porte K (bleu = Hebb,
orange = moderne), le style de trait porte J ; marqueur creux = cycle non valide.

---

## 5. Contrôle multi-graines

Seeds 42, 43, 44 aux quatre corrélations décisives. Comptage de cycles valides :

| c | K hebbien (2 bras × 3 graines) | JH_KP | JP_KP |
|---|---|---|---|
| 0.2 | **6/6** | 3/3 | 3/3 |
| 0.4 | 0/6 | **3/3** | **3/3** |
| 0.6 | 0/6 | **3/3** | **3/3** |
| 0.9 | 0/6 | **0/3** | **3/3** |

Aucune ambiguïté sur aucune graine. La variabilité de réalisation touche
uniquement la *manière* d'échouer des bras hebbiens (à `c = 0.6`, seed 43 donne
couverture 0.42 / avant 0.45 au lieu du gel complet) — jamais le verdict.

---

## 6. Réconciliation avec E27 : une mesure prise trop tôt

E37 donne `forward = 0.19` à `c = 0.4` là où E27 publiait `0.65`. Deux
expériences sur le même générateur, la même architecture, le même λ. Il fallait
trancher, puisque ce bras est la ligne de référence.

**Première hypothèse, réfutée.** J'avais supposé que l'écart venait de
l'observable — E27 lit le meneur sur les coefficients `a`, E35 impose l'overlap
physique `m` pour des motifs corrélés. Le protocole E27 rejoué et la **même**
trajectoire relue avec les deux lectures (`e37_e27_reconciliation.py`) :

| c | meneur lu sur `a` | meneur lu sur `m` | E27 publié |
|---|---|---|---|
| 0.0 | 1.00 | 1.00 | 1.00 |
| 0.2 | 1.00 | 1.00 | 1.00 |
| 0.4 | **0.65** | 0.67 | **0.65** |
| 0.6 | **0.50** | 0.52 | **0.50** |
| 0.8 | 0.69 | 0.59 | 0.60 |

L'observable n'explique rien. En revanche le protocole reproduit E27 au chiffre
près — donc la divergence vient bien du protocole, pas d'un désaccord numérique.

**Cause réelle : la fenêtre.** E27 analyse 400 u.t. (0.37 période) ouvertes après
200 u.t. seulement. `e37_arrest_in_time.py` suit le nombre de changements de
meneur par fenêtre de 400 u.t. sur 8000 u.t., depuis la CI amorcée d'E27 :

| | fenêtres successives de 400 u.t. |
|---|---|
| `c=0.6`, JH_KH | 24, 16, 1, **0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0** |
| `c=0.6`, JP_KP | 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37 |
| `c=0.4`, JH_KH | 30, 11, 16, 10, 6, 15, 12, 10, 16, 9, 10, 16, 11, 9, 15, 13, 9 |
| `c=0.4`, JP_KP | 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37 |

À `c = 0.6` le système usuel **s'arrête à t ≈ 1200** et ne bouge plus. E27
mesurait entre t=200 et t=600, entièrement dans le transitoire. Sa « fraction
avant 0.50 » décrit une bouffée de mouvement qui précède le gel, pas un régime.
À `c = 0.4` il ne gèle pas mais s'agite sans ordre.

**Conséquence.** Les conclusions qualitatives d'E27 restent valides (« cycle
dégradé/détruit à corrélation forte »). Mais les valeurs `0.65 / 0.50 / 0.60` ne
doivent pas être citées comme des fractions avant stationnaires : ce sont des
mesures de transitoire. À corriger si elles apparaissent dans un texte destiné à
publication.

---

## 7. Un fait non anticipé : la période devient insensible aux motifs

`JP_KP` donne **`T = 1080.7`, `period_cv = 4.6e−05`** à *toutes* les corrélations
et sur *toutes* les graines testées — `c ∈ {0, 0.2, 0.4, 0.6, 0.8, 0.9}`,
seeds 42/43/44 — identiques aux quatre chiffres significatifs. Pendant que
`JH_KP` dérive : 1078.2 → 1078.8 → 1088.1 → 1164.1.

Explication mécanique proposée : K moderne applique `G†` **avant** le décalage
cyclique, donc décorrèle les coordonnées avant de propager. Dans ces coordonnées
blanchies la dynamique de séquence redevient celle du cas iid, et la période
n'est plus fixée par la statistique des motifs mais seulement par `(λ, τ, β, P)`.
C'est le sens précis de « moderniser K » : **décoder avant de décaler**.

Statut : **hypothèse cohérente avec les données, non testée directement.** Le
test décisif serait de comparer `T` à celui du cas iid à même `(λ, τ, β, P)` et
de vérifier l'invariance sur d'autres générateurs corrélés (GP-cercle d'E28).
Non fait.

---

## 8. Balayage en λ — INCOMPLET, aucune conclusion

Objectif : dire si l'apport déplace le seuil de dépiégeage ou change la nature du
régime. Lancé à `c ∈ {0.6, 0.9}` et `λ ∈ {0.35, 0.4, 0.5, 0.7}`, `t_total = 10000`.
**Interrompu à la demande de l'utilisateur après les deux premiers bras de chaque
cellule.** Les 8 cellules `JH_KP` et `JP_KP` n'ont pas été mesurées.

Ce qui est acquis (bras à K hebbien, 16 cellules) :

| c | λ = 0.7 | 0.5 | 0.4 | 0.35 |
|---|---|---|---|---|
| 0.6, JH_KH | 0.01/0.00 | 0.01/0.00 | 0.01/0.00 | 0.01/0.00 |
| 0.6, JP_KH | 0.01/0.00 | 0.01/0.00 | 0.01/0.00 | 0.06/0.10 |
| 0.9, JH_KH | 0.01/0.00 | 0.01/0.00 | 0.01/0.00 | 0.01/0.00 |
| 0.9, JP_KH | 0.01/0.00 | 0.01/0.00 | 0.01/0.00 | 0.01/0.00 |

Lecture autorisée : **baisser λ ne sauve pas le système usuel** — il échoue aux
quatre valeurs de λ testées, aux deux corrélations. Lecture NON autorisée :
quoi que ce soit sur la plage en λ des bras modernes, qui n'a pas été mesurée.

---

## 9. Ce que le run établit, et ce qu'il n'établit pas

**Établi.**
- La plage de fonctionnement en corrélation passe de `c ≤ 0.2` à `c ≥ 0.9`,
  sur 3 réalisations, avec le critère de cycle gelé en E35.
- K est le facteur dominant ; J n'agit qu'à `c = 0.9`.
- La défaillance du système usuel à `c ≥ 0.6` est un **arrêt**, daté (t ≈ 1200).
- Le désaccord E27/E37 est un artefact de fenêtre d'analyse, pas un désaccord.

**Non établi.**
- La plage en λ des bras modernes (§8).
- L'invariance de période comme mécanisme (§7) — hypothèse, pas mesure.
- Toute affirmation de qualité de reconstruction d'image : E37 ne mesure aucune
  métrique d'image. Le NO-GO d'E35-V1 sur ce point reste entier.
- Le comportement à `N` plus grand. E27 avait montré que la frontière de
  destruction se déplace avec `N` (`c ∈ (0.2,0.4)` à `N=2000` →
  `c ∈ (0.4,0.6)` à `N=10⁴`) ; E37 n'a tourné qu'à `N = 2000`. Le contraste
  entre bras étant de l'ordre de « tout ou rien », un effet de taille est peu
  susceptible de l'inverser, mais ce n'est pas vérifié.
- La généralisation à d'autres corrélations. E28 avait montré que la corrélation
  lisse (GP-cercle) se comporte à l'opposé de la corrélation locale (Markov) pour
  le système usuel. E37 n'a testé que Markov.

---

## 10. Incidents

**Noms de fichiers écrasés (perte de 6 runs).** `Path.with_suffix` sur un nom
contenant `c0.40_seed42_…` traite `.40_seed42_…` comme l'extension : les six runs
du premier factoriel ont tous écrit dans `E37_modernK_c0.json`. Les métriques
étaient sauvées dans les logs, mais les artefacts structurés ont été perdus.
Corrigé (concaténation explicite de la chaîne) et le factoriel relancé —
**reproductibilité exacte constatée sur les 24 cellules**, chiffre pour chiffre,
ce qui a converti l'incident en contrôle de reproductibilité.

**Hypothèse fausse publiée en cours de route.** L'écart avec E27 a d'abord été
attribué à l'observable ; mesure faite, l'observable n'y était pour rien (§6).
Corrigé dans ce document.

## 11. Coût

| phase | temps mur |
|---|---|
| smoke + benchmark | 30 s |
| factoriel principal (6 c × 4 bras), ×2 pour cause d'incident | 2 × ~28 min |
| traces pour figures (3 c × 2 bras) | ~13 min |
| réconciliation E27 | ~3 min |
| arrêt-dans-le-temps (4 runs × 8000 u.t.) | ~13 min |
| multi-graines (2 seeds × 4 c × 4 bras) | ~35 min |
| balayage λ (interrompu) | ~26 min |

Débit mesuré : ~4650 pas/s à `N=2000, P=100`, mono-thread. Jamais plus de 10
processus simultanés sur 10 cœurs ; BLAS épinglé à 1 thread par processus
(`OMP/OPENBLAS/VECLIB/MKL/NUMEXPR/ACCELERATE`).
