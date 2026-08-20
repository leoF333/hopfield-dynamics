# Conclusion scientifique de N1

Date : **2026-07-29**.

## Portée exacte

Les résultats N1 de base sont complets pour 20 réalisations indépendantes
(\(N=2000,P=100\), graines 42--61) : seuils statiques, seuils dynamiques,
contrôles d’histoire et identité des motifs. L’audit exact des branches
stationnaires est également complet pour les 20 réalisations. Le contrôle
dynamique fin à \(dt=0.005\), largeur de bracket au plus \(5\times10^{-5}\)
et deux histoires initiales a été effectué sur quatre réalisations
diagnostiques (42, 44, 48 et 61), et non sur les 20.

N1 est donc terminé pour tester les mécanismes possibles et corriger la
conclusion scientifique. Une campagne fine sur les 20 graines ne serait
nécessaire que pour estimer la fréquence statistique de la bistabilité ou de
l’hystérésis opérationnelle.

## Résultats établis sur 20 réalisations

1. Le motif du pli statique extrême, la liaison dynamique la plus lente et le
   motif d’arrêt coïncident pour **20/20** réalisations. L’intervalle de Wilson
   à 95% est \([0.839,1]\). Ce résultat est indépendant du contrôle statique :
   aucune information statique n’a servi à sélectionner le motif dynamique.
2. Le gate préenregistré d’égalité quantitative des seuils échoue :
   seulement **11/20** brackets statiques originaux et dynamiques se
   recouvrent, contre 90% requis.
3. Après résolution augmentée exacte des plis, le pli de la branche mémoire
   appartient au bracket dynamique original pour **10/20** réalisations.
4. La branche atteinte après arrêt est numériquement la même que la branche
   mémoire pour **13/20** réalisations. Une branche stationnaire distincte et
   convergée est trouvée pour les graines 44, 45, 46, 48, 50 et 56. Le solveur
   augmenté de la branche d’arrêt de la graine 43 ne converge pas et cette
   réalisation reste non interprétée sur ce point.
5. Parmi les 19 branches d’arrêt convergées, leur pli appartient au bracket
   dynamique original pour **14/19** réalisations. La distance absolue moyenne
   entre milieu dynamique et pli passe de \(3.92\times10^{-4}\) pour la
   branche mémoire à \(1.22\times10^{-4}\) pour la branche d’arrêt; la médiane
   passe de \(1.42\times10^{-4}\) à \(8.67\times10^{-5}\).

Le changement de branche explique donc une fraction substantielle des écarts,
mais pas tous les cas.

## Audit dynamique fin

| Graine | Bracket depuis le cycle | Bracket depuis l’arrêt | Pli mémoire | Pli d’arrêt | Verdict |
|---|---:|---:|---:|---:|---|
| 42 | [0.3276250, 0.3276602] | [0.3276250, 0.3276602] | 0.3276299 | 0.3276299 | Convergence numérique vers un pli unique; aucune dépendance à l’histoire résolue. |
| 44 | [0.3392946, 0.3393393] | [0.3392946, 0.3393393] | 0.3386429 | 0.3393189 | La transition suit le pli d’une branche stationnaire distincte; aucune dépendance à l’histoire résolue. |
| 48 | [0.3322925, 0.3323372] | [0.3325610, 0.3326058] | 0.3288609 | 0.3325906 | Branche distincte et fenêtre de coexistence dépendant de l’histoire d’au moins \(2.24\times10^{-4}\). |
| 61 | [0.3245625, 0.3246018] | [0.3249953, 0.3250347] | 0.3250185 | 0.3250185 | Même branche stationnaire mais fenêtre de coexistence dépendant de l’histoire d’au moins \(3.94\times10^{-4}\). |

Le déplacement des milieux descendants entre le calcul original
\(dt=0.01\) et l’audit \(dt=0.005\) est inférieur à
\(6.1\times10^{-5}\) pour ces quatre graines. Les écarts de l’ordre de
\(10^{-4}\) à \(10^{-3}\) observés pour 44, 48 et 61 ne peuvent donc pas être
attribués uniquement au pas de temps ou à la largeur du bracket.

Les deux histoires initiales établissent une coexistence opérationnelle de
résultats dans les fenêtres non recouvrantes de 48 et 61. Comme le protocole
réinitialise chaque valeur de \(\lambda\) depuis une histoire cyclique ou
arrêtée de référence, il ne constitue pas encore une boucle quasistatique
continue. Le terme rigoureux est donc « dépendance à l’histoire / bistabilité
opérationnelle »; « hystérésis quasistatique » demanderait un balayage continu
montant et descendant.

## Conclusion pour l’article

N1 réfute la formulation forte selon laquelle la frontière dynamique est
toujours égale au plus grand pli de la branche mémoire-connectée. Elle établit
en revanche un résultat plus structurel :

> Dans chaque réalisation testée, l’extrême de l’ensemble des bifurcations
> statiques sélectionne correctement le motif qui devient le goulot
> d’étranglement dynamique et sur lequel la séquence s’arrête. La valeur du
> seuil dynamique opérationnel peut toutefois être déplacée par l’accès à une
> branche stationnaire distincte et par la coexistence de bassins dépendant de
> l’histoire.

La statistique des bifurcations organise donc la transition en sélectionnant le
défaut critique. Elle ne suffit pas, à elle seule, à prédire une frontière
dynamique unique lorsque la géométrie globale des branches et des bassins est
multistable.

## Claims autorisés

- Accord 20/20 entre motif statique extrême, liaison dynamique lente et motif
  d’arrêt.
- Existence démontrée de branches stationnaires alternatives dans plusieurs
  réalisations.
- Existence démontrée de dépendance à l’histoire/bistabilité opérationnelle
  pour les cas diagnostiques 48 et 61.
- Compatibilité exacte seuil--pli dans le cas simple 42 et avec le pli de la
  branche alternative dans le cas 44.

## Claims non autorisés

- Égalité universelle entre seuil dynamique et plus grand pli mémoire.
- Fréquence populationnelle de la bistabilité à partir des quatre audits fins.
- Hystérésis quasistatique sans balayages continus.
- Interprétation de la branche d’arrêt de la graine 43, dont le solveur
  augmenté n’a pas convergé.

