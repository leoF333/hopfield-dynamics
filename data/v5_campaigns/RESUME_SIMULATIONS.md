# Reprise sûre des simulations numériques

## Campagne 24 h lancée le 29 juillet

Le point de reprise prioritaire est maintenant :

`/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics/runs/NEXT24H/state.json`

Les journaux individuels sont dans :

`/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics/logs/NEXT24H/`

Le plan, les benchmarks et les raisons scientifiques sont dans
`NEXT_24H_PLAN.md`. L’ordre est audit N1, N2, N6, N8, N5A, N5B, puis
agrégation/gates. L’orchestrateur `src/run_next24h.py` ne lance jamais plus de
quatre processus monothreads et chaque campagne est restart-safe à la graine ou
au point.

Après un arrêt de l’ordinateur, relire d’abord `runs/NEXT24H/state.json`, vérifier
qu’aucun ancien processus n’est actif, puis reprendre avec :

```bash
cd "/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics"
/opt/homebrew/bin/conda run --no-capture-output -n mcmc_env \
  python src/run_next24h.py --start-phase PHASE
```

Remplacer `PHASE` par la première phase non terminée indiquée dans l’état :
`N1_AUDIT`, `N2`, `N6`, `N8`, `N5A`, `N5B_MEMORY`, `N5B_PINNED` ou
`FINALIZE`. Les sorties existantes sont relues; ne pas définir `V5_FORCE=1`.

Dernière mise à jour : **2026-07-29 03:30 CDT**.

## Mise à jour prioritaire du 29 juillet

La campagne N1 est **terminée** : les 20 graines 42--61 possèdent chacune un
résultat statique et un résultat dynamique finaux, intègres et accompagnés de
leur provenance. Il n'y a aucun calcul actif et il ne faut pas relancer la
commande N1.

Le gate scientifique préenregistré a échoué. Les 20/20 graines choisissent
indépendamment le même motif pour le maximum statique, le lien dynamique le plus
lent et l'arrêt, mais seulement 11/20 brackets statiques et dynamiques se
recouvrent, contre 90% requis. Le résultat de sélection d'identité est donc
établi; l'égalité quantitative multi-graines des seuils ne l'est pas avec les
définitions et résolutions actuelles. Aucun résultat n'a été forcé.

Résultats canoniques à lire avant toute décision :

- `runs/N1/comparison.json`
- `reports/N1_static_dynamic.csv`
- `figures/N1_static_dynamic.pdf`
- `RUN_STATUS.md`

Archive complète finale :

`runs/N1_completion_backup_2026-07-29T0330CDT.tar.gz`

SHA-256 :

`11e0f7b9dfd73558c33b69f7a11e5972d66761b3a7214cd5f23da8e54d6923b8`

N4 n'a pas été reconstruit et N2 n'a pas été lancé, conformément au gate.

Un défaut de comptage a été découvert lors des reprises longues : au-delà de
200 000 états enregistrés, l'ancien code recomptait les tours seulement sur la
fenêtre d'états conservée et oubliait les événements antérieurs. Ce défaut
n'affecte aucun des huit résultats finaux (ils ont tous atteint cinq tours avant
la troncature), mais il invalide comme verdict physique les anciennes timeouts
longues. Le code corrigé conserve désormais un flux complet des événements de
relais indépendamment de la fenêtre d'états.

- SHA-256 corrigé de `src/dynamics.py` :
  `4a63e6681a0862c4314e6debbe522c09f9fe8d5cbdd87964b64ad2838f25d128`.
- SHA-256 antérieur explicitement admis pour les huit résultats déjà complets :
  `bfe196db07865c23a0870341833d6681dd38a8f7cdef0d328adf22bba7df6b2d`.
- Les six tests du noyau, dont un cycle lent franchissant la limite de
  troncature, et le smoke N1 passent.
- Le pilote haute-\(N\) corrigé, graine **43**, s'est terminé à 00:59 CDT en
  environ 17 minutes : bracket dynamique \([0.31825,0.31850]\), motif lent et
  motif d'arrêt 67, contrôle d'histoire réussi et recouvrement avec le bracket
  statique indépendant.
- Les onze autres reprises se sont ensuite terminées sans erreur. Les 20
  résultats dynamiques sont complets.

La commande générale est conservée plus bas uniquement comme procédure de
récupération historique. Ne pas l'exécuter dans l'état actuel : aucun résultat
N1 ne manque.

Ce document est le point d'entrée à lire après le redémarrage de l'ordinateur.
Il décrit comment reprendre N1 sans recalculer les résultats déjà validés.

## État de l'arrêt du 28 juillet (archive historique)

Cette section conserve l'état exact du premier arrêt et est remplacée, pour toute
reprise actuelle, par la mise à jour prioritaire ci-dessus. L'orchestrateur et
les quatre processus actifs avaient été interrompus manuellement à
09:11 PDT. Les journaux et checkpoints sont restés stables après l'arrêt. Le
heartbeat Codex `surveillance-horaire-simulations-v5` a également été mis en
pause afin qu'aucun processus ne soit relancé avant le redémarrage volontaire.

- Résultats N1 finaux, avec schéma et provenance complets :
  graines **42, 44, 48, 49, 51 et 52**.
- Graines limitées par le temps lors du passage primaire et conservées pour une
  reprise avec \(T_{\max}=40000\) :
  **43, 45, 46, 47, 50, 53, 54, 55, 56 et 57**.
- Graines interrompues pendant le dernier lot primaire :
  **58, 59, 60 et 61**.
- Pour chacune de ces quatre graines, le checkpoint contient 21 points complets :
  dernier point \(\lambda=0.360\), prochain point \(\lambda=0.358\).
- Les **20 checkpoints N1** ont été rouverts après l'arrêt. Aucun fichier corrompu
  n'a été détecté.

## Emplacements sur l'ordinateur

Racine de tout le travail numérique produit :

`/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics`

Résultats et checkpoints N1 :

`/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics/runs/N1`

Les fichiers finaux validés suivent les formes :

`dynamic_N2000_P100_seed<SEED>.npz`

`dynamic_N2000_P100_seed<SEED>.json`

Les reprises suivent la forme :

`dynamic_N2000_P100_seed<SEED>.checkpoint.npz`

Journaux par graine :

`/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics/logs/N1_production_seed<SEED>.log`

État détaillé et historique des contrôles :

`/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics/RUN_STATUS.md`

État machine lisible automatiquement :

`/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics/runs/N1/overnight_status.json`

Snapshot d'audit :

`/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics/runs/N1/shutdown_snapshot_2026-07-28T0859PDT.json`

Archive locale créée après l'arrêt propre :

`/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics/runs/N1/N1_shutdown_backup_2026-07-28T0912PDT.tar.gz`

SHA-256 de cette archive :

`cce14d62e3e6e38a38fd2e80d9adbc382ace2740700cb9471aa414da3546e5a1`

Plan scientifique source :

`/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5/NUMERICAL_WORKPLAN.md`

Le dossier historique suivant reste strictement en lecture seule :

`/Users/leoflack/Desktop/Recherche/Chicago/numerics copie théorie claude + implémentation GPT`

## Commande de reprise N1 — archive, ne pas exécuter maintenant

Cette commande ne doit être utilisée que si une vérification future démontre la
perte ou la corruption d'un résultat N1 :

```bash
cd "/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5_numerics"

/opt/homebrew/bin/conda run --no-capture-output -n mcmc_env \
  python src/n1_overnight.py \
  --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
  --wait-active \
  --max-parallel 4 \
  --poll-seconds 60 \
  --stale-minutes 20 \
  --retry-seeds 50 53 54 55 \
  --retry-max-time 40000
```

Après le lancement, réactiver le heartbeat Codex
`surveillance-horaire-simulations-v5` pour reprendre les contrôles horaires.

Cette commande est idempotente et restart-safe :

1. elle refuse de recalculer les vingt sorties finales dont le schéma, le contrôle
   d'histoire et le hash de `dynamics.py` sont valides ;
2. elle détecte les graines terminées par `timeout_other` et les reprend ensuite
   avec \(T_{\max}=40000\), sans réduire le critère des cinq tours ;
3. elle route aussi les reprises interrompues 50, 53, 54 et 55 directement vers
   le passage \(T_{\max}=40000\) ;
4. une timeout de descente reprend au point non résolu ; une timeout de bisection
   reconstruit le coarse bracket sauvegardé ;
5. après les reprises, elle exécute `n1_compare.py`, applique le gate scientifique
   d'au moins 15 graines admissibles, puis reconstruit N4 seulement si le gate
   passe.

## Précautions

- Ne pas supprimer ou renommer `runs/N1`, `logs`, les fichiers `.checkpoint.npz`
  ou les vingt couples finaux `.npz`/`.json`.
- Ne pas définir `V5_FORCE=1`.
- Ne pas modifier `src/dynamics.py` avant d'avoir terminé N1. Son SHA-256 corrigé
  attendu est
  `4a63e6681a0862c4314e6debbe522c09f9fe8d5cbdd87964b64ad2838f25d128`.
  Le SHA antérieur `bfe196...` n'est accepté que pour les huit sorties déjà
  terminées avant que la troncature puisse intervenir.
- Ne pas lancer une seconde campagne lourde en parallèle.
- Si la commande est de nouveau interrompue, la relancer à l'identique : elle
  reprendra les derniers checkpoints complets.
- Internet n'est pas nécessaire pour les simulations. MLX/Metal n'étant pas
  accessible dans le sandbox Codex, ces runs utilisent le CPU float64.

## Après N1

Lire `RUN_STATUS.md` et `runs/N1/comparison.json`. La phase numérique N1 est
terminée, mais `completion_test_passed` vaut `false`. Ne pas démarrer N2 ou N4
automatiquement. La prochaine étape doit être un audit scientifique ciblé des
neuf écarts de seuil, en particulier l'outlier seed 48, ou une révision explicite
de la narration de l'article fondée sur le résultat robuste 20/20 de sélection
d'identité.
