# Numerical work plan — execution and restart status

## Campagnes demandées sur 24 h — 2026-07-29 11:00 CDT

- Audit de branches N1 terminé pour 20/20 réalisations. La résolution augmentée
  exacte révèle la même branche mémoire/arrêt pour 13/20 graines et une branche
  d’arrêt distincte convergée pour 44, 45, 46, 48, 50 et 56. Le solveur de la
  branche d’arrêt de la graine 43 n’a pas convergé et reste non interprété.
- En particulier, les écarts des graines 44 et 48 s’alignent sur le pli d’une
  branche d’arrêt distincte : ils ne peuvent pas être attribués uniquement à la
  largeur de bracket ou au pas de temps.
- Le contrôle dynamique de convergence/hystérésis est préparé pour les graines
  42, 44, 48 et 61 à `dt=0.005`, cinq tours et tolérance de bracket `5e-5`.
- N2 a été corrigé pour utiliser le pli raffiné par le système augmenté et pour
  checkpoint­er séparément chaque point statique et dynamique. Les graines
  retenues sont 42, 60, 47, 52 et 51, toutes sans changement de branche et
  couvrant la distribution des écarts extrêmes.
- Les tests centraux (7/7), les smoketests N2 et N5B, ainsi que les smoketests
  antérieurs N5A/N6/N8 passent. Un point N2 haute résolution a pris 215 s.
- Benchmarks : N6 haute-N `1.84 s` pour `T=20+20`, N8 haute charge `1.37 s`,
  N5A cellule réelle `17.9 s`, N5B racines `28.0 s` (M48) et `39.3 s` (M64).
- MLX/Metal échoue dans l’environnement Codex; toutes les productions utilisent
  le CPU float64. L’orchestrateur limite explicitement l’ensemble à quatre
  processus monothreads.
- Plan détaillé : `NEXT_24H_PLAN.md`. État machine restart-safe :
  `runs/NEXT24H/state.json`. Journaux : `logs/NEXT24H/`.
- Orchestrateur lancé à **11:02 CDT**. Le premier lot contient uniquement les
  quatre audits N1 (graines 42, 44, 48, 61). État initial : charge
  2.58/3.09/2.84, mémoire libre 69%, aucune page throttled, disque libre
  614 GiB. Les quatre tâches ont chacune un PID et un journal distincts.
- Surveillance horaire active :
  `surveillance-horaire-campagnes-num-riques-24h`.
- Contrôle horaire à **12:03 CDT** : les quatre raffinements descendants N1
  sont terminés et intègres à `dt=0.005`, cinq tours et largeur
  `<=5e-5`. Brackets : graine 42 `[0.3276250,0.3276602]`, 44
  `[0.3392946,0.3393393]`, 48 `[0.3322925,0.3323372]`, 61
  `[0.3245625,0.3246018]`. Pour 42, le milieu diffère du pli mémoire de
  `1.27e-5`, compatible avec la résolution. Pour 44, il diffère du pli de la
  branche d’arrêt de seulement `-1.90e-6`, tandis que l’écart au pli mémoire
  reste `6.74e-4` : le changement de branche est confirmé et n’est pas un
  artefact de pas. Pour 48, la branche distincte explique l’essentiel de
  l’écart, mais un résidu `-2.76e-4` subsiste et doit attendre le contrôle
  ascendant. Pour 61, un écart `-4.36e-4` persiste malgré le pas et le bracket
  raffinés, ce qui favorise une fenêtre opérationnelle/bassin plutôt qu’une
  simple discrétisation.
- Les quatre contrôles ascendants sont actifs et ont produit de nouveaux points
  jusqu’à 12:03. Depuis le lancement, 90 fichiers de points nouveaux ont été
  écrits. Charge 4.96/5.12/5.03, mémoire libre 65%, aucune page throttled,
  disque libre 614 GiB. Aucun signe de stagnation ni erreur.
- État à **12:11 CDT** : les quatre processus actifs sont toujours uniquement
  les audits ascendants 42, 44, 48 et 61; N2 n’a pas encore démarré. Résultats
  ascendants provisoires : pour 61, la frontière est déjà resserrée entre
  `0.3249953` (arrêt) et `0.3250741` (cycle), intervalle qui contient le pli
  exact `0.3250185`, alors que la frontière descendante est proche de
  `0.3245822`. Cela constitue une première signature nette d’hystérésis/bassin.
  Pour 48, le bracket ascendant provisoire
  `[0.3325163,0.3328744]` contient le pli de la branche d’arrêt
  `0.3325906`. Les brackets ascendants de 42 et 44 sont encore en cours de
  raffinement. Charge 4.90/4.95/4.97, mémoire libre 65%, aucune page throttled,
  disque libre 614 GiB.
- **Audit N1 terminé à 12:45 CDT.** Les huit raffinements
  (quatre graines, directions cyclique et arrêtée) sont résolus, intègres et
  agrégés dans
  `runs/N1_AUDIT/summary_dynamic_seeds_42_44_48_61_dt0p005_tours5_tol5e-05_hyst1.json`.
  Les graines 42 et 44 donnent exactement le même bracket dans les deux
  directions. La graine 48 donne `[0.3322925,0.3323372]` depuis le cycle et
  `[0.3325610,0.3326058]` depuis l’arrêt; ce dernier contient le pli exact de
  la branche d’arrêt `0.3325906`. La graine 61 donne
  `[0.3245625,0.3246018]` depuis le cycle et
  `[0.3249953,0.3250347]` depuis l’arrêt; ce dernier contient le pli exact
  `0.3250185`. N1 démontre donc simultanément convergence numérique (42),
  sélection d’une branche distincte (44) et fenêtres d’hystérésis résolues
  (48, 61).
- Incident N2 à **12:45 CDT** : l’amorçage fixe à `lambda_fold-0.03` ne
  converge pas pour la graine 52 (résidu `9.80e-4`). L’orchestrateur a arrêté
  proprement tout le lot N2 et n’a lancé aucune campagne suivante. Aucun
  résultat final/checkpoint n’est perdu.
- Correction validée : N2 choisit désormais le premier point régulier
  résidu-gaté et de même identité parmi des offsets proches. Pour la graine 52,
  l’offset `0.005` converge avec résidu machine; la continuation comporte 503
  points, tourne correctement et donne `lambda_fold=0.32502749`, en accord avec
  le pli augmenté indépendant. Compilation et 7/7 tests centraux passent.
- N2 relancé à **13:05 CDT** sur les graines 42, 60, 47 et 52, quatre
  processus monothreads. Charge initiale 2.65/3.26/3.12, mémoire libre 67%,
  aucune page throttled. La graine 51 restera le second lot. Le code
  d’orchestration a aussi été durci pour effacer les marqueurs d’échec obsolètes
  lors d’une reprise future.
- L’analyse scientifique consolidée de N1, avec portée, table exacte, claims
  autorisés et claims exclus, est enregistrée dans
  `N1_AUDIT_CONCLUSION.md`.

Last manual update: **2026-07-28 23:31 CDT**.

This is the authoritative restart log for
`paper_v5/NUMERICAL_WORKPLAN.md`.  All new code and outputs are under this
`paper_v5_numerics/` directory.  The historical folder

`/Users/leoflack/Desktop/Recherche/Chicago/numerics copie théorie claude + implémentation GPT`

is read-only input.  Its seven recorded source hashes still match the initial
snapshot in `manifests/reference_baseline.json`.

## Current resumed execution

- Machine preflight at 21:15 PDT: 66% system memory free and 615 GiB disk free.
  Process enumeration was unavailable; later file provenance showed that the
  three legacy pilots had in fact survived.
- N5A coherence provenance refreshed and five-seed analysis completed:
  midpoints 1.28225, 1.23475, 1.33786, 1.23384 and 1.23338 for seeds 42--46.
- N1A current-schema static thresholds completed at about 21:22 PDT:
  20/20 seeds, 100/100 branches per seed.  None of the 20 extreme branches has a
  twin warning; their alternate-solution separations are \(8\,10^{-12}\) to
  \(1.6\,10^{-9}\).
- Active run: N1 final dynamics.  The first batch, seeds 42--45, ran from
  21:22 PDT to about 00:01 PDT.  Seeds 42 and 44 completed under the final
  schema; seeds 43 and 45 remain explicitly unresolved after `timeout_other`.
  The second batch, seeds 46--49, started at 00:17 PDT as four mono-thread
  processes.
  Per-seed logs are `logs/N1_production_seed{seed}.log`; per-point checkpoints
  appear in `runs/N1/`.
- A restart-safe overnight orchestrator is active and waiting for the first
  batch.  It will then run seeds 46--61 in batches of at most four, without
  relaunching any completed or freshly active seed.  Its machine-readable state
  is `runs/N1/overnight_status.json` and its event log is
  `logs/N1_overnight_orchestrator.log`.  On a stale pre-existing run it stops
  instead of silently duplicating it.  A final unresolved seed is excluded, not
  force-classified; the planned gate then requires at least 15 admissible
  current-schema seeds.
- Correction discovered from file provenance at 21:46 PDT: the old pre-checkpoint
  pilot for seed 44 had survived the earlier process check and finished at 21:36,
  producing the scientifically plausible bracket \([0.33925,0.33950]\).
  Process enumeration was unavailable in the sandbox, which caused the false
  initial conclusion that no old process survived.  That legacy output lacks the
  final source hashes and independent-history flag and is therefore not accepted
  by the final gate.  The active current-schema seed-44 run will overwrite it.
  The orchestrator now requires the current `dynamics.py` hash,
  `arrest_primary_motif` and a successful independent-history control before
  treating any seed as complete.
- Initial resource check at 21:23 PDT: 67% system memory free, no throttled pages,
  615 GiB disk free.  Direct process/CPU and GPU telemetry are denied by the
  sandbox; liveness is tracked from the running execution session, logs and
  checkpoint timestamps.
- Hourly check at 22:16 PDT: all four final runs reached lambda 0.376 as ordered
  cycles with forward fraction 1.000.  Logs and checkpoints advanced at
  22:13--22:14, so there is no sign of stagnation.  Load averages were 8.59,
  8.21 and 8.80; memory free 64%, throttled pages 0, disk free 615 GiB.  No
  error marker was found in any final-run log.
- Hourly check at 23:16 PDT: seed 42 reached lambda 0.342, seed 44 reached
  lambda 0.342, and seeds 43 and 45 reached lambda 0.340; all remain ordered
  cycles with forward fraction 1.000.  Every log/checkpoint advanced between
  23:12 and 23:15, with no error marker and no stagnation.  Load averages were
  5.54, 5.22 and 5.56; memory free 65%, throttled pages 0, disk free 615 GiB.
  GPU/Metal remains unavailable in the sandbox.  Seed 44 is now immediately
  above its independently known static threshold; its final bracketing and
  independent-history controls should be the next completion event.
- Hourly check at 00:16 PDT:
  - seed 42 completed with dynamic bracket \([0.32750,0.32775]\), slowest bond
    and arrest motif 91, and successful independent-history controls.  Its
    dynamic bracket overlaps the static bracket
    \([0.3275017,0.3276383]\);
  - seed 44 completed with dynamic bracket \([0.33925,0.33950]\), slowest bond
    and arrest motif 95, and successful independent-history controls.  Its
    motif identity agrees, but this dynamic bracket does not overlap its static
    bracket \([0.3386270,0.3387841]\); this is retained as a measured
    discrepancy;
  - seeds 43 and 45 each reached an ordered cycle at lambda 0.322, then returned
    `timeout_other` at lambda 0.320 after 4.61 forward tours and the prescribed
    maximum simulated time 20000.  No class is forced and neither seed enters
    the current gate.
  Load averages were 1.50, 1.59 and 2.42 after the first batch ended; memory
  free 66%, throttled pages 0 and disk free 615 GiB.
- The orchestrator was corrected and validated to recognize those terminal
  unresolved logs, exclude them without relaunching, and continue the campaign.
  Seeds 46--49 were launched at 00:17 PDT.  The status currently records
  completed seeds `[42,44]`, unresolved seeds `[43,45]`, and running seeds
  `[46,47,48,49]`.
- Hourly check at 01:16 PDT: seeds 46--49 all reached lambda 0.368 as ordered
  cycles with forward fraction 1.000.  Their four logs and checkpoints advanced
  between 01:13 and 01:15, with no error marker or stagnation.  Load averages
  were 7.63, 5.98 and 5.75; memory free 66%, throttled pages 0 and disk free
  615 GiB.  GPU/Metal remains unavailable in the sandbox.  No action or new
  launch was needed.
- Hourly check at 02:17 PDT: seeds 46--49 all reached lambda 0.336 as ordered
  cycles with forward fraction 1.000.  Logs and checkpoints advanced between
  02:13 and 02:16, with no error marker or stagnation.  Load averages were
  5.07, 5.11 and 5.14; memory free 66%, throttled pages 0 and disk free
  615 GiB.  Seeds 48 and 49 are now close to their independently measured static
  extremes (about 0.329 and 0.330), so their bracketing phases should begin
  before those of seeds 46 and 47.
- Hourly check at 03:17 PDT:
  - seed 48 completed with dynamic bracket \([0.33225,0.33250]\), slowest bond
    and arrest motif 13, and successful independent-history controls.  The motif
    agrees with the static extreme, while the dynamic bracket lies above the
    static bracket \([0.3288204,0.3289570]\);
  - seed 49 completed with dynamic bracket \([0.33000,0.33025]\), slowest bond
    and arrest motif 77, and successful independent-history controls.  Its
    dynamic and static brackets overlap;
  - seeds 46 and 47 remained unresolved at lambda 0.326 and 0.318,
    respectively, after strictly forward trajectories reached only 4.42 and
    4.63 tours by the prescribed cap \(T_{\max}=20000\);
  - the next primary batch, seeds 50--53, launched automatically and reached
    lambda 0.388 by 03:16 with fresh checkpoints and no errors.
  Load averages were 6.74, 5.65 and 5.28; memory free 66%, throttled pages 0
  and disk free 615 GiB.
- Four of the first eight seeds are currently time-cap limited.  A validated
  restart-safe extension is therefore prepared in `src/n1_overnight.py`: after
  the primary pass, every `timeout_other` seed will resume from its last complete
  descent/bracket checkpoint with \(T_{\max}=40000\).  A descent timeout restarts
  at the failed lambda; a bisection timeout reconstructs the saved coarse bracket
  and repeats the unresolved bisection with the larger cap.
  The five-tour criterion and every classification gate remain unchanged.
  **Do not interrupt the currently loaded orchestrator to activate this edit.**
  After its primary pass terminates, restart the orchestrator with the same
  command before starting N2; it will discover terminal logs, perform the
  extended retries in batches of four, and only then rerun the N1 gate.
- Hourly check at 04:17 PDT: seeds 50 and 51 reached lambda 0.354, while seeds
  52 and 53 reached lambda 0.352; all four remain ordered cycles with forward
  fraction 1.000.  Logs and checkpoints advanced between 04:13 and 04:17, with
  no error marker or stagnation.  Load averages were 5.09, 5.06 and 5.07;
  memory free 65%, throttled pages 0 and disk free 615 GiB.  No intervention
  was needed.
- Hourly check at 05:17 PDT:
  - seed 51 completed with dynamic bracket \([0.33925,0.33950]\), slowest bond
    and arrest motif 27, and successful independent-history controls.  The motif
    agrees with the static extreme, while the dynamic bracket lies slightly
    below the static bracket \([0.3396682,0.3398253]\);
  - seed 52 entered bisection and classified lambda 0.32525 as an ordered cycle;
  - seed 50 reached lambda 0.322 and seed 53 lambda 0.324, both still ordered
    cycles with forward fraction 1.000.
  All active checkpoints/logs are fresh and contain no error marker.  Load
  averages were 5.05, 4.49 and 4.65; memory free 65%, throttled pages 0 and
  disk free 615 GiB.
- Hourly check at 06:18 PDT:
  - seed 52 completed with dynamic bracket \([0.32500,0.32525]\), slowest bond
    and arrest motif 40, successful independent-history controls and overlap
    with its static bracket;
  - seed 50 remained unresolved at lambda 0.320 after 4.49 forward tours by
    \(T_{\max}=20000\);
  - seed 53 remained unresolved at lambda 0.322 after 4.61 forward tours by the
    same cap.  Both are queued for the validated extended-time continuation;
  - seeds 54--57 launched automatically and all reached lambda 0.372 as ordered
    cycles with fresh checkpoints.
  The primary pass now has six final-schema completions and six time-cap-limited
  seeds.  Load averages were 5.30, 5.52 and 5.50; memory free 65%, throttled
  pages 0 and disk free 615 GiB.  No process is stagnant.
- Hourly check at 07:19 PDT: seeds 54--57 all reached lambda 0.338 as ordered
  cycles with forward fraction 1.000.  Their logs and checkpoints advanced
  between 07:16 and 07:18, with no error marker or stagnation.  Load averages
  were 7.79, 5.78 and 5.39; memory free 64%, throttled pages 0 and disk free
  615 GiB.  No intervention was needed.
- Hourly check at 08:20 PDT:
  - seeds 54, 55 and 57 reached a stationary-arrest coarse bracket, then remained
    unresolved during bisection at lambda 0.32525, 0.32050 and 0.32550 after
    4.64, 4.67 and 4.59 tours, respectively;
  - seed 56 remained unresolved during descent at lambda 0.320 after 4.65
    strictly forward tours;
  - all four are time-cap limitations, not forced attractor assignments, and are
    queued for the \(T_{\max}=40000\) continuation;
  - the last primary batch, seeds 58--61, launched automatically and reached
    lambda 0.390 with four fresh, error-free checkpoints.
  The primary pass has six final-schema completions and ten time-cap-limited
  seeds so far.  Load averages were 5.39, 5.59 and 5.57; memory free 65%,
  throttled pages 0 and disk free 615 GiB.
- User-return estimate at 08:55 PDT: seeds 58 and 59 reached lambda 0.368 and
  seeds 60 and 61 lambda 0.370, with fresh checkpoints.  Based on the measured
  3.3--3.5 minutes per descent point and their static-threshold neighborhood,
  the last primary batch should finish in about 1.5--2.5 hours.  The expected
  10--12 extended-time continuations should then require about 2--4 additional
  hours in batches of four, followed by less than 20 minutes for the N1 gate and
  N4 rebuild.  Best estimate for a completed N1 campaign is therefore
  13:00--16:00 PDT, with a conservative allowance to about 17:00 if several
  \(T_{\max}=40000\) points are needed.
- Shutdown audit at 09:07 PDT: all six final-schema NPZ/JSON pairs opened
  successfully, and every one of the 20 N1 checkpoint NPZ files opened with
  valid history and `next_lambda` fields.  No corruption was detected.  The
  active seeds 58--61 are durably checkpointed after 20 completed descent
  points, with their next lambda equal to 0.360.  A machine-readable snapshot is
  `runs/N1/shutdown_snapshot_2026-07-28T0859PDT.json`.  A 96 MiB local archive
  containing N1 results/checkpoints, logs, source, runbook and status is
  `runs/N1/N1_shutdown_backup_2026-07-28T0907PDT.tar.gz`, SHA-256
  `31323a41a79c9c0a73609148c046da63c19c8dc6635a40ac22af1d49a73c835b`.
  `sync` completed after writing the archive.
- Clean stop at 09:11 PDT: the orchestrator and seeds 58--61 were interrupted
  manually before computer shutdown.  Their log/checkpoint sizes and mtimes
  remained unchanged over the following five seconds.  Each final active
  checkpoint was reopened successfully and contains 21 completed points, last
  lambda 0.360 and next lambda 0.358.  All 20 N1 checkpoints reopen after the
  stop.  A post-stop archive was written and synced at
  `runs/N1/N1_shutdown_backup_2026-07-28T0912PDT.tar.gz`, SHA-256
  `cce14d62e3e6e38a38fd2e80d9adbc382ace2740700cb9471aa414da3546e5a1`.
  Full restart instructions are in `RESUME_SIMULATIONS.md`.  No numerical
  process should now be active.  The hourly heartbeat
  `surveillance-horaire-simulations-v5` was paused to prevent an automatic
  relaunch before the deliberate restart.
- Pre-restart review at 22:29--22:32 CDT:
  - the N1 theory, evidentiary purpose, completion gate, runbook, complete
    `dynamics.py`, `n1_overnight.py`, `n1_compare.py` and `run.sh` paths were
    reread;
  - the post-stop archive SHA-256 still matches
    `cce14d62e3e6e38a38fd2e80d9adbc382ace2740700cb9471aa414da3546e5a1`;
  - `./run.sh smoke` passed all four core tests and both N1 smoke paths;
  - all three production drivers compile;
  - the restart-selection audit gives exactly completed
    `[42,44,48,49,51,52]`, extended-time retries
    `[43,45,46,47,50,53,54,55,56,57]`, and primary checkpoint resumes
    `[58,59,60,61]`;
  - the ten censored trajectories had 4.42--4.67 strictly forward tours at
    \(T=20000\).  Linear extrapolation reaches five tours at
    \(T\simeq21400\)--22600, supporting \(T_{\max}=40000\) as a conservative
    censoring-resolution control while leaving the five-tour, forward-order,
    arrest and independent-history criteria unchanged;
  - machine preflight: load averages 1.82, 1.96 and 1.78; memory free 70%;
    disk free 614 GiB; no checkpoint/log activity since the clean stop.
  The remaining N1 campaign is estimated at 3--6 hours and is therefore below
  the user's ten-hour campaign limit.
- Restart launched at 22:30 CDT with the audited command in
  `RESUME_SIMULATIONS.md`.  The orchestrator correctly skipped the six valid
  finals, recovered the ten terminal primary timeouts for later extended retry,
  and launched only seeds 58--61.  All four reported
  `resumed at lambda=0.358000 after 21 completed points`; no earlier descent
  point was recomputed.  Initial load averages were 8.06, 4.35 and 2.75 during
  process startup, memory free 69% and disk free 614 GiB.  The hourly heartbeat
  is active again with the ten-hour-per-campaign ceiling and mandatory
  theory/objective/code/smoke/benchmark review before any later heavy campaign.
- Hourly check at 23:30 CDT: the four resumed checkpoints have advanced normally.
  Seed 58 has already obtained a coarse cycle/arrest bracket and classified the
  bisection point lambda 0.33125 as stationary arrest.  Seed 59 reached lambda
  0.328, seed 60 lambda 0.330 and seed 61 lambda 0.328; all three remain ordered
  cycles with forward fraction 1.000.  Logs/checkpoints advanced between 23:22
  and 23:30, with no new error marker or stagnation.  Load averages were 6.84,
  6.61 and 6.41; memory free 64%, throttled pages 0 and disk free 614 GiB.
  No intervention or additional campaign was needed.
- The legacy seed-42 pilot finished at 22:05 with bracket
  \([0.32750,0.32775]\) and slowest bond 91, consistent with the static extreme.
  The legacy seed-43 pilot terminated honestly at 22:10 on `timeout_other` near
  lambda 0.320 and wrote no final result.  These observations are diagnostics
  only; both legacy outputs are excluded by schema/source validation.
- Hourly heartbeat `surveillance-horaire-simulations-v5` is active.  It checks
  progression, memory, disk and observable process state.  GPU/Metal telemetry is
  unavailable inside the Codex sandbox and is recorded as such.
- Do not start another independent heavy campaign while the N1 overnight
  orchestrator is active.

## State recorded before the computer shutdown

### N1 dynamic pilot — seeds 42, 43, 44

- Started: 2026-07-27 17:41 PDT.
- Parameters: \(N=2000\), \(P=100\), \(\beta=20\), \(\tau=10\),
  \(dt=0.01\), descent \(0.40\to0.25\) by 0.002, five ordered tours,
  maximum simulated time 20000, final bracket width at most \(5\,10^{-4}\).
- State at 19:18 PDT: all three processes were still active; no final NPZ had yet
  been written and no error had been reported.  At 21:15, sandbox restrictions
  prevented reliable process enumeration and there was no final output yet.
  Subsequent provenance showed that all three pilots survived: seed 44 finished
  at 21:36, seed 42 at 22:05, and seed 43 stopped on an explicitly unresolved
  `timeout_other` at 22:10.  Their logs are retained as timing/scientific
  diagnostics and they are excluded from final-schema completion.
- Original estimate: 75--110 minutes total; about 97 minutes had elapsed at this
  update.  The likely remaining time was **0--13 minutes**.  A conservative upper
  allowance remained roughly **two additional hours** if a trajectory reached the
  near-boundary integration cap.
- Important restart fact: this pilot was launched with the pre-checkpoint driver
  and writes only at the end.  An interrupted seed cannot resume.
- This is a timing/scientific pilot only.  Seeds 42--44 must in any case be rerun
  with the current driver, because the final driver now stores raw relay events,
  dwell matrices, derivative histories, independent-history controls, arrest
  identities and complete source provenance.
- If the computer is shut down now, these three unfinished pilot processes are
  expected to be terminated.  Do not treat the absent NPZs as failed scientific
  outcomes; simply restart seeds 42--44 with the current driver when work resumes.
- Logs: `logs/N1_dynamic_seed42.log`, `logs/N1_dynamic_seed43.log`,
  `logs/N1_dynamic_seed44.log`.

### N5A short-delay coherence — completed

- Started: 2026-07-27 18:22 PDT.
- Seed 42 had already completed before this launch.
- Parameters: \(N=2000\), \(P=100\), \(\lambda=0.9\), 19 prescribed delay
  values, plus \(dt/2\) controls at \(\tau=1.2,1.3,1.4\): 22 files per seed.
- Completed at 18:59 PDT.
- Seeds 42--46: 22/22 files each; total 110/110.
- Campaign summary written to `runs/N5A/summary.json`.
- Every completed cell is already an independent NPZ, so this campaign is safely
  restartable even if the computer is closed.
- Log: `logs/N5A_coherence_seeds43_46.log`.

This subsection records the pre-shutdown state only; the current resumed state is
the first section above.

## Completed production work and established results

### N1A — full static thresholds

- Completed all 20 disorder seeds 42--61 at \(N=2000,P=100\).
- Every seed resolved 100/100 memory-connected branches, so all 20 static
  realizations pass the N1 admissibility count.
- Largest-threshold lower endpoints range from 0.3123828809 to 0.3396681646.
- First--second extreme gaps range from \(1.18818\,10^{-4}\) to
  \(1.58439\,10^{-2}\).
- Static extreme motifs:

  | Seed | motif | lower bracket | upper bracket |
  |---:|---:|---:|---:|
  | 42 | 91 | 0.3275016992 | 0.3276383403 |
  | 43 | 67 | 0.3181516807 | 0.3183088179 |
  | 44 | 95 | 0.3386269541 | 0.3387840914 |
  | 45 | 69 | 0.3171884771 | 0.3173456143 |
  | 46 | 22 | 0.3251796875 | 0.3252830078 |
  | 47 | 79 | 0.3159566406 | 0.3160754590 |
  | 48 | 13 | 0.3288203809 | 0.3289570220 |
  | 49 | 77 | 0.3300000000 | 0.3301796875 |
  | 50 | 39 | 0.3181516807 | 0.3183088179 |
  | 51 | 27 | 0.3396681646 | 0.3398253018 |
  | 52 | 40 | 0.3250000000 | 0.3251796875 |
  | 53 | 87 | 0.3218160156 | 0.3219348340 |
  | 54 | 48 | 0.3250000000 | 0.3251796875 |
  | 55 | 62 | 0.3206250000 | 0.3208046875 |
  | 56 | 77 | 0.3123828809 | 0.3125195220 |
  | 57 | 89 | 0.3252393359 | 0.3253759771 |
  | 58 | 21 | 0.3315375591 | 0.3316946964 |
  | 59 | 67 | 0.3141768359 | 0.3143134771 |
  | 60 | 65 | 0.3275016992 | 0.3276383403 |
  | 61 | 34 | 0.3250000000 | 0.3251796875 |

- The current-schema rerun is complete.  All branches include
  `twin_separation`; secondary near-twins are retained as warnings.  None of the
  20 extreme branches has a twin warning, with extreme-branch separations between
  about \(8\,10^{-12}\) and \(1.6\,10^{-9}\).

### N3 — E30 seed-aware reanalysis

- Completed on all 303 archived E30 files.
- Bootstrap: 5000 disorder-seed resamples; motifs were never treated as
  independent disorder realizations.
- Fixed \(\alpha=0.05\):
  - threshold-width exponent \(b=0.43078\), 95% CI
    \([0.39638,0.46520]\);
  - extreme-interval exponent \(d=0.27582\), 95% CI
    \([0.22538,0.33040]\);
  - pure power is ahead of the additive-floor width model by
    \(\Delta\mathrm{AICc}=2.11\), which is weak evidence, not a definitive
    model selection;
  - the independent-Gaussian extreme prediction has mean signed maximum error
    \(+0.00734\) and maximum absolute error about 0.02745; its extreme-gap form is
    disfavored relative to a free power by \(\Delta\mathrm{AICc}=8.37\).
- Fixed \(P=100\):
  - width exponent \(b=0.90312\), 95% CI \([0.88596,0.92102]\);
  - the power-plus-floor model is better than the pure power by
    \(\Delta\mathrm{AICc}=10.64\);
  - extreme-interval exponent about 0.75496, 95% CI
    \([0.73520,0.77489]\).
- Evidence wording: the fixed-\(\alpha\) result is consistent with
  central-limit-like narrowing, but the present CI does not support claiming an
  exactly measured \(1/2\) exponent.
- Main outputs:
  `runs/N3/E30_seed_aware_analysis.json` and
  `reports/N3_E30_seed_statistics.csv`.

### N4 — first publication composites

- Figures 1--4 have been assembled and visually checked in `figures/`.
- Figure 2c now contains the normalized shapes of all 100 branches, with raw
  transparent curves, median and interquartile band.
- Figure 2 marks the reference twin/threshold-resolution band.
- Figure 3 shows seed-aware standard errors and bootstrap exponent intervals;
  panel 3e remains blank until N1 passes.
- Figure 4 represents machine-floor Floquet multipliers only as contraction lower
  bounds, never as measured multiplier values.
- The plotting code is now configured for 600 dpi PNG plus vector PDF.  The files
  currently on disk were built before the final 600 dpi edit and should be rebuilt
  once N1 fills panel 3e.
- Panel-level source/key provenance:
  `manifests/N4_figure_provenance.json`.

### N5A — completed five-seed coherence result

- All 110 prescribed coherence files are complete for seeds 42--46.
- Coherence midpoints are 1.28225, 1.23475, 1.33786, 1.23384 and 1.23338.
- Direct multi-observable jump brackets are \([1.25,1.30]\),
  \([1.20,1.25]\), \([1.30,1.35]\), \([1.20,1.25]\) and
  \([1.20,1.25]\), respectively.
- At \(\tau=1.3\) and 1.4, the \(dt=0.01\) and \(dt=0.005\) observables agree to
  numerical precision.
- At \(\tau=1.2\), chaotic/desynchronized time averages change by about 5% under
  step halving; retain this as a measured sensitivity rather than hiding it.
- The five-seed analysis is saved; the separate boundary/Lyapunov campaign
  remains pending.

## Completed smoke tests and code validation

- Core tests: 4/4 passed:
  dense/operator coupling equivalence, exact reduced/full trajectory equivalence,
  ordered-relay event logic, and correlated-pattern generator correlations.
- N1 small-\(N\) static/dynamic code paths exercised; unresolved dynamics remain
  unresolved instead of being forced into a class.
- N2 pseudo-arclength smoke:
  \(N=120,P=6\), 53 continuation points, branch turned, fold near 0.297624.
  The revised smoke also resolved physical DDE roots with scaled residuals below
  \(3\,10^{-15}\).
- N5A smoke: seven cells.
- N5B smoke: one memory-branch scan with two spectral resolutions.
- N6 smoke: one streaming Lyapunov spectrum.
- N7 smoke: four correlated-pattern ensembles.
- N8 smoke: four high-load cells.
- N9 revised smoke: one cell; the adaptive wandering-history filter retained 4/4
  small-\(N\) candidates.
- N10 smoke:
  exact reduced/full relative error \(4.86\,10^{-16}\), reference static and
  Lyapunov checks, no forbidden-source hits, historical hashes unchanged.
  The full Floquet smoke was not included in that particular audit invocation.
- MLX is installed in `mcmc_env`, but Metal is unavailable inside the Codex
  sandbox and `mlx.core` aborts there.  All validated runs here therefore use the
  NumPy float64 backend.  `./run.sh mlx-preflight` can test MLX from a normal
  macOS Terminal.

## Implemented but not yet run at production scale

- N1 current dynamic driver: point checkpoints, no use of static information,
  raw relay events and dwell times, true second-memory history control, arrest
  identity and strict chaos/unresolved separation.
- N2: pseudo-arclength through the fold, \(M=48\) physical DDE roots with
  \(M=64\) controls, dynamic \(dt/2\) offsets, three dynamic fit models and local
  null/stable normal-form projections.
- N5A boundary: corrected selection of the last two moving and first two
  post-boundary Lyapunov points.
- N5B: continuous complex-root tracking, \(M=48/64\) residual gates,
  \(10^{-6}\) crossing refinement, eigenmode-aligned nonlinear probes, final-state
  classification and pinned controls for all 20 N1 seeds.  Spectral points are
  restartable.
- N6: streaming float64 Benettin/QR calculation with memory independent of
  accumulation time.
- N7: restart-safe correlated-pattern scans; refined points now start from the
  correct bracketing history and save their own raw files.
- N8: restart-safe high-load trajectories and Lyapunov files.
- N9: bounded float64 batches until 32 wandering, non-periodic histories are
  retained; restart-safe cell summaries.
- N10: numerical audit driver and historical hash guard.  Point-level figure
  provenance and the clean-build audit still need strengthening before release.

## Remaining work, in dependency order

1. **N1 final dynamics — active:** complete seeds 42--61 with the current
   checkpointed driver.  Seed 44 is complete; the overnight orchestrator owns the
   remaining queue.  A four-seed batch is expected to take roughly 2--4 hours,
   with longer near-boundary cases possible.
2. **N1 comparison/gate:** run `n1_compare.py`, require at least 15 admissible
   seeds, at least 90% bracket overlap, identity agreement except resolution-scale
   near-ties, and successful independent-history controls.  Then rebuild N4.
3. **N2 production:** seed 42 plus four N1-admissible seeds spanning the observed
   extreme-gap distribution.  Expected duration: several hours to more than one
   day because \(\delta=10^{-5}\) may require very long tours.
4. **N5A boundary/Lyapunov:** first benchmark one high-\(N\), \(k=1\) Lyapunov
   point; then launch the nine-delay, five-seed boundary scan.  Expected duration:
   approximately 1--3 days on CPU.
5. **N5B production:** five \(\alpha=0.07\) memory scans plus 20-seed pinned
   controls.  Expected duration: many hours to several days.
6. **N6 production:** five seeds, five lambda values, \(k=8\), \(T=4000\), with
   all prescribed half-step controls.  Expected duration: several days on CPU.
7. **N7 production:** five seeds for both five-point correlation families,
    static branches plus dynamic descents.  Expected duration: about 8--24 hours.
8. **N8 production:** classify the high-load grid and select the two chaotic
    parameter values needed by N9.  Expected duration: 1--3 days.
9. **N9 production:** five seeds, three parameter points, 32 retained wandering
    histories per seed, float64 \(T=6000\).  Expected duration: several days.
10. **N10 final audit:** full/reduced checks at \(N=500,2000\), three thresholds,
    dynamic/Floquet/Lyapunov reruns, exact point provenance and clean plotting
    build.  Expected duration: hours.

Before every production item, retain the established rule: run or re-run its smoke
test, benchmark the relevant high-\(N\) kernel, and report the estimated duration
before launching.

## Primary restart commands

Run from `paper_v5_numerics/` using `mcmc_env`.

```bash
# Confirm all current coherence files and refresh the full five-seed summary.
./run.sh N5A --stage production --confirm-heavy --mode coherence
conda run -n mcmc_env python src/n5a_analyse.py

# One current-schema static seed.
./run.sh N1-static --N 2000 --P 100 --seed 42 --nproc 4

# One current-schema dynamic seed.
./run.sh N1-dynamic --N 2000 --P 100 --seed 42 --dt 0.01 \
  --lambda-start 0.40 --lambda-stop 0.25 --lambda-step 0.002 \
  --bracket-tol 0.0005 --max-time 20000 --required-tours 5

# Final N1 statistics after all seeds.
conda run -n mcmc_env python src/n1_compare.py
```

The full command catalogue and scientific reasons remain in `RUNBOOK.md`.

## Corrective audit and restart — 2026-07-29 00:30--00:40 CDT

- The resumed primary pass produced two additional valid final-schema results:
  seed 58 has dynamic bracket \([0.33125,0.33150]\), arrest motif 21 and a
  successful independent-history control; seed 61 has bracket
  \([0.32450,0.32475]\), arrest motif 34 and a successful history control.
  Seeds 59 and 60 remained unresolved at the primary time cap.
- The first \(T_{\max}=40000\) retry batch (43, 45, 46, 47) again returned only
  about 4.1--4.3 tours.  Comparing their event records at \(T=20000\) and
  \(T=40000\) exposed a bookkeeping defect: after 200,000 recorded states the
  bounded-memory trajectory window was pruned, and the relay-tour count was
  recomputed only from that tail.  Thus longer integration erased earlier tours.
  These retry failures are computationally censored and must not be interpreted
  as a different attractor or as a physical failure of N1.
- The active second retry batch (50, 53, 54, 55) was stopped cleanly as soon as
  the defect was identified.  Its checkpoints were preserved; its last log lines
  are explicit `KeyboardInterrupt` markers.  No other heavy campaign was active
  after the stop.
- `src/dynamics.py` now accumulates every relay event in a streaming record while
  retaining only a bounded state tail for closure diagnostics.  The complete
  five-tour count, forward fraction, dwell times and slowest bond are computed
  from the untruncated event record.  The corrected source SHA-256 is
  `4a63e6681a0862c4314e6debbe522c09f9fe8d5cbdd87964b64ad2838f25d128`.
- Six core tests pass, including exact agreement with the original full-record
  event detector and a synthetic slow cycle that needs more than 10,000 time
  units and crosses the pruning boundary.  The N1 smoke test also passes without
  forcing its deliberately unresolved small-\(N\) boundary.
- The eight successful results made with the previous source remain admissible:
  every one stopped after five tours before the pruning boundary could affect
  classification.  Their approved source SHA is
  `bfe196db07865c23a0870341833d6681dd38a8f7cdef0d328adf22bba7df6b2d`.
  Only long censored trajectories require recomputation.
- Restart selection was tested explicitly: completed
  `[42,44,48,49,51,52,58,61]`; corrected extended retries
  `[43,45,46,47,50,53,54,55,56,57,59,60]`; no primary seed remains pending.
  A new `--retry-seeds` option routes the four cleanly interrupted retries
  50, 53, 54 and 55 directly to \(T_{\max}=40000\).
- Scientific purpose and gate were re-read before restart: the corrected runs
  measure the independent dynamic boundary without static input and leave every
  unresolved/chaotic outcome unresolved.  Based on the observed forward speeds,
  five tours should now be reached near \(T=21400\)--22600.  A single high-\(N\)
  corrected pilot (seed 43) is estimated at 20--45 minutes; if it validates the
  fix, the remaining eleven retries in batches of four are estimated at
  1.5--3.5 hours, well below the ten-hour ceiling.
- High-\(N\) corrected pilot completed at 00:59 CDT in about 17 minutes.  Seed 43
  reached five strictly forward tours at \(\lambda=0.320\), bracketed the
  transition as \([0.31825,0.31850]\), selected slowest/arrest motif 67 and
  passed both independent-history controls.  Its dynamic bracket overlaps the
  independently computed static maximum bracket
  \([0.3181517,0.3183088]\), whose motif is also 67.  The final NPZ and JSON
  reopen successfully; the corrected result contains 21,789 raw relay events.
  This validates both the streaming repair and the high-\(N\) runtime estimate.
- Machine check at 01:00 CDT: load averages 1.37, 1.46 and 1.59; memory free
  65%; throttling not reported; disk free 614 GiB.  No second heavy campaign was
  active during the pilot.
- With nine valid dynamic results now present, the remaining eleven corrected
  retries are `[45,46,47,50,53,54,55,56,57,59,60]`.  Using the measured pilot,
  the revised estimate is 1--2.5 hours in three batches of at most four CPU
  processes, still conservatively below the ten-hour ceiling.
- Corrected restart launched at 01:31 CDT.  The restart-safe selection skipped
  all nine valid finals, placed every incomplete seed directly in the extended
  pass and launched only `[45,46,47,50]` as the first four-process batch.
  Their logs confirm checkpoint resumes at lambda 0.320, 0.326, 0.318 and 0.320,
  respectively.  Startup load averages were 3.27, 2.00 and 1.78; memory free
  65% and disk free 614 GiB.
- Hourly check at 02:30 CDT: the first corrected batch completed cleanly and all
  four final NPZ/JSON pairs reopen:
  - seed 45: dynamic bracket \([0.31725,0.31750]\), motif 69, history control
    passed, static bracket overlap;
  - seed 46: dynamic bracket \([0.32575,0.32600]\), motif 22, history control
    passed, no static bracket overlap (static
    \([0.3251797,0.3252830]\));
  - seed 47: dynamic bracket \([0.31575,0.31600]\), motif 79, history control
    passed, static bracket overlap;
  - seed 50: dynamic bracket \([0.31850,0.31875]\), motif 39, history control
    passed, no static bracket overlap (static
    \([0.3181517,0.3183088]\)).
  In all four cases the independently selected dynamic slowest motif equals both
  the arrest motif and the static maximum motif.  The two non-overlaps are
  retained as genuine resolved discrepancies; no bracket or classification was
  changed to force the proposed equality.
- The second batch `[53,54,55,56]` is active.  Seeds 53--55 have completed their
  bisection classifications and are executing independent-history controls;
  seed 56 obtained a coarse cycle/arrest bracket after reaching
  \(\lambda=0.312\).  Logs/checkpoints advanced during this hour, so no
  stagnation criterion is met.  Load averages are 5.08, 5.00 and 4.83; memory
  free 65%, throttling not reported and disk free 614 GiB.
- N1 dynamics and comparison finished at 03:24 CDT.  All 20 static NPZ/JSON
  pairs and all 20 dynamic NPZ/JSON pairs reopen without error; every dynamic
  seed passes its independent-history control and has an approved provenance
  hash.  Twelve dynamics use the corrected streaming source and eight valid
  early completions use the approved pre-streaming source.
- **The predeclared N1 scientific gate failed.**  All 20 disorder seeds are
  admissible and the independently selected slowest motif, arrest motif and
  static-maximum motif agree for 20/20 seeds (Wilson 95% interval
  \([0.839,1]\)).  However, only 11/20 static and dynamic threshold brackets
  overlap, far below the required 90%.  The mean absolute midpoint discrepancy
  is \(3.92\times10^{-4}\), and the maximum is
  \(3.49\times10^{-3}\) for seed 48.  Non-overlap seeds are
  `[44,46,48,50,51,55,56,58,61]`.
- This gate failure is not reclassified or hidden: at the current numerical
  definitions and resolutions, N1 does **not** establish the claimed multi-seed
  equality of static and dynamic thresholds.  It does establish the distinct
  20/20 identity-selection result.  The comparison, full seed table and figures
  are saved in `runs/N1/comparison.json`, `reports/N1_static_dynamic.csv`,
  `figures/N1_static_dynamic.pdf` and `figures/N1_static_dynamic.png`.
- The orchestrator stopped after the failed gate, as designed.  N4 was not
  rebuilt and N2 was not started.  No numerical process remains active.  The
  03:30 CDT machine check gives load averages 1.56, 1.70 and 2.25; memory free
  65%, throttling not reported and disk free 614 GiB.
- A completion archive was written and synced at
  `runs/N1_completion_backup_2026-07-29T0330CDT.tar.gz` (303 MiB), SHA-256
  `11e0f7b9dfd73558c33b69f7a11e5972d66761b3a7214cd5f23da8e54d6923b8`.
  It contains all N1 runs/checkpoints, logs, comparison tables/figures, corrected
  source/tests, the workplan and the restart/status documentation.

## NEXT24H hourly check — 2026-07-29 14:04 CDT

- N2 is active on the corrected four-process batch for seeds
  `[42,47,52,60]`, restarted at 13:05 CDT.  The stale top-level error and the
  12:45 EXIT 1 line in `state.json`/the seed-52 log record the superseded first
  attempt; seed 52 has since passed the corrected near-fold initialization and
  is producing normal outputs.
- All four arclength traces, augmented folds, and the full ten-point static-side
  sweeps now exist.  The dynamic-side sweep is advancing: for every seed the
  \(\delta=10^{-5}\) runs at \(dt=0.01\) and \(dt=0.005\), and the
  \(\delta=2\times10^{-5}\) run at \(dt=0.01\), are complete.  The newest
  atomic NPZ/JSON pair was written for seed 42 at 14:03 CDT, so there is direct
  progress during this check and no stagnation condition.
- No N6/N8/N5 campaign has been started concurrently.  The orchestrator remains
  in N2 and will gate this phase before advancing.
- Machine state: load averages 6.74/5.84/5.61; system-wide memory free 66%;
  zero throttled pages; swap has historical activity but no critical memory
  pressure is reported.  Disk free is 613 GiB (33% used).  No resource or disk
  alert is active.

## NEXT24H hourly check — 2026-07-29 15:04 CDT

- The first four N2 production seeds completed cleanly: seed 52 at 14:48, seed
  42 at 14:51, seed 60 at 14:52 and seed 47 at 14:55 CDT.  Every summary JSON
  reopens, all four arclength traces turn through a fold, the augmented fixed and
  null residuals pass, and none of the 64 dynamic measurements is censored.
- The fifth and final requested N2 realization, seed 51, started automatically at
  14:55 CDT.  Its arclength trace and all ten static points are complete, and its
  first dynamic NPZ/JSON pair was written at 15:01.  This is current progress;
  there is no stagnation.
- Preliminary dynamic-side free exponents for completed seeds are 0.539 (42),
  0.520 (47), 0.410 (52) and 0.439 (60).  The logarithmic alternative is
  disfavored by very large AICc differences for all four.  These are provisional
  single-seed fits; the requested pooled/bootstrap and window-sensitivity gate
  must wait for seed 51.
- A validity problem was detected in the preliminary N2B static output and is
  explicitly retained as unresolved: for seeds 42, 47 and 52, the reported
  leading real root below the fold is positive, whereas N2B requires the stable
  memory-connected side with \(z_{\rm real}<0\).  The present Newton step from
  the exact fold appears able to select the unstable saddle branch.  Therefore
  the current static exponents for these seeds are not scientifically
  admissible.  Seed 51 is allowed to finish because its dynamic and geometric
  outputs remain useful; before the N2 gate, N2B must be audited and its static
  branch continued from the known stable N1 terminal state.  No conclusion is
  forced from the invalid static fits.
- Machine state: load averages 1.60/2.18/3.21; memory free 68%; zero throttled
  pages; no new swap-out count during this hour; disk free 608 GiB.  Only the
  single seed-51 N2 worker remains active and no other campaign overlaps it.

## NEXT24H hourly check — 2026-07-29 16:05 CDT

- N2 finished all five requested seeds at 15:35 CDT.  All five summary files
  reopen, all five augmented folds satisfy their residual gates, all arclength
  traces turn through the selected fold, and none of the 80 dynamic
  measurements is censored.
- **The strict N2 scientific gate is not passed by these raw summaries.**  Four
  dynamic free-power fits give exponents 0.539, 0.520, 0.410 and 0.439 and
  strongly reject the logarithmic alternative.  Seed 51 instead saturates near
  the static fold: its free-power fit degenerates toward zero exponent and the
  logarithmic curve has the best AICc.  This is consistent with the independently
  measured seed-51 dynamic bracket lying about \(2.5\times10^{-4}\) below the
  exact static fold, which is negligible for the \(10^{-3}\) equality plot but
  not for a local scaling scan down to \(10^{-5}\).  No universal dynamic
  exponent is inferred from seed 51.
- The previously detected N2B implementation error is confirmed: four seeds
  followed the unstable local saddle branch when Newton iteration was initialized
  at the exact fold.  Their positive leading real roots cannot be fitted to the
  required stable-side law.  A non-writing five-seed audit initialized from the
  known stable N1 terminal state gives negative leading zero-frequency roots at
  \(\delta=10^{-5}\) for every seed
  (`[-0.02076,-0.01223,-0.01896,-0.02501,-0.02035]` for
  `[42,47,51,52,60]`), validating the diagnosis and correction.
- `src/n2_local_laws.py` now labels and continues the stable N1 branch and adds
  `--refresh-static`, which reuses all completed dynamic trajectories.  The 60
  superseded static/summary JSON files were preserved in
  `runs/N2_INVALID_STATIC_BRANCH_20260729/`; the corrected source compiles and
  has SHA-256
  `45a8b9f61e34d366c7c070cd07ca1dc6e287d9fdb3445fac09659b6c5a571d26`.
  The corrected production refresh has not been launched because N6 is active;
  it will be smoke-tested/benchmarked and run after the current heavy campaign.
- N6 began at 15:35 CDT on seeds 42--45, four monothread processes.  Fresh
  outputs through 16:03 prove progress.  The completed \(dt=0.01\) points
  provisionally show 3--6 positive Lyapunov exponents across several seeds and
  \(\lambda=0.29,0.30,0.31\), a promising multi-seed hyperchaos signal; the
  prescribed half-step controls and full grid remain required before the gate.
- Machine state: load averages 5.24/5.36/4.96; memory free 66%; zero throttled
  pages and no new swap-outs; disk free 608 GiB.  No resource alert or
  stagnation is present.

## NEXT24H hourly check — 2026-07-29 17:05 CDT

- N6 seeds 43, 44 and 45 completed cleanly at 16:37--16:41 CDT, each with all
  five \(dt=0.01\) lambda values plus the prescribed \(dt=0.005\) control at
  \(\lambda=0.31\).  All summary and spectrum files reopen.
- The half-step/window evidence already establishes a significant provisional
  result for these three disorder realizations: at \(\lambda=0.31\), at least
  the first three Lyapunov exponents remain clearly positive under
  \(T=1000,2000,4000\) and \(dt=0.01\to0.005\).  The first three \(T=4000\)
  exponents at \(dt=0.005\) are approximately
  `[0.0515,0.0339,0.0226]` (seed 43),
  `[0.0553,0.0313,0.0206]` (seed 44), and
  `[0.0524,0.0363,0.0266]` (seed 45).  Thus hyperchaos is robust across these
  seeds; near-zero fourth-to-sixth estimates are not counted as part of this
  conservative conclusion.
- Seed 42 remains active because its specification includes a half-step control
  at all five lambda values (ten spectra rather than six).  Its newest completed
  file was written at 16:53 CDT, within the expected per-cell runtime, so there
  is no stagnation.  Seed 46 is queued in the orchestrator's second batch and
  will start only after all first-batch workers have exited; this explains the
  currently unused CPU slots and is not a missing task or duplicate.
- N2 corrected static production remains deferred until N6 ends, preserving the
  no-overlap rule for heavy campaigns.
- Machine state: load averages 2.30/2.43/2.78; memory free 66%; zero throttled
  pages and no new swap-outs; disk free 607 GiB.  No resource alert is active.

## NEXT24H hourly check — 2026-07-29 18:06 CDT

- Seed 42 completed all ten requested N6 spectra at 17:56 CDT.  The queued seed
  46 then started automatically and wrote its first \(T=4000\) spectrum at
  18:04, so progress is current and no stagnation criterion is met.
- Seed 42 confirms robust hyperchaos at \(\lambda=0.30,0.31,0.325\): at least
  three exponents remain positive under both \(dt=0.01\) and \(0.005\), and
  under the three accumulation windows.  Its \(\lambda=0.29\) and 0.32 controls
  do **not** pass: one time step ends on a negative spectrum while the other
  remains chaotic.  These two cells are therefore treated as basin/attractor or
  time-step-sensitive, not as validated hyperchaos.
- The discrepancy exposes an implementation limitation relevant to the final
  gate: the current campaign initializes each time step independently from the
  same random coefficient vector after a finite transient, rather than first
  retaining only an independently classified chaotic history as specified in
  N6.  The valid conservative conclusion remains the cross-seed
  \(\lambda=0.31\) result for seeds 42--45, where at least three exponents are
  stable under the controls.  The ambiguous seed-42 cells remain explicitly
  unresolved and will not be used to broaden the phase region.
- Seed 46's first point at \(\lambda=0.29,dt=0.01\) has five raw positive
  estimates, including three well-separated values
  `[0.0591,0.0346,0.0205]`; its half-step control at 0.31 and remaining grid
  points are still pending.
- N2 static refresh remains queued until N6 completes.  Machine state: load
  averages 6.61/4.13/3.11; memory free 66%; zero throttled pages and no new
  swap-outs; disk free 607 GiB.  No resource alert is active.

## NEXT24H hourly check — 2026-07-29 19:06 CDT

- N6 completed at 18:54 CDT.  All 34 requested spectrum files reopen and the
  scientific gate has been audited in `reports/N6_GATE_STATUS.md`.
- The conservative N6 hyperchaos gate **passes at \(\lambda=0.31\)**: all five
  seeds retain at least three exponents separated from zero under
  \(T=1000,2000,4000\) and the prescribed \(dt=0.01\to0.005\) controls.
  Seed-42 cells at 0.29 and 0.32 remain attractor/time-step sensitive and are
  excluded from any interval claim.  The neutral exponent and Kaplan--Yorke
  dimension are not resolved by eight exponents; only a finite-spectrum lower
  bound is supported.
- N8 started at 18:54 CDT on seeds 42--45.  Forty-one trajectory/checkpoint
  files exist and new files were written at 19:06, so the campaign is active
  and not stagnant.
- An N8 gate defect was detected before classification: the 1600-time-unit
  observation contains about 0.5--1.5 complete tours because the cycle period
  grows with \(P\), while the code required two tours before computing any
  Lyapunov exponent.  The resulting trajectory data are valid but the original
  process would label sustained forward motion as unresolved and skip every
  Lyapunov calculation.
- The corrected code now uses at least 50 consecutive relay transitions with
  forward fraction \(>0.99\) as the production moving-state preselection.  The
  already saved examples have 142--148 strictly forward transitions and pass
  this criterion.  `src/n8_highload.py` and `src/observables.py` compile with
  SHA-256 values
  `adc7db19c09e98da526ea2a5eab6ccbfb8c1a09ec306eddddb11019ba32d5955`
  and
  `9590fee8765262627d3b034c05deda818539137732bf1168ae0b5b2b31b137c6`.
  The running workers retain the old loaded code, but the restart-safe N8
  finalize pass will reuse every trajectory and compute the missing spectra.
- The orchestrator could not be paused from the sandbox (`operation not
  permitted`), so it is allowed to finish trajectory generation without
  interruption.  No duplicate campaign was launched.
- Machine state: load averages 5.07/4.83/3.93; memory free 63%; zero throttled
  pages and no new swap-outs; disk free 606 GiB.  No resource alert is active.

## User-requested workplan status — 2026-07-29 19:21 CDT

- Completed phases: N1 audit, raw N2 production and N6 production.
- N1 establishes 20/20 terminal-motif identity and, after the declared
  reliability exclusion of seed 48, 19/19 static/dynamic threshold agreement at
  \(10^{-3}\).  The finer audit retains branch-switching qualifications.
- N2 dynamic trajectories are complete for five seeds with no censoring.  Four
  seeds yield free dynamic exponents 0.410--0.539 and strongly reject a
  logarithmic law; seed 51 does not diverge about the static fold and remains an
  explicit exception.  The raw N2B static sweep selected an unstable saddle
  branch for several seeds.  Its corrected stable-branch code is validated, but
  the production static refresh and final pooled/window-sensitive analysis are
  still pending.
- N6 is complete.  Its conservative multi-seed hyperchaos gate passes at
  \(\lambda=0.31\): all five seeds retain at least three positive exponents
  under window and half-step controls.  Seed-42 cells at 0.29 and 0.32 remain
  basin/time-step sensitive; the neutral exponent and full Kaplan--Yorke
  dimension are unresolved.
- N8 is active on the first four seeds.  Trajectories for all \(\alpha=0.05\)
  and 0.09 cells and the first \(\alpha=0.13\) cells are saved.  The original
  workers are collecting valid trajectories but skip Lyapunov runs because of
  an overly strict two-tour preselection.  Corrected restart-safe code is ready
  and will reuse those trajectories to compute the missing classifications and
  spectra.  Seed 46 remains queued after the first batch.
- N5A and N5B have not started in this pass.  They remain ordered after N8:
  short-delay boundary/coherence, long-delay memory branches, then pinned-front
  controls.
- Current resources: load averages 5.60/5.55/5.04, memory free 63%, no
  throttling, disk free 606 GiB.  N8 outputs advanced at 19:18; no stagnation or
  resource alert.

## Full-workplan continuation and ETA — 2026-07-29 19:25 CDT

- The user authorized continued autonomous execution through the complete
  workplan.  `FULL_WORKPLAN_EXECUTION.md` now records the post-NEXT24H sequence,
  restart/resource rules and current ETA.
- Seed policy is fixed prospectively: no seed is removed merely for being an
  outlier.  Fit/headline exclusions require an independent convergence,
  precision, branch-identity or attractor-class criterion; all excluded results
  and reasons remain reported.  Full-sample and admissible-set sensitivity
  results will be shown where relevant.
- Revised ETA from measured and high-\(N\) benchmarked runtimes:
  - priority N2/N5/N6/N8 campaign: 12--18 h remaining;
  - full workplan through N3, N7, N9, N4 and N10: approximately 3--6 days of
    continuous CPU computation, central estimate about four days.
  N9 is the main uncertainty because its runtime depends on the fraction of
  candidate histories retained in the wandering ensemble.  Its estimate will be
  tightened after a high-\(N\), \(K=4\) benchmark.
- The hourly automation was extended from the current 24-hour campaign to the
  complete workplan.  It will remain active until all numerical gates,
  aggregation and figure provenance checks through N10 are complete.

## Scope revision — 2026-07-29 19:30 CDT

- At the user's request, N7, N9 and N10 are removed from the execution scope.
- N4 is not a numerical campaign: it is the later recomposition of validated
  outputs into article figures, derived tables and captions.  It is postponed
  to the manuscript/figure-production phase and excluded from the current
  calculation ETA.
- The retained work is now: finish active N8; complete queued N5A/N5B; run the
  corrected restart-safe N8 and N2 passes; execute N3 reanalysis; aggregate and
  gate N1/N2/N3/N5/N6/N8.
- Revised ETA for all retained calculations: **14--22 h remaining**.  The
  hourly automation prompt and `FULL_WORKPLAN_EXECUTION.md` have been updated
  accordingly.

## Article-figure deadline — 2026-07-29 19:45 CDT

- Priority changed from completing every retained numerical refinement before
  figure work to delivering the essential manuscript figures by 2026-07-30
  12:00 CDT.
- The complete, placeholder-free figure package is now stored in
  `/Users/leoflack/Documents/Codex/2026-07-25/automate/outputs/paper_v5/figures_ready`.
- Figures 1--4 were regenerated from existing validated data. Figure 3 now includes
  the completed N1 paired test: 19/19 admissible seeds agree at \(10^{-3}\), with
  seed 48 excluded by the independent non-identifiability criterion of overlapping
  leading static brackets.
- A new N6 publication panel was generated from the completed campaign. All five
  seeds retain at least three positive Lyapunov exponents at \(\lambda=0.31\) under
  the factor-two time-step control; no Kaplan--Yorke dimension is inferred.
- Delay, short-delay, chaos-geometry and both correlation panels reuse the validated
  historical results already copied into the paper workspace. The historical
  Desktop directory remains unchanged.
- `paper_v5/main.pdf` compiles successfully (30 pages) with no placeholder panels.
  Visual inspection of all eight main-figure pages found no clipping, overlaps or
  broken rendering.
- The active N8 campaign continues and is writing outputs (53 N8 files modified in
  the preceding hour at the 19:42 control). Machine state remains safe: no throttled
  pages and 606 GiB free on disk.
- Corrected N2, N5 and N8 refinements can strengthen or replace panels only after
  their gates pass; none currently blocks the core figure deadline.

## Hourly monitor — 2026-07-29 20:07 CDT

- N8 remains active on seeds 42--45. Although the worker logs are line-buffered and
  still show only their start records, trajectory checkpoints advanced through the
  \(\alpha=0.15,\lambda=0.35\) cells at 20:04--20:05. This is direct progress and
  is not a stagnation.
- Four new \(\alpha=0.15\) trajectory files and the preceding
  \(\alpha=0.13,\lambda=0.90\) files reopen at nonzero sizes. No duplicate
  campaign was launched.
- Machine state is safe: load averages 4.35/4.84/5.08 for four monothread workers,
  79,496 free memory pages, zero throttled pages and 606 GiB free disk. Existing
  cumulative swap counters are nonzero, but no critical memory-pressure signal is
  present.
- The article package remains intact: `paper_v5/main.pdf` reopens as a 30-page,
  2,753,356-byte PDF. No figure replacement was triggered because N8 has not
  completed its scientific gate.

## Hourly monitor — 2026-07-29 21:08 CDT

- The first N8 batch completed normally for seeds 42--45 at 20:51--20:52, with
  per-seed summaries and all requested trajectory files present. The orchestrator
  then started the queued seed-46 batch without overlap or duplication.
- Seed 46 is progressing rapidly: trajectories and one-exponent Lyapunov files
  advanced successively through the \(\alpha=0.05\) and \(\alpha=0.09\) cells,
  most recently \(\alpha=0.09,\lambda=0.70\) at 21:07. The N8 phase gate remains
  pending until this final batch and the restart-safe corrected analysis complete.
- Machine state remains acceptable. Load averages are 6.22/5.28/5.26; system-wide
  memory-free percentage is 48%, no pages are throttled, and disk free is 605 GiB.
  The low instantaneous free-page count is offset by inactive/compressed memory and
  does not constitute critical pressure.
- The article package is unchanged and intact: 17 files in `figures_ready` and a
  valid 30-page, 2,753,356-byte `main.pdf`. No panel replacement is scientifically
  justified yet.

## Hourly monitor — 2026-07-29 22:08 CDT

- Original N8 trajectory collection completed normally for all five seeds at 21:33.
  All per-seed summaries and restart-safe trajectory outputs are present. This is
  completion of raw collection only: the corrected moving-state preselection and
  final N8 scientific gate remain pending, so no phase-boundary conclusion is yet
  reported.
- The orchestrator started N5A boundary scans for seeds 42--45 at 21:33 without
  overlapping another heavy campaign. Outputs continue to advance through the
  \(\tau=1.2\) grid at 22:08, including stored maximal-Lyapunov controls near the
  moving-state boundary. No stagnation is present.
- Machine state remains safe for the four-worker phase: load averages
  6.58/7.49/7.77, system-wide memory-free percentage 45%, zero throttled pages and
  605 GiB free disk. No new swap-outs were observed relative to the preceding
  control.
- The figure deadline package remains unchanged and valid: 17 files in
  `figures_ready` and a readable 30-page `main.pdf`. N5A has not yet reached its
  gate, so the validated historical delay panel is retained.

## N8 preliminary analysis — 2026-07-29 22:23 CDT

- All 100 expected N8 trajectory files reopen and pass the required finite-array
  integrity checks.  This is a grid of four loads, five non-reciprocity values
  and five disorder seeds, with one prescribed retrieval-like initial history
  per seed and cell; it is not a basin-volume measurement.
- The saved legacy `classification` field is not used.  Its two-tour requirement
  is incompatible with most fixed-duration high-load records, and the worker's
  dictionary merge order can overwrite the Lyapunov-aware label.
- A prospective trajectory-quality audit requiring at least 50 relay events,
  nearest-neighbour forward fraction above 0.99, reversal fraction below 0.01
  and a nonzero fixed residual finds 96/100 strict sequential trajectories.
  Nineteen of the twenty parameter cells are strict sequences for all 5/5
  disorder realizations.
- The only heterogeneous cell is
  \((\alpha,\lambda)=(0.15,0.35)\): seed 44 is a strict sequence; seed 42 is a
  degraded jumpy forward state; seeds 43 and 46 are irregular non-sequential
  states; seed 45 is a fixed point.  No seed was discarded.
- Strict sequence motion is stationary across the saved record: for every one
  of the 96 trajectories, the smallest-to-largest transition count among the
  four observation quarters is at least 0.96.  Conditional relay speed depends
  mainly on \(\lambda\), while binary retrieval quality decreases with load and
  improves with \(\lambda\).
- Existing Lyapunov coverage is 19/96 strict trajectories, all from seed 46.
  The cumulative leading estimates at accumulation time 1000 lie in
  \([-0.008271,-0.002940]\), but their last-250-time-unit block means lie in
  \([-0.001341,+0.001202]\), with 0/19 above the predeclared \(0.002\)
  positivity threshold.  The estimates approach the neutral direction and
  resolve no chaos for seed 46; multi-seed certification remains pending.
- The reproducible analysis, audit tables and preliminary figure are in
  `reports/N8_PRELIMINARY_ANALYSIS.md`,
  `reports/N8_trajectory_audit.csv`, `reports/N8_cell_summary.csv`, and
  `figures/N8_preliminary_audit.{pdf,png,svg}`.  No N8 figure has replaced an
  article panel because the multi-seed Lyapunov gate and multi-tour periodic
  closure are not complete.

## Hourly monitor — 2026-07-29 23:08 CDT

- N5A remains active on four monothread workers, seeds 42--45.  All four have
  entered the \(\tau=3\) boundary grid and wrote new checkpoints at
  23:07:38--23:07:52.  Current per-seed output counts are 221--223 files, so
  there is direct progress and no stagnation.
- The newest files reopen with all required stored arrays finite.  `dwell`
  contains expected NaNs for motifs not visited in the finite observation and
  is not treated as corruption.
- Machine state remains acceptable: load averages 6.02/5.92/5.98, zero
  throttled pages, no memory-pressure failure signal, and 604 GiB free disk.
  Cumulative swap counters are nonzero but no new critical-pressure condition is
  established.
- The N8 preliminary audit is complete and preserved; the multi-seed Lyapunov
  completion remains queued after the active heavy family.  The article package
  is unchanged and intact: 17 files in `figures_ready` and a readable
  30-page `main.pdf`.

## Hourly monitor — 2026-07-30 00:08 CDT

- N5A boundary production completed normally for seeds 42--45 at
  23:40--23:41 CDT.  Each completed seed reached the full \(\tau=10\) grid and
  saved 297--298 boundary/Lyapunov files plus its summary.
- The orchestrator then started the remaining seed-46 boundary scan at 23:41
  without overlap.  It has progressed through the \(\tau=1.0\) and 1.2 grids
  and is writing new checkpoints at 00:07:51; there is no stagnation.
- Latest outputs for every seed reopen successfully with all required numerical
  arrays finite.  Expected NaNs in unvisited-pattern `dwell` entries are
  excluded from the integrity criterion.
- Machine state is safe: load averages 3.12/2.71/2.97 with one active scientific
  worker, system-wide memory free 51%, zero throttled pages and 603 GiB free
  disk.  No critical swap or memory-pressure event is present.
- N5A scientific aggregation remains pending until seed 46 completes.  The
  article package remains unchanged and intact: 17 files in `figures_ready` and
  a valid 30-page `main.pdf`.

## Error, correction and gate audit — 2026-07-30 01:13 CDT

- N5A completed normally for all five seeds at 01:05:57.  Each seed produced
  all 279 requested boundary cells.  The lightweight aggregation was executed
  and saved to `runs/N5A/analysis.json`.
- The N5A coherence crossover is seed-consistent: interpolated midpoints span
  1.233--1.338 and the largest direct jumps lie in brackets 1.20--1.35.
  However, the full N5A completion gate **fails**.  Most delay/seed boundary
  rows have only two Lyapunov-resolved points rather than the required four, and
  the trajectory classifications do not resolve a seed-robust periodic/chaotic
  moving boundary.  Consequently the new N5A figure is diagnostic only and does
  not replace the validated historical article panel.
- N5B memory production started at 01:05:57.  Seed 42 failed immediately and
  unambiguously because its production static file path collided with a
  pre-existing smoke checkpoint whose bracket tolerance was \(2\times10^{-3}\)
  instead of the production \(10^{-6}\).  Seeds 43--45 continue normally; the
  failed seed-42 process had already exited and required no termination.
- All six legacy seed-42 smoke artifacts were preserved, not deleted, in
  `runs/N5B/archive_smoke_legacy_20260729/`.  `src/n5b_long_delay.py` now places
  smoke static, spectral and pinned-control checkpoints in an explicit
  `_smoke` namespace while leaving production names unchanged.  Its corrected
  SHA-256 is
  `35301f6fd5f34860a907ee1712db49a2b3cd1a74ca249cd42c614df660994f2c`.
- The corrected N5B smoke test passed end to end in 25.14 s at \(N=2000\),
  producing one memory scan and one pinned control with separate smoke paths.
  After this validation, seed-42 production was relaunched as the fourth and
  only recovery worker (execution session 83521).  Its production static
  checkpoint was written successfully at 01:12:35.  Together with the active
  seed-43--45 workers this respects the four-monothread limit; seed 46 remains
  pending.
- At the 01:08 resource check, load averages were 1.78/2.41/2.55,
  system-wide memory free 52%, throttled pages zero and disk free 603 GiB.  The
  article package remains intact and unchanged.

## User status and N5B recovery — 2026-07-30 01:24 CDT

- A direct process audit found that the failed original N5B orchestrator had
  exited and silently terminated its seed-43--45 children, although
  `runs/NEXT24H/state.json` still marked them as running.  Their production
  static checkpoints are intact, but they had written no spectral rows.  This
  stale state is now documented and is no longer trusted as a liveness source.
- Corrected seed-42 production is active in execution session 83521 and has
  already written six of forty spectral rows, progressing from \(\tau=40\) into
  \(\tau=45\).  The measured early rate is approximately one row every
  1.5--2 minutes.
- Seeds 43, 44 and 45 were relaunched from their valid static checkpoints in
  execution sessions 61311, 33274 and 50885 respectively.  All three sessions
  are alive.  Together with seed 42, exactly four monothread scientific
  processes are active.  Seed 46 remains queued until one slot is released.
- Current machine state remains safe: load averages 2.12/2.17/2.24,
  system-wide memory free 55%, zero throttled pages and 603 GiB free disk.
- Conservative overnight estimate from measured spectral throughput:
  memory branches for seeds 42--45 should finish around 02:30--03:00, seed 46
  around 03:30--04:15, and the split pinned-front controls around
  05:00--06:30.  Corrected N8/N2 completion, N3 reanalysis and final gates then
  require roughly another 3--5 h.  The retained numerical program is therefore
  expected to finish around 08:00--11:30 CDT if no further gate or code defect
  appears.  The complete article figure package is already safe and does not
  depend on these refinements.

## Hourly monitor — 2026-07-30 02:10 CDT

- All four manually recovered N5B memory sessions are alive and advancing.
  Completed spectral-row counts are currently seed 42: 16/40, seed 43: 9/40,
  seed 44: 6/40 and seed 45: 5/40.  The newest checkpoints were written between
  02:03 and 02:09, so none is stagnant.  Seed 46 remains queued.
- The differing row counts reflect seed-dependent root-solver cost; no output
  error or failed gate has appeared.  All work is in the same N5B family and the
  four-process limit remains respected.
- Load averages are elevated at 8.16/8.73/8.68 during the four spectral
  solvers, but memory and storage remain safe: system-wide memory free 55%,
  zero throttled pages and 603 GiB free disk.  These float64 CPU eigensolvers
  have no suitable MLX/GPU path in the validated implementation.
- The article package remains unchanged and intact: 17 files in
  `figures_ready` and a valid 30-page `main.pdf`.  No new figure passes a gate
  warranting replacement.

## Hourly monitor — 2026-07-30 03:12 CDT

- All four N5B memory sessions remain alive and checkpointed.  Spectral-row
  counts advanced to seed 42: 24/40, seed 43: 17/40, seed 44: 11/40 and
  seed 45: 11/40.  Latest writes occurred at 03:04--03:10, so no task is
  stagnant; seed 46 remains queued.
- Across the 63 completed production rows, every selected \(M=64\) complex root
  passes the \(10^{-10}\) residual gate and none has positive real part.  This
  is interim evidence against a resolved Hopf preemption in the portion scanned,
  not a final conclusion before all delays and seeds complete.
- Four-way spectral throughput is lower than the isolated benchmark because of
  seed-dependent eigensolver cost and shared-resource contention.  The earlier
  completion estimate is therefore optimistic; it will be revised when the
  first full seed establishes a production wall time.  Checkpointing makes this
  slowdown safe and no process is running without output.
- Machine state remains acceptable: load averages 5.47/5.41/5.68,
  system-wide memory free 52%, zero throttled pages and 603 GiB free disk.
  The article package remains unchanged and valid.

## Hourly monitor and N5B conjugacy audit — 2026-07-30 04:13 CDT

- All four N5B memory sessions continue to progress.  Counts are now
  seed 42: 30/40, seed 43: 24/40, seed 44: 15/40 and seed 45: 15/40, with
  latest checkpoints at 04:06--04:12.  Seed 46 remains queued and no
  stagnation is present.
- Across the 84 completed rows, every selected complex root passes its scaled
  residual gate and none has positive real part.  Current maximum real parts
  are -0.00853, -0.03064, -0.02144 and -0.02306 for seeds 42--45 respectively;
  these are interim values until each full branch is complete.
- A comparison defect was identified in `resolution_agreement`: a real system
  has conjugate eigenvalue pairs, but the code compared the signed imaginary
  representatives directly.  Seeds 42--43 often selected opposite members of
  the same pair at \(M=48\) and 64, producing a false discrepancy
  \(2|\operatorname{Im}z|\).  For the audited \(\tau=40\) points, the apparent
  differences 0.0857 and 0.0536 reduce to \(6.9\times10^{-18}\) and
  \(2.1\times10^{-17}\) after conjugacy-aware matching.
- `src/n5b_long_delay.py` now compares
  \(\min(|z_{48}-z_{64}|,|z_{48}-\bar z_{64}|)\), including refined crossings.
  It compiles and the correction is validated against all four existing
  \(\tau=40\) examples; SHA-256 is
  `205311c3fecc963a2d8bc52749d936edc4420e3a78bfacc865a815768eda70e4`.
  The running workers retain their loaded code, so the final aggregation must
  recompute this diagnostic from stored roots rather than trust the legacy
  scalar field.  No raw spectral result is affected.
- Resources remain safe: load averages 5.25/5.19/5.11, system-wide memory free
  55%, zero throttled pages and 607 GiB free disk.  The article package is
  intact and unchanged.

## Hourly monitor and recovery scheduling — 2026-07-30 05:14 CDT

- N5B spectral-row counts reached seed 42: 35/40, seed 43: 31/40,
  seed 44: 19/40 and seed 45: 19/40.  Latest checkpoints were written at
  04:59--05:11; all four execution sessions remain alive and no stagnation is
  present.
- None of the 104 completed \(M=64\) complex roots has positive real part, and
  all selected-root residuals pass.  Conjugacy-aware resolution agreement is
  below \(10^{-6}\) for every row except two seed-45 cells
  (\(\tau=40,50\), offset \(5\times10^{-4}\)), where \(M=48\) and 64 select
  genuinely different root families despite individually small residuals.
  Those two cells remain unresolved for the resolution-stability gate.
- Because the original orchestrator is dead, a restart-safe recovery supervisor
  was compiled and launched in execution session 44459.  It currently waits
  without CPU load for the first seed-42--45 production completion, then starts
  seed 46 in the released slot.  After all five memory branches finish it will
  launch four disjoint pinned-control chunks and maintain the four-scientific-
  process limit.  Its state is persisted in `runs/N5B/recovery_state.json`;
  source SHA-256 is
  `b52e709725467d96261ce90543faddd9bcb9c5feb06758f3970d5ab6f5fb19f5`.
- Current load averages are elevated but acceptable at 10.34/7.04/5.95 during
  the eigensolvers; system-wide memory free is 50%, throttled pages remain zero
  and disk free is 607 GiB.  The article package is unchanged and valid.

## Hourly monitor and first completed N5B seeds — 2026-07-30 06:15 CDT

- Seeds 42 and 43 completed their full 40-row memory-branch scans normally at
  06:08--06:10.  Seed 42 has fold 0.2340007391 and seed 43 fold 0.2191971208.
  Neither seed has a candidate complex crossing, refined crossing or nonlinear
  probe: the \(M=64\) complex real parts remain strictly negative throughout,
  with maxima -0.0047675 and -0.0139634 respectively.
- All 80 completed rows for these two seeds pass the residual and
  conjugacy-aware resolution gates to numerical precision.  This is a complete
  two-seed result supporting complex spectral softening without resolved Hopf
  preemption.
- Seeds 44 and 45 remain active at 24/40 rows each and wrote checkpoints at
  06:10.  The recovery supervisor detected the released slot and launched
  seed 46 at 06:08:32; its first row was saved at 06:14.  Thus three memory
  workers are active, no duplicate was launched and no task is stagnant.
- Machine state is safe: load averages 4.30/4.63/4.90, system-wide memory free
  50%, zero throttled pages and 607 GiB free disk.  The article package remains
  unchanged and valid.

## Hourly monitor — 2026-07-30 07:16 CDT

- N5B memory progress is now seed 42: 40/40 complete, seed 43: 40/40 complete,
  seed 44: 32/40, seed 45: 32/40 and seed 46: 14/40.  The three active workers
  wrote checkpoints at 07:13--07:14, so no stagnation is present.
- None of the 158 completed rows has a positive \(M=64\) complex real part and
  all selected-root residuals pass.  Resolution agreement remains exact modulo
  conjugacy except for the two previously identified seed-45 root-family
  ambiguities, which remain explicitly unresolved.
- The recovery supervisor is healthy and continues to wait for all five memory
  checkpoints before starting the four pinned-control chunks.  Current
  throughput suggests memory completion around 09:00 CDT.
- Machine state remains safe: load averages 3.94/4.35/4.34, system-wide memory
  free 51%, zero throttled pages and 607 GiB free disk.  The article package is
  intact and unchanged.

## Hourly monitor — 2026-07-30 08:17 CDT

- Seed 44 completed its 40/40 N5B memory rows normally at 08:16.  Seed 45 is at
  39/40 and seed 46 at 26/40, with current checkpoints at 08:12--08:15.
  Seeds 42--44 are complete; no task is stagnant.
- All 185 completed rows retain negative complex real parts and pass the
  residual gate.  Seed 44 completes with maximum complex real part
  -0.0078914 and no candidate crossing.  Its resolution agreement passes at
  every point.  Seed 45 retains the two previously declared root-family
  ambiguities, with no positive real part.
- The recovery supervisor remains in `seed46_running` state and will launch
  pinned controls only after the final memory checkpoint.  Current throughput
  places memory completion near 09:15--09:45 CDT.
- Machine state remains safe: load averages 3.34/4.16/4.30, system-wide memory
  free 50%, zero throttled pages and 607 GiB free disk.  The article package is
  intact and unchanged.

## N5B memory completion and pinned start — 2026-07-30 09:18 CDT

- All five N5B memory-branch scans are complete: 200/200 production spectral
  rows.  The folds for seeds 42--46 are respectively 0.23400074, 0.21919712,
  0.22506768, 0.24499230 and 0.22983400.
- At both \(M=48\) and 64, every computed complex real part is negative for
  every seed.  The least-negative \(M=64\) values are -0.0047675, -0.0139634,
  -0.0078914, -0.0123338 and -0.0151893.  No candidate or refined complex
  crossing was found and therefore no nonlinear Hopf probe was triggered.
- All 200 selected roots pass the residual gate.  Resolution selection agrees
  modulo conjugacy in 198/200 rows.  In the two seed-45 ambiguous cells the
  resolutions select different root families, but both selected real parts
  remain negative.  The supported five-seed conclusion is therefore spectral
  softening without a resolved Hopf preemption; no Hopf-born attractor is
  claimed.
- The recovery supervisor completed seed 46 at 08:31:51 and launched four
  disjoint pinned-control chunks immediately.  Each chunk has written 7/90
  controls, most recently at 09:12--09:13.  The four-process limit is respected
  and no task is stagnant.
- Measured four-way pinned throughput is about 37 controls/hour in aggregate,
  much slower than the isolated benchmark; completion is currently estimated
  around 17:30--19:00 CDT.  These controls are supplementary and do not block
  the already complete noon figure package.
- Machine state remains safe: load averages 5.17/5.34/5.41, system-wide memory
  free 50%, zero throttled pages and 607 GiB free disk.  The article package is
  intact and unchanged.

## Hourly monitor — 2026-07-30 10:20 CDT

- The four restart-safe N5B pinned-control chunks have advanced to 71/360
  production controls: chunk counts are 18/90, 17/90, 18/90 and 18/90.
  Checkpoints were written continuously through 10:18:54, including all four
  disjoint seed groups, so there is no stagnation.
- All 71 production JSON files reopen successfully.  Direct process inspection
  is denied by the current sandbox, but file progression is the authoritative
  liveness signal for this checkpointed campaign.
- The measured aggregate rate since the previous hour remains about 40
  controls/hour.  With 289 controls remaining, the conservative completion
  window remains approximately 17:30--19:00 CDT; no duplicate or overlapping
  scientific campaign has been launched.
- Machine state is safe: load averages 5.88/6.01/6.13, system-wide memory free
  53%, zero throttled pages and 607 GiB free disk.  No memory or storage alert
  is present.
- The article package is unchanged and intact: 17 files in `figures_ready` and
  a readable 30-page `main.pdf`.

## Hourly monitor — 2026-07-30 11:21 CDT

- N5B pinned controls have advanced from 71/360 to 112/360.  Each of the four
  disjoint chunks is at 28/90, with fresh checkpoints from all four groups at
  11:16--11:17.  No task is stagnant and no duplicate is present.
- All 112 production JSON files reopen successfully.  The measured throughput
  is stable at about 41 controls/hour; 248 remain, keeping the projected
  completion near 17:15--19:00 CDT.
- Resources remain safe during the four CPU workers: load averages
  5.98/6.37/6.38, system-wide memory free 51%, zero throttled pages and
  607 GiB free disk.
- The noon deliverable remains complete before the deadline: 17 files are
  present in `figures_ready`, and `main.pdf` is a readable 30-page PDF.  No
  incomplete N5B result has been substituted into the article.

## Hourly monitor — 2026-07-30 12:23 CDT

- N5B pinned controls have advanced to 155/360, with chunk counts
  38/90, 39/90, 39/90 and 39/90.  All four groups wrote checkpoints between
  12:20 and 12:22, so no stagnation is present.
- All 155 production JSON files reopen successfully.  Throughput remains stable
  at roughly 42 controls/hour; 205 controls remain and the current completion
  estimate stays near 17:15--19:00 CDT.
- Resources remain safe: load averages 5.52/5.23/5.38, system-wide memory free
  51%, zero throttled pages and 607 GiB free disk.
- The 12:00 CDT figure deadline is satisfied: the unchanged article package
  contains 17 ready figure files and a readable 30-page `main.pdf`.  No
  provisional result was promoted to the paper to meet the deadline.

## Hourly monitor — 2026-07-30 13:23 CDT

- N5B pinned controls have advanced to 198/360.  The four restart-safe chunks
  are at 48/90, 49/90, 52/90 and 49/90 and all wrote fresh checkpoints through
  13:22:56.  No stagnation or duplicate is present.
- All 198 production JSON files reopen successfully.  Throughput remains about
  43 controls/hour; 162 controls remain, giving a central completion estimate
  near 17:10 CDT with the conservative window unchanged at 17:15--19:00.
- Machine state remains acceptable under the four CPU workers: load averages
  5.44/5.85/5.93, system-wide memory free 47%, zero throttled pages and
  607 GiB free disk.
- The article package remains unchanged and readable at 30 pages.  No interim
  pinned-control trend is being interpreted before the complete gate.

## Hourly monitor — 2026-07-30 14:24 CDT

- N5B pinned controls have advanced to 237/360.  Chunk counts are
  59/90, 58/90, 62/90 and 58/90, with fresh checkpoints through 14:22.
  All groups progressed during the hour; there is no stagnation or duplicate.
- All 237 production JSON files reopen successfully.  At the current aggregate
  rate, 123 controls remain and completion is expected around
  17:15--18:15 CDT.
- Resources remain safe: load averages 5.20/5.32/5.36, system-wide memory free
  49%, zero throttled pages and 607 GiB free disk.
- The article PDF remains intact and readable at 30 pages.  Scientific
  aggregation of the pinned controls remains deferred until the complete grid
  is present.

## Hourly monitor — 2026-07-30 15:24 CDT

- N5B pinned controls have advanced to 277/360.  Chunk counts are
  69/90, 68/90, 72/90 and 68/90, with new checkpoints through 15:22.
  All groups continue to progress and no stagnation or duplication is present.
- All 277 production JSON files reopen successfully.  Eighty-three controls
  remain; at the measured rate the central completion estimate is
  17:20--17:40 CDT.
- Machine state is safe: load averages 6.60/5.40/5.30, system-wide memory free
  50%, zero throttled pages and 607 GiB free disk.
- The article PDF remains unchanged and readable at 30 pages.  Final N5B
  aggregation will start only after the complete 360-control grid is verified.

## On-demand status — 2026-07-30 15:41 CDT

- The only active scientific campaign is the N5B pinned-front control.  It is
  at 289/360 outputs, split 72/90, 71/90, 75/90 and 71/90 across the four
  restart-safe chunks.  All four groups wrote fresh checkpoints through 15:40,
  so no task is stagnant.
- Seventy-one controls remain.  At the measured aggregate throughput, raw
  completion is expected near 17:20--17:40 CDT, followed by integrity checks
  and the N5B scientific aggregation/gate.
- The original `NEXT24H/state.json` remains marked failed because its first
  orchestrator exited during the recovered N5B incident.  It is not the
  liveness authority; `runs/N5B/recovery_state.json` and the advancing
  checkpoints are authoritative.
- Resources remain safe: load averages 5.70/6.01/5.84, system-wide memory free
  52% and 607 GiB free disk.  No other heavy family is running.

## Scope decision and revised ETA — 2026-07-30 16:00 CDT

- The user explicitly removed the remaining N8 Lyapunov and multi-tour
  calculations from scope.  Existing N8 trajectories, audit tables and the
  qualified preliminary conclusion remain preserved; no new N8 simulation will
  be launched.
- N5B pinned controls are at 302/360, split 75/90, 74/90, 79/90 and 74/90.
  All four chunks wrote fresh checkpoints through 15:57, so there is no
  stagnation.  Fifty-eight controls remain, corresponding to about 1.5 h at the
  measured aggregate rate, followed by 15--30 min for integrity and the N5B
  gate.
- After N5B, only the corrected N2 static refresh, N2 fit/window gate and the
  final retained aggregations remain.  The N2 refresh reuses all completed
  dynamic trajectories.  Based on the previous full five-seed runtime, a
  provisional budget is 1--2 h for production plus 1--2 h for smoke/benchmark,
  analysis, gates and integration.
- The revised conservative total is **5--7 h from 16:00 CDT**, with a central
  completion near 22:00 CDT if no new defect or scientifically required
  follow-up appears.  The hourly automation prompt has been updated to forbid
  any new N8 run.

## Hourly monitor — 2026-07-30 16:24 CDT

- N5B pinned controls have advanced to 320/360.  The four chunks are at
  79/90, 79/90, 83/90 and 79/90 and all wrote fresh checkpoints through 16:22.
  No stagnation or duplicate is present.
- All 320 production JSON files reopen successfully.  Forty controls remain;
  at the measured rate raw completion remains expected near 17:20--17:35 CDT.
- Resources remain safe under the four CPU workers: load averages
  6.51/7.07/6.74, system-wide memory free 50%, zero throttled pages and
  607 GiB free disk.
- The article PDF remains unchanged and readable at 30 pages.  No N8 run is
  active or queued under the revised scope.

## N5B completion and positive-margin audit — 2026-07-30 17:30 CDT

- All four pinned-control chunks completed normally at 17:14--17:15 with
  return code zero.  The full N5B grid is complete and readable: five
  40-point memory scans plus 360/360 pinned controls.  A cache-only finalization
  assembled the combined summary without recomputation.
- The five memory-connected scans retain the previously reported negative
  complex margins at every point and show no candidate crossing.  However, the
  broader pinned-front control contains **12 positive complex margins**:
  five cells for seed 52 and seven for seed 58.  The other 18 seeds have no
  positive pinned-control margin.
- These are candidate complex instabilities, not yet Hopf classifications.
  All 12 stored M=64 roots pass the residual gate, but the production pinned
  grid did not include an independent resolution or nonlinear perturbation.
  Therefore the negative N5B gate cannot yet be accepted and no Hopf claim is
  made.
- A dedicated resolution-audit script was added and benchmarked on the first
  positive cell, seed 52 at tau=50 and relative distance 2e-4.  The root is
  positive and identical modulo conjugacy at M=48, 64 and 80:
  Re(z)=0.0006747199944, |Im(z)|=0.0476952391, with maximum inter-resolution
  difference 4.1e-17 and all residual gates passing.  This confirms that at
  least this candidate is not a pseudospectral-resolution artifact.
- The remaining 11 positive cells are now being audited restart-safely in
  execution session 49397.  The measured cost is about 97 s per cell for the
  M=48/80 pair, giving approximately 18 minutes.  N2 has not been started and
  will remain deferred until this N5B gate audit is complete.
- This is a scientifically significant new result requiring follow-up:
  resolution stability alone is insufficient to call Hopf.  Crossing
  refinement/transversality and eigenmode-aligned nonlinear tests are required
  before changing the manuscript or final conclusion.

## N5B resolution audit complete — 2026-07-30 18:27 CDT

- The resolution audit finished all 12 positive pinned-front cells normally.
  Every candidate remains positive at M=48, 64 and 80, every residual gate
  passes, and the largest conjugacy-aware inter-resolution difference is only
  7.84e-17.  The positive margins are therefore genuine roots of the discretized
  characteristic equation to the tested precision, not spectral-grid noise.
- The 12 cells belong to eight (seed, tau) slices: seed 52 at tau 50, 70 and
  100, and seed 58 at tau 20, 30, 50, 70 and 100.  In each slice the three
  lambda offsets contain a negative-to-positive complex-margin bracket.
- A restart-safe crossing-refinement script has been added.  It bisects each
  bracket in lambda to width 1e-6, tracks the same complex-root family, checks
  M=48/64/80 again, estimates transversality and verifies that the leading real
  mode remains stable.  The code compiles and the first full-resolution pilot
  is active in execution session 72272; no other heavy campaign overlaps it.
- N2 remains deferred.  Even if the refined spectral gate passes, a Hopf claim
  will require eigenmode-aligned nonlinear growth/frequency tests on both sides
  of the refined crossings.  No manuscript figure or wording has been changed.

## First N5B crossing refined — 2026-07-30 19:30 CDT

- The first full-resolution crossing refinement completed for seed 52,
  tau=50.  The complex root crosses within
  lambda=[0.3246630859, 0.3246636719], width 5.86e-7, at relative distance
  4.26465e-4 below the real fold.
- At the refined point the M=48/64/80 roots agree within 2.1e-17 and have
  Re(z)=2.02e-7, |Im(z)|=0.04935545.  The measured slope
  d Re(z)/d lambda is 1.732, while the leading real root remains safely
  negative at -0.02098.  This is a resolution-stable transverse complex-root
  preemption of the real fold for this pinned branch.
- This satisfies the spectral part of the Hopf gate for one crossing, but not
  the nonlinear part.  It is therefore still called a candidate Hopf
  bifurcation, not a resolved Hopf-born attractor.
- The pilot took 556 s.  A first attempt to parallelize through Python's
  `ProcessPoolExecutor` failed before launching workers because the sandbox
  denies the semaphore system query; no checkpoint was damaged.  The runner
  was corrected to accept disjoint explicit crossing indices.
- The seven remaining refinements are now running as four independent
  monothread sessions 9627, 13981, 93223 and 67145.  This preserves the
  four-process limit and is estimated to require about 20 minutes.  N2 remains
  deferred.

## Revised completion estimate — 2026-07-30 20:00 CDT

- The four refinement sessions remain active.  The first crossing required
  556 s in isolation; shared-resource contention makes the initial 20-minute
  four-way estimate optimistic, but the current load and memory state remain
  safe and this is not yet a stagnation.
- If all eight spectral crossings pass, the predeclared N5B gate requires
  48 nonlinear probes: two sides, three eigenmode amplitudes and eight
  crossings.  Their runtime has not yet been measured.  The first probe will be
  used as a benchmark before the remaining restart-safe batches are launched.
- Based on the already measured DDE throughput, a provisional budget is
  1--3 h for the N5B nonlinear probes and final gate, followed by 3--5 h for the
  corrected N2 refresh, fits and final integration.
- The revised total is therefore **5--9 h from 20:00 CDT**, corresponding
  approximately to 01:00--05:00 CDT on July 31.  This range will be tightened
  after the first nonlinear probe; a failed nonlinear gate will be reported
  rather than extended or forced.

## User-directed N5B stop and transition to N2 — 2026-07-30 20:25 CDT

- The user explicitly removed the N5B/Hopf-margin investigation from the
  manuscript and requested that the active N5B refinements stop.  The four
  unambiguous refinement sessions were interrupted individually; no broad
  process kill was used.
- All completed N5B scans, the 360/360 pinned controls, the 12-cell resolution
  audit and the completed seed-52/tau-50 crossing refinement remain preserved
  under `runs/N5B/`.  The other in-flight crossing refinements had not written
  final checkpoints when stopped.  No existing output was deleted or
  overwritten.
- No further N5B calculation or nonlinear Hopf probe will be launched.  These
  archived results are outside the current manuscript scope.
- The corrected N2 static refresh is now the sole heavy numerical priority.  It
  starts from the stable N1 terminal branch and reuses the already completed N2
  dynamic trajectories.  The source audit confirms that cached dynamic
  trajectory/summary files are read rather than reintegrated when present.
- The hourly automation was updated accordingly: N5B and new N8 calculations
  are forbidden; N2 validation, production, gate, final figure generation and
  PDF inspection are the remaining path.
- Machine state immediately after the stop was safe: load averages
  4.99/5.92/5.67, 48% system-wide memory free and 606 GiB free disk.  No heavy
  campaign was left running.

## Corrected N2 validation and production start — 2026-07-30 20:39 CDT

- The corrected source has SHA-256
  `45a8b9f61e34d366c7c070cd07ca1dc6e287d9fdb3445fac09659b6c5a571d26`.
  It compiles and all seven central tests pass.
- The explicit `--refresh-static` smoke test passed in 150.38 s.  Both smoke
  points stayed on the selected memory branch, converged with residuals below
  \(4\times10^{-16}\), had negative leading real roots, passed every spectral
  residual gate, and agreed between \(M=48\) and \(M=64\) to
  \(8.4\times10^{-17}\) or better.  The pre-existing dynamic trajectory
  modification times remained unchanged.
- The high-\(N\), seed-42 production benchmark completed in 401.92 s.  All ten
  corrected points have the required stable branch source and identity, all
  finite leading real roots are negative, and the three \(M=48/64\) controls
  agree within \(3.5\times10^{-17}\).  One \(M=48\) real-root observable is
  missing at \(\delta=0.002\); it is retained as missing rather than imputed and
  does not invalidate the other nine points.
- Based on the measured benchmark, the remaining seeds 47, 51, 52 and 60 were
  launched as four independent single-threaded sessions 74411, 27662, 25828
  and 75536.  Expected raw completion is approximately 10--20 minutes under
  shared-resource contention, followed by analysis, gate and figure
  generation.

## N2 complete, gated and integrated — 2026-07-30 21:18 CDT

- All five corrected N2 realizations (42, 47, 51, 52 and 60) are complete:
  50 stable-branch stationary points, 80 uncensored dynamic measurements and
  30 step-halving controls.  Four-way spectral execution was measured to reduce
  aggregate throughput, so the campaign was completed serially using the
  restart-safe point checkpoints and explicit one-thread library limits.
- A second numerical audit found that the generic reduced-IG candidate list can
  omit a purely real soft mode.  The production summaries were upgraded from
  the saved stationary states with the dedicated real-axis
  `sigmin(T_P)` search (`n=300`), without rerunning dynamics or recomputing the
  complex spectra.  All 50 real roots are negative and have scaled
  characteristic residual below \(10^{-10}\).
- Seed 51 is retained but excluded from the local dynamic-exponent aggregation
  by an independent threshold-centering criterion: its entire measured dynamic
  onset bracket lies below the exact static fold by more than the smallest N2
  offsets.  Seeds 42, 47, 52 and 60 are used for that aggregation.
- The full-window disorder-mean dynamic free exponent is 0.477 with nested
  bootstrap 95% interval [0.424, 0.529].  On
  \(\delta\le5\times10^{-4}\) it is 0.470 [0.435, 0.504].  The logarithmic
  alternative is rejected by at least \(\Delta\mathrm{AICc}=35.1\), and every
  step-halving control agrees at the reported passage-time resolution.
- The strict measured-\(1/2\) gate **fails**: individual fixed-half fits are
  disfavored by as much as \(\Delta\mathrm{AICc}=24.5\), and the static free
  exponent is sample- and window-dependent (mean 0.654 through \(10^{-4}\)
  versus 0.474 through \(5\times10^{-4}\)).  The allowed conclusion is
  **compatible with a saddle-node square-root law**, not a universal measured
  exponent exactly equal to one half.
- Outputs:
  `reports/N2_GATE_STATUS.md`, `reports/N2_summary.json`,
  `reports/N2_fit_windows.csv`, `reports/N2_timestep_controls.csv`,
  `figures/N2_critical_laws.pdf` and
  `manifests/N2_figure_provenance.json`.
- The final panel was copied to
  `paper_v5/figures_ready/Figure4b_N2_critical_laws.pdf` and integrated into the
  manuscript.  All manuscript discussion and supplementary figures about the
  N5B/Hopf-margin direction were removed at the user's request.
- `paper_v5/main.pdf` compiles successfully with `latexmk`: 29 A4 pages,
  no undefined references or overfull boxes.  Pages 11--16, 23 and 26 were
  rendered and visually inspected; the new N2 figure, captions and section
  transitions are legible and unclipped.
- No heavy numerical process remains active.  N2 production, its gate and final
  figure/PDF generation are complete.
