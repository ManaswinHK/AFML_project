# v3 progress log (intervention credit, Phases 0–5)

**ALL PHASES DONE (12 Nov 2026, 07:45).** Start with `docs/v3/README.md` and `docs/v3/report.md`. PR: ManaswinHK/AFML_project#2.

Branch: `intervention-credit-v3` on `ManaswinHK/AFML_project` (never push master). One PR at the end of each phase.
Plan source: `docs/assessment_and_plan.md` (§8, §9). Deadline: about one week from 11 Nov 2026. Compute: this Mac only (M5, 10 cores, 16 GB).
Env: `.venv` (Python 3.12, torch CPU). Run tests: `.venv/bin/python -m pytest -q`.

## Rules for whoever resumes
1. Read this file, then `git log --oneline -15`, then the last "Session notes" entry.
2. Take the first unchecked item. Never redo a checked item.
3. Update `.progress/heartbeat` (`date +%s > .progress/heartbeat`) at least every 20 minutes.
4. Commit after each item; push the branch after each phase or every ~2 hours.
5. Every result goes in a doc with: what was run, seeds, numbers, and whether it was pre-registered.

## Checklist
- [x] R. Research: uploaded papers + SHAQ, CSCA, C3, Mesnard CCA, dropout (2304.12458), fault tolerance (2412.00534) -> `docs/v3/related_work.md`
- [x] R2. Repo code read; theory written -> `docs/v3/theory.md` (commitment split, pending-game step semivalues, identifiability)
- [x] 0.1 pending amount in state  - [x] 0.2 env.stop()/clone(), acting flags  - [x] 0.3 finisher feasibility
- [x] 0.4 table baseline in pipeline  - [x] 0.5 top-k (tie-aware)  - [x] 0.6 closed-loop truths in every cell
- [ ] 0.7 re-run stored partial cells (h3/h5 partial, d in {25,50}) with the fixed state
- [x] 0.8 errata -> `docs/v3/errata.md`
- [x] 1.2 exact truth in pipeline  - [x] 1.3 intervention data (stratified)  - [x] 1.4 atd registered; reproduction check passed (L1 0.006 vs 0.02, r 0.98 vs 0.96)
- [x] 2.1 shuffled roles  - [x] V2 control implemented (mappo/qmix _int, _int_masked)
- [x] 3.1 lanes layout  - [x] 3.2 feasibility (IPPO K=1: 98% test success by 120k steps)  - [x] 3.3 difference rewards via shadow copies
- [x] 4.1/4.2 code: dropout option in adapter; BranchableMacTeam (exact branching of learned policies); phase4 grid
- [x] Exploratory pilots (seeds 0-4) -> `docs/v3/pilot_findings.md` (pilot_fixed and pilot_budget stopped early on purpose)
- [x] Phase 3 pre-registration frozen (5a73c9c) + Addendum A (post-hoc, labelled); gate running
- [x] Mode P pre-registration frozen: commit a08b267 "Freeze pre-registration v3"
- [ ] RUNNING: `scripts/v3/run_confirm_all.sh` (log written to a local file); re-run it to resume (skips done cells)
- [ ] Confirmatory: confirm_main (60), confirm_h5 (20), confirm_perm (10), size transfer (seeds 5-9), gate (Phase 3, 60 runs), phase4 (6 runs)
- [ ] 0.7 re-run partial cells (rerun_partial_h3, rerun_partial_h5)
- [ ] Analysis: scripts/v3/analyse_prereg.py -> docs/v3/confirmatory_results.md; Phase 3/4 summaries
- [ ] Write-up: docs/v3/report.md + figures; PR
- [ ] 1.1 Pre-registration v3 committed before any confirmatory run
- [ ] 1.x Phase 1
- [ ] 2.x Phase 2
- [ ] 3.x Phase 3 gate
- [ ] 4.x Phase 4 (reduced)
- [ ] 5. Write-up

## Session notes
- 2026-11-11 (session 1): cloned, venv built, branch created. Phase 0 code + tests, ablation library, critics
  (MLP, ensemble, set), V2 control, shuffled roles, lanes + difference rewards. Pilots running.
  Early pilot signal (exploratory): set critic on shuffled roles L1 0.004 (b=2) / 0.006 (b=4) vs MLP 0.46 / 0.86.
  Partial layout: realised credit flips with exploration luck (1/3 episodes one finisher completes alone);
  structural read-out + expected-credit floor added. Active design: no reliable gain (exploratory negative).
- 2026-11-12 00:30 (session 1): pre-registration frozen (a08b267); confirmatory chain launched (WORKERS=6);
  gate (pre-registered) running, addendum grids gate_fine/gate_ns chained after it.
  If resuming: check `ls results/v3/confirm_*`, re-run `WORKERS=6 scripts/v3/run_confirm_all.sh`, then
  `.venv/bin/python scripts/v3/run_phase3.py --grid gate_fine --workers 3` and `--grid gate_ns`, then phase4
  (`--grid phase4 --workers 3`) and `scripts/v3/analyse_prereg.py`, `scripts/v3/analyse_phase3.py`.
- 2026-11-12 02:30: Phase 3 gate DONE (H3.1-H3.3 supported; addendum A1 agrees); gate_ns (A2, post-hoc) running.
  Phase 4 training running (2 workers) then eval. FOUND: pre-registered masked kappa read-out omits gamma^t
  and is swamped by critic calibration offsets (MAPPO standardised units); post-hoc structural read-out added
  (Phase 4 addendum; V2 post-hoc methods *_kdisc, *_struct; grid v2_posthoc queued last).
  Queue after confirm chain: double_k4 -> v1_coverage -> v2_posthoc (waiters use bracket regex patterns).
- 2026-11-12 07:45: ALL DONE. Confirmatory Mode P (60 + 20 + 10 cells + size transfer), Phase 3 gate + Addendum A1/A2,
  Phase 4, Phase 0.7 re-runs, order-4 check, V1, V2 post-hoc. 253 tests pass. Report final. PR #2 updated.
  Possible next steps (not started): mixed-size training for the set critic; Level-Based Foraging replication;
  learned critic + structural read-out as an online credit signal in non-shared IPPO (where Phase 3 shows headroom).
