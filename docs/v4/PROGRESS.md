# v4 progress log (next steps after v3)

Branch `v4-next-steps` (from master 60261bd). Plan: merge into master when done (user's instruction).

## Parts
- [x] C (frozen fe679a6): expected-value credit vs realised difference rewards, non-shared IPPO, K = 4, 8, seeds 10-14 - RUNNING (`results/v3/phase3/online_credit`)
- [ ] A: mixed-size set critic (pilot: interp done, extrap running) -> pre-register -> confirm (seeds 10-14)
- [ ] B: Level-Based Foraging replication (deterministic LBF tested against lbforaging 2.0.0) -> pilot -> pre-register -> confirm
- [ ] D: randomised start cells (deferred v3 item 2.5)
- [ ] Analysis, write-up (`docs/v4/report.md`), PR, merge

## Notes
- 2026-11-12 23:25: started. lbforaging 2.0.0 installed in .venv.
