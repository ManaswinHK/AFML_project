# v4 progress log (next steps after v3)

Branch `v4-next-steps` (from master 60261bd). Plan: merge into master when done (user's instruction).

## Parts
- [x] C (frozen fe679a6): expected-value credit vs realised difference rewards, non-shared IPPO, K = 4, 8, seeds 10-14. Done: C1 not supported, C2, C3 supported (`docs/v4/results.md`).
- [x] A (frozen 21285a0): mixed-size set critic. Done: A1-A5 supported.
- [ ] B: Level-Based Foraging. Design phase: encoding fix (E6), heavy food + removal null (E7), per-food critic pilot (E9) -> pre-register -> confirm
- [ ] D (frozen fae1df9): random start cells. RUNNING `results/v4/confirm_rs`
- [ ] Analysis, write-up (`docs/v4/report.md`), PR, merge

## Notes
- 2026-11-12 23:25: started. lbforaging 2.0.0 installed in .venv.
- 2026-11-13 07:00: A and C confirmatory runs finished and analysed.
- To resume: `.venv/bin/python scripts/v3/run_cells.py --grid confirm_rs --workers 3 --out results/v4/confirm_rs` (skips finished cells); `.venv/bin/python scripts/v4/analyse_v4.py`.
