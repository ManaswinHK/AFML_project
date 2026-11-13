# v4 progress log (next steps after v3)

**ALL PARTS DONE (13 Nov 2026).** Start with `docs/v4/README.md` and `docs/v4/report.md`.

Branch `v4-next-steps` (from master 60261bd), merged into master when done (the user's instruction).

## Parts
- [x] C (frozen fe679a6): expected-value credit vs realised difference rewards (non-shared IPPO, K = 4, 8). C1 not supported, C2, C3 supported; post-hoc drawdown analysis.
- [x] A (frozen 21285a0): mixed-size set critic. A1-A5 supported.
- [x] D (frozen fae1df9): random start cells. D2, D3 supported; D1 partial.
- [x] B (frozen e0198de): Level-Based Foraging, heavy food, removal null, two budgets. B1-B4 supported; B5, B6 not supported; 6 agents descriptive.
- [x] Analysis (`results.md`), figures, report, decision log (E1-E12), LBF design notes.
- [x] Full test suite: 274 passed, 1 deselected (`test_stas_learns_to_decompose_the_return`, see v3 D3).

## Notes
- 2026-11-12 23:25: started. lbforaging 2.0.0 installed in .venv.
- 2026-11-13 07:00: A and C confirmatory runs finished and analysed.
- 2026-11-13: LBF design phase (pilots on seeds 0-4, `lbf_design.md`); D and B frozen and run.
