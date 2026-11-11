# v3 progress log (intervention credit, Phases 0–5)

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
- [ ] R. Research: read the 4 uploaded papers + SHAQ, CSCA (Li/Lee/Bareinboim), Exact-Is-Easier, CXAI, STAS; write `docs/v3/related_work.md`
- [ ] R2. Read repo code in full; write `docs/v3/design_notes.md` (what changes where, and the novel idea)
- [ ] 0.x Phase 0 items (see plan §8)
- [ ] 1.1 Pre-registration v3 committed before any confirmatory run
- [ ] 1.x Phase 1
- [ ] 2.x Phase 2
- [ ] 3.x Phase 3 gate
- [ ] 4.x Phase 4 (reduced)
- [ ] 5. Write-up

## Session notes
- 2026-11-11 (session 1): cloned, venv built, branch created.
