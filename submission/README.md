# Final submission

| File | What it is |
|---|---|
| `Final_Report.docx` / `Final_Report.pdf` | 10-page report: abstract, introduction, background, related work, benchmark, results, discussion and limitations, conclusion, references |
| `Final_Presentation.pptx` / `Final_Presentation.pdf` | 15 main slides + 2 backup slides, with speaker notes on every slide (about 15–18 minutes) |
| `AFML_Final_Presentation_team_04.pptx` | **Submitted deck**: the course template (17 slides: 12 required + a key-ideas slide + 4 slides on the failure and our fix), written for a non-MARL audience, with a timed ~6.5-minute script including a 60-second demo in the speaker notes; built by `build/make_template_deck.py` |
| `recording_script.md` | Word-for-word video script, recorded in four takes, with on-screen cues for the demo |
| `presentation_guide.md` | Slide-by-slide explanation of the deck: what each slide means, its numbers, likely questions |
| `video_script.md` | Earlier single-take script, with a recording checklist |
| `build/` | Scripts that regenerate everything (`make_deck.js`, `make_report.py`, `make_overview_fig.py`) |

Every number comes from the result documents in `docs/` (`report.md`, `main_results.md`, `h3_results.md`, `h5_results.md`, `mode_t_results.md`, `sensitivity.md`). Nothing was re-run.

## How the deliverables are structured, and why

**Slides: one claim per slide, with the evidence under it.** Each title is a full sentence stating the slide's point ("Learned credit rarely beats simply splitting the reward equally"), not a topic label ("Results"). In Michael Alley's studies of this *assertion–evidence* style, audiences recalled and understood more than with topic-and-bullet slides. Text is kept short so the audience listens instead of reading.

**Talk: one running example, intuition before formalism.** Following Peyton Jones, Hughes and Launchbury's *How to give a good research talk*, the talk builds on a single concrete example (A pulls a lever, B and C push a crate, D and E stand around). Every definition and finding refers back to it. Details that matter only in Q&A are on the backup slides.

**Report: context, then content, then conclusion, at every level.** Following Mensh and Kording's *Ten simple rules for structuring papers* (PLOS Comp Biol 2017): one central message (credit read-outs and "who earned it" are different things), an abstract that tells the whole story, an introduction about the gap, results that justify each claim, and a discussion that states the limitations openly.

**Honest framing.** Failed predictions are shown in a scorecard with the reason for each failure. Analyses chosen after seeing the data are labelled post-hoc, as the project's own documents already do.

## Presenting

Slide numbers in the speaker notes match the deck. Suggested split for four speakers: 1–3 problem, 4–6 approach, 7–10 findings, 11–15 estimator and wrap-up. Backup slides 16–17 answer the likely "is it robust?" and "what went wrong?" questions.

## Rebuild

```bash
cd submission/build
npm install                    # pptxgenjs
../../.venv/Scripts/python make_overview_fig.py
../../.venv/Scripts/python make_report.py
NODE_PATH=./node_modules PPTX_SKILL=<path to the pptx skill> node make_deck.js
```
