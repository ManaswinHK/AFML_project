"""Fill the course slide template (AFML-ppt-template.pptx) with the project content.

Usage:  python make_template_deck.py <template.pptx>
Writes: submission/AFML_Final_Presentation.pptx

Keeps every template slide, header, table, logo and footer; removes the template's
example images; adds a key-ideas slide and three results slides.
Speaker notes hold the timed script for the <= 7 minute video (also in submission/video_script.md).
"""
import copy
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
import os as _os
OUT = Path(_os.environ.get("DECK_OUT", ROOT / "submission" / "AFML_Final_Presentation_team_04.pptx"))
NAVY = RGBColor(0x1F, 0x2A, 0x6B)
LOGO = "Picture 12"


# ---------------------------------------------------------------- helpers
def shape(slide, name):
    return next(s for s in slide.shapes if s.name == name)


def drop_example_pictures(slide):
    for s in list(slide.shapes):
        if s.shape_type == 13 and s.name != LOGO:
            s._element.getparent().remove(s._element)


def bullets(sh, items, size=18):
    """Replace a text box's paragraphs with `items`, reusing the first paragraph's formatting.
    An item is a string or (bold_lead, rest)."""
    tf = sh.text_frame
    proto = copy.deepcopy(tf.paragraphs[0]._p)
    for p in list(tf.paragraphs):
        p._p.getparent().remove(p._p)
    for it in items:
        p_el = copy.deepcopy(proto)
        for r in p_el.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}r"):
            p_el.remove(r)
        tf._txBody.append(p_el)
        p = tf.paragraphs[-1]
        lead, rest = (it if isinstance(it, tuple) else ("", it))
        if lead:
            r = p.add_run()
            r.text = lead
            r.font.bold = True
            r.font.size = Pt(size)
        r = p.add_run()
        r.text = rest
        r.font.bold = False
        r.font.size = Pt(size)
        p.space_after = Pt(6)
        p.alignment = PP_ALIGN.LEFT


def cell(c, text, size=12, bold=False):
    tf = c.text_frame
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    p = tf.paragraphs[0]
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold


def fill_table(slide, rows, size=12, start=1):
    t = next(s for s in slide.shapes if s.has_table).table
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell(t.cell(start + i, j), val, size=size)
    # blank any unused body rows' text (keep the grid)
    return t


def rich(t, row, col, head, detail, label="", hs=11, ds=10):
    """A cell with a bold headline and a plain-language second line underneath."""
    tf = t.cell(row, col).text_frame
    cell(t.cell(row, col), head, size=hs, bold=True)
    q = tf.add_paragraph()
    q.space_before = Pt(2)
    if label:
        r = q.add_run(); r.text = label + " "; r.font.size = Pt(ds); r.font.bold = True
    r = q.add_run(); r.text = detail; r.font.size = Pt(ds)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def picture(slide, path, left, top, width=None, height=None):
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top),
                                    Inches(width) if width else None,
                                    Inches(height) if height else None)


def duplicate(prs, src):
    """Copy a slide (shapes + image relationships) to the end of the deck."""
    new = prs.slides.add_slide(src.slide_layout)
    for s in list(new.shapes):
        s._element.getparent().remove(s._element)
    rid_map = {}
    for rid, rel in src.part.rels.items():
        if "image" in rel.reltype:
            rid_map[rid] = new.part.relate_to(rel._target, rel.reltype)
    for s in src.shapes:
        el = copy.deepcopy(s._element)
        for node in el.iter():
            for k, v in list(node.attrib.items()):
                if k.endswith("}embed") and v in rid_map:
                    node.set(k, rid_map[v])
        new.shapes._spTree.append(el)
    return new


def move(prs, slide, index):
    lst = prs.slides._sldIdLst
    el = next(e for e in lst if prs.slides.part.related_part(e.rId) is slide.part)
    lst.remove(el)
    lst.insert(index, el)


# ---------------------------------------------------------------- timing
import re

MARK = re.compile(r"\[\d+:\d\d – \d+:\d\d\]")


def retime(prs, wps=145 / 60, demo_s=60):
    """Rewrite every '[m:ss – m:ss]' marker in the notes from cumulative word counts."""
    t = 0.0
    fmt = lambda x: f"{int(x // 60)}:{int(round(x % 60)):02d}" if round(x % 60) < 60 else f"{int(x // 60) + 1}:00"
    for slide in prs.slides:
        tf = slide.notes_slide.notes_text_frame
        text = tf.text
        parts = MARK.split(text)
        if len(parts) < 2:
            continue
        out = parts[0]
        for seg in parts[1:]:
            dur = demo_s if "DEMO" in seg else 20 if "Each member" in seg else len(seg.split()) / wps
            out += f"[{fmt(t)} – {fmt(t + dur)}]" + seg
            t += dur
        tf.text = out


# ---------------------------------------------------------------- content
B = ROOT / "submission" / "build"


def set_title(slide, text):
    p = shape(slide, "Text Box 34").text_frame.paragraphs[0]
    p.runs[0].text = text
    for r in p.runs[1:]:
        r.text = ""


def build(template):
    prs = Presentation(template)
    S = list(prs.slides)
    for s in S[1:6]:
        drop_example_pictures(s)

    # added slides (copied before the originals are edited)
    gloss = duplicate(prs, S[2])
    extra1, extra2, extra4, extra3 = (duplicate(prs, S[5]) for _ in range(4))
    move(prs, gloss, 2)
    move(prs, extra1, 7)
    move(prs, extra2, 8)
    move(prs, extra4, 9)
    move(prs, extra3, 10)

    # 1 ---- title
    tb = shape(S[0], "Google Shape;26;p3").text_frame
    members = [
        "PES1UG24AM356 – Daruru Bhuvan Sai",
        "PES1UG24AM108 – H K Manaswin",
        "PES1UG24AM118 – Harshitha B T",
        "PES1UG24AM117 – Harshith R",
    ]
    k = 0
    for p in tb.paragraphs:
        t = p.text.strip()
        if t.startswith("Project Title"):
            r = p.add_run()
            r.text = "  Who Earned the Reward? Checking whether teams of learning agents give credit " \
                     "to the agents who actually earned a shared reward"
            r.font.bold = False
        elif t.startswith("Team member") and k < 4:
            for r in list(p.runs):
                r._r.getparent().remove(r._r)
            r = p.add_run()
            r.text = f"Team member {k + 1} SRN and name:  {members[k]}"
            k += 1
    notes(S[0], "[0:00 – 0:10]  Hi, we are Team Aporia: Bhuvan, Manaswin, Harshitha and Harshith. Our project: "
                "Who Earned the Reward?")

    # 2 ---- topic: the basics and the problem
    bullets(shape(S[1], "Content Placeholder 2"), [
        ("Reinforcement learning: ", "an agent learns by trial and error. It acts, gets points (a reward), and does more of what earned points."),
        ("Many agents, one reward: ", "like a group project with one grade. Usually only some members did the work."),
        ("Credit assignment: ", "deciding who earned the reward, so each agent learns the right lesson."),
        ("Our question: ", "do popular methods give the credit to the agents who actually earned it?"),
    ], size=14)
    sh = shape(S[1], "Content Placeholder 2")
    sh.width, sh.height = Inches(10.6), Inches(2.0)
    picture(S[1], B / "fig_background.png", 2.1, 3.6, width=7.8)
    notes(S[1], "[0:10 – 0:40]  In reinforcement learning, an agent learns by trial and error: it acts, gets points, "
                "and does more of what earned points. A team of agents often shares one reward, like a group project with one "
                "grade, though only some members did the work. Working out who earned it is called credit assignment. Do "
                "popular methods get it right?")

    # 3 ---- key ideas in plain English (added)
    set_title(gloss, "Key ideas in plain English")
    bullets(shape(gloss, "Content Placeholder 2"), [
        ("Episode: ", "one play-through of the game, from start to finish."),
        ("Delayed reward: ", "the points arrive some time after the useful actions, so it is hard to tell who caused them."),
        ("Idle agents: ", "agents that move around but help with nothing. They should get zero credit."),
        ("Critic: ", "a neural network that watches the game and predicts how many points the team will end up with. "
                     "Most methods work out each agent's credit from it."),
        ("Equal split: ", "give everyone the same share. It needs no learning, so a method should at least beat it."),
        ("Credit error: ", "how far a method's split is from the true split. 0 = exactly right, 2 = completely wrong."),
    ], size=15)
    notes(gloss, "[0:40 – 1:05]  Some words we'll use. Idle agents help with nothing, so they should get zero credit. A "
                 "critic is a neural network that predicts how many points the team will get. The equal split is the "
                 "baseline to beat, and credit error measures how far a method is from the truth: zero is perfect.")

    # 4 ---- uniqueness
    bullets(shape(S[2], "Content Placeholder 2"), [
        ("Nobody checks the credit. ", "Methods are judged only by the team's score, because real tasks have no answer key."),
        ("We build the answer key: ", "games where we know exactly who earned what."),
        ("We test six popular methods against it ", "(VDN, QMIX, COMA, MAPPO, NA²Q, STAS), and they mostly fail."),
        ("We fix it: ", "one change to the training data makes the credit almost exactly right."),
    ], size=18)
    notes(S[2], "[1:05 – 1:20]  Methods are judged only by team score, because real tasks have no answer key. So we "
                "built games with one, tested six popular methods, and found a fix.")

    # 5 ---- dataset
    bullets(shape(S[3], "Content Placeholder 2"), [
        ("No downloaded dataset: ", "in reinforcement learning the data comes from playing a game (a simulator), which our code does."),
        ("Our game (Planted Coalition Gridworld): ", "3–7 agents on a grid; reward delayed by 0–50 steps; 0–4 idle agents."),
        ("A standard game (Level-Based Foraging): ", "public benchmark where agents team up to pick up food; new random layout every game."),
        ("Why not a fixed dataset? ", "To know the truth we must re-play each game with agents taken out, which recorded data can't do."),
        ("Size: ", "120 test settings × 5 runs in the main experiment; about 2,500 result files, all in the repo."),
    ], size=16)
    notes(S[3], "[1:20 – 1:30]  There's no downloaded dataset: the data comes from simulated games, because finding "
                "the truth means re-playing games with agents taken out.")

    # 6 ---- design
    bullets(shape(S[4], "Content Placeholder 2"), [""], size=4)
    picture(S[4], B / "fig_plain_overview.png", 1.6, 1.95, width=9.6)
    notes(S[4], "[1:30 – 2:00]  Our example: A pulls a lever that opens a gate, later B and C push a crate together, "
                "and the points come even later. D and E just wander. So A, B and C each earned a third. True credit comes "
                "from re-playing each game with agents taken out: it is how much reward is lost without you.")

    # 7

    # 7

    # 7 ---- final results (1)
    bullets(shape(S[5], "Content Placeholder 2"), [
        ("Only 5 of 96 ", "method-and-setting combinations beat simply splitting the reward equally."),
        ("", "Generally, the later the reward arrives and the more idle agents there are, the worse the credit gets."),
    ], size=15)
    shape(S[5], "Content Placeholder 2").height = Inches(1.3)
    picture(S[5], B / "fig_results_plain.png", 1.9, 3.3, width=9.4)
    notes(S[5], "[2:45 – 3:10]  So how do the popular methods do? Up means more wrong. Left to right, the reward arrives "
                "later. The dashed line is the equal split. With no idle agents it is exactly right, at zero, yet every "
                "method is well above it. Overall only 5 of 96 combinations "
                "beat just splitting the reward equally, and it gets worse with delay and with more idle agents.")

    # 8 ---- why they fail: the wrong question
    set_title(extra1, "Why existing methods fail: they ask the wrong question")
    bullets(shape(extra1, "Content Placeholder 2"), [
        ("Also blind to teamwork: ", "VDN, QMIX and NA²Q put 0.00 of the value on groups (truth: 1.0). "
                                     "Our first method, a coalition estimator trained on replays with agents removed, finds the groups (0.78–0.97)."),
    ], size=13)
    sh = shape(extra1, "Content Placeholder 2")
    sh.top, sh.height = Inches(5.95), Inches(0.9)
    picture(extra1, B / "fig_two_questions.png", 2.0, 1.95, width=9.0)
    notes(extra1, "[3:10 – 3:45]  Why do they fail? They ask the wrong question. Their credit means: what if A paused for one "
                  "step? A opened the gate early, so it would just open it a step later. Nothing is lost, so A gets almost no "
                  "credit. The question that matters is: what if A weren't there at all? Then the gate stays shut and all ten "
                  "points are lost, so A earned a third. They are also blind to teamwork; our first method, a coalition "
                  "estimator, does find the groups.")

    # 9 ---- our fix: how it works
    set_title(extra2, "Our fix: credit by intervention")
    picture(extra2, B / "fig_how_it_works.png", 1.0, 1.95, width=11.0)
    bullets(shape(extra2, "Content Placeholder 2"), [
        ("Like a coach benching a player in practice ", "to see how the team does without them."),
        ("Both pieces are needed: ", "trained without sitting out, the same critic fails (error 0.85–1.50); "
                                     "MAPPO trained on the same sitting-out games but asking its one-step question still fails (2.00)."),
    ], size=13)
    sh = shape(extra2, "Content Placeholder 2")
    sh.top, sh.height = Inches(5.2), Inches(1.6)
    notes(extra2, "[3:45 – 4:35]  So we make the methods ask the right question, and give them the data to answer it. Like a "
                  "coach benching a player in practice: while collecting training games, a random agent or group sometimes sits "
                  "out the rest of the game. A neural network, the critic, learns how many points the team will still get, "
                  "given who is still playing. Then an agent's credit is simply points with you minus points without you. "
                  "Why does this fix A? Before A pulls the lever, if A stopped the team would get nothing, so the team "
                  "needs A completely. The moment A pulls it, stopping A no longer matters. That change is A's credit, "
                  "whether or not A had slack. "
                  "Both pieces matter: without the sitting-out games the same critic fails, and an existing method given the "
                  "same games but asking its old one-step question still fails.")

    # 10 ---- our fix: results
    set_title(extra4, "Our fix gets the credit right")
    bullets(shape(extra4, "Content Placeholder 2"), [
        ("Left: ", "in a real test game, ours gives A its full third at the exact step A pulls the lever; QMIX never does. "),
        ("Right: ", "error below 0.07 in every setting, and the right team every time; existing methods 0.5–2.0."),
    ], size=13)
    sh = shape(extra4, "Content Placeholder 2")
    sh.height = Inches(0.95)
    picture(extra4, B / "fig_credit_over_time.png", 0.6, 2.95, width=6.0)
    picture(extra4, B / "fig_fix.png", 6.65, 3.0, width=6.1)
    notes(extra4, "[4:35 – 5:10]  Does it work? On the left, one real test game. The blue line is the credit our method gives A "
                  "over time. It jumps to exactly a third at the step A pulls the lever, because after that the team no longer "
                  "needs A. QMIX, in orange, never gets close. On the right, across all twelve settings: every existing method "
                  "has an error between 0.5 and 2. Ours is below 0.07, and finds the right team every time.\n"
                  "[5:10 – 6:10]  DEMO, 60 s (open docs/credit_viz.html; choose delay 25, enabler slack, test episode #1). "
                  "1) Method QMIX, panel 'Who got the credit': say 'grey is the truth; QMIX gives D and E credit and A very "
                  "little'. 2) Click 'Ours: trained with sitting out': 'our bars sit on the truth: a third each for A, B, C, "
                  "nothing for D and E'. 3) Per-step heatmap: 'and A's credit lands exactly when it pulls the lever'. 4) Press "
                  "Play on the episode replay to show the agents moving. 5) Click 'Ours: coalition estimator' and point at the "
                  "coalition panel: 'it finds the group A, B, C'. 6) Table at the bottom: 'on all 30 test games, ours has the "
                  "lowest error'.")

    # 10 ---- generalises
    set_title(extra3, "Our fix generalises, with limits")
    bullets(shape(extra3, "Content Placeholder 2"), [
        ("New starting positions every game: ", "a critic that treats agents alike stays near exact (0.02–0.03, vs 0.52–0.96 for a plain network)."),
        ("Team sizes it never trained on: ", "trained on a few team sizes, it picks exactly the right team at new sizes (error 0.01–0.14)."),
        ("Standard foraging game (new random map each game): ", "with 4 agents, much better than an equal split; with 6, better but far from exact."),
    ], size=14)
    shape(extra3, "Content Placeholder 2").height = Inches(2.1)
    picture(extra3, B / "fig_lbf_plain.png", 2.2, 4.0, width=7.6)
    notes(extra3, "[4:50 – 5:15]  It also generalises: near exact when starting positions change every game, and the right "
                  "team at team sizes it never trained on. In a standard foraging game it clearly beats the equal split with "
                  "four agents; with six it is better, but far from exact.")

    # 11 ---- done vs remaining: each row says what it was and what it showed
    done = [
        ("A game with a known answer",
         "Enabler A, finishers B and C, idle D and E. True credit: re-play each game with agents removed.", "Done"),
        ("Test of four popular methods: VDN, QMIX, COMA, MAPPO",
         "120 settings × 5 runs. Only 5 of 96 combinations beat an equal split; worse with longer delay and more idle agents.", "Done"),
        ("Teamwork-level methods: NA²Q and STAS rebuilt from their papers; our first (coalition) estimator",
         "NA²Q finds no teamwork; STAS splits about equally. Ours finds 96% of the groups, but needs re-plays with agents removed.", "Done"),
        ("Agents learning from scratch, with credit checked during training",
         "Team score is a weak guide to credit while learning. COMA and MAPPO never learned the game.", "Done (COMA, MAPPO failed)"),
        ("Our fix: credit by intervention (training with agents sitting out)",
         "Error 0.002–0.07 vs 0.5–2 for existing methods; right team in all 12 settings. Drop either piece and it fails.", "Done"),
        ("Generalisation tests and an external review",
         "Near exact with new start positions and unseen team sizes (up to 8 agents); beats equal split on foraging.", "Done"),
        ("Bigger standard games, and settings where agents can't sit out",
         "StarCraft-style tasks, and real systems where taking an agent out in practice is unsafe.", "To be done"),
    ]
    t = fill_table(S[6], [(str(k + 1), "", st) for k, (_, _, st) in enumerate(done)], size=11)
    for k, (head, detail, _) in enumerate(done):
        rich(t, k + 1, 1, head, detail, "Showed:" if k < 6 else "Why:", hs=11, ds=9.5)
    notes(S[6], "[5:45 – 5:52]  Everything planned is done; bigger games and settings where agents can't sit out are next.")

    # 12 ---- contributions: what each member did, then what they investigated
    t = fill_table(S[7], [
        ("Daruru Bhuvan Sai", "PES1UG24AM356", ""),
        ("H K Manaswin", "PES1UG24AM108", ""),
        ("Harshitha B T", "PES1UG24AM118", ""),
        ("Harshith R", "PES1UG24AM117", ""),
    ], size=12)
    for i, (did, studied) in enumerate([
        ("Designed and implemented the intervention-based credit assignment methodology. Developed counterfactual "
         "agent-removal experiments and learned credit-estimation approaches to identify individual contributions "
         "to delayed team rewards.",
         "how delayed rewards affect the attribution of credit to agents, particularly early enablers whose actions "
         "contribute to rewards received much later; also how intervention-based methods improve credit estimation."),
        ("Formulated the research problem and developed the foundational simulation environment. Implemented "
         "ground-truth credit computation using Shapley values and Harsanyi dividends, designed evaluation metrics, "
         "integrated existing MARL algorithms and developed a low-order coalition credit estimation approach.",
         "how rewards can be decomposed into individual and subgroup contributions, how ground-truth credit can be "
         "established, and whether coalition-based estimation can capture interactions between cooperating agents."),
        ("Evaluated coalition-level credit assignment across different cooperative structures, including overlapping "
         "coalitions and partial contributions. Analysed the performance of coalition estimation approaches against "
         "the established ground truth.",
         "whether the correct contributing groups can be identified when multiple agents cooperate, their "
         "contributions overlap, or only a subset of a coalition contributes; coalition recovery and group-level "
         "credit accuracy."),
        ("Conducted comparative experiments using existing MARL algorithms, including VDN, QMIX, COMA and MAPPO. "
         "Evaluated their individual-agent credit assignment performance under different experimental configurations.",
         "how accurately different algorithms assign credit to individual agents, particularly with delayed rewards "
         "and non-contributing agents; credit error, reward leakage and missed contributions."),
    ]):
        tf = t.cell(1 + i, 2).text_frame
        r = tf.paragraphs[0].runs[0]
        r.text = did
        r.font.size = Pt(10.5)
        q = tf.add_paragraph()
        q.space_before = Pt(2)
        for txt, bold in [("Investigated: ", True), (studied, False)]:
            rr = q.add_run()
            rr.text = txt
            rr.font.size = Pt(10)
            rr.font.bold = bold
            rr.font.italic = not bold
    notes(S[7], "[5:52 – 6:12]  Our contributions. [Each member: one short sentence.]")

    # 13 ---- quantity and quality
    fill_table(S[8], [
        ("1", "Our game + scripted team", "100", "Y", "–"),
        ("2", "True credit by re-playing without agents", "100", "Y", "–"),
        ("3", "VDN, QMIX, COMA, MAPPO (via EPyMARL)", "100", "Y", "MAPPO's standard credit is unusable; a corrected version is used"),
        ("4", "NA²Q and STAS rebuilt from the papers", "100", "Y", "One slow STAS test skipped in quick runs"),
        ("5", "Our first (teamwork) estimator", "100", "Y", "Needs re-plays with agents removed"),
        ("6", "Learning from scratch with credit checks", "100", "Y", "COMA, MAPPO fail to learn"),
        ("7", "Our fix: training with agents sitting out", "100", "Y", "–"),
        ("8", "Standard foraging game (checked against the official one)", "100", "Y", "–"),
        ("9", "Analysis, figures, viewer; 22 test files, ~15k lines", "100", "Y", "Unzip stored results before analysis"),
    ], size=12)
    notes(S[8], "[6:12 – 6:17]  Every part runs end to end, with tests.")

    # 14 ---- learnings: the lesson, then why we believe it
    learn = [
        ("A high team score does not mean credit is fair.",
         "A team can score perfectly while idle agents get credit. We could only see this because our game has an answer key; normal benchmarks only report the score."),
        ("The question a method asks decides what it can learn.",
         "“What if A paused for one step?” gives the enabler almost nothing, because A can act a step later. Even a perfect one-step critic is off by 0.9–1.2. Fair credit needs “what if A weren’t there?”."),
        ("The right data beats a fancier model.",
         "A critic can only learn “points without you” if it has seen games where someone was missing. Without the sitting-out games the same critic fails; a 6× larger critic only made a good result slightly better."),
        ("Credit should land when the outcome is decided, not when the reward arrives.",
         "With delayed rewards, A's credit belongs to the moment it pulls the lever, long before the points arrive. Our method puts it there; one-step methods give the early enabler almost nothing."),
        ("Write predictions down first and report what fails.",
         "We wrote our expected results before each experiment. An outside review caught claims that did not hold up (e.g. a shuffled-roles result, and “credit for free” from dropout), and we corrected them."),
    ]
    t = fill_table(S[9], [(str(k + 1), "") for k in range(len(learn))], size=13)
    for k, (head, detail) in enumerate(learn):
        rich(t, k + 1, 1, head, detail, hs=13, ds=11)
    notes(S[9], "[6:17 – 6:27]  Our main lessons: a score can't show fair credit, ask the right question, and the right data beats a fancier model.")

    # 15 ---- challenges: the problem, then why it is hard / what would help
    chal = [
        ("Only small games so far (up to 8 agents)",
         "The true credit needs one extra re-play per agent removed, and more for groups, so checking it gets expensive in big tasks like StarCraft.",
         "Scope"),
        ("Our fix needs agents that can sit out during training",
         "Easy in a simulator. Real robots, traffic or trading systems may not allow it; there, natural absences in logged data or a simulator would be needed.",
         "Data"),
        ("COMA and MAPPO failed to learn from scratch",
         "So their credit was checked only on a scripted team. Sitting-out training repaired MAPPO's critic (error 1.7 → 0.45–0.74) but still did not beat the equal split.",
         "Implementation"),
        ("Luck in a single game can't be credited",
         "When chance decides the outcome, a critic that predicts averages cannot say who was lucky this time. It learns average credit; exact per-game credit needs re-plays.",
         "Design"),
        ("Standard foraging game with 6 agents: better than an equal split, but far from exact",
         "More agents means many more sit-out combinations to learn from, so the critic likely needs more data or a better design.",
         "Design"),
    ]
    t = fill_table(S[10], [(str(k + 1), "", ty) for k, (_, _, ty) in enumerate(chal)], size=13)
    for k, (head, detail, _) in enumerate(chal):
        rich(t, k + 1, 1, head, detail, hs=12, ds=10.5)
    notes(S[10], "[6:27 – 6:37]  Open problems: bigger games, settings where agents can't sit out, and luck in a single game.")

    # 16 ---- references
    fill_table(S[11], [
        ("1", "Value-decomposition networks for cooperative multi-agent learning (VDN)", "Sunehag, Lever, Gruslys, et al.", "2018"),
        ("2", "QMIX: Monotonic value function factorisation for deep MARL", "Rashid, Samvelyan, Schroeder de Witt, Farquhar, Foerster, Whiteson", "2018"),
        ("3", "Counterfactual multi-agent policy gradients (COMA)", "Foerster, Farquhar, Afouras, Nardelli, Whiteson", "2018"),
        ("4", "The surprising effectiveness of PPO in cooperative multi-agent games (MAPPO)", "Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu", "2022"),
        ("5", "NA²Q: Neural attention additive model for interpretable multi-agent Q-learning", "Liu, Zhu, Chen", "2023"),
        ("6", "STAS: Spatial-temporal return decomposition for sparse rewards in MARL", "Chen, Zhang, Yang, Du", "2024"),
        ("7", "Benchmarking multi-agent deep RL algorithms in cooperative tasks (EPyMARL, LBF)", "Papoudakis, Christianos, Schäfer, Albrecht", "2021"),
    ], size=12)
    notes(S[11], "[6:37 – 6:42]  These are our references. Thank you.")

    retime(prs)
    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "Downloads" / "AFML-ppt-template.pptx"))
