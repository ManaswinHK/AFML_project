"""Builds submission/Final_Report.docx (convert to PDF with LibreOffice afterwards).
    ../../.venv/Scripts/python make_report.py
Every number comes from docs/ in this repository (report.md, main_results.md, h3_results.md,
h5_results.md, mode_t_results.md, sensitivity.md).
"""
import os
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(ROOT, "docs", "figures")
OUT = os.path.join(ROOT, "submission", "Final_Report.docx")

INK = RGBColor(0x1E, 0x22, 0x35)
TEAL = RGBColor(0x2E, 0x8C, 0x7E)
GREY = RGBColor(0x4A, 0x50, 0x68)

doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.2))
sec.top_margin = sec.bottom_margin = Cm(2.0)

st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.12
for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Title", 22)):
    h = doc.styles[name]
    h.font.name = "Cambria"
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.color.rgb = INK
    h.element.rPr.rFonts.set(qn("w:asciiTheme"), "") if False else None
    rf = h.element.rPr.rFonts
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), "Cambria")
doc.styles["Heading 1"].paragraph_format.space_before = Pt(14)
doc.styles["Heading 2"].paragraph_format.space_before = Pt(10)

# page numbers in footer
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    r = fp.add_run()
    if t:
        el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), t); r._r.append(el)
    else:
        el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt; r._r.append(el)
    r.font.size = Pt(9)


def runs(p, text):
    """**bold** and *italic* inline markup."""
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**"):
            p.add_run(part[2:-2]).bold = True
        elif part.startswith("*"):
            p.add_run(part[1:-1]).italic = True
        else:
            p.add_run(part)
    return p


def P(text, style=None, align=None):
    p = doc.add_paragraph(style=style)
    runs(p, text)
    if align:
        p.alignment = align
    return p


def B(items, numbered=False):
    for it in items:
        p = doc.add_paragraph(style="List Number" if numbered else "List Bullet")
        runs(p, it)
        p.paragraph_format.space_after = Pt(3)


def H1(t):
    doc.add_heading(t, level=1)


def H2(t):
    doc.add_heading(t, level=2)


fig_no = [0]
tab_no = [0]


def FIGURE(fname, caption, width=16.4, folder=FIG):
    fig_no[0] += 1
    doc.add_picture(os.path.join(folder, fname), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    r = p.add_run(f"Figure {fig_no[0]}. ")
    r.bold = True
    runs(p, caption)
    for r in p.runs:
        r.font.size = Pt(9.5)
        r.font.color.rgb = GREY
    p.paragraph_format.space_after = Pt(10)


def shade(cell, hex_):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hex_)
    tcPr.append(sh)


def TABLE(caption, rows, widths):
    tab_no[0] += 1
    p = doc.add_paragraph()
    r = p.add_run(f"Table {tab_no[0]}. ")
    r.bold = True
    runs(p, caption)
    for r in p.runs:
        r.font.size = Pt(9.5)
        r.font.color.rgb = GREY
    p.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Cm(widths[j])
            c.paragraphs[0].text = ""
            runs(c.paragraphs[0], val)
            for r in c.paragraphs[0].runs:
                r.font.size = Pt(9.5)
                if i == 0:
                    r.bold = True
                    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
            if i == 0:
                shade(c, "1E2235")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ---------------------------------------------------------------- title block
doc.add_paragraph(style="Title").add_run("Who Earned the Reward?")
p = P("Ground-Truth Evaluation of Subgroup Credit Assignment under Delayed Coalition Rewards in Cooperative Multi-Agent Reinforcement Learning")
p.runs[0].font.size = Pt(13); p.runs[0].font.color.rgb = GREY
P("**Team Aporia:** Daruru Bhuvan Sai (PES1UG24AM356), H K Manaswin (PES1UG24AM108), Harshitha B T (PES1UG24AM118), Harshith R (PES1UG24AM117)")
P("AFML Course Project, Custom Track. Final report, November 2026.")

H1("Abstract")
P("In cooperative multi-agent reinforcement learning (MARL) a team receives one shared reward, and a credit-assignment method must decide which agents earned it. These methods are normally judged by the reward the team reaches, not by whether the credit they assign is correct, because in real tasks the correct answer is unknown. We build a benchmark where it is known. In a small gridworld we **plant** which group of agents earns each reward and when (an early *enabler*, later *finishers*, a payout delay, and *bystanders* who contribute nothing), and we compute exact credit by replaying each episode with every subset of agents removed. We then grade six well-known methods (VDN, QMIX, COMA, MAPPO, STAS and NA²Q) against this ground truth, with an equal split as the baseline. Four findings stand out. (1) Learned credit rarely beats the equal split: 5 of 96 method-settings do. (2) The credit these methods compute measures the effect of a **one-step** deviation, which is a different quantity from \"who earned the reward\": even a perfect per-step critic misses 99% of the credit of an enabler with slack. (3) Low credit error can be accidental: 30 of the 32 best-scoring settings have critics that barely track their own target. (4) Value-decomposition methods put essentially no value on groups (0.000–0.002 against a true 1.0), because their per-step read-out cannot link agents who act at different times. A small episodic coalition estimator we propose, trained with agent-removed replays, recovers the planted group (dividend error 0.43 against 2.00 for NA²Q) and beats the equal split in 8 of 8 settings, but it relies on being able to replay episodes without some agents. Team reward was only weakly related to credit accuracy (Spearman ρ = 0.15–0.20).")

# ---------------------------------------------------------------- 1
H1("1. Introduction")
P("Consider a team of five agents. Agent A pulls a lever that opens a gate. Some steps later, agents B and C push a crate through the gate together. The team receives 10 points, but only after a delay. Agents D and E wander around and never do anything useful. Anyone watching can say who earned the reward: A, B and C, one third each. A learning algorithm, however, only sees the team's +10. Working out which agents caused it is the **credit-assignment problem**, and getting it wrong matters, because each agent learns from the credit it is given. Wrong credit trains wrong behaviour.")
P("Two kinds of error are especially likely in this example. **Leakage**: D and E absorb credit they did not earn. **Missed credit**: A acted long before the reward arrived, so it is under-credited. Both get worse as the reward is delayed and as more bystanders are added, at least in principle.")
P("Many MARL methods address credit assignment, from value decomposition (VDN, QMIX) to counterfactual critics (COMA) and methods that redistribute delayed rewards (STAS). They are evaluated almost entirely on **team performance**. A team can succeed while its internal credit is wrong, so team performance cannot tell us whether credit is right. This project asks a direct question:")
P("*When a subgroup of agents earns a delayed reward, do standard credit-assignment methods recover who earned it?*")
P("Answering it requires an answer key. Our contributions are:")
B([
    "**A planted-coalition benchmark with exact ground truth.** A gridworld whose events require specific agents in a specific order, with controllable delay, bystanders, overlapping groups and partial synergy. Exact credit at the level of groups (Harsanyi dividends) and individuals (Shapley values, leave-one-out) is computed by replaying episodes with agents removed.",
    "**A fair grading protocol.** Each method is compared only with the ground-truth quantity its credit read-out actually estimates, critics are trained on fixed scripted data to separate credit from exploration, and credit is graded at two levels (fidelity to the method's own target, and the gap between that target and \"who earned it\").",
    "**A pre-registered evaluation of six methods**, whose main findings are that standard credit read-outs rarely beat an equal split, measure a one-step counterfactual rather than contribution, and cannot represent groups whose members act at different times.",
    "**A lightweight episodic coalition estimator** that recovers the planted group when replay data with agents removed is available.",
])

FIGURE("fig0_overview.png", "(a) The running example used throughout. A, B and C earn the reward together; D and E are bystanders. (b) How credit is graded: plant the structure, record episodes of a scripted team, replay each episode with every subset of agents removed, compute exact credit, and compare each method's credit with it.", width=15.5, folder=HERE)

# ---------------------------------------------------------------- 2
H1("2. Background: what does “who earned it” mean?")
P("Let N be the set of n agents. For any subset S of agents, the **coalition value** v(S) is the reward the team earns when only the agents in S act and the others are removed. With n = 5 there are 2⁵ = 32 subsets. From v we compute three standard quantities from cooperative game theory (Table 1).")
TABLE("Three ground-truth notions of credit, with values for the running example (reward 10 earned by {A, B, C}).", [
    ["Quantity", "Question it answers", "Running example"],
    ["**Harsanyi dividend** Δ(S) [Harsanyi 1963]", "Which group created value that no smaller group creates by itself?", "Δ({A,B,C}) = 10; every other dividend is 0"],
    ["**Shapley value** φᵢ [Shapley 1953]", "What is agent i's fair share? (each dividend split equally among its members)", "φ_A = φ_B = φ_C = 10/3; φ_D = φ_E = 0"],
    ["**Leave-one-out** mᵢ = v(N) − v(N∖{i})", "How much is lost if agent i alone is removed?", "m_A = m_B = m_C = 10; m_D = m_E = 0"],
], [4.2, 6.4, 5.8])
P("After normalising to shares the three agree in this example (a third each for A, B and C). They differ when groups overlap or when partial teams earn partial reward. Dividends matter because individual shares can hide group structure: in a game where A and B earn 2 only together, and in one where each earns 1 alone, both agents get Shapley value 1, yet one game is pure teamwork and the other has none. Only the dividends tell them apart.")
P("**Metrics.** Credit error is the L1 distance between a method's normalised credit shares and the true shares, from 0 (exact) to 2 (completely wrong). Leakage Λ is the share given to bystanders (ideal 0). Missed enabler credit M is the fraction of the enabler's true credit that the method fails to give it (ideal 0; above 1 when the enabler gets negative credit). At the group level we report dividend error (0 to 2), recall and precision of the true groups, and **interaction mass**, the share of value assigned to groups of two or more agents (1.0 in our coalition layouts). Every result is compared with the **equal split** (1/n to every agent), which needs no learning.")

# ---------------------------------------------------------------- 3
H1("3. Related work")
P("**Credit assignment in cooperative MARL.** Value-decomposition methods learn per-agent utilities whose combination approximates the team value: a sum in VDN [Sunehag et al. 2018], a monotone mixing network in QMIX [Rashid et al. 2018], and more expressive factorisations in QTRAN, Weighted QMIX and QPLEX [Son et al. 2019; Rashid et al. 2020; Wang et al. 2021]. COMA [Foerster et al. 2018] gives each agent a counterfactual advantage against a baseline that marginalises its own action. MAPPO [Yu et al. 2022] uses a centralised critic with a shared advantage. NA²Q [Liu et al. 2023] adds explicit terms for pairs of agents, making the mixer interpretable at the group level. STAS [Chen et al. 2024] and TAR² [Kapoor et al. 2025] redistribute a delayed episodic reward across time steps and agents. All of these are evaluated primarily by team return.")
P("**Shapley-based credit.** SHAQ [Wang et al. 2022] and Shapley Counterfactual Credits [Li et al. 2021] build Shapley values into learning. Exact Shapley computation needs all 2ⁿ coalitions, which motivates approximations; our ground truth pays that cost once, offline, to grade other methods.")
P("**Checking credit against a known answer.** Mahjoub et al. [2023] compare an agent-importance estimate with true Shapley values in standard benchmarks, but for their own estimator and per agent. In explainable machine learning, grading attributions against planted ground truth is established practice (shapiq [Muschalik et al. 2024], XAI-Bench [Liu et al. 2021], AttributionLab [Zhang et al. 2023]); we adapt that methodology to multi-agent credit. To our knowledge, standard MARL learners have not been graded against planted *coalitions* under *delayed* reward, nor have their credit read-outs been validated against the quantity they claim to estimate. Our original proposal also cited MA-POCA [Cohen et al. 2021] for agents that leave mid-episode; we kept that out of scope, and Saleh Abadi and Soh [2025] survey the wider challenges in open agent systems.")

# ---------------------------------------------------------------- 4
H1("4. The benchmark")
H2("4.1 Planted Coalition Gridworld")
P("Agents move on a small grid (actions: no-op, four moves, interact) and can block each other. Each **event** has enablers, who must each pull their own lever, and finishers, who must interact together on their goal cells while the gate is open. Stations are keyed, so only the designated agent can use them: the earning group is planted by identity. A gate opened at step t is usable from t + 1, so enablers always act before finishers. The reward of an event completed at step t is paid at t + d. Bystanders can move and block but never activate anything. Other layouts add overlapping events (an agent in two groups) and partial synergy (a partial team earns part of the reward). Agents observe a fixed-size vector with presence flags; the centralised critic's state additionally holds a payout countdown, which we added after a pilot showed the critic's state was otherwise not Markov under delay.")
H2("4.2 Exact ground truth by removal")
P("A scripted team completes the events, with exploration noise (each coalition member takes a random action with probability 0.2). For each recorded episode we replay the full team's actions with every subset of agents removed (open-loop replay), read off v(S) for all 2ⁿ subsets, and compute dividends, Shapley values and leave-one-out credit. This costs about 60 ms per episode for n = 5. Physical interference sometimes makes the realised group differ from the planted one, so we always grade against the **realised** truth and report how often the two agree. The calculator is unit-tested: it reproduces the running example and agrees with brute-force formulas on random games.")
H2("4.3 Reading credit out of each method")
P("A method is compared only with the ground-truth quantity its read-out estimates. For VDN and QMIX, an agent's credit at a step is the drop in team value Q_tot when that agent's action is replaced by a no-op; COMA's is its counterfactual advantage; these are summed over the episode and compared with leave-one-out credit. MAPPO's advantage is shared by all agents, so after normalisation every agent receives ±1/n by construction; we report it only as a reference. STAS's credit is the reward it assigns to each agent; NA²Q's is the Shapley value of its mixer game. For group-level analysis we read VDN, QMIX and NA²Q out as coalition games and take their exact dividends. NA²Q and STAS are our own reimplementations, because the official repositories have no licence.")
H2("4.4 Protocol")
P("**Fixed-data evaluation (our “Mode P”).** Each method's critic or return model is trained on 200 scripted episodes and tested on 30 held-out episodes (1,500 updates, minibatches of 32). Actors are never trained, so credit estimation is isolated from exploration. **Two-level grading.** We also build a per-step counterfactual truth (replace one agent's action with a no-op at one step, let the scripted team continue, and record the change in discounted return, γ = 0.99). *Fidelity* compares a method's read-out with this per-step truth; the *meaning gap* compares the per-step truth with removal credit. **Pre-registration.** Pass thresholds were fixed after two pilots and before the main runs. Because the smallest possible Wilcoxon p with 5 seeds (1/32) can never survive Holm correction, the primary test was changed to a one-sided paired t-test on per-seed means before any main result was seen; the amendment is committed in the repository. A hypothesis family is supported only if every eligible condition passes.")
H2("4.5 Experiments")
B([
    "**Main grid:** delay d ∈ {0, 10, 25, 50}, bystanders b ∈ {0, 2, 4}, enabler with slack or on the critical path, 5 seeds: 120 cells, VDN, QMIX, COMA and MAPPO, about 20 CPU-hours.",
    "**Group-level grid:** overlap and partial-synergy layouts, d ∈ {0, 25, 50}, 5 seeds, six methods (30 cells, 87 minutes on 8 workers).",
    "**Estimator grid:** three layouts, d ∈ {0, 50}, 5 seeds, our estimator against NA²Q and STAS (30 cells plus 10 reference cells).",
    "**Full training (“Mode T”):** VDN and QMIX learning their own behaviour, d ∈ {0, 25}, 5 seeds, credit scored at every checkpoint.",
])

# ---------------------------------------------------------------- 5
H1("5. Results")
H2("5.1 Learned credit rarely beats an equal split")
FIGURE("fig1_l1_vs_uniform.png", "Credit error against delay (slack enabler; mean ± sd over 5 seeds). Lower is better; the dashed line is the equal split. With no bystanders the equal split is exactly right, so its line is at 0. MAPPO is omitted because its credit is ±1/n by construction.")
P("Across the main grid, a method's mean credit error is below the equal split in only **5 of 96** method-conditions, almost all at d ≤ 10 (one is MAPPO's ±1/n artefact landing near uniform). No method beats it at d ≥ 25, and COMA never does. Error grows with delay and with the number of bystanders (Figure 2). On the group-level layouts the result is starker: the equal split scores 0.45 (overlap) and 0.89 (partial synergy), and no method's mean beats it in any of 36 conditions. This comparison was added after seeing the data (post-hoc), but it changes how every error number should be read: most of the credit these methods compute carries little information about who earned the reward beyond \"share equally\".")

H2("5.2 The credit is a one-step counterfactual, not contribution")
P("The read-outs of VDN, QMIX and COMA ask: *if this agent did nothing at this one step, how much reward would be lost?* An enabler with slack simply pulls the lever one step later, so the answer is close to zero. The question \"who earned it\" asks instead: *if this agent were absent, how much would be lost?* For the enabler, everything. To show this is a property of the definition rather than of training, we built a **perfect per-step critic** from the simulator. Its missed enabler credit is **0.96–0.99** in all 24 slack cells, independent of delay. More data or larger networks cannot close this gap.")
P("When the enabler is on the critical path, the per-step notion does credit it: splitting episodes by whether they were actually critical (post-hoc), missed credit falls to 0.11–0.12 in critical episodes. The pre-registered test of this prediction (H1b) still failed, because only 54% of the episodes we intended to be critical actually were.")

H2("5.3 Low error is not faithful credit")
P("Two-level grading shows that the best end-to-end numbers are mostly not faithful to the method's own target. In **30 of the 32** best-scoring cells, the correlation between the read-out and its per-step target is below 0.3, and it is at most 0.36 for every method. QMIX at d = 0 with two bystanders has the lowest credit error in the grid (0.44) but a correlation of only 0.14 with its own target. A good score can therefore be an accident. COMA, whose counterfactual baseline was expected to reduce leakage, leaks the **most** (0.42 at b = 2 against 0.14 for VDN and 0.07 for QMIX), because its noisy per-agent signal spreads credit evenly, including onto bystanders.")

H2("5.4 Value-decomposition methods cannot see groups")
FIGURE("fig2_interaction_mass.png", "Interaction mass, the share of value assigned to groups of two or more agents (d = 0, mean ± sd over 5 seeds). The true value is 1.0. The orange and pink bars are our estimator (Section 5.5).", width=14.5)
P("Read out as coalition games, VDN, QMIX and NA²Q assign **0.000–0.002** of their value to groups, against a true 1.0. Dividend error is at its maximum (about 2.0) for all of them. The cause is the read-out, not the networks' capacity: it swaps actions within a single time step, but our groups act at different steps (A at step 3, B and C at step 9). Each per-step game is therefore additive, and a sum of additive games stays additive. QMIX can represent monotone interactions and NA²Q has explicit pair terms, and neither helps. Because every method sits at the maximum error, the pre-registered group-level test (H3) could not separate methods and is uninformative; Section 5.5 is the fair test.")

H2("5.5 An episodic coalition estimator recovers the group")
P("This is the mechanism our proposal called for. The **episodic coalition estimator (ECE-k)** predicts an episode's total return from which agents took part, with a term for every group of up to k agents (k = 2 or 3). We read off these terms as low-order dividends and split each equally among its members, rescaled so the credit sums to the return. It is trained on normal episodes plus open-loop replays with random agents removed (\"replay data\").")
FIGURE("fig3_dividend_error.png", "Dividend error (0 is exact; 2 is the maximum) at d = 0, mean ± sd over 5 seeds. Only the estimator trained with replay data moves below 2. ECE-2 cannot represent an order-3 group, so it stays at 2 on the coalition layout.", width=14.5)
P("ECE-3 with replay data brings dividend error on the order-3 coalition layout from **2.00** (NA²Q, with or without the same replay data) to **0.43**, with recall and precision of the true group of 0.96 (Figure 4). It is significantly better than NA²Q on dividend error in **8 of 8** pre-registered conditions, and it beats the equal split on individual credit error in 8 of 8, by 0.29–0.61. It is about 3.5× cheaper than NA²Q and STAS on the same machine. Two limits: it does **not** leak significantly less than NA²Q (2 of 8 conditions; the replay data alone removes most of NA²Q's leakage), and **without the replay data it only reproduces the equal split**, because when every agent is always present, the model cannot tell agents apart. ECE-2 is a useful check: it cannot represent the three-agent group, so its dividend error stays at 2.0, yet its individual credit error (0.14) is close to ECE-3's (0.12). Individual credit can look right while the group structure is wrong.")

H2("5.6 Team reward says little about credit quality")
FIGURE("fig4_return_vs_credit.png", "Return against credit accuracy (1 − L1) at each checkpoint of full training. Hollow markers have fewer than 10 of 30 test episodes with defined credit. COMA and MAPPO collapsed during training and could not be assessed.", width=13.5)
P("With full training from scratch, no method ever earned reward: QMIX and MAPPO for 400k steps and QMIX for 1M steps stayed at return 0, because a uniformly random policy completes the task in 0 of 20,000 episodes. Seeding the replay buffer with 200 scripted episodes let the off-policy learners (VDN and QMIX) learn. Within each method, return is only weakly correlated with credit accuracy: Spearman **ρ = +0.20** for QMIX (95% CI [−0.10, +0.48]) and **+0.15** for VDN ([−0.19, +0.35]), below the 0.7 we set in advance for \"strong\". QMIX earns about twice VDN's return (6.5 against 3.1, p = 0.013) with similar credit error (0.64 against 0.75, p = 0.44). At the final checkpoint QMIX's credit does beat the equal split (0.64 against 0.79, 7 of 10 runs), so online-trained QMIX carries some information about who earned the reward. COMA and MAPPO, even after behavioural-cloning pretraining, collapsed online; credit was defined at too few checkpoints to assess them. This part is descriptive, not confirmatory.")

H2("5.7 Robustness")
P("Using Shapley shares instead of leave-one-out changes credit error by at most 0.03–0.10; method rankings agree in 27 of 30 conditions and no verdict changes. Re-running the scripted team for each subset (closed-loop removal, so the remaining agents can react) recovers the planted group in 100% of 3,600 test episodes, against 83% with open-loop replay. Under that truth the benchmark-stability prediction (H0) passes in all 120 cells, and the delay prediction (H1a) passes through one QMIX condition (Holm p 0.074 → 0.030), so we treat H1a as borderline rather than a firm negative. The pre-registered verdicts remain the open-loop ones.")

TABLE("Pre-registered predictions and outcomes. Thresholds were fixed before the main runs; a family passes only if every eligible condition passes.", [
    ["Prediction", "Outcome", "Note"],
    ["H0: the benchmark gives a stable answer key", "Narrowly not supported (104/120)", "All 120 pass under closed-loop removal"],
    ["H1a: missed enabler credit grows with delay", "Not supported", "QMIX right sign everywhere, Holm p = 0.074; borderline under closed loop"],
    ["H1b: slack, not delay, removes per-step enabler credit", "Not supported as written", "Slack half 24/24; critical manipulation too weak (54% critical)"],
    ["H2: leakage grows with bystanders", "COMA supported (8/8); VDN, QMIX 1/8", "VDN/QMIX right direction in 16/16 (post-hoc sign test)"],
    ["H2′: COMA leaks less than VDN/QMIX", "Reversed (0/8)", "COMA leaks the most"],
    ["H3: individual metrics hide group errors", "Uninformative", "All methods at maximum dividend error"],
    ["H-fid: accuracy does not imply fidelity", "Supported", "30 of the 32 best cells"],
    ["H4: return is a weak proxy for credit", "Consistent (descriptive)", "ρ = 0.15–0.20; VDN and QMIX only"],
    ["H5: coalition estimator beats NA²Q", "Supported on dividend error (8/8) and vs equal split (8/8); not on leakage", "Needs replay data"],
], [5.6, 4.9, 5.9])

# ---------------------------------------------------------------- 6
H1("6. Discussion")
P("**What the results mean.** The central lesson is that \"credit\" in standard MARL methods and \"who earned the reward\" are different quantities. The per-step counterfactual these methods compute is useful for learning: it tells an agent whether its current action matters right now. But it is not a measure of contribution, and it systematically ignores agents whose contribution is not time-critical. Methods built to split the team value inherit a related blind spot at the group level: a read-out that compares values within one time step cannot represent a team whose members act at different times. Since team return was only weakly related to credit accuracy, these failures would not be visible from learning curves.")
P("**For practitioners.** Any table of credit error should include the equal split as a row; without it, numbers that look reasonable can carry no information. A method's credit should be checked for fidelity to its own target as well as against the truth, since a good end-to-end score can be accidental.")
P("**On the negative results.** Several predictions failed, and each failure has a specific explanation: too few seeds for VDN and QMIX leakage (every difference had the expected sign), a weak manipulation of enabler criticality, and a group-level test that every method failed at the same maximum. We report them as findings rather than hide them, as our proposal committed to (\"a negative Stage 1 result is still a valid finding\").")
H2("6.1 Limitations")
B([
    "One environment family, at most 7 agents, 5 seeds per cell; power is limited and the pass rules are strict.",
    "Most results train critics on scripted data. Full training covers only VDN and QMIX, and needed a seeded replay buffer, which is not standard online training.",
    "VDN and QMIX learn the optimal value Q*, while COMA, MAPPO and NA²Q (with the SARSA target we used, because NA²Q's original target diverged on fixed data) learn the behaviour policy's value.",
    "NA²Q and STAS are our reimplementations; the official code was compared line by line but not run. STAS ran on a mix of devices in the estimator grid.",
    "Our estimator needs replays with agents removed, a stronger assumption than any compared method makes and one real systems may not allow.",
    "The equal-split comparison, the split by realised criticality, the VDN/QMIX sign test and the robustness checks were decided after seeing data and are labelled post-hoc.",
])

# ---------------------------------------------------------------- 7
H1("7. Conclusion")
P("We built a benchmark that knows who earned each reward and used it to grade six multi-agent credit-assignment methods. Their credit rarely beats an equal split, measures a one-step what-if rather than contribution, and cannot see groups whose members act at different times; and team reward says little about any of this. Group-level credit is recoverable by a small estimator when episodes can be replayed without some agents. Natural next steps are testing whether the same gaps appear in standard benchmarks such as Level-Based Foraging, and finding a version of the estimator that does not need agent-removed replays.")
P("**Reproducibility.** Code, configurations, raw results and per-hypothesis write-ups are in the project repository. docs/running_on_another_machine.md covers setup and every grid; scripts/make_figures.py regenerates Figures 2–5; docs/credit_viz.html is an interactive per-episode credit viewer.")

# ---------------------------------------------------------------- refs
H1("References")
refs = [
    "Chen, S., Zhang, Z., Yang, Y., & Du, Y. (2024). STAS: Spatial-temporal return decomposition for solving sparse rewards problems in multi-agent reinforcement learning. AAAI. arXiv:2304.07520.",
    "Cohen, A., Teng, E., Berges, V.-P., et al. (2021). On the use and misuse of absorbing states in multi-agent reinforcement learning (MA-POCA). arXiv:2111.05992.",
    "Foerster, J., Farquhar, G., Afouras, T., Nardelli, N., & Whiteson, S. (2018). Counterfactual multi-agent policy gradients. AAAI. arXiv:1705.08926.",
    "Harsanyi, J. C. (1963). A simplified bargaining model for the n-person cooperative game. International Economic Review, 4(2).",
    "Kapoor, A., Tessera, K., Baranwal, M., Khadilkar, H., Peters, J., Albrecht, S., & Sun, M. (2025). Redistributing rewards across time and agents for multi-agent reinforcement learning (TAR²). arXiv:2502.04864.",
    "Li, J., Kuang, K., Wang, B., et al. (2021). Shapley counterfactual credits for multi-agent reinforcement learning. KDD. arXiv:2106.00285.",
    "Liu, Y., Khandagale, S., White, C., & Neiswanger, W. (2021). Synthetic benchmarks for scientific research in explainable machine learning (XAI-Bench). arXiv:2106.12543.",
    "Liu, Z., Zhu, Y., & Chen, C. (2023). NA²Q: Neural attention additive model for interpretable multi-agent Q-learning. ICML.",
    "Mahjoub, O., de Kock, R., Singh, S., Khlifi, W., Vall, A., Tessera, K., & Pretorius, A. (2023). Efficiently quantifying individual agent importance in cooperative MARL. arXiv:2312.08466.",
    "Muschalik, M., Baniecki, H., Fumagalli, F., Kolpaczki, P., Hammer, B., & Hüllermeier, E. (2024). shapiq: Shapley interactions for machine learning. NeurIPS Datasets and Benchmarks.",
    "Papoudakis, G., Christianos, F., Schäfer, L., & Albrecht, S. V. (2021). Benchmarking multi-agent deep reinforcement learning algorithms in cooperative tasks (EPyMARL). NeurIPS Datasets and Benchmarks. arXiv:2006.07869.",
    "Rashid, T., Samvelyan, M., Schroeder de Witt, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018). QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning. ICML. arXiv:1803.11485.",
    "Rashid, T., Farquhar, G., Peng, B., & Whiteson, S. (2020). Weighted QMIX: Expanding monotonic value function factorisation. NeurIPS. arXiv:2006.10800.",
    "Saleh Abadi, A., & Soh, L.-K. (2025). Challenges in credit assignment for multi-agent reinforcement learning in open agent systems. arXiv:2510.27659.",
    "Shapley, L. S. (1953). A value for n-person games. In Contributions to the Theory of Games II, Princeton University Press.",
    "Son, K., Kim, D., Kang, W. J., Hostallero, D. E., & Yi, Y. (2019). QTRAN: Learning to factorize with transformation for cooperative multi-agent reinforcement learning. ICML. arXiv:1905.05408.",
    "Sunehag, P., Lever, G., Gruslys, A., et al. (2018). Value-decomposition networks for cooperative multi-agent learning based on team reward. AAMAS. arXiv:1706.05296.",
    "Wang, J., Ren, Z., Liu, T., Yu, Y., & Zhang, C. (2021). QPLEX: Duplex dueling multi-agent Q-learning. ICLR. arXiv:2008.01062.",
    "Wang, J., Zhang, Y., Gu, Y., & Kim, T.-K. (2022). SHAQ: Incorporating Shapley value theory into multi-agent Q-learning. NeurIPS. arXiv:2105.15013.",
    "Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2022). The surprising effectiveness of PPO in cooperative multi-agent games (MAPPO). NeurIPS Datasets and Benchmarks. arXiv:2103.01955.",
    "Zhang, Y., et al. (2023). AttributionLab: Faithfulness of feature attribution under controllable environments. arXiv:2310.06514.",
]
for r in refs:
    p = doc.add_paragraph()
    runs(p, r)
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.8)
    p.paragraph_format.space_after = Pt(3)
    for rr in p.runs:
        rr.font.size = Pt(9.5)

doc.save(OUT)
print("wrote", OUT)
