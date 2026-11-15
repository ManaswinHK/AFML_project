// Builds submission/Final_Presentation.pptx
// Run from submission/build:  node make_deck.js
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const FIG = path.join(__dirname, "..", "..", "docs", "figures");
const OUT = path.join(__dirname, "..", "Final_Presentation.pptx");

const THEME = {
  name: "Aporia Credit",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1E2235", lt1: "FFFFFF", dk2: "4A5068", lt2: "EEF1F6",
    accent1: "E0A030", // gold: the reward / coalition members
    accent2: "2E8C7E", // teal: correct / truth
    accent3: "C4563F", // rust: leakage / failure
    accent4: "6B7390", // slate: bystanders, neutral
    accent5: "2A78D6", accent6: "8E5BB5",
    hlink: "2A78D6", folHlink: "8E5BB5",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "Who Earned the Reward?";
pres.author = "Team Aporia";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const W = 13.333, M = 0.6;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.0, w: 8.2, h: 1.5, fontSize: 48, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.65, w: 7.6, h: 1.2, fontSize: 20, color: "D6DAE6", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 1.05, fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Who Earned the Reward?  ·  Team Aporia", options: { x: M, y: 7.05, w: 6, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: C.text2, align: "right" },
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.45, w: W - 2 * M, h: 1.0, fontSize: 30, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: "AEB4C8", align: "right" },
});

// ---------- helpers ----------
let n = 0;
function tb(slide, text, o) {
  slide.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 16, color: C.text1, valign: "top", objectName: "t" + (n++) }, o));
}
// agent token: the deck's visual motif
function agent(slide, letter, x, y, d, kind) {
  const fill = kind === "coal" ? C.accent1 : kind === "by" ? "C9CDD8" : kind === "truth" ? C.accent2 : C.accent3;
  const txt = kind === "by" ? C.text2 : C.background1;
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: "agent" + letter + (n++) });
  tb(slide, letter, { x, y, w: d, h: d, align: "center", valign: "middle", bold: true, fontSize: Math.round(d * 26), color: txt });
}
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill || C.background2 }, line: { color: fill || C.background2 }, objectName: "card" + (n++) });
}
function arrow(slide, x1, y1, x2, y2, color) {
  slide.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color: color || C.text2, width: 2, endArrowType: "triangle" }, objectName: "arrow" + (n++) });
}
function stat(slide, big, label, x, y, w, color) {
  tb(slide, big, { x, y, w, h: 0.95, fontSize: 54, bold: true, color: color || C.accent3, fontFace: "Cambria" });
  tb(slide, label, { x, y: y + 1.0, w, h: 0.9, fontSize: 15, color: C.text2 });
}
function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}

// =====================================================================
// 1. Title
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
  s.addText("Who Earned the Reward?", { placeholder: "title" });
  s.addText("Checking whether multi-agent learning gives credit to the agents who actually earned a delayed team reward", { placeholder: "body" });
  tb(s, "Team Aporia: Daruru Bhuvan Sai · H K Manaswin · Harshitha B T · Harshith R", { x: M, y: 5.6, w: 8.5, h: 0.4, fontSize: 15, color: C.background1 });
  tb(s, "AFML Course Project, Custom Track · November 2026", { x: M, y: 6.0, w: 8.5, h: 0.4, fontSize: 13, color: "AEB4C8" });
  // motif: three earners, two bystanders
  const d = 0.85, y0 = 2.2;
  agent(s, "A", 9.5, y0, d, "coal"); agent(s, "B", 10.55, y0, d, "coal"); agent(s, "C", 11.6, y0, d, "coal");
  agent(s, "D", 10.0, y0 + 1.25, d, "by"); agent(s, "E", 11.05, y0 + 1.25, d, "by");
  tb(s, "earned it", { x: 9.5, y: y0 - 0.5, w: 3, h: 0.35, fontSize: 13, color: HEX.accent1, italic: true });
  tb(s, "just standing there", { x: 10.0, y: y0 + 2.2, w: 3, h: 0.35, fontSize: 13, color: "AEB4C8", italic: true });
  s.addNotes("Good morning. Our project asks a simple question: when a team of learning agents gets one shared reward, does the learning method give credit to the agents who actually earned it? The gold agents earned the reward; the grey ones did nothing. By the end of this talk you will see that standard methods mostly cannot tell the two apart, why that happens, and one way that can.");
}

// 2. Running example
pres.addSection({ title: "Problem" });
{
  const s = content("A team can earn its reward while the credit goes to the wrong agents", "Problem");
  // timeline
  const y = 2.05;
  s.addShape(pres.shapes.LINE, { x: 0.9, y: y + 0.55, w: 7.3, h: 0, line: { color: C.text2, width: 1.5, endArrowType: "triangle" }, objectName: "timeline" });
  // step 1
  agent(s, "A", 1.0, y, 0.7, "coal");
  tb(s, "Step 3\nA pulls a lever and opens a gate", { x: 0.7, y: y + 0.95, w: 2.2, h: 0.9, fontSize: 14 });
  // step 2
  agent(s, "B", 3.5, y, 0.7, "coal"); agent(s, "C", 4.25, y, 0.7, "coal");
  tb(s, "Step 9\nB and C push a crate together", { x: 3.3, y: y + 0.95, w: 2.4, h: 0.9, fontSize: 14 });
  // reward
  card(s, 6.4, y - 0.05, 1.5, 0.8, C.accent2);
  tb(s, "+10", { x: 6.4, y: y - 0.05, w: 1.5, h: 0.8, fontSize: 26, bold: true, color: C.background1, align: "center", valign: "middle" });
  tb(s, "Step 9 + d\nreward arrives d steps late", { x: 6.2, y: y + 0.95, w: 2.2, h: 0.9, fontSize: 14 });
  // bystanders
  agent(s, "D", 1.0, 4.35, 0.6, "by"); agent(s, "E", 1.7, 4.35, 0.6, "by");
  tb(s, "D and E wander nearby and never touch anything", { x: 2.5, y: 4.4, w: 5.5, h: 0.5, fontSize: 14, color: C.text2 });
  // right panel: right answer + two failure modes
  card(s, 8.7, 1.75, 4.05, 1.45);
  tb(s, "The right answer", { x: 8.95, y: 1.9, w: 3.6, h: 0.35, fontSize: 15, bold: true, color: C.accent2 });
  tb(s, "A, B and C each earned a third. D and E earned nothing.", { x: 8.95, y: 2.3, w: 3.6, h: 0.8, fontSize: 15 });
  card(s, 8.7, 3.4, 4.05, 1.35);
  tb(s, "Failure 1: leakage", { x: 8.95, y: 3.55, w: 3.6, h: 0.35, fontSize: 15, bold: true, color: C.accent3 });
  tb(s, "D and E absorb credit they did not earn.", { x: 8.95, y: 3.95, w: 3.6, h: 0.7, fontSize: 15 });
  card(s, 8.7, 4.95, 4.05, 1.55);
  tb(s, "Failure 2: missed credit", { x: 8.95, y: 5.1, w: 3.6, h: 0.35, fontSize: 15, bold: true, color: C.accent3 });
  tb(s, "A acted long before the reward, so it gets too little.", { x: 8.95, y: 5.5, w: 3.6, h: 0.8, fontSize: 15 });
  tb(s, "In multi-agent RL the team only sees the +10. The learner has to work out who caused it: this is credit assignment.", { x: M, y: 5.45, w: 7.6, h: 0.9, fontSize: 15, italic: true, color: C.text2 });
  s.addNotes("Here is the running example we use throughout. Agent A pulls a lever early. Later, B and C push a crate together. The team gets 10 points, but only d steps later. D and E are bystanders: they move around but never do anything useful. The right answer is obvious to us: A, B and C each earned a third. A learning method only sees the +10 and has to figure this out. Two things can go wrong. Leakage: the bystanders get some of the credit. Missed credit: A acted so long before the reward that it gets too little. Wrong credit matters because each agent learns from its credit, so wrong credit trains the wrong behaviour.");
}

// 3. Gap
{
  const s = content("Credit methods are judged by the team's score, never by whether their credit is right", "Problem");
  card(s, M, 1.7, 5.8, 3.0);
  tb(s, "What papers report", { x: M + 0.3, y: 1.9, w: 5.2, h: 0.4, fontSize: 18, bold: true, color: C.accent2 });
  tb(s, [
    { text: "Team reward goes up over training", options: { bullet: true, breakLine: true } },
    { text: "Win rates on benchmark games", options: { bullet: true, breakLine: true } },
    { text: "Comparisons against other methods' scores", options: { bullet: true } },
  ], { x: M + 0.3, y: 2.45, w: 5.2, h: 2.0, fontSize: 16, paraSpaceAfter: 8 });
  card(s, 6.93, 1.7, 5.8, 3.0);
  tb(s, "What is almost never checked", { x: 7.23, y: 1.9, w: 5.2, h: 0.4, fontSize: 18, bold: true, color: C.accent3 });
  tb(s, [
    { text: "Did credit go to the agents who earned the reward?", options: { bullet: true, breakLine: true } },
    { text: "Did bystanders get anything?", options: { bullet: true, breakLine: true } },
    { text: "Was the right group found, not just the right individuals?", options: { bullet: true } },
  ], { x: 7.23, y: 2.45, w: 5.2, h: 2.0, fontSize: 16, paraSpaceAfter: 8 });
  tb(s, "Why not? In real tasks nobody knows the right answer, so there is nothing to grade against.", { x: M, y: 4.95, w: W - 2 * M, h: 0.45, fontSize: 16, color: C.text2 });
  card(s, M, 5.55, W - 2 * M, 1.2, C.text1);
  tb(s, [
    { text: "Our question: ", options: { bold: true, color: HEX.accent1 } },
    { text: "when a subgroup earns a delayed reward, do standard credit-assignment methods recover who earned it?", options: { color: C.background1 } },
  ], { x: M + 0.3, y: 5.65, w: W - 2 * M - 0.6, h: 1.0, fontSize: 19, valign: "middle" });
  s.addNotes("Why has nobody measured this? Methods like QMIX and COMA are evaluated on how much reward the team ends up getting. That tells you the team works, but not whether the credit inside the team is right. A team can succeed with wrong credit. The reason nobody checks is that in real tasks there is no answer key. So our project is about building the answer key. Our question is on the slide.");
}

// 4. Approach
pres.addSection({ title: "Approach" });
{
  const s = content("We plant the answer in a gridworld, then remove agents to measure the exact truth", "Approach");
  const steps = [
    ["1", "Plant", "Design events where we choose who must act and when"],
    ["2", "Play", "A scripted team completes the events; we record each episode"],
    ["3", "Remove", "Replay each episode with every subset of agents taken out (32 replays for 5 agents)"],
    ["4", "Compute", "Exact credit for every agent and every group, from those replays"],
    ["5", "Grade", "Compare each method's credit with the exact answer"],
  ];
  const bw = 2.2, gap = 0.27, y = 1.8;
  steps.forEach((st, i) => {
    const x = M + i * (bw + gap);
    card(s, x, y, bw, 2.55, i === 4 ? C.text1 : C.background2);
    const dark = i === 4;
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.2, w: 0.55, h: 0.55, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "num" + i });
    tb(s, st[0], { x: x + 0.2, y: y + 0.2, w: 0.55, h: 0.55, align: "center", valign: "middle", bold: true, color: C.background1, fontSize: 18 });
    tb(s, st[1], { x: x + 0.2, y: y + 0.9, w: bw - 0.4, h: 0.4, bold: true, fontSize: 18, color: dark ? C.background1 : C.text1 });
    tb(s, st[2], { x: x + 0.2, y: y + 1.35, w: bw - 0.35, h: 1.15, fontSize: 14, color: dark ? "D6DAE6" : C.text1 });
    if (i < 4) arrow(s, x + bw + 0.02, y + 1.27, x + bw + gap - 0.02, y + 1.27);
  });
  tb(s, "Knobs we turn", { x: M, y: 4.75, w: 4, h: 0.4, fontSize: 18, bold: true, color: C.text1 });
  const knobs = [
    ["Reward delay", "0, 10, 25 or 50 steps"],
    ["Bystanders", "0, 2 or 4 useless agents"],
    ["Enabler timing", "A has slack, or A is on the critical path"],
    ["Team shape", "overlapping teams; partial teams earning part of the reward"],
  ];
  knobs.forEach((k, i) => {
    const x = M + i * 3.05;
    tb(s, k[0], { x, y: 5.25, w: 2.85, h: 0.35, fontSize: 15, bold: true, color: C.accent2 });
    tb(s, k[1], { x, y: 5.62, w: 2.85, h: 0.8, fontSize: 14, color: C.text2 });
  });
  s.addNotes("Our approach in one line: plant the answer, then measure it exactly. We built a small gridworld where we decide which agents must act together, and when. A scripted team plays it. Then, for every recorded episode, we replay it with every possible subset of agents removed. With five agents that is 32 replays. From those replays we can compute exactly how much each agent and each group contributed. Then we grade each method against that. We vary four things: how late the reward arrives, how many bystanders there are, whether the early agent has slack, and the shape of the team.");
}

// 5. Three definitions
{
  const s = content("There are three exact ways to answer “who deserves credit?”, and we compute all of them", "Approach");
  const rows = [
    [{ text: "Measure", options: { bold: true, color: C.background1, fill: { color: C.text1 } } },
     { text: "Question it answers", options: { bold: true, color: C.background1, fill: { color: C.text1 } } },
     { text: "Running example (reward 10)", options: { bold: true, color: C.background1, fill: { color: C.text1 } } }],
    ["Harsanyi dividend", "Which group created value that no smaller group creates alone?", "Group {A, B, C} = 10; every other group = 0"],
    ["Shapley value", "What is each agent's fair share of that value?", "A, B, C get 10/3 each; D, E get 0"],
    ["Leave-one-out", "How much is lost if this one agent is removed?", "A, B, C lose 10 each; D, E lose 0"],
  ];
  s.addTable(rows, { x: M, y: 1.75, w: W - 2 * M, colW: [2.6, 5.0, 4.53], fontSize: 16, color: C.text1, border: { type: "solid", pt: 0.75, color: "D5D9E3" }, fill: { color: C.background1 }, rowH: 0.62, valign: "middle", margin: 0.1, objectName: "deftable" });
  card(s, M, 4.6, W - 2 * M, 1.95);
  tb(s, "Why groups matter, not only individual shares", { x: M + 0.3, y: 4.75, w: 8, h: 0.4, fontSize: 17, bold: true, color: C.accent2 });
  tb(s, [
    { text: "Game 1: A and B earn 2 only together.  ", options: { breakLine: true } },
    { text: "Game 2: A and B each earn 1 on their own.", options: { breakLine: true } },
    { text: "Both give the same individual shares (1 each), yet one is pure teamwork and the other is none. Only the group measure tells them apart.", options: { color: C.text2 } },
  ], { x: M + 0.3, y: 5.2, w: W - 2 * M - 0.6, h: 1.25, fontSize: 15 });
  s.addNotes("Credit can mean different things, so we compute three exact versions. The Harsanyi dividend tells you which group created the value: here the group A, B, C created all 10 points. The Shapley value splits that fairly: 10 over 3 each. Leave-one-out asks what you lose if one agent is removed: 10 for each of A, B, C. After normalising, they agree here; they differ when teams overlap. The box at the bottom is why we care about groups at all: two games can give identical individual shares while one is pure teamwork and the other has none.");
}

// 6. Methods and metrics
{
  const s = content("We grade six well-known methods, and an equal split as the baseline to beat", "Approach");
  const fams = [
    ["Split the team value into agent parts", "VDN, QMIX, NA²Q", "Credit = how much the team value drops if the agent does nothing"],
    ["Learn a team critic", "COMA, MAPPO", "Credit = the agent's advantage from a centralised critic"],
    ["Spread the reward over time and agents", "STAS", "Credit = the reward the model hands to each agent"],
  ];
  fams.forEach((f, i) => {
    const x = M + i * 4.1;
    card(s, x, 1.7, 3.9, 2.35);
    tb(s, f[0], { x: x + 0.25, y: 1.85, w: 3.4, h: 0.7, fontSize: 16, bold: true, color: C.text1 });
    tb(s, f[1], { x: x + 0.25, y: 2.55, w: 3.4, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
    tb(s, f[2], { x: x + 0.25, y: 3.0, w: 3.4, h: 0.95, fontSize: 14, color: C.text2 });
  });
  tb(s, "How we score them", { x: M, y: 4.3, w: 5, h: 0.4, fontSize: 18, bold: true });
  const mets = [
    ["Credit error", "distance from the true shares: 0 is perfect, 2 is completely wrong"],
    ["Leakage", "share of credit given to bystanders (ideal 0)"],
    ["Missed credit", "how much of the early enabler's credit is lost (ideal 0)"],
    ["Group recovery", "did the method find the group that earned the reward?"],
  ];
  mets.forEach((m, i) => {
    const x = M + (i % 2) * 6.1, y = 4.8 + Math.floor(i / 2) * 0.85;
    tb(s, [{ text: m[0] + ": ", options: { bold: true, color: C.accent2 } }, { text: m[1] }], { x, y, w: 5.9, h: 0.75, fontSize: 15 });
  });
  tb(s, "Critics are trained on fixed scripted play, so we test credit, not exploration. Main grid: 120 settings × 5 seeds, about 20 CPU-hours.", { x: M, y: 6.5, w: W - 2 * M, h: 0.4, fontSize: 13, italic: true, color: C.text2 });
  s.addNotes("We tested six methods from three families. VDN, QMIX and NA-squared-Q split the team's value into per-agent parts. COMA and MAPPO learn a team critic. STAS spreads the final reward over time and agents. For each one we read out its credit in the way that matches what it actually estimates. We score four things: overall credit error, leakage to bystanders, missed credit for the early agent, and whether the right group was found. Every result is compared with the simplest possible baseline: give everyone an equal share. To isolate credit from exploration, the critics learn from fixed scripted play.");
}

// 7. Finding 1
pres.addSection({ title: "Findings" });
{
  const s = content("Finding 1: learned credit rarely beats simply splitting the reward equally", "Findings");
  s.addImage({ path: path.join(FIG, "fig1_l1_vs_uniform.png"), x: M, y: 1.6, w: 9.0, h: 9.0 * 612 / 1785, altText: "Credit error against delay for VDN, QMIX and COMA with 0, 2 and 4 bystanders, compared with the equal-split line", objectName: "fig1" });
  stat(s, "5 of 96", "method-settings beat the equal split, almost all with a short delay (10 steps or less)", 9.95, 1.75, 2.8);
  tb(s, [
    { text: "Lower is better. The dashed line is the equal split.", options: { breakLine: true } },
    { text: "Error grows with delay and with more bystanders, and COMA is worst throughout.", options: {} },
  ], { x: M, y: 4.85, w: 9.0, h: 0.9, fontSize: 15, color: C.text2 });
  tb(s, "With no bystanders the equal split is exactly right, so its line sits at 0.", { x: M, y: 5.8, w: 9.0, h: 0.4, fontSize: 13, italic: true, color: C.text2 });
  s.addNotes("First finding. On the y-axis is credit error: zero is perfect. The dashed line is what you get by giving everyone an equal share without learning anything. Across the main grid, only 5 of 96 method-settings beat that equal split, and almost all of those have a short delay. As the reward is delayed further, or more bystanders are added, the error grows. So most of the credit these methods compute carries little information about who earned the reward.");
}

// 8. Finding 2
{
  const s = content("Finding 2: their credit answers a different question, “what if I paused for one step?”", "Findings");
  card(s, M, 1.7, 5.9, 3.3);
  tb(s, "What the methods measure", { x: M + 0.3, y: 1.85, w: 5.3, h: 0.4, fontSize: 18, bold: true, color: C.accent3 });
  tb(s, "“If A does nothing at this one step, how much reward is lost?”", { x: M + 0.3, y: 2.35, w: 5.3, h: 0.8, fontSize: 17, italic: true });
  tb(s, "A simply pulls the lever one step later. The team still succeeds, so A's credit is about zero.", { x: M + 0.3, y: 3.25, w: 5.3, h: 1.0, fontSize: 15, color: C.text2 });
  agent(s, "A", M + 0.3, 4.3, 0.55, "by"); tb(s, "credit ≈ 0", { x: M + 1.0, y: 4.35, w: 3, h: 0.45, fontSize: 17, bold: true, color: C.accent3 });
  card(s, 6.83, 1.7, 5.9, 3.3);
  tb(s, "What “who earned it” means", { x: 7.13, y: 1.85, w: 5.3, h: 0.4, fontSize: 18, bold: true, color: C.accent2 });
  tb(s, "“If A were not there at all, how much reward is lost?”", { x: 7.13, y: 2.35, w: 5.3, h: 0.8, fontSize: 17, italic: true });
  tb(s, "No lever, no gate, no crate: the whole reward is lost. A deserves a full share.", { x: 7.13, y: 3.25, w: 5.3, h: 1.0, fontSize: 15, color: C.text2 });
  agent(s, "A", 7.13, 4.3, 0.55, "coal"); tb(s, "credit = a full third", { x: 7.83, y: 4.35, w: 4, h: 0.45, fontSize: 17, bold: true, color: C.accent2 });
  stat(s, "0.99", "missed credit for an enabler with slack, even for a perfect per-step critic, at every delay we tried (1.0 = all of it missed)", M, 5.2, 7.5);
  tb(s, "This is a gap in the definition, not in training. More data or a bigger network cannot close it.", { x: 8.6, y: 5.35, w: 4.13, h: 1.3, fontSize: 16, bold: true, color: C.text1 });
  s.addNotes("Second finding, and the one we think matters most. The credit VDN, QMIX and COMA compute asks: if this agent did nothing at this one step, what would change? If A has slack, it simply pulls the lever one step later, nothing is lost, so A gets about zero credit. But the question we care about is: if A were not there at all, what would be lost? Then the answer is everything. We checked this with a perfect per-step critic built from the simulator: it still misses 99 percent of A's credit, at every delay. So this is not a training problem; the read-out answers a different question.");
}

// 9. Finding 3
{
  const s = content("Finding 3: a low credit error can be luck; the critics barely track their own target", "Findings");
  // two-level diagram
  const y = 1.95;
  const boxes = [["Method's credit", C.accent1], ["Its own target\n(one-step what-if)", C.accent4], ["Who earned it\n(remove the agent)", C.accent2]];
  boxes.forEach((b, i) => {
    const x = M + i * 4.2;
    card(s, x, y, 3.3, 1.15, b[1]);
    tb(s, b[0], { x, y, w: 3.3, h: 1.15, align: "center", valign: "middle", bold: true, fontSize: 16, color: C.background1 });
    if (i < 2) arrow(s, x + 3.35, y + 0.57, x + 4.15, y + 0.57);
  });
  tb(s, "Fidelity: is the critic right about what it estimates?", { x: M, y: y + 1.3, w: 4.6, h: 0.7, fontSize: 14, color: C.text2 });
  tb(s, "Meaning gap: is that even the right question? (Finding 2)", { x: M + 4.2, y: y + 1.3, w: 4.6, h: 0.7, fontSize: 14, color: C.text2 });
  stat(s, "30 of 32", "best-scoring settings have a critic that barely matches its own target (correlation below 0.3)", M, 4.35, 5.6);
  stat(s, "0.44 vs 0.14", "QMIX's best setting: lowest credit error overall, yet correlation with its own target is only 0.14", 6.83, 4.35, 5.9, C.accent1);
  s.addNotes("Third finding. We graded credit at two levels. First, fidelity: does the method's credit match the thing it is actually trying to estimate? Second, the meaning gap from the last slide. What we found is that the best-looking results are mostly not faithful. In 30 of the 32 best-scoring settings, the critic's credit correlates less than 0.3 with its own target. QMIX's single best setting has credit error 0.44, the lowest anywhere, but a correlation of only 0.14 with what it is supposed to estimate. So a good score can be an accident, and you have to check both levels before trusting it.");
}

// 10. Finding 4
{
  const s = content("Finding 4: methods that split the team value cannot see teamwork at all", "Findings");
  s.addImage({ path: path.join(FIG, "fig2_interaction_mass.png"), x: M, y: 1.55, w: 7.6, h: 7.6 * 663 / 1445, altText: "Interaction mass: VDN, QMIX and NA2Q near 0, the episodic estimator near the true 1.0", objectName: "fig2" });
  stat(s, "≈ 0", "share of value VDN, QMIX and NA²Q put on groups (0.000–0.002). The true share is 1.0: all of it.", 8.6, 1.7, 4.1);
  card(s, 8.6, 3.9, 4.13, 2.6);
  tb(s, "Why", { x: 8.85, y: 4.05, w: 3.6, h: 0.4, fontSize: 17, bold: true, color: C.accent2 });
  tb(s, "Their value is compared one step at a time. A acts at step 3 and B, C at step 9, so no single step ever contains the whole team.", { x: 8.85, y: 4.5, w: 3.7, h: 1.9, fontSize: 15 });
  tb(s, "The orange and pink bars are our estimator (next slide).", { x: M, y: 5.2, w: 7.6, h: 0.4, fontSize: 13, italic: true, color: C.text2 });
  s.addNotes("Fourth finding, at the group level. We asked VDN, QMIX and NA-squared-Q which groups created the value. In our layouts, all of the value comes from groups, so the right answer is 1.0. They put essentially nothing on groups: between 0 and 0.002. The reason is structural. They compare values one time step at a time, but our team acts at different times: A at step 3, B and C at step 9. No single step contains the whole team, so the team is invisible. NA-squared-Q has explicit pair terms, and it still cannot see it.");
}

// 11. Our estimator
pres.addSection({ title: "Our estimator" });
{
  const s = content("Our small coalition estimator, trained on replays with agents removed, finds the team", "Our estimator");
  card(s, M, 1.6, 4.4, 4.0);
  tb(s, "How it works", { x: M + 0.3, y: 1.75, w: 3.8, h: 0.4, fontSize: 18, bold: true, color: C.accent2 });
  tb(s, [
    { text: "1.  Predict the episode's total reward from which agents took part, with terms for groups of up to 3 agents", options: { breakLine: true } },
    { text: "2.  Train it on normal episodes plus replays with random agents removed", options: { breakLine: true } },
    { text: "3.  Read off each group's value; split it equally among its members", options: {} },
  ], { x: M + 0.3, y: 2.25, w: 3.85, h: 3.2, fontSize: 15, paraSpaceAfter: 10 });
  s.addImage({ path: path.join(FIG, "fig3_dividend_error.png"), x: 5.3, y: 1.6, w: 7.43, h: 7.43 * 663 / 1445, altText: "Dividend error: NA2Q at the maximum 2.0, ECE-3 with replay data at 0.43 on the coalition layout", objectName: "fig3" });
  tb(s, [
    { text: "0.43 vs 2.00", options: { bold: true, color: C.accent2 } },
    { text: " group error against NA²Q (2.00 is the maximum). Recall and precision of the true team: 0.96. Beats the equal split in 8 of 8 settings, at about 3.5× less compute than NA²Q.", options: {} },
  ], { x: 5.3, y: 5.15, w: 7.43, h: 0.95, fontSize: 15 });
  tb(s, "Catch: it needs to replay episodes without some agents. Without that data it only reproduces the equal split.", { x: M, y: 6.15, w: W - 2 * M, h: 0.5, fontSize: 15, italic: true, color: C.accent3 });
  s.addNotes("This is the fix we proposed in our project proposal. We call it the episodic coalition estimator. It predicts the episode's total reward from which agents took part, including terms for groups of up to three agents. The key ingredient is the training data: normal episodes plus replays with random agents removed. Then we read off each group's value and split it among its members. On the layout where a team of three earns the reward, its group error is 0.43, where NA-squared-Q stays at the maximum 2.0, even when NA-squared-Q gets the same replay data. It finds the right team 96 percent of the time and beats the equal split in every setting. The catch: it needs the ability to replay without agents. Without that data, it only reproduces the equal split.");
}

// 12. Return vs credit
pres.addSection({ title: "Wrap-up" });
{
  const s = content("A higher team score does not mean better credit", "Wrap-up");
  const h = 5.15, w = h * 1700 / 1258;
  s.addImage({ path: path.join(FIG, "fig4_return_vs_credit.png"), x: M, y: 1.5, w, h, altText: "Return against credit accuracy at each checkpoint for QMIX, VDN, COMA and MAPPO", objectName: "fig4" });
  const x = M + w + 0.4, ww = W - M - x;
  stat(s, "ρ = 0.20", "correlation between team reward and credit accuracy for QMIX (0.15 for VDN): weak", x, 1.6, ww, C.accent1);
  tb(s, [
    { text: "QMIX earns twice VDN's reward ", options: {} },
    { text: "(6.5 vs 3.1)", options: { bold: true } },
    { text: " but its credit error is about the same ", options: {} },
    { text: "(0.64 vs 0.75)", options: { bold: true } },
    { text: ".", options: {} },
  ], { x, y: 3.7, w: ww, h: 1.1, fontSize: 15 });
  tb(s, "Here agents learn their own behaviour. Training from scratch never earned any reward (random play finishes the task 0 times in 20,000), so we seeded learning with scripted episodes. COMA and MAPPO collapsed, so they could not be assessed.", { x, y: 4.9, w: ww, h: 1.8, fontSize: 13, color: C.text2 });
  s.addNotes("Finally, does team reward tell you anything about credit quality? Here the agents learn their own behaviour, and we score credit at every checkpoint. The correlation between team reward and credit accuracy is weak: 0.20 for QMIX and 0.15 for VDN. QMIX earns about twice as much reward as VDN, but its credit error is about the same. One honest caveat: training from scratch never got any reward, because random play never completes the task, so we seeded the learners with scripted episodes. COMA and MAPPO collapsed during training, so we could only assess VDN and QMIX. This is a descriptive result, not a confirmatory one.");
}

// 13. Scorecard
{
  const s = content("Scorecard: what we predicted before running, and what we found", "Wrap-up");
  const hd = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 } } });
  const yes = (t) => ({ text: t, options: { bold: true, color: C.accent2 } });
  const no = (t) => ({ text: t, options: { bold: true, color: C.accent3 } });
  const mid = (t) => ({ text: t, options: { bold: true, color: "B07A10" } });
  const rows = [
    [hd("Prediction"), hd("Result"), hd("In plain words")],
    ["The benchmark gives a stable answer key", mid("Narrowly no; yes on re-check"), "104 of 120 settings passed; all 120 pass when the team can react to removals"],
    ["Early-agent credit shrinks as delay grows", no("Not supported"), "QMIX points the right way in every setting, but 5 seeds were too few (p = 0.074)"],
    ["Bystanders get more credit as more are added", mid("COMA yes; others weak"), "COMA 8 of 8 settings; VDN and QMIX right direction, too few seeds"],
    ["COMA leaks less than VDN and QMIX", no("Reversed"), "COMA leaks the most"],
    ["Good scores do not mean faithful credit", yes("Supported"), "30 of the 32 best settings"],
    ["Team reward is a weak guide to credit", yes("Consistent"), "Correlation 0.15 to 0.20 (descriptive)"],
    ["Our estimator recovers the team", yes("Supported"), "8 of 8 settings on group error; not on leakage"],
  ];
  s.addTable(rows, { x: M, y: 1.6, w: W - 2 * M, colW: [4.3, 2.9, 4.93], fontSize: 13, color: C.text1, border: { type: "solid", pt: 0.75, color: "D5D9E3" }, rowH: 0.56, valign: "middle", margin: 0.08, objectName: "scoretable" });
  tb(s, "Every threshold was fixed in writing before the main runs. Extra analyses done after seeing data are labelled as such in the report.", { x: M, y: 6.6, w: W - 2 * M, h: 0.5, fontSize: 13, italic: true, color: C.text2 });
  s.addNotes("Here is the honest scorecard. We wrote down our predictions and the pass thresholds before running the main experiments. Some held, some did not. The early-agent prediction failed the strict test: QMIX points the right way everywhere, but five seeds were not enough. COMA, which we expected to leak least because of its counterfactual baseline, actually leaks the most. The two predictions about faithfulness, and our estimator's group recovery, held. We think the failures are as informative as the successes, and each one comes with an explanation in the report.");
}

// 14. Limitations
{
  const s = content("What we cannot claim", "Wrap-up");
  const lim = [
    ["One environment family", "A small gridworld, up to 7 agents, 5 seeds per setting. Real tasks are messier."],
    ["Scripted data", "Most results train critics on fixed scripted play, to separate credit from exploration."],
    ["Our own reimplementations", "NA²Q and STAS were rewritten from the papers; the official code was compared but not run."],
    ["A strong assumption for our fix", "The estimator needs replays with agents removed, which real systems may not allow."],
  ];
  lim.forEach((l, i) => {
    const x = M + (i % 2) * 6.1, y = 1.75 + Math.floor(i / 2) * 2.35;
    card(s, x, y, 5.9, 2.1);
    agent(s, "!", x + 0.25, y + 0.25, 0.5, "fail");
    tb(s, l[0], { x: x + 0.95, y: y + 0.3, w: 4.7, h: 0.45, fontSize: 18, bold: true });
    tb(s, l[1], { x: x + 0.95, y: y + 0.85, w: 4.7, h: 1.1, fontSize: 15, color: C.text2 });
  });
  s.addNotes("Four limits on what we can claim. It is one environment family, small, with five seeds per setting. Most results use fixed scripted play rather than full training. Two of the methods are our own reimplementations, because the official code has no licence. And our estimator relies on replays with agents removed, which a real robot team or game may not allow.");
}

// 15. Conclusion
{
  const s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Wrap-up" });
  s.addText("Three things to take away", { placeholder: "title" });
  const pts = [
    ["Check credit against a known answer.", "A team's score cannot tell you whether credit is right. Planted environments with exact removal truth can."],
    ["Standard credit is a one-step what-if.", "It misses agents with slack and rarely beats an equal split, however well trained."],
    ["Team-level credit is recoverable.", "A small group-aware model finds the team, if you can replay episodes without some agents."],
  ];
  pts.forEach((p, i) => {
    const y = 1.75 + i * 1.5;
    agent(s, String(i + 1), M, y, 0.7, "coal");
    tb(s, p[0], { x: M + 1.0, y: y - 0.02, w: 11, h: 0.45, fontSize: 22, bold: true, color: C.background1 });
    tb(s, p[1], { x: M + 1.0, y: y + 0.5, w: 11, h: 0.7, fontSize: 16, color: "D6DAE6" });
  });
  tb(s, "Code, results and an interactive per-episode credit viewer (docs/credit_viz.html) are in the project repository.", { x: M, y: 6.3, w: 9, h: 0.4, fontSize: 13, color: "AEB4C8" });
  tb(s, "Questions?", { x: 9.8, y: 6.15, w: 2.93, h: 0.6, fontSize: 26, bold: true, color: HEX.accent1, align: "right", fontFace: "Cambria" });
  s.addNotes("To close, three things. One: if you want to know whether credit is right, you have to check it against a known answer; team score is not enough, and planted environments make that possible. Two: the credit standard methods compute is a one-step what-if, which misses agents with slack and rarely beats an equal split. Three: team-level credit can be recovered by a small group-aware model, as long as you can replay episodes without some agents. Thank you; we are happy to take questions.");
}

// Backup
pres.addSection({ title: "Backup" });
{
  const s = content("Backup: the main conclusions survive two robustness checks", "Backup");
  card(s, M, 1.7, 5.9, 4.6);
  tb(s, "Fair share vs loss-if-removed", { x: M + 0.3, y: 1.85, w: 5.3, h: 0.4, fontSize: 18, bold: true, color: C.accent2 });
  tb(s, [
    { text: "The two answers differ by at most 0.03 to 0.10 credit error", options: { bullet: true, breakLine: true } },
    { text: "Method rankings agree in 27 of 30 settings", options: { bullet: true, breakLine: true } },
    { text: "The best method is always the same; no verdict changes", options: { bullet: true } },
  ], { x: M + 0.3, y: 2.4, w: 5.3, h: 3.6, fontSize: 16, paraSpaceAfter: 10 });
  card(s, 6.83, 1.7, 5.9, 4.6);
  tb(s, "Replay as recorded vs let the team react", { x: 7.13, y: 1.85, w: 5.3, h: 0.4, fontSize: 18, bold: true, color: C.accent2 });
  tb(s, [
    { text: "When the remaining agents can react to a removal, the planted team is recovered in 100% of 3,600 test episodes (83% without)", options: { bullet: true, breakLine: true } },
    { text: "The answer-key prediction then passes in all 120 settings", options: { bullet: true, breakLine: true } },
    { text: "Early-agent prediction becomes borderline (one QMIX setting passes, p = 0.030), so we call it borderline, not a firm negative", options: { bullet: true } },
  ], { x: 7.13, y: 2.4, w: 5.3, h: 3.6, fontSize: 16, paraSpaceAfter: 10 });
  s.addNotes("Backup for questions about robustness. Using Shapley shares instead of leave-one-out changes nothing important. Letting the team react to removals, rather than replaying recorded actions, fixes the few answer-key failures and makes one early-agent setting pass. We keep the pre-registered verdicts as the main ones and report these as sensitivity results.");
}
{
  const s = content("Backup: problems the pilots exposed, and how we fixed them", "Backup");
  const rows = [
    [{ text: "Problem", options: { bold: true, color: C.background1, fill: { color: C.text1 } } }, { text: "Fix", options: { bold: true, color: C.background1, fill: { color: C.text1 } } }],
    ["Under delay the critic could not tell when the reward would arrive", "Added a payout countdown to the critic's view (not the agents'). Fixed QMIX, COMA, MAPPO; VDN cannot see it"],
    ["NA²Q's original learning target blew up on fixed data (value 115–140 against a true 8)", "Used a target that follows the recorded behaviour (SARSA); it converges to about 8.3"],
    ["One-step credit is not removal credit", "Graded in two levels: fidelity to its own target, and the meaning gap"],
    ["MAPPO gives every agent the same advantage, so its credit is ±1/n by construction", "Reported as reference only; not used as evidence"],
    ["5 seeds make the planned test unable to reach significance after correction", "Switched to a paired t-test before looking at results; amendment committed"],
  ];
  s.addTable(rows, { x: M, y: 1.6, w: W - 2 * M, colW: [5.6, 6.53], fontSize: 14, color: C.text1, border: { type: "solid", pt: 0.75, color: "D5D9E3" }, rowH: 0.75, valign: "middle", margin: 0.1, objectName: "fixtable" });
  s.addNotes("Backup for method questions: the problems two pilot runs exposed, and how each was fixed before the main runs.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
