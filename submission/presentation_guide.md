# Presentation guide: slide by slide

For each slide this covers what is on screen, what it means, the numbers and where they come from, the questions you
might get, and the exact words to say. The words match the speaker notes in `AFML_Final_Presentation.pptx` and
`video_script.md`, and the timings fit inside 7 minutes (about 6:10 spoken, 6:30–6:45 with pauses).

## The story in five sentences

1. When a team of AI agents shares one reward, someone has to work out **who earned it** (credit assignment).
2. Nobody checks whether popular methods get this right, because real tasks have no answer key. **We built games that
   have one.**
3. Checked against it, the popular methods are **mostly worse than just splitting the reward equally**.
4. The reason: they ask **"what if this agent paused for one step?"** when the real question is **"what if this agent
   weren't there at all?"**
5. **Our fix:** during training, let random agents **sit out** for the rest of the game, and train a critic on that.
   It then gives the right credit almost exactly (error under 0.07, against 0.5–2.0 for the others).

## Time plan

| Part | Slides | Time | Suggested speaker |
|---|---|---|---|
| Setup | 1–6 | 0:00–1:37 | Manaswin |
| Existing methods fail | 7–8 | 1:37–2:39 | Manaswin (7), Bhuvan (8) |
| Our fix, results, demo | 9–10 + demo | 2:39–5:12 | Bhuvan |
| Generalisation | 11 | 5:12–5:31 | Bhuvan |
| Admin | 12–17 | 5:31–6:10 | Harshitha (12–14), Harshith (15–17) |

---

## Recording in four takes (use this for the video)

You record four separate clips in Clipchamp, then put them in order on its timeline and export once. The scripts
below replace the per-slide "Say" lines further down. The demo is woven into slides 6 and 10, so there is no separate
60-second demo block.

| Take | Slides | Target | Words to speak |
|---|---|---|---|
| 1 | 1 (intro) | 0:25 | ~60 |
| 2 | 2–5 (basics, terms, what's new, data) | 2:10 | ~310 |
| 3 | 6–11 (game, failure, fix, results + demo, generalisation) | 3:05 | ~400 + clicks |
| 4 | 12–17 (status, team, quality, lessons, challenges, refs) | 0:55 | ~110 + team lines |
| | **Total** | **≈ 6:35** | leaves ~25 s of margin under 7:00 |

Planning exactly 30 s + 2:30 + 3:00 + 1:00 lands at 7:00 with no margin, and take 3 runs long in practice. That is
why take 2 is trimmed and the spare time goes to take 3.

### Screen setup (once, before take 1)

1. **Don't record PowerPoint's edit view.** It shows the ribbon, the thumbnails and "Product Activation Failed" in
   the title bar. Instead go to **Slide Show → Set Up Slide Show → Show type: "Browsed by an individual (window)" → OK**,
   then press **F5**. The slides play in a plain window you can dock on the right half, with no ribbon. Use the arrow
   keys or Page Down to advance.
2. **Browser on the left half:** open `docs/credit_viz.html`. Press **Ctrl+Shift+B** to hide the bookmarks bar, and
   zoom to about **80%** (Ctrl and −) so the replay and heatmaps fit. Set it to **delay 25 · slack · test episode #1**.
3. Turn on **Do Not Disturb** in Windows quick settings, and close WhatsApp and Spotify.
4. In Clipchamp, record **Full screen** for each take, and stop and start between takes.

### Take 1: slide 1 (≈ 25 s)

> Hi, we are Team Aporia: Daruru Bhuvan Sai, H K Manaswin, Harshitha B T and Harshith R, from PES University. Our
> project is called "Who Earned the Reward?". It checks whether teams of learning agents give credit to the agents who
> actually earned a shared reward, and when they don't, it fixes that.

### Take 2: slides 2–5 (≈ 2:10)

**Slide 2 (≈ 45 s)**
> First, the basics. In reinforcement learning, an agent learns by trial and error. It takes an action, gets points,
> called a reward, and over many tries it does more of what earned points. Now put several agents in one team that
> shares a single reward. It's like a group project with one grade: often only some members did the work, but everyone
> gets the same mark. If each agent learns from that shared mark, the ones who did nothing are told they did well.
> Deciding who actually earned the reward is called credit assignment. Our question is simple: do popular methods give
> the credit to the right agents?

**Slide 3 (≈ 45 s)**
> A few words we'll use. An episode is one play-through of the game. A delayed reward arrives some time after the
> useful actions, which makes it hard to tell who caused it. Idle agents move around but help with nothing, so they
> should get zero credit. A critic is a neural network that predicts how many points the team will end up with; most
> methods read each agent's credit off it. The equal split gives everyone the same share. It needs no learning, so any
> method should beat it. And credit error measures how far a method's split is from the true split: zero is exactly
> right, two is completely wrong.

**Slide 4 (≈ 20 s)**
> What's new? Methods are normally judged only by the team's score, because real tasks have no answer key. But a team
> can score perfectly while giving credit to the wrong agents. So we built games with an answer key, tested six popular
> methods against it, and found a fix.

**Slide 5 (≈ 20 s)**
> There's no downloaded dataset. In reinforcement learning the data comes from playing a game in a simulator. We use
> our own gridworld and a public benchmark, Level-Based Foraging. A fixed dataset wouldn't work, because finding the
> true credit means re-playing each game with agents taken out.

### Take 3: slides 6–11 with the demo (≈ 3:05)

Lines in *[brackets]* are actions in the browser, not words to say.

**Slide 6 (≈ 30 s)**
> Here's our game. *[Browser: press Play on the episode replay.]* A, in blue, pulls a lever that opens a gate. Later B
> and C, in orange, must push a crate together, and the ten points arrive even later. D and E, in grey, just wander. So
> A, B and C each earned a third. To get the true credit, we re-play the same game with each agent taken out. Without
> A, B or C the team scores nothing; without D or E it still scores ten. True credit is the reward lost without you.

**Slide 7 (≈ 20 s)**
> How do popular methods do? Up means more wrong, and to the right the reward arrives later. The dashed line is the
> equal split. Only 5 of 96 combinations beat simply splitting the reward equally, and it gets worse with delay and
> with more idle agents.

**Slide 8 (≈ 28 s)**
> Why? They ask the wrong question. Their credit means: what if A paused for one step? A opened the gate early, so it
> would just open it a step later. Nothing is lost, so A gets almost no credit. The real question is: what if A weren't
> there at all? Then the gate stays shut and all ten points are lost, so A earned a third.

**Slide 9 (≈ 40 s)**
> Our fix gives the critic the right question and the data to answer it. Like a coach benching a player in practice:
> while collecting training games, a random agent or group sits out for the rest of the game. The critic learns how
> many points the team will still get, given who is still playing. An agent's credit is points with you minus points
> without you. Before A pulls the lever, the team without A gets nothing; right after, it no longer needs A. That jump
> is A's credit. Both pieces matter: without the sitting-out games, or with the old one-step question, it fails.

**Slide 10 + demo (≈ 47 s)**
> Does it work? Across all twelve settings, existing methods have an error between 0.5 and 2. Ours is below 0.07 and
> finds the right team every time. Here is one real game. *[Browser: click QMIX, scroll to "Per-step credit".]* The top
> row is QMIX, the bottom is the true credit. They don't match, and D and E get credit. *[Click "Ours: trained with
> sitting out".]* Now the top matches the bottom: A's credit lands at the lever pull, B and C's at the push, and nothing
> for D and E.

**Slide 11 (≈ 20 s)**
> It also generalises: near exact when starting positions change every game, and the right team at team sizes it never
> trained on. In a standard foraging game it clearly beats the equal split with four agents; with six it is better, but
> far from exact.

If take 3 runs past 3:15, cut the last sentence of slide 9.

### Take 4: slides 12–17 (≈ 55 s)

> **12:** Everything we planned is done: the game and its true credit, the test of six methods, our fix, and the
> generalisation tests. Next are bigger games, and settings where agents can't sit out.
> **13:** *[Each member, one short sentence about what they did.]*
> **14:** Every part runs end to end, with tests, and all results are in the repository.
> **15:** Our main lessons: a team's score can't tell you whether credit is fair, and the right question and the right
> data beat a fancier model.
> **16:** Open problems: bigger games, settings where agents can't sit out, and luck in a single game, which no critic
> can attribute.
> **17:** These are our references. Thank you.

### Joining the takes

In Clipchamp, drag the four recordings onto the timeline in order and trim the dead air at each start and end (the
moment you click Stop). Check that the total under the timeline is **below 7:00**, then **Export → 1080p**.

---

## Slide 1: Title

**On screen:** course code, project title, the four SRNs and names.

**Say** [0:00–0:07]
> Hi, we are Team Aporia: Bhuvan, Manaswin, Harshitha and Harshith. Our project: Who Earned the Reward?

---

## Slide 2: Topic

**On screen:** four bullets (reinforcement learning, many agents with one reward, credit assignment, our question)
plus a picture.

**What it means**
- **Reinforcement learning (RL):** an agent tries actions, gets points (the *reward*), and gradually does more of
  what earned points. No one shows it the right answer; it learns from the score.
- **Multi-agent RL (MARL):** several agents act in the same game. In a *cooperative* game they all receive the
  **same** team reward.
- **The problem:** if five agents all receive +10, each one "thinks" its own actions earned it, including the ones that
  did nothing. That is like a group project with one grade. If the method can't tell who earned it, agents learn the
  wrong lessons.
- **Credit assignment** means splitting the shared reward into how much each agent contributed.

**Possible question:** *Why does it matter?* Because each agent learns from its own credit. Wrong credit means slow
learning, or agents rewarded for doing nothing.

**Say** [0:07–0:31]
> In reinforcement learning, an agent learns by trial and error: it acts, gets points, and does more of what earned
> points. A team of agents often shares one reward, like a group project with one grade, though only some members did
> the work. Working out who earned it is called credit assignment. Do popular methods get it right?

---

## Slide 3: Key ideas in plain English

**On screen:** six definitions.

**What it means** (the vocabulary for the rest of the talk)

| Term | Meaning | Why it matters later |
|---|---|---|
| Episode | One play-through, start to finish | Credit is measured per episode |
| Delayed reward | The points arrive steps after the useful actions | The delay is what breaks existing methods |
| Idle agents | Agents that move but can't help | Their true credit is **0**; a good method gives them 0 |
| Critic | A neural network predicting the points still to come | Most methods read credit off a critic, and so do we |
| Equal split | Everyone gets the same share | The no-learning baseline: a method must at least beat it |
| Credit error | Distance between a method's split and the true split: 0 = exact, 2 = completely wrong | The y-axis of every results chart |

**How credit error is computed:** turn each method's credit into **shares** that add up to 1 (e.g. A 0.33, B 0.33,
C 0.33, D 0, E 0), then add up the absolute differences from the true shares. If the truth is a third each for A, B, C
and a method gives the whole reward to D, the error is 2. An equal split with 2 idle agents (0.2 each) gives
0.8, which is why "0.80" appears everywhere as the equal-split line.

**Say** [0:31–0:53]
> Some words we'll use. Idle agents help with nothing, so they should get zero credit. A critic is a neural network
> that predicts how many points the team will get. The equal split is the baseline to beat, and credit error measures
> how far a method is from the truth: zero is perfect.

---

## Slide 4: Uniqueness

**On screen:** what is new.

**What it means**
- Papers judge credit methods by **team score**. But a team can score fully with terrible credit (in our game the
  team always collects its +10 no matter who gets credited), so score says nothing about credit.
- Our contribution is an **answer key**: games where we know exactly who earned what.
- We tested **six** methods: VDN, QMIX, COMA, MAPPO (the standard ones, via the EPyMARL library), NA²Q and STAS
  (newer ones that claim to explain credit, rebuilt from their papers).
- Then the **fix**.

**Possible question:** *Why are none of the six "wrong" in their own papers?* None of them claims to output true
credit; they are judged by score. We are the first to grade their credit read-outs directly.

**Say** [0:53–1:04]
> Methods are judged only by team score, because real tasks have no answer key. So we built games with one, tested
> six popular methods, and found a fix.

---

## Slide 5: Dataset

**On screen:** why there is no downloaded dataset, the two games, and the size.

**What it means**
- In RL the data **is** the game: a simulator produces it. So there's no dataset to download.
- **Planted Coalition Gridworld** (ours): a 5×10 grid with 3 to 7 agents. Reward delay of 0, 25 or 50 steps; 0, 2 or 4
  idle agents. "Planted" means we decide who the team that earns the reward is.
- **Level-Based Foraging (LBF):** a public benchmark where agents with skill levels must team up to lift food. We
  re-implemented it and checked it step for step against the official package (`lbforaging` 2.0.0). It gets a new
  random map every game.
- **Why not recorded data:** the truth needs **re-playing the same game with agents taken out**, using the same random
  numbers. Only a simulator allows that.
- Size: 120 settings × 5 runs in the main experiment, about 2,500 result files, all in the repo.

**Say** [1:04–1:13]
> There's no downloaded dataset: the data comes from simulated games, because finding the truth means re-playing games
> with agents taken out.

---

## Slide 6: Overall design (diagram)

**On screen:** the running example and how the truth is computed.

**What it means.** This is the slide everything else hangs on, so know it well.
- **The game:** agent **A** (the *enabler*) pulls a lever that opens a gate. Later, **B and C** (the *finishers*)
  must push a crate **together**. The +10 arrives **25 steps after** the push. **D and E** wander and can't help.
- **True credit = what the team loses without you.** We re-play the same game (same random numbers) with one agent
  stopped from the start:
  - without A: the gate never opens, so 0 points. A's credit is 10.
  - without B (or C): nobody can push the crate alone, so 0 points. B's and C's credit is 10 each.
  - without D (or E): still 10 points. Their credit is 0.
  - Shares: **A, B, C a third each; D, E zero.**
- Teammates may **react** in the re-play (e.g. finishers wait for a gate that never opens). That is the "team reacts",
  closed-loop truth, and it is exact.
- We also compute each agent's credit **per step**: it lands when the outcome becomes certain (A's at the lever pull,
  B and C's at the push), **not** when the delayed reward is paid.

**Possible questions**
- *Why "stopped" and not "deleted"?* Both are tested. *No-op*: the agent stands still. *Removal*: it leaves the
  board. The main grid uses no-op; LBF uses removal.
- *What about the scripted team?* The agents follow a near-optimal script with 20% random actions, so the truth is
  exact for that team. That is a stated limitation.

**Say** [1:13–1:37]
> Our example: A pulls a lever that opens a gate, later B and C push a crate together, and the points come even later.
> D and E just wander. So A, B and C each earned a third. True credit comes from re-playing each game with agents
> taken out: it is how much reward is lost without you.

---

## Slide 7: Final results so far (existing methods)

**On screen:** credit-error chart, with panels by number of idle agents, reward delay on the x-axis and a dashed
equal-split line.

**What it means**
- **Up means more wrong.** Left to right, the reward arrives later.
- **With no idle agents the equal split is exactly right** (3 workers, a third each), so its error is 0. Every
  method is still above zero: they do worse than doing nothing clever.
- **Only 5 of the 96 method-and-setting combinations beat the equal split.** One of the 5 is MAPPO's glitch (below).
- Credit gets worse with **more delay** and **more idle agents**.
- Main-grid ranges (v3 confirmatory, 12 settings): VDN 0.71–1.72, QMIX 0.50–1.42, COMA 1.45–1.62. The equal split
  scores 0.80 (2 idle) and 1.14 (4 idle). VDN and QMIX beat it only when there is **no delay**; with any delay, never.

**Possible questions**
- *Isn't this just bad training?* No. Their per-step credit barely correlates even with **their own** target
  (r ≤ 0.19), and a **perfect** one-step critic would still be off by about 1.2 (next slide). More data didn't
  help (tested with 4× and 16× data).
- *MAPPO?* MAPPO's advantage is the same number for every agent, so it can only produce an equal split. With the
  sign flipped it scores 2.00, which is a glitch, not a result (erratum E11). A corrected read-out scores 0.38–0.85.

**Say** [1:37–2:04]
> So how do the popular methods do? Up means more wrong. Left to right, the reward arrives later. The dashed line is
> the equal split. With no idle agents it is exactly right, at zero, yet every method is well above it. Overall only
> 5 of 96 combinations beat just splitting the reward equally, and it gets worse with delay and with more idle agents.

---

## Slide 8: Why existing methods fail: the wrong question

**On screen:** two panels: "What if A paused for one step?" (red) against "What if A weren't there at all?" (green).
Plus a bullet on teamwork.

**What it means.** This is the core idea of the project.
- Existing methods compute credit with a **one-step counterfactual**: swap agent A's action at **one** step for "do
  nothing", let the team carry on as usual, and see how much the predicted value drops.
  - VDN/QMIX: the change in the team's value when A's action becomes a no-op.
  - COMA: A's action value minus the average over A's other actions.
- **Why that fails:** if A pauses for one step, A just pulls the lever one step later. The team still gets +10.
  Almost nothing is lost, so **A gets almost no credit**, even though A was essential.
- **The right question is persistent:** if A is gone for good, the gate never opens and all 10 points are lost, so A
  earned a third.
- **Proof that this is about the question, not the critic:** we computed the one-step counterfactual **exactly**
  with the simulator (a perfect critic). It is still off by **1.22** when A acts early and **0.91** when A acts late,
  and it misses **99%** of an early enabler's credit. A better critic cannot close that gap.
- **Teamwork bullet:** B and C only score **together**, so the value sits in the *group* {B, C} (with A). VDN, QMIX and
  NA²Q add up per-agent values, so they put **0.00** of the value on groups (truth: 1.0). Our first method, a
  coalition estimator trained on replays with agents removed, finds the groups (0.78–0.97).
  - Be honest if asked: a simple table on the same replay data does about as well. The gain comes from the
    **data with agents removed**, which is the same lesson as our main fix.

**Possible question:** *Then why do VDN/QMIX sometimes get it right with no delay?* With no delay, the reward lands
right after the push, so a one-step change can occasionally line up with the truth. QMIX was "right for the wrong
reason" in 77% of no-delay, early-enabler games.

**Say** [2:04–2:39]
> Why do they fail? They ask the wrong question. Their credit means: what if A paused for one step? A opened the gate
> early, so it would just open it a step later. Nothing is lost, so A gets almost no credit. The question that matters
> is: what if A weren't there at all? Then the gate stays shut and all ten points are lost, so A earned a third. They
> are also blind to teamwork; our first method, a coalition estimator, does find the groups.

---

## Slide 9: Our fix: credit by intervention

**On screen:** a 3-step diagram: (1) practice games where someone sits out, (2) train a critic, (3) credit = with you
minus without you. Plus the coach analogy and "both pieces are needed".

**What it means, step by step**
1. **Collect data with interventions.** 200 normal games plus 200 games where, from a random moment, a random agent
   or group **stops for the rest of the game**. Half are single agents, half groups; half stop before the crate push,
   half during the payout wait. The game state includes an **"acting" flag per agent**, so "A has stopped" is just a
   different state.
2. **Train an ordinary critic** to predict the points still to come from each state, including who is still acting.
   Plain regression on actual returns (Adam, 3,000 steps). Nothing exotic.
3. **Read out credit on normal test games** (no one stops):
   - ask the critic "points to come **with** everyone" and "points to come if A's flag were **off**";
   - A's credit at step t = reward at t + (value without A next step) − (value without A now). Summed over the
     game, this is exactly *"points with A − points without A"*, the true removal credit.
- **Why it fixes A:** before the lever pull, "without A" predicts 0, because the team needs A completely. Right after
  the pull, "without A" predicts 10, because the gate is already open. That jump is A's credit, landing **exactly
  at the pull**, whether A acted early or late.
- **Two versions:**
  - **MLP**, a plain network: error 0.012–0.063.
  - **Set critic**, which treats agents alike: one shared encoder per agent plus attention, so relabelling the agents
    can't change the answer. Error 0.002–0.026, and it is the version that generalises (slide 11).
- **Both pieces are needed:**
  - Same critic **without** the sit-out games: error 0.85–1.50, worse than the equal split. It has never seen a
    state where someone stopped, so its guess is arbitrary.
  - MAPPO trained **on the same sit-out games** but read out the old one-step way: still 2.00. Read the same MAPPO
    critic our way: 0.41/0.71, which is better but still far from ours. **Data and question together** do it.

**Possible questions**
- *Is this new?* The idea of "reward with you minus without you" is Wolpert & Tumer's *difference rewards* (2001).
  What's new is the evidence: an exact benchmark, why one-step read-outs miss it, and the finding that a plain critic
  **learns** it once the data contains stops.
- *How much sit-out data?* Modest. With 100 stops the error is already 0.02 (2 idle agents); 200 is the default.
- *Can you do this in the real world?* Only where agents can be made to stop during data collection (a simulator).
  That is our main limitation (slide 16).

**Say** [2:39–3:40]
> So we make the methods ask the right question, and give them the data to answer it. Like a coach benching a player
> in practice: while collecting training games, a random agent or group sometimes sits out the rest of the game. A
> neural network, the critic, learns how many points the team will still get, given who is still playing. Then an
> agent's credit is simply points with you minus points without you. Why does this fix A? Before A pulls the lever, if
> A stopped the team would get nothing, so the team needs A completely. The moment A pulls it, stopping A no longer
> matters. That change is A's credit, whether or not A had slack. Both pieces matter: without the sitting-out games
> the same critic fails, and an existing method given the same games but asking its old one-step question still
> fails.

---

## Slide 10: Our fix gets the credit right, then the demo

**On screen:** left, one real test game: A's credit share over time for ours (blue), QMIX (orange) and the truth
(dashed). Right, error ranges for every method across the 12 main settings.

**What it means**
- **Left:** the blue line jumps to a third **at the step A pulls the lever** and stays there; the +10 arriving later
  moves nothing. QMIX's line never reaches the truth.
- **Right:** every existing method is between 0.5 and 2.0. Ours is **below 0.07 in every setting**: MLP 0.012–0.063,
  set critic 0.002–0.026. It ranks exactly A, B, C on top in **all 1,800 test games** (12 settings × 5 runs × 30
  games). The per-step credit correlates 0.97 (MLP) and 0.996 (set critic) with the exact per-step truth.

**Say** [3:40–4:12]
> Does it work? On the left, one real test game. The blue line is the credit our method gives A over time. It jumps to
> exactly a third at the step A pulls the lever, because after that the team no longer needs A. QMIX, in orange,
> never gets close. On the right, across all twelve settings: every existing method has an error between 0.5 and 2.
> Ours is below 0.07, and finds the right team every time.

### Demo [4:12–5:12], 60 seconds

Have `docs/credit_viz.html` open **before recording**, already set to **delay 25 · enabler slack · test episode #1**.

| # | Click | Say |
|---|---|---|
| 1 | Method **QMIX**, panel "Who got the credit" | "Grey is the truth. QMIX gives D and E credit and A very little." |
| 2 | Method **Ours: trained with sitting out** | "Our bars sit on the truth: a third each for A, B and C, nothing for D and E." |
| 3 | Point at the **per-step heatmap** | "And A's credit lands exactly when it pulls the lever." |
| 4 | Press **Play** on the episode replay | (let the agents move for a few seconds) |
| 5 | Method **Ours: coalition estimator**, point at the coalition panel | "It finds the group A, B, C." |
| 6 | Scroll to the **table** at the bottom | "On all 30 test games, ours has the lowest error." |

If you're running long, stop after step 3.

---

## Slide 11: Our fix generalises, with limits

**On screen:** three bullets and the LBF chart.

**What it means.** A fixed layout could be memorised, so we tested configurations the critic never saw:
- **New start positions (and roles) every game:** the set critic stays near exact, 0.018 (2 idle) and 0.030 (4 idle),
  against 0.52/0.96 for the plain MLP. The MLP improves with 4× data (0.21/0.58), so it is short on symmetry, not
  capacity.
- **Unseen team sizes:** trained on several sizes, it picks exactly the right team at new sizes. Between trained
  sizes the error is 0.008/0.014, as good as training at that size. One or two agents beyond the largest trained size
  it is 0.06/0.14. Trained on one size only, it leaks 9–46% of the credit to unfamiliar bystanders.
- **Level-Based Foraging** (new random map every game, 4 agents): 0.26 (0.16 with 4× data) against 0.42 for the
  equal split. Without sit-out games it is worse than the equal split again (1.02/1.34), the same pattern. With 6
  agents it is better than the equal split (0.87/0.81 vs 1.65) but far from exact.
- **Honest misses:** two pre-registered LBF predictions (error below 0.25, and improvement with more data) were not
  significant (p = 0.10 and 0.12): four runs were at 0.07–0.13, one at 0.40.

**Possible question:** *Why not "shuffled roles" as generalisation?* An external review pointed out that a critic
that treats agents alike passes that test **by construction**, so we retracted it (erratum E12) and ran these real
tests instead.

**Say** [5:12–5:31]
> It also generalises: near exact when starting positions change every game, and the right team at team sizes it never
> trained on. In a standard foraging game it clearly beats the equal split with four agents; with six it is better,
> but far from exact.

---

## Slide 12: Done vs remaining

**On screen:** 7-row table; rows 1–6 done, row 7 (bigger games, settings without sitting out) to be done.

**Background on row 4 (learning from scratch):** we also asked whether *accurate* credit makes a team **learn
faster**. Mostly not: average credit used as a reward is mathematically equivalent to reward shaping, and with a critic
that sees the whole state it changed learning speed by a factor of only 1.05. What helps is **actual difference
rewards** (re-simulating without each agent at every step): 12–26× fewer games with 8 parallel tasks, at the cost of
17 simulator copies per step.

**Say** [5:31–5:34]
> Everything planned is done; bigger games are next.

---

## Slide 13: Team member contribution

**On screen:** names, SRNs, tasks. **Harshitha's and Harshith's rows still say "[fill in]"**: send me their tasks
and I'll rebuild the deck.

**Say** [5:34–5:54]: each member says one short sentence about what they did.

---

## Slide 14: Quantity and quality of work

**On screen:** 9 components, all 100% and running. Notes: MAPPO's standard credit is unusable (a corrected read-out
is used); one slow STAS test skipped in quick runs; COMA and MAPPO failed to learn from scratch; stored results must be
unzipped before analysis.

**Say** [5:54–5:57]
> Every part runs end to end, with tests.

---

## Slide 15: Top learning points

**On screen:** four lessons: score can't verify credit; ask the right question; the right data beats a fancier model;
write predictions first and report failures.

**Say** [5:57–6:03]
> Our main lesson: ask the right question, and the right data beats a fancier model.

---

## Slide 16: Top unresolved challenges

**On screen:** small games only; needs agents that can sit out; COMA/MAPPO didn't learn from scratch; luck in a single
game can't be attributed; 6-agent foraging far from exact.

**Notes**
- Row 1 says "up to 7 agents". The team-size test went up to **8**, so if asked, say "at most eight agents".
- "Luck" means a critic learns **average** credit. If an idle agent's random move happened to matter in one game, only
  a real re-play can tell; no critic can.

**Say** [6:03–6:08]
> Open problems: bigger games, and settings where agents can't sit out.

---

## Slide 17: References

**On screen:** VDN, QMIX, COMA, MAPPO, NA²Q, STAS, EPyMARL/LBF.

**Say** [6:08–6:10]
> These are our references. Thank you.

---

## Quick numbers sheet (memorise these)

| What | Number |
|---|---|
| Equal split error | 0.80 (2 idle), 1.14 (4 idle) |
| Existing methods | 0.5–2.0; only 5 of 96 beat the equal split |
| Perfect one-step critic | still 1.22 off (early enabler), 0.91 (late) |
| Ours, plain critic | 0.012–0.063 |
| Ours, set critic | 0.002–0.026 |
| Ours without sit-out data | 0.85–1.50 |
| Right team picked | 100% of 1,800 test games |
| Random starts (set critic) | 0.018 / 0.030 |
| Foraging, 4 agents | 0.26 (0.16 with 4× data) vs 0.42 equal split |
| Learning speed from average credit | ×1.05 (no real gain) |
| Difference rewards, 8 tasks | 12× fewer games (MAPPO), 26× (IPPO), at 17 simulators per step |
