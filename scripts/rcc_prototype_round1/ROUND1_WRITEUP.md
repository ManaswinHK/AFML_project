# Who Earned the Reward? Review, new tests, and the plan from here

**Date:** 10 November 2026 · **Repo reviewed:** `DBS13062006/AFML_project` at commit `5943480` · **Reviewer:** Claude

**What this covers.** I read the one-pager, `who_earned_the_reward_problem_statement_v2.md`, every file in `docs/`, the source in `src/whoearned/`, and the stored results in `results/mode_p/`. I ran the 131 unit tests that do not need PyTorch (all pass). I then ran seven new tests on your environment, with your seeds, layouts and metrics. The scripts and raw outputs are in the attached bundle.

**What it does not cover.** PyTorch would not install in my sandbox, so I did not re-run VDN, QMIX, COMA, MAPPO, NA²Q or STAS. Every number for those methods below is read from your stored JSON files. My own estimator is a scikit-learn prototype, not an EPyMARL module. I did not read problem statement v1. Papers I could only read as abstracts or search snippets are listed in §9.

Everything I ran today is **exploratory**: it uses seeds 0–4, the same seeds as your runs, with no pre-registration. §8 gives the pre-registration for a confirmatory repeat.

---

## 1. Bottom line

**Is the work so far useful?** Partly. The measuring instrument is good and one finding is real. But you do not yet have a positive result, and the one-pager's central claim ("wrong credit trains wrong behaviour") is untested. Of your seven confirmatory verdicts, one is a clean pass (H5), one is a partial pass (H2, COMA only), and five are fails or uninformative (H0, H1a, H1b, H2′, H3). The useful findings all came post hoc.

**What I found today.**

| # | Test | Result |
|---|---|---|
| 1 | Uniform-credit baseline on your main grid (stored results, no retraining) | VDN, QMIX and COMA beat an equal split in **4 of 72** conditions. With no bystanders, the equal split is exactly right (L1 = 0.00) and those three methods are 0.51–1.53 off. |
| 2 | Why per-step credit fails | It is partly an identity, not an empirical accident. With a perfect critic, COMA's expected episode credit is **zero for every agent**. Checked numerically: −0.02, −0.01, +0.07, 0.00, 0.00 (all within about one standard error of 0). |
| 3 | A per-step credit whose counterfactual *persists* (the agent is removed for the rest of the episode, not for one step) | The "semantic gap" closes. Missed enabler credit falls from 0.99 (your one-step value, slack cells) to **0.00**. The enabler is credited at the step it pulls the lever. |
| 4 | A critic trained on data that contains removal interventions | It estimates that persistent credit with per-step correlation **0.83–0.97** on the single-coalition cells (your methods: ≤ 0.36 against their own targets). Credit error L1 is **0.09–0.26** against 0.71–1.03 for uniform and 0.44–1.51 for your best stored method. The same critic without interventions scores 0.86–1.44. |
| 5 | Coalition level, overlap and partial-synergy layouts | Interaction mass **0.80–0.97** (truth 1.0; your methods 0.000–0.002). Dividend error ≈ 1.0 at 200 episodes (your methods 2.0, the maximum), falling to 0.3–0.5 at 400–800 episodes on the single-coalition and overlap layouts and to 0.8 on the partial layout. Not solved, but no longer blind. |
| 6 | Two problems in the current setup | (a) In the partial-synergy layout the critic's state is not Markov under delay: it omits the pending payout *amount*. Adding it halves the value error at d = 25. (b) "The per-step game can only be additive" is slightly overstated: a perfect critic's per-step game has interaction mass 0.26–0.35. |
| 7 | Does exact credit speed up learning? (toy game) | With one coalition: **no difference at all** (215 vs 215 rounds). With 8 independent events: 3× faster. With 16: team reward fails in 30,000 rounds, exact credit solves it in about 1,500. |

**What I recommend.**

1. Reframe the project as **credit by intervention**: one-step read-outs cannot identify who earned the reward; removal interventions in the training data can, with no new architecture.
2. Fix the three setup problems in §7 Phase 0 before running anything else.
3. Repeat tests 3–5 inside the repo on fresh seeds under a new pre-registration (§8). That is your positive result.
4. **Do not run Mode T on the current single-coalition layout.** Exact credit provably cannot help there (§4, argument 5), so H4 would come out "true" for a trivial reason.
5. Run the headroom test (§7 Phase 2) on a multi-event layout first. It decides whether there is a learning story or only a measurement-and-estimation story. I do not know which way it will go in your gridworld.

One thing I want to be plain about: the estimator in test 4 is good at *attribution*. I do not expect its credit, used as a reward, to speed up learning. §4 argument 6 gives the reason. The learning benefit, if there is one, comes from a different quantity (difference rewards), which your simulator can compute exactly.

---

## 2. What the work so far is worth

### 2.1 What is solid

- **The instrument.** Deterministic environment, exact replay for all 2ⁿ coalitions, per-event ground truth, unit tests, realised-versus-planted reporting. My regenerated test episodes reproduce yours action for action (30 of 30 in the cell I checked), and my uniform-baseline and semantic-gap numbers match your documents.
- **Two-level grading.** Separating "is the critic right about its own target" from "is the target the right notion" is the best idea in the project. Most credit-assignment papers conflate the two.
- **The discipline.** A pre-registration, an amendment written before looking at results, and honest labels on post-hoc work.
- **The finding.** Per-step read-outs give a slack enabler no credit at any delay. That is correct and was not obvious to you going in (v2 predicted delay would be the cause).

### 2.2 What is weaker than the summary says

| # | Issue | Evidence | What to do |
|---|---|---|---|
| 1 | The read-outs are your construction. VDN, QMIX and COMA never claimed to output removal credit. | v2 §2.4 defines the read-outs. | Claim "no standard read-out of these methods recovers removal credit", not "the methods assign credit wrongly". |
| 2 | Part of the headline is a mathematical identity. | §4 arguments 1–2. Summed advantages equal the return surprise. COMA's expected credit is zero for everyone. | State these as propositions and cite the action-gap literature (§10). The contribution is measuring it and fixing it, not discovering it. |
| 3 | The training data has no outcome variance. | `train_return_mean` is exactly 10.0 in all 120 main-grid cells and all 15 overlap cells. No training episode ever shows what happens when a contribution is missing. | This is the root cause of several results (below). Interventions fix it. Also run the coverage control in §6. |
| 4 | The STAS conclusion does not follow. | With a constant return of 10 in every training episode, any return-decomposition model has a zero-loss solution that gives every agent the same amount. Your synthetic check (25–27%) is the only evidence about STAS itself. The port was compared line by line but never executed against the official code. | Say "our reimplementation, on constant-return data". Either validate the port on STAS's own Alice-and-Bob task or drop the general claim. |
| 5 | No method was compared with the trivial baseline on the main grid. | Test 1: 4 of 72. | Report uniform next to every L1. H5's "30 of 32 best cells have step r < 0.3" needs rereading: most of those "best" cells are worse than an equal split. |
| 6 | Nothing shows that credit matters for learning. | Mode T not run. In the main layout it cannot matter for coalition members (§4 argument 5). | Phase 2. |
| 7 | The partial-synergy state is not Markov under delay. | Test 6a. | Phase 0. The 10 stored partial cells at d = 25 and d = 50 should be re-run for all six methods. |
| 8 | "Blind by construction" at the coalition level is slightly overstated. | Test 6b. | Reword `docs/h3_results.md` §2 and add the truth-level row. |
| 9 | Pilot seeds are inside the confirmatory grid. | `scripts/run_mode_p.py`: pilot seeds (0, 1); main seeds (0, 1, 2, 3, 4). Thresholds were set on data that the confirmatory test then reused. | Small effect, but use fresh seeds (5–9) for the next confirmatory round. |
| 10 | Mode P is one regime. | A near-optimal scripted policy that recovers from any one-step lapse is exactly where one-step advantages vanish. | Say so. Mode T, or a degraded-policy control, is needed before generalising. |
| 11 | Prior work is missing. | Difference rewards (Wolpert & Tumer): your leave-one-out ground truth *is* their Wonderful Life Utility, and COMA was derived from it. Also missing: Dr.Reinforce, the action-gap line, removal-based explanation, model-based counterfactual credit (MACD), and the 2026 work on exact credit for LLM agent teams. | §10. |

### 2.3 Against the one-pager

| One-pager promise | Status |
|---|---|
| Stage 1: "baseline credit error grows with reward delay and agent turnover, even when team return looks good" | Not shown. Delay: H1a not supported. Turnover: dropped. Return: untested (no Mode T). What you found instead is a different and better-supported claim: slack, not delay. |
| Stage 2: "our mechanism lowers credit error vs baselines at matched compute … keeps team return … far cheaper than exact Shapley" | Not started in the repo. Test 4 below is a first version of the credit-error part. "Keeps team return" is open. |

---

## 3. New tests

All tests use your code for the environment, scripted team, replay ground truth and metrics. γ = 0.99 throughout. "L1" is your metric: the L1 distance between credit shares, 0 = exact, 2 = disjoint. "Open-loop truth" is your default ground truth (replay recorded actions with agents removed). "Closed-loop truth" is your sensitivity protocol (re-run the scripted team with agents removed).

### Test 1. Uniform baseline on the main grid

**Question.** Your "still to do" list asks whether the main-grid L1 values carry information.

**Method.** `uniform_main.py` reads the per-episode truth and per-episode L1 stored in `results/mode_p/main/*.json`. Uniform credit gives 1/n to every agent. Target: leave-one-out, as in your main tables. A condition is one (delay, enabler setting) pair, 8 per bystander count.

**Result.** Conditions where the method's mean L1 is below uniform's:

| Bystanders | Uniform L1 | VDN | QMIX | COMA | MAPPO |
|---|---|---|---|---|---|
| 0 | 0.00 | 0/8 (0.56–1.53) | 0/8 (0.51–1.05) | 0/8 (1.19–1.35) | 0/8 |
| 2 | 0.72–0.73 | 0/8 | 1/8 | 0/8 | 1/8 |
| 4 | 1.00–1.02 | 1/8 | 2/8 | 0/8 | 0/8 |

The four wins for VDN and QMIX: QMIX at d = 0, slack, b = 2 (0.44 vs 0.73) and b = 4 (0.66 vs 1.02); QMIX at d = 10, critical, b = 4 (0.85 vs 1.00); VDN at d = 0, slack, b = 4 (0.79 vs 1.02). Each holds in 5 of 5 seeds. MAPPO's one win (0.704 vs 0.715) is the case where its shared advantage has a positive sign, which makes it the uniform split.

**Reading.** QMIX at d = 0 with a slack enabler does carry information: it separates bystanders from the coalition. With any delay that edge survives in 1 of 12 conditions (d ∈ {10, 25, 50} × enabler setting × b ∈ {2, 4}), and at d = 0 with a critical enabler QMIX also loses. With no bystanders, where the right answer is one third each, every method is far from it. This is a cleaner statement of your delay result than H1a.

### Test 2. COMA's expected credit is zero for everyone

**Claim (§4 argument 2).** COMA's advantage subtracts the average over the agent's own actions under its policy. So its expectation over that agent's action is zero at every step, for every agent, with any policy and a perfect critic.

**Check.** `coma_zero_mean.py` uses your exact per-step counterfactual truth (`replay.step_counterfactuals`), which is what a perfect critic would output. Layout: 5×10 grid; agent A pulls a lever, agents B and C must then interact on their goal cells in the same step for a reward of 10; D and E are bystanders; d = 0, slack enabler, scripted team with ε = 0.2. 30 test episodes (15 each from seeds 0 and 1).

| | A (enabler) | B | C | D | E |
|---|---|---|---|---|---|
| COMA-style credit, mean ± SE | −0.02 ± 0.06 | −0.01 ± 0.12 | +0.07 ± 0.06 | 0.00 ± 0.01 | 0.00 ± 0.02 |
| median | 0.00 | 0.10 | 0.11 | 0.00 | 0.00 |
| episodes with negative credit | 3% | 33% | 23% | 7% | 7% |
| Noop-swap credit (VDN/QMIX target), mean | 0.03 | 0.32 | 0.79 | 0.00 | 0.00 |

**Reading.** The means are zero within noise. The distribution is skewed: usually a small positive number, occasionally a large negative one. So a perfect COMA critic credits each agent with *how lucky its exploration noise was this episode*. This explains three of your results at once: H2′ reversed (credit is noise, noise spreads evenly, bystanders collect it); "median 0, mean 0.54" for the critical enabler; and the mechanism you flagged as "likely, not yet verified" in `main_results.md` §1.

### Test 3. A persistent counterfactual closes the semantic gap

**Idea.** Your level-2 result says one-step counterfactuals miss the enabler because the team recovers. So do not let it recover: remove the agent at step t and keep it removed.

**Definition.** V⁻ⁱ(t) is the discounted return from step t when agent i is removed at step t and the rest of the scripted team carries on (closed loop, same random streams). Per-step credit:

  c(i, t) = r_t + γ·V⁻ⁱ(t+1) − V⁻ⁱ(t)

Its γᵗ-weighted sum over the episode equals G − V⁻ⁱ(0) exactly (§4 argument 3), where G is the discounted return. That is the closed-loop leave-one-out credit.

**One episode, in full.** Layout as in Test 2 but with d = 25: reward 10 is paid 25 steps after B and C complete; horizon 60; ε = 0.2; cell (d = 25, b = 2, slack, seed 0), test episode 0. In this episode A pulls the lever at t = 4, B and C complete at t = 14, the payout arrives at t = 39, and G = 6.757.

| Agent | V⁻ⁱ(0) | Steps with non-zero credit |
|---|---|---|
| A (enabler) | 0.00 | t = 4: **+7.03** (the lever pull) |
| B (finisher) | 0.00 | t = 14: **+7.78** (the joint interaction) |
| C (finisher) | 0.00 | t = 14: **+7.78** |
| D, E (bystanders) | 6.76 | none |

A gets its credit at the pull, 35 steps before the reward. The delay plays no role. (7.03 = 10·γ³⁵.)

**Across cells** (48 main-grid cells; exact, from the simulator, no learning):

| Enabler | Your one-step missed credit (noop notion) | Persistent missed credit | Persistent L1 vs open-loop truth | Closed- vs open-loop truth L1 |
|---|---|---|---|---|
| slack | 0.99 | 0.00 | 0.08 (b = 2), 0.16 (b = 4) | 0.08, 0.17 |
| critical | 0.65–0.67 | 0.00 | 0.10 (b = 2), 0.19 (b = 4) | 0.10, 0.19 |

**Reading.** The residual L1 is exactly the difference between your two ground-truth protocols. It is not an error of the credit. So the semantic gap is caused by the *continuation* in the counterfactual (the agent comes back next step), not by looking at one step at a time.

This also gives you most of Step 9. Closed- and open-loop truths differ by L1 ≈ 0.08–0.10 at b = 2 and in the overlap layout, 0.17–0.19 at b = 4, and **0.37–0.38 in the partial layout**. The partial layout's ground truth depends heavily on the protocol, which is an H0 problem you have not reported.

### Test 4. A critic trained with removal interventions

**Idea.** V⁻ⁱ is a value function. It can be learned, but only from data in which agents are sometimes removed. Your global state already carries a presence flag per agent, so a centralised critic on that state *is* a removal-conditioned critic once the data contains removals. No new architecture.

**Method** (`rcc_lib.py`, `run_rcc.py`). I call it RCC, for removal-conditioned critic.

- **Data.** The same 200 training episodes per cell as yours (same seeds). In each, with probability 0.5, one removal: half at t = 0 and half at a uniformly random step; half remove one random agent and half a random subset. That gave 88–110 intervened episodes per cell. Removed agents leave the grid, as in your `reset(present=mask)`.
- **Critic.** An MLP (2 × 128, scikit-learn) from the global state to the discounted and undiscounted return-to-go, fitted by regression for 100 epochs.
- **Read-out.** Zero agent i's position and presence flag in each test state and query the critic. Episode credit: G − v̂(N∖i) ("pathwise"), which is what I scored first, and v̂(N) − v̂(N∖i) ("expected"), added after seeing the partial-layout result below.
- **Control.** The identical critic trained on your original 200 episodes (no removals), queried the same way.
- **Scripted team change.** For mid-episode removal I subclassed `ScriptedTeam` so that finishers still proceed when an absent enabler has already pulled its lever. For removal at t = 0 it is identical to yours.
- **Scoring.** Your 30 test episodes per cell, your `evaluate_against`, leave-one-out target, open-loop truth.

**Single-coalition cells** (A enables, B and C finish, b bystanders; 5 seeds at b = 2, 3 seeds at b = 4; mean over seeds):

| d | b | Enabler | Uniform | Best stored method | RCC | Same critic, no interventions | Floor¹ | RCC step r² | Stored step r, max³ |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2 | slack | 0.73 | QMIX 0.44 | **0.10** | 0.99 | 0.08 | 0.97 | 0.29 |
| 0 | 2 | critical | 0.71 | VDN 1.02 | **0.11** | 0.97 | 0.10 | 0.86 | 0.31 |
| 0 | 4 | slack | 1.03 | QMIX 0.71 | **0.19** | 1.19 | 0.17 | 0.91 | 0.17 |
| 0 | 4 | critical | 1.02 | VDN 1.10 | **0.26** | 1.14 | 0.19 | 0.83 | 0.25 |
| 25 | 2 | slack | 0.73 | QMIX 0.94 | **0.10** | 0.92 | 0.08 | 0.97 | 0.23 |
| 25 | 2 | critical | 0.71 | QMIX 0.94 | **0.11** | 0.92 | 0.10 | 0.87 | 0.34 |
| 25 | 4 | slack | 1.03 | VDN 1.51 | **0.20** | 1.44 | 0.17 | 0.93 | 0.14 |
| 25 | 4 | critical | 1.02 | QMIX 1.49 | **0.21** | 1.39 | 0.19 | 0.87 | 0.20 |
| 50 | 2 | slack | 0.73 | QMIX 0.98 | **0.09** | 0.98 | 0.08 | 0.95 | 0.23 |
| 50 | 2 | critical | 0.71 | QMIX 0.99 | **0.12** | 0.86 | 0.10 | 0.90 | 0.26 |
| 50 | 4 | slack | 1.03 | QMIX 1.40 | **0.18** | 1.36 | 0.17 | 0.92 | 0.18 |
| 50 | 4 | critical | 1.02 | QMIX 1.35 | **0.22** | 1.36 | 0.19 | 0.86 | 0.16 |

¹ Floor: L1 of the planted structure against the realised per-episode truth. No estimator of *expected* credit can do better, because in some episodes the realised coalition differs from the planted one.
² Pearson correlation between RCC's per-step credit and the exact persistent credit of Test 3, over all (episode, step, agent). This is "level 1" for RCC: is the critic right about its own target.
³ The largest level-1 step correlation of any stored method in any seed of that condition, each against its own one-step target. MAPPO is excluded from "best stored method" because of its read-out artifact.

RCC is below uniform in 48 of 48 seeds and below the best stored method in 12 of 12 conditions. Leakage is 0.01–0.06 and missed enabler credit 0.00–0.04. Delay makes no difference.

**Overlap and partial-synergy cells** (your H3 layouts, 5 seeds each):

| Layout | d | Uniform | Best stored (LOO target) | RCC pathwise | RCC expected | No interventions | Floor | RCC step r |
|---|---|---|---|---|---|---|---|---|
| overlap | 0 | 0.49 | STAS 0.52 | 0.11 | 0.11 | 0.53 | 0.09 | 0.92 |
| overlap | 25 | 0.49 | STAS 0.52 | 0.11 | 0.12 | 0.64 | 0.09 | 0.92 |
| overlap | 50 | 0.49 | STAS 0.76 | 0.10 | 0.11 | 0.55 | 0.09 | 0.84 |
| partial | 0 | 0.90 | QMIX 1.01 | 0.59 | 0.42 | 1.23 | 0.26 | 0.76 |
| partial | 25 | 0.90 | STAS 1.18 | 0.59 | 0.41 | 1.29 | 0.26 | 0.36 |
| partial | 50 | 0.90 | STAS 1.06 | 0.59 | 0.43 | 1.29 | 0.26 | 0.28 |

**Reading.**

- The persistent target is learnable where the one-step target is not. The reason is scale (§4 argument 4): persistent effects are as large as the value itself (about 7–10), one-step effects are about 0.1.
- The intervention data is what does the work. The same network without it is at or worse than uniform.
- **The partial layout is not solved.** Two separate causes:
  - *My pathwise read-out leaks.* G − v̂(N∖i) contains the episode's return surprise, G − E[G], which goes to every agent including bystanders (leakage 0.28). In the other layouts G is always 10, so this never showed. The expected read-out removes it (leakage 0.09–0.11, L1 0.41–0.43). I changed the read-out after seeing the data; treat it as post hoc.
  - *The state is not Markov under delay* (Test 6a). That is why step r falls from 0.76 to 0.28.
- Cost: interventions lower the return of the training data from 10.0 to 7.5–8.4 in the single-coalition cells.

### Test 5. Coalition level

**Method.** Query the RCC critic at the initial state for every coalition S (all 2ⁿ presence masks) to get v̂(S). Then compute Harsanyi dividends two ways: the plain Möbius transform, as you do, and a sparse fit (Lasso over the full Möbius basis, penalty chosen by cross-validation over coalitions, no order limit and no ground truth). Metrics as in `experiments.coalition_scores`.

**Result at 200 episodes** (ranges over d ∈ {0, 25, 50}; truth: interaction mass 1.0):

| Layout | Method | Interaction mass | Dividend error (0–2) | Recall | Shapley L1 (uniform) |
|---|---|---|---|---|---|
| overlap | VDN, QMIX, NA²Q (stored) | 0.000–0.002 | 2.000 | 0.36–0.39 | 0.80–1.36 (0.45) |
| overlap | RCC, plain Möbius | 0.94–0.96 | 1.51–1.57 | 0.71–0.92 | 0.16 |
| overlap | RCC, sparse | 0.93–0.97 | 0.98–1.08 | 0.94 | 0.15–0.16 |
| overlap | same critic, no interventions, sparse | 0.26–0.33 | 1.99 | 0.36–0.39 | 0.51–0.59 |
| partial | VDN, QMIX, NA²Q (stored) | 0.000–0.002 | 2.000 | 0.38–0.46 | 0.99–1.52 (0.88) |
| partial | RCC, sparse | 0.80–0.89 | 0.99–1.18 | 0.93–0.94 | 0.32–0.34 |

In the overlap layout (truth: {A, B, C} = 6 and {B, D} = 4), seed 0, d = 0, the sparse dividends are {A,B,C} = 4.36, {B,D} = 3.62, then a spurious {A,C} = 1.22 and smaller terms. The right two coalitions are on top, but about 30% of the mass is on wrong ones.

**How it scales with data** (`scaling.py`; 3 seeds; d = 0 for overlap and partial, d = 25 for the single coalition):

| Layout | Episodes | Removal probability | Intervened episodes | Training return | LOO L1 | Dividend error | Interaction mass |
|---|---|---|---|---|---|---|---|
| single coalition | 200 | 0.10 | 19 | 9.55 | 0.26 | 1.91 | 0.72 |
| | 200 | 0.25 | 47 | 8.95 | 0.22 | 1.79 | 0.87 |
| | 200 | 0.50 | 102 | 7.65 | 0.09 | 0.90 | 0.94 |
| | 400 | 0.50 | 207 | 7.69 | 0.08 | 0.41 | 1.00 |
| | 800 | 0.50 | 413 | 7.63 | 0.08 | 0.31 | 1.00 |
| overlap | 200 | 0.10 | 19 | 9.61 | 0.21 | 1.78 | 0.74 |
| | 200 | 0.50 | 102 | 8.03 | 0.11 | 0.91 | 0.94 |
| | 400 | 0.50 | 207 | 7.98 | 0.12 | 0.51 | 0.98 |
| | 800 | 0.50 | 413 | 7.92 | 0.12 | 0.47 | 0.99 |
| partial | 200 | 0.50 | 102 | 6.74 | 0.60 | 1.10 | 0.83 |
| | 800 | 0.50 | 413 | 6.80 | 0.60 | 0.78 | 0.98 |

**Reading.** Individual credit needs few interventions: 19 intervened episodes already beat uniform (0.26 vs 0.73; 0.21 vs 0.49). Coalition structure needs many: on the single-coalition and overlap layouts the dividend error is 1.8–1.9 with 19 intervened episodes, 0.9 with about 100, and 0.4–0.5 with about 200. This is the trade-off to report. The plain Möbius transform amplifies small value errors, which is why the sparse fit matters.

### Test 6. Two problems in the current setup

**6a. The partial-synergy state is not Markov under delay.** A partial event pays 5 or 10 depending on how many finishers took part. Once it completes, the global state shows "payout pending" and the countdown, but not the amount. `aug_test.py` trains the same critic with and without the pending amount appended to the state (partial layout, 3 seeds):

| d | Amount in state | Full-team value RMSE | RCC value r | RCC step r |
|---|---|---|---|---|
| 0 | no | 1.10 | 0.94 | 0.75 |
| 0 | yes | 1.11 | 0.94 | 0.72 |
| 25 | no | 1.71 | 0.84 | 0.35 |
| 25 | yes | **0.88** | **0.98** | **0.75** |

At d = 0 it makes no difference (nothing is ever pending). At d = 25 it halves the error. This is the same class of problem as your pilot Problem 1. It affects every critic in your 10 stored partial cells at d = 25 and d = 50, and probably explains "COMA's critic loss is high on the partial layout (about 3.5)".

**6b. The true per-step game is not exactly additive.** `docs/h3_results.md` §2 argues that a per-step game "can only be additive" because coalition members act at different steps. `stepgame_truth.py` computes that game exactly (what a perfect Q_tot would give), with your definition: at step t, agents in S take their real action, the rest take noop, and the team then continues. 10 test episodes per layout, seed 0, d = 0:

| Layout | Interaction mass (perfect critic) | Dividend error | Learned mixers (stored) |
|---|---|---|---|
| overlap | 0.26 | 1.94 | 0.000–0.002, 2.00 |
| partial | 0.35 | 2.00 | 0.000–0.002, 2.00 |

B and C interact in the *same* step, so the per-step game does contain a {B, C} term. It never contains {A, B, C} or {B, D}, so the dividend error stays at the maximum and your conclusion stands. But the learned mixers also miss the same-step term that a perfect critic would show. So the coalition-level failure is mostly semantic and partly estimation. You applied two-level grading to individuals but not to coalitions; this is the missing coalition-level "level 2" row. The sample is small (10 episodes, one seed).

### Test 7. Does exact credit speed up learning? A toy

**Why a toy.** Mode T in the gridworld needs PyTorch and hours. This checks the logic first.

**Game** (`toy_headroom.py`). One-shot and repeated. K events. Each event has 2 planted members and pays 10 if both choose it in the same round. Every agent picks among K + 1 actions (idle, or any event). Team reward is the sum, so the optimum is 10K. Learners: independent softmax REINFORCE with a running-mean baseline, learning rate 0.02, 20 seeds, 30,000 rounds. Only the learning signal differs:

- **team:** every agent receives the team reward;
- **exact leave-one-out:** every agent receives the team reward minus the team reward with its own action set to idle (the Wolpert–Tumer difference reward).

"Rounds to 90%" is the first round at which the 200-round running mean return exceeds 90% of the optimum (median over seeds).

| Events K | Agents | Team reward | Exact leave-one-out |
|---|---|---|---|
| 1 | 2 | 215 | 215 |
| 1 (+4 bystanders) | 6 | 216 | 216 |
| 2 | 4 | 260 | 258 |
| 4 | 8 | 405 | 358 |
| 8 | 16 | 1,961 (18 of 20 seeds solved) | 648 |
| 16 | 32 | not reached (0 of 10 seeds; final return 82% of optimum) | 1,546 |

**Reading.** With one coalition, exact credit is *identical* to the team reward for the members, with or without bystanders. The benefit appears only when an agent's reward is mixed with rewards from events it has nothing to do with, and it grows with the number of such events. This is the classical difference-rewards result in your setting. It is a toy: one step, no delay, no exploration problem in space. It tells you where to look in the gridworld, not what you will find.

### What these tests do not show

- Nothing here is confirmatory. Same seeds as your runs, analysis choices made today.
- RCC was tested only with a scripted, stationary policy. With learning policies the critic chases a moving target.
- RCC's comparison with your methods is not like for like: it had different data (with interventions) and a different trainer (regression, not TD). The fair controls are the no-intervention ablation, which I ran, and your methods trained on the intervention data, which I could not run (§6).
- RCC estimates *expected* credit under the policy. It cannot say that in episode 17 a bystander happened to matter. Your open-loop truth can.
- At the initial state, RCC is close to "measure v(S) by running the coalition, then smooth". That is the point, but a reviewer will say it. The per-step fidelity (0.83–0.97) is the part that shows generalisation across states.

---

## 4. Why it works: six short arguments

Notation: s_t state, r_t team reward, γ discount, G = Σ_t γᵗ r_t, V the value of the team's policy, N the set of all agents.

**1. Summed advantages measure surprise, not contribution.** For any function V with V(s_T) = 0,

  Σ_t γᵗ [r_t + γV(s_{t+1}) − V(s_t)] = G − V(s_0).

The terms cancel in pairs. With V the team's own value, the left side is the summed team advantage. So it equals how much better this episode went than expected. A good deterministic policy has G ≈ V(s_0), so the sum is about zero. This is why MAPPO's read-out is a small negative number shared by all agents and scores L1 = 2.000.

**2. COMA's expected credit is zero for every agent.** COMA's advantage is A_i = Q(s, a) − Σ_{a′} π_i(a′) Q(s, (a′, a_{−i})). Averaging over a_i ~ π_i gives exactly 0, at every state, for every agent. So E[Σ_t A_i] = 0 whether the agent is the enabler or a bystander. Realised credit is the agent's exploration luck (Test 2).

**3. Removal credit is a sum of persistent per-step terms.** Apply argument 1 with V⁻ⁱ, the value when agent i is gone from now on:

  Σ_t γᵗ [r_t + γV⁻ⁱ(s_{t+1}) − V⁻ⁱ(s_t)] = G − V⁻ⁱ(s_0).

In expectation the right side is J(team) − J(team without i), the closed-loop leave-one-out credit. By the performance difference lemma (Kakade & Langford, 2002), each term's expectation is the advantage of what the team actually did, measured in the world where i leaves afterwards, over i leaving now. So the right per-step counterfactual is "the agent stops contributing from here on", not "the agent skips one step". The same construction with V^S for every coalition S gives per-step dividends that sum to the closed-loop Harsanyi dividends.

**4. Scale explains the fidelity gap.** If the team recovers a one-step lapse by agent i within k steps, the one-step effect is at most (1 − γᵏ)·V. With γ = 0.99 and k = 1 that is 1% of the value, about 0.1 here, and exactly 0 when the agent has slack. The persistent effect V − V⁻ⁱ can be as large as V itself, about 7–10 here. A critic with error ε has signal-to-error ratio (1 − γᵏ)V/ε on the first target and up to V/ε on the second: roughly 100 times better. Observed: step correlations ≤ 0.36 versus 0.83–0.97. This is the known action-gap problem (Baird, 1994; Tallec et al., 2019) applied to credit read-outs.

**5. With a single all-or-nothing coalition, exact credit cannot help its members.** If the only reward needs every member of coalition C, then removing any member i of C gives return 0. So i's leave-one-out credit equals the team return on every trajectory, and its Shapley value is the team return divided by |C|. Exact credit changes only the bystanders' signal. In your main layout bystanders cannot produce reward. Test 7 confirms: 215 vs 215.

**6. RCC's credit, used as a reward, should not speed up learning.** Two parts.

- *Do not reward remaining importance.* Let M_i(s) = V(s) − V⁻ⁱ(s), how much agent i still matters from state s. Example: A must pull a lever before B and C can finish for a reward of 10. Before the pull M_A = 10·γᵏ; after it M_A = 0, because the team no longer needs A. An agent rewarded for M_i, or for increases in it, learns to stay indispensable and never pull.
- *The right-signed version is neutral.* c(i, t) from argument 3 rewards the pull. But it is the team reward plus γΦ(s′) − Φ(s) with Φ = V⁻ⁱ, which is potential-based shaping (Ng, Harada & Russell, 1999). That preserves optimal policies, and with Monte-Carlo returns it is the same as using V⁻ⁱ(s_t) as agent i's baseline. For a bystander V⁻ⁱ = V, the standard baseline. For a member who has not yet contributed V⁻ⁱ = 0, a worse baseline. So no variance reduction.

The learning benefit in Test 7 comes from a different object: the *open-loop* difference reward, which subtracts what the others' recorded actions would have earned without you. It cancels the noise from other agents' events along the same trajectory. A state-based value cannot do that.

**Identifiability, in one paragraph.** A return-decomposition model fitted to full-team returns satisfies Σ_{i,t} r̂(i, t) = R. If r̂ is a solution, so is r̂ + h for any h whose entries sum to zero, including any transfer of credit between agents. The data do not choose among them; the architecture does. On your data R = 10 in every training episode, so r̂ = 10/(nT) for everyone has zero loss. With removal interventions the data contain returns for coalitions other than N, so v(S) = E[R | S present] is an ordinary regression target, and dividends and Shapley values follow uniquely by Möbius inversion. This is the same lesson as in removal-based explanation of ML models: replacing a feature with a default value queries the model off its data manifold, and the fix is a surrogate trained with the features actually masked (Frye et al., 2021; Covert, Lundberg & Lee, 2021). Swapping an agent's action to noop inside a critic is the off-manifold query. Removing the agent in the environment is the masked training.

---

## 5. The proposal: credit by intervention

This is a data-collection protocol and a read-out. It is not a new network, and it is not a combination of two existing architectures.

**Protocol.**

1. During data collection, in a fraction p of episodes, remove a random set of agents at a random step for the rest of the episode. Removed agents leave the environment; their observation is zeroed and their presence flag is 0 (your environment already does this at reset).
2. Train any centralised state-value critic on all episodes in the usual way. Its input must include the presence flags. In EPyMARL that is MAPPO's critic unchanged.

**Read-outs.**

| Level | Formula | Cost per query |
|---|---|---|
| Per step, per agent | c(i, t) = r_t + γ·V̂(s_{t+1} without i) − V̂(s_t without i). Luck-free variant: subtract the team's own TD error. | n forward passes per step |
| Episode, per agent | V̂(s_0) − V̂(s_0 without i) | n forward passes |
| Coalitions | v̂(S) = V̂(s_0 with only S present) for all S, then sparse Möbius fit → dividends, Shapley | 2ⁿ forward passes (n ≤ 8), or sampled masks |

**Cost.** No extra parameters. A fraction p of episodes earn less (Test 5: return 9.55 at p = 0.1, 7.65 at p = 0.5, against 10). No simulator resets or replays are needed, only the ability to switch an agent off.

**Limits, stated up front.**

- It estimates expected credit under the current policy, with the remaining agents reacting to the absence (closed loop). It is not trajectory-specific.
- Coalition structure needs many interventions (Test 5).
- Under learning, the remaining agents will adapt to absences. That changes both the policies (they become robust to dropout) and what the credit means. This must be measured, not assumed.
- It does not by itself improve learning (§4 argument 6).

**How it relates to prior work.** From my searches today; "not found" means not found in about fifteen queries, not proven absent.

| Work | What it does | Difference |
|---|---|---|
| Wolpert & Tumer difference rewards (WLU); Agogino & Tumer | Agent's reward = G minus G with the agent clamped to a null action | The definition of your leave-one-out truth. Needs the reward function or a resettable simulator. A learning signal, not graded against truth. |
| COMA; Shapley Counterfactual Credits (Li et al., 2021); SQDDPG; Multiagent Model-based Credit Assignment (AAMAS 2022) | Replace an agent's *action* by a default or marginal inside a learned critic or model, for one step | One-step, and queries the critic where it has no data. This is what your benchmark shows failing. |
| MACD (Chai et al., AAMAS 2024) | World-model rollouts with one agent's first action replaced by a default | One-step replacement; model trained on full-team data; evaluated by return only. Its authors note that a default action is not the same as the agent leaving. |
| Dr.Reinforce (Castellini et al., 2021) | Learns a reward network and differences it | Per-step reward differencing; no persistent removal. |
| SHAQ (Wang et al., 2022) | Defines a state-conditional coalition value in theory (Markov Shapley value) | I could not read the full text. As far as I can tell coalition values are approximated from full-team Q-values, not learned from coalition play. **Read this before any novelty claim.** |
| Heuillet et al. (2022); Mahjoub et al. (2023) | Shapley or leave-one-out by re-running trained policies with agents replaced, at evaluation time | Post-hoc, needs the simulator, not a learned estimator. |
| MA-POCA; COPA; REFIL | Train with varying team membership | For robustness or generalisation, not for credit read-outs. REFIL's sub-groups are imagined inside the mixer, not real removals. |
| C3 (Chen et al., 2026); CCPO (2026); "Agents that Matter" (2026) | Exact or removal-based credit for LLM agent teams | Same philosophy (measure, do not infer). C3 reports that a learned critic reaches rank correlation 0.29 where the measured counterfactual reaches 0.69, which matches your level-1 finding in a different field. C3 also argues that removing an agent shifts the others' observation distribution; training with removals is the answer to that. |
| Li, Lee & Bareinboim (2026), Counterfactual Shapley Credit Assignment | Shapley credit over *time steps* for one agent, with replacement actions from a baseline policy | Single agent. I could not read §3, so I do not know whether replacements persist. **Read before claiming the per-step construction.** |
| Frye et al. (2021); Covert et al. (2021) | Removal-based explanation with surrogates trained under masking | The idea transferred here, from features to agents. |

**What I would claim, if the confirmatory round holds.** (i) A ground-truth benchmark with two-level grading. (ii) Three propositions explaining why one-step read-outs cannot recover removal credit. (iii) Removal interventions make it identifiable with a standard critic, shown against truth, with a no-intervention ablation. (iv) An honest account of when credit matters for learning.

---

## 6. Is the problem validated enough?

**"Standard read-outs do not recover who earned the reward": nearly.** It has a measurement, a mechanism and now a mathematical reason. Four cheap checks would close it:

| Check | Why | Cost |
|---|---|---|
| V1. Coverage control. Re-run Mode P for QMIX and COMA on data with outcome variance: ε = 0.4, and a mix with 25% episodes from a degraded script. | Is low level-1 fidelity a property of the methods or of training data in which the reward always arrives? | 2 data settings × 6 cells × 5 seeds |
| V2. Your methods on the intervention data. Train QMIX and COMA on the Test 4 dataset; keep their standard read-outs; also read QMIX as a masked value (max over actions of Q_tot at the state without i). | Separates "better data" from "better read-out". My prediction: one-step read-outs stay poor; the masked-value read-out becomes good. If so, the fix is the protocol, not the method. | same grid as Test 4 |
| V3. Coalition-level level 2 on all 30 H3 cells (`stepgame_truth.py`). | Test 6b is 10 episodes and one seed. | about 8 s per episode |
| V4. Uniform and planted-oracle rows in every table. | Test 1. | none |

**"It matters for learning": not validated at all.** This needs Phase 2. Until then, do not write "wrong credit trains wrong behaviour".

**External validity: thin.** One custom environment, scripted policies. A second environment family (the one-pager lists Level-Based Foraging) would help a paper but is not the next step.

---

## 7. Plan, step by step

Effort is in person-days for someone who knows the repo. I do not know your deadline; the order is by dependency, and each phase ends with something you can report. Roles A–D are for your four members; assign as you like.

### Phase 0. Fixes (1 day, do first)

| # | Task | File | Done when |
|---|---|---|---|
| 0.1 | Add each event's pending payout amount (÷ its reward) to the global state. | `envs/gridworld.py::get_state`, `state_size` (+K) | A new unit test: two partial episodes with payouts 5 and 10 have different states while pending. |
| 0.2 | Re-run the 10 partial cells at d = 25, 50 for all six methods. | `scripts/run_mode_p.py --grid h3` | Critic losses at d = 25 within 2× of d = 0. |
| 0.3 | Add `remove_agents(agents)` to the environment for mid-episode removal. | `envs/gridworld.py` (3 lines, see `rcc_lib.drop_agents`) | Unit test: removal at t = 0 equals `reset(present=mask)`. |
| 0.4 | Let finishers proceed when an absent enabler has already pulled. | `scripted.py::_finish_feasible` (see `rcc_lib.BankedScriptedTeam`) | Existing tests still pass; closed-loop v(S) unchanged. |
| 0.5 | Uniform and planted-oracle baselines in the summaries. | `scripts/summarise_mode_p.py` (logic in `uniform_main.py`) | Both rows appear in `summary.md`. |
| 0.6 | Correct the documents. | `docs/h3_results.md` §2, `docs/main_results.md` §1 (COMA mechanism), `docs/step7_ports.md` (STAS wording), results overview | Items 2, 4, 7, 8 of §2.2 addressed. |

### Phase 1. Confirmatory repeat of Tests 3–5 in the repo (3–4 days)

| # | Task | Detail | Done when |
|---|---|---|---|
| 1.1 | Write `docs/preregistration_v3.md` **before any run**. | §8 below. Fresh seeds 5–9. | Committed before the first result. |
| 1.2 | Port RCC to PyTorch. | `src/whoearned/rcc.py`: reuse MAPPO's centralised V critic from the EPyMARL harness; train with `train_critic_sequential` on batches built from intervention episodes. Add `"rcc"` to `mode_p.ALL_ALGS` and to `read_credit`. | On cell (d = 25, b = 2, seed 0) it reproduces my prototype within 0.05 L1. |
| 1.3 | Intervention data generator. | `experiments.py`: `DataCell.episodes("train", p_drop=…)` using the sampler in `rcc_lib.sample_intervention`. | Unit test: with p_drop = 0 the episodes are byte-identical to today's. |
| 1.4 | Persistent ground truth. | `replay.py`: `persistent_values(config, make_team)` (logic in `rcc_lib.true_persistent_values`); add to `TestTruth`. About 1.3 s per 5-agent episode. | Unit test: γᵗ-weighted sum equals G − V⁻ⁱ(0). |
| 1.5 | Run the grid: main (d ∈ {0, 25, 50} × b ∈ {2, 4} × slack/critical) and H3, seeds 5–9, with RCC, the no-intervention ablation, and controls V1–V2. | | Tables as in Tests 4–5. |
| 1.6 | Sparse dividends and the data-scaling curve. | `rcc_lib.sparse_dividends`; `scaling.py`. | Dividend error against number of intervened episodes, with seeds. |
| 1.7 | Size k+1 honesty check. | New layout with a 4-member coalition (2 enablers, 2 finishers) and n = 6. The sparse fit has no order limit, so the check is whether it finds a size-4 dividend from the same budget. | Reported whatever the result. |
| 1.8 | Coalition-level level 2 (V3). | `stepgame_truth.py` on all H3 cells. | Row added to the H3 table. |

### Phase 2. Headroom test: does exact credit help learning here? (4–6 days, decision gate)

| # | Task | Detail |
|---|---|---|
| 2.1 | Multi-event layout. | `layouts.py::multi_event_layout(K)`: K separate lanes divided by walls, each with one enabler, one lever, one finisher, one goal, reward 10. n = 2K, K ∈ {1, 2, 4, 8}. Walls remove physical interference, so realised = planted by design. Keep lanes short (lever and goal 2–3 cells away) and the horizon about 25 so that exploration is feasible. |
| 2.2 | Feasibility check. | IPPO on K = 1 must reach 90% success within about 300k steps. If not, shorten the lane. Do this before anything else in the phase. |
| 2.3 | Exact difference rewards with shadow environments. | `ShadowDifferenceEnv`: alongside the real environment keep n copies, copy i reset without agent i. Each step, feed every copy the same joint action. Agent i's reward is r_real − r_copy_i. This is your open-loop removal replay run in lockstep, so per-step exact leave-one-out rewards come out online. It needs per-agent rewards in EPyMARL (`common_reward=False`; check that your pinned commit supports it). |
| 2.4 | Runs. | IPPO and MAPPO with (a) team reward, (b) shadow difference rewards. QMIX and COMA with team reward as references. K ∈ {1, 2, 4, 8}, 5 seeds. Count environment steps both with and without the shadow copies. |
| 2.5 | Measures. | Episodes to 90% of optimal return; final return; per-agent policy-gradient SNR (you already have this code). |
| 2.6 | **Gate.** | If (b) needs fewer than 0.67× the episodes of (a) at the largest K you can run: there is a learning story; go to Phase 4. If not: stop the learning track and write the measurement-and-estimator paper. Pre-register the K = 1 prediction too: no difference. |

### Phase 3. RCC under learning (3–4 days; can run alongside Phase 2)

| # | Task | Detail |
|---|---|---|
| 3.1 | `InterventionWrapper(env, p_drop)` in the EPyMARL runner. | Samples (step, set) per episode and calls `remove_agents`. Absent agents get noop-only available actions (your adapter already does this at reset). |
| 3.2 | Train MAPPO with p_drop ∈ {0, 0.1, 0.25} on the overlap layout and on K = 4. | The critic is the RCC. |
| 3.3 | At 10 checkpoints: RCC read-outs on 30 clean greedy evaluation episodes, against exact persistent truth (replay the prefix, remove, continue greedily). | Fidelity r and L1 over training. |
| 3.4 | Cost and side effects. | Return on clean episodes with and without interventions; whether agents become robust to dropout. |

### Phase 4. Only if the gate passes: make exact credit cheap (1 week)

Sample one agent per episode for the shadow copy (2× instead of (n+1)× steps) and learn a model of the removed-world reward from those copies. Compare with (a) and (b) per environment step. I would not design this further until Phase 2 says it is worth it.

### Phase 5. Write-up (1 week)

Structure: benchmark and two-level grading → three propositions → intervention protocol and results against truth → headroom result, whichever way it went → limits.

**Suggested split.** A: pre-registration, integration, documents. B: environment (0.1, 0.3, 2.1, 2.3, 3.1). C: RCC port and Phase 1 runs. D: Mode T runs, analysis and figures.

---

## 8. Pre-registration v3 (draft)

Fresh seeds 5–9. One-sided paired t-test on per-seed means, Holm within each hypothesis, exact Wilcoxon reported alongside, as in your amendment. Thresholds are set from today's exploratory numbers with a margin.

| ID | Hypothesis | Supported if | Today (exploratory) |
|---|---|---|---|
| R1 | A persistent counterfactual recovers enabler credit. | Exact persistent missed credit ≤ 0.10 in every single-coalition cell, slack and critical. | 0.00 |
| R2 | The persistent target is learnable; the one-step targets are not. | RCC step r ≥ 0.75 in every single-coalition and overlap condition; no stored-method step r above 0.5. | 0.83–0.97; ≤ 0.36 |
| R3 | Interventions are necessary. | No-intervention ablation L1 exceeds RCC L1 by ≥ 0.3 in every condition (paired). | gaps 0.42–1.24 |
| R4 | RCC beats the trivial baseline. | RCC expected-read-out L1 below uniform L1 in every condition (paired), and ≤ 0.35 on single-coalition and overlap cells. | 0.11–0.28 |
| R5 | Coalitions are visible. | Sparse interaction mass ≥ 0.75 and recall ≥ 0.80 on overlap and partial at 200 episodes; dividend error falls by ≥ 30% from 200 to 800 episodes. | 0.80–0.97; 0.93–0.94; −48% to −66% (overlap, single), −29% (partial) |
| R6 | Fixing the state restores the partial layout. | With the pending amount in the state, RCC step r ≥ 0.6 at d = 25. | 0.75 |
| R7 | Honesty check. | Report the size-4 result. No threshold; cannot fail, must be reported. | not run |
| R8 | No headroom with one coalition. | K = 1: episodes-to-90% of difference rewards within ±20% of team reward. | toy: 215 vs 215 |
| R9 | Headroom with many events. | Largest K: difference rewards need ≤ 0.67× the episodes. | toy: 0.33× at K = 8 |
| R10 | RCC survives learning. | Step r ≥ 0.6 at the final checkpoint with p_drop = 0.25. | not run |

R5's partial-layout fall (−29%) is just under the 30% threshold today. Either accept that it may fail or set the threshold for overlap and single-coalition layouts only; decide before running, not after.

---

## 9. What I need from you, and resources

**Data sets.** None. Everything is generated by your environment.

**Models.** None. No LLM is involved, so Ollama is not needed. I would not add an LLM judge for credit: "Agents that Matter" (2026) reports that introspective LLM judges do not reproduce ablation-based attribution.

**Hardware.**

- Your MacBook Air (16 GB) is enough for Phases 0, 1 and 3.3. My 2-core sandbox ran all 78 cells of Tests 3–5, including exact truths, in about 90 minutes.
- Phases 2 and 3 are many EPyMARL training runs (about 30 configurations × 5 seeds). Use the Ryzen 5 7600 + RTX 4060 machine your documents mention. On the Air alone, cut to 3 seeds and time one run before committing.

**Papers.** I read these only as abstracts, search snippets or automated summaries. If you upload the PDFs I will read them properly; the first two decide the novelty wording.

1. Wang et al., *SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning* (arXiv 2105.15013): the Markov Shapley value definition and how coalition values are obtained.
2. Li, Lee & Bareinboim, *Counterfactual Shapley Credit Assignment* (arXiv 2607.16999): §3.1 and Theorem 3.7.
3. Chen et al., *The Trace Is the State: Exact Credit Assignment for LLM Agent Teams* (arXiv 2603.06859): §2.3 on distribution shift under agent removal.
4. Heuillet et al., *Collective eXplainable AI* (arXiv 2110.01307).
5. Chen et al., *STAS* (arXiv 2304.07520), to check the port against the paper's own Alice-and-Bob numbers.

**Decisions.** Your deadline, and whether the Ryzen machine is available, decide how much of Phases 2–4 is realistic.

---

## 10. References to add

Checked today (abstract or snippet seen) unless marked †, which are from memory and must be verified before citing.

**Difference rewards and their descendants**
- Wolpert & Tumer, *Optimal Payoff Functions for Members of Collectives* (2001); Wolpert & Tumer, *Optimal Reward Functions in Distributed Reinforcement Learning* (NASA TR). Wonderful Life Utility, clamping, factoredness, learnability.
- Agogino & Tumer, *Analyzing and Visualizing Multiagent Rewards in Dynamic and Stochastic Environments*, JAAMAS 17, 2008.
- Proper & Tumer, *Modeling Difference Rewards for Multiagent Learning*, AAMAS 2012. Colby et al., *Local Approximation of Difference Evaluation Functions*, AAMAS 2016.
- Devlin, Yliniemi, Kudenko & Tumer, *Potential-Based Difference Rewards for Multiagent Reinforcement Learning*, AAMAS 2014. Devlin & Kudenko, *Theoretical Considerations of Potential-Based Reward Shaping for Multi-Agent Systems*, AAMAS 2011.
- Castellini, Devlin, Oliehoek & Savani, *Difference Rewards Policy Gradients*, arXiv 2012.11258.
- Han et al., *Multiagent Model-based Credit Assignment for Continuous Control*, AAMAS 2022, arXiv 2112.13937.
- Chai, Fu, Zhao & Zhu, *Aligning Credit for Multi-Agent Cooperation via Model-based Counterfactual Imagination*, AAMAS 2024.

**Why one-step effects are small and hard to learn**
- Tallec, Blier & Ollivier, *Making Deep Q-learning Methods Robust to Time Discretization*, ICML 2019, arXiv 1901.09732. Builds on Baird (1994), Advantage Updating†.
- Bellemare et al., *Increasing the Action Gap*, AAAI 2016†. Farahmand, *Action-Gap Phenomenon in Reinforcement Learning*, NeurIPS 2011†.

**Identities used in §4**
- Kakade & Langford, *Approximately Optimal Approximate Reinforcement Learning*, ICML 2002† (performance difference lemma).
- Ng, Harada & Russell, *Policy Invariance under Reward Transformations*, ICML 1999†.
- Kuba et al., *Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning*, ICLR 2022† (multi-agent advantage decomposition).

**Removal-based explanation**
- Covert, Lundberg & Lee, *Explaining by Removing*, JMLR 22(209), 2021, arXiv 2011.14878.
- Frye et al., *Shapley Explainability on the Data Manifold*, ICLR 2021, arXiv 2006.01272.
- Jethani et al., *FastSHAP*, ICLR 2022, arXiv 2107.07436.

**Recent, close in spirit**
- Chen et al., *The Trace Is the State: Exact Credit Assignment for LLM Agent Teams* (earlier title *Exact Is Easier*), arXiv 2603.06859.
- *Counterfactual Credit Policy Optimization for Multi-agent Collaboration*, arXiv 2603.21563.
- *Agents that Matter*, arXiv 2605.27621.
- *COSAC: Counterfactual Credit Assignment in Sequential Cooperative Teams*, arXiv 2604.17693.
- Bui, Mai & Nguyen, *Preference-Guided Learning for Sparse-Reward Multi-Agent Reinforcement Learning*, arXiv 2509.21828 (v2, 8 November 2026): temporal and agent credit from terminal outcomes, evaluated by return on MAMuJoCo and SMACv2.

Your v2 references [7] (arXiv 2607.16999) and [27] (arXiv 2610.02847) exist and say what you report; I checked both abstracts.

---

## Appendix A. Reproducing today's numbers

Unzip the bundle into `<repo>/experiments/rcc_prototype/`, or anywhere and set the environment variable `WHOEARNED_REPO` to the repository path. Requires numpy, scipy and scikit-learn. No PyTorch. Run from inside the folder.

| Test | Command | Time (2 cores) | Output |
|---|---|---|---|
| check | `python aggregate.py results` | seconds | reproduces `results/agg_final.txt` from my per-cell JSONs |
| 1 | `python uniform_main.py <repo>/results/mode_p/main` | seconds | stdout |
| 2 | `python coma_zero_mean.py 15` | 5 min | stdout |
| 3–5 | `python run_rcc.py h3 results/out_h3 2`, then the same with `main_b2` and `main_b4` into `results/out_main`; then `python aggregate.py results` and `python readout_variants.py results` | 90 min | one JSON per cell; the two summary tables |
| 5 (scaling) | `python scaling.py 2` with environment variable `RCC_RESULTS=results` | 15 min | `scaling.json` |
| 6a | `python aug_test.py` | 12 min | `aug_test.json` |
| 6b | `python stepgame_truth.py overlap 0 10 1`; same with `partial` | 3 min | `stepgame_*.json` |
| 7 | `python toy_headroom.py`; `python toy16.py` | 6 min | `toy_headroom.json` |

`run_rcc.py` skips cells whose JSON already exists, so delete `results/out_*` to recompute from scratch.

## Appendix B. Terms

| Term | Meaning |
|---|---|
| Removal credit | An agent's credit defined by what the team earns without it: leave-one-out (v(N) − v(N∖i)) or Shapley. |
| Open-loop / closed-loop truth | Replay the recorded actions with agents removed / re-run the team with agents removed. |
| One-step counterfactual | Change one agent's action at one step, then let the team continue as usual. What your level-1 targets are. |
| Persistent counterfactual | Remove the agent at a step and keep it removed. |
| RCC | Removal-conditioned critic: a centralised value critic whose input includes presence flags, trained on data with removal interventions. |
| Pathwise / expected read-out | G − v̂(N∖i) / v̂(N) − v̂(N∖i). |
| Floor | L1 between the planted structure and the realised per-episode truth; the best any expected-credit estimator can do. |
| Interaction mass | Share of total |dividend| on coalitions of two or more agents. |
| Headroom | How much faster learning is with exact credit than with the team reward. |
