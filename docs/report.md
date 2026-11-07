# Who Earned the Reward? Ground-Truth Evaluation of Subgroup Credit Assignment

Draft report, 7 November 2026. Every number is from a document in this repository; the source for each section is named in it. Full results: `docs/results_overview.md`.

## 1. Question

In cooperative multi-agent RL a team often receives one delayed reward for something a *subgroup* did. Credit-assignment methods are judged by the return they reach, not by whether the credit they assign is right. We ask: **when a coalition earns a delayed reward, do standard credit read-outs recover who earned it?** The only way to answer is to know the answer, so the project builds a benchmark with an exact ground truth.

## 2. Benchmark

- **Planted Coalition Gridworld** (`docs/environment_design.md`). Events have enablers and finishers (A pulls a lever, then B and C push together); the reward is paid d steps after completion; bystanders may wander; layouts include overlapping events and partial synergy.
- **Exact truth by true removal.** For each test episode we remove any subset of agents (replaying the episode, or re-running the scripted team), read off the coalition value v(S) for all 2^n subsets, and compute Harsanyi dividends, Shapley values and leave-one-out credit.
- **Metrics.** Individual credit error L1, leakage Λ to bystanders, missed enabler credit M, and, at the coalition level, dividend error, recall/precision and interaction mass.
- **Mode P.** Critics are trained on scripted episodes, so credit can be separated from exploration. Mode T (online training) was planned but could not be completed (§7).

## 3. Per-step credit read-outs (main grid, H0–H2)

120 cells (delay 0–50, 0–4 bystanders, slack or critical enabler, 5 seeds) with VDN, QMIX, COMA and MAPPO (`docs/main_results.md`).

![L1 against delay](figures/fig1_l1_vs_uniform.png)

*Figure 1. Individual error against leave-one-out credit, slack enabler. With no bystanders an equal split is exactly right, so the dashed line is at 0. MAPPO is omitted: its shared advantage gives every agent ±1/n by construction.*

- **Credit from per-step read-outs is a different quantity from "who earned the reward".** Even a perfect per-step critic gives a slack enabler essentially zero credit (missed credit 0.99 under the no-op notion in all 12 slack conditions, independent of delay), whereas removal credit gives it a full share.
- **Learned read-outs rarely beat an equal split.** Across the main grid 5 of 96 method-conditions beat uniform credit (4 if MAPPO's artefact is excluded), almost all at d ≤ 10.
- **COMA leaks to bystanders more, not less** (H2′ reversed in 8/8 comparisons). H2 holds for COMA (8/8); VDN and QMIX miss it, although every VDN/QMIX difference has the expected sign (a post-hoc sign test, p = 1.5e-5, indicative only because the conditions share data).
- **Low end-to-end error is not evidence of faithful credit** (H-fid): 30 of the 32 best end-to-end cells have per-step fidelity r < 0.3.
- Pre-registered H0, H1a and H1b are not supported as written (narrowly for H0; H1b because only 54% of "critical" episodes were critical).

## 4. Coalition level (H3)

NA²Q and STAS were reimplemented, and VDN, QMIX and NA²Q were read out as coalition games (`docs/h3_results.md`). Their games are almost purely additive (interaction mass 0.000–0.002 against a true 1.0), because a per-step game cannot represent coalitions whose members act at different times. Dividend error is at its maximum for all of them, so the pre-registered H3 test cannot separate methods and is **uninformative**.

## 5. A coalition estimator (H5)

The episodic coalition estimator (ECE-k) fits an episode-return model with coalition terms up to size k, reads out low-order dividends and splits each among its members. It is trained on episodes plus open-loop replays with random agents removed (`docs/h5_results.md`, pre-registered in `docs/h5_preregistration.md`).

![Interaction mass](figures/fig2_interaction_mass.png)

![Dividend error](figures/fig3_dividend_error.png)

*Figures 2 and 3. Coalition recovery at d = 0 (the results are the same at d = 50).*

- ECE-3 with replay data recovers the order-3 coalition: dividend error **0.43 against 2.00** for NA²Q (with or without the same replay data), recall and precision 0.96. It is significantly better in **8/8** conditions on dividend error and on individual L1 against uniform credit (gain 0.29–0.61).
- **Not supported:** lower leakage (2/8 against NA²Q, 1/8 against data-matched NA²Q). The replay data alone removes most of NA²Q's leakage.
- **The replay data is essential.** Trained on episodes alone, ECE reproduces uniform credit and recovers no coalition.
- **Honesty check.** ECE-2 cannot represent an order-3 coalition (dividend error stays 2.0 on the coalition layout and 1.77 on overlap) although its individual L1 stays close to ECE-3's.
- **Cost.** On one machine ECE-3 is about 3.5× cheaper than NA²Q and STAS with replay data.

## 6. Sensitivity (`docs/sensitivity.md`)

- **Shapley vs leave-one-out.** The two targets differ by L1 ≤ 0.03 on the coalition layout (0.06 on partial, 0.10 on overlap); rankings agree in 27 of 30 conditions and the best method is always the same. No verdict changes.
- **Open- vs closed-loop removal.** The closed-loop truth recovers the planted coalition in 100% of 3,600 test episodes (open loop: 83%; 66% at 4 bystanders), so the H0 failures are an artefact of open-loop replay. H2, H2′ and H-fid are unchanged, but **H1a becomes supported through one QMIX condition** (Holm p 0.074 → 0.030). H1a is therefore borderline, not a robust negative.

## 7. Mode T and H4: not completed

QMIX and MAPPO for 400k steps and QMIX for 1M steps never earned reward (a uniform-random policy completes the task in 0 of 20,000 episodes), so return is constant and H4 (return is a weak proxy for credit faithfulness) is **untested** (`docs/mode_t_results.md`). There is no return-versus-L1 figure for the same reason.

## 8. Limitations

- One environment family, scripted behaviour data, at most 7 agents and 5 seeds per cell; power is limited, and the pre-registered rules are strict (a family is supported only if every condition passes).
- VDN and QMIX learn Q*; COMA, MAPPO and NA²Q (SARSA) learn the behaviour policy's value.
- ECE with replay data needs the ability to replay episodes with agents removed, a stronger assumption than any compared method makes.
- STAS and NA²Q are our reimplementations. STAS ran on a mix of devices in H5 (laptop GPU, laptop CPU, RTX 4060); it is descriptive there.
- The uniform baseline, the H1b split by criticality, the VDN/QMIX sign test and the sensitivity checks are post-hoc or exploratory and are labelled so in their documents.
- H0 and H1b were not re-tested under the closed-loop truth.

## 9. Reproducing

`docs/running_on_another_machine.md` covers environment setup and each grid. Figures: `scripts/make_figures.py` (needs `matplotlib`). Mode T needs a Python 3.12 environment, because `sacred` does not import under 3.14.
