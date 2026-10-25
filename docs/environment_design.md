# Planted Coalition Gridworld: Design

This environment makes the team reward a **known** sum of coalition events (problem statement, Section 2.1). It is deliberately small and fully deterministic, so that replaying an episode with any subset of agents removed gives an exact v(S).

## World

- An `H × W` grid with optional wall cells. Each cell holds at most one agent when `collisions=True` (the default), so agents can physically block each other. This is what makes "no-op" and "true removal" differ.
- `n` agents with fixed start cells. Episodes have a fixed horizon `T`.
- Actions: `0 noop, 1 up, 2 down, 3 left, 4 right, 5 interact`.
- Moves are resolved in agent-index order. A move into a wall, off the grid, or into a cell occupied by a present agent fails, and the agent stays put. This rule is simple, deterministic and documented, and it is the only order effect in the environment.

## Events (the planted coalitions)

Each event `k` has:

| Field | Meaning |
|---|---|
| `enablers` | Agents that must each pull their own **lever** (interact on their lever cell) to open gate `k` |
| `finishers` | Agents that must **interact together, in the same step,** on their own goal cells while gate `k` is open |
| `reward` | r_k |
| `partial` | If `False` (all-or-nothing), the event completes only when all finishers interact together. If `True`, it completes when at least one finisher does, and pays r_k × (participating finishers / all finishers). |
| `lever_cells`, `goal_cells` | One cell per enabler / finisher |

Stations are **keyed**: only the designated agent can use its lever or goal cell. This is what makes the coalition planted by identity. Agents in no event are **bystanders**: they can move, and can block others, but can never activate anything.

An agent may appear in several events (**overlap**), but not as both enabler and finisher of the same event.

**Gate timing.** Finishers use the gate state from the *start* of the step. A gate opened by a pull at step t can therefore be used from step t+1, so enablers always act strictly before finishers.

## Delay

When event k completes at step t, its payout r_k (or the partial fraction) is delivered as team reward at step **t + d**.

- If t + d ≥ T, the payout is delivered at the last step T−1 and the event log marks it `clipped`. Configurations should choose T large enough that this does not happen. Tests check for it.
- The observation shows physical facts (gate open, event completed) but **not** the payout countdown. The critic must learn that a completed event will pay later, as with real delayed rewards.

## Removal (the null policy)

`reset(present=mask)` starts an episode with only the agents in `mask`:

- Absent agents are not placed on the grid, cannot block anyone, and their actions are ignored. Their only available action is `noop`.
- Observation vectors keep a fixed size. An absent agent's own observation is all zeros, and other agents see it with presence flag 0 and zeroed position.

With no randomness after `reset`, replaying the same per-agent action sequences gives the same trajectory exactly. This is the **open-loop replay** used for ground truth in Step 3.

## Observations (per agent, fixed size)

| Block | Size |
|---|---|
| One-hot agent id | n |
| Own position (normalised row, col) + own presence | 3 |
| Every agent j: relative position (normalised) + presence | 3n |
| Every event k: gate open, done, I have pulled my lever, I am an enabler, I am a finisher, relative position of my station | 7K |
| Time t / T | 1 |

Total `4n + 7K + 4`.

## Global state (for centralised critics)

All positions (normalised) and presence flags, plus per event: gate open, done, **payout pending**, **steps until payout / T**, and which enablers have pulled. Then t / T. Total `3n + K(4 + n) + 1`.

The payout countdown was added after pilot 1 (`docs/pilot_findings.md`, Problem 1). Without it, the payout step under delay cannot be predicted from the state, so the state is not Markov and feedforward critics hit an irreducible error floor. Agent observations still do **not** show the countdown: agents remain partially observant, and the temporal-credit problem (linking an early action to a late reward) is unchanged.

## Scripted team (Step 3, `whoearned/scripted.py`)

Mode P data comes from a scripted team that reads the environment directly (it is privileged, not learned):

- **Tasks.** Each agent's tasks are its roles across events, in event order. At every step it pursues the first task that is unfinished and *feasible*. A finisher task is feasible only if every enabler of that event is present and, for all-or-nothing events, every co-finisher is present. So under closed-loop removal, agents skip events that can no longer happen instead of waiting forever.
- **Enablers** wait until step `enabler_start`, walk to their lever and pull it.
- **Finishers** walk to their goal cell and interact when (i) the gate is open, (ii) t ≥ `finish_not_before`, and (iii) every present co-finisher is on its goal. Every finisher checks the same condition, so they interact in the same step. `finish_not_before` sets the enabler–finisher gap.
- **Bystanders** have three modes:
  - `"wander"` (default): random moves, but never onto a cell of any coalition member's planned route (start → stations, ignoring other agents). This keeps realised roles equal to planted roles, which Mode P needs.
  - `"roam"`: random moves anywhere. A roaming bystander can stray into a lane, and then removing a *different* bystander can change the outcome. This is the planted-vs-realised-roles case. For example, `running_example` with seed 7 gives a planted bystander a non-zero dividend; there is a test for it.
  - `"idle"`: never moves.
- **Pathing** is breadth-first search around walls and other agents. If no free path exists, the agent follows the path that ignores agents and waits when the next cell is occupied.
- **Exploration noise.** With probability `epsilon`, a coalition member takes a uniformly random action instead.
- **Random-number streams.** Each agent has its own generator, seeded by `(seed, agent)`, and draws the same numbers every step whether it is present or not. Removing one agent therefore changes nothing in the others' randomness; the removal is the only change, as the closed-loop protocol requires.

## Replay ground truth (Step 3, `whoearned/replay.py`)

- **Open-loop (default).** Record the full team's actions once, then replay them for every subset S. Agents outside S follow the null policy:
  - `remove` (default): absent from the grid;
  - `noop`: present but only do nothing, so they can still block others;
  - `random`: present and take uniform random actions from their own fixed stream.
- **Closed-loop (sensitivity check).** Re-run the scripted team for every subset, so the remaining agents can react to who is missing.
- Both produce v(S) for all 2^n subsets, plus **per-event** values v_k(S) (the payout of event k alone). Since the return is the sum of event payouts, v = Σ_k v_k. From these come dividends, Shapley values and leave-one-out marginals, in total and per event. Per-event credit is what the missed-credit metric needs when events overlap.

## Event log

`env.event_log` records, for each event: lever pull times per enabler, gate-open time, completion time, participating finishers, payout time, payout amount, and whether the payout was clipped. Steps 3–4 use it for per-event ground truth and the missed-credit metric.
