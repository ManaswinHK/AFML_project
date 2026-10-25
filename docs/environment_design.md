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

All positions (normalised) and presence flags, plus per event: gate open, done, and which enablers have pulled. Then t / T. Total `3n + K(2 + n) + 1`.

## Event log

`env.event_log` records, for each event: lever pull times per enabler, gate-open time, completion time, participating finishers, payout time, payout amount, and whether the payout was clipped. Steps 3–4 use it for per-event ground truth and the missed-credit metric.
