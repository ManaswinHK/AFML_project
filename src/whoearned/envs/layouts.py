"""Hand-built layouts. Agents A..E are indices 0..4.

running_example (5 x 10), matching Section 1 of the problem statement:

    A...l.....      A pulls lever (0,4), which opens the gate
    ..........
    B.......g.      B and C interact together on goals (2,8), (3,8)
    C.......g.
    ....D.E...      D and E are bystanders

overlap_example (5 x 10): B belongs to two events.

    A...l.....      event 0: enabler A, finishers B, C, reward 6
    ..........
    B....g..g.      event 1: enabler D, finisher B,   reward 4
    C....g....
    D...l....E      E is a bystander
"""

from whoearned.envs.gridworld import EventSpec, GridConfig

A, B, C, D, E = range(5)

# Bystander start cells, in order; none lies on a coalition route, and bystanders
# cannot reach row 1 without crossing the finishers' lanes.
BYSTANDER_SLOTS = ((4, 4), (4, 6), (4, 2), (4, 8), (4, 0))


def coalition_layout(
    n_bystanders: int = 2, delay: int = 15, horizon: int = 60, partial: bool = False, role_perm=None
) -> GridConfig:
    """The running-example coalition (A enables, B and C finish) plus 0-5 bystanders.

    Agents 0-2 are A, B, C; agents 3.. are bystanders on row 4. With
    n_bystanders=2 this is exactly :func:`running_example`.

    ``role_perm`` (v3 Phase 2.1, shuffled roles): a permutation with ``role_perm[role] = agent index``,
    where roles are 0 = enabler, 1-2 = finishers, 3.. = bystanders. Start cells move with the role; the
    lever and goal cells stay where they are. ``None`` is the identity (the fixed layout).
    """
    if not 0 <= n_bystanders <= len(BYSTANDER_SLOTS):
        raise ValueError(f"n_bystanders must be in 0..{len(BYSTANDER_SLOTS)}")
    n = 3 + n_bystanders
    perm = list(range(n)) if role_perm is None else [int(x) for x in role_perm]
    if sorted(perm) != list(range(n)):
        raise ValueError(f"role_perm must be a permutation of 0..{n - 1}")
    role_starts = ((0, 0), (2, 0), (3, 0), *BYSTANDER_SLOTS[:n_bystanders])
    starts = [None] * n
    for role, agent in enumerate(perm):
        starts[agent] = role_starts[role]
    a, b, c = perm[0], perm[1], perm[2]
    event = EventSpec(
        enablers=(a,),
        finishers=tuple(sorted((b, c))),
        reward=10.0,
        lever_cells={a: (0, 4)},
        goal_cells={b: (2, 8), c: (3, 8)},
        partial=partial,
    )
    return GridConfig(
        n_agents=n,
        height=5,
        width=10,
        horizon=horizon,
        delay=delay,
        start_cells=tuple(starts),
        events=(event,),
    )


def running_example(delay: int = 15, horizon: int = 60, partial: bool = False) -> GridConfig:
    event = EventSpec(
        enablers=(A,),
        finishers=(B, C),
        reward=10.0,
        lever_cells={A: (0, 4)},
        goal_cells={B: (2, 8), C: (3, 8)},
        partial=partial,
    )
    return GridConfig(
        n_agents=5,
        height=5,
        width=10,
        horizon=horizon,
        delay=delay,
        start_cells=((0, 0), (2, 0), (3, 0), (4, 4), (4, 6)),
        events=(event,),
    )


def overlap_example(delay: int = 15, horizon: int = 60) -> GridConfig:
    event0 = EventSpec(
        enablers=(A,),
        finishers=(B, C),
        reward=6.0,
        lever_cells={A: (0, 4)},
        goal_cells={B: (2, 5), C: (3, 5)},
    )
    event1 = EventSpec(
        enablers=(D,),
        finishers=(B,),
        reward=4.0,
        lever_cells={D: (4, 4)},
        goal_cells={B: (2, 8)},
    )
    return GridConfig(
        n_agents=5,
        height=5,
        width=10,
        horizon=horizon,
        delay=delay,
        start_cells=((0, 0), (2, 0), (3, 0), (4, 0), (4, 9)),
        events=(event0, event1),
    )


def multi_event_layout(n_events: int = 2, delay: int = 0, horizon: int = 25, lane_length: int = 4,
                       n_bystanders: int = 0) -> GridConfig:
    """K walled lanes, each with one enabler and one finisher (v3 Phase 3, the headroom gate).

    Lane k uses rows 3k (enabler) and 3k + 1 (finisher); row 3k + 2 is a wall between lanes. The enabler
    (agent 2k) starts at column 0 and pulls a lever at column ``lane_length - 2``; the finisher (agent
    2k + 1) starts at column 0 and interacts on its goal at column ``lane_length - 1`` once the gate is
    open. Each event pays 10. Lanes cannot interact (walls), so the team reward is a sum of K
    independent events and each agent's exact difference reward is its own lane's reward.

    K = 2, lane_length = 4:
        A.l.      A pulls the lever (0, 2)
        B..g      B interacts on (1, 3) after the gate opens
        ####
        C.l.
        D..g
    """
    if n_events < 1 or lane_length < 3:
        raise ValueError("need n_events >= 1 and lane_length >= 3")
    if not 0 <= n_bystanders <= lane_length:
        raise ValueError("need 0 <= n_bystanders <= lane_length")
    # Optional bystanders (v3 Phase 4): agents 2K.. in their own walled row below the lanes, with no task.
    height = 3 * n_events - 1 + (2 if n_bystanders else 0)
    walls = frozenset((3 * k + 2, c) for k in range(n_events - 1 + (1 if n_bystanders else 0))
                      for c in range(lane_length))
    events, starts = [], []
    for k in range(n_events):
        e, f = 2 * k, 2 * k + 1
        events.append(EventSpec(enablers=(e,), finishers=(f,), reward=10.0,
                                lever_cells={e: (3 * k, lane_length - 2)}, goal_cells={f: (3 * k + 1, lane_length - 1)}))
        starts += [(3 * k, 0), (3 * k + 1, 0)]
    starts += [(3 * n_events, c) for c in range(n_bystanders)]
    return GridConfig(n_agents=2 * n_events + n_bystanders, height=height, width=lane_length, horizon=horizon, delay=delay,
                      start_cells=tuple(starts), events=tuple(events), walls=walls)
