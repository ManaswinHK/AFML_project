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
