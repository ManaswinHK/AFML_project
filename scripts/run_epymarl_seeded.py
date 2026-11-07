"""Online EPyMARL training with a replay buffer seeded by scripted-team episodes (Mode T, H4).

Same as ``scripts/run_epymarl.py`` (it imports and reuses it; EPyMARL stays unmodified), plus two flags
that must come before ``with``:

    --seed-episodes=N     scripted full-team episodes put in the replay buffer before training starts
    --seed-epsilon=E      their exploration noise (default 0.2, as in Mode P)

    .venv312/Scripts/python scripts/run_epymarl_seeded.py --config=qmix --env-config=planted_coalition \
        --seed-episodes=200 with seed=0 t_max=400000 use_cuda=False buffer_size=20000 \
        env_args.layout_kwargs.delay=0

The seeded episodes are converted by ``whoearned.epymarl_bridge.episodes_to_batch``. They stay in the
buffer until it wraps, so give ``buffer_size`` room (the QMIX default of 5000 episodes would overwrite
them after about 5000 online episodes). Only off-policy learners (VDN, QMIX) use a replay buffer across
updates; COMA and MAPPO clear theirs, so seeding has no effect on them.
"""

import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_epymarl as base  # noqa: E402  (sets up paths, the sacred patches and EPyMARL imports)
import importlib.machinery  # noqa: E402

# run_epymarl stubs `smaclite` without a __spec__, which makes importlib.util.find_spec raise later.
if "smaclite" in sys.modules and sys.modules["smaclite"].__spec__ is None:
    sys.modules["smaclite"].__spec__ = importlib.machinery.ModuleSpec("smaclite", None)

from components.episode_buffer import ReplayBuffer  # noqa: E402

from whoearned.envs.layouts import coalition_layout  # noqa: E402
from whoearned.epymarl_bridge import episodes_to_batch  # noqa: E402
from whoearned.replay import rollout  # noqa: E402
from whoearned.scripted import ScriptedTeam  # noqa: E402

SEEDING = {"episodes": 0, "epsilon": 0.2, "layout_kwargs": {}, "seed": 0}
SEED_BASE = 900_000  # disjoint from Mode P's training (s * 100000 + i) and test (+ 50000 + i) seeds


class SeededReplayBuffer(ReplayBuffer):
    """A ReplayBuffer that starts with scripted-team episodes in it."""

    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super().__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        n = SEEDING["episodes"]
        if n <= 0:
            return
        if n > buffer_size:
            raise ValueError(f"seed-episodes={n} exceeds buffer_size={buffer_size}")
        cfg = coalition_layout(**SEEDING["layout_kwargs"])
        base_seed = SEED_BASE + 1000 * SEEDING["seed"]
        episodes = [rollout(cfg, ScriptedTeam(cfg, seed=base_seed + i, epsilon=SEEDING["epsilon"])) for i in range(n)]
        holder = SimpleNamespace(scheme=scheme, groups=groups, preprocess=preprocess, device="cpu")
        self.insert_episode_batch(episodes_to_batch(holder, episodes))
        returns = [e.total for e in episodes]
        print(f"seeded the replay buffer with {n} scripted episodes (mean return {sum(returns) / n:.2f})", flush=True)


base.epymarl_run.ReplayBuffer = SeededReplayBuffer


def main(argv):
    params = list(argv)
    n = base._pop_arg(params, "--seed-episodes")
    eps = base._pop_arg(params, "--seed-epsilon")
    SEEDING["episodes"] = int(n) if n else 0
    SEEDING["epsilon"] = float(eps) if eps else 0.2
    config = base._load_yaml(os.path.join(ROOT, "configs", "envs", "planted_coalition.yaml"))
    lk = dict(config["env_args"]["layout_kwargs"])
    for p in params:
        for key in ("n_bystanders", "delay", "horizon"):
            if p.startswith(f"env_args.layout_kwargs.{key}="):
                lk[key] = int(p.split("=", 1)[1])
        if p.startswith("seed="):
            SEEDING["seed"] = int(p.split("=", 1)[1])
    SEEDING["layout_kwargs"] = lk
    base.main(params)


if __name__ == "__main__":
    import torch as th

    th.set_num_threads(1)
    main(sys.argv)
