"""Build EPyMARL controllers and learners outside its training loop.

Mirrors the setup in ``external/epymarl/src/run.py::run_sequential`` so we can
(i) feed our own episodes (e.g. scripted Mode P data) to EPyMARL learners and
(ii) read credit out of the trained networks (``whoearned.readouts``).
EPyMARL itself is not modified.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EPYMARL_SRC = os.path.join(ROOT, "external", "epymarl", "src")


def ensure_epymarl_importable() -> None:
    """Put EPyMARL's src on sys.path and stub `smaclite` (imported by EPyMARL but unused here)."""
    if not os.path.isdir(EPYMARL_SRC):
        raise ImportError(f"EPyMARL not found at {EPYMARL_SRC}; clone github.com/uoe-agents/epymarl into external/")
    if EPYMARL_SRC not in sys.path:
        sys.path.insert(0, EPYMARL_SRC)
    if importlib.util.find_spec("smaclite") is None:
        sys.modules.setdefault("smaclite", types.ModuleType("smaclite"))


ensure_epymarl_importable()

import torch as th  # noqa: E402
import yaml  # noqa: E402
from components.episode_buffer import EpisodeBatch  # noqa: E402
from components.transforms import OneHot  # noqa: E402
from controllers import REGISTRY as mac_REGISTRY  # noqa: E402
from learners import REGISTRY as le_REGISTRY  # noqa: E402

from whoearned.envs import NOOP, PlantedCoalitionGridworld  # noqa: E402
from whoearned.epymarl_env import PlantedEPyMARLEnv  # noqa: E402
from whoearned.replay import Episode  # noqa: E402


def _load_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _update(d, u):
    for k, v in u.items():
        d[k] = _update(d.get(k, {}), v) if isinstance(v, dict) else v
    return d


class _NullLogger:
    """Stands in for EPyMARL's Logger: learners call log_stat during train()."""

    def __init__(self):
        self.console_logger = SimpleNamespace(info=lambda *a, **k: None)
        self.stats: dict[str, list] = {}

    def log_stat(self, key, value, t):
        self.stats.setdefault(key, []).append((t, value))


@dataclass
class Harness:
    args: Any
    env: PlantedEPyMARLEnv
    scheme: dict
    groups: dict
    preprocess: dict
    mac: Any
    learner: Any
    logger: _NullLogger

    @property
    def device(self) -> str:
        return self.args.device


def load_args(alg: str, env_args: dict | None = None, overrides: dict | None = None) -> SimpleNamespace:
    """default.yaml <- configs/envs/planted.yaml <- algs/<alg>.yaml <- env_args / overrides."""
    config = _load_yaml(os.path.join(EPYMARL_SRC, "config", "default.yaml"))
    config = _update(config, _load_yaml(os.path.join(ROOT, "configs", "envs", "planted.yaml")))
    config = _update(config, _load_yaml(os.path.join(EPYMARL_SRC, "config", "algs", f"{alg}.yaml")))
    if env_args:
        config = _update(config, {"env_args": deepcopy(env_args)})
    if overrides:
        config = _update(config, deepcopy(overrides))
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
    args = SimpleNamespace(**config)
    args.device = "cuda" if args.use_cuda else "cpu"
    return args


def build_harness(alg: str, env_args: dict | None = None, overrides: dict | None = None, seed: int = 0) -> Harness:
    """Controller + learner for ``alg`` on the planted gridworld, set up exactly as run.py does."""
    overrides = {"use_cuda": False, **(overrides or {})}
    args = load_args(alg, env_args, overrides)
    args.seed = seed
    np.random.seed(seed)
    th.manual_seed(seed)

    env_kwargs = {k: v for k, v in args.env_args.items() if k != "seed"}
    env = PlantedEPyMARLEnv(**env_kwargs, seed=seed)
    info = env.get_env_info()
    args.n_agents, args.n_actions, args.state_shape = info["n_agents"], info["n_actions"], info["state_shape"]

    scheme = {
        "state": {"vshape": info["state_shape"]},
        "obs": {"vshape": info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (info["n_actions"],), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "reward": {"vshape": (1,)},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    # The learner needs the scheme *after* preprocessing (with actions_onehot and filled).
    template = EpisodeBatch(scheme, groups, 1, info["episode_limit"] + 1, preprocess=preprocess, device="cpu")
    mac = mac_REGISTRY[args.mac](template.scheme, groups, args)
    logger = _NullLogger()
    learner = le_REGISTRY[args.learner](mac, template.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()
    return Harness(args, env, scheme, groups, preprocess, mac, learner, logger)


def episodes_to_batch(harness: Harness, episodes: list[Episode]) -> EpisodeBatch:
    """Turn recorded episodes into an EpisodeBatch laid out exactly as EpisodeRunner fills it.

    Each episode is replayed in the gridworld to recover per-step state, obs and
    available actions. Slot T holds the final observation; its action is NOOP
    (the runner also stores an action there, which learners mask out).
    """
    if not episodes:
        raise ValueError("need at least one episode")
    cfg = episodes[0].config
    T = cfg.horizon
    batch = EpisodeBatch(
        harness.scheme, harness.groups, len(episodes), T + 1,
        preprocess=harness.preprocess, device=harness.device,
    )
    for b, ep in enumerate(episodes):
        if ep.config != cfg:
            raise ValueError("all episodes must share one configuration")
        env = PlantedCoalitionGridworld(cfg)
        env.reset(ep.present)
        for t in range(T):
            batch.update(
                {
                    "state": [env.get_state()],
                    "avail_actions": [env.get_avail_actions()],
                    "obs": [env.get_obs()],
                },
                bs=slice(b, b + 1), ts=t,
            )
            _, r, done, _ = env.step(ep.actions[t])
            if r != ep.rewards[t]:
                raise ValueError(f"episode {b} does not replay to its recorded rewards at t={t}")
            batch.update(
                {
                    "actions": [[[int(a)] for a in ep.actions[t]]],
                    "reward": [(float(r),)],
                    "terminated": [(bool(done),)],
                },
                bs=slice(b, b + 1), ts=t,
            )
        batch.update(
            {
                "state": [env.get_state()],
                "avail_actions": [env.get_avail_actions()],
                "obs": [env.get_obs()],
                "actions": [[[NOOP]] * cfg.n_agents],
            },
            bs=slice(b, b + 1), ts=T,
        )
    return batch
