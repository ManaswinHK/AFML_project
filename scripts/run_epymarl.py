"""Run an EPyMARL algorithm on the Planted Coalition Gridworld.

Mirrors ``external/epymarl/src/main.py`` but (i) registers our environment as
``"planted"``, (ii) reads env configs from ``configs/envs`` in this repo, and
(iii) writes results to ``results/`` in this repo. EPyMARL itself is unmodified.

Usage (from the project root):

    .venv/Scripts/python scripts/run_epymarl.py --config=vdn --env-config=planted with t_max=20000

Anything after ``with`` is a sacred config override, e.g. ``seed=3``,
``env_args.layout_kwargs.delay=25``, ``runner=episode batch_size_run=1``.
"""

import os
import sys
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPYMARL_SRC = os.path.join(ROOT, "external", "epymarl", "src")
sys.path[:0] = [EPYMARL_SRC, os.path.join(ROOT, "src")]

import importlib.util  # noqa: E402
import types  # noqa: E402

# EPyMARL's envs package imports `smaclite` only to register SMAClite environments,
# which we never use. Stub it if it is not installed. This runs at module level so
# that spawned worker processes (parallel runner on Windows) get the stub too.
if importlib.util.find_spec("smaclite") is None:
    sys.modules["smaclite"] = types.ModuleType("smaclite")

import yaml  # noqa: E402
from sacred import SETTINGS  # noqa: E402
from sacred.observers import FileStorageObserver  # noqa: E402

import envs as epymarl_envs  # noqa: E402
from whoearned.epymarl_env import make_env  # noqa: E402

epymarl_envs.REGISTRY["planted"] = make_env

import main as epymarl_main  # noqa: E402  (defines the sacred Experiment `ex`)

SETTINGS["CAPTURE_MODE"] = "sys"  # "fd" capture is unreliable on Windows


def _load_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _pop_arg(params, name):
    for i, p in enumerate(params):
        if p.split("=")[0] == name:
            del params[i]
            return p.split("=", 1)[1]
    return None


def main(argv):
    params = deepcopy(argv)
    alg = _pop_arg(params, "--config")
    env = _pop_arg(params, "--env-config") or "planted"
    if alg is None:
        raise SystemExit("pass --config=<algorithm>, e.g. --config=vdn")

    config = _load_yaml(os.path.join(EPYMARL_SRC, "config", "default.yaml"))
    config = epymarl_main.recursive_dict_update(config, _load_yaml(os.path.join(ROOT, "configs", "envs", f"{env}.yaml")))
    config = epymarl_main.recursive_dict_update(config, _load_yaml(os.path.join(EPYMARL_SRC, "config", "algs", f"{alg}.yaml")))
    config["local_results_path"] = os.path.join(ROOT, "results")

    ex = epymarl_main.ex
    ex.add_config(config)
    layout = config["env_args"]["map_name"]
    for p in params:
        if p.startswith("env_args.map_name="):
            layout = p.split("=", 1)[1]
    ex.observers.append(FileStorageObserver.create(os.path.join(ROOT, "results", "sacred", config["name"], layout)))
    ex.run_commandline(params)


if __name__ == "__main__":
    import torch as th

    th.set_num_threads(1)
    main(sys.argv)
