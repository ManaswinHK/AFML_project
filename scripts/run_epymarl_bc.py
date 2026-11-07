"""Online EPyMARL training for the on-policy learners (COMA, MAPPO) after behavioural-cloning pretraining.

Plain online training never earns reward on this task (docs/mode_t_results.md §1), and seeding the replay buffer
(``run_epymarl_seeded.py``) cannot help COMA and MAPPO because they clear their buffer after every update. This launcher
instead initialises them from demonstrations before the first online step:

1. **Actor**: behavioural cloning (cross-entropy) of the shared actor on N scripted full-team episodes;
2. **Critic**: a warm-up on the same episodes, by running the learner's own ``train`` with the actor's learning rate set
   to 0, so only the critic (and its target and reward normaliser) learns;
3. online training then proceeds unchanged.

EPyMARL stays unmodified; the learner registry and replay buffer are patched from here. Extra flags (before ``with``):

    --bc-episodes=N       scripted episodes used for both steps (default 200; 0 disables pretraining)
    --bc-epsilon=E        their exploration noise (default 0.2, as in Mode P)
    --bc-epochs=K         passes over the episodes for the actor (default 100)
    --critic-warmup=K     critic updates on the scripted episodes (default 300)

    .venv312/Scripts/python scripts/run_epymarl_bc.py --config=coma --env-config=planted_coalition \
        --bc-episodes=200 with seed=0 t_max=400000 use_cuda=False env_args.layout_kwargs.delay=0
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_epymarl_seeded as seeded  # noqa: E402  (also imports run_epymarl and sets up the EPyMARL paths)

base = seeded.base

import torch as th  # noqa: E402
from components.episode_buffer import ReplayBuffer  # noqa: E402
from learners import REGISTRY as LEARNERS  # noqa: E402

from whoearned.envs.layouts import coalition_layout  # noqa: E402
from whoearned.epymarl_bridge import episodes_to_batch  # noqa: E402
from whoearned.replay import rollout  # noqa: E402
from whoearned.scripted import ScriptedTeam  # noqa: E402
from types import SimpleNamespace  # noqa: E402

BC = {"episodes": 0, "epsilon": 0.2, "epochs": 100, "critic_warmup": 300, "batch": 25, "lr": 1e-3}
RECORD = {}  # scheme, groups, preprocess and episode length, as run.py builds them


class RecordingBuffer(ReplayBuffer):
    """The run's ReplayBuffer, remembering how it was built so the demonstrations can use the same layout."""

    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super().__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        RECORD.update(scheme=scheme, groups=groups, preprocess=preprocess, max_seq_length=max_seq_length)


def demonstrations():
    n = BC["episodes"]
    cfg = coalition_layout(**seeded.SEEDING["layout_kwargs"])
    base_seed = seeded.SEED_BASE + 1000 * seeded.SEEDING["seed"]
    episodes = [rollout(cfg, ScriptedTeam(cfg, seed=base_seed + i, epsilon=BC["epsilon"])) for i in range(n)]
    holder = SimpleNamespace(scheme=RECORD["scheme"], groups=RECORD["groups"], preprocess=RECORD["preprocess"],
                             device="cpu")
    buf = ReplayBuffer(RECORD["scheme"], RECORD["groups"], n, RECORD["max_seq_length"],
                       preprocess=RECORD["preprocess"], device="cpu")
    buf.insert_episode_batch(episodes_to_batch(holder, episodes))
    return buf, sum(e.total for e in episodes) / n


def clone_actor(mac, buf):
    """Cross-entropy of the actor's action probabilities against the scripted team's actions."""
    opt = th.optim.Adam(mac.parameters(), lr=BC["lr"])
    steps = max(1, BC["episodes"] // BC["batch"])
    loss = None
    for epoch in range(BC["epochs"]):
        total = 0.0
        for _ in range(steps):
            batch = buf.sample(BC["batch"])
            mac.init_hidden(batch.batch_size)
            probs = th.stack([mac.forward(batch, t=t) for t in range(batch.max_seq_length - 1)], dim=1)
            actions = batch["actions"][:, :-1]
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - batch["terminated"][:, :-1].float()[:, :-1])
            nll = -th.log(th.gather(probs, 3, actions).squeeze(3) + 1e-10)
            loss = (nll * mask).sum() / (mask.sum() * probs.size(2))
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(mac.parameters(), 10.0)
            opt.step()
            total += loss.item()
        if epoch % 20 == 0 or epoch == BC["epochs"] - 1:
            print(f"BC epoch {epoch}: action NLL {total / steps:.3f}", flush=True)


def warm_critic(learner, args, buf):
    """Train only the critic: the learner's own update with the actor's learning rate at 0."""
    groups = learner.agent_optimiser.param_groups
    saved = [g["lr"] for g in groups]
    for g in groups:
        g["lr"] = 0.0
    for i in range(BC["critic_warmup"]):
        learner.train(buf.sample(args.batch_size), 0, i)
    for g, lr in zip(groups, saved):
        g["lr"] = lr
    learner.agent_optimiser = type(learner.agent_optimiser)(learner.agent_params, lr=args.lr)  # drop warm-up moments


def with_pretraining(factory):
    def make(mac, scheme, logger, args):
        learner = factory(mac, scheme, logger, args)
        if BC["episodes"] > 0:
            buf, mean_return = demonstrations()
            print(f"pretraining on {BC['episodes']} scripted episodes (mean return {mean_return:.2f})", flush=True)
            clone_actor(mac, buf)
            warm_critic(learner, args, buf)
            print("pretraining done", flush=True)
        return learner
    return make


base.epymarl_run.ReplayBuffer = RecordingBuffer
base.epymarl_run.le_REGISTRY = {k: with_pretraining(v) for k, v in LEARNERS.items()}


def main(argv):
    params = list(argv)
    for flag, key, cast in (("--bc-episodes", "episodes", int), ("--bc-epsilon", "epsilon", float),
                            ("--bc-epochs", "epochs", int), ("--critic-warmup", "critic_warmup", int)):
        v = base._pop_arg(params, flag)
        if v is not None:
            BC[key] = cast(v)
    seeded.main(params)


if __name__ == "__main__":
    th.set_num_threads(1)
    main(sys.argv)
