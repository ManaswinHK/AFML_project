"""Mode T (Step 8, H4): score the credit of policies trained online by EPyMARL.

For each saved checkpoint of a run:
1. Load the learner (``learner.load_models``) into a harness built as in Mode P.
2. Roll out test episodes with the trained policy as behaviour, mixed with epsilon-uniform noise
   (as the scripted team is in Mode P), so episodes differ and the ground truth has variation.
   Q methods act epsilon-greedily on their utilities; policy methods sample their softmax.
   The behaviour probabilities are recorded, so COMA's baseline uses the policy that acted.
3. Ground truth per episode by open-loop replay with true removal (leave-one-out, Shapley),
   exactly as in Mode P.
4. The method's own read-out (``mode_p.read_credit``) on those episodes, scored end to end
   against leave-one-out.

No per-step (level-1) truth: it would need closed-loop re-runs of the learned policy for every
(step, agent, action), and H4 asks only about return vs end-to-end credit error.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch as th

from whoearned.epymarl_bridge import EpisodeBatch, Harness, build_harness, episodes_to_batch
from whoearned.experiments import MATCHED_TARGET, summarise
from whoearned.metrics import evaluate_credit
from whoearned.mode_p import episode_credit, read_credit
from whoearned.replay import ground_truth_open_loop, rollout


class MacTeam:
    """A trained EPyMARL controller acting in the gridworld, with epsilon-uniform noise."""

    def __init__(self, harness: Harness, epsilon: float = 0.2, seed: int = 0):
        self.h, self.epsilon = harness, epsilon
        self.rng = np.random.default_rng(seed)
        self.q_values = harness.args.agent_output_type == "q"

    def reset(self, env) -> None:
        h = self.h
        self.batch = EpisodeBatch(h.scheme, h.groups, 1, env.config.horizon + 1,
                                  preprocess=h.preprocess, device=h.device)
        h.mac.init_hidden(1)
        self.t = 0

    @th.no_grad()
    def act(self, env) -> list[int]:
        avail = np.array(env.get_avail_actions(), dtype=float)
        self.batch.update({"state": [env.get_state()], "avail_actions": [avail.astype(int).tolist()],
                           "obs": [env.get_obs()]}, ts=self.t)
        out = self.h.mac.forward(self.batch, t=self.t)[0].cpu().numpy()        # (n, A)
        if self.q_values:
            greedy = np.where(avail > 0, out, -np.inf).argmax(-1)
            policy = np.eye(out.shape[-1])[greedy]
        else:
            policy = out * avail
            policy /= policy.sum(-1, keepdims=True)
        uniform = avail / avail.sum(-1, keepdims=True)
        probs = (1 - self.epsilon) * policy + self.epsilon * uniform
        actions = [int(self.rng.choice(len(p), p=p)) for p in probs]
        self.last_probs = probs
        self.batch.update({"actions": [[[a] for a in actions]]}, ts=self.t)
        self.t += 1
        return actions


def checkpoints(model_dir: str) -> list[tuple[int, str]]:
    """(t_env, path) for every saved checkpoint under one run's model directory, in order."""
    out = [(int(os.path.basename(p)), p) for p in glob.glob(os.path.join(model_dir, "*")) if os.path.basename(p).isdigit()]
    return sorted(out)


def evaluate_checkpoint(alg: str, env_args: dict, path: str, n_test: int = 30, epsilon: float = 0.2,
                        seed: int = 0) -> dict:
    """Return, credit error and per-episode scores of one checkpoint."""
    h = build_harness(alg, env_args=env_args, seed=seed)
    h.learner.load_models(path)
    cfg = h.env.config
    team = MacTeam(h, epsilon=epsilon, seed=seed)
    episodes = [rollout(cfg, team) for _ in range(n_test)]
    greedy = rollout(cfg, MacTeam(h, epsilon=0.0, seed=seed))
    batch = episodes_to_batch(h, episodes)
    credit = episode_credit(read_credit(h, alg, batch, episodes)["per_step"])
    rows = []
    for b, ep in enumerate(episodes):
        gt = ground_truth_open_loop(cfg, ep.actions)
        rows.append({"return": ep.total, "matched": evaluate_credit(credit[b], gt, cfg,
                                                                    target=MATCHED_TARGET[alg]).as_dict(),
                     "loo": gt.leave_one_out.tolist(), "credit": credit[b].tolist()})
    return {
        "return_mean": float(np.mean([e.total for e in episodes])),
        "greedy_return": greedy.total,
        "summary": summarise(rows),
        "episodes": rows,
    }


__all__ = ["MacTeam", "checkpoints", "evaluate_checkpoint"]
