"""NA²Q (Liu, Zhu & Chen, ICML 2023) for Mode P, plus its coalition read-out.

A clean reimplementation that follows the official code, github.com/zichuan-liu/NA2Q at
commit adc10b4 (src/modules/mixers/qnam.py, src/learners/qnam_learner.py,
src/modules/agents/rnn_sd_agent.py, src/modules/intrinsic/qnam_context.py, and
config/algs/qnam.yaml). The repository has no licence file, so no code is copied. The
architecture, losses and hyperparameters match the official ones except where marked
"Deviation".

Q_tot(s, a) = sum_k w_k(s, z) f_k(Q_{S_k}) + v(s), where k runs over every agent
(order 1) and every pair (order 2). Each f_k is a small MLP with non-negative weights,
so it is monotone. The weights w are a softmax attention over terms, with the query
from the state and the key from the agents' VAE latents z.

Deviation, Mode P only: ``target="sarsa"`` bootstraps from the data's next joint action
instead of the double-Q max. On fixed scripted data the official max target diverges:
- On 200 noisy episodes, Q_tot(t=0) reaches about 115–140 against a true discounted return
  of about 8.
- The max picks per-agent actions the data never takes, and the per-agent heads and
  monotone pairwise terms amplify their overestimates.
- The SARSA target converges to the right values on the same data and network.

With "sarsa", NA²Q estimates the behaviour policy's Q (as COMA's critic does), not Q*.
``target="max"`` restores the official update.
"""

from __future__ import annotations

import copy
import math
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from whoearned.envs import NOOP
from whoearned.epymarl_bridge import EpisodeBatch
from whoearned.ground_truth import harsanyi_dividends

DEFAULTS = dict(
    rnn_hidden_dim=64, hypernet_embed=64, mixing_embed_dim=32, mix_nary=(1, 2),
    term_hidden_dims=(8, 4), lr=5e-4, optim_alpha=0.99, optim_eps=1e-5, grad_norm_clip=10.0,
    gamma=0.99, double_q=True, target_update_interval=200, beta_vae=0.1, target="max",
)


class Agent(nn.Module):
    """rnn_sd agent: shared GRU, Q = shared head + per-agent head. Input: obs + last action."""

    def __init__(self, input_dim: int, n_agents: int, n_actions: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.fc2 = nn.Linear(hidden, n_actions)
        self.local = nn.ModuleList(nn.Linear(hidden, n_actions) for _ in range(n_agents))

    def forward(self, inputs: th.Tensor):
        """inputs (B, n, T, d) -> Q (B, T, n, A), hidden (B, T, n, h)."""
        B, n, T, _ = inputs.shape
        x = F.relu(self.fc1(inputs.reshape(B * n, T, -1)))
        h, _ = self.rnn(x)                                        # (B*n, T, h)
        h = h.reshape(B, n, T, -1)
        local = th.stack([head(h[:, i]) for i, head in enumerate(self.local)], dim=1)
        q = self.fc2(h) + local                                   # (B, n, T, A)
        return q.permute(0, 2, 1, 3), h.permute(0, 2, 1, 3)


class VAE(nn.Module):
    """Encodes each agent's GRU state; the decoder reconstructs a mask over its observation."""

    def __init__(self, h_dim: int, obs_dim: int, hidden: int = 32):
        super().__init__()
        self.e1, self.e2 = nn.Linear(h_dim, hidden), nn.Linear(hidden, hidden)
        self.mean, self.log_std = nn.Linear(hidden, h_dim), nn.Linear(hidden, h_dim)
        self.d1, self.d2, self.d3 = nn.Linear(h_dim, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, obs_dim)

    def encode(self, h):
        z = F.relu(self.e2(F.relu(self.e1(h))))
        mean, std = self.mean(z), th.exp(self.log_std(z))
        return mean + std * th.randn_like(std), mean, std

    def forward(self, h):
        z, mean, std = self.encode(h)
        return th.sigmoid(self.d3(F.relu(self.d2(F.relu(self.d1(z)))))), mean, std


class _Terms(nn.Module):
    """One monotone MLP per coalition of a given order (grouped 1x1 convs, |weights| as in the original)."""

    def __init__(self, order: int, count: int, hidden_dims):
        super().__init__()
        dims = [order, *hidden_dims]
        self.layers = nn.ModuleList(
            nn.Conv1d(a * count, b * count, kernel_size=1, groups=count) for a, b in zip(dims[:-1], dims[1:])
        )
        self.last = nn.Conv1d(dims[-1] * count, count, kernel_size=1, groups=count)

    def forward(self, x):  # x (B, count * order, 1) -> (B, count)
        for layer in self.layers:
            with th.no_grad():
                layer.weight.abs_()
            x = F.elu(layer(x))
        with th.no_grad():
            self.last.weight.abs_()
        return self.last(x).squeeze(-1)


class Mixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, a: SimpleNamespace):
        super().__init__()
        self.n_agents, self.h = n_agents, a.rnn_hidden_dim
        self.coalitions = {o: list(combinations(range(n_agents), o)) for o in a.mix_nary}
        self.terms = nn.ModuleDict({str(o): _Terms(o, len(c), a.term_hidden_dims) for o, c in self.coalitions.items()})
        self.n_terms = sum(len(c) for c in self.coalitions.values())
        self.query = nn.Linear(state_dim, a.hypernet_embed)
        self.key = nn.Linear(self.h * n_agents, a.hypernet_embed)
        self.to_keys = nn.Linear(a.hypernet_embed, self.n_terms, bias=False)
        self.to_queries = nn.Linear(a.hypernet_embed, self.n_terms * self.n_terms, bias=False)
        self.v = nn.Sequential(
            nn.Linear(state_dim, a.hypernet_embed), nn.ReLU(),
            nn.Linear(a.hypernet_embed, a.mixing_embed_dim), nn.ReLU(), nn.Linear(a.mixing_embed_dim, 1),
        )

    @property
    def term_coalitions(self) -> list[tuple[int, ...]]:
        return [g for o in self.coalitions for g in self.coalitions[o]]

    def parts(self, qs, states, latent):
        """qs (..., n), states (..., s), latent (..., n, h) -> term outputs f (N, K), weights w (N, K), v (N,)."""
        lead = qs.shape[:-1]
        qs, states = qs.reshape(-1, self.n_agents), states.reshape(-1, states.shape[-1])
        latent = latent.reshape(-1, self.n_agents * self.h)
        f = th.cat([self.terms[str(o)](qs[:, [i for g in c for i in g]].unsqueeze(-1))
                    for o, c in self.coalitions.items()], dim=-1)
        K, N = self.n_terms, qs.shape[0]
        keys = F.relu(self.to_keys(self.key(latent))).view(N, K, 1)
        queries = self.to_queries(self.query(states)).view(N, K, K)
        w = F.softmax(th.bmm(queries, keys) / math.sqrt(K), dim=1).squeeze(-1)
        return f, w, self.v(states).squeeze(-1), lead

    def forward(self, qs, states, latent):
        f, w, v, lead = self.parts(qs, states, latent)
        return ((f * w).sum(-1) + v).reshape(*lead, 1)


class NA2Q:
    """Agents + VAE + mixer, trained with the official qnam_learner loss on a fixed EpisodeBatch."""

    def __init__(self, n_agents: int, n_actions: int, obs_dim: int, state_dim: int, seed: int = 0, **overrides):
        self.args = SimpleNamespace(**{**DEFAULTS, **overrides})
        a = self.args
        th.manual_seed(seed)
        self.n_agents, self.n_actions = n_agents, n_actions
        self.agent = Agent(obs_dim + n_actions, n_agents, n_actions, a.rnn_hidden_dim)
        self.vae = VAE(a.rnn_hidden_dim, obs_dim)
        self.mixer = Mixer(n_agents, state_dim, a)
        self.target_agent, self.target_vae, self.target_mixer = (copy.deepcopy(m) for m in (self.agent, self.vae, self.mixer))
        self.agent_params = list(self.agent.parameters())
        self.mixer_params = list(self.mixer.parameters()) + list(self.vae.parameters())
        self.opt_agent = th.optim.RMSprop(self.agent_params, lr=a.lr, alpha=a.optim_alpha, eps=a.optim_eps)
        self.opt_mixer = th.optim.Adam(self.mixer_params, lr=a.lr)
        self.last_target_update = 0

    def _inputs(self, batch: EpisodeBatch) -> th.Tensor:
        """(B, n, T+1, obs + A): observation and one-hot previous action (zeros at t = 0)."""
        onehot = batch["actions_onehot"]
        last = th.cat([th.zeros_like(onehot[:, :1]), onehot[:, :-1]], dim=1)
        return th.cat([batch["obs"], last], dim=-1).permute(0, 2, 1, 3)

    def train(self, batch: EpisodeBatch, episode_num: int) -> float:
        a = self.args
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail = batch["avail_actions"]
        inputs = self._inputs(batch)

        q, hidden = self.agent(inputs)
        obs = batch["obs"][:, :-1]
        recon, mean, std = self.vae(hidden[:, :-1])
        vae_loss = (F.mse_loss(recon * obs, obs)
                    - 0.5 * (1 + th.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
                    + F.l1_loss(recon, th.zeros_like(recon)))
        with th.no_grad():
            latent, _, _ = self.vae.encode(hidden[:, :-1])
        chosen = th.gather(q[:, :-1], 3, actions).squeeze(3)

        with th.no_grad():
            target_q, target_hidden = self.target_agent(inputs)
            target_latent, _, _ = self.target_vae.encode(target_hidden[:, 1:])
            target_q = target_q[:, 1:]
            target_q[avail[:, 1:] == 0] = -9999999
            if a.target == "sarsa":
                target_max = th.gather(target_q, 3, batch["actions"][:, 1:]).squeeze(3)
            elif a.double_q:
                live = q.detach().clone()
                live[avail == 0] = -9999999
                target_max = th.gather(target_q, 3, live[:, 1:].argmax(3, keepdim=True)).squeeze(3)
            else:
                target_max = target_q.max(3)[0]
            target_tot = self.target_mixer(target_max, batch["state"][:, 1:], target_latent)

        q_tot = self.mixer(chosen, batch["state"][:, :-1], latent)
        td = (q_tot - (rewards + a.gamma * (1 - terminated) * target_tot)) * mask
        loss = (td ** 2).sum() / mask.sum() + a.beta_vae * vae_loss

        self.opt_agent.zero_grad()
        self.opt_mixer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent_params, a.grad_norm_clip)
        nn.utils.clip_grad_norm_(self.mixer_params, a.grad_norm_clip)
        self.opt_agent.step()
        self.opt_mixer.step()
        if episode_num - self.last_target_update >= a.target_update_interval:
            for src, dst in ((self.agent, self.target_agent), (self.vae, self.target_vae), (self.mixer, self.target_mixer)):
                dst.load_state_dict(src.state_dict())
            self.last_target_update = episode_num
        return float(loss.item())

    @th.no_grad()
    def mixer_game(self, batch: EpisodeBatch, null_action: int = NOOP) -> np.ndarray:
        """Per-step coalition values (B, T, 2^n): Q_tot with agents outside S switched to the null
        action, minus Q_tot with every agent null.

        Deviation: the latent is the VAE *mean*, not a sample, so the read-out is deterministic.
        """
        q, hidden = self.agent(self._inputs(batch))
        _, latent, _ = self.vae.encode(hidden[:, :-1])
        q = q[:, :-1]
        chosen = th.gather(q, 3, batch["actions"][:, :-1]).squeeze(3)
        return coalition_game(lambda qs: self.mixer(qs, batch["state"][:, :-1], latent).squeeze(-1),
                              chosen, q[..., null_action])


def coalition_game(mix, chosen: th.Tensor, null: th.Tensor) -> np.ndarray:
    """g_t(S) = mix(Q with agents outside S at their null-action value) - mix(all null), shape (B, T, 2^n).

    ``mix`` maps (B, T, n) utilities to (B, T) joint values (a learned mixer with its inputs bound).
    """
    n = chosen.shape[-1]
    values = []
    for s in range(2 ** n):
        member = th.tensor([(s >> i) & 1 for i in range(n)], dtype=th.bool, device=chosen.device)
        values.append(mix(th.where(member, chosen, null)))
    g = th.stack(values, dim=-1)
    return (g - g[..., :1]).cpu().numpy()


def game_dividends(game: np.ndarray) -> np.ndarray:
    """Episode-summed Harsanyi dividends (B, 2^n) of a per-step game (B, T, 2^n)."""
    total = game.sum(axis=1)
    return np.stack([harsanyi_dividends(v) for v in total])


def game_shapley_per_step(game: np.ndarray) -> np.ndarray:
    """Per-step Shapley values (B, T, n) of a per-step game, from its dividends: phi_i = sum_{S ∋ i} d_S / |S|."""
    B, T, size = game.shape
    n = size.bit_length() - 1
    d = np.apply_along_axis(harsanyi_dividends, -1, game)
    out = np.zeros((B, T, n))
    for s in range(1, size):
        members = [i for i in range(n) if (s >> i) & 1]
        out[..., members] += d[..., s:s + 1] / len(members)
    return out


__all__ = ["NA2Q", "coalition_game", "game_dividends", "game_shapley_per_step"]
