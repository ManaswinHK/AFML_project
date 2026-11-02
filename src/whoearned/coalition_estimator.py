"""Low-order episodic coalition estimator (problem statement v2, H5), for Mode P.

The return model is a sum of coalition terms of order at most ``k``:

    R_hat(tau) = sum_{S subset of P, 1 <= |S| <= k} g_S(h_S),

where P is the set of agents present in the episode and h_i encodes agent i's identity and its
own *action* sequence, nothing else. Because a term sees only its members, removing an agent
removes exactly the terms it belongs to. The model's coalition game is therefore
v_hat(M) = sum_{S subset of M} g_S, and its Harsanyi dividends are the terms themselves:
d_hat(S) = g_S for |S| <= k and 0 above. Individual credit is the Shapley value of that game,
phi_i = sum_{S containing i} g_S / |S| (the equal split of each dividend among its members).

Why actions only (docs/h5_preregistration.md, pilot): the ground truth replays the recorded
actions open loop with agents removed, so actions are what stays fixed under removal. Positions
and event flags do not: a finisher's own path shows whether the enabler opened the gate, so a
model that sees them fits an order-3 coalition with a pair of finishers and never needs A.

Training: MSE between R_hat and the episode return, plus ``sparsity`` times sum_S (|g_S| + eps)^p
over the present terms (p = 1 is L1). Identification needs data in which presence varies while
actions stay fixed: the ``_mix`` data adds one open-loop replay of each training episode with a
random subset removed (``DataCell.episodes("mix")``). On full-team data with a constant return,
nothing identifies the split (any set of terms summing to the return fits); that is the expected
failure, run on purpose as a control.

Terms of the same order share weights and are symmetric in their members (sums and
elementwise products of member encodings), so the parameter count does not grow with n.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from whoearned.epymarl_bridge import EpisodeBatch
from whoearned.ground_truth import mask_of

DEFAULTS = dict(hidden_size=64, lr=1e-3, sparsity=0.05, sparsity_p=1.0, sparsity_eps=0.0)


def own_feature_index(n_agents: int, n_events: int) -> list[int]:
    """Observation entries the encoder sees: the agent's identity one-hot (the first n entries).
    Everything else in the observation (positions, gate and event flags) can depend on other agents
    and is excluded; the agent's actions are fed separately. ``n_events`` is kept for the layout check."""
    return list(range(n_agents))


def _mlp(d_in: int, d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))


class CoalitionModel(nn.Module):
    def __init__(self, feat_dim: int, n_actions: int, n_agents: int, k: int, d: int):
        super().__init__()
        self.n, self.k, self.n_actions = n_agents, k, n_actions
        self.step = nn.Sequential(nn.Linear(feat_dim + n_actions, d), nn.ReLU())
        self.gru = nn.GRU(d, d, batch_first=True)
        self.heads = nn.ModuleList(_mlp(d if order == 1 else 2 * d, d) for order in range(1, k + 1))
        self.subsets = [list(itertools.combinations(range(n_agents), order)) for order in range(1, k + 1)]

    def encode(self, feats, actions, lengths):
        """feats (B, n, T, f), actions (B, n, T) -> h (B, n, d): last valid GRU state per agent."""
        B, n, T, _ = feats.shape
        x = th.cat([feats, F.one_hot(actions, self.n_actions).float()], dim=-1)
        out, _ = self.gru(self.step(x).reshape(B * n, T, -1))
        last = (lengths - 1).clamp(min=0).repeat_interleave(n)
        return out[th.arange(B * n), last].reshape(B, n, -1)

    def terms(self, h, present):
        """-> list over orders of (B, n_subsets) term values, zeroed where a member is absent."""
        out = []
        for order, (head, subsets) in enumerate(zip(self.heads, self.subsets), 1):
            idx = th.tensor(subsets, device=h.device)          # (S, order)
            hs = h[:, idx]                                     # (B, S, order, d)
            x = hs[..., 0, :] if order == 1 else th.cat([hs.sum(-2), hs.prod(-2)], dim=-1)
            out.append(head(x).squeeze(-1) * present[:, idx].prod(-1))
        return out


class CoalitionEstimator:
    """Episodic coalition-term return model trained on a fixed EpisodeBatch (Mode P)."""

    def __init__(self, n_agents: int, n_actions: int, obs_dim: int, n_events: int, k: int = 2,
                 seed: int = 0, **overrides):
        self.args = SimpleNamespace(**{**DEFAULTS, **overrides})
        self.n, self.k = n_agents, k
        self.feat_idx = own_feature_index(n_agents, n_events)
        if obs_dim != 4 * n_agents + 7 * n_events + 4:
            raise ValueError("observation size does not match the gridworld layout")
        th.manual_seed(seed)
        self.model = CoalitionModel(len(self.feat_idx), n_actions, n_agents, k, self.args.hidden_size)
        self.opt = th.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def _inputs(self, batch: EpisodeBatch):
        obs = batch["obs"][:, :-1].permute(0, 2, 1, 3)        # (B, n, T, o)
        feats = obs[..., self.feat_idx]
        actions = batch["actions"][:, :-1, :, 0].permute(0, 2, 1)
        mask = batch["filled"][:, :-1, 0].float()
        mask[:, 1:] = mask[:, 1:] * (1 - batch["terminated"][:, :-2, 0].float())
        present = obs[:, :, 0, self.n + 2]                      # own presence flag at t = 0
        return feats, actions, mask.sum(1).long(), mask, present

    def _forward(self, batch: EpisodeBatch):
        feats, actions, lengths, mask, present = self._inputs(batch)
        h = self.model.encode(feats, actions, lengths)
        return self.model.terms(h, present), mask

    def train(self, batch: EpisodeBatch) -> float:
        self.model.train()
        terms, mask = self._forward(batch)
        episode_return = (batch["reward"][:, :-1, 0] * mask).sum(1)
        pred = sum(t.sum(-1) for t in terms)
        a = self.args
        sparsity = sum(((t.abs() + a.sparsity_eps) ** a.sparsity_p - a.sparsity_eps ** a.sparsity_p).sum(-1)
                       for t in terms).mean()
        loss = F.mse_loss(pred, episode_return) + a.sparsity * sparsity
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(F.mse_loss(pred, episode_return).item())

    @th.no_grad()
    def dividends(self, batch: EpisodeBatch) -> np.ndarray:
        """(B, 2^n): the term values, placed at their coalition's mask; 0 above order k."""
        self.model.eval()
        terms, _ = self._forward(batch)
        out = np.zeros((batch.batch_size, 2 ** self.n))
        for values, subsets in zip(terms, self.model.subsets):
            for j, members in enumerate(subsets):
                out[:, mask_of(members)] = values[:, j].numpy()
        return out

    def credit(self, batch: EpisodeBatch, dividends: np.ndarray | None = None) -> np.ndarray:
        """Per-step credit (B, T, n): the episode's Shapley values, placed at the last step."""
        d = self.dividends(batch) if dividends is None else dividends
        phi = np.zeros((d.shape[0], self.n))
        for s in range(1, d.shape[1]):
            members = [i for i in range(self.n) if (s >> i) & 1]
            phi[:, members] += d[:, s:s + 1] / len(members)
        per_step = np.zeros((d.shape[0], batch.max_seq_length - 1, self.n))
        per_step[:, -1] = phi
        return per_step

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


__all__ = ["CoalitionEstimator", "CoalitionModel", "own_feature_index"]
