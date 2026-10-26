"""STAS (Chen, Zhang, Yang & Du, AAAI 2024): spatial-temporal return decomposition, for Mode P.

A clean reimplementation that follows the official code, github.com/zowiezhang/STAS at
commit 6312725 (models/reward_model/mard/{mard.py, modules.py}, utils/util.py::make_train_step,
and configs/STAS_AandB.py). The repository has no licence file, so no code is copied.

It uses the default model, STAS_ML (reward_model_version "v2"). Each layer is:
1. temporal self-attention per agent (causal);
2. a "Shapley attention" over agents. Each agent attends to a random coalition (Bernoulli(0.5)
   mask, always including itself), and the output is averaged over ``sample_num`` sampled masks:
   a Monte Carlo estimate of its marginal contribution.

A linear head on the concatenated layer outputs gives per-agent, per-step rewards r_{i,t}. It is
trained so that sum_{i,t} r_{i,t} matches the episode return (MSE).

Kept as in the original:
- the odd residual wiring in ``EncoderLayer``;
- agent identity embeddings that are randomly permuted on every forward;
- dropout = emb_dropout = 0.5 (the runner passes ``dropout=0.3``, but STAS_ML never uses it).

Deviations:
- The time mask is repeated per agent with ``repeat_interleave``. The original's ``repeat`` pairs
  masks with the wrong episodes when lengths differ; our episodes all have the same length, so
  the result is identical.
- The read-out averages ``readout_passes`` stochastic forwards in eval mode. The original uses a
  single forward, which is a noisier estimate of the same expectation.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from whoearned.epymarl_bridge import EpisodeBatch

DEFAULTS = dict(hidden_size=128, n_heads=4, n_layers=3, sample_num=3, emb_dropout=0.5,
                lr=1e-4, weight_decay=1e-5, readout_passes=16)


def _attend(q, k, v, mask, dropout):
    scores = th.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    prob = dropout(F.softmax(scores.masked_fill(mask == 0, -1e9), dim=-1))
    return th.matmul(prob, v)


class _Attention(nn.Module):
    """Multi-head attention. With ``n_query`` set, each position (agent) has its own query projection."""

    def __init__(self, d: int, heads: int, dropout: float, n_query: int | None = None):
        super().__init__()
        self.heads, self.size = heads, d // heads
        self.W_Q = nn.ModuleList(nn.Linear(d, d) for _ in range(n_query)) if n_query else nn.Linear(d, d)
        self.W_K, self.W_V, self.out = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        return x.view(*x.shape[:-1], self.heads, self.size).transpose(1, 2)

    def forward(self, x, mask):  # x (N, L, d), mask (N, L, L)
        if isinstance(self.W_Q, nn.ModuleList):
            q = th.stack([w(x[:, i]) for i, w in enumerate(self.W_Q)], dim=1)
        else:
            q = self.W_Q(x)
        ctx = _attend(self._split(q), self._split(self.W_K(x)), self._split(self.W_V(x)),
                      mask.unsqueeze(1), self.dropout)
        return self.dropout(self.out(ctx.transpose(1, 2).reshape(*x.shape)))


class _EncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.attn = _Attention(d, heads, dropout)
        self.ln1, self.ln2 = nn.LayerNorm(d, eps=1e-12), nn.LayerNorm(d, eps=1e-12)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d), nn.Dropout(dropout))

    def forward(self, x, mask):
        att = self.attn(x, mask)
        out = self.ff(self.ln1(x + att)) + att
        return self.ln2(out + att)


class _ShapleyAttention(nn.Module):
    def __init__(self, d: int, heads: int, n_agents: int, sample_num: int, dropout: float):
        super().__init__()
        self.n, self.sample_num = n_agents, sample_num
        self.phi = _Attention(d, heads, dropout, n_query=n_agents)
        self.agent_embedding = nn.Embedding(n_agents, d)

    def forward(self, x):  # x (B, n, T, d)
        B, n, T, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * T, n, d)
        x = x + self.agent_embedding(th.randperm(n, device=x.device))[None]
        out = 0
        for _ in range(self.sample_num):
            mask = th.bernoulli(th.full((n, n), 0.5, device=x.device))
            mask = (mask - th.diag(th.diag(mask)) + th.eye(n, device=x.device)).expand(B * T, n, n)
            out = out + self.phi(x, mask)
        return (out / self.sample_num).reshape(B, T, n, d).permute(0, 2, 1, 3)


class STASModel(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, n_agents: int, seq_length: int, a: SimpleNamespace):
        super().__init__()
        d = a.hidden_size
        self.seq_length = seq_length
        self.state_emb = nn.Linear(obs_dim, d)
        self.action_emb = nn.Embedding(n_actions + 1, d)
        self.pos_emb = nn.Embedding(seq_length, d)
        self.layers = nn.ModuleList(
            nn.ModuleList([_EncoderLayer(d, a.n_heads, a.emb_dropout),
                           _ShapleyAttention(d, a.n_heads, n_agents, a.sample_num, a.emb_dropout)])
            for _ in range(a.n_layers)
        )
        self.head = nn.Linear(d * a.n_layers, 1)

    def forward(self, obs, actions, lengths):
        """obs (B, n, T, o), actions (B, n, T) long, lengths (B,) -> rewards (B, n, T)."""
        B, n, T, _ = obs.shape
        x = self.state_emb(obs) + self.action_emb(actions) + self.pos_emb(th.arange(T, device=obs.device))
        valid = (th.arange(T, device=obs.device)[None] < lengths[:, None]).float()
        time_mask = th.tril(valid.unsqueeze(-1) * valid.unsqueeze(1)).repeat_interleave(n, dim=0)
        x = x.reshape(B * n, T, -1)
        outs = []
        for temporal, spatial in self.layers:
            x = temporal(x, time_mask)
            x = spatial(x.reshape(B, n, T, -1)).reshape(B * n, T, -1)
            outs.append(x)
        return self.head(th.cat(outs, dim=-1)).reshape(B, n, T)


class STAS:
    """Return-decomposition model trained on a fixed EpisodeBatch (Mode P)."""

    def __init__(self, n_agents: int, n_actions: int, obs_dim: int, horizon: int, seed: int = 0,
                 device: str | None = None, **overrides):
        self.args = SimpleNamespace(**{**DEFAULTS, **overrides})
        # The transformer is the one network here big enough to gain from a GPU; others stay on CPU.
        self.device = device or ("cuda" if th.cuda.is_available() else "cpu")
        th.manual_seed(seed)
        self.model = STASModel(obs_dim, n_actions, n_agents, horizon, self.args).to(self.device)
        self.opt = th.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def _inputs(self, batch: EpisodeBatch):
        obs = batch["obs"][:, :-1].permute(0, 2, 1, 3)
        actions = batch["actions"][:, :-1, :, 0].permute(0, 2, 1)
        mask = batch["filled"][:, :-1, 0].float()
        mask[:, 1:] = mask[:, 1:] * (1 - batch["terminated"][:, :-2, 0].float())
        return tuple(x.to(self.device) for x in (obs, actions, mask.sum(1).long(), mask))

    def train(self, batch: EpisodeBatch) -> float:
        obs, actions, lengths, mask = self._inputs(batch)
        episode_return = (batch["reward"][:, :-1, 0].to(self.device) * mask).sum(1)
        self.model.train()
        pred = self.model(obs, actions, lengths)
        loss = F.mse_loss((pred * mask[:, None]).sum(dim=(1, 2)), episode_return)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    @th.no_grad()
    def credit(self, batch: EpisodeBatch) -> np.ndarray:
        """Per-step decomposed rewards (B, T, n), averaged over ``readout_passes`` forwards in eval mode."""
        obs, actions, lengths, mask = self._inputs(batch)
        self.model.eval()
        pred = sum(self.model(obs, actions, lengths) for _ in range(self.args.readout_passes)) / self.args.readout_passes
        return (pred * mask[:, None]).permute(0, 2, 1).cpu().numpy()


__all__ = ["STAS", "STASModel"]
