"""State-value critics for persistent-ablation credit.

All critics regress the undiscounted return-to-go on the environment's global state (which carries
per-agent ``acting`` flags, so a stopped agent is just another state). Monte-Carlo targets, Adam.

* :func:`fit_mlp` - flat MLP (round-2 prototype: 2 x 128, 3,000 steps, batch 256, lr 1e-3).
* :func:`fit_ensemble` - bootstrap ensemble of MLPs; ``predict.std`` gives epistemic spread.
* :func:`fit_set_critic` - permutation-invariant critic (plan Section 7.2). The flat state is re-sliced
  into one token per agent (its position, presence, acting flag and its own lever flags) plus one
  shared context vector (event flags and time). A shared encoder, one self-attention layer and sum/mean
  pooling make W(pi s, pi mask) = W(s, mask) hold by construction. It sees exactly the information the
  MLP sees; only the symmetry differs.
"""

from __future__ import annotations

import numpy as np
import torch as th

SCALE = 10.0


def _train(net, X, y, seed, steps, batch, lr, weights=None):
    th.manual_seed(seed)
    opt = th.optim.Adam(net.parameters(), lr=lr)
    Xt, yt = th.tensor(X, dtype=th.float32), th.tensor(y / SCALE, dtype=th.float32)
    w = None if weights is None else th.tensor(weights, dtype=th.float32)
    g = th.Generator().manual_seed(seed)
    loss = th.tensor(0.0)
    for _ in range(steps):
        idx = th.randint(0, len(Xt), (batch,), generator=g)
        err = (net(Xt[idx]).squeeze(-1) - yt[idx]) ** 2
        loss = err.mean() if w is None else (err * w[idx]).sum() / w[idx].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())


def _predictor(net):
    def predict(S):
        with th.no_grad():
            return net(th.tensor(np.asarray(S), dtype=th.float32)).squeeze(-1).numpy().astype(np.float64) * SCALE
    return predict


def mlp(d_in: int, hidden: int = 128) -> th.nn.Module:
    return th.nn.Sequential(th.nn.Linear(d_in, hidden), th.nn.ReLU(), th.nn.Linear(hidden, hidden), th.nn.ReLU(),
                            th.nn.Linear(hidden, 1))


def fit_mlp(X, y, seed=0, hidden=128, steps=3000, batch=256, lr=1e-3, weights=None):
    th.set_num_threads(1)
    th.manual_seed(seed)
    net = mlp(X.shape[1], hidden)
    final = _train(net, X, y, seed, steps, batch, lr, weights)
    predict = _predictor(net)
    predict.final_loss = final
    predict.n_parameters = sum(p.numel() for p in net.parameters())
    return predict


def fit_ensemble(X, y, seed=0, members=5, **kw):
    """Bootstrap ensemble: each member sees a resample of the training *rows*."""
    rng = np.random.default_rng([seed, 31])
    preds = []
    for m in range(members):
        idx = rng.integers(0, len(X), len(X))
        preds.append(fit_mlp(X[idx], y[idx], seed=seed * 100 + m, **kw))

    def predict(S):
        return np.mean([p(S) for p in preds], axis=0)

    def std(S):
        return np.std([p(S) for p in preds], axis=0)
    predict.std = std
    predict.members = preds
    return predict


# --- permutation-invariant critic --------------------------------------------------------------------


class StateTokens:
    """Slices the flat global state (see PlantedCoalitionGridworld.get_state) into agent tokens + context.

    Agent token: position (2), present, acting, pulled flag per event (k), role flags per event (2k).
    Context: per event gate/done/pending/countdown/amount (5k), and t / T.
    """

    def __init__(self, n: int, k: int, roles: bool = True):
        self.n, self.k, self.roles = n, k, roles
        per_event = 5 + n
        roles = 4 * n + k * per_event                                    # start of the role-flag block
        self.d_state = roles + 2 * n * k + 1
        agent_idx = []
        for i in range(n):
            cols = [2 * i, 2 * i + 1, 2 * n + i, 3 * n + i]
            cols += [4 * n + e * per_event + 5 + i for e in range(k)]   # this agent's pulled flag per event
            if self.roles:
                cols += [roles + 2 * k * i + j for j in range(2 * k)]   # its role flags per event
            agent_idx.append(cols)
        self.agent_idx = th.tensor(agent_idx)                          # (n, 4 + 3k), or (n, 4 + k) without roles
        ctx = [4 * n + e * per_event + j for e in range(k) for j in range(5)] + [self.d_state - 1]
        self.ctx_idx = th.tensor(ctx)                                    # (5k + 1,)
        self.d_agent, self.d_ctx = 4 + (3 if self.roles else 1) * k, 5 * k + 1

    def __call__(self, S: th.Tensor):
        return S[:, self.agent_idx], S[:, self.ctx_idx]                  # (B, n, d_agent), (B, d_ctx)


class SetCritic(th.nn.Module):
    def __init__(self, tokens: StateTokens, hidden: int = 128, heads: int = 4):
        super().__init__()
        self.tokens = tokens
        h = hidden
        self.enc = th.nn.Sequential(th.nn.Linear(tokens.d_agent + tokens.d_ctx, h), th.nn.ReLU(), th.nn.Linear(h, h))
        self.attn = th.nn.MultiheadAttention(h, heads, batch_first=True)
        self.post = th.nn.Sequential(th.nn.ReLU(), th.nn.Linear(h, h), th.nn.ReLU())
        self.head = th.nn.Sequential(th.nn.Linear(2 * h + tokens.d_ctx, h), th.nn.ReLU(), th.nn.Linear(h, 1))

    def forward(self, S):
        a, c = self.tokens(S)
        n = a.shape[1]
        z = self.enc(th.cat([a, c[:, None, :].expand(-1, n, -1)], -1))      # (B, n, h)
        z = z + self.attn(z, z, z, need_weights=False)[0]
        z = self.post(z)
        pooled = th.cat([z.sum(1), z.mean(1), c], -1)
        return self.head(pooled)


def role_columns(n: int, k: int) -> np.ndarray:
    """Indices of the role-flag block of the global state."""
    start = 4 * n + k * (5 + n)
    return np.arange(start, start + 2 * n * k)


def drop_roles(X: np.ndarray, n: int, k: int) -> np.ndarray:
    """The state without its role flags (roles must then be inferred from positions, as in round 2)."""
    return np.delete(np.asarray(X), role_columns(n, k), axis=-1)


def fit_set_critic(X, y, n, k, seed=0, hidden=128, steps=3000, batch=256, lr=1e-3, weights=None, roles=True):
    th.set_num_threads(1)
    th.manual_seed(seed)
    net = SetCritic(StateTokens(n, k, roles=roles), hidden)
    final = _train(net, X, y, seed, steps, batch, lr, weights)
    predict = _predictor(net)
    predict.net = net
    predict.final_loss = final
    predict.n_parameters = sum(p.numel() for p in net.parameters())
    return predict


def transfer_set_critic(net: SetCritic, n: int, k: int):
    """The same weights read out for a team of a different size n (agent tokens are size-independent)."""
    import copy
    other = copy.deepcopy(net)
    other.tokens = StateTokens(n, k)
    pred = _predictor(other)
    pred.net = other
    return pred


def fit_set_critic_multi(datasets: dict, k: int, seed=0, hidden=128, steps=3000, batch=256, lr=1e-3, roles=True):
    """One set critic trained on several team sizes at once (v4 Part A).

    ``datasets`` maps n (number of agents) to (X, y) with X in that size's flat state layout. Each Adam step
    draws one size (probability proportional to its rows) and a batch from it, so the shared per-agent encoder
    and the pooled value head must fit every size. Returns ``predict_for(n)`` -> predictor for size n (any n,
    including sizes never trained on).
    """
    th.set_num_threads(1)
    th.manual_seed(seed)
    sizes = sorted(datasets)
    tokens = {n: StateTokens(n, k, roles=roles) for n in sizes}
    net = SetCritic(tokens[sizes[0]], hidden)
    opt = th.optim.Adam(net.parameters(), lr=lr)
    data = {n: (th.tensor(X, dtype=th.float32), th.tensor(y / SCALE, dtype=th.float32)) for n, (X, y) in datasets.items()}
    weights = np.array([len(datasets[n][1]) for n in sizes], dtype=float)
    weights /= weights.sum()
    g = th.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    loss = th.tensor(0.0)
    for _ in range(steps):
        n = sizes[rng.choice(len(sizes), p=weights)]
        Xt, yt = data[n]
        net.tokens = tokens[n]
        idx = th.randint(0, len(Xt), (batch,), generator=g)
        loss = ((net(Xt[idx]).squeeze(-1) - yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def predict_for(n):
        p = transfer_set_critic(net, n, k) if roles else _transfer_noroles(net, n, k)
        p.final_loss = float(loss.detach())
        return p
    predict_for.net = net
    return predict_for


def _transfer_noroles(net, n, k):
    import copy
    other = copy.deepcopy(net)
    other.tokens = StateTokens(n, k, roles=False)
    pred = _predictor(other)
    pred.net = other
    return pred


# --- two-set critic for Level-Based Foraging (v4 Part B) ---------------------------------------------------


class TwoSetCritic(th.nn.Module):
    """Permutation-invariant over agents AND over food items (LBF state from ``envs.lbf.LBFEnv.get_state``).

    Agent token: position (2), present, acting, level. Food token: position (2), level, reward share, present
    (``food_feats`` entries). Context: t / T.
    Type-specific encoders, one self-attention layer over all tokens (agents and food attend to each other), then
    sum and mean pooling per type and a value head.
    """

    def __init__(self, n: int, n_food: int, hidden: int = 128, heads: int = 4, food_feats: int = 5):
        super().__init__()
        self.n, self.F = n, n_food
        self.agent_idx = th.tensor([[2 * i, 2 * i + 1, 2 * n + i, 3 * n + i, 4 * n + i] for i in range(n)])
        self.food_idx = th.tensor([[5 * n + food_feats * k + j for j in range(food_feats)] for k in range(n_food)])
        h = hidden
        self.enc_a = th.nn.Sequential(th.nn.Linear(5 + 1, h), th.nn.ReLU(), th.nn.Linear(h, h))
        self.enc_f = th.nn.Sequential(th.nn.Linear(food_feats + 1, h), th.nn.ReLU(), th.nn.Linear(h, h))
        self.attn = th.nn.MultiheadAttention(h, heads, batch_first=True)
        self.post = th.nn.Sequential(th.nn.ReLU(), th.nn.Linear(h, h), th.nn.ReLU())
        self.head = th.nn.Sequential(th.nn.Linear(4 * h + 1, h), th.nn.ReLU(), th.nn.Linear(h, 1))

    def forward(self, S):
        t = S[:, -1:]
        a, f = S[:, self.agent_idx], S[:, self.food_idx]
        za = self.enc_a(th.cat([a, t[:, None, :].expand(-1, self.n, -1)], -1))
        zf = self.enc_f(th.cat([f, t[:, None, :].expand(-1, self.F, -1)], -1))
        z = th.cat([za, zf], 1)
        z = self.post(z + self.attn(z, z, z, need_weights=False)[0])
        za, zf = z[:, :self.n], z[:, self.n:]
        return self.head(th.cat([za.sum(1), za.mean(1), zf.sum(1), zf.mean(1), t], -1))


def fit_two_set_critic(X, y, n, n_food, seed=0, hidden=128, steps=3000, batch=256, lr=1e-3):
    th.set_num_threads(1)
    th.manual_seed(seed)
    net = TwoSetCritic(n, n_food, hidden)
    final = _train(net, X, y, seed, steps, batch, lr)
    predict = _predictor(net)
    predict.final_loss = final
    predict.n_parameters = sum(p.numel() for p in net.parameters())
    predict.net = net
    return predict


class FoodHeadCritic(TwoSetCritic):
    """Reward-decomposed LBF critic (v4 Part B; Hybrid Reward Architecture, van Seijen et al. 2017).

    LBF's return is a sum over food items, so V(s) = sum_k value_k(s) * P(food k is collected | s) exactly, with
    value_k = 10 x reward share x present read from the state. The two-set encoder is shared; each food token
    (after attending to the agents) predicts its own collection probability. Training is per food, on whether the
    food, present at this step, was gone by the end of the trajectory: F binary targets per row instead of one sum,
    and a food that is gone contributes exactly zero.
    """

    def __init__(self, n: int, n_food: int, hidden: int = 128, heads: int = 4, food_feats: int = 5):
        super().__init__(n, n_food, hidden, heads, food_feats)
        h = hidden
        self.food_head = th.nn.Sequential(th.nn.Linear(2 * h + 1, h), th.nn.ReLU(), th.nn.Linear(h, 1))

    def logits(self, S):
        t = S[:, -1:]
        a, f = S[:, self.agent_idx], S[:, self.food_idx]
        za = self.enc_a(th.cat([a, t[:, None, :].expand(-1, self.n, -1)], -1))
        zf = self.enc_f(th.cat([f, t[:, None, :].expand(-1, self.F, -1)], -1))
        z = th.cat([za, zf], 1)
        z = self.post(z + self.attn(z, z, z, need_weights=False)[0])
        za, zf = z[:, :self.n], z[:, self.n:]
        ctx = za.sum(1, keepdim=True).expand(-1, self.F, -1)          # the whole team, seen by every food
        return self.food_head(th.cat([zf, ctx, t[:, None, :].expand(-1, self.F, -1)], -1)).squeeze(-1)

    def food_values(self, S):
        f = S[:, self.food_idx]
        return 10.0 * f[..., 3] * f[..., 4] / SCALE                     # share x present, in critic units

    def forward(self, S):
        return (self.food_values(S) * th.sigmoid(self.logits(S))).sum(-1, keepdim=True)


def food_targets(X, final, n: int, n_food: int, food_feats: int = 5) -> np.ndarray:
    """(M, F) 1 if food k is present in the row's state and gone at the end of its trajectory, else 0."""
    pres = lambda S: np.stack([S[:, 5 * n + food_feats * k + food_feats - 1] for k in range(n_food)], 1)
    return ((pres(X) > 0.5) & (pres(final) < 0.5)).astype(np.float32)


def fit_food_critic(X, final, n, n_food, seed=0, hidden=128, steps=3000, batch=256, lr=1e-3):
    """Fit :class:`FoodHeadCritic` with a per-food binary cross-entropy, masked to foods present in the state."""
    th.set_num_threads(1)
    th.manual_seed(seed)
    net = FoodHeadCritic(n, n_food, hidden)
    Y = th.tensor(food_targets(X, final, n, n_food))
    Xt = th.tensor(X, dtype=th.float32)
    mask = (Xt[:, net.food_idx][..., 4] > 0.5).float()
    opt = th.optim.Adam(net.parameters(), lr=lr)
    g = th.Generator().manual_seed(seed)
    loss = th.tensor(0.0)
    for _ in range(steps):
        idx = th.randint(0, len(Xt), (batch,), generator=g)
        bce = th.nn.functional.binary_cross_entropy_with_logits(net.logits(Xt[idx]), Y[idx], reduction="none")
        loss = (bce * mask[idx]).sum() / mask[idx].sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    predict = _predictor(net)
    predict.final_loss = float(loss.detach())
    predict.n_parameters = sum(p.numel() for p in net.parameters())
    predict.net = net
    return predict
