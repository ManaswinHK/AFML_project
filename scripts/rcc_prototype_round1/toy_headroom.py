"""Toy headroom test: does exact credit speed up learning, and when?

Planted coalition team game, one shot, repeated. n = c*K agents; event k needs its c planted members to all
choose action k in the same round and then pays 10. Every agent chooses among K+1 actions (idle or any event).
Independent softmax REINFORCE learners with a per-agent running-mean baseline; the only thing varied is the
learning signal:
  team    : G(a)
  loo     : G(a) - G(a with a_i := idle)        (Wolpert-Tumer difference reward with idle clamping)
  shapley : Shapley value of v(S) = G(a_S, idle elsewhere)   (here = 10/c if i's own event fired, else 0)
Optimal return = 10*K. Reported: mean return over the last 10% of rounds, and rounds until the running mean
return first exceeds 90% of optimal (None if never).
"""
import numpy as np, sys, json

def simulate(K, c, signal, rounds, lr, seed, extra_bystanders=0):
    rng = np.random.default_rng(seed)
    n = c * K + extra_bystanders
    member_of = np.array([i // c for i in range(c * K)] + [-1] * extra_bystanders)   # -1: bystander
    logits = np.zeros((n, K + 1))
    base = np.zeros(n)
    hist = np.zeros(rounds)
    for t in range(rounds):
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        a = (p.cumsum(1) > rng.random((n, 1))).argmax(1)            # action 0 = idle, k+1 = event k
        on_own = np.zeros(n, bool)
        on_own[:c * K] = a[:c * K] == member_of[:c * K] + 1
        fired = on_own[:c * K].reshape(K, c).all(1)                    # event k fires iff all its members chose it
        G = 10.0 * fired.sum()
        hist[t] = G
        if signal == "team":
            r = np.full(n, G)
        else:
            own = np.zeros(n)
            own[:c * K] = np.repeat(fired, c) * 10.0                   # removing i kills exactly its own fired event
            r = own if signal == "loo" else own / c
        adv = r - base
        base += 0.05 * (r - base)
        grad = -p
        grad[np.arange(n), a] += 1.0
        logits += lr * adv[:, None] * grad
    return hist

def summarise(hist, K):
    opt = 10.0 * K
    final = hist[-len(hist) // 10:].mean() / opt
    run = np.convolve(hist, np.ones(200) / 200, mode="valid") / opt
    hit = np.flatnonzero(run >= 0.9)
    return final, (int(hit[0]) + 200 if len(hit) else None)

if __name__ == "__main__":
    rounds, seeds, lr = 30000, 20, 0.02
    out = {}
    print(f"rounds={rounds} seeds={seeds} lr={lr}; coalition size c=2")
    print("K  bystanders  signal    final/optimal (mean, sd)   solved seeds   median rounds to 90%")
    for K, b in [(1, 0), (1, 4), (2, 0), (4, 0), (8, 0), (4, 4)]:
        for sig in ("team", "loo", "shapley"):
            res = [summarise(simulate(K, 2, sig, rounds, lr, s, b), K) for s in range(seeds)]
            fin = np.array([r[0] for r in res]); hits = [r[1] for r in res if r[1] is not None]
            med = int(np.median(hits)) if len(hits) > seeds // 2 else None
            out[f"K{K}_b{b}_{sig}"] = dict(final_mean=float(fin.mean()), final_sd=float(fin.std()), solved=len(hits), median_rounds=med)
            print(f"{K}  {b}           {sig:8s}  {fin.mean():.3f} +- {fin.std():.3f}            {len(hits):2d}/{seeds}         {med}")
    json.dump(out, open("toy_headroom.json", "w"), indent=1)
