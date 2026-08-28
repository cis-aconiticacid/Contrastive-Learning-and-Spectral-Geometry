"""C13: PCA intrinsic-dim of Push-T trajectories.

Test whether uvlowr r=4 architectural rank matches data effective dim.

Measures:
- State (7-d) — PCA over frames
- State delta (7-d) — z_{t+1} - z_t within episodes (frameskip=5)
- Proprio (4-d) and Action (2-d)
- (Optional) Latent (192-d) from trained baseline encoder + latent deltas

Reports: effective_rank (exp of entropy of normalized eigenvalues^2),
participation_ratio (sum λ)² / sum λ², and cumulative explained variance.
"""
import h5py
import numpy as np
import json
from pathlib import Path

H5 = '/workspace/stablewm_home/pusht_expert_train.h5'
FRAMESKIP = 5
SAMPLE = 50_000  # frames to subsample for PCA — plenty


def effective_rank(eigvals):
    """exp(H(p)) where p = λ / Σλ. Same metric as our Jacobian effective_rank."""
    p = np.asarray(eigvals, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0: return 0.0
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(eigvals):
    """(Σλ)² / Σλ²  — another classic intrinsic-dim proxy."""
    e = np.asarray(eigvals, dtype=np.float64)
    e = e[e > 0]
    if e.size == 0: return 0.0
    return float((e.sum()**2) / (e**2).sum())


def pca_eig(X):
    """Centered PCA via covariance eigvals."""
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(C)
    return np.sort(eigvals)[::-1]


def report(name, X):
    ev = pca_eig(X)
    ev_n = ev / max(ev.sum(), 1e-12)
    cum = np.cumsum(ev_n)
    # dim needed for 90% / 95% / 99% variance
    d90 = int(np.searchsorted(cum, 0.90) + 1)
    d95 = int(np.searchsorted(cum, 0.95) + 1)
    d99 = int(np.searchsorted(cum, 0.99) + 1)
    er = effective_rank(ev)
    pr = participation_ratio(ev)
    print(f"\n=== {name}  (n={X.shape[0]}, d={X.shape[1]}) ===")
    print(f"  eigvals (top): {np.array2string(ev[:min(10, len(ev))], precision=4, suppress_small=True)}")
    print(f"  normalized:    {np.array2string(ev_n[:min(10, len(ev))], precision=3, suppress_small=True)}")
    print(f"  cum var:       {np.array2string(cum[:min(10, len(cum))], precision=3, suppress_small=True)}")
    print(f"  → effective_rank (entropy) = {er:.3f}")
    print(f"  → participation_ratio       = {pr:.3f}")
    print(f"  → dim @ 90%/95%/99% var     = {d90} / {d95} / {d99}")
    return {'name': name, 'n': int(X.shape[0]), 'd': int(X.shape[1]),
            'eigvals': ev.tolist(), 'eigvals_normalized': ev_n.tolist(),
            'cum_var': cum.tolist(),
            'effective_rank': er, 'participation_ratio': pr,
            'dim_at_90': d90, 'dim_at_95': d95, 'dim_at_99': d99}


def main():
    rng = np.random.default_rng(0)
    results = {}

    with h5py.File(H5, 'r') as f:
        ep_len = f['ep_len'][:]
        ep_off = f['ep_offset'][:]
        N_total = int(ep_len.sum())
        print(f"Push-T: {len(ep_len)} episodes, {N_total} total frames, "
              f"median ep_len={np.median(ep_len):.0f}, frameskip={FRAMESKIP}")
        print(f"After frameskip: median {np.median(ep_len)/FRAMESKIP:.1f} steps per ep")

        # ---- random subsample for whole-dataset PCA ----
        idx = rng.choice(N_total, size=min(SAMPLE, N_total), replace=False)
        idx = np.sort(idx)

        state = f['state'][:][idx]
        proprio = f['proprio'][:][idx]
        action = f['action'][:][idx]
        print(f"\nLoaded subsample: state {state.shape}, proprio {proprio.shape}, action {action.shape}")

        results['state'] = report("Push-T STATE (all frames, subsampled)", state)
        results['proprio'] = report("Push-T PROPRIO", proprio)
        results['action'] = report("Push-T ACTION", action)
        results['state_proprio_action'] = report(
            "Push-T STATE+PROPRIO+ACTION concatenated",
            np.concatenate([state, proprio, action], axis=1))

        # ---- per-episode state deltas (frameskip-aware) ----
        # iterate over a sample of episodes, build deltas z_{t+k} - z_t for k=frameskip
        ep_sample = rng.choice(len(ep_len), size=min(2000, len(ep_len)), replace=False)
        state_all = f['state']
        deltas = []
        for ep in ep_sample:
            o = int(ep_off[ep]); L = int(ep_len[ep])
            if L <= FRAMESKIP: continue
            s = state_all[o:o+L]
            ds = s[FRAMESKIP:] - s[:-FRAMESKIP]
            deltas.append(ds)
        deltas = np.concatenate(deltas, axis=0)
        print(f"\nState-delta sample: {deltas.shape}")
        results['state_delta'] = report(
            f"Push-T STATE DELTA (s_{{t+{FRAMESKIP}}} - s_t)", deltas)

    out = Path('/workspace/le-wm/refine-logs/c13_pusht_pca.json')
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
