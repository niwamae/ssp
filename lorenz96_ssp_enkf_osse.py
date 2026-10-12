# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///
"""
Reproduction code for:
    Saito, T. & Kotsuki, S. (2026)
    "Sparse Sensor Placement for Reducing Forecast Errors in Ensemble Kalman
    Filtering"

Run with uv (recommended):
    uv run lorenz96_ssp_enkf_osse.py [options]

`uv run` reads the inline PEP 723 metadata above and automatically creates
an ephemeral environment with the required dependencies (numpy, matplotlib)
the first time it is run -- no manual venv/pip setup needed.

This script reproduces the Lorenz-96 OSSE (Observing System Simulation
Experiment) described in Section 5 of the paper: adaptive observation
network design via A-/D-/E-optimality greedy algorithms (AG-M, DG-E, EG-M)
built from Fisher information matrices (FIM) in the model space / ensemble
space of the ETKF.

Experiment design (Section 5.1 of the paper)
----------------------------------------------
- Lorenz-96 model, Nx = 40 grid points, F = 8.0, dt = 0.05 (4 steps = 1 day).
- 20 grid points = "land" (always observed, fixed network).
  20 grid points = "ocean" (candidate sites for adaptive observations).
- ETKF, ensemble size Ne = 32, R = I (unit variance), multiplicative
  inflation alpha = 1.01, no localization.
- Assimilation interval  ta = 6h  = 1 model step.
  Forecast lead time      tf = 2 days = 8 model steps.
- At every cycle:
    1. A "control" cycle (20 land obs + 5 random ocean obs) keeps a
       background ensemble climatology running (not affected by the
       adaptive-selection experiment; results are *not* fed back).
    2. Starting from that cycle's background ensemble Xb, the background
       perturbations dXb are propagated tf steps ahead (using the
       background trajectory itself as a surrogate TLM, M_b) to obtain the
       surrogate forecast perturbations dXf used to build the FIM.
    3. AG-M / DG-E / EG-M / homogeneous select Nadapt ocean sites from
       (dXb, dXf).
    4. Observations are generated from the nature run at the fixed + chosen
       adaptive sites, assimilated into Xb -> Xa (branch, evaluation only),
       and Xa is propagated tf steps ahead to evaluate forecast spread and
       RMSE against the nature run.

Known simplifications relative to the paper
---------------------------------------------
- Only single-grid-point scalar observations are used (same as the paper).
- No localization (same as the paper).
- EFSO-based a-posteriori observation-impact diagnostic (Fig. 10 of the
  paper) is not reproduced here; everything else (Figs. 5-9, 11) is.
- Default `--n-cycles` is small (a few hundred) for a quick demo; pass
  `--n-cycles 14600` to reproduce the paper's full 10-year experiment
  (6-hourly cycles), which will take considerably longer to run.
"""

import argparse
import time

import numpy as np

# ----------------------------------------------------------------------------
# 1. Lorenz-96 model
# ----------------------------------------------------------------------------


def l96_tendency(x, F):
    """Lorenz-96 tendency. x has shape (Nx,) or (Nx, Ne); the grid axis is
    always axis 0 (NOT the last axis -- for an ensemble array (Nx, Ne), the
    last axis is the ensemble-member axis, and must not be rolled)."""
    return (np.roll(x, -1, axis=0) - np.roll(x, 2, axis=0)) * np.roll(x, 1, axis=0) - x + F


def rk4_step(x, dt, F):
    k1 = l96_tendency(x, F)
    k2 = l96_tendency(x + 0.5 * dt * k1, F)
    k3 = l96_tendency(x + 0.5 * dt * k2, F)
    k4 = l96_tendency(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(x, nsteps, dt, F):
    """Integrate state/ensemble x (..., Nx) forward `nsteps` model steps."""
    for _ in range(nsteps):
        x = rk4_step(x, dt, F)
    return x


# ----------------------------------------------------------------------------
# 2. ETKF analysis update  (Bishop et al. 2001 / Hunt et al. 2007 form)
# ----------------------------------------------------------------------------


def etkf_analysis(Xb, obs_idx, y, r_var, inflation=1.0):
    """
    Xb       : (Nx, Ne) background ensemble
    obs_idx  : (Nobs,) indices of observed grid points
    y        : (Nobs,) observation values
    r_var    : (Nobs,) observation error variances (diagonal R)
    inflation: multiplicative inflation factor on background perturbations

    Returns Xa (Nx, Ne) analysis ensemble.
    """
    Ne = Xb.shape[1]
    xb_bar = Xb.mean(axis=1)
    dXb = (Xb - xb_bar[:, None]) * np.sqrt(inflation)

    dYb = dXb[obs_idx, :]                       # (Nobs, Ne)
    ybar = xb_bar[obs_idx]
    inv_sig = 1.0 / np.sqrt(r_var)
    d = (y - ybar) * inv_sig
    dYb_n = dYb * inv_sig[:, None]

    Ne_m1 = Ne - 1
    Pa_inv = Ne_m1 * np.eye(Ne) + dYb_n.T @ dYb_n
    Pa = np.linalg.inv(Pa_inv)

    wbar = Pa @ (dYb_n.T @ d)
    evals, evecs = np.linalg.eigh(Ne_m1 * Pa)
    evals = np.clip(evals, 0.0, None)
    Wa = evecs @ (np.sqrt(evals)[:, None] * evecs.T)

    Xa = xb_bar[:, None] + dXb @ (wbar[:, None] + Wa)
    return Xa


# ----------------------------------------------------------------------------
# 3. FIM-based greedy observation selection (Section 3.5 / 4 of the paper)
# ----------------------------------------------------------------------------


class EnsembleFIM:
    """Maintains the ensemble-space analysis precision matrix

        Pa_inv^e = (Ne-1) I + sum_j w_j w_j^T / r_j

    while observations are added incrementally (shared by all greedy
    algorithms below, cf. Eq. 15 / Algorithm 1-2 of the paper).
    """

    def __init__(self, Ne):
        self.Ne = Ne
        self.Pa_inv = (Ne - 1) * np.eye(Ne)

    def add(self, w, r_var):
        self.Pa_inv = self.Pa_inv + np.outer(w, w) / r_var

    @property
    def Pa(self):
        return np.linalg.inv(self.Pa_inv)


def greedy_select(method, dXb, dXf, fixed_idx, candidate_idx, r_var_fixed, r_var_cand, n_select):
    """
    Fast greedy observation-site selection, vectorized over candidates.

    method : 'AG-M' (Eq. 29/30, fast), 'DG-E' (Eq. 31/32, fast), 'EG-M' (Eq. 23)
    dXb    : (Nx, Ne) background ensemble perturbations
    dXf    : (Nx, Ne) surrogate forecast ensemble perturbations (= M_b dXb)

    Returns the list of selected grid-point indices, in selection order.
    """
    r_var_fixed = np.broadcast_to(r_var_fixed, (len(fixed_idx),)).astype(float)
    r_var_cand = np.broadcast_to(r_var_cand, (len(candidate_idx),)).astype(float)
    rvar_of = dict(zip(candidate_idx, r_var_cand))

    fim = EnsembleFIM(dXb.shape[1])
    for idx, rv in zip(fixed_idx, r_var_fixed):
        fim.add(dXb[idx, :], rv)

    remaining = list(candidate_idx)
    selected = []

    for _ in range(n_select):
        Pa = fim.Pa
        W = dXb[remaining, :].T                 # (Ne, Ncand)
        rv = np.array([rvar_of[i] for i in remaining])
        PaW = Pa @ W                             # (Ne, Ncand)
        den = rv + np.einsum("ij,ij->j", W, PaW)  # w^T Pa w + r, per candidate

        if method == "AG-M":
            # maximize || dXf Pa w ||^2 / (r + w^T Pa w)  <=> minimize tr(Pf)
            num = np.sum((dXf @ PaW) ** 2, axis=0)
            score = num / den
            best_pos = int(np.argmax(score))

        elif method == "DG-E":
            # maximize w^T Pa w / r  <=> maximize det(Pa_inv_new) (Shannon info)
            num = np.einsum("ij,ij->j", W, PaW)
            score = num / rv
            best_pos = int(np.argmax(score))

        elif method == "EG-M":
            # minimize max eigenvalue of the trial forecast error covariance
            # (Sherman-Morrison update of Pf, avoids a fresh matrix inverse)
            Pf_base = dXf @ Pa @ dXf.T
            V = dXf @ PaW                         # (Nx, Ncand)
            best_pos, best_val = None, None
            for j in range(len(remaining)):
                Pf_trial = Pf_base - np.outer(V[:, j], V[:, j]) / den[j]
                val = np.linalg.eigvalsh(Pf_trial)[-1]
                if best_val is None or val < best_val:
                    best_val, best_pos = val, j

        else:
            raise ValueError(f"unknown method: {method}")

        best_i = remaining[best_pos]
        fim.add(dXb[best_i, :], rvar_of[best_i])
        selected.append(best_i)
        remaining.pop(best_pos)

    return selected


def greedy_select_AGM_baseline(dXb, dXf, fixed_idx, candidate_idx, r_var_fixed, r_var_cand, n_select):
    """
    Baseline (non-fast) AG-M: rebuilds Pa^e from scratch and inverts it for
    every candidate at every step (Algorithm 1 of the paper). Mathematically
    identical to greedy_select('AG-M', ...) but much slower; kept only for
    the computational-cost comparison (cf. Fig. 11 of the paper).
    """
    Ne = dXb.shape[1]
    r_var_fixed = np.broadcast_to(r_var_fixed, (len(fixed_idx),)).astype(float)
    r_var_cand = np.broadcast_to(r_var_cand, (len(candidate_idx),)).astype(float)
    rvar_of = dict(zip(candidate_idx, r_var_cand))

    sel_idx = list(fixed_idx)
    sel_rvar = list(r_var_fixed)
    remaining = list(candidate_idx)
    chosen = []

    for _ in range(n_select):
        best_i, best_tr = None, None
        for i in remaining:
            idx_trial = sel_idx + [i]
            rv_trial = np.array(sel_rvar + [rvar_of[i]])
            W = dXb[idx_trial, :]
            Pa_inv = (Ne - 1) * np.eye(Ne) + (W / rv_trial[:, None]).T @ W
            Pa = np.linalg.inv(Pa_inv)
            tr_pf = np.trace(dXf @ Pa @ dXf.T)
            if best_tr is None or tr_pf < best_tr:
                best_i, best_tr = i, tr_pf
        sel_idx.append(best_i)
        sel_rvar.append(rvar_of[best_i])
        chosen.append(best_i)
        remaining.remove(best_i)

    return chosen


def select_homogeneous(candidate_idx, n_select):
    """Equally spaced candidate sites (the paper's non-optimized baseline)."""
    candidate_idx = sorted(candidate_idx)
    n = len(candidate_idx)
    positions = [int(round(k * n / n_select)) % n for k in range(n_select)]
    return [candidate_idx[p] for p in positions]


# ----------------------------------------------------------------------------
# 4. OSSE driver (Section 5 of the paper)
# ----------------------------------------------------------------------------


class L96Config:
    Nx = 40
    F = 8.0
    dt = 0.05
    steps_per_day = 4           # Lorenz & Emanuel (1998): 4 steps = 1 model day
    ta_steps = 1                # 6h assimilation interval  = 1 step
    tf_steps = 8                # 2-day forecast lead time = 8 steps
    Ne = 32
    inflation = 1.01
    r_var = 1.0                 # R = I
    ocean_idx = list(range(0, 20))
    land_idx = list(range(20, 40))
    n_control_ocean_obs = 5     # "control" cycle: 5 random ocean obs


METHODS = ["AG-M", "DG-E", "EG-M", "homogeneous"]


def make_nature_run(cfg, n_steps, warmup=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = cfg.F * np.ones(cfg.Nx) + rng.normal(0, 0.1, cfg.Nx)
    x = integrate(x, warmup, cfg.dt, cfg.F)

    traj = np.empty((n_steps + 1, cfg.Nx))
    traj[0] = x
    for k in range(n_steps):
        x = rk4_step(x, cfg.dt, cfg.F)
        traj[k + 1] = x
    return traj


def make_initial_ensemble(cfg, x0, rng):
    return x0[:, None] + rng.normal(0, 1.0, size=(cfg.Nx, cfg.Ne))


def run_osse(cfg, n_cycles, nadapt_list, burnin=200, seed=0):
    """
    Runs the OSSE described above and returns a dict of per-(method, Nadapt)
    statistics: mean forecast spread, mean forecast RMSE, first-site
    selection-frequency histogram, and mean consecutive-selection distance.
    """
    rng = np.random.default_rng(seed)

    total_steps = burnin + n_cycles * cfg.ta_steps + cfg.tf_steps + 1
    nature = make_nature_run(cfg, total_steps, seed=seed)

    Xa_main = make_initial_ensemble(cfg, nature[0], rng)

    # --- burn-in: let the control cycle equilibrate -------------------------
    t = 0
    for _ in range(burnin):
        Xb = integrate(Xa_main, cfg.ta_steps, cfg.dt, cfg.F)
        t += cfg.ta_steps
        ocean_ctrl = rng.choice(cfg.ocean_idx, size=cfg.n_control_ocean_obs, replace=False)
        obs_idx = cfg.land_idx + list(ocean_ctrl)
        y = nature[t, obs_idx] + rng.normal(0, np.sqrt(cfg.r_var), len(obs_idx))
        Xa_main = etkf_analysis(Xb, obs_idx, y, np.full(len(obs_idx), cfg.r_var), cfg.inflation)

    # --- statistics containers ----------------------------------------------
    stats = {
        (m, n): {"spread": [], "rmse": [], "first_site_count": np.zeros(cfg.Nx),
                  "mean_consec_dist": []}
        for m in METHODS for n in nadapt_list
    }

    for _ in range(n_cycles):
        Xb = integrate(Xa_main, cfg.ta_steps, cfg.dt, cfg.F)
        t += cfg.ta_steps
        t_fcst = t + cfg.tf_steps

        xb_bar = Xb.mean(axis=1)
        dXb = Xb - xb_bar[:, None]
        Xf_surrogate = integrate(Xb, cfg.tf_steps, cfg.dt, cfg.F)
        xf_bar = Xf_surrogate.mean(axis=1)
        dXf = Xf_surrogate - xf_bar[:, None]

        for n_adapt in nadapt_list:
            for method in METHODS:
                if method == "homogeneous":
                    chosen = select_homogeneous(cfg.ocean_idx, n_adapt)
                else:
                    chosen = greedy_select(
                        method, dXb, dXf, cfg.land_idx, cfg.ocean_idx,
                        cfg.r_var, cfg.r_var, n_adapt,
                    )

                obs_idx = cfg.land_idx + list(chosen)
                r_arr = np.full(len(obs_idx), cfg.r_var)
                y = nature[t, obs_idx] + rng.normal(0, np.sqrt(cfg.r_var), len(obs_idx))

                Xa_method = etkf_analysis(Xb, obs_idx, y, r_arr, cfg.inflation)
                Xf_eval = integrate(Xa_method, cfg.tf_steps, cfg.dt, cfg.F)

                spread = Xf_eval.std(axis=1, ddof=1).mean()
                rmse = np.sqrt(np.mean((Xf_eval.mean(axis=1) - nature[t_fcst]) ** 2))

                s = stats[(method, n_adapt)]
                s["spread"].append(spread)
                s["rmse"].append(rmse)
                s["first_site_count"][chosen[0]] += 1
                if len(chosen) > 1:
                    s["mean_consec_dist"].append(
                        np.mean(np.abs(np.diff(sorted(chosen))))
                    )

        # --- advance the control cycle (independent of the branches above) --
        ocean_ctrl = rng.choice(cfg.ocean_idx, size=cfg.n_control_ocean_obs, replace=False)
        obs_idx = cfg.land_idx + list(ocean_ctrl)
        y = nature[t, obs_idx] + rng.normal(0, np.sqrt(cfg.r_var), len(obs_idx))
        Xa_main = etkf_analysis(Xb, obs_idx, y, np.full(len(obs_idx), cfg.r_var), cfg.inflation)

    summary = {}
    for key, s in stats.items():
        summary[key] = {
            "spread": float(np.mean(s["spread"])),
            "rmse": float(np.mean(s["rmse"])),
            "first_site_freq": s["first_site_count"] / n_cycles,
            "mean_consec_dist": float(np.mean(s["mean_consec_dist"])) if s["mean_consec_dist"] else np.nan,
        }
    return summary


# ----------------------------------------------------------------------------
# 5. Baseline vs. fast AG-M: correctness + speed check (Fig. 11 of the paper)
# ----------------------------------------------------------------------------


def check_fast_agm(cfg, n_trials=5, n_adapt=5, seed=1):
    rng = np.random.default_rng(seed)
    nature = make_nature_run(cfg, 50, warmup=1000, seed=seed)
    Xa = make_initial_ensemble(cfg, nature[0], rng)

    t_base, t_fast = 0.0, 0.0
    all_match = True
    for _ in range(n_trials):
        Xb = integrate(Xa, cfg.ta_steps, cfg.dt, cfg.F)
        xb_bar = Xb.mean(axis=1)
        dXb = Xb - xb_bar[:, None]
        Xf = integrate(Xb, cfg.tf_steps, cfg.dt, cfg.F)
        dXf = Xf - Xf.mean(axis=1)[:, None]

        t0 = time.perf_counter()
        sel_base = greedy_select_AGM_baseline(
            dXb, dXf, cfg.land_idx, cfg.ocean_idx, cfg.r_var, cfg.r_var, n_adapt)
        t_base += time.perf_counter() - t0

        t0 = time.perf_counter()
        sel_fast = greedy_select(
            "AG-M", dXb, dXf, cfg.land_idx, cfg.ocean_idx, cfg.r_var, cfg.r_var, n_adapt)
        t_fast += time.perf_counter() - t0

        all_match &= (sel_base == sel_fast)
        Xa = Xb  # just advance state for the next trial

    print(f"[check_fast_agm] baseline == fast selections every trial: {all_match}")
    print(f"[check_fast_agm] total time baseline={t_base:.4f}s  fast={t_fast:.4f}s "
          f"(speed-up x{t_base / max(t_fast, 1e-12):.1f})")


# ----------------------------------------------------------------------------
# 6. Plotting (optional)
# ----------------------------------------------------------------------------


def make_plots(summary, cfg, nadapt_list, outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    os.makedirs(outdir, exist_ok=True)
    colors = {"AG-M": "tab:orange", "DG-E": "tab:purple", "EG-M": "tab:blue",
              "homogeneous": "gray"}
    markers = {"AG-M": "o", "DG-E": "^", "EG-M": "s", "homogeneous": "x"}

    # Fig. 8-like: spread & RMSE vs. Nadapt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m in METHODS:
        spread = [summary[(m, n)]["spread"] for n in nadapt_list]
        rmse = [summary[(m, n)]["rmse"] for n in nadapt_list]
        axes[0].plot(nadapt_list, spread, marker=markers[m], color=colors[m], label=m)
        axes[1].plot(nadapt_list, rmse, marker=markers[m], color=colors[m], label=m)
    axes[0].set_xlabel("Number of adaptive observations")
    axes[0].set_ylabel("Forecast ensemble spread")
    axes[1].set_xlabel("Number of adaptive observations")
    axes[1].set_ylabel("Forecast RMSE")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig8_spread_rmse_vs_nadapt.png"), dpi=150)
    plt.close(fig)

    # Fig. 6-like: first-site selection frequency, for the largest Nadapt tested
    n_ref = nadapt_list[-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in ["AG-M", "DG-E", "EG-M"]:
        ax.plot(range(cfg.Nx), summary[(m, n_ref)]["first_site_freq"],
                 color=colors[m], label=m)
    ax.axvspan(-0.5, len(cfg.ocean_idx) - 0.5, color="tab:blue", alpha=0.08)
    ax.axvspan(len(cfg.ocean_idx) - 0.5, cfg.Nx - 0.5, color="tab:orange", alpha=0.08)
    ax.set_xlabel("Grid point")
    ax.set_ylabel("First-site selection frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig6_first_site_frequency.png"), dpi=150)
    plt.close(fig)

    # Fig. 7-like: mean consecutive-selection distance vs. Nadapt
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ["AG-M", "DG-E", "EG-M"]:
        dist = [summary[(m, n)]["mean_consec_dist"] for n in nadapt_list]
        ax.plot(nadapt_list, dist, marker=markers[m], color=colors[m], label=m)
    ax.set_xlabel("Number of adaptive observations")
    ax.set_ylabel("Mean consecutive-selection distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig7_mean_consecutive_distance.png"), dpi=150)
    plt.close(fig)

    print(f"Saved figures to: {outdir}")


# ----------------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cycles", type=int, default=200,
                         help="number of 6h assimilation cycles (paper uses 14600 = 10 years)")
    parser.add_argument("--burnin", type=int, default=200,
                         help="number of control-cycle burn-in steps before recording stats")
    parser.add_argument("--nadapt", type=int, nargs="+", default=[1, 3, 5, 7, 10],
                         help="list of adaptive-observation counts to sweep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true", help="save PNG figures")
    parser.add_argument("--outdir", type=str, default="osse_figs")
    parser.add_argument("--skip-check", action="store_true",
                         help="skip the baseline-vs-fast-AG-M equivalence/speed check")
    args = parser.parse_args()

    cfg = L96Config()

    if not args.skip_check:
        check_fast_agm(cfg)

    print(f"Running OSSE: n_cycles={args.n_cycles}, nadapt={args.nadapt}, seed={args.seed}")
    t0 = time.perf_counter()
    summary = run_osse(cfg, args.n_cycles, args.nadapt, burnin=args.burnin, seed=args.seed)
    print(f"Done in {time.perf_counter() - t0:.1f}s")

    print(f"\n{'method':<12}{'Nadapt':>8}{'spread':>12}{'RMSE':>12}{'mean_dist':>12}")
    for n in args.nadapt:
        for m in METHODS:
            s = summary[(m, n)]
            print(f"{m:<12}{n:>8}{s['spread']:>12.4f}{s['rmse']:>12.4f}{s['mean_consec_dist']:>12.3f}")

    if args.plot:
        make_plots(summary, cfg, args.nadapt, outdir=args.outdir)


if __name__ == "__main__":
    main()
