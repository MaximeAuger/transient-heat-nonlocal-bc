"""
Master script: run all experiments of the transient PINN vs VPINN benchmark
and regenerate every table and figure in ../results/.

Experiments
-----------
E1. Linear case (beta = 0, alpha = 1), two manufactured solutions
E2. Nonlinearity  beta in {0.5, 1, 2, 3, 5}
E3. Nonlocal coefficient alpha in {0.5, 1, 1.5, 1.9, 1.99, 2.5, 3}
    (alpha* = 2 is the critical value) + source-free problem vs FD reference
E4. Ablations: test space K_x x K_t, horizon T, quadrature points, network
E5. Extension: nonlocal initial condition u(x,0) = gamma u(x,T) + phi(x)

Every configuration is run with 3 seeds; tables report mean +/- std.

Usage
-----
    python run_all.py                 # full benchmark
    python run_all.py --quick         # tiny budget, 1 seed (-> results/quick/)
    python run_all.py --only E1,E2    # subset of experiments
    python run_all.py --workers 4     # number of parallel processes
    python run_all.py --force         # ignore the per-run cache

Runs are independent single-threaded processes; finished runs are cached in
results/cache/ so that an interrupted benchmark can be resumed.

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import argparse
import csv
import functools
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(SRC_DIR, "..", "results")

SEEDS = (0, 1, 2)
BETAS = (0.5, 1.0, 2.0, 3.0, 5.0)
ALPHAS = (0.5, 1.0, 1.5, 1.9, 1.99, 2.5, 3.0)
ALPHAS_FREE = (1.0, 2.5, 3.0)                 # source-free problem (FD ref.)
TEST_SPACES = ((4, 4), (8, 6), (12, 8), (16, 12), (24, 16))
QUAD_POINTS = (16, 24, 32, 48)
NETWORKS = ((16, 4), (32, 2), (32, 4), (32, 6), (64, 4))   # (width, depth)
GAMMAS = (-1.0, 0.5, 2.0)                     # nonlocal initial condition
FD_TABLE_GRID = 200          # FD grid shown next to the networks in tables
FD_REFERENCE_GRID = 1600     # FD grid used as reference (source-free cases)

SCALARS = ("L2", "L2_rel", "Linf", "H1_T", "nonlocal_max", "flux_ad_rel",
           "flux_lam_rel", "time", "n_lbfgs_evals")


# ===================================================================
# Job definition
# ===================================================================

def full_config(method, cfg):
    """Merge a job configuration with the solver defaults."""
    if method == "pinn":
        from pinn_solver import DEFAULT_CONFIG
    else:
        from vpinn_solver import DEFAULT_CONFIG
    return {**DEFAULT_CONFIG, **cfg}


def job_id(method, cfg):
    full = full_config(method, cfg)
    if full["gamma"] == 0.0:       # keeps the keys of the local-IC runs stable
        del full["gamma"]
    blob = json.dumps({"method": method, "cfg": full}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def build_jobs(only, quick):
    """Return the list of (experiment, label, method, cfg, save_fields)."""
    base = {}
    seeds = SEEDS
    if quick:
        base = {"n_adam": 300, "n_lbfgs": 300, "n_x": 16, "n_t": 16}
        seeds = (0,)
    jobs = []

    def add(exp, label, methods=("pinn", "vpinn"), fields=False, **cfg):
        if only and exp.split("_")[0] not in only:
            return
        for method in methods:
            for seed in seeds:
                c = {**base, **cfg, "seed": seed}
                if method == "pinn":       # VPINN-only settings
                    c = {k: v for k, v in c.items()
                         if k not in ("n_test_x", "n_test_t")}
                jobs.append({"exp": exp, "label": label, "method": method,
                             "cfg": c, "fields": fields and seed == seeds[0],
                             "id": job_id(method, c)})

    # --- E1: linear case, two manufactured solutions ---
    for kind in ("decay", "oscillating"):
        add("E1", kind, fields=True, problem=kind)

    # --- E2: nonlinearity ---
    add("E2", 0.0, problem="decay", beta=0.0)
    for beta in BETAS:
        add("E2", beta, problem="decay", beta=beta)

    # --- E3: nonlocal coefficient ---
    for alpha in ALPHAS:
        add("E3", alpha, problem="decay", alpha=alpha)
    for alpha in ALPHAS_FREE:
        add("E3_free", alpha, fields=True, problem="source_free", alpha=alpha)

    # --- E4: ablations ---
    for kx, kt in TEST_SPACES:
        add("E4_tests", f"{kx}x{kt}", methods=("vpinn",),
            n_test_x=kx, n_test_t=kt)
    add("E4_tests", "PINN", methods=("pinn",))
    for T in (1.0, 5.0):
        add("E4_horizon", f"T={T:g}", T=T)
    add("E4_horizon", "T=5, 12x16", methods=("vpinn",), T=5.0,
        n_test_x=12, n_test_t=16)
    for n in QUAD_POINTS:
        if quick and n > 24:
            continue
        add("E4_quad", n, n_x=n, n_t=n)
    for width, depth in NETWORKS:
        add("E4_net", f"{width}x{depth}", n_hidden=width, n_layers=depth)

    # --- E5: nonlocal initial condition (Sapagovas type) ---
    add("E5", 0.0, problem="decay")
    for gamma in GAMMAS:
        add("E5", gamma, problem="decay", gamma=gamma)
    return jobs


# ===================================================================
# Worker
# ===================================================================

@functools.lru_cache(maxsize=None)
def fd_reference(alpha, beta, T, n=FD_REFERENCE_GRID):
    from exact_solution import source_free
    from fd_reference import solve_crank_nicolson
    return solve_crank_nicolson(source_free(alpha, beta, T), n, n)


def run_job(job, cache_dir, force=False):
    """Train one model (single thread) and return its scalar metrics."""
    import torch
    torch.set_num_threads(1)
    from pinn_solver import train_pinn
    from vpinn_solver import train_vpinn

    path = os.path.join(cache_dir, job["id"] + ".json")
    npz = os.path.join(cache_dir, job["id"] + ".npz")
    if (not force and os.path.exists(path)
            and (not job["fields"] or os.path.exists(npz))):
        with open(path) as fh:
            return json.load(fh)

    cfg = job["cfg"]
    reference = None
    if cfg.get("problem") == "source_free":
        reference = fd_reference(cfg.get("alpha", 1.0), cfg.get("beta", 0.0),
                                 cfg.get("T", 1.0))
    train = train_pinn if job["method"] == "pinn" else train_vpinn
    res = train(cfg, verbose=False, reference=reference)

    err = res["errors"]
    rec = {"id": job["id"], "method": job["method"], "cfg": res["config"],
           "time": res["time"], "n_lbfgs_evals": res["n_lbfgs_evals"],
           "lbfgs_failed": bool(res["lbfgs_failed"]),
           "final_loss": res["history"]["loss"][-1]}
    rec.update({k: float(v) for k, v in err.items() if np.ndim(v) == 0})

    if job["fields"]:
        arrays = {k: v for k, v in err.items() if np.ndim(v) > 0}
        arrays.update({"hist_" + k: np.asarray(v)
                       for k, v in res["history"].items()})
        np.savez_compressed(npz, **arrays)
    with open(path, "w") as fh:
        json.dump(rec, fh)
    return rec


def run_all_jobs(jobs, cache_dir, workers, force):
    """Run the unique jobs in parallel; return {id: record}."""
    unique = {}
    for job in jobs:
        if job["id"] in unique:
            unique[job["id"]]["fields"] |= job["fields"]
        else:
            unique[job["id"]] = dict(job)
    todo = sorted(unique.values(),          # most expensive first
                  key=lambda j: -(full_config(j["method"], j["cfg"])["n_x"] ** 2
                                  * full_config(j["method"], j["cfg"])["n_hidden"]))
    print(f"{len(jobs)} runs requested, {len(todo)} unique, {workers} workers")
    records, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, j, cache_dir, force): j for j in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            rec = fut.result()
            records[job["id"]] = rec
            flag = "  L-BFGS FAILED" if rec["lbfgs_failed"] else ""
            print(f"  [{n:3d}/{len(todo)}] {time.time() - t0:6.0f}s  "
                  f"{job['exp']:10s} {str(job['label']):12s} "
                  f"{job['method']:5s} seed={job['cfg']['seed']}  "
                  f"L2={rec['L2']:.2e}  ({rec['time']:.0f}s){flag}", flush=True)
    return records, time.time() - t0


# ===================================================================
# Aggregation
# ===================================================================

class Results:
    """Access to per-configuration statistics (mean / std over seeds)."""

    def __init__(self, jobs, records, cache_dir):
        self.cache_dir = cache_dir
        self.groups = {}
        for job in jobs:
            key = (job["exp"], job["label"], job["method"])
            self.groups.setdefault(key, []).append(
                (job, records[job["id"]]))

    def has(self, exp, label, method):
        return (exp, label, method) in self.groups

    def values(self, exp, label, method, metric):
        return np.array([rec.get(metric, np.nan)
                         for _, rec in self.groups[(exp, label, method)]])

    def stat(self, exp, label, method, metric):
        v = self.values(exp, label, method, metric)
        return float(np.mean(v)), float(np.std(v))

    def n_failed(self, exp, label, method):
        return sum(rec["lbfgs_failed"]
                   for _, rec in self.groups[(exp, label, method)])

    def fields(self, exp, label, method):
        for job, _ in self.groups[(exp, label, method)]:
            path = os.path.join(self.cache_dir, job["id"] + ".npz")
            if job["fields"] and os.path.exists(path):
                return dict(np.load(path))
        return None

    def labels(self, exp):
        out = []
        for (e, label, _) in self.groups:
            if e == exp and label not in out:
                out.append(label)
        return out


def fmt(mean, std=None):
    if mean is None or not np.isfinite(mean):
        return "n/a"
    if std is None:
        return f"{mean:.2e}"
    return f"{mean:.2e} ± {std:.1e}"


def cell(res, exp, label, method, metric):
    if not res.has(exp, label, method):
        return "n/a"
    return fmt(*res.stat(exp, label, method, metric))


def fd_manufactured_error(kind, alpha, beta, T=1.0, n=FD_TABLE_GRID):
    """L2 / Linf / flux errors of the Crank-Nicolson scheme (n x n grid)."""
    from exact_solution import manufactured
    from fd_reference import fd_errors
    t0 = time.time()
    prob = manufactured(kind, alpha, beta, T)
    e = fd_errors(prob, n, n)
    t = e["sol"]["t"]
    flux_ref = prob.flux0(t)
    e["flux_rel"] = (np.sqrt(np.trapezoid((e["sol"]["flux0"] - flux_ref) ** 2, t))
                     / np.sqrt(np.trapezoid(flux_ref ** 2, t)))
    xg, tg = np.meshgrid(e["sol"]["x"], t, indexing="ij")
    e["L2_rel"] = e["L2"] / np.sqrt(np.trapezoid(
        np.trapezoid(prob.u(xg, tg) ** 2, t, axis=1), e["sol"]["x"]))
    e["time"] = time.time() - t0
    return e


# ===================================================================
# Tables
# ===================================================================

def winner(res, exp, label, metric="L2"):
    """Name the method with the lower mean error and say whether the
    mean +/- std intervals overlap."""
    mp, sp_ = res.stat(exp, label, "pinn", metric)
    mv, sv = res.stat(exp, label, "vpinn", metric)
    name = "VPINN" if mv < mp else "PINN"
    ratio = max(mp, mv) / min(mp, mv)
    overlap = (min(mp, mv) + (sv if mv < mp else sp_)
               >= max(mp, mv) - (sp_ if mv < mp else sv))
    return name, ratio, overlap


def write_tables(res, out_dir, meta):
    from critical_alpha import leading_eigenvalue
    L = []
    L.append("# Results tables\n")
    L.append(f"Generated by `src/run_all.py` — {meta['n_runs']} training runs, "
             f"{meta['wall']:.0f} s wall-clock with {meta['workers']} "
             f"single-thread workers on {meta['cpu']} (training time summed "
             f"over runs: {meta['cpu_time']:.0f} s).  "
             f"Budget per run: {meta['n_adam']} Adam iterations + "
             f"{meta['n_lbfgs']} L-BFGS evaluations.  "
             "Values are mean ± std over the seeds "
             f"{list(meta['seeds'])}.\n")
    fd = {}

    # ---------------- E1 ----------------
    rows = [("L2 error (space-time)", "L2"), ("relative L2 error", "L2_rel"),
            ("L∞ error", "Linf"), ("H1 error at t = T", "H1_T"),
            ("nonlocal condition (max over t)", "nonlocal_max"),
            ("flux −k u_x(0,t) by autograd (rel. L2)", "flux_ad_rel"),
            ("flux λ(t) learned (rel. L2)", "flux_lam_rel"),
            ("training time (s)", "time"),
            ("L-BFGS evaluations used", "n_lbfgs_evals")]
    for kind in res.labels("E1"):
        e = fd[("E1", kind)] = fd_manufactured_error(kind, 1.0, 0.0)
        L.append(f"\n## E1 — linear case (β = 0, α = 1), `{kind}` solution\n")
        L.append(f"| Metric | PINN | VPINN | FD Crank–Nicolson "
                 f"{FD_TABLE_GRID}×{FD_TABLE_GRID} |")
        L.append("|---|---|---|---|")
        fd_col = {"L2": e["L2"], "L2_rel": e["L2_rel"], "Linf": e["Linf"],
                  "flux_ad_rel": e["flux_rel"], "time": e["time"]}
        for name, key in rows:
            f = fmt(fd_col[key]) if key in fd_col else "—"
            pinn = "—" if key == "flux_lam_rel" else cell(res, "E1", kind, "pinn", key)
            L.append(f"| {name} | {pinn} | "
                     f"{cell(res, 'E1', kind, 'vpinn', key)} | {f} |")

    # ---------------- E2 ----------------
    if res.labels("E2"):
        L.append("\n## E2 — nonlinearity k(u) = 1 + β u² (α = 1, `decay`)\n")
        L.append("| β | PINN L2 | VPINN L2 | lower mean | ratio | PINN flux (AD) "
                 "| VPINN flux λ(t) | FD L2 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for beta in res.labels("E2"):
            name, ratio, overlap = winner(res, "E2", beta)
            tag = f"{name} ×{ratio:.1f}" + (" (within ±1 std)" if overlap else "")
            e = fd[("E2", beta)] = fd_manufactured_error("decay", 1.0, beta)
            L.append(f"| {beta:g} | {cell(res, 'E2', beta, 'pinn', 'L2')} | "
                     f"{cell(res, 'E2', beta, 'vpinn', 'L2')} | {tag} | "
                     f"{ratio:.2f} | "
                     f"{cell(res, 'E2', beta, 'pinn', 'flux_ad_rel')} | "
                     f"{cell(res, 'E2', beta, 'vpinn', 'flux_lam_rel')} | "
                     f"{fmt(e['L2'])} |")

    # ---------------- E3 ----------------
    if res.labels("E3"):
        L.append("\n## E3 — nonlocal coefficient α (β = 0, `decay`), "
                 "critical value α* = 2\n")
        L.append("| α | σ₁(α) | e^{σ₁T} | PINN L2 | VPINN L2 | lower mean | "
                 "PINN nonlocal | VPINN nonlocal | FD L2 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for alpha in res.labels("E3"):
            s1 = leading_eigenvalue(alpha)
            name, ratio, overlap = winner(res, "E3", alpha)
            tag = f"{name} ×{ratio:.1f}" + (" (within ±1 std)" if overlap else "")
            e = fd[("E3", alpha)] = fd_manufactured_error("decay", alpha, 0.0)
            L.append(f"| {alpha:g} | {s1:+.3f} | {np.exp(s1):.3g} | "
                     f"{cell(res, 'E3', alpha, 'pinn', 'L2')} | "
                     f"{cell(res, 'E3', alpha, 'vpinn', 'L2')} | {tag} | "
                     f"{cell(res, 'E3', alpha, 'pinn', 'nonlocal_max')} | "
                     f"{cell(res, 'E3', alpha, 'vpinn', 'nonlocal_max')} | "
                     f"{fmt(e['L2'])} |")
    if res.labels("E3_free"):
        from exact_solution import source_free
        from fd_reference import solve_crank_nicolson
        L.append("\n## E3 — source-free problem (f = 0, g0 = g1 = 0), no exact "
                 f"solution: errors against the FD reference "
                 f"({FD_REFERENCE_GRID}×{FD_REFERENCE_GRID})\n")
        L.append("| α | σ₁(α) | PINN rel. L2 | VPINN rel. L2 | PINN flux (AD) | "
                 "VPINN flux λ(t) | FD reference self-error (rel. L2, est.) |")
        L.append("|---|---|---|---|---|---|---|")
        for alpha in res.labels("E3_free"):
            T = 1.0
            fine = fd_reference(alpha, 0.0, T)
            half = solve_crank_nicolson(source_free(alpha, 0.0, T),
                                        FD_REFERENCE_GRID // 2,
                                        FD_REFERENCE_GRID // 2)
            d = fine["u"][::2, ::2] - half["u"]
            norm = lambda a: np.sqrt(np.trapezoid(np.trapezoid(
                a ** 2, half["t"], axis=1), half["x"]))
            self_err = norm(d) / 3.0 / norm(half["u"])     # Richardson, order 2
            fd[("E3_free", alpha)] = self_err
            L.append(f"| {alpha:g} | {leading_eigenvalue(alpha):+.3f} | "
                     f"{cell(res, 'E3_free', alpha, 'pinn', 'L2_rel')} | "
                     f"{cell(res, 'E3_free', alpha, 'vpinn', 'L2_rel')} | "
                     f"{cell(res, 'E3_free', alpha, 'pinn', 'flux_ad_rel')} | "
                     f"{cell(res, 'E3_free', alpha, 'vpinn', 'flux_lam_rel')} | "
                     f"{fmt(self_err)} |")

    # ---------------- E4 ----------------
    if res.labels("E4_tests"):
        L.append("\n## E4a — size of the test space K_x × K_t (VPINN)\n")
        L.append("| Test space | residuals | L2 | H1(T) | flux λ(t) | time (s) |")
        L.append("|---|---|---|---|---|---|")
        for label in res.labels("E4_tests"):
            if label == "PINN":
                continue
            kx, kt = (int(s) for s in label.split("x"))
            L.append(f"| {kx} × {kt} | {kx * kt} | "
                     f"{cell(res, 'E4_tests', label, 'vpinn', 'L2')} | "
                     f"{cell(res, 'E4_tests', label, 'vpinn', 'H1_T')} | "
                     f"{cell(res, 'E4_tests', label, 'vpinn', 'flux_lam_rel')} | "
                     f"{res.stat('E4_tests', label, 'vpinn', 'time')[0]:.0f} |")
        if res.has("E4_tests", "PINN", "pinn"):
            L.append(f"| PINN (reference) | — | "
                     f"{cell(res, 'E4_tests', 'PINN', 'pinn', 'L2')} | "
                     f"{cell(res, 'E4_tests', 'PINN', 'pinn', 'H1_T')} | — | "
                     f"{res.stat('E4_tests', 'PINN', 'pinn', 'time')[0]:.0f} |")

    def two_method_table(exp, title, head):
        if not res.labels(exp):
            return
        L.append(f"\n## {title}\n")
        L.append(f"| {head} | PINN L2 | VPINN L2 | PINN rel. L2 | VPINN rel. L2 "
                 "| PINN time (s) | VPINN time (s) |")
        L.append("|---|---|---|---|---|---|---|")
        for label in res.labels(exp):
            tp = (f"{res.stat(exp, label, 'pinn', 'time')[0]:.0f}"
                  if res.has(exp, label, "pinn") else "—")
            tv = (f"{res.stat(exp, label, 'vpinn', 'time')[0]:.0f}"
                  if res.has(exp, label, "vpinn") else "—")
            L.append(f"| {label} | {cell(res, exp, label, 'pinn', 'L2')} | "
                     f"{cell(res, exp, label, 'vpinn', 'L2')} | "
                     f"{cell(res, exp, label, 'pinn', 'L2_rel')} | "
                     f"{cell(res, exp, label, 'vpinn', 'L2_rel')} | {tp} | {tv} |")

    two_method_table("E4_horizon", "E4b — time horizon T", "Horizon")
    two_method_table("E4_quad", "E4c — quadrature / collocation points "
                     "n_x = n_t", "Points per direction")
    two_method_table("E4_net", "E4d — network size (width × depth)", "Network")

    if res.labels("E5"):
        L.append("\n## E5 — nonlocal initial condition "
                 "u(x,0) = γ u(x,T) + φ(x) (β = 0, α = 1, `decay`)\n")
        L.append("| γ | PINN L2 | VPINN L2 | lower mean | PINN H1(T) | "
                 "VPINN H1(T) | PINN flux (AD) | VPINN flux λ(t) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for gamma in res.labels("E5"):
            name, ratio, overlap = winner(res, "E5", gamma)
            tag = f"{name} ×{ratio:.1f}" + (" (within ±1 std)" if overlap else "")
            lab = f"{gamma:g}" + (" (local IC)" if gamma == 0.0 else "")
            L.append(f"| {lab} | {cell(res, 'E5', gamma, 'pinn', 'L2')} | "
                     f"{cell(res, 'E5', gamma, 'vpinn', 'L2')} | {tag} | "
                     f"{cell(res, 'E5', gamma, 'pinn', 'H1_T')} | "
                     f"{cell(res, 'E5', gamma, 'vpinn', 'H1_T')} | "
                     f"{cell(res, 'E5', gamma, 'pinn', 'flux_ad_rel')} | "
                     f"{cell(res, 'E5', gamma, 'vpinn', 'flux_lam_rel')} |")

    failed = [(k, res.n_failed(*k)) for k in res.groups if res.n_failed(*k)]
    L.append("\n## Optimiser failures\n")
    if failed:
        for (exp, label, method), n in failed:
            L.append(f"- {exp} / {label} / {method}: L-BFGS produced a "
                     f"non-finite loss in {n} run(s); the Adam state was kept.")
    else:
        L.append("No run produced a non-finite loss.")

    with open(os.path.join(out_dir, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    # one line per run
    with open(os.path.join(out_dir, "all_runs.csv"), "w", newline="",
              encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["experiment", "label", "method", "seed", *SCALARS,
                     "final_loss", "lbfgs_failed"])
        for (exp, label, method), items in res.groups.items():
            for job, rec in items:
                wr.writerow([exp, label, method, job["cfg"]["seed"]]
                            + [f"{rec.get(k, float('nan')):.6e}" for k in SCALARS]
                            + [f"{rec['final_loss']:.6e}", rec["lbfgs_failed"]])
    return fd


# ===================================================================
# Figures
# ===================================================================

# Categorical colours (fixed assignment): PINN, VPINN, finite differences
C_PINN, C_VPINN, C_FD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
STYLE = {"pinn": dict(color=C_PINN, marker="o", label="PINN"),
         "vpinn": dict(color=C_VPINN, marker="s", label="VPINN")}
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
             "#184f95", "#0d366b"]


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
        "font.family": "sans-serif",
        "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
        "axes.titlelocation": "left",
    })
    return plt


def _errorbar(ax, x, res, exp, labels, method, metric, **kw):
    stats = [res.stat(exp, lab, method, metric) for lab in labels]
    mean = np.array([s[0] for s in stats])
    std = np.array([s[1] for s in stats])
    st = {**STYLE[method], **kw}
    lower = np.minimum(std, 0.9 * mean)          # keep log-scale bars positive
    ax.errorbar(x, mean, yerr=[lower, std], capsize=3, elinewidth=1.2,
                markeredgecolor=SURFACE, markeredgewidth=1.0, **st)
    for xi, lab in zip(x, labels):               # individual seeds
        v = res.values(exp, lab, method, metric)
        ax.scatter(np.full_like(v, xi), v, s=10, color=st["color"], alpha=0.35,
                   linewidths=0, zorder=1)
    return mean, std


def fig_error_maps(plt, res, out_dir):
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    kinds = [k for k in res.labels("E1")
             if res.fields("E1", k, "pinn") is not None]
    if not kinds:
        return
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    fig, axes = plt.subplots(len(kinds), 3, figsize=(13, 3.6 * len(kinds)),
                             squeeze=False)
    for r, kind in enumerate(kinds):
        fp, fv = res.fields("E1", kind, "pinn"), res.fields("E1", kind, "vpinn")
        x, t = fp["x"], fp["t"]
        ax = axes[r, 0]
        im = ax.pcolormesh(t, x, fp["u_ref"], cmap=cmap, shading="auto")
        fig.colorbar(im, ax=ax, label="u")
        ax.set_title(f"Exact solution ({kind})")
        ep, ev = np.abs(fp["error"]), np.abs(fv["error"])
        vmax = max(ep.max(), ev.max())
        norm = LogNorm(vmin=vmax * 1e-3, vmax=vmax)
        for ax, e, name in ((axes[r, 1], ep, "PINN"), (axes[r, 2], ev, "VPINN")):
            im = ax.pcolormesh(t, x, np.maximum(e, vmax * 1e-3), cmap=cmap,
                               norm=norm, shading="auto")
            fig.colorbar(im, ax=ax, label="|u_θ − u|")
            ax.set_title(f"{name} error — max {e.max():.1e}")
        for ax in axes[r]:
            ax.set_xlabel("t")
            ax.set_ylabel("x   (x = 0: nonlocal boundary)")
            ax.grid(False)
    fig.suptitle("E1 — linear case: space-time error maps (seed 0, common "
                 "colour scale for the two methods)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_error_maps.png"))
    plt.close(fig)


def fig_flux(plt, res, out_dir):
    kinds = [k for k in res.labels("E1")
             if res.fields("E1", k, "vpinn") is not None]
    if not kinds:
        return
    fig, axes = plt.subplots(len(kinds), 2, figsize=(12, 3.8 * len(kinds)),
                             squeeze=False)
    for r, kind in enumerate(kinds):
        fp, fv = res.fields("E1", kind, "pinn"), res.fields("E1", kind, "vpinn")
        t = fv["t"]
        ax = axes[r, 0]
        ax.plot(t, fv["flux_ref"], color=INK, lw=4, alpha=0.25,
                label="exact  −k(u) u_x(0, t)")
        ax.plot(t, fv["flux_lam"], color=C_VPINN, label="VPINN  learned λ(t)")
        ax.plot(t, fp["flux_ad"], color=C_PINN, ls=(0, (4, 2)),
                label="PINN  autograd flux")
        ax.set_xlabel("t")
        ax.set_ylabel("boundary flux at x = 0")
        ax.set_title(f"Boundary flux ({kind})")
        ax.legend()
        ax = axes[r, 1]
        ax.semilogy(t, np.abs(fv["flux_lam"] - fv["flux_ref"]) + 1e-16,
                    color=C_VPINN, label="VPINN  learned λ(t)")
        ax.semilogy(t, np.abs(fv["flux_ad"] - fv["flux_ref"]) + 1e-16,
                    color=C_VPINN, lw=1.2, ls=(0, (1, 1.5)),
                    label="VPINN  autograd flux")
        ax.semilogy(t, np.abs(fp["flux_ad"] - fp["flux_ref"]) + 1e-16,
                    color=C_PINN, ls=(0, (4, 2)), label="PINN  autograd flux")
        ax.set_xlabel("t")
        ax.set_ylabel("|flux error|")
        ax.set_title("Absolute error of the flux")
        ax.legend()
    fig.suptitle("E1 — boundary flux: learned Lagrange multiplier λ(t) vs "
                 "exact flux (seed 0)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_boundary_flux.png"))
    plt.close(fig)


def fig_beta(plt, res, out_dir, fd):
    betas = res.labels("E2")
    if len(betas) < 2:
        return
    x = np.array(betas, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    mp, _ = _errorbar(ax, x, res, "E2", betas, "pinn", "L2")
    mv, _ = _errorbar(ax, x, res, "E2", betas, "vpinn", "L2")
    ax.plot(x, [fd[("E2", b)]["L2"] for b in betas], color=C_FD, marker="^",
            markeredgecolor=SURFACE, label=f"FD {FD_TABLE_GRID}×{FD_TABLE_GRID}")
    ax.set_yscale("log")
    ax.set_xlabel("nonlinearity β")
    ax.set_ylabel("space-time L2 error")
    ax.set_title("Error vs β (mean ± std, 3 seeds)")
    ax.legend()

    ax = axes[1]
    ratio = mp / mv
    ax.axhline(1.0, color=MUTED, lw=1.0)
    ax.plot(x, ratio, color=INK2, marker="D", markeredgecolor=SURFACE)
    ax.set_yscale("log")
    lo, hi = min(ratio.min(), 0.5), max(ratio.max(), 2.0)
    ax.set_ylim(lo / 1.5, hi * 1.5)
    ax.text(0.98, 0.96, "VPINN more accurate", transform=ax.transAxes,
            ha="right", va="top", color=INK2)
    ax.text(0.98, 0.04, "PINN more accurate", transform=ax.transAxes,
            ha="right", va="bottom", color=INK2)
    ax.set_xlabel("nonlinearity β")
    ax.set_ylabel("L2(PINN) / L2(VPINN)")
    ax.set_title("Error ratio (1 = tie)")

    ax = axes[2]
    _errorbar(ax, x, res, "E2", betas, "pinn", "flux_ad_rel",
              label="PINN  autograd flux")
    _errorbar(ax, x, res, "E2", betas, "vpinn", "flux_lam_rel",
              label="VPINN  learned λ(t)")
    ax.set_yscale("log")
    ax.set_xlabel("nonlinearity β")
    ax.set_ylabel("relative L2 error of the flux")
    ax.set_title("Boundary flux error vs β")
    ax.legend()
    fig.suptitle("E2 — effect of the nonlinearity k(u) = 1 + β u²  (α = 1)",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_error_vs_beta.png"))
    plt.close(fig)


def fig_alpha(plt, res, out_dir, fd):
    from critical_alpha import discrete_leading_eigenvalue, leading_eigenvalue
    alphas = res.labels("E3")
    if len(alphas) < 2:
        return
    x = np.array(alphas, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    _errorbar(ax, x, res, "E3", alphas, "pinn", "L2")
    _errorbar(ax, x, res, "E3", alphas, "vpinn", "L2")
    ax.plot(x, [fd[("E3", a)]["L2"] for a in alphas], color=C_FD, marker="^",
            markeredgecolor=SURFACE, label=f"FD {FD_TABLE_GRID}×{FD_TABLE_GRID}")
    ax.axvline(2.0, color=MUTED, lw=1.0)
    ax.text(2.03, 0.97, "α* = 2", transform=ax.get_xaxis_transform(),
            va="top", color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("nonlocal coefficient α")
    ax.set_ylabel("space-time L2 error")
    ax.set_title("Error vs α (mean ± std, 3 seeds)")
    ax.legend(loc="lower right")

    ax = axes[1]
    a_line = np.linspace(0.3, 3.2, 300)
    ax.axhline(0.0, color=MUTED, lw=1.0)
    ax.axvline(2.0, color=MUTED, lw=1.0)
    ax.plot(a_line, [leading_eigenvalue(a) for a in a_line], color=INK2,
            label="characteristic equation")
    ax.plot(x, [discrete_leading_eigenvalue(a).real for a in x], ls="none",
            marker="^", color=C_FD, markeredgecolor=SURFACE,
            label="FD operator spectrum")
    ax.text(0.04, 0.9, "decay", transform=ax.transAxes, color=INK2)
    ax.text(0.8, 0.9, "growth", transform=ax.transAxes, color=INK2)
    ax.set_xlabel("nonlocal coefficient α")
    ax.set_ylabel("leading eigenvalue σ₁")
    ax.set_title("Homogeneous problem: σ₁(α) changes sign at α* = 2")
    ax.legend(loc="center left")

    ax = axes[2]
    free = [a for a in res.labels("E3_free")
            if res.fields("E3_free", a, "pinn") is not None]
    if free:
        a = max(free)
        fp = res.fields("E3_free", a, "pinn")
        fv = res.fields("E3_free", a, "vpinn")
        t = fp["t"]
        ax.semilogy(t, np.abs(fp["u_ref"][0]), color=INK, lw=4, alpha=0.25,
                    label="FD reference")
        ax.semilogy(t, np.abs(fv["u_pred"][0]), color=C_VPINN, label="VPINN")
        ax.semilogy(t, np.abs(fp["u_pred"][0]), color=C_PINN, ls=(0, (4, 2)),
                    label="PINN")
        s1 = leading_eigenvalue(a)
        ax.set_title(f"Source-free problem, α = {a:g}: growth rate "
                     f"σ₁ = {s1:.2f}")
        ax.set_xlabel("t")
        ax.set_ylabel("|u(0, t)|")
        ax.legend()
    fig.suptitle("E3 — nonlocal coefficient α and the growing-mode regime "
                 "(β = 0)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_error_vs_alpha.png"))
    plt.close(fig)


def fig_ablations(plt, res, out_dir):
    if not res.labels("E4_tests"):
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    # (a) test space
    ax = axes[0, 0]
    labels = [l for l in res.labels("E4_tests") if l != "PINN"]
    x = np.arange(len(labels))
    _errorbar(ax, x, res, "E4_tests", labels, "vpinn", "L2")
    if res.has("E4_tests", "PINN", "pinn"):
        m, s = res.stat("E4_tests", "PINN", "pinn", "L2")
        ax.axhspan(max(m - s, 0.1 * m), m + s, color=C_PINN, alpha=0.15, lw=0)
        ax.axhline(m, color=C_PINN, label="PINN (mean ± std)")
    ax.set_xticks(x, [l.replace("x", "×") for l in labels])
    ax.set_yscale("log")
    ax.set_xlabel("test space  K_x × K_t")
    ax.set_ylabel("space-time L2 error")
    ax.set_title("(a) Size of the VPINN test space")
    ax.legend()

    def grouped(ax, exp, xlabel, title, metric="L2",
                ylabel="space-time L2 error", connect=True):
        labels = [l for l in res.labels(exp) if res.has(exp, l, "pinn")]
        x = np.arange(len(labels))
        kw = {} if connect else {"ls": "none"}     # unordered categories
        _errorbar(ax, x - 0.04, res, exp, labels, "pinn", metric, **kw)
        _errorbar(ax, x + 0.04, res, exp, labels, "vpinn", metric, **kw)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_xticks(x, [str(l).replace("x", "×") for l in labels])
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    grouped(axes[0, 1], "E4_horizon", "time horizon", "(b) Horizon T",
            metric="L2_rel", ylabel="relative space-time L2 error",
            connect=False)
    grouped(axes[1, 0], "E4_quad", "Gauss–Legendre points per direction",
            "(c) Quadrature / collocation points")
    grouped(axes[1, 1], "E4_net", "network  width × depth", "(d) Network size",
            connect=False)
    fig.suptitle("E4 — ablations on the linear case (mean ± std, 3 seeds)",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_ablations.png"))
    plt.close(fig)


def fig_history(plt, res, out_dir):
    if not res.labels("E1"):
        return
    kind = res.labels("E1")[0]
    fp, fv = res.fields("E1", kind, "pinn"), res.fields("E1", kind, "vpinn")
    if fp is None or fv is None:
        return
    n_adam = res.groups[("E1", kind, "pinn")][0][1]["cfg"]["n_adam"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharex=True)
    for ax, f, name, parts in (
            (axes[0], fp, "PINN", (("hist_pde", "strong residual"),
                                   ("hist_I", "nonlocal condition"))),
            (axes[1], fv, "VPINN", (("hist_weak", "weak residual"),
                                    ("hist_I", "nonlocal condition")))):
        it = np.arange(1, len(f["hist_loss"]) + 1)
        ax.semilogy(it, f["hist_loss"], color=INK2, lw=1.2, label="total loss")
        for (key, lab), col in zip(parts, (C_PINN, C_VPINN)):
            ax.semilogy(it, f[key], color=col, lw=1.0, label=lab + " (unweighted)")
        ax.axvline(n_adam, color=MUTED, lw=1.0)
        ax.text(n_adam, 0.97, " Adam → L-BFGS", transform=ax.get_xaxis_transform(),
                va="top", color=INK2)
        ax.set_xlabel("loss evaluation")
        ax.set_ylabel("loss")
        ax.set_title(f"{name} training history ({kind}, seed 0)")
        ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_training_history.png"))
    plt.close(fig)


def make_figures(res, out_dir, fd):
    plt = setup_style()
    fig_error_maps(plt, res, out_dir)
    fig_flux(plt, res, out_dir)
    fig_beta(plt, res, out_dir, fd)
    fig_alpha(plt, res, out_dir, fd)
    fig_ablations(plt, res, out_dir)
    fig_history(plt, res, out_dir)


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget and a single seed (smoke test)")
    ap.add_argument("--only", default="",
                    help="comma-separated subset of E1,E2,E3,E4,E5")
    ap.add_argument("--workers", type=int,
                    default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--force", action="store_true",
                    help="recompute runs even if cached")
    args = ap.parse_args()

    out_dir = os.path.join(RESULTS_ROOT, "quick") if args.quick else RESULTS_ROOT
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    jobs = build_jobs(only, args.quick)
    records, wall = run_all_jobs(jobs, cache_dir, args.workers, args.force)
    res = Results(jobs, records, cache_dir)

    any_cfg = next(iter(records.values()))["cfg"]
    meta = {"n_runs": len(records), "wall": wall, "workers": args.workers,
            "cpu": platform.processor() or platform.machine(),
            "cpu_time": sum(r["time"] for r in records.values()),
            "n_adam": any_cfg["n_adam"], "n_lbfgs": any_cfg["n_lbfgs"],
            "seeds": sorted({j["cfg"]["seed"] for j in jobs})}
    fd = write_tables(res, out_dir, meta)
    make_figures(res, out_dir, fd)
    print(f"\nDone: {len(records)} runs in {wall:.0f}s.  "
          f"Tables and figures written to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
