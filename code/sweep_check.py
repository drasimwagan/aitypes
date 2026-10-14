"""Randomised numerical check of Theorem 1, Proposition 3 and the Q* approximation."""
import numpy as np
from model import analytic_curves, crossover_approx
from scipy.optimize import brentq
rng = np.random.default_rng(0)
fails = {"thm_small": 0, "thm_large": 0, "hive_ind": 0, "hive_lin": 0, "lin_dar": 0, "single_cross": 0}
apx_err = []
n = 400
for _ in range(n):
    p = dict(N=int(10**rng.uniform(0, 5)), L=int(rng.integers(2, 60)), s2=10**rng.uniform(-1, 1),
             r=10**rng.uniform(-1, 1), rho=rng.uniform(0.05, 1), rhoL=rng.uniform(0.05, 1),
             tau=int(rng.integers(1, 200)), Lam=int(rng.integers(0, 500)))
    abar = p["Lam"] + (p["tau"] + 1) / 2
    Q = np.logspace(-12, 4, 400)
    a = analytic_curves(Q, p)
    d = a["lineage"] - a["individual"]
    if p["N"] * p["rhoL"] > p["rho"] and abar > 1:
        if not d[0] < 0: fails["thm_small"] += 1
        if not d[-1] > 0: fails["thm_large"] += 1
        sign_changes = np.sum(np.diff(np.sign(d)) != 0)
        if sign_changes != 1: fails["single_cross"] += 1
        else:
            i = np.where(np.diff(np.sign(d)) != 0)[0][0]
            f = lambda lq: (lambda c: c["lineage"][0] - c["individual"][0])(analytic_curves(np.array([10**lq]), p))
            qx = 10 ** brentq(f, np.log10(Q[i]), np.log10(Q[i + 1]))
            apx_err.append(abs(crossover_approx(p) / qx - 1))
    # hive with same N and rho = max(rho, rhoL) must dominate both
    ph = dict(p, rho=max(p["rho"], p["rhoL"]))
    h = analytic_curves(Q, ph)["hive"]
    if np.any(h > a["individual"] * (1 + 1e-12)): fails["hive_ind"] += 1
    if np.any(h > a["lineage"] * (1 + 1e-12)): fails["hive_lin"] += 1
    if np.any(a["lineage"] >= a["darwinian"]): fails["lin_dar"] += 1
apx_err = np.array(apx_err)
print("cases:", n, "fails:", fails)
print("Q* approx rel. error: median %.3g, 90th pct %.3g, max %.3g, n=%d" % (np.median(apx_err), np.percentile(apx_err, 90), apx_err.max(), len(apx_err)))
print("share with error < 5%%: %.2f" % np.mean(apx_err < 0.05))
