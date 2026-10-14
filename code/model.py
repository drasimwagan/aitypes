"""
Episodic linear-Gaussian drift environment and adaptation profiles.

World:
    g_{e+1} = g_e + w_e,           w_e ~ N(0, Q)        (slow, shared, drifting)
    instance i in episode e:
        l_{e,i} ~ N(0, s2)                               (episode-specific offset)
        y_{e,i,n} = g_e + l_{e,i} + v,  v ~ N(0, r),  n = 1..L
    Agent predicts y before seeing it. Excess loss = E[(u - u_hat)^2], u = g + l.

Every agent runs the Bayes-optimal in-episode filter (in-context learning) from a
calibrated Gaussian prior N(m, P) on g. Agents differ only in how the prior is
produced, i.e. in their consolidation structure:

    individual  : N=1, consolidates own episode with Fisher fraction rho every episode
    hive        : N instances, shared slow memory, consolidates every episode
    lineage     : N instances, no consolidation within deployment (rho_inst = 0);
                  pooled data trained every tau episodes, deployed after lag Lam
    darwinian   : same lineage, but no in-episode learning (predicts prior mean)
"""
import numpy as np


# ---------------------------------------------------------------- analytic
def f_excess(P, s2, r, L):
    """Mean excess loss over an episode, given prior variance P on g (Prop. 1)."""
    P = np.asarray(P, dtype=float)
    n = np.arange(L)
    return np.mean(1.0 / (1.0 / (P[..., None] + s2) + n / r), axis=-1)


def P_ss(Q, sig2, x):
    """Steady-state prior variance of a scalar KF receiving x evidence units per step."""
    Q = np.asarray(Q, dtype=float)
    return 0.5 * (Q + np.sqrt(Q**2 + 4.0 * Q * sig2 / x))


def lineage_prior_vars(Q, sig2, x, tau, Lam):
    """Deployed prior variances P_A(j) = P_ss^+ + (Lam + j) Q, j = 1..tau."""
    Q = np.asarray(Q, dtype=float)
    Pplus = P_ss(Q, sig2, x) - Q
    j = np.arange(1, tau + 1)
    return Pplus[..., None] + (Lam + j) * Q[..., None]


def analytic_curves(Q, p):
    s2, r, L = p["s2"], p["r"], p["L"]
    sig2 = s2 + r / L
    out = {}
    out["individual"] = f_excess(P_ss(Q, sig2, p["rho"]), s2, r, L)
    out["hive"] = f_excess(P_ss(Q, sig2, p["N"] * p["rho"]), s2, r, L)
    PA = lineage_prior_vars(Q, sig2, p["N"] * p["rhoL"], p["tau"], p["Lam"])
    out["lineage"] = f_excess(PA, s2, r, L).mean(axis=-1)
    out["darwinian"] = (PA + s2).mean(axis=-1)
    return out


def crossover_approx(p):
    """Small-Q approximation of the individual/lineage crossing drift rate Q*."""
    sig2 = p["s2"] + p["r"] / p["L"]
    abar = p["Lam"] + (p["tau"] + 1) / 2.0
    num = p["rho"] ** -0.5 - (p["N"] * p["rhoL"]) ** -0.5
    return sig2 * num**2 / (abar - 1.0) ** 2


# ------------------------------------------------------------- Monte Carlo
def episode_excess(m, P, y, u, s2, r):
    """Exact sequential Bayes prediction within an episode.

    m, P : prior mean/var on g, broadcastable to y[..., 0]
    y    : observations, shape (..., L)
    u    : true g + l, shape (...)
    returns mean squared error of predictions of u over the episode, shape (...)
    """
    L = y.shape[-1]
    n = np.arange(L)
    S = np.concatenate([np.zeros(y.shape[:-1] + (1,)), np.cumsum(y, -1)[..., :-1]], -1)
    a = 1.0 / (np.asarray(P)[..., None] + s2)
    prec = a + n / r
    mean = (np.asarray(m)[..., None] * a + S / r) / prec
    return np.mean((u[..., None] - mean) ** 2, axis=-1)


def kf_update(m, P, z, R):
    K = P / (P + R)
    return m + K * (z - m), (1 - K) * P


def monte_carlo(Qs, p, E=30000, burn=10000, seed=0):
    rng = np.random.default_rng(seed)
    Qs = np.asarray(Qs, float)
    nQ = len(Qs)
    N, L, s2, r = p["N"], p["L"], p["s2"], p["r"]
    rho, rhoL, tau, Lam = p["rho"], p["rhoL"], p["tau"], p["Lam"]
    sig2 = s2 + r / L
    big = 1e3

    g = np.zeros(nQ)
    # individual (uses instance 0), hive (all N), lineage training KF
    mI, PI = np.zeros((nQ, N)), np.full((nQ, N), big)  # N independent individuals
    mV, PV = np.zeros(nQ), np.full(nQ, big)
    mT, PT = np.zeros(nQ), np.full(nQ, big)
    snaps = {}  # cutoff episode -> (m, P) posterior after that episode's data
    acc = {k: np.zeros(nQ) for k in ["individual", "hive", "lineage", "darwinian"]}
    reg = {k: np.zeros(nQ) for k in acc}  # paired regret vs an oracle that knows g
    count = 0

    for e in range(E):
        if e > 0:
            g = g + np.sqrt(Qs) * rng.standard_normal(nQ)
        l = np.sqrt(s2) * rng.standard_normal((nQ, N))
        u = g[:, None] + l
        y = u[..., None] + np.sqrt(r) * rng.standard_normal((nQ, N, L))
        ybar = y.mean(-1)

        # deployed lineage snapshot for this episode
        c = ((e - Lam - 1) // tau) * tau  # most recent cutoff deployed by episode e
        if c in snaps and e >= burn:
            ms, Ps = snaps[c]
            Pdep = Ps + (e - c) * Qs
            lin = episode_excess(ms[:, None], Pdep[:, None], y, u, s2, r).mean(-1)
            dar = ((u - ms[:, None]) ** 2).mean(-1)
            ind = episode_excess(mI, PI, y, u, s2, r).mean(-1)
            hiv = episode_excess(mV[:, None], PV[:, None], y, u, s2, r).mean(-1)
            ora = episode_excess(g[:, None], np.zeros((nQ, 1)), y, u, s2, r).mean(-1)
            for k, v in [("lineage", lin), ("darwinian", dar), ("individual", ind), ("hive", hiv)]:
                acc[k] += v
                reg[k] += v - ora
            count += 1

        # ---- consolidation (end of episode) ----
        # individual: own episode, Fisher fraction rho (extra channel noise)
        zI = ybar + np.sqrt(sig2 * (1 / rho - 1)) * rng.standard_normal((nQ, N))
        mI, PI = kf_update(mI, PI, zI, sig2 / rho)
        # hive: N instances pooled, each with fraction rho
        zV = (ybar + np.sqrt(sig2 * (1 / rho - 1)) * rng.standard_normal((nQ, N))).mean(-1)
        mV, PV = kf_update(mV, PV, zV, sig2 / (rho * N))
        # lineage training: N instances pooled with fraction rhoL
        zT = (ybar + np.sqrt(sig2 * (1 / rhoL - 1)) * rng.standard_normal((nQ, N))).mean(-1)
        mT, PT = kf_update(mT, PT, zT, sig2 / (rhoL * N))
        if e % tau == 0:
            snaps[e] = (mT.copy(), PT.copy())
            for k in [k for k in snaps if k < e - 3 * tau - Lam]:
                del snaps[k]
        # drift
        PI, PV, PT = PI + Qs[:, None], PV + Qs, PT + Qs

    return ({k: v / count for k, v in acc.items()},
            {k: v / count for k, v in reg.items()})
