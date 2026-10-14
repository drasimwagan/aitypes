import numpy as np
from model import monte_carlo, analytic_curves
p = dict(N=10, L=20, s2=1.0, r=1.0, rho=0.5, rhoL=0.5, tau=20, Lam=10)
Qs = np.array([0.316, 10.0])
raws = [monte_carlo(Qs, p, E=110000, burn=10000, seed=s)[0] for s in (21, 22, 23)]
an = analytic_curves(Qs, p)
for k in an:
    mc = np.mean([r[k] for r in raws], axis=0); se = np.std([r[k] for r in raws], axis=0)/np.sqrt(3)
    print(k, "analytic", np.round(an[k],4), "mc", np.round(mc,4), "rel err", np.round(np.abs(mc-an[k])/an[k],4), "rel se", np.round(se/an[k],4))
