#!/usr/bin/env python3
"""RK4 verification of eta = <v>/v_p (Fig. 3 and Table 4 of the JGCD Technical Note).

Integrates the planar two-body equations of motion with tangential thrust inside the
thrust zone r <= R_t, normalized so that mu = (1 - kappa)/2, v_p = 1, r_p = 1, R_t = lambda,
and compares the simulated efficiency with the analytical value, Eq. (15).
Outputs: table4_initial_conditions.txt, fig_verification.pdf / .png in the current directory.
"""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from scipy.integrate import quad

def eta_analytical(kappa, lam, n_quad=500):
    u_min = 1.0 / lam
    s_nodes, s_weights = np.polynomial.legendre.leggauss(n_quad)
    s = 0.5 * (s_nodes + 1.0)
    ws = 0.5 * s_weights
    u = 1.0 - (1.0 - u_min) * s**2
    v_over_vp = np.sqrt(kappa + (1.0 - kappa) * u)
    denom = u**2 * np.sqrt(u + kappa)
    return np.sum(ws * v_over_vp / denom) / np.sum(ws / denom)

def simulate_flyby(kappa, lam, epsilon_param, n_steps=500000):
    mu = (1.0 - kappa) / 2.0
    v_p = 1.0
    r_p = 1.0
    R_t = lam * r_p
    h = r_p * v_p
    e = (1.0 + kappa) / (1.0 - kappa)

    cos_f0 = np.clip((r_p * (1.0 + e) / R_t - 1.0) / e, -1, 1)
    f0 = np.arccos(cos_f0)
    r0 = R_t
    theta0 = np.pi - f0

    vr_mag = mu / h * e * np.sin(f0)
    vt_mag = mu / h * (1.0 + e * np.cos(f0))

    x0 = r0 * np.cos(theta0)
    y0 = r0 * np.sin(theta0)
    vx0 = (-vr_mag) * np.cos(theta0) + vt_mag * (-np.sin(theta0))
    vy0 = (-vr_mag) * np.sin(theta0) + vt_mag * np.cos(theta0)

    def dt_dr(r):
        u = r_p / r
        vr = v_p * np.sqrt(max(0, (1-u)*(u+kappa)))
        return 1.0 / max(vr, 1e-15)
    T_half, _ = quad(dt_dr, r_p * 1.0001, R_t, limit=200)
    T_total = 2 * T_half
    a_thr = epsilon_param * v_p / T_total

    state = np.array([x0, y0, vx0, vy0])
    dt = T_total / n_steps
    thrust_integral = 0.0
    dv_total = 0.0

    def derivs(s, thrust_on):
        x, y, vx, vy = s
        r = np.sqrt(x**2 + y**2)
        v = np.sqrt(vx**2 + vy**2)
        ax = -mu * x / r**3
        ay = -mu * y / r**3
        if thrust_on and v > 0:
            ax += a_thr * vx / v
            ay += a_thr * vy / v
        return np.array([vx, vy, ax, ay])

    passed_peri = False
    for _ in range(n_steps):
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        v = np.sqrt(vx**2 + vy**2)
        thrust_on = (r <= R_t)
        if thrust_on:
            thrust_integral += a_thr * v * dt
            dv_total += a_thr * dt
        vr = (x*vx + y*vy) / r
        if vr > 0: passed_peri = True
        if passed_peri and r > R_t: break
        k1 = derivs(state, thrust_on)
        k2 = derivs(state + 0.5*dt*k1, thrust_on)
        k3 = derivs(state + 0.5*dt*k2, thrust_on)
        k4 = derivs(state + dt*k3, thrust_on)
        state += (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    delta_eps_O = v_p * dv_total
    eta_sim = thrust_integral / delta_eps_O if delta_eps_O > 0 else 0
    eta_an = eta_analytical(kappa, lam)
    return eta_sim, eta_an, dv_total / v_p

# Eight (kappa, lambda) cases of Table 4 / Fig. 3
param_sets = [(0.01, 10), (0.05, 30), (0.05, 5), (0.1, 5), (0.1, 20), (0.3, 10), (0.5, 5), (0.8, 10)]
eps_vals = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2]
step_counts = [800000, 600000, 500000, 300000, 200000, 200000, 200000]

# Table 4: explicit initial conditions
with open('table4_initial_conditions.txt', 'w') as f:
    f.write('kappa  lambda      e     f0_rad  eta_analytical\n')
    for kv, lv in sorted(param_sets):
        e = (1 + kv) / (1 - kv)
        f0 = np.arccos((1 + e) / (lv * e) - 1 / e)
        f.write(f'{kv:5.2f} {lv:7.0f} {e:7.3f} {f0:9.3f} {eta_analytical(kv, lv):10.3f}\n')
print(open('table4_initial_conditions.txt').read())

results = []
for kv, lv in param_sets:
    for ev, ns in zip(eps_vals, step_counts):
        eta_s, eta_a, actual_eps = simulate_flyby(kv, lv, ev, n_steps=ns)
        err = abs(eta_a - eta_s)
        results.append((kv, lv, actual_eps, eta_s, eta_a, err))
        print(f"k={kv}, l={lv}, eps={actual_eps:.2e}: eta_sim={eta_s:.6f}, eta_an={eta_a:.6f}, err={err:.2e}", flush=True)

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 13, 'legend.fontsize': 9, 'figure.dpi': 200, 'savefig.bbox': 'tight'})
fig, ax = plt.subplots(figsize=(7, 5))
for kv, lv in param_sets:
    subset = [(r[2], r[5]) for r in results if r[0] == kv and r[1] == lv and r[5] > 1e-15]
    if subset:
        eps_arr, err_arr = zip(*subset)
        ax.loglog(eps_arr, err_arr, 'o-', label=rf'$\kappa={kv},\;\lambda={lv}$', markersize=5)
eps_ref = np.logspace(-4, -1, 50)
ax.loglog(eps_ref, 0.5 * eps_ref, 'k--', linewidth=0.8, label=r'$\propto \varepsilon$')
ax.set_xlabel(r'$\varepsilon = \Delta v / v_p$')
ax.set_ylabel(r'$|\eta_{\rm analytical} - \eta_{\rm sim}|$')
ax.legend(ncol=2)
ax.grid(True, alpha=0.3, which='both')
fig.savefig('fig_verification.pdf')
fig.savefig('fig_verification.png')
plt.close(fig)
print("done")
