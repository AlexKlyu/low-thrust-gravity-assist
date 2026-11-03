"""
Numerical verification of the analytical formula eta = <v>/v_p
for the efficiency of low-thrust gravity assist.

Uses RK4 integration of the two-body problem with tangential thrust.
Normalization: mu = (1-kappa)/2, v_p = 1, r_p = 1, R_t = lambda.

Author: Alexander Klyuchnikov (MIPT)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad


def eta_analytical(kappa, lam, n_quad=500):
    """Compute analytical eta via Gauss-Legendre quadrature.

    Uses singularity-removing substitution u = 1 - (1 - u_min) * s^2.
    """
    u_min = 1.0 / lam
    s_nodes, s_weights = np.polynomial.legendre.leggauss(n_quad)
    s = 0.5 * (s_nodes + 1.0)
    ws = 0.5 * s_weights
    u = 1.0 - (1.0 - u_min) * s**2
    v_over_vp = np.sqrt(kappa + (1.0 - kappa) * u)
    denom = u**2 * np.sqrt(u + kappa)
    return np.sum(ws * v_over_vp / denom) / np.sum(ws / denom)


def simulate_flyby(kappa, lam, epsilon_param, n_steps=500000):
    """Simulate a powered flyby with tangential thrust using RK4.

    Parameters
    ----------
    kappa : float
        v_inf^2 / v_p^2
    lam : float
        R_t / r_p (thrust zone radius)
    epsilon_param : float
        Target Delta v / v_p
    n_steps : int
        Number of RK4 steps

    Returns
    -------
    eta_sim : float
        Simulated efficiency
    eta_an : float
        Analytical efficiency
    actual_eps : float
        Actual Delta v / v_p achieved
    """
    mu = (1.0 - kappa) / 2.0
    v_p = 1.0
    r_p = 1.0
    R_t = lam * r_p
    h = r_p * v_p
    e = (1.0 + kappa) / (1.0 - kappa)

    # Initial position: r = R_t on incoming branch
    cos_f0 = np.clip((r_p * (1.0 + e) / R_t - 1.0) / e, -1, 1)
    f0 = np.arccos(cos_f0)
    r0 = R_t
    theta0 = np.pi - f0

    # Velocity from Keplerian solution
    vr_mag = mu / h * e * np.sin(f0)
    vt_mag = mu / h * (1.0 + e * np.cos(f0))

    x0 = r0 * np.cos(theta0)
    y0 = r0 * np.sin(theta0)
    vx0 = (-vr_mag) * np.cos(theta0) + vt_mag * (-np.sin(theta0))
    vy0 = (-vr_mag) * np.sin(theta0) + vt_mag * np.cos(theta0)

    # Flight time on unperturbed trajectory
    def dt_dr(r):
        u = r_p / r
        vr = v_p * np.sqrt(max(0, (1 - u) * (u + kappa)))
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
        vr = (x * vx + y * vy) / r
        if vr > 0:
            passed_peri = True
        if passed_peri and r > R_t:
            break
        k1 = derivs(state, thrust_on)
        k2 = derivs(state + 0.5 * dt * k1, thrust_on)
        k3 = derivs(state + 0.5 * dt * k2, thrust_on)
        k4 = derivs(state + dt * k3, thrust_on)
        state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    delta_eps_O = v_p * dv_total
    eta_sim = thrust_integral / delta_eps_O if delta_eps_O > 0 else 0
    eta_an = eta_analytical(kappa, lam)
    return eta_sim, eta_an, dv_total / v_p


if __name__ == '__main__':
    # Test cases covering wide range of kappa and lambda
    eps_vals = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 0.1]
    step_counts = [800000, 600000, 500000, 300000, 200000, 200000, 200000, 200000]
    param_sets = [(0.1, 5), (0.3, 10), (0.05, 30), (0.3, 5)]

    test_cases = []
    for kv, lv in param_sets:
        for ev, ns in zip(eps_vals, step_counts):
            test_cases.append((kv, lv, ev, ns))

    results = []
    for kv, lv, ev, ns in test_cases:
        eta_s, eta_a, actual_eps = simulate_flyby(kv, lv, ev, n_steps=ns)
        err = abs(eta_a - eta_s)
        results.append((kv, lv, actual_eps, eta_s, eta_a, err))
        print(f"kappa={kv}, lambda={lv}, eps={actual_eps:.2e}: "
              f"eta_sim={eta_s:.6f}, eta_an={eta_a:.6f}, err={err:.2e}")

    # Plot convergence
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 13,
                         'legend.fontsize': 10, 'figure.dpi': 200,
                         'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(figsize=(7, 5))
    for kv, lv in param_sets:
        subset = [(r[2], r[5]) for r in results
                  if r[0] == kv and r[1] == lv and r[5] > 1e-15]
        if subset:
            eps_arr, err_arr = zip(*subset)
            ax.loglog(eps_arr, err_arr, 'o-',
                      label=rf'$\kappa={kv},\;\lambda={lv}$', markersize=5)

    eps_ref = np.logspace(-4, -0.8, 50)
    ax.loglog(eps_ref, 0.5 * eps_ref, 'k--', linewidth=0.8,
              label=r'$\propto \varepsilon$')
    ax.set_xlabel(r'$\varepsilon = \Delta v / v_p$')
    ax.set_ylabel(r'$|\eta_{\rm analytical} - \eta_{\rm sim}|$')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    fig.savefig('fig_verification.pdf')
    fig.savefig('fig_verification.png')
    plt.close(fig)
    print("\nVerification plot saved to fig_verification.pdf/png")
