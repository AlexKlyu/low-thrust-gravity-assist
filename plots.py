"""
Generate the figures and Table 3 of the JGCD Technical Note
"Efficiency of Low-Thrust Gravity Assist":
  Fig. 1  Contour map eta(kappa, lambda)
  Fig. 2  a) eta(lambda) at fixed kappa (dashed: asymptotes sqrt(kappa)),
          b) eta(kappa) at fixed lambda
  Table 3 Efficiency for typical flybys with realistic periapsis distances

Author: Alexander V. Kliuchnikov (MIPT)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def eta_analytical(kappa, lam, n_quad=500):
    """Efficiency eta(kappa, lambda), Eq. (15), via Gauss-Legendre quadrature
    after the substitution u = 1 - (1 - u_min) s^2, Eqs. (16)-(17)."""
    u_min = 1.0 / lam
    s_nodes, s_weights = np.polynomial.legendre.leggauss(n_quad)
    s = 0.5 * (s_nodes + 1.0)
    ws = 0.5 * s_weights
    u = 1.0 - (1.0 - u_min) * s**2
    v_over_vp = np.sqrt(kappa + (1.0 - kappa) * u)
    denom = u**2 * np.sqrt(u + kappa)
    return np.sum(ws * v_over_vp / denom) / np.sum(ws / denom)


def eta_approx(kappa, lam):
    """Compact approximation, Eqs. (18)-(19); max relative error 1.9% for
    kappa in [0.01, 0.9], lambda in [1.5, 50]."""
    alpha = 0.40 - 0.21 * np.sqrt(kappa)
    return np.sqrt(kappa + (1.0 - kappa) / (alpha * lam + 1.0 - alpha))


plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
})

# === Fig. 1: contour map ===
print("Fig. 1: contour map...")
kappas = np.linspace(0.01, 0.95, 80)
lambdas = np.linspace(1.01, 30, 80)
ETA = np.array([[eta_analytical(k, l) for l in lambdas] for k in kappas])

fig, ax = plt.subplots(figsize=(8, 6))
levels_fill = np.linspace(0.05, 1.0, 30)
levels_line = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
cs = ax.contourf(kappas, lambdas, ETA.T, levels=levels_fill, cmap='viridis')
cl = ax.contour(kappas, lambdas, ETA.T, levels=levels_line,
                colors='white', linewidths=0.8)
ax.clabel(cl, inline=True, fontsize=9, fmt='%.2f')
fig.colorbar(cs, ax=ax, label=r'$\eta$')
ax.set_xlabel(r'$\kappa = v_\infty^2 / v_p^2$')
ax.set_ylabel(r'$\lambda = R_t / r_p$')
fig.savefig('fig_contour.pdf')
fig.savefig('fig_contour.png')
plt.close(fig)

# === Fig. 2: cross sections ===
print("Fig. 2: eta(lambda) and eta(kappa)...")
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams.update({'legend.fontsize': 9.5})
fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

lambdas_fine = np.linspace(1.01, 50, 300)
for c, kv in zip(colors, [0.01, 0.05, 0.1, 0.3, 0.5, 0.8]):
    axa.plot(lambdas_fine, [eta_analytical(kv, l) for l in lambdas_fine],
             color=c, linewidth=1.8, label=rf'$\kappa = {kv}$')
    axa.axhline(np.sqrt(kv), color=c, linestyle='--', linewidth=0.8, alpha=0.7)
axa.set_xlabel(r'$\lambda = R_t / r_p$')
axa.set_ylabel(r'$\eta = \langle v \rangle / v_p$')
axa.set_xlim(1, 50)
axa.set_ylim(0, 1.05)
axa.grid(True, alpha=0.3)
axa.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
axa.text(0.03, 0.04, 'a)', transform=axa.transAxes, fontsize=13)

kappas_fine = np.linspace(0.001, 0.99, 300)
for c, lv in zip(colors, [2, 5, 10, 20, 50]):
    axb.plot(kappas_fine, [eta_analytical(k, lv) for k in kappas_fine],
             color=c, linewidth=1.8, label=rf'$\lambda = {lv}$')
axb.set_xlabel(r'$\kappa = v_\infty^2 / v_p^2$')
axb.set_xlim(0, 1)
axb.grid(True, alpha=0.3)
axb.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
axb.text(0.03, 0.04, 'b)', transform=axb.transAxes, fontsize=13)

fig.tight_layout(w_pad=1.5)
fig.savefig('fig_eta_combined.pdf')
fig.savefig('fig_eta_combined.png')
plt.close(fig)

# === Table 3: typical flybys with realistic periapsis distances ===
print("\nTable 3:")
# name, mu [km^3/s^2], planet radius [km], periapsis r_p [km], v_inf [km/s]
flybys = [
    ("Jupiter", 126686534.0, 71492.0, 5 * 71492.0,    6.0),   # 5 R_J
    ("Saturn",   37931187.0, 60268.0, 3 * 60268.0,    5.0),   # 3 R_S
    ("Earth",      398600.4,  6378.0, 6378.0 + 500.0, 3.0),   # 500 km altitude
    ("Venus",      324859.0,  6052.0, 6052.0 + 300.0, 3.5),   # 300 km altitude
    ("Mars",        42828.0,  3396.0, 3396.0 + 250.0, 2.5),   # 250 km altitude
]
print(f"  {'Planet':<8} {'r_p, km':>9} {'v_inf':>6} {'v_p':>6} {'kappa':>6}"
      f" {'eta(5)':>7} {'eta(10)':>8} {'eta(20)':>8}")
for name, mu, R, rp, vinf in flybys:
    vp = np.sqrt(vinf**2 + 2.0 * mu / rp)          # Eq. (9) at periapsis
    kv = vinf**2 / vp**2                            # Eq. (11)
    e5, e10, e20 = (eta_analytical(kv, l) for l in (5, 10, 20))
    print(f"  {name:<8} {rp:9.0f} {vinf:6.1f} {vp:6.2f} {kv:6.3f}"
          f" {e5:7.3f} {e10:8.3f} {e20:8.3f}")

print("\nAll figures generated.")
