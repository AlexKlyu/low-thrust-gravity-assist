"""
Generate all plots for the article:
  1. Contour map eta(kappa, lambda)
  2. eta(lambda) curves for fixed kappa
  3. eta(kappa) curves for fixed lambda
  4. Mission examples table

Author: Alexander Klyuchnikov (MIPT)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def eta_analytical(kappa, lam, n_quad=500):
    """Compute analytical eta via Gauss-Legendre quadrature."""
    u_min = 1.0 / lam
    s_nodes, s_weights = np.polynomial.legendre.leggauss(n_quad)
    s = 0.5 * (s_nodes + 1.0)
    ws = 0.5 * s_weights
    u = 1.0 - (1.0 - u_min) * s**2
    v_over_vp = np.sqrt(kappa + (1.0 - kappa) * u)
    denom = u**2 * np.sqrt(u + kappa)
    I_num = np.sum(ws * v_over_vp / denom)
    I_den = np.sum(ws * 1.0 / denom)
    return I_num / I_den


def eta_grid(kappas, lambdas):
    """Compute eta on a 2D grid."""
    result = np.zeros((len(kappas), len(lambdas)))
    for i, k in enumerate(kappas):
        for j, l in enumerate(lambdas):
            result[i, j] = eta_analytical(k, l)
    return result


plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
})

# === Plot 1: Contour map ===
print("Plot 1: contour map...")
kappas = np.linspace(0.01, 0.95, 80)
lambdas = np.linspace(1.01, 30, 80)
ETA = eta_grid(kappas, lambdas)

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
print("  done")

# === Plot 2: eta(lambda) curves ===
print("Plot 2: eta(lambda)...")
lambdas_fine = np.linspace(1.01, 50, 300)
kappa_vals = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8]

fig, ax = plt.subplots(figsize=(8, 5.5))
for kv in kappa_vals:
    etas = [eta_analytical(kv, l) for l in lambdas_fine]
    ax.plot(lambdas_fine, etas, label=rf'$\kappa = {kv}$', linewidth=1.8)
    ax.axhline(y=np.sqrt(kv), color='gray', linestyle=':', linewidth=0.5)
ax.set_xlabel(r'$\lambda = R_t / r_p$')
ax.set_ylabel(r'$\eta = \langle v \rangle / v_p$')
ax.legend(loc='upper right')
ax.set_ylim(0, 1.05)
ax.set_xlim(1, 50)
ax.grid(True, alpha=0.3)
fig.savefig('fig_eta_lambda.pdf')
fig.savefig('fig_eta_lambda.png')
plt.close(fig)
print("  done")

# === Plot 3: eta(kappa) curves ===
print("Plot 3: eta(kappa)...")
kappas_fine = np.linspace(0.001, 0.99, 300)
lambda_vals = [2, 5, 10, 20, 50]

fig, ax = plt.subplots(figsize=(8, 5.5))
for lv in lambda_vals:
    etas = [eta_analytical(k, lv) for k in kappas_fine]
    ax.plot(kappas_fine, etas, label=rf'$\lambda = {lv}$', linewidth=1.8)
ax.set_xlabel(r'$\kappa = v_\infty^2 / v_p^2$')
ax.set_ylabel(r'$\eta = \langle v \rangle / v_p$')
ax.legend(loc='lower right')
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 1)
ax.grid(True, alpha=0.3)
fig.savefig('fig_eta_kappa.pdf')
fig.savefig('fig_eta_kappa.png')
plt.close(fig)
print("  done")

# === Mission examples ===
print("\nMission examples:")
missions = [
    ("Jupiter",  6.0, 59.5, 1.05),
    ("Saturn",   5.0, 35.5, 1.1),
    ("Earth",    3.0, 11.2, 1.05),
    ("Venus",    3.5, 10.4, 1.05),
    ("Mars",     2.5,  5.0, 1.1),
]
for name, vinf, vesc, rp_R in missions:
    vp = np.sqrt(vinf**2 + vesc**2 / rp_R)
    kv = vinf**2 / vp**2
    e5 = eta_analytical(kv, 5)
    e10 = eta_analytical(kv, 10)
    e20 = eta_analytical(kv, 20)
    print(f"  {name:<8} kappa={kv:.4f}  "
          f"eta(5)={e5:.4f}  eta(10)={e10:.4f}  eta(20)={e20:.4f}")

print("\nAll plots generated.")
