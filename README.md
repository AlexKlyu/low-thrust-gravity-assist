# Efficiency of Low-Thrust Gravity Assist

Analytical formula for the efficiency of a low-thrust gravity assist maneuver:

$$\eta(\kappa, \lambda) = \frac{\langle v \rangle}{v_p}$$

where $\kappa = v_\infty^2 / v_p^2$ and $\lambda = R_t / r_p$.

The formula gives the ratio of energy gain from continuous low-thrust propulsion during a planetary flyby to that of an equivalent impulsive maneuver at periapsis (Oberth effect). The result is expressed as a ratio of definite integrals (closed-form in quadratures) and is valid to first order in $\varepsilon = \Delta v / v_p$.

## Files

| File | Description |
|------|-------------|
| `article.tex` | Article source (Russian) |
| `verify.py` | RK4 numerical verification of the analytical formula |
| `plots.py` | Generate all figures (contour map, parametric curves, mission table) |

## Quick start

```bash
pip install numpy scipy matplotlib
python verify.py    # runs verification, produces fig_verification.pdf
python plots.py     # generates all article figures
```

## Normalization

The verification code uses dimensionless variables:
- $\mu = (1 - \kappa)/2$, $v_p = 1$, $r_p = 1$, $R_t = \lambda$
- Eccentricity: $e = (1 + \kappa)/(1 - \kappa)$
- Initial position: $r = R_t$ on the incoming hyperbolic branch

## Author

Alexander Klyuchnikov, Moscow Institute of Physics and Technology (MIPT)
