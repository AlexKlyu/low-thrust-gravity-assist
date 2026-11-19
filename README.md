# Efficiency of Low-Thrust Gravity Assist

Code accompanying the Technical Note *"Efficiency of Low-Thrust Gravity Assist"*
by A. Kliuchnikov (manuscript under review in the *Journal of Guidance, Control, and Dynamics*;
preprint: [doi:10.13140/RG.2.2.21880.00007](https://doi.org/10.13140/RG.2.2.21880.00007)).

The efficiency of a low-thrust gravity assist is defined as the ratio of the orbital energy gain under
continuous tangential thrust to that of an impulsive maneuver at periapsis with the same total velocity
increment. To first order in $\varepsilon = \Delta v / v_p$ it equals

$$\eta(\kappa, \lambda) = \frac{\langle v \rangle}{v_p}
= \frac{\displaystyle\int_{1/\lambda}^{1} \frac{\sqrt{\kappa + (1-\kappa)u}}{u^2\sqrt{(1-u)(u+\kappa)}}\,du}
       {\displaystyle\int_{1/\lambda}^{1} \frac{du}{u^2\sqrt{(1-u)(u+\kappa)}}},$$

where $\kappa = v_\infty^2 / v_p^2$, $\lambda = R_t / r_p$, and $u = r_p / r$. The result is given in
quadratures: the numerator is an elliptic integral, and the denominator (proportional to the time of
flight through the thrust zone) is elementary; both are evaluated by Gauss–Legendre quadrature after the
substitution $u = 1 - (1 - 1/\lambda)\,s^2$, which removes the integrable singularity at periapsis.

A compact closed-form approximation (maximum relative error 1.9% for $\kappa \in [0.01, 0.9]$,
$\lambda \in [1.5, 50]$) is

$$\eta \approx \sqrt{\kappa + \frac{1-\kappa}{\alpha\lambda + 1 - \alpha}}, \qquad \alpha = 0.40 - 0.21\sqrt{\kappa}.$$

## Files

| File | Description |
|------|-------------|
| `plots.py` | Fig. 1 (contour map of $\eta$), Fig. 2 (cross sections $\eta(\lambda)$, $\eta(\kappa)$), and the values of Table 3 |
| `verify.py` | RK4 verification against direct integration of the equations of motion: Fig. 3 and Table 4 |

## Quick start

```bash
pip install -r requirements.txt
python plots.py     # fig_contour.*, fig_eta_combined.*, prints Table 3   (~ a few minutes)
python verify.py    # table4_initial_conditions.txt, fig_verification.*  (~ 40 min, pure Python RK4)
```

## Verification setup

The verification uses dimensionless variables:
- $\mu = (1 - \kappa)/2$, $v_p = 1$, $r_p = 1$, $R_t = \lambda$;
- eccentricity $e = (1 + \kappa)/(1 - \kappa)$;
- the spacecraft starts at $r = R_t$ on the inbound branch with the exact Keplerian state, and the engine
  operates while $r \le R_t$;
- eight $(\kappa, \lambda)$ cases (Table 4) and $\varepsilon \in [10^{-4}, 5 \times 10^{-2}]$;
- fixed-step RK4 with $2 \times 10^5$ to $8 \times 10^5$ steps per passage through the thrust zone.

For $\varepsilon \gtrsim 10^{-3}$ the error of the first-order formula is about $0.5\,\varepsilon$.

## Author

Alexander V. Kliuchnikov, Moscow Institute of Physics and Technology (MIPT)
