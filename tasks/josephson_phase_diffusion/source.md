# Source

- V. Ambegaokar and B. I. Halperin, "Voltage due to thermal noise in the dc Josephson
  effect", Physical Review Letters 22, 1364-1366 (1969). Exact mean voltage of the
  overdamped junction with thermal noise.
- Yu. M. Ivanchenko and L. A. Zil'berman, "The Josephson effect in small tunnel
  contacts", Soviet Physics JETP 28, 1272-1276 (1969). The same problem from the
  Fokker-Planck equation.
- R. L. Stratonovich, Topics in the Theory of Random Noise, Vol. II (Gordon and Breach,
  1967). Stationary current of Brownian motion in a tilted periodic potential.
- P. Reimann, C. Van den Broeck, H. Linke, P. Hanggi, J. M. Rubi and A. Perez-Madrid,
  "Giant acceleration of free diffusion by use of tilted periodic potentials", Physical
  Review Letters 87, 010602 (2001), and Physical Review E 65, 031104 (2002). Exact
  effective diffusion coefficient and the giant-diffusion peak near the critical tilt.
- B. Lindner, M. Kostur and L. Schimansky-Geier, "Optimal diffusive transport in a tilted
  periodic potential", Fluctuation and Noise Letters 1, R25-R39 (2001). Diffusion and
  first-passage moments in the tilted washboard.
- S. Lifson and J. L. Jackson, "On the self-diffusion of ions in a polyelectrolyte
  solution", Journal of Chemical Physics 36, 2410-2414 (1962). Diffusion in an untilted
  periodic potential.
- H. Risken, The Fokker-Planck Equation, 2nd ed. (Springer, 1989), chapters 10 and 11.
  Kramers equation, Brinkman hierarchy, Brownian motion in periodic potentials, matrix
  continued fractions.
- H. D. Vollmer and H. Risken, "Eigenvalues and their connection to transition rates for
  the Fokker-Planck equation with a periodic potential", Zeitschrift fur Physik B 52,
  259-266 (1983). Noisy underdamped junction (RCSJ) by matrix continued fractions.
- W. C. Stewart, Applied Physics Letters 12, 277 (1968), and D. E. McCumber, Journal of
  Applied Physics 39, 3113 (1968). The RCSJ model and the parameter beta_c.

The combination of steps (exact voltage and differential resistance, effective phase
diffusion, location of the giant-diffusion peak, inversion of the voltage for the noise
temperature and its thermometric sensitivity, criterion current and voltage noise of a series array with junction-
dependent noise strengths and time scales), the parameter ranges and the accuracy
requirements are our own design.

Consistency checks run while building the task:
- The reference (Stratonovich and Reimann double integrals: periodic trapezoid rule in
  the position, composite Gauss-Legendre in the shift graded toward the tilt boundary
  layer) and the high-precision Fourier continued fractions agree to about 1e-13
  (relative) on 40 random points over |i| <= 10, 0.02 <= theta <= 50, including
  exponentially small voltages (down to about 1e-17) and diffusion coefficients.
- The second solution is independent of the reference: it solves the stationary
  Fokker-Planck equation and the homogenization cell problem by Galerkin (Fourier)
  truncation with tridiagonal eliminations in extended precision, gets dv/di from the
  linearized system, evaluates D_eff = theta int (1 + chi')**2 p from the Fourier
  coefficients, finds the diffusion peak by a double-precision scan followed by secant
  iterations on high-precision differences, the noise temperature by Illinois regula falsi in 1 / theta with the thermometric
  sensitivity from the Galerkin system differentiated in theta, and the
  criterion current by safeguarded Newton iterations on ln V.

## How the test targets are obtained

No target is copied from the reference solution. Every test computes its own target:

| Target | Method in the test | Note / limit of validity |
|---|---|---|
| v and dv/di (step 1) | decaying solution of the Fourier recurrence of the stationary Fokker-Planck equation as a continued fraction in extended precision (30 + 1.8 / theta digits), v = i + Im r_1, dv/di from the differentiated recurrence | about 1e-15 relative, also for exponentially small v |
| v at low noise (steps 1, 4, 5, general) | single-integral form with the inner integral done exactly, v = theta (1 - exp(-2 pi i / theta)) / int_0^{2 pi} I_0(2 sin(y/2) / theta) exp(-i y / theta) dy, mpmath quadrature at 40 digits | agrees with the continued fraction to 1e-12 |
| dv/di at i = 0 (step 1) | linear response, 1 / I_0(1/theta)**2 | exact |
| D_eff (steps 2, 3, 5, general) | homogenization: D = theta int (1 + chi')**2 p dx with the corrector chi and the density p both from continued fractions in extended precision; Fourier convolution for the average | about 1e-15 relative |
| D_eff at i = 0 (step 2) | Lifson-Jackson, theta / I_0(1/theta)**2 | exact |
| diffusion peak (step 3) | one Newton step on dD/di = 0 with central differences (h = 2e-5) of the continued-fraction D must move i_peak by less than 1e-6; D lower at i_peak +- 0.05 | truncation of the differences below 1e-8 |
| noise temperature (step 4) | the voltage is generated at a known theta (including the endpoints 0.02 and 50) by the continued fraction (or the Bessel integral); the returned theta must recover it | relative 1e-7 |
| thermometric sensitivity kappa (step 4) | central difference of ln v in ln theta with step 1e-15 theta, the continued fraction evaluated at 60 + 1.8 / theta digits | about 1e-15 relative |
| array criterion (step 5, general) | one Newton step on V(J) = v_crit with V and dV/dJ from the continued fractions must move j_star by less than 1e-9 relative; r_diff and s_v compared with the continued-fraction sums; for one junction or identical junctions j_star is known from the Bessel-integral voltage | relative 1e-9 / 1e-8 |

## Version 3 changes

Editorial clean-up only: wording of the source notes and of the problem description.
No change to any prompt requirement, function header, solution, test or mutant.

## Version 4 changes

Step 4: the endpoint tolerance is now part of the stated contract (a voltage at most a
relative 1e-11 beyond the voltage at theta = 0.02 or 50 returns that endpoint; farther out
raises ValueError) and is tested on both sides (1e-12 accepted, 1e-9 rejected). Step 5: the
validation tests cover 2-D arrays, empty arrays, a NaN entry, a zero resistance, theta0 / c_k
above 50 and v_crit = 0, and three more computed-value cases were added.

## Version 5 changes (difficulty increase)

New final step 6: exact stationary voltage and differential resistance of the noisy
junction with capacitance (RCSJ), |i| <= 2.5, 0.1 <= theta <= 2, beta_c = 0 or 0.1 to 2.
The reference solves the Kramers equation in Hermite functions (velocity) and Fourier
modes (phase) by a matrix continued fraction over the Hermite index (N = 110, |p| <= 56),
with dv/di carried analytically through the same recursion; it is converged to 2e-9
relative against N = 260, |p| <= 140 on 29 points of the range (random and corners). The test oracle expands in Hermite functions centred at a different velocity
v0 = 0.6 i (a different truncated system) with N = 220, |p| <= 110; the two agree to 6e-10
relative over a 112-point grid of the range. The second solution centres the basis at the
overdamped mean velocity. A sparse direct solve of the same hierarchy and a continued
fraction over the Fourier index were tried and rejected: both lose accuracy in the running
state at weak noise and strong inertia. Step 1 also tests the endpoint i = -10.

## Version 6 changes

Step 4: the endpoint tolerance is applied exactly as stated, as a relative difference of the
voltage to the endpoint voltage (no longer through ln v), and is tested just around the
cutoff at both endpoints (0.9e-11 accepted, 1.1e-11 rejected). Step 6: non-finite input,
i = -2.6, theta = 2.01 and beta_c = 2.01 are tested as ValueError. Step 2: i = -10.01 too.

## Version 7 changes

Step 6 runtime cut from about 150 s to under 15 s: the reference truncation is N = 110,
|p| <= 56 (converged to 2e-9 relative, tolerance 1e-7) and dv/di is computed analytically
by differentiating the continued-fraction recursion (agrees with finite differences to
their own accuracy), so one call takes about 0.25 s instead of 7 s. The test oracle uses
N = 150, |p| <= 72 in the shifted basis (unchanged to 5e-9 against N = 220, |p| <= 110) and
a central difference for dv/di.
