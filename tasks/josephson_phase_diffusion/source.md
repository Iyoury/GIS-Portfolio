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
- H. Risken, The Fokker-Planck Equation, 2nd ed. (Springer, 1989), chapter 11. Brownian
  motion in periodic potentials, matrix continued fractions.

The combination of steps (exact voltage and differential resistance, effective phase
diffusion, location of the giant-diffusion peak, inversion of the voltage for the noise
temperature, criterion current and voltage noise of a series array with junction-
dependent noise strengths and time scales), the parameter ranges and the accuracy
requirements are our own design.

Checks used while building the task:
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
  iterations on high-precision differences, the noise temperature by Illinois regula falsi in 1 / theta, and the
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
| noise temperature (step 4) | the voltage is generated at a known theta by the continued fraction (or the Bessel integral); the returned theta must recover it | relative 1e-7 |
| array criterion (step 5, general) | one Newton step on V(J) = v_crit with V and dV/dJ from the continued fractions must move j_star by less than 1e-9 relative; r_diff and s_v compared with the continued-fraction sums; for one junction or identical junctions j_star is known from the Bessel-integral voltage | relative 1e-9 / 1e-8 |
