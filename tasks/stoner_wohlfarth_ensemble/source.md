# Source

- E. C. Stoner and E. P. Wohlfarth, "A mechanism of magnetic hysteresis in
  heterogeneous alloys", Philosophical Transactions of the Royal Society of London A
  240, 599-642 (1948). The single-domain model, the switching field and the hysteresis
  loops for different field angles.
- L. Neel, "Theorie du trainage magnetique des ferromagnetiques en grains fins avec
  applications aux terres cuites", Annales de Geophysique 5, 99-136 (1949), and W. F.
  Brown, "Thermal fluctuations of a single-domain particle", Physical Review 130,
  1677-1686 (1963). Thermally activated escape over the anisotropy barrier with an
  attempt frequency (Neel-Brown model).
- M. P. Sharrock, "Time dependence of switching fields in magnetic recording media",
  Journal of Applied Physics 76, 6413-6418 (1994). Dependence of the switching field and
  of the coercivity on the sweep rate and the thermal stability ratio K V / (k_B T).
- W. T. Coffey and Yu. P. Kalmykov, The Langevin Equation, 4th ed. (World Scientific, 2017),
  chapter on the Brown model: Legendre-polynomial (matrix continued fraction) solution of the
  axially symmetric Fokker-Planck equation, smallest eigenvalue and integral relaxation time.
- D. A. Garanin, "Integral relaxation time of single-domain ferromagnetic particles", Physical
  Review E 54, 3250-3256 (1996). Exact integral formula for the integral relaxation time.
- C. Tannous and J. Gieraltowski, "The Stoner-Wohlfarth model of ferromagnetism",
  European Journal of Physics 29, 475-487 (2008). Teaching account of the model and the
  astroid.

The combination of steps (zero-temperature branch, barriers, survival probability under
a linear sweep, switching-field statistics, dynamic coercivity of a weighted ensemble),
the escape over both maxima without back jumps, and the parameter ranges are our own
design.

Checks used while building the task:
- The barriers were computed by three independent methods (quartic in a shifted
  tan(theta/2) solved by companion-matrix eigenvalues, bracketing of each extremum
  between known neighbours, and a grid search on the circle refined by bisection); they
  agree to better than 1e-15. At psi = 0 and psi = pi/2 they equal the closed forms
  (1 + h)**2 / 2 and (1 -+ |h|)**2 / 2.
- At psi = 0 the survival probability has the closed form
  P(h) = exp(-(f0 / rate) sqrt(pi / a) [erfc(sqrt(a) (1 + h)) - erfc(2 sqrt(a))]); the
  reference reproduces it to about 1e-15 and its median to about 1e-14.
- The second solution is independent of the reference: bracketing instead of the
  quartic for the barriers, an adaptive Gauss-Legendre rule instead of a library
  quadrature for the survival probability, one ODE run with dense output for the
  switching statistics instead of a windowed quadrature, and per-particle ODE runs with
  bisection for the ensemble instead of nested quadratures inside Brent's method. The
  two agree to about 1e-14.

## How the test targets are obtained

No target is copied from the reference solution. Every test computes its own target:

| Target | Method in the test | Note / limit of validity |
|---|---|---|
| zero-temperature branch (step 1) | all minima of e on a 20000-point circle grid refined by bisection; with two minima the descending branch is the one with cos(theta) > 0 | exact to rounding for points at least 1e-3 from the jump |
| barriers (step 2) | minima and maxima of e from the same grid search, refined by bisection; original minimum = the one with cos(theta) > 0 | exact to rounding; closed forms at h = 0, psi = 0 and psi = pi/2 |
| survival probability (step 3) | closed form with erfc at psi = 0; for other angles an adaptive quadrature (purely relative tolerance) of the escape rate built on the grid barriers | about 1e-14 |
| median switching field (steps 4, 5) | closed form with the inverse erfc at psi = 0; otherwise one Newton step on P = 1/2 from the returned value, with dP/dh = (f0 / rate) (Gamma / f0) P | must move it by less than 1e-9 |
| mean switching field (step 4) | E[H_s] = h_sw - int P dh, with P from the closed form (psi = 0) or from one ODE run for the escape integral on the grid barriers | about 1e-12 |
| h_c and h_half (step 5, general) | one Newton step on the expected ensemble magnetization (or on the switched weight minus 1/2), with analytic derivatives: dP/dh as above and dm/dh = sin(theta - psi)**2 / e''(theta) along each minimum | must move them by less than 1e-8; for one particle at psi = 0 both equal minus the closed-form median |
| sanity values | barriers 1/2 at h = 0; P = 1 above h_sw and 0 below -h_sw; P depends on f0 and rate only through f0 / rate | exact |

## Version 3 changes (difficulty increase)

New final step 6, brown_relaxation(sigma, h): the smallest nonzero eigenvalue lam1 and the
integral relaxation time tau_int of Brown's axially symmetric Fokker-Planck equation,
0 <= sigma <= 60, |h| <= 0.9, both to 1e-8 relative.
- Reference: Sturm-Liouville form with the Boltzmann weight w = exp(-beta E); inverse iteration
  in which every step is a cumulative integral of w g (flux) and of 2 F / ((1 - z^2) w),
  represented by piecewise Chebyshev interpolation in theta (panels keep their own relative
  accuracy, so wells whose weights differ by 1e-94 are both resolved). The flux is taken from
  the nearer end on each side of the barrier to avoid cancelling the two well masses, and the
  iteration starts from a step at the barrier so that a slow mode living in the shallow well is
  not lost below rounding. lam1 is the Rayleigh quotient of positive integrals; tau_int is
  Garanin's formula 2 int Q^2 / ((1 - z^2) w) dz / int (z - <z>)^2 w dz.
- Test oracle and second solution: Legendre expansion W = sum b_l P_l, pentadiagonal operator
  from the recurrences of z P_l and (1 - z^2) P_l', solved in extended precision (mpmath,
  30 + sigma (1 + |h|)^2 / 2 digits) by banded elimination: inverse iteration for lam1, a
  linear solve for tau_int.
- The two agree to 3e-15 (lam1) and 2e-13 (tau_int) on 48 points (random and corners);
  lam1 = 1, tau_int = 1 at sigma = 0; Brown's asymptote is approached at sigma = 60; a third
  check by symmetric finite differences reproduces lam1 = 5.150415 at sigma = 60, h = 0.9,
  where a smooth starting vector converges instead to an intrawell mode of the deep well.

Version 4: the sigma = 0 checks of step 6 use 1e-9 instead of 1e-12 (inside the stated 1e-8; a 1e-12 bound left no room for a result perturbed at the 1e-12 level).

Version 5 (content-check fixes):
- Steps 3-5: the domain 40 <= a <= 1000, 1e5 <= f0 / rate <= 1e13 is now enforced (ValueError
  outside it) and stated in the prompts and docstrings. In it f0 / rate * int Gamma / f0 over
  (-h_sw, h_sw) is at least about 430 for every psi (checked on a psi grid at a = 1000,
  f0 / rate = 1e5), so the survival probability always crosses 1/2 inside (-h_sw, h_sw) and the
  median is a proper root; the tests cover all four corners of the domain.
- psi = pi/2: h_sw = 1 is taken exactly (cos(pi/2) = 6e-17 in floating point would put it 2e-11
  below 1, which matters for f0 / rate up to 1e13); quadrature nodes that round onto +-h_sw are
  moved one ulp inside. New closed-form tests at psi = pi/2 (barriers (1 -+ h)^2 / 2) in steps 3
  and 4, and an ensemble with a particle at psi = pi/2 and one with a zero weight in step 5.
- Step 2 returns numpy float scalars for a scalar h (tested with isinstance(..., np.floating));
  the near-edge fields are 1.1e-3 inside the range, where the 1e-10 accuracy is promised.
- Step 1 is tested exactly at |h + h_sw| = 1e-3 (psi = 0, h = -0.999 and -1.001).
- Step 3 is tested at h = +-h_sw exactly (psi = 0 and pi/2).
- Step 5: fewer invalid-input entries, three new computed cases.
- Step 6: the whole computation (panel construction, inverse iteration, Rayleigh quotient,
  Garanin's tau_int) is in the body of brown_relaxation; h = -0.9 is tested against the oracle
  and against h = +0.9; the docstring gives the units as 1/tau_N and tau_N, respectively.
- Step 5 (v5 follow-up): the ValueError test now covers every invalid input named in the prompt
  (equal-shaped 2-D arrays, empty arrays, infinite and NaN weights, rate = 0, negative f0, NaN a,
  infinite rate) next to the domain bounds.

Version 7: step 6 states a time budget (each call within 30 s on one CPU core; the reference takes
about 1.2 s at sigma = 60, |h| = 0.9 and the second solution about the same), and the step-6 and
general tests assert it on every checked call.

Version 8: v7 reached 1/8 with 3 of 8 attempts cut by the time limit. Failures were spread over
all six steps and were mostly numerical-method misses while chasing 1e-10. The required
accuracies are relaxed by a factor of 100, everywhere at once (prompts, docstrings and tests); the
science and the test cases are unchanged:
- steps 1-3: 1e-10 to 1e-8 absolute;
- step 4: 1e-9 to 1e-7;
- step 5: 1e-8 to 1e-6;
- step 6: 1e-8 to 1e-6 relative.
Every mutant still fails by many orders of magnitude more than the new bounds.
