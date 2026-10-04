# Source

- H. Bateman, "Solution of a system of differential equations occurring in the theory of radioactive
  transformations", Proceedings of the Cambridge Philosophical Society 15, 423-427 (1910). The chain
  solution.
- J. Cetnar, "General solution of Bateman equations for nuclear transmutations", Annals of Nuclear
  Energy 33, 640-645 (2006). Branching networks, production and the numerical problems of the
  Bateman form for close decay constants.
- A. C. McCurdy, K. C. Ng and B. N. Parlett, "Accurate computation of divided differences of the
  exponential function", Mathematics of Computation 43, 501-528 (1984), and A. H. Al-Mohy and
  N. J. Higham, "A new scaling and squaring algorithm for the matrix exponential", SIAM Journal on
  Matrix Analysis and Applications 31, 970-989 (2009). Scaling and squaring, and recomputing the
  diagonal of triangular exponentials exactly.
- M. Pusa and J. Leppanen, "Computing the matrix exponential in burnup calculations", Nuclear Science
  and Engineering 164, 140-150 (2010). Stiffness of depletion matrices and the failure of standard
  methods for short-lived nuclides.
- Half-lives and cross sections: NUBASE2020 (F. G. Kondev et al., Chinese Physics C 45, 030001, 2021)
  and the Atlas of Neutron Resonances (S. F. Mughabghab, 6th ed., Elsevier, 2018): 197Au 98.65 b,
  198Au 2.6941 d and about 25100 b, 199Au 3.139 d, 59Co 20.7 b to 60mCo, 60Co 5.2714 y and 2 b, and
  the 238U series. The 199Au cross section is rounded to 30 b, and the 214Bi -> 210Tl branch to
  2e-4.

The combination of steps, the subtraction-free scaling and squaring, the parameter ranges and the
accuracy requirements are our own design.

Checks used while building the task:
- Two independent oracles agree to all printed digits on the full 238U series (15 members, decay
  constants over 21 decades) at 1 y, 1e4 y and 1e9 y: the Bateman sum in 600-digit arithmetic and
  mpmath's matrix exponential at 300 digits.
- The reference reproduces them to about 1e-14 relative on random chains and networks, including:
  - nearly equal decay constants (relative differences down to 1e-15);
  - entries down to 1e-250;
  - production terms and nuclides given in a non-topological order.
- Closed forms are reproduced:
  - single decay;
  - Erlang amounts for equal decay constants;
  - saturation with production, including lam t = 1e-10;
  - two-way branching;
  - target burnup exp(-sigma phi t) down to exp(-200).
- The second solution (mpmath expm in adaptively doubled precision, bisection plus secant for the
  flux) agrees with the reference to about 1e-14.
- Standard methods fail the tests:
  - scipy.linalg.expm and the Bateman sum in double precision both fail step 1;
  - squaring the diagonal in scaling and squaring (instead of resetting it) loses the amounts on the
    238U series completely.
- The gold-monitor response: d ln A / d ln phi for 198Au after 5 d of irradiation and 1 d of cooling
  is 0.96 at 1e13, 0.65 at 1e14, and 0 near 1.6e15 (burnup maximum); 199Au rises with slope at least
  1.1 on [1e-2, 1e15].

Version 2 (content-check fixes):
- Accuracy rule: the relative bound applies to exact entries that are positive and at least
  1e-250 * S; every other entry, exact zeros included, must lie within max(1e-250 * S, 1e-300). This
  covers S = 0 (an empty inventory), which is now tested in steps 1-3.
- Branching column sums: one rule everywhere, at most 1 + 1e-12 (1 up to rounding). Step 2 tests a
  sum of 1 + 5e-13 (accepted) and 1 + 1e-9 (rejected). Step 3 checks the branching matrix itself
  before the history, so an empty history no longer skips the check.
- New endpoint tests:
  - step 1: t = 0, lam = 1e10, n = 30 (equal and distinct constants), 2-D and empty arrays;
  - step 2: t = 0, t = 1e20, lam = 1e10, n = 30, n = 31;
  - step 3: sigma = 1e7 and above, capture_to = -2 and n, flux = 1e18 and above, duration = 1e12
    and above, an empty history;
  - step 4: t_cool = 0, t_irr = t_cool = 1e9, flux_hi = 1e18, the times above 1e9, k = -1 and n,
    a NaN activity.
