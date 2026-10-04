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

Version 3 (content-check fixes):
- Step 3 new tests:
  - lam = 1e10 under no flux, and lam above 1e10 rejected;
  - a combined rate of exactly 1e10 per s (lam = 1e10 - 10, capture rate 10 per s), with its
    closed form;
  - n = 30 with an empty history, and n = 31 rejected;
  - a NaN cross section and an infinite duration rejected.
- Step 4 now states the step-3 data contract (types, shapes, ranges, acyclicity) itself.
- Step 4 makes the interval explicitly closed: an activity within a relative 1e-9 of A(flux_lo) or
  A(flux_hi) gives that end. This is tested at both ends.

Version 4 (content-check fixes):
- Step 4: the end rule is two-sided and checked before the bracket. If
  |ln(activity / A(flux_lo))| <= 1e-9 the result is flux_lo, else if the same holds at flux_hi it is
  flux_hi; on either side of the end value. Tested at both ends with activities equal to the end
  values and 5e-10 inside them.
- Step 2 tests a branching column summing exactly to the float 1 + 1e-12 (accepted).
- Step 3:
  - accepts the same bound with an empty history;
  - rejects a nonzero diagonal with an empty history;
  - ignores capture links of nuclides without cross section (a zero-sigma capture loop is accepted);
  - turns a history that is not a sequence (None) into ValueError;
  - rescales the period fractions so that rounding cannot push a column at the bound above it.

Version 5 (content-check fixes):
- Step 4 tests both sides of each end: activities 5e-10 outside A(flux_lo) and A(flux_hi) also give
  the ends.
- Step 3 requires history to be a list or tuple (ordered); None, a set and a generator are rejected.
- Steps 1-3 test amounts of exp(-550), about 1e-239, near the bottom of the relative-accuracy range:
  - step 1: pure decay;
  - step 2: pure decay in a network;
  - step 3: capture burnup at sigma = 1e7 b and 1e18 n / (cm^2 s) for 5.5e6 s.

Version 6 (content-check fixes):
- Step 3 new tests:
  - a valid history given as a tuple;
  - a history entry with a non-numeric value rejected;
  - a capture loop with positive cross sections rejected even with an empty history;
  - sum(n0) = 1e100 accepted and above rejected.
- Domain bound on the amounts, so that every exact result is representable in double precision:
  - sum(n0) <= 1e100 in steps 1, 3 and 4;
  - sum(n0) + t * sum(source) <= 1e100 in step 2.
  Each bound is stated, enforced (ValueError), and tested at the bound and above it.

Version 7 (advisory fixes after the v6 rollouts):
- Steps 1-3 test a positive exact amount below the 1e-250 S threshold: exp(-600), about 3e-261, is
  checked with the absolute rule.
  - step 1: pure decay;
  - step 2: pure decay in a network;
  - step 3: burnup at sigma = 100 b, flux 1e18, 6e6 s.
- Step 2 rejects an empty network, a scalar source and a 2-D n0.
- Step 3 rejects a scalar sigma, a branching matrix that is not n x n, and an n0 of the wrong length.

Version 8 (content-check fixes):
- Steps 1-2: the number of halvings is 0 whenever t * max(lam) <= 0.5. The logarithm is no longer
  taken of a product that underflows to 0 (lam = 1e-300, t = 1e-300), and that case is now tested.
- Step 4: activities 2e-9 (in |ln|) outside A(flux_lo) and A(flux_hi) must raise ValueError,
  next to the accepted 5e-10 cases.

Version 9 (content-check fixes):
- Domain of the cross sections: sigma = 0 or 1e-6 <= sigma <= 1e7 b (steps 3-4). Every capture
  rate sigma * 1e-24 * phi stays far above the floating-point underflow, so the promised flux
  accuracy is achievable.
  - Tested in step 3 with 1e-6 b (accepted) and 1e-7 b (rejected).
  - Tested in step 4 with 1e-300 b (rejected) and a round trip on a 1e-6 b monitor.
- Step 4:
  - activity = 0 and flux_lo = flux_hi raise ValueError;
  - the end tolerance is tested at |ln| = 0.999e-9 on both sides of each end (the end is
    returned) and at 1.01e-9 outside (ValueError).

Version 10 (content-check fixes):
- Step 4 domain closed: A(flux_lo) >= 1e-250 * lam[k] * sum(n0), i.e. the measured nuclide holds at
  least 1e-250 of the initial atoms at flux_lo, the range where step 3 promises relative accuracy.
  Otherwise ValueError. This excludes the case where the per-atom product fraction underflows
  (t_irr = 1e-300 with n0 = 1e100), which is tested as a ValueError.
- Step 4 end rule: the tests use the activities closest to |ln| = 1e-9 from inside (1e-9 - 1e-14,
  a few rounding units of the logarithms; end returned, at both ends and on both sides) and
  1.01e-9 outside (ValueError).
- Step 4 new tests: a round trip at slope 0.52 (flux 1.547e14 on the gold monitor), and the
  precedence of flux_lo when both end activities are within the tolerance.
- Steps 1-3 test exp(-575), about 8e-250, just inside the relative-accuracy range.

## Version 11

- Step 4 end rule tested at the boundary itself: on both sides of both ends, the activity closest to
  |ln(activity / A(end))| = 1e-9 from inside (within 1e-14) returns the end, so a cutoff anywhere
  measurably below 1e-9 fails. The prompt no longer says which activities are used in the tests.
- Step 4 domain check done on the atoms in separate logarithms, ln x[k] >= ln 1e-250 + ln sum(n0),
  since 1e-250 * lam[k] * sum(n0) underflows to 0 for small lam[k] (then -inf >= -inf passed). The
  prompt states that the rule holds in that case; a new test with lam[k] = 1e-75 and a two-capture
  product holding 5e-267 of the atoms at flux_lo raises ValueError (the v10 reference returned flux_hi).
  ln A is formed as ln lam[k] + ln x[k] so that lam[k] * x[k] cannot underflow.
