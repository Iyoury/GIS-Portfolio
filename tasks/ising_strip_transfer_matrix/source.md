# Source

- L. Onsager, "Crystal statistics. I. A two-dimensional model with an order-disorder
  transition", Phys. Rev. 65, 117 (1944). Exact critical temperature
  Tc = 2 / ln(1 + sqrt(2)) for J = k_B = 1.
- B. Kaufman, "Crystal statistics. II. Partition function evaluated by spinor
  analysis", Phys. Rev. 76, 1232 (1949). Exact eigenvalues of the transfer matrix of a
  periodic strip; used for the step 2 targets and in the second solution.
- C. N. Yang, "The spontaneous magnetization of a two-dimensional Ising model", Phys.
  Rev. 85, 808 (1952). Limit of m_L below Tc.
- T. D. Lee and C. N. Yang, "Statistical theory of equations of state and phase
  transitions. II. Lattice gas and Ising model", Phys. Rev. 87, 410 (1952). Zeros of the
  partition function on the imaginary field axis.
- M. E. Fisher, "Yang-Lee edge singularity and phi^3 field theory", Phys. Rev. Lett. 40,
  1610 (1978).
- C. Itzykson, R. B. Pearson and J. B. Zuber, "Distribution of zeros in Ising and gauge
  models", Nucl. Phys. B 220, 415 (1983). Finite-size scaling of the zeros.
- M. P. Nightingale, "Scaling theory and finite systems", Physica A 83, 561 (1976).
  Finite-size scaling with strip transfer matrices.
- J. L. Cardy, "Conformal invariance and universality in finite-size scaling", J. Phys. A
  17, L385 (1984). xi_L = L / (2 pi x) on a periodic strip at criticality.
- H. W. J. Bloete, J. L. Cardy and M. P. Nightingale, Phys. Rev. Lett. 56, 742 (1986);
  I. Affleck, Phys. Rev. Lett. 56, 746 (1986). f_L = f_inf - pi c T / (6 L**2).
- M. Luscher, "Volume dependence of the energy spectrum in massive quantum field
  theories", Commun. Math. Phys. 104, 177 (1986). Exponential finite-size corrections with
  a power-law prefactor, which is why strips converge slowly near Tc in a weak field.
- T. Nishino and K. Okunishi, "Corner transfer matrix renormalization group method",
  J. Phys. Soc. Jpn. 65, 891 (1996). Used for the step 7 reference.
- V. Zauner-Stauber, L. Vanderstraeten, M. T. Fishman, F. Verstraete and J. Haegeman,
  "Variational optimization algorithms for uniform matrix product states", Phys. Rev. B 97,
  045145 (2018). Used for the step 7 second solution.

The row-labelling rule, the equal sharing of the in-row bonds and of the field between the
two factors, and the exact way the steps are combined are our own choices, made so that
every number in the tests is fixed.

## How the test targets are obtained

No target is copied from the reference solution. Every expected value is written inline in
the tests and was produced by the method below; the column "check" gives the agreement with
an independent route.

| Target | Method | Check |
|---|---|---|
| M entries (step 1) | brute-force sum over the energy, entry by entry, inside the test; spot entries worked out by hand | exact |
| f, xi_spin, xi_energy (step 2) | Kaufman's exact strip spectrum, now computed inside the test file in double precision (no pinned numbers); xi_energy = 1 / (2 g_1), the two lowest fermion modes q = 1 and 2L - 1; tolerances 1e-10 (f) and 1e-6 (lengths) | reference within 1.7e-8 (xi_spin at L = 10, T = 1, limited by the 1.4e-8 relative gap), elsewhere within 1e-14 |
| m_L (step 3) | reference (separate spin-flip sector blocks) | second solution (rotation of the leading pair of the whole matrix into parity states) within 3e-16; coefficient of the slowest term of G(r) at 40 digits for L <= 6 within 1e-15 |
| chi (step 4) | reference (spectral sum over the odd states) | second solution (odd-sector linear solve) within 2.4e-9; 40-digit finite differences of ln lam0(h) for L <= 6 within 1e-13 |
| H_edge (step 5) | reference (root of the discriminant of the leading pair) | second solution (bisection on the leading pair leaving the real axis) within 1.3e-12; 40-digit bisection for L <= 4 within 1e-15; the leading pair was checked real and distinct below the edge on a fine grid for every L from 3 to 8 and T from 1.5 to 10 |
| final dict (step 6) | the formulas of step 6 applied to the values above | second solution within 1e-12 |
| bulk f, m, chi, c (step 7, general) | corner-transfer-matrix renormalization at bond dimension 28 (the reference uses 16), free energy from kappa = Z(3x3) Z(4 corners) / Z(corners + 2 edges)**2, chi from Richardson central differences of m, c from Richardson second differences of f | VUMPS (second solution: per-site eigenvalue of the row transfer matrix, chi = -d2f/dh2 and c from five-point stencils) within 4e-14 (f), 5e-11 (m), 6e-7 relative (chi) and 3e-8 relative (c) on all eight test points; at h = 0 the same free energy reproduces Onsager's to 1e-14 at T = 2, 2.5 and 3 |

## Why each step is not a direct transcription

Each step was run against the default that a practitioner would reach for first, on the
test points of that step. The tolerance in parentheses is the one stated in the prompt.

| Step | Default route | Error on the test points | Correct alternatives that pass |
|---|---|---|---|
| 2 (rel 1e-6) | xi_energy from the second eigenvalue of M | 1.1 to 6.8 (every case) | eigenvectors of M classified by spin-flip parity; sector blocks; free fermions |
| 2 (rel 1e-6) | xi_energy from the third eigenvalue of M | 0.22, 0.74, 0.83 at T = 3, 5, 10 (the third eigenvalue is odd above Tc) | as above |
| 3 (abs 1e-9) | matrix element between the two leading eigenvectors of one eigensolver call | 1.0 at (L, T) = (10, 0.5), 1.8e-3 at (10, 0.6), 3.3e-4 at (8, 0.5): the pair is degenerate to machine precision and the solver returns an arbitrary mixture | sector blocks; rotation of the pair into parity states |
| 3 (abs 1e-9) | Yang's bulk magnetization | 0.08 at (6, 2.2), 0.51 at (7, 4.0) | |
| 4 (rel 1e-6) | central second difference of f(h), step 1e-3 to 1e-7 | every step fails at least one case (relative error up to 0.99 below Tc, roundoff-dominated at Tc and above for the small steps); Richardson on 1e-4 and 5e-5 fails at 1.0 | spectral sum (with or without sector blocks); odd-sector linear solve |
| 4 (rel 1e-6) | correlation summed over r >= 0 only | 0.11 to 0.50 | |
| 5 (rel 1e-8) | first field at which any eigenvalue becomes complex | 0.32 at (6, Tc), 0.99 at (8, 3.0): degenerate lower pairs leave the real axis first | root of the discriminant of the leading pair; bisection on the leading pair |
| 5 (rel 1e-8) | linear scan of H from 1e-3 | factor 6.6 at (7, 1.5): the edge is 1.3e-4 below Tc | geometric bracketing |
| 5 (rel 1e-8) | Hermitian eigensolver on the complex symmetric matrix | factor 1.6 to 8.5e3 | general (non-Hermitian) eigensolver |

The strip routes pass only at the two test points far from the critical region
((T, h) = (2.0, 0.05) and (3.0, 0.2)), where the correlation length is about one lattice
spacing.

| 7 (f 1e-10, m 1e-9, chi and c rel 1e-5) | widest diagonalizable strip, or extrapolation of strips | m alone was already off by 1e-4 to 1e-2 at h = 0.02 near Tc; the new range goes down to h = 0.005 (longer correlation length) | corner-transfer-matrix renormalization or VUMPS with the correct per-site partition function and converged derivatives |
| 7 | partition function per site without dividing out the corner contribution | f off by about 0.3 | |
| 7 | specific heat as -d2f/dT2 (no factor T) | factor T | |
| 7 | susceptibility of free spins (1 - m**2) / T | factor 3 to 60 | |
| 7 | central differences with steps that are too large at h = 0.005 | chi off by 2e-5 relative and more (truncation error of a second difference with step 5e-4) | Richardson extrapolation or smaller steps on a well-converged f |

Every test case takes under 15 s with the reference solution (numpy 2.4.6, scipy 1.17.1), except the step 7 cases, which take up to about 15 s each (five warm-started CTMRG runs).

Version 5: the free-energy tolerance of step 2 is 1e-10 instead of 1e-12 (a 1e-12 statement sat at the rounding level of the eigenvalue solver), and the step 2 targets are evaluated inside the test from Kaufman's formula.

Version 6: endpoint cases added to the tests — strip_magnetization(3, 10), strip_susceptibility(3, 2.5) and (3, 10), lee_yang_edge(3, 1.5) and (3, 10), critical_exponents(3) and (7); every value cross-checked by the two solutions (agreement 1e-12 or better). The Kaufman helper of the step 2 tests is written inside every test case (no preamble).

Version 7 (difficulty increase): the last step now returns the full thermodynamics of the
infinite lattice in a field, (f, m, chi, c), instead of m alone, over a wider and harder
range (h from 0.005 instead of 0.02, so the correlation length near Tc is about twice as
long). The free energy needs the correctly normalized per-site partition function of the
infinite-lattice method, and chi = dm/dh and c = -T d2f/dT2 need converged derivatives to
1e-5 relative. Targets from CTMRG at bond dimension 28, cross-checked with VUMPS.

Version 8: step 7 states a time budget of 60 s per call (the reference needs at most about
15 s), asserted on every call in the step 7 and whole-task tests; two rollout attempts had
timed out on step 7 without a stated budget. Step 6 states the scaling laws (Cardy's relation,
finite-size power laws between two widths, the conformal law of the free energy for three
widths) instead of the final estimator formulas; the expected values are unchanged.
