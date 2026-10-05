# Sources, method and ground truth

## Physics and formulas
- **Prism potential, attraction and gradients.** Nagy (1966) and Nagy, Papp and Benedek (2000,
  J. Geodesy 74, 552), as corner sums with the limits at faces, edges and corners.
- **Columns with an exponentially decaying density contrast.** Cordell (1973); Chakravarthi and
  Sundararajan (2007, Geophysics 72, I23).
- **Basin inversion.** Bott (1960); Cordell and Henderson (1968).
- **Constant.** G = 6.6743e-11 (CODATA 2018).

## Reference solution
- **Prism potential, attraction and gradients, near the prism (within two diagonals of its centre).**
  The corner sums, with these details:
  - ln(a + r) for a < 0 is taken as ln((b^2 + c^2)/(r - a)), with no cancellation.
  - On the line of an edge (b^2 + c^2 = 0) the ln(b^2 + c^2) part is dropped. It cancels exactly between
    the two corners of that edge.
  - 0 ln 0 is taken as 0, and atan(n/0) as sign(n) pi/2. These are the limits used where the sums are
    continuous.
- **Prism, beyond two diagonals.** A 16^3-point Gauss-Legendre volume rule of 1/r and its derivatives.
  It is exact to rounding that far. The corner sums lose about (distance/size)^2 of their digits:
  1e-8 relative at 100 diagonals, 4e-5 at 1000, and nothing correct at 1e4.
- **Columns.** g_z = G drho0 int_0^h e^{-lam z} L(z) dz, with L the corner sum of atan(x y/(z r)).
  It is evaluated with geometric panels from the top (ratio 2, down to 1e-18 h), 20-point
  Gauss-Legendre each. This resolves the lamina term near the top for a station at any distance from an
  edge. A single 64-point rule gives 4e-5 relative error 1 mm from an edge, and 6e-6 even with 1024 nodes.
- **Basin depths.** Newton's method from the infinite-slab thicknesses, with the analytic Jacobian
  dg_i/dh_j = G drho0 e^{-lam h_j} L_ij(h_j) and step halving when the residual grows. Convergence is
  to 1e-13. The depth panels are refined at the top on a quarter of the smallest cell width (the
  stations are at the cell centres).

## Second solution
- The corner sums evaluated point by point, with math.fsum of all the terms.
- A 24^3-point rule beyond three diagonals.
- The columns by adaptive QUADPACK with geometric breakpoints.
- The basin by scipy.optimize.root (hybrid Powell) with the analytic Jacobian, polished by Newton steps.

## Targets
- **Prism (steps 1 and 2).** The closed forms evaluated with mpmath at 60 digits, which have no
  cancellation at any distance, with the same limit conventions. The far-field points (2.5 to 1e6
  diagonals) agree with 48^3-point Gauss-Legendre volume integrals in double precision to 1e-15.
- **Columns (step 3).** mpmath tanh-sinh quadrature, at 30 digits, of the lamina term over depth, with
  breakpoints at h 2^-k. For lam = 0, the 60-digit closed form of the prism.
- **Basin (step 4).** A 5 x 4 basin with known depths. Its anomaly is computed with 25-digit mpmath
  quadrature of all 400 station-column pairs.

The reference agrees with all targets within the stated tolerances: U to 1e-12, g to 0.09 of its
tolerance, T to 1e-4 of its tolerance, columns to 1.5e-13, depths to 7e-14.
