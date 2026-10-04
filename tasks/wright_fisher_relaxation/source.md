# Source

- R. A. Fisher, The Genetical Theory of Natural Selection (Clarendon Press, 1930), and S. Wright,
  "Evolution in Mendelian populations", Genetics 16, 97-159 (1931). The Wright-Fisher model and
  the stationary distribution of the diffusion limit.
- M. Kimura, "On the probability of fixation of mutant genes in a population", Genetics 47,
  713-719 (1962), and M. Kimura and T. Ohta, "The average number of generations until fixation of a
  mutant gene in a finite population", Genetics 61, 763-771 (1969). Fixation probability and
  conditional fixation time.
- W. Feller, "Diffusion processes in genetics", Proc. Second Berkeley Symposium, 227-246 (1951),
  and C. Cannings, "The latent roots of certain Markov chains arising in genetics", Advances in
  Applied Probability 6, 260-290 (1974). Exact eigenvalues of the neutral chain.
- S. Karlin, Total Positivity, Vol. I (Stanford University Press, 1968). Real, positive, distinct
  eigenvalues of totally positive kernels.
- W. K. Grassmann, M. I. Taksar and D. P. Heyman, "Regenerative analysis and steady state
  distributions for Markov chains", Operations Research 33, 1107-1116 (1985), and C. A. O'Cinneide,
  "Entrywise perturbation theory and error analysis for Markov chains", Numerische Mathematik 65,
  109-120 (1993). State reduction without subtraction, and its entrywise relative accuracy.
- W. J. Ewens, Mathematical Population Genetics I, 2nd ed. (Springer, 2004). Absorption
  probabilities and times of the Wright-Fisher chain, and the origin-fixation regime.

The combination of steps, the haploid life cycle (selection, then mutation, then sampling), the
parameter ranges and the accuracy requirements are our own design.

Checks used while building the task:
- Transition matrix: every entry of matrices up to N = 60 and selected rows at N = 300 and 400
  agree with mpmath binomials (50 digits) to about 1e-13 relative, including entries down to
  1e-250. Row sums, means N p_mut and variances N p_mut (1 - p_mut) are exact. For N = 1 the matrix
  is [[1 - v, v], [u, 1 - u]].
- Reference: subtraction-free elimination (state reduction) for the absorption probabilities and
  times, the stationary distribution, the relaxation rate (inverse iteration on a GTH-factored
  I - P, eliminating towards the most probable state) and the passage times. Second solution:
  the same linear systems solved by plain LU in mpmath, with the precision doubled until two runs
  agree to 1e-13. The two agree to about 1e-14 relative on all tested cases.
- Analytic checks:
  - neutral fixation probability i0 / N;
  - exact neutral first two stationary moments;
  - neutral relaxation rate u + v, down to 2e-12;
  - N = 1 closed forms;
  - exact allele-relabeling symmetry s -> -s / (1 + s), u <-> v, i -> N - i, which tests the
    results at N = 400 where they span more than 100 decades.
- Rare-mutation consistency:
  - the relaxation rate equals 1 / t_up + 1 / t_down to 2e-10;
  - t_up equals 1 / (N v p_fix(1)) to 2e-10;
  - pi[N] / pi[0] equals t_down / t_up to 4e-7.
- Dense linear algebra fails these tests:
  - np.linalg.solve for the passage times is off by 2e-5;
  - eig for the gap is off by 5e-3 at u + v = 2e-12;
  - the dense eigenvector for pi gives relative errors up to 1e10 on the small entries.

Version 2 (content-check fixes):
- Step 2 tests the smallest population N = 2 (neutral: p_fix = p_loss = 1/2, both conditional times 2
  generations).
- Step 4 tests u = 0.1 (neutral gap 0.11 for u = 0.1, v = 0.01).
- Step 5 tests v = 0.1 (N = 1: t_up = 1 / v = 10, t_down = 1 / u = 20).
- Step 4 prompt: the lambda_2**t decay is stated for a generic initial distribution (nonzero
  component along the second eigenvector).

Version 3 (difficulty increase, the quick probe solved v2 three times out of three):
- New step 6: the quasi-stationary distribution and absorption rate with one-way mutation (u = 0). The
  rate is down to about 1e-246 and is required to 1e-8 relative, with the distribution entrywise.
  - Reference: inverse iteration for the left Perron vector with a subtraction-free LU of I - Q. Each
    pivot is the sum of the remaining off-diagonal rates plus the accumulated absorption probability,
    and the iteration stops when every entry has settled, not only the rate.
  - Second solution: plain LU in mpmath with doubled precision.
  - Tests:
    - the neutral rate is exactly v, for any N (Feller-Cannings spectrum);
    - extended-precision oracle;
    - exact identities at N = 400: rate = sum_i q_i P[i, N] and (q Q)_j = (1 - rate) q_j;
    - the general test links it to step 5: rate * t_up = 1 to 8e-11.
- Steps 2-6 state a time budget of 20 s per call at N up to 400, asserted in the tests. The
  reference needs well under 2 s; extended-precision linear algebra at N = 400 does not fit.
- The prompts and background no longer describe the numerical remedy: no state reduction, no
  remark on dense eigensolvers or on diagonal entries close to 1.

Version 4 (content-check fixes):
- Independent N = 400 targets, computed once by plain Gaussian elimination with partial pivoting in
  mpmath at 400 digits (transition matrix built from mpmath binomials). They are written into the
  tests and compared at 1e-8 relative; the reference agrees with them to about 1e-13:
  - step 2: all four outputs for s = -1/3, i0 = 1 (p_fix about 3e-141) and for s = 0.5, i0 = 399;
  - step 3: 18 entries of pi for s = 0.5, u = v = 1e-12, from the largest down to about 6e-150;
  - step 5: t_up and t_down (about 8.7e149) for s = 0.5, u = v = 1e-12;
  - step 6: the rate (about 2.3e-246) and 15 entries of the distribution for s = -0.5, v = 1e-9,
    including its tail between 1e-250 and 1e-240.
- Steps 1 and 3: entries whose exact value is below 1e-250 are compared with that exact value
  (|got - exact| <= 1e-250), as the prompts state.

Version 5: the v4 rollouts (5/8) lost credit only on one step-6 check, the eigenvector equation at the
far tail of the N = 400, s = 0.5, v = 0.1 distribution (qsd[0] about 4e-221). The requirement was
already covered by the general accuracy rule; the prompt now states it explicitly, with that case and
the size of its tail, and says that every tail entry is checked against (qsd Q)_j = rho qsd_j.
