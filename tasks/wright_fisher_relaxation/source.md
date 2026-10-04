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
