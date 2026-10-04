# Background: mutation, selection and drift in the Wright-Fisher model

## The model

The Wright-Fisher model (Fisher 1930, Wright 1931) describes a population of constant size N
with non-overlapping generations. Here it is haploid with two alleles, A and a, and the state is
the number i of copies of A. In each generation, selection changes the frequency p = i / N to
p_sel = (1 + s) p / (1 + s p) (fitness 1 + s for A, 1 for a). Mutation then turns A into a with
probability u and a into A with probability v, giving p_mut = (1 - u) p_sel + v (1 - p_sel). The
next generation is a binomial sample of size N with success probability p_mut. Genetic drift is
this sampling noise. Its variance p (1 - p) / N per generation makes small populations lose their
variation quickly.

The result is a Markov chain on {0, ..., N} with a dense transition matrix. Without mutation the
fixed states 0 and N are absorbing. With mutation the chain is irreducible and converges to a
stationary distribution, the mutation-selection-drift balance.

## Fixation

Without mutation every allele is eventually fixed or lost. The fixation probability h_i solves
h = P h with h_0 = 0 and h_N = 1. The mean times conditional on fixation or loss come from the
chain conditioned on its fate (the Doob h-transform), t_i = (G h)_i / h_i, where G is the
fundamental matrix of the transient states. Kimura's diffusion result
(1 - exp(-2 N s p)) / (1 - exp(-2 N s)) shows how fast the fixation probability falls for
deleterious alleles: for N s = -200 it is far below 1e-100.

## Stationary distribution and relaxation

In the diffusion limit the stationary distribution is Wright's density, proportional to
p**(2 N v - 1) (1 - p)**(2 N u - 1) exp(2 N s p). When N u and N v are small it piles up on the
two fixed states. A population then spends almost all its time fixed for one allele and switches
only when a new mutation sweeps to fixation (the origin-fixation regime). The waiting time for
such a substitution is about 1 / (N v p_fix).

The eigenvalues of the transition matrix are real, positive and distinct when the mutation rates
are positive. This follows from the total positivity of the binomial kernel when p_mut increases
with i (Karlin). The second largest eigenvalue lambda_2 sets the slowest relaxation. For the
neutral model, Feller and Cannings showed that the eigenvalues are
(1 - u - v)**k prod_{m<k} (1 - m / N). The gap 1 - lambda_2 is then exactly u + v, the switching
rate between the two fixed states, whatever N is.

## Quasi-stationary state

With mutation in one direction only (a -> A, u = 0), fixation of A is permanent. Before that, a
population settles into a quasi-stationary distribution (the Yaglom limit): the distribution of the
allele count conditional on A not yet being fixed. It is the left Perron eigenvector of the
transition matrix restricted to the states that are not fixed for A. Its eigenvalue rho gives the
constant per-generation probability 1 - rho that a population in this state fixes A. When A is
strongly deleterious this probability is exponentially small in N s, even though new copies of A
keep arising by mutation.
