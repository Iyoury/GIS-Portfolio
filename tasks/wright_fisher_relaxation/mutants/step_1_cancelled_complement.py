import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def wf_transition_matrix(N, s, u, v):
    '''One-generation transition matrix of the haploid Wright-Fisher model with selection and mutation.

    Inputs:
      N: int, population size, 1 <= N <= 400.
      s: float, selection coefficient of allele A (fitness 1 + s against 1 for a), -0.5 <= s <= 0.5.
      u: float, mutation probability A -> a per generation, 0 <= u <= 0.1.
      v: float, mutation probability a -> A per generation, 0 <= v <= 0.1.

    Output:
      P: numpy array of shape (N + 1, N + 1), P[i, j] = probability that a population with i
         copies of A has j copies in the next generation. Relative error below 1e-10 for every
         entry whose exact value is at least 1e-250; smaller entries within 1e-250 of the exact value.

    Raises:
      ValueError if N is not an integer in [1, 400], or if s, u or v is not finite or is outside
      its range.
    '''
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)):
        raise ValueError("N must be an integer")
    N = int(N)
    if not 1 <= N <= 400:
        raise ValueError("need 1 <= N <= 400")
    s, u, v = float(s), float(u), float(v)
    if not (np.isfinite(s) and np.isfinite(u) and np.isfinite(v)):
        raise ValueError("s, u and v must be finite")
    if not (-0.5 <= s <= 0.5 and 0.0 <= u <= 0.1 and 0.0 <= v <= 0.1):
        raise ValueError("need -0.5 <= s <= 0.5 and 0 <= u, v <= 0.1")
    i = np.arange(N + 1, dtype=float)
    # frequencies after selection and after mutation; the complements are formed directly, never as
    # 1 - p, because (1 - p)**(N - j) multiplies any relative error of 1 - p by N - j
    den = N + s * i
    p_sel = (1.0 + s) * i / den
    q_sel = (N - i) / den
    p_mut = (1.0 - u) * p_sel + v * q_sel
    # MUTANT: complement of the mutated frequency formed by subtraction
    q_mut = 1.0 - p_mut
    j = np.arange(N + 1, dtype=float)[None, :]
    log_binom = gammaln(N + 1.0) - gammaln(j + 1.0) - gammaln(N - j + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # 0 * log(0) = 0: a population fixed for one allele without mutation stays fixed
        log_p = np.where(j > 0, j * np.log(p_mut)[:, None], 0.0)
        log_q = np.where(j < N, (N - j) * np.log(q_mut)[:, None], 0.0)
    P = np.exp(log_binom + log_p + log_q)
    return P
