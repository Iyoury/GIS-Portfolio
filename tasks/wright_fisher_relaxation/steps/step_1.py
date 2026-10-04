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
    raise NotImplementedError
