import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def stationary_distribution(N, s, u, v):
    '''Stationary distribution of the number of copies of A under selection, mutation and drift.

    Inputs:
      N: int, population size, 1 <= N <= 400.
      s: float, selection coefficient of A, -0.5 <= s <= 0.5.
      u: float, mutation probability A -> a per generation, 1e-12 <= u <= 0.1.
      v: float, mutation probability a -> A per generation, 1e-12 <= v <= 0.1.

    Output:
      pi: numpy array of shape (N + 1,), pi[i] = stationary probability of i copies of A, summing to 1.
          Relative error below 1e-8 for every entry whose exact value is at least 1e-250; smaller
          entries within 1e-250 of the exact value.

    Raises:
      ValueError if N is not an integer in [1, 400], or if s, u or v is not finite or is outside its
      range.
    '''
    raise NotImplementedError
