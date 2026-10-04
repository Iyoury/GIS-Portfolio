import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def relaxation_rate(N, s, u, v):
    '''Rate of the slowest approach to the stationary distribution: 1 - lambda_2 of the transition matrix.

    Inputs:
      N: int, population size, 1 <= N <= 400.
      s: float, selection coefficient of A, -0.5 <= s <= 0.5.
      u: float, mutation probability A -> a per generation, 1e-12 <= u <= 0.1.
      v: float, mutation probability a -> A per generation, 1e-12 <= v <= 0.1.

    Output:
      gap: Python float, 1 - lambda_2, where lambda_2 is the second largest eigenvalue of the
           transition matrix (all its eigenvalues are real, positive and distinct). Relative error
           below 1e-8, however small the gap is.

    Raises:
      ValueError if N is not an integer in [1, 400], or if s, u or v is not finite or is outside its
      range.
    '''
    raise NotImplementedError
