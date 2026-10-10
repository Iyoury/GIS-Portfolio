import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def substitution_times(N, s, u, v):
    '''Mean waiting times between the two monomorphic states under selection, mutation and drift.

    Inputs:
      N: int, population size, 1 <= N <= 400.
      s: float, selection coefficient of A, -0.5 <= s <= 0.5.
      u: float, mutation probability A -> a per generation, 1e-12 <= u <= 0.1.
      v: float, mutation probability a -> A per generation, 1e-12 <= v <= 0.1.
      Inputs for which t_up or t_down would exceed 1e300 generations are outside the domain (within these
      ranges only t_up can be that large, near N = 400, s = -0.5, u = 0.1, v = 1e-12).

    Output:
      (t_up, t_down): tuple of two Python floats.
        t_up: mean number of generations for a population with no copy of A (i = 0) to reach i = N
              for the first time.
        t_down: mean number of generations for a population fixed for A (i = N) to reach i = 0 for
                the first time.
        Each with a relative error below 1e-8, however large (t_down is about 1e150 at
        N = 400, s = 0.5, u = v = 1e-12).

    Raises:
      ValueError if N is not an integer in [1, 400], or if s, u or v is not finite or is outside its
      range.
    '''
    raise NotImplementedError
