import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def quasi_stationary(N, s, v):
    '''Quasi-stationary distribution and absorption rate with one-way mutation a -> A.

    Inputs:
      N: int, population size, 1 <= N <= 400.
      s: float, selection coefficient of A, -0.5 <= s <= 0.5.
      v: float, mutation probability a -> A per generation, 1e-12 <= v <= 0.1 (u = 0: no mutation A -> a).

    Output:
      (rate, qsd): rate is a Python float, the probability per generation that a population in the
        quasi-stationary state fixes A, rate = 1 - rho with rho the largest eigenvalue of the transition
        matrix restricted to i = 0, ..., N - 1; relative error below 1e-8, however small it is.
        qsd is a numpy array of shape (N,), the quasi-stationary distribution over i = 0, ..., N - 1
        (left eigenvector of that restricted matrix for rho, summing to 1); every entry whose exact value
        is positive and at least 1e-250 has a relative error below 1e-8, every other entry lies within
        1e-250 of its exact value.

    Raises:
      ValueError if N is not an integer in [1, 400] (bool and float are not accepted), or if s or v is
      not finite or is outside its range.
    '''
    raise NotImplementedError
