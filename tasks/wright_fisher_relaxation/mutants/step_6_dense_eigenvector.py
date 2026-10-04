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
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)):
        raise ValueError("N must be an integer")
    N = int(N)
    if not 1 <= N <= 400:
        raise ValueError("need 1 <= N <= 400")
    s, v = float(s), float(v)
    if not (np.isfinite(s) and np.isfinite(v)):
        raise ValueError("s and v must be finite")
    if not (-0.5 <= s <= 0.5 and 1e-12 <= v <= 0.1):
        raise ValueError("need -0.5 <= s <= 0.5 and 1e-12 <= v <= 0.1")
    # MUTANT: Perron root and left eigenvector of the restricted matrix from a dense eigensolver
    P = wf_transition_matrix(N, s, 0.0, v)
    w, vl = np.linalg.eig(P[:N, :N].T)
    k = int(np.argmax(np.real(w)))
    qsd = np.abs(np.real(vl[:, k]))
    qsd = qsd / qsd.sum()
    result = (float(1.0 - np.real(w[k])), qsd)
    return result
