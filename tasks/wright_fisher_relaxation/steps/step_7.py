import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def fastest_mode(N, s, u, v):
    '''Smallest eigenvalue of the Wright-Fisher transition matrix and its right eigenvector.

    Inputs:
      N: int, population size, 1 <= N <= 400.
      s: float, selection coefficient of A, -0.5 <= s <= 0.5.
      u, v: float, mutation probabilities A -> a and a -> A per generation, 0 <= u, v <= 0.1
            (v = 0 makes i = 0 absorbing, u = 0 makes i = N absorbing).

    Output:
      lam_min: Python float, the smallest eigenvalue of the transition matrix P of step 1
               (wf_transition_matrix), with a relative error below 1e-8.
      mode: numpy float array of shape (N + 1,), the right eigenvector (P mode = lam_min mode), scaled so
            that max_i |mode[i]| = 1 and its first entry that is not on an absorbing state is positive;
            every entry with an error below 1e-8 times its absolute value, except on the absorbing states,
            where the exact entries are zero and the returned entries must be below 1e-250 in absolute value.

    Raises:
      ValueError if N is not an integer in [1, 400] (bool and float are not accepted), if s, u or v
      is not finite or is outside its range, or if N = 1 and u = v = 0.
    '''
    raise NotImplementedError
