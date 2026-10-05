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
            that max_i |mode[i]| = 1 and its first nonzero entry is positive; every entry with an error
            below 1e-8 times its absolute value (entries that are exactly zero returned as 0.0).

    Raises:
      ValueError if N is not an integer in [1, 400] (bool and float are not accepted), if s, u or v
      is not finite or is outside its range, or if N = 1 and u = v = 0.
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
    if N == 1 and u == 0.0 and v == 0.0:
        raise ValueError("with N = 1 and u = v = 0 both states are absorbing")
    # eigenvalues and right eigenvectors from a dense eigensolver, the smallest eigenvalue taken
    P = wf_transition_matrix(N, s, u, v)
    ev, W = np.linalg.eig(P)
    j = int(np.argmin(np.abs(ev)))
    lam_min = float(abs(ev[j]))
    mode = W[:, j].real
    mode = mode / np.max(np.abs(mode))
    first = mode[np.nonzero(mode)[0][0]]
    mode = mode / np.sign(first)
    return lam_min, mode
