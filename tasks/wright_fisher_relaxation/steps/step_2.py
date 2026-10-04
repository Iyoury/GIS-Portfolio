import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def fixation_statistics(N, s, i0):
    '''Fixation and loss of allele A under selection and drift, without mutation.

    Inputs:
      N: int, population size, 2 <= N <= 400.
      s: float, selection coefficient of A, -0.5 <= s <= 0.5.
      i0: int, initial number of copies of A, 1 <= i0 <= N - 1.

    Output:
      (p_fix, p_loss, t_fix, t_loss): tuple of four Python floats.
        p_fix: probability that A is eventually fixed (i = N).
        p_loss: probability that A is eventually lost (i = 0).
        t_fix: mean number of generations until fixation, given that A is fixed.
        t_loss: mean number of generations until loss, given that A is lost.
        Each with a relative error below 1e-8, however small it is.

    Raises:
      ValueError if N is not an integer in [2, 400], if i0 is not an integer in [1, N - 1], or if s
      is not finite or is outside [-0.5, 0.5].
    '''
    raise NotImplementedError
