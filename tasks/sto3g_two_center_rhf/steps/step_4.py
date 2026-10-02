import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


def fci_energy(ZA, ZB, zetaA, zetaB, R):
    '''Full-CI and closed-shell RHF ground-state energies of a two-electron diatomic in STO-3G.

    Inputs:
      ZA, ZB: float, nuclear charges (for example 1, 1 for H2 and 2, 1 for HeH+), > 0.
      zetaA, zetaB: float, Slater exponents of the 1s functions on A and B, > 0.
      R: float, distance between the nuclei in bohr, R > 0.

    Output:
      (E_fci, E_rhf): tuple of two Python floats, total energies in hartree (electronic
      energy plus the nuclear repulsion ZA * ZB / R), absolute errors below 1e-9.
        E_fci: lowest eigenvalue of the electronic Hamiltonian in the space of all
               two-electron states built from the two basis functions (full CI).
        E_rhf: closed-shell restricted Hartree-Fock ground-state energy.
    '''
    raise NotImplementedError
