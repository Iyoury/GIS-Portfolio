# Background: minimal-basis Hartree-Fock, full CI and vibrational levels of two-electron diatomics

## The problem

Two nuclei with charges ZA and ZB are fixed at distance R (Born-Oppenheimer
approximation). Two electrons with opposite spins move around them. In restricted
Hartree-Fock (RHF) both electrons sit in the same spatial orbital psi, which is written
as a linear combination of basis functions: psi = c_0 phi_0 + c_1 phi_1. Atomic units
are used throughout (hbar = m_e = e = 4 pi eps_0 = 1; lengths in bohr, energies in
hartree).

## The STO-3G basis

A Slater 1s function exp(-zeta r) is hard to integrate, so STO-3G replaces it by a
fixed least-squares fit with three Gaussians (Hehre, Stewart and Pople, 1969). For
zeta = 1 the exponents are 0.109818, 0.405771, 2.22766 and the coefficients (for
normalized primitives) are 0.444635, 0.535328, 0.154329. For another zeta the exponents
are multiplied by zeta**2, because exp(-zeta r) is the zeta = 1 function with r scaled
by zeta. The usual molecular values are zeta = 1.24 for H in molecules and
zeta = 2.0925 for He. With the six rounded numbers the contracted function has a norm
of 1.0000014, not exactly 1; the task keeps it that way, as in Szabo and Ostlund.

## Integrals over Gaussians

The product of two Gaussians on different centres is one Gaussian on a point between
them (Gaussian product theorem). This gives closed forms for the overlap, kinetic and
nuclear-attraction integrals and for the two-electron integrals. The Coulomb integrals
contain the Boys function F0(t) = integral_0^1 exp(-t u**2) du, with F0(0) = 1. F0 can
be written with a standard special function, but that form gives 0/0 at t = 0, which is
a common bug: t = 0 happens for the attraction of a function on A to nucleus A, and for
(00|00).

## Roothaan-Hall equations

With the basis fixed, the RHF condition becomes a generalized eigenvalue problem
F c = e S c (Roothaan-Hall), where the Fock matrix F contains the core Hamiltonian plus
the Coulomb and exchange fields of the electrons, so it depends on the orbital it is
solving for. It is solved by iteration until the orbital no longer changes (self-consistent
field). With only two basis functions the orbital has one free parameter, so the energy
can also be minimized directly. For H2 at R = 1.4 bohr with zeta = 1.24 the total energy
is -1.1167 hartree, and for HeH+ at R = 1.4632 bohr with zeta(He) = 2.0925,
zeta(H) = 1.24 it is -2.8607 hartree (Szabo and Ostlund report -1.117 and -2.860662; the
small difference for HeH+ comes from the approximate erf used in their program).

## Full configuration interaction

Two electrons in two spatial functions span three singlet configurations (both in one
orbital, both in the other, one in each). Diagonalizing the Hamiltonian in this space is
full CI, the exact answer in the basis. Near equilibrium it lies below RHF by the
correlation energy; on stretching, RHF keeps a fixed ionic weight and goes to a wrong
limit, while full CI separates into the lowest fragment arrangement. For H2 at 1.4 bohr
with zeta = 1.24, Szabo and Ostlund give -1.1373 hartree (full CI) and -1.1167 (RHF).

## Dissociation limit

As R -> infinity all two-centre integrals vanish and the lowest singlet is the lowest of:
both electrons on A, both on B, or one on each centre, with energies built from
one-centre integrals of each function only (normalized by its norm, which is not exactly
1 here). The remaining energy at large R is the Coulomb interaction q_A q_B / R of the
fragment charges. For H + H and He + H+ one fragment is neutral and the approach is
exponential; for an ion pair (H- + H+, which becomes the lowest limit in this basis when
the two exponents are very different) the tail is -1/R, and for fragments that both
carry a fraction of a charge (non-integer Z) it is repulsive. Taking E(R) at a large but
finite R as the limit is then wrong by q_A q_B / R.

## Vibrational levels

With the full-CI curve as the Born-Oppenheimer potential, the rotationless nuclear motion
obeys a one-dimensional radial Schrodinger equation with the reduced mass of the nuclei.
Its bound states below the dissociation limit are the vibrational levels; they are
anharmonic (spacings shrink with v) and their positions depend on the exact limit through
the reference energy. Accurate levels need a converged representation of the radial
equation (discrete-variable, shooting or spectral methods).

## The harmonic picture (for orientation)

Repeating the calculation for many R gives the potential energy curve. Its minimum
is the equilibrium bond length R_e. The curvature k at the minimum gives the harmonic
vibrational wavenumber omega_e = sqrt(k/mu)/(2 pi c), where mu is the reduced mass of
the nuclei. For H2 the RHF/STO-3G values are R_e = 1.3459 bohr (0.712 angstrom),
E_e = -1.11751 hartree and omega_e of about 5481 cm^-1 (the experimental values are
1.401 bohr and 4401 cm^-1; the gap is the known error of minimal-basis Hartree-Fock).
