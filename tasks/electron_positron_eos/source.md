# Sources, method and ground truth

## Physics
Ideal relativistic Fermi gas of electrons and positrons with pairs in equilibrium with radiation
(Chandrasekhar 1939; Cox and Giuli 1968; Timmes and Arnett 1999). Definitions follow the textbook
integrals over momentum. The constants are CODATA 2018 (m_e c^2, hbar/(m_e c)) and the exact SI-2019
values (k, N_A).

## Reference solution
- **Quadrature.** Composite 24-point Gauss-Legendre.
  - Below the Fermi edge, in t = sqrt(E) (E the kinetic energy in m_e c^2), with doubling panels.
  - Over the edge E_F +- 60 kT, in y = (E - E_F)/kT with panels of width 1. The electron occupation
    argument is then y itself, not a difference of two large energies; this keeps the entropy accurate
    to 1e-15 even at psi = 1e6.
- **Net electron density.** Computed from f(b - psi) - f(b + psi) = sinh(psi)/(cosh b + cosh psi). The
  integrand is positive and summed in logarithms, so there is no cancellation and no underflow.
- **Entropy.** Uses the positive integrand -[f ln f + (1-f) ln(1-f)] = log1p(e^{-|x|}) + |x|/(e^{|x|}+1).
- **psi from rho Y_e.** A safeguarded secant/bisection in ln psi on ln n_net. The variable psi includes
  the rest energy, so a pair plasma with psi = 2e-21 is represented exactly; mu/kT would round it away.
- **Specific heat.** Computed as c_V = (1/kT^2) det / <dN^2>. The determinant
  det = <dE^2><dN^2> - <dE dN>^2 = 1/2 sum_ij w_i w_j (e_i n_j - e_j n_i)^2 is evaluated term by term:
  - same-species pairs give two-pass centred second moments;
  - electron-positron pairs give (e_i + e_j)^2 > 0.

  There is no cancellation. The Schur form loses (kT/E_F)^2, which is 2.8e-6 relative at
  rho_Ye = 1e13, T = 1e7 K.

## Second solution
- Adaptive QUADPACK (scipy.integrate.quad) in E, split at the edge window, instead of fixed panels.
- Brent's method for psi.
- Centred moments for c_V, by the same adaptive quadrature.
- Agrees with the reference to about 1e-13.

## Targets (all independent of the solution code)
- **Classical gas.** The Maxwell-Juettner closed forms with modified Bessel functions, in mpmath:
  - n = theta K2(1/theta) e^{+-psi}/(pi^2 lambda^3);
  - P = n_tot k T;
  - <eps> = K1/K2 + 3 theta;
  - s per particle = k(<eps>/theta -+ psi + 1);
  - psi from n in closed form.

  These are used only where the Fermi corrections are below 1e-11. Positron densities of a classical
  positron gas also come from the Maxwell-Juettner formula.
- **Twelve (T, psi) states, from psi = 2.2e-21 to 1e6 and from 1e7 to 1e11 K.** mpmath tanh-sinh
  quadrature at 40 digits of the defining integrals (n_minus, n_plus, n_minus - n_plus computed in
  40-digit arithmetic, P, u, s), with breakpoints every kT over the Fermi edge.
- **Ten (rho_Ye, T) states.** psi from an mpmath root search on those integrals. c_V from a centred
  difference of u at constant n_net (T +- 1e-7 T, psi re-solved at each temperature), all at 40 digits.
  This is a different route from the fluctuation formula of the reference.
- **Four (rho, T, Ye) states for the complete equation of state.** Same methods.

The reference agrees with all targets to 1e-13 or better: about 1e-13 for c_V, below 2e-14 for the
others.

## Why the naive formulas fail the stated accuracy
- n_minus - n_plus: 100 % error in the pair plasma (rho_Ye = 1e-10, T = 1e11 K).
- (u_tot + P - mu n)/T for the entropy: 1.6e-5 at psi = 1.3e5, T = 1e7 K (required 1e-10).
- The Schur form of c_V: 2.8e-6 at rho_Ye = 1e13, T = 1e7 K, and 4.5e-8 at 1e11 (required 1e-8).
