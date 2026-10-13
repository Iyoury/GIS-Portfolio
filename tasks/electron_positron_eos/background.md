# Electron-positron equation of state

In stellar interiors, white dwarfs and the cores of massive stars, the electrons form an ideal Fermi
gas. Depending on density and temperature it can be classical or completely degenerate, and
non-relativistic or ultra-relativistic. Above about 10^9 K, photons create electron-positron pairs.
At low density the pairs then outnumber the electrons supplied by the ions by many orders of
magnitude.

Number densities, pressure, energy, entropy and specific heat all follow from the Fermi-Dirac occupation
numbers of electrons and positrons.

The pairs are in equilibrium with radiation, so the positron chemical potential is minus that of the
electrons, rest energy included. With psi = (mu + m_e c^2)/(k T), the occupation numbers are
1/(exp(eps/kT -+ psi) + 1). Here eps is the total energy of a particle in units of m_e c^2.

The gas spans pair plasmas, in which the net electron density lies many decades below the electron and
positron densities, classical gases, and completely degenerate gases, whose thermal properties are tiny
compared with their zero-temperature values.

A stellar model also needs the response of the gas to a reversible adiabatic change, at constant
entropy per unit mass:

- **First adiabatic exponent** Gamma1 = (d ln P / d ln rho) at constant entropy. It decides dynamical
  stability: a self-gravitating body needs a pressure-weighted Gamma1 above 4/3.
- **Adiabatic temperature gradient** nabla_ad = (d ln T / d ln P) at constant entropy, the threshold of
  the Schwarzschild criterion for convection.

With the pairs in equilibrium only the net electron number is conserved during the change, so pair
creation enters every derivative. For the completely degenerate gas Gamma1 falls from 5/3 to 4/3 as the
electrons become relativistic, which sets the Chandrasekhar mass. In a pair plasma part of the heat
goes into the rest energy of new pairs, and Gamma1 of the pair gas falls below 4/3.

All of these quantities are needed with a small relative error over the whole range of stellar
conditions.

References:

- Chandrasekhar (1939), *An Introduction to the Study of Stellar Structure*, ch. X.
- Cox and Giuli (1968), *Principles of Stellar Structure*, ch. 24.
- Timmes and Arnett (1999), ApJS 125, 277.
- Timmes and Swesty (2000), ApJS 126, 501 (the Helmholtz EOS).
- Landau and Lifshitz, *Statistical Physics*, sec. 56-61 and 112-114.
