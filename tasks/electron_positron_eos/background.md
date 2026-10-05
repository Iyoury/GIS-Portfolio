# Electron-positron equation of state

In stellar interiors, white dwarfs and the cores of massive stars, the electrons form an ideal Fermi
gas. Depending on density and temperature it can be classical or completely degenerate, and
non-relativistic or ultra-relativistic. Above about 10^9 K, photons create electron-positron pairs.
At low density the pairs then outnumber the electrons supplied by the ions by many orders of
magnitude.

Every quantity follows from the Fermi-Dirac occupation numbers of electrons and positrons:

- number densities;
- pressure;
- energy;
- entropy;
- specific heat.

The pairs are in equilibrium with radiation, so the positron chemical potential is minus that of the
electrons, rest energy included. With psi = (mu + m_e c^2)/(k T), the occupation numbers are
1/(exp(eps/kT -+ psi) + 1). Here eps is the total energy of a particle in units of m_e c^2.

The quantities used in a stellar model are often small differences of large ones:

- **Net electron density.** In a pair plasma it can be 20 decades below the electron and positron
  densities, which nearly cancel.
- **Entropy of a degenerate gas.** It is of order kT/E_F times the particle number. That makes it a
  minute remainder of the Euler relation u + P - mu n = T s.
- **Specific heat at constant density.** In the grand canonical ensemble it is
  (<dE^2> - <dE dN>^2 / <dN^2>)/(k T^2). In a degenerate gas this difference is smaller than its terms
  by (kT/E_F)^2.

All of these quantities are needed with a small relative error over the whole range of stellar
conditions.

References:

- Chandrasekhar (1939), *An Introduction to the Study of Stellar Structure*, ch. X.
- Cox and Giuli (1968), *Principles of Stellar Structure*, ch. 24.
- Timmes and Arnett (1999), ApJS 125, 277.
- Timmes and Swesty (2000), ApJS 126, 501 (the Helmholtz EOS).
- Landau and Lifshitz, *Statistical Physics*, sec. 56-61 and 112-114.
