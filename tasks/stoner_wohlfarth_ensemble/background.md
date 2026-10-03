# Background: thermally activated switching of Stoner-Wohlfarth particles

## One particle at zero temperature

A small ferromagnetic particle (below about 50 nm) has no domain walls: all its spins
turn together, so its magnetization is a vector of fixed length M_s. With uniaxial
anisotropy constant K, volume V and an applied field H, the energy is

E = K V sin(theta)**2 - mu0 M_s H V cos(theta - psi) = 2 K V e(theta),

where theta is the angle of the magnetization from the easy axis, psi the angle of the
field axis from the easy axis, h = H / H_K with H_K = 2K / (mu0 M_s), and
e(theta) = (1/2) sin(theta)**2 - h cos(theta - psi).

For small |h| there are two minima, one on each side of the hard axis, separated by two
maxima. A slowly changing field moves a minimum continuously until it merges with a
maximum and vanishes; then the magnetization jumps to the other minimum. The field
where this happens is the switching field h_sw(psi); in the plane of the field
components along and across the easy axis these fields form the Stoner-Wohlfarth
astroid. It equals 1 at psi = 0 and psi = pi/2 and 0.5 at psi = pi/4. A particle does
not go to the global minimum: between -h_sw and h_sw its state depends on its history
(hysteresis).

## Thermal activation

At a temperature T the magnetization fluctuates and can cross an energy maximum before
its minimum disappears. In the Neel-Brown picture the escape over a barrier dE happens
at the rate f0 exp(-dE / (k_B T)), with an attempt frequency f0 of order 1e9 to 1e10 per
second. Because E = 2 K V e, a reduced barrier Delta_e corresponds to
dE / (k_B T) = 2 a Delta_e with the thermal stability ratio a = K V / (k_B T); at zero
field both barriers are Delta_e = 1/2, i.e. dE = K V. Particles with a above about 40
are stable for years at zero field (magnetic recording needs a of 40 to 60).

During a field sweep the barrier of the occupied minimum shrinks as the field comes
down, so the escape rate rises by many orders of magnitude over a short field interval.
The particle therefore switches at a random field spread over a narrow window, well
before the zero-temperature switching field; slower sweeps and smaller a switch at
smaller fields. Sharrock's law, h_s ~ h_sw [1 - (ln(f0 t) / a)**n] with n about 1/2 to
2/3, captures this trend only approximately; an exact treatment needs the full field
dependence of both barriers along the sweep.

## Ensembles and dynamic coercivity

Without interactions, the magnetization of an ensemble is the weighted mean of the
expected magnetizations of its particles. Each particle contributes the magnetization
of its original minimum with the probability that it is still there, and the
magnetization of the other minimum otherwise. The coercive field measured in a sweep
(the field where the ensemble magnetization crosses zero) depends on the sweep rate
and the temperature: this is the dynamic coercivity measured in magnetic recording
media. It differs from the field where half of the particles have switched, because the
magnetization inside each minimum also rotates reversibly with the field.

## Exact thermal relaxation: Brown's Fokker-Planck equation

The Arrhenius rate f0 exp(-dE / k_B T) of the earlier steps rests on an assumed attempt
frequency. Brown (1963) showed that the thermally fluctuating magnetization obeys a
Fokker-Planck equation; for a field along the easy axis it depends on z = cos(theta) only:
2 tau_N dW/dt = d/dz[(1 - z^2)(dW/dz + W d(beta E)/dz)], beta E = sigma (1 - z^2 - 2 h z).
Its smallest nonzero eigenvalue lam1 is the true switching rate, exponentially small for high
barriers; Brown's asymptote (2 / sqrt(pi)) sigma^1.5 exp(-sigma) (h = 0) holds only for large
sigma. The integral relaxation time tau_int of the magnetization weights every mode by its
contribution; in a strong field the shallow well is almost empty, the slow mode barely shows in
<z(t) z(0)>, and tau_int falls orders of magnitude below 1 / lam1 (Coffey, Kalmykov, Garanin).
Accurate numbers need care: lam1 can be 1e-24 next to eigenvalues of order 100, and the two
wells can differ in Boltzmann weight by 1e-94.
