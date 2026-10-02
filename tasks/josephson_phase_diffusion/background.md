# Background: phase diffusion in overdamped Josephson junctions

## The noisy RSJ model

A Josephson junction shunted by a resistance R (resistively shunted junction, RSJ) with
negligible capacitance is overdamped: the current through the resistor, (hbar / 2e R)
dphi/dt, plus the supercurrent I_c sin(phi) equals the bias current I plus the
Johnson-Nyquist noise current of the resistor. The noise is Gaussian and white with
<I_N(t) I_N(t')> = (2 k_B T / R) delta(t - t'). In the reduced time
tau = (2 e I_c R / hbar) t the phase obeys

dphi/dtau = i - sin(phi) + xi(tau),  <xi(tau) xi(tau')> = 2 theta delta(tau - tau'),

with i = I / I_c and theta = k_B T / E_J, E_J = hbar I_c / 2e. This is the overdamped
Brownian motion of a particle in the tilted washboard potential U(phi) = -cos(phi) - i phi.
The Josephson relation V = (hbar / 2e) dphi/dt turns the mean phase velocity into the dc
voltage, <V> = I_c R v.

Without noise the phase is trapped in a minimum of U for |i| < 1 (v = 0) and runs for
|i| > 1 with v = sqrt(i**2 - 1). Thermal noise lets the phase slip over the barriers in
both directions, so a small voltage appears below the critical current (phase diffusion
regime) and the current-voltage curve is rounded (Ambegaokar and Halperin 1969; Ivanchenko
and Zil'berman 1969).

## Exact stationary results

The Fokker-Planck equation for the phase density on the circle has a stationary solution
with a constant probability current, which gives v exactly as a ratio of integrals of
Boltzmann factors (Stratonovich formula). The long-time variance of the phase grows
linearly, and the effective diffusion coefficient D_eff has an exact expression of the
same type (Reimann et al. 2001); it can also be obtained from homogenization theory
(a periodic corrector function) or from the first two moments of the first passage time
over one period. At zero bias D_eff reduces to the Lifson-Jackson formula. Out of
equilibrium D_eff is not given by the Einstein relation theta dv/di.

Below the critical current at small theta both v and D_eff are exponentially small
(Arrhenius-like phase-slip rates), so methods whose error is only absolute (for example a
truncated Fourier solution in double precision, where v appears as a difference of
numbers of order one) lose all relative accuracy there.

## Giant diffusion

Close to the critical tilt i ~ 1 the phase alternates between long dwell times in an
almost flat minimum and fast running stretches. The dispersion of its displacement is
then much larger than for free diffusion: D_eff / theta has a pronounced maximum that grows
as theta -> 0 (giant diffusion; Reimann et al. 2001, Lindner et al. 2001). As theta grows
the peak broadens and moves above the critical current.

## Thermometry and junction arrays

Because the voltage in the phase-diffusion regime depends exponentially on E_J / k_B T,
a measured voltage at a known bias fixes the effective noise temperature of a junction.
Kramers-rate formulas are only asymptotic and give a biased temperature.

In a series array biased by one current source every junction has its own critical
current, normal resistance and Johnson noise. Weaker junctions have a smaller E_J and are
therefore noisier at the same temperature. The array voltage is the sum of the junction
voltages; experiments define the critical current of the array through a voltage
criterion, which then depends on temperature and on the criterion itself. The zero-
frequency voltage noise of each junction follows from its phase diffusion: the
accumulated phase is (2e / hbar) times the time integral of the voltage, and the reduced
time of junction k runs at the rate 2 e I_ck R_k / hbar.

## Junctions with capacitance

A real junction also has a capacitance C, which gives the phase a mass: the RCSJ model
beta_c phi'' + phi' + sin(phi) = i + noise, with the Stewart-McCumber parameter
beta_c = 2 e I_c R**2 C / hbar. Without noise and for beta_c of order one or larger, a locked
and a running state coexist over a range of bias (hysteresis, retrapping current). With
noise the stationary state is unique: its distribution of phase and phase velocity solves a
Kramers equation, and the dc voltage is its mean velocity. There is no closed formula; the
standard exact numerical route expands the velocity dependence in Hermite functions and the
phase dependence in Fourier modes and solves the resulting hierarchy by matrix continued
fractions (Risken; Vollmer and Risken). The overdamped Ambegaokar-Halperin result is the
limit beta_c -> 0.
