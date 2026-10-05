# Scientific background: heat flow from an inclined Shield borehole

## Why this workflow exists

Surface heat flow constrains crustal radiogenic heat production, lithospheric
thickness and geothermal potential. On the Canadian Shield, most heat-flow
determinations come from **mineral-exploration diamond-drill holes** that have
been left to re-equilibrate after drilling. Examples are the Abitibi,
Sudbury, Thompson and Flin Flon compilations (Jessop et al. 1984; Mareschal &
Jaupart 2004).

These holes are drilled at an angle to cut steeply dipping ore bodies. A
collar dip of 45–60° is typical, and the path keeps bending as it deepens.
Conductivity is measured on core from each rock unit. Shallow temperatures
still carry the signature of the last glaciation, which ended only about
10 ka ago.

Producing a heat-flow value therefore takes three corrections: convert depth
along the hole to vertical depth, integrate thermal resistance through the
layered column, and remove the paleoclimatic transient.

## 1. Borehole trajectory: minimum curvature

A deviation survey gives, at measured depths (MD) along the hole, the
inclination I from vertical (0° = straight down) and the azimuth A (clockwise
from north). The unit tangent in (north, east, down) coordinates is

    t(I, A) = (sin I cos A, sin I sin A, cos I)

The **minimum-curvature** method is the industry standard. It assumes that
between two stations the hole follows the circular arc along which the tangent
turns at a constant rate on the great circle from t1 to t2. The total turning
angle between the two stations is the **dogleg** β, with cos β = t1 · t2.

The vertical depth of any point follows by integrating the down component of
the tangent along this arc. The closed form contains the "ratio factor" of
the method (Sawaryn & Thorogood 2005); a point inside a segment is handled
with the partial arc that ends at that point.

Interpolating TVD, or the angles, linearly between stations is not the
minimum-curvature path. Neither is ignoring the ratio factor (the
"balanced tangential" method). Treating MD as depth is a gross error in
inclined holes: T(z) stretched along the hole lowers the apparent gradient by
cos I, typically 15–30 %.

Layer contacts are logged in MD along the core. In this task they are
horizontal, so their depth is the TVD of the contact.

## 2. Steady conduction in a layered column (Bullard method)

In one dimension, with conductivity k(z) and uniform radiogenic heat
production A (W m^-3), the heat flow decreases with depth as q(z) = q0 − A z,
and Fourier's law gives dT/dz = q(z)/k(z). Integrating from the surface:

    T_ss(z) = T0 + q0 R(z) − A S(z)

where R(z) is the thermal resistance between the surface and depth z
(m^2 K W^-1) and S(z) (m^3 K W^-1) is the second depth function that comes
from the heat-production term; both are zero at the surface.

Resistances add in series, so the effective conductivity of a stack is the
thickness-weighted **harmonic** mean. An arithmetic mean, or a single average
k, biases both q0 and the curvature of the profile. The "Bullard plot" of T
against R is a straight line when there is no heat production and no
transient.

## 3. Paleoclimate perturbation

Suppose the ground surface of a homogeneous half-space (diffusivity κ, in
m^2 s^-1) undergoes a step change ΔT that starts a time t before present and
persists until now. The present-day perturbation at depth z is

    ΔT(z) = ΔT · erfc( z / (2 √(κ t)) )

A step that began at t_old and ended at t_new (both before present, with
t_new < t_old) is the difference of two such responses. A piecewise-constant
surface history is the sum of these terms by superposition (Birch 1948;
Beck 1977).

Some practical points:

- Times must be in seconds. This task uses a Julian year of 365.25 days.
- The erfc argument contains the factor 2. Dropping it is a common error.
- For the interval that ends at the present (t_new = 0), erfc(z/0) → 0 for
  z > 0.

On the Shield, the last glaciation left temperatures in the upper 1–2 km
several kelvin warmer than a steady profile would predict. If the correction
is ignored, heat flow from holes shallower than about 1 km is underestimated
by several mW m^-2.

The *shape* of the history comes from paleoclimate proxies: Holocene,
glacial, and the Eemian interglacial. Its *amplitude* is poorly known,
because the ground-surface temperature under an ice sheet is set by the
basal thermal regime. The amplitude is therefore estimated from the borehole
itself.

## 4. Joint estimate

With R, S and the unit-amplitude perturbation P(z) evaluated at every
reading, the observation model is linear in the unknowns T0, q0 and g:

    T_obs(z) = T0 + q0 R(z) − A S(z) + g P(z) + ε

A is known from geochemistry on core, so the term A·S is moved to the left
side. The three parameters follow from ordinary least squares on the design
matrix [1, R, P], with the usual least-squares parameter covariance and the
unbiased residual variance RSS/(n − 3).

Readings shallower than a cut-off depth are excluded. They are disturbed by
recent (post-Little-Ice-Age) warming and by ground-water circulation in
fractured near-surface rock.

## References

- Beck, A.E. (1977) Climatically perturbed temperature gradients and their effect on regional and continental heat-flow means. Tectonophysics 41, 17–39.
- Birch, F. (1948) The effects of Pleistocene climatic variations upon geothermal gradients. American Journal of Science 246, 729–760.
- Bullard, E.C. (1939) Heat flow in South Africa. Proceedings of the Royal Society A 173, 474–502.
- Jessop, A.M., Lewis, T.J., Judge, A.S., Taylor, A.E., Drury, M.J. (1984) Terrestrial heat flow in Canada. Tectonophysics 103, 239–261.
- Mareschal, J.-C., Jaupart, C. (2004) Variations of surface heat flow and lithospheric thermal structure beneath the North American craton. Earth and Planetary Science Letters 223, 65–77.
- Sawaryn, S.J., Thorogood, J.L. (2005) A compendium of directional calculations based on the minimum curvature method. SPE Drilling & Completion 20, 24–36.
- Carslaw, H.S., Jaeger, J.C. (1959) Conduction of Heat in Solids, 2nd ed. Oxford.
