# Gravity of prisms and sedimentary basins

Gravity surveys measure small variations of the Earth's attraction. These variations come from density
contrasts under the surface. To interpret them, forward models are built from simple bodies, and the
right rectangular prism of uniform density is the workhorse. Its potential, attraction and gradient
tensor are known in closed form, as sums over its eight corners of logarithmic and arctangent terms
(Nagy 1966; Nagy, Papp and Benedek 2000).

These closed forms are exact, but two features matter numerically:

- **Cancellation far from the prism.** The corner terms grow like the square of the distance, while the
  potential decreases like 1/distance and the attraction like 1/distance^2. Far from the body, the sum
  is a small difference of large numbers.
- **Limits at the boundary.** On the faces, edges and corners of the prism, some terms take limiting
  forms such as 0 ln 0 or the arctangent of 0/0. The gradient tensor is discontinuous across the faces
  and singular along the edges.

Sedimentary basins are commonly modelled as columns of sediment over a denser basement. Compaction makes
the density contrast decrease with depth, and an exponential law is classical:
drho(z) = drho0 exp(-lam z) (Cordell 1973; Chakravarthi and Sundararajan 2007). A column then has no
closed-form attraction. Its horizontal laminae do, so the attraction is an integral over depth of a
lamina term. That term varies sharply near the surface when a station lies close to the edge of the
column.

With one station above each column, the gravity anomaly of the basin determines the depths of the
columns (Bott 1960; Cordell and Henderson 1968).
