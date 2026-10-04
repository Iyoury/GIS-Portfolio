# Background: decay chains, activation and flux monitors

## Decay chains and networks

A radioactive nuclide with decay constant lam (half-life ln 2 / lam) decays at the rate lam x. In a
chain each decay feeds the next member, and in a network a fraction of the decays goes to each
daughter. With constant production the amounts obey dx/dt = A x + q. A has nonnegative
off-diagonal entries (feeding rates) and minus the total removal rates on its diagonal. The
solution is x(t) = exp(t A) x0 plus the integral of exp(s A) q; the second term is the solution of an
augmented system with one extra constant state.

Bateman (1910) wrote the chain solution as a sum of exponentials divided by products of differences
of decay constants. The formula is exact but numerically treacherous. With close decay constants
the terms are huge and of alternating sign. With equal ones it has to be replaced by its limit, a
divided difference of the exponential. Natural chains are also extremely stiff: in the 238U series
the decay constants run from 4.9e-18 per s (238U) to 4.2e3 per s (214Po).

Every entry of exp(t A) is a sum of positive contributions, one for each path through the network.
A method whose operations are all additions and multiplications of nonnegative numbers keeps the
relative accuracy of every entry, even one of 1e-200. Dense algorithms (Pade approximants,
eigendecompositions, the Bateman sum in double precision) only reach an error of about 1e-16 times
the largest entry.

## Activation and burnup

In a neutron flux phi (neutrons per cm^2 per s), a nuclide with capture cross section sigma
(1 barn = 1e-24 cm^2) captures neutrons at the rate sigma phi. It is removed (burnup) and turned into
the next heavier isotope. Activation adds these rates to the decay network. A history of periods
with constant flux is the product of the matrix exponentials of the periods. Burnup matters at high
fluence (phi times time): 198Au has a capture cross section of 25100 b, so above about 1e14
n / (cm^2 s) its own captures compete with its decay (2.7 d half-life).

## Flux monitors

Gold, cobalt and other foils are irradiated together with a sample, and their induced activity is
measured afterwards. Inverting the activation equations gives the flux. At low fluence the activity
of a first-order product is proportional to the flux and that of a second-order product (199Au from
two captures) to its square. At high fluence burnup makes the response saturate and turn over, so
the inversion needs the full activation model, solved on a monotone branch.
