# Background: the 2D Ising model on long strips, by transfer matrix

## The model

Spins s = +1 or -1 sit on a square lattice. The energy is
E = -J * sum over nearest-neighbour pairs of s_i s_j - h * sum over sites of s_i, with
J = 1. We set k_B = 1, so the inverse temperature is K = 1/T. In infinite size and zero
field the model orders below the Onsager temperature Tc = 2 / ln(1 + sqrt(2)) = 2.269185...

## Strips and the transfer matrix

Take a strip that is L spins wide and N rows long, periodic across the width (spin L is
spin 0). A row is one of 2**L spin patterns. The Boltzmann weight of the strip is a product
of factors that each involve two neighbouring rows, so Z = trace(M**N) for a 2**L by 2**L
matrix M. Sharing each row's own weight (its in-row bonds and its field term) half and half
between the two factors next to it gives

M[a, b] = exp( (1/2) w(a) + K sum_i a_i b_i + (1/2) w(b) ),  w(a) = K sum_i a_i a_(i+1) + (h/T) sum_i a_i.

For real h this M is real, symmetric and positive definite. For a purely imaginary field
h = i H it is complex symmetric, not Hermitian, and its eigenvalues need not be real.

For N going to infinity the free energy per spin is f = -T ln(lam0) / L, with lam0 the
largest eigenvalue. A correlation function along the strip is a sum over the other
eigenstates of M: a state k contributes a term proportional to (lam_k / lam0)**r at
distance r, provided the operator has a non-zero matrix element between it and the top
state. The slowest term fixes the correlation length of that operator.

The strip has no true phase transition. Below Tc the spin correlation length grows
exponentially with L and the strip behaves, over shorter distances, like an ordered
system with magnetization close to Yang's (1 - sinh(2K)**-4)**(1/8).

## Thermodynamic response and the Yang-Lee edge

The zero-field susceptibility is the second field derivative of -f. By the fluctuation
relation it is also the sum along the whole strip of the correlation of the row
magnetization, divided by T and by the width.

Lee and Yang (1952) showed that the zeros of the Ising partition function in a complex
field lie on the imaginary field axis. For an infinitely long strip the zeros sit where the
two leading eigenvalues of M(iH) have equal modulus. They start at an edge H_edge(L, T),
where the two leading eigenvalues meet on the real axis and turn into a complex-conjugate
pair. Above Tc the edge tends to a finite value, a critical point of its own (Fisher's
Yang-Lee edge singularity, 1978). At Tc it closes like L**(-y_h) with the magnetic
exponent y_h = 15/8, and below Tc it closes exponentially fast.

## Finite-size scaling and conformal invariance at Tc

At the critical point the strip is described by a conformal field theory of central
charge c = 1/2. Cardy (1984) showed that the correlation length of an operator of scaling
dimension x on a periodic strip of width L is xi = L / (2 pi x), with x = 1/8 for the spin
and x = 1 for the energy density. Bloete, Cardy and Nightingale (1986) and Affleck (1986)
showed that f_L = f_inf - pi c T / (6 L**2) + ...; for the periodic Ising strip the next
correction is of order 1/L**4. The magnetization amplitude decays like L**(-beta/nu), the
susceptibility grows like L**(gamma/nu) and the Yang-Lee edge closes like L**(-y_h), with
beta/nu = x_sigma = 1/8, gamma/nu = 2 - 2 x_sigma = 7/4 and y_h = 2 - x_sigma = 15/8. The
two-width estimates of step 6 approach these values as L grows.

## From strips to the infinite lattice

In a field h > 0 the infinite lattice has a single equilibrium state and a finite
correlation length xi(T, h). A quantity of the periodic strip of width L approaches its
bulk value with a correction of order exp(-L / xi) times a power of L (Luscher-type
finite-size corrections). Away from the critical region xi is about one lattice spacing
and modest widths are enough; near Tc in a weak field xi reaches several lattice spacings,
and the strips that can be diagonalized stay visibly away from the bulk value. The last
step asks for the bulk free energy, magnetization, susceptibility and specific heat to
accuracies that make this distinction matter.

## Thermodynamics of the infinite lattice in a field

For h > 0 the infinite lattice has a unique equilibrium state and a finite correlation
length, which near Tc grows like h**(-8/15). Its free energy per spin is analytic there;
the magnetization m = -df/dh, the susceptibility chi = dm/dh and the specific heat
c = -T d2f/dT2 follow from it. Infinite-lattice methods (corner transfer matrices, uniform
matrix product states) give f through the partition function per site, in which the
arbitrary normalizations of the environment tensors must cancel; at h = 0 they reproduce
Onsager's free energy.
