# Sources, data provenance and ground-truth evidence

## Scientific sources

| Element | Source | What was taken |
|---|---|---|
| Minimum-curvature trajectory | Sawaryn & Thorogood (2005) | Ratio factor, great-circle interpolation between stations |
| Layered conduction, Bullard method | Bullard (1939); Carslaw & Jaeger (1959) | R(z), S(z), steady geotherm with uniform heat production |
| Paleoclimate correction | Birch (1948); Beck (1977) | erfc step response, superposition of a piecewise history |
| Shield workflow and magnitudes | Jessop et al. (1984); Mareschal & Jaupart (2004) | Inclined exploration holes, typical q0 of 35–55 mW m^-2, glacial correction |
| Layer over a half-space (step 5 targets) | Carslaw & Jaeger (1959), ch. 12 | Image series of the step response, alpha = (e1 − e2)/(e1 + e2) with the effusivities e = √(k ρc) |
| Fixed Talbot inversion (step 5 reference, mpmath targets) | Abate & Valkó (2004) | Contour s = rθ(cot θ + i), r = 2M/5t, and the trapezoid weights |
| Hyperbolic contour (step 5 second solution) | Weideman & Trefethen (2007) | s(u) = μ(1 + sin(iu − α)) with α = 1.1721, h = 1.0818/N, μ = 4.4921 N/t |
| Gaver–Stehfest inversion (step 5 mutant, cross-check of the 50-layer targets) | Stehfest (1970) | Weights V_k and nodes k ln 2 / t |
| Branch-cut (real-axis) inversion (steps 6 and 7 and whole-task test oracle) | Author's derivation (version 10, difficulty_increase, item 4) | Bromwich contour collapsed on the cut of √s; checked against the stored 90-digit values and erfc |
| Wald covariance and profile-likelihood interval of t_end (step 6) | Standard nonlinear least squares; derivation in the version 10 (difficulty_increase) note, item 3 | s² (JᵀJ)⁻¹ with n − 4 degrees of freedom; threshold RSS_min (1 + F₀.₉₅(1, n − 4)/(n − 4)) |
| Maximum likelihood with an unknown variance per hole, expected Fisher information, likelihood-ratio interval (step 7) | Standard likelihood theory; derivation in the version 10 (difficulty_increase) note, item 12c | Concentrated likelihood −Σ (n_h/2)(log(2π RSS_h/n_h) + 1); information Σ J_hᵀJ_h/σ_h²; threshold l_max − χ²₀.₉₅(1)/2 |

Full references: Abate, J., Valkó, P.P. (2004) Multi-precision Laplace transform inversion. International
Journal for Numerical Methods in Engineering 60, 979–993. Weideman, J.A.C., Trefethen, L.N. (2007) Parabolic
and hyperbolic contours for computing the Bromwich integral. Mathematics of Computation 76, 1341–1356.
Stehfest, H. (1970) Algorithm 368: Numerical inversion of Laplace transforms. Communications of the ACM 13,
47–49. Carslaw, H.S., Jaeger, J.C. (1959) Conduction of Heat in Solids, 2nd ed., Oxford. The other works are
listed in background.md. Each formula taken from these works was checked numerically against the independent
targets of the tests (closed forms, image series, extended-precision values), not only taken on trust.

Everything else is the author's own design:

- the task decomposition;
- the joint estimation of the glacial amplitude together with T0 and q0;
- the joint estimation of the end of the glaciation t_end with the layered perturbation (version 10,
  difficulty_increase);
- the regional maximum-likelihood inversion of several boreholes with their own noise levels and the
  weighted-score construction of its exact targets (version 10, difficulty_increase, item 12);
- the conventions (reference temperature, history intervals, z_min cut in
  TVD, OLS covariance);
- all test cases.

No public code was copied.

## Data

Since version 10 (difficulty_increase, at the end of this file) the boreholes of `tests/step_6.py` are built by
a different generator (orthogonal-residual construction for `deglaciation_heat_flow_inversion`), and those of
`tests/step_7.py` and `tests/general.py` by the weighted-score construction of item 12 for the final function
`regional_deglaciation_inversion`; they are described there. The rest of this section describes the
boreholes and the generator of the `surface_heat_flow` tests of versions 5 to 11 (historical).

All data are **synthetic**. Real boreholes do not come with a known true heat
flow, and the tests need an analytic truth. There is **no external data
file**: each borehole case of `tests/step_6.py` and `tests/general.py`
regenerates its borehole itself, with fixed seeds and only numpy and math,
before calling `surface_heat_flow`. The generating parameters (the truth) are
constants in `tests/step_6.py` and `tests/general.py`, and the least-squares
targets of the noisy holes are recomputed inside each case from the
generator's own model terms (`numpy.linalg.lstsq`, pseudo-inverse
covariance). The solver never sees these files.

| Borehole | Seed | Length (MD) | Inclination | q0 (W m^-2) | T0 (°C) | g (K) | A (µW m^-3) | κ (m^2 s^-1) | Logging noise |
|---|---|---|---|---|---|---|---|---|---|
| noiseless | 1 | 1500 m, survey every 30 m | 35→44°, azimuth 180→196° | 0.041 | 5.2 | 6.5 | 0.6 | 1.2e-6 | none |
| noisy | 20260929 | 1800 m, every 25 m | 28→40°, azimuth 330→350° | 0.037 | 4.1 | 7.8 | 0.9 | 1.1e-6 | 5 mK |
| deep | 4242 | 2400 m, every 50 m | 8→15°, azimuth 90→110° | 0.052 | 3.6 | 5.0 | 1.8 | 1.3e-6 | 10 mK |

All three boreholes share the same history shape: intervals ending at 10, 100
and 120 ka, with departures (0, −1, +0.25), i.e. the Holocene, the last
glaciation and the Eemian. The z_min cut is at 150 m. The layers have
conductivities of 2.3–4.6 W m^-1 K^-1, typical of Abitibi volcanics,
intrusions and metasediments.

Why these data are representative:

- Heat flows, heat production and diffusivities are typical of the Superior
  Province.
- The trajectories reproduce the lift and swing of real NQ holes.
- The glacial amplitudes (5–8 K) and the 5–10 mK logging noise are
  realistic.

The generator works forward from the truth and shares no code with the task
solutions:

1. **TVD** comes from 64-point Gauss–Legendre integration of the vertical
   component of the great-circle tangent.
2. **R and S** come from adaptive quadrature of the layered profile (the
   copy in the test files uses the exact layer sums; see below).
3. **P** comes from its own erfc implementation.
4. **Noise:** seeded Gaussian noise is added to the temperatures.

The authoring generator (`generate_test_data.py`) is reproduced verbatim
below. It used adaptive quadrature for R and S and wrote an HDF5 file during
authoring. The copy embedded in the borehole cases of `tests/step_6.py` and
`tests/general.py` uses the exact layer sums instead of quadrature and needs
no data file. Both produce identical inputs: the maximum difference over
every array of the three boreholes is below 1e-11 (1.4e-14 when re-measured
in revision 10a). No target is computed with solution.py or
second_solution.py; both are compared with these targets (see the evidence
section).

```python
"""Synthetic inclined-borehole datasets for the borehole-heat-flow task.

Forward model, written independently of the task solution:
  * trajectory: the survey stations define a minimum-curvature path; TVD at
    any MD is obtained by numerically integrating the vertical component of
    the unit tangent, which on each segment rotates at constant rate on the
    great circle between the station directions (Gauss-Legendre, 64 nodes);
  * temperature: T(z) = T0 + q0 R(z) - A S(z) + g P(z) with R, S evaluated by
    adaptive quadrature of the layered conductivity profile and P from the
    half-space step-response solution;
  * Gaussian noise (seeded) is added to temperatures for the noisy cases.
Run `python generate_test_data.py` to rebuild harness/test_data.h5.
"""
from pathlib import Path
import numpy as np
import h5py
from scipy.integrate import quad
from scipy.special import erfc

YEAR = 365.25 * 86400.0
GL_X, GL_W = np.polynomial.legendre.leggauss(64)


def tangent(inc, azi):
    return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])


def tvd_numeric(md, inc_deg, azi_deg, m):
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    total = 0.0
    for i in range(len(md) - 1):
        a, b = md[i], min(md[i + 1], m)
        if b <= a:
            break
        t1, t2 = tangent(inc[i], azi[i]), tangent(inc[i + 1], azi[i + 1])
        beta = np.arccos(np.clip(t1 @ t2, -1, 1))
        L = md[i + 1] - md[i]
        s = 0.5 * (b - a) * GL_X + 0.5 * (b + a)
        f = (s - a) / L
        if beta < 1e-12:
            tz = np.full_like(s, t1[2])
        else:
            tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
        total += 0.5 * (b - a) * np.sum(GL_W * tz)
    return total


def k_of(z, tops, k):
    return k[np.searchsorted(tops, z, side="right") - 1]


def RS(z, tops, k):
    pts = [t for t in tops if 0 < t < z]
    R = quad(lambda x: 1 / k_of(x, tops, k), 0, z, points=pts or None, limit=200, epsabs=1e-12)[0]
    S = quad(lambda x: x / k_of(x, tops, k), 0, z, points=pts or None, limit=200, epsabs=1e-12)[0]
    return R, S


def P_of(z, t_years, shape, kappa):
    edges = np.concatenate(([0.0], np.asarray(t_years) * YEAR))
    out = 0.0
    for i, d in enumerate(shape):
        e_old = erfc(z / (2 * np.sqrt(kappa * edges[i + 1])))
        e_new = 0.0 if edges[i] == 0 else erfc(z / (2 * np.sqrt(kappa * edges[i])))
        out += d * (e_old - e_new)
    return out


def make(seed, survey, layers, A, kappa, q0, T0, g, hist_t, hist_shape, log_step,
         log_top, noise):
    rng = np.random.default_rng(seed)
    md, inc, azi = (np.asarray(v, float) for v in survey)
    tops_md, k = (np.asarray(v, float) for v in layers)
    log_md = np.arange(log_top, md[-1] + 1e-9, log_step)
    tops_tvd = np.array([tvd_numeric(md, inc, azi, m) for m in tops_md])
    z = np.array([tvd_numeric(md, inc, azi, m) for m in log_md])
    T = np.empty(z.size)
    for j, zz in enumerate(z):
        R, S = RS(zz, tops_tvd, k)
        T[j] = T0 + q0 * R - A * S + g * P_of(zz, hist_t, hist_shape, kappa)
    if noise > 0:
        T = T + noise * rng.standard_normal(T.size)
    return dict(survey_md=md, survey_inc=inc, survey_azi=azi, log_md=log_md, log_temp=T,
                layer_top_md=tops_md, layer_k=k, heat_production=A, kappa=kappa,
                hist_t_years=np.asarray(hist_t, float), hist_dT_shape=np.asarray(hist_shape, float))


def survey_curving(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
    azi = azi0 + (azi1 - azi0) * f**1.3
    return md, inc, azi


HIST_T = [1.0e4, 1.0e5, 1.2e5]          # years before present (interval ends)
HIST_SHAPE = [0.0, -1.0, 0.25]          # Holocene, last glaciation, Eemian

DATASETS = {
    # Abitibi-type hole, noiseless: recovered parameters must equal the truth
    "noiseless": dict(
        seed=1, survey=survey_curving(1500, 30, 35, 44, 180, 196, 2.0, 1),
        layers=([0, 180, 420, 610, 900, 1150, 1330], [2.9, 3.4, 2.6, 4.6, 3.1, 2.4, 3.8]),
        A=0.6e-6, kappa=1.2e-6, q0=0.041, T0=5.2, g=6.5, hist_t=HIST_T,
        hist_shape=HIST_SHAPE, log_step=5.0, log_top=20.0, noise=0.0),
    # same geology type, different hole, 5 mK logging noise
    "noisy": dict(
        seed=20260929, survey=survey_curving(1800, 25, 28, 40, 330, 350, 1.5, 2),
        layers=([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
        A=0.9e-6, kappa=1.1e-6, q0=0.037, T0=4.1, g=7.8, hist_t=HIST_T,
        hist_shape=HIST_SHAPE, log_step=5.0, log_top=15.0, noise=0.005),
    # near-vertical deep hole, higher heat production, 10 mK noise
    "deep": dict(
        seed=4242, survey=survey_curving(2400, 50, 8, 15, 90, 110, 1.0, 3),
        layers=([0, 260, 900, 1400, 2000], [3.6, 2.9, 3.2, 2.6, 3.5]),
        A=1.8e-6, kappa=1.3e-6, q0=0.052, T0=3.6, g=5.0, hist_t=HIST_T,
        hist_shape=HIST_SHAPE, log_step=10.0, log_top=30.0, noise=0.010),
}
Z_MIN = 150.0


def main():
    out = Path(__file__).resolve().parents[1] / "harness" / "test_data.h5"
    out.parent.mkdir(exist_ok=True)
    with h5py.File(out, "w") as h:
        for name, kw in DATASETS.items():
            d = make(**kw)
            g = h.create_group(name)
            for key, v in d.items():
                if np.ndim(v) == 0:
                    g.attrs[key] = v
                else:
                    g.create_dataset(key, data=v)
            g.attrs["z_min"] = Z_MIN
            g.attrs["true_q0"] = kw["q0"]
            g.attrs["true_T0"] = kw["T0"]
            g.attrs["true_amplitude"] = kw["g"]
            g.attrs["seed"] = kw["seed"]
    print("wrote", out)


if __name__ == "__main__":
    main()
```

## Ground-truth evidence for stored targets

**Step 1, true_vertical_depth.**

- Vertical and straight inclined holes are exact (MD·cos I).
- A uniform build from 0° to 90° over 900 m is a circular arc. Its TVD is
  R sin(MD/R) at every measured depth, including between stations, which is
  what rejects linear interpolation.
- The two three-dimensional cases (a build-and-turn segment, and a turn at
  constant inclination) were computed with the closed-form ratio factor. They
  were checked by 48-point Gauss–Legendre integration of the tangent and agree
  to 1e-9 m; the stored 8-decimal values agree with adaptive quadrature of
  the arc tangent to 5e-9 m (their rounding).
- A constant-azimuth segment with inclination linear in MD has the closed
  form (sin I(m) − sin I1)/k, evaluated in the test without cancellation
  (the array-query case and the three tiny-dogleg cases on a 20 km segment).
- Eight random three-dimensional surveys (seeds 1–8) are compared with
  adaptive quadrature (`scipy.integrate.quad`) of the down component of the
  great-circle tangent, computed inside each case.

**Step 2, layer_integrals.** The fixed cases are hand-computed closed forms;
one checks that R gives the harmonic, not the arithmetic, effective
conductivity. Six random layered columns (seeds 11–16) are compared with
adaptive quadrature (`scipy.integrate.quad`) of 1/k and z/k computed inside
each case. Since revision 10b, case 11 checks shallow depths (0 to 37 m) under a
deep layer stack against R = z/k1 and S = z²/(2k1), and R = S = 0 at the
surface (absolute 1e-12). R and S are elementary integrals: the second
solution (ramp superposition of the conductivity contrasts and integration by
parts, revision 10b item 14) and the reference give the same values up to
rounding, so neither is evidence for the other; the evidence is the closed
forms and the quadrature.

**Step 3, paleoclimate_perturbation.**

- A single step, a past pulse and equal consecutive steps are checked against
  the analytic erfc forms, computed in the test with `math.erfc`.
- The multi-step history was checked by numerical Duhamel convolution with
  the half-space impulse response z/(2√(πκτ³)) exp(−z²/4κτ) in log-time.
  The two agree to 1e-9 K; the stored 8-decimal values agree with the
  convolution to 5e-9 K (their rounding).
- Linearity in the departure and decay with depth are analytic properties.
- Since revision 10b, case 11 checks depths of 0.05 to 2.5 m under a recent
  two-step history (2 and 150 years) against the erfc forms (`math.erfc`).

**Step 4, fit_heat_flow.**

- Exact recovery of the generating parameters on noise-free synthetic
  profiles (including the four-reading design and the heat-production sign
  case).
- On a noisy profile, the estimates and covariance are compared with
  `numpy.linalg.lstsq` computed inside the test.
- The resolvability threshold uses a P column built in the test with
  prescribed scaled singular values (see the v5 error-contract section).

**Step 5, layered_paleoclimate_perturbation.** The targets and their
provenance are listed in the Version 8 section (now `tests/step_5.py`); five
of the stored 90-digit values were recomputed independently in revision 10a.
Since revision 10b the 50-layer column of case 6 (k from 1.5 to 5 W m^-1 K^-1,
20 intervals) has independent targets at six depths (30 m to 3000 m, the last
one in the bottom half-space); their computation is described in the revision 10b
section. Before revision 10b that column was only checked for self-consistency.

**Step 6 and whole task since version 10 (difficulty_increase).** Every borehole case constructs its data so
that the exact least-squares solution of the four-parameter problem is known (orthogonal residual), with P and
dP/dt_end from the branch-cut oracle or from erfc; see the version 10 (difficulty_increase) note, items 3-6.
The text below describes the `surface_heat_flow` tests of versions 5-9 and revisions 10a-10b (historical).

**Step 6 and whole task, versions 5-9 and revisions 10a-10b (historical: `surface_heat_flow`).**

- *noiseless*: the truth is the generator input. solution.py recovers
  q0 = 0.041 to 4e-15 W m^-2 and second_solution.py to 6e-17 W m^-2
  (re-measured in revision 10b).
- *noisy* and *deep*: **no value is taken from the reference solution.** The
  expected estimates are recomputed at test time inside each case of
  `tests/step_6.py` and `tests/general.py` (`_independent_fit`). They come
  from `numpy.linalg.lstsq`, with the covariance from the pseudo-inverse,
  applied to the generator's own model terms: Gauss–Legendre TVD, exact layer
  sums for R and S, and `math.erfc` for P. They agree with `solution.py` and
  `second_solution.py` to 1e-10 W m^-2 and 1e-7 K (re-measured in revision 10b:
  1e-14 W m^-2 on q0, 3e-13 K on T0, 3e-12 K on g, 2e-13 relative on the
  standard errors). These values are valid
  for the stated 1-D model: conduction only, horizontal layers, uniform κ,
  and OLS with n − 3 degrees of freedom.
- *n_used* is the count of readings whose generator TVD is ≥ z_min, by
  definition. The value 46 in step 6 case 5 / general case 5 (z_min applied
  in vertical depth) is the count of generator TVDs in [150, 350) m of the
  noisy hole (an MD cut would remove 40 readings).
- Since revision 10b: step 6 case 6 puts a reading at the collar (MD 0, hence
  TVD 0 on any trajectory) of a vertical hole and cuts at z_min = 0, so the
  reading sits exactly at the cut and every valid TVD method computes it
  exactly; n_used = 151 (all readings) by definition, and the noise-free
  T0, q0, g are the generating values. Step 6 case 4 and general case 4 add a
  vertical noise-free hole with six readings where z_min = 300 m keeps exactly
  4 (valid, generating values recovered) and z_min = 500 m keeps 3
  (ValueError). Step 6 case 0 and general case 0 compare the rms residual of
  the noisy hole with its 5 mK logging noise (expected σ√((n − 3)/n), relative
  scatter 1/√(2(n − 3)) = 4 % for n = 326; band 15 %; measured ratio 0.993).
- The same records (target, test, method, note, validity) were listed in the
  `target_provenance` field of the old-format problem.yaml; since revision 10a
  they are in the header comments of `tests/step_6.py` and `tests/general.py`.
- For the noisy and deep holes, the generator truth lies within 4σ of the
  estimates (checked in their cases). For the noisy hole, q0 differs by 1.1σ
  and g by 0.7σ; for the deep hole, by 0.6σ and 0.2σ.

**Limitations.**

- The model is 1-D, homogeneous in κ for the transient, and ignores
  topography and the temperature dependence of conductivity. These
  simplifications are stated in the problem and applied identically in the
  generator.
- The noisy targets are least-squares estimates, not the truth; noise moves
  them.

## Tolerance rationale and rejected incorrect approaches (historical: v7 tests and step numbers)

This section and its table were measured on the v7 test files (workflow = step 5 then); the current
per-case mutant results are in the revision 10b table (steps 1-5) and in the platform version 10 note (step 6
and whole task) at the end of this file. The table compares the reference with each mutant. Δq is in
mW m^-2 and Δg in K.

| Mutant | noiseless | noisy | deep | Caught by |
|---|---|---|---|---|
| measured depth used as depth | Δq −12.3, Δg −2.9 | Δq −9.1, Δg −2.6 | Δq −1.2, Δg −0.35 | general, all |
| linear TVD interpolation between stations | Δg +1.3e-3, Δq +2.0e-3 | Δg +2e-4 | Δg +1e-4 | general (noiseless), step 1 |
| balanced tangential (no ratio factor) | Δg +7e-5, Δq +1.6e-4 | Δg +3e-5 | Δg +7e-6 | general (noiseless: q0 tolerance 1e-8 W m^-2), step 1 |
| single arithmetic-mean conductivity | Δq +6.1, Δg +4.9 | Δg +2.6 | Δq +4.5, Δg +1.4 | general |
| heat production ignored | Δq −0.8, Δg −0.3 | Δq −1.2, Δg −0.4 | Δq −2.8, Δg −1.4 | general, step 4 |
| years not converted to seconds | P ≡ 0 → error | error | error | general, step 3 |
| erfc argument missing the factor 2 | Δq −7.2 | Δq −8.6 | Δq −3.5 | general, step 3 |
| wrong sign of the paleoclimate term | Δg −13 | Δg −15.6 | Δg −10 | general, step 3 |
| amplitude fixed at 1 (not estimated) | Δq −7.0 | Δq −6.8 | Δq −1.4 | general |
| z_min applied to MD instead of TVD | n_used differs | n_used differs | n_used differs | general |
| s² = RSS/n instead of RSS/(n−3) | sigmas −0.5 % | −0.5 % | −0.7 % | step 4 only |

The tolerances sit between numerical noise and the smallest mutant effects:

- The two valid pipelines agree to 1e-10 W m^-2 and 1e-7 K.
- The trajectory mutants move g by only 1e-5–1e-3 K and q0 by 1e-7–2e-6 W m^-2
  on the densely surveyed holes. That is why the noiseless tolerances are
  tight (1e-8 W m^-2, 1e-4 K); step 1 rejects these mutants independently.
- The noisy tolerances (2e-6 W m^-2 ≈ 0.3σ on q0; 2e-3 K ≈ 0.3σ on g) reject
  every physical mutant by a factor of more than 100.
- The degrees-of-freedom error is deliberately tested only in step 4. A
  0.5 % change is below the 2 % sigma tolerance of the integration tests,
  which admits any valid covariance evaluation (normal equations, QR, SVD).

## Error-contract tests (v5 additions; historical step numbers: workflow = step 5 in v5-v9)

Every stated ValueError has a test built by construction (no value from the reference):

- step 1: a dogleg of 180 deg - 5e-7 rad (opposite azimuths, inclinations 10 and 170 - 2.86e-5 deg)
  must be rejected (inside the 1e-6 rad band), a dogleg of 180 deg - 2e-6 rad must be accepted
  (outside the band; the near-half-circle path turns back up, so TVD(100) < TVD(40)); non-finite
  survey MD, inclination or azimuth are rejected before the ordering checks;
- step 2: a non-finite layer top ([0, NaN, 500]) is rejected;
- step 3: a non-finite history time or amplitude is rejected;
- step 4: the resolvability threshold is enforced from both sides: a P column built in the test as
  0.01 R plus a tiny component outside span{1, R}, scaled by the test's own singular values to a
  smallest scaled singular value of 3e-9 (rejected) and 3e-8 (fitted, finite results);
- step 5 (the workflow; step 6 since revision 10a) / whole task: a survey not starting at MD 0, a non-increasing survey, a non-increasing
  history and a NaN history time all propagate as ValueError through surface_heat_flow.

## Registered mutants (problem.yaml) (historical: v5-v7 test functions and step numbers)

Every mutant below is a complete implementation that contains one deliberate
scientific error. Each one fails the tests it targets, and the reference and
the second solution pass all of them.

| Where | Mutant | Error | Tests failed |
|---|---|---|---|
| step 1 | linear_tvd_between_stations | TVD interpolated linearly between stations | 3 of 20 |
| step 1 | balanced_tangential | ratio factor omitted | 3 of 20 |
| step 2 | arithmetic_conductivity | arithmetic mean k instead of resistances in series | 2 of 12 |
| step 2 | heat_production_integral_missing_half | S missing the factor 1/2 | 2 of 12 |
| step 3 | years_not_converted | times left in years | 6 of 17 |
| step 3 | erfc_missing_factor_2 | z/√(κt) instead of z/(2√(κt)) | 6 of 17 |
| step 3 | interval_sign_reversed | erfc(new) − erfc(old) | 7 of 17 |
| step 4 | heat_production_sign | +A·S instead of −A·S | 3 of 11 |
| step 4 | dof_n | s² = RSS/n | 1 of 11 |
| whole task | measured_depth_as_depth | MD used as depth | 7 of 12 |
| whole task | no_paleoclimate_correction | steady geotherm only | 4 of 12 |
| whole task | single_mean_conductivity | one arithmetic-mean layer | 4 of 12 |
| whole task | z_min_on_measured_depth | shallow cut applied to MD | 5 of 12 |

Undefined cases are rejected with ValueError, as the prompt states:
- a 180° dogleg between consecutive survey stations, where the arc is
  undefined;
- a rank-deficient design [1, R, P], where T0, q0 and g cannot be separated.

A test covers each case.

Version 7: the dogleg rejection band is inclusive (a dogleg exactly 1e-6 rad from 180 degrees is rejected, as the prompt says); the step 1 and step 2 tests assert the container types (ndarray with the query shape; tuple of two Python floats or of two ndarrays); step 5 (the workflow; step 6 since revision 10a) tests invalid layer conductivities (zero, negative, non-finite, wrong length) through the whole workflow; the overview no longer mentions how the task is graded.

## Version 8 (difficulty increase; v7 rollouts solved 8/8)

New step 6 (step 5 since revision 10a), `layered_paleoclimate_perturbation(z, layer_tops, layer_k, rho_c, t_years, dT)`: the
present-day paleoclimate departure in the layered conductivity column (diffusivity k_i / rho_c in each
layer, temperature and heat flux continuous at the tops), to 1e-8 K per kelvin of the largest |dT|, within
10 s for up to 1000 depths, 50 layers and 20 history intervals. The erfc formula of step 3 no longer
applies. Steps 1-5 and the workflow are unchanged.

- Reference: Laplace transform in time; in each layer F = a e^{-qz} + b e^{qz}, q = sqrt(s / kappa_i);
  the admittance Y = k F'/F is carried from the bottom half-space (Y = -kq) up to the surface with the
  tanh recursion, and F between and inside layers is written with e^{-2qh} factors only (no cancellation,
  no overflow), in logarithms; inversion of F/s on the fixed Talbot contour (Abate and Valko 2004, 32
  nodes). Accuracy about 1e-10 K per K against all targets; 0.4 s for 1000 depths, 50 layers and 20
  intervals.
- Second solution: one global linear system per Laplace node for the scaled layer coefficients, and the
  hyperbolic contour of Weideman and Trefethen (2007); agrees with the reference to about 1e-9.
- Targets (tests/step_6.py at the time, now tests/step_5.py): the erfc half-space for a uniform column; the image series of Carslaw and
  Jaeger (1959, sec. 12.8) for a layer over a half-space, alpha = (e1 - e2)/(e1 + e2) with the
  effusivities e = sqrt(k rho_c), evaluated in the test with math.erfc (a poorly conducting, a highly
  conducting and a thin cover); and for a five-layer Shield column with the glacial history of the task
  and a strongly contrasted three-layer column (k = 0.8 / 6.5 / 2.0, recent warming), values computed once
  with mpmath at 90 digits (transfer-matrix transform marched in mp, mpmath's own Talbot inversion);
  the reference agrees to 3e-11 and 2e-10 (4e-11 per K).
- Mutants: erfc with the mean diffusivity, erfc with the local diffusivity, and Gaver-Stehfest inversion
  of the exact transform in double precision (fails the 1e-8 K requirement); all fail.
- Comparisons between two computed outputs use twice the per-output tolerance.

Version 9 (content-check fixes on v8; step numbers of v8-v9: step 5 = surface_heat_flow, step 6 =
layered_paleoclimate_perturbation, swapped in revision 10a): step 6 no longer reads the clock; the prompt states the size it must
handle (1000 depths, 50 layers, 20 intervals in one call) and that case is now checked on values (50 equal
conductivities against the erfc half-space, varying conductivities against single-depth calls). Step 6 also
rejects two-dimensional and empty layer arrays in the tests. Step 4 fits a resolvable four-reading design
(smallest valid n). Step 5 has a vertical hole (TVD = MD exactly) with a reading exactly at z_min, which must
be kept (n_used = 131).

## Internal revision 10a (grading_fix: authoring-guide conformance; before the platform version 10 upload)

Naming: this note and the next were first headed "Version 10" and "Version 11". Both are internal revisions
written before the platform version 10 upload, which also contains the difficulty_increase note at the end of
this file; there has been no platform version 11. Elsewhere in this file "revision 10a" and "revision 10b" refer
to these two notes.

Audit of v9 against docs/CROWN_AUTHORING_GUIDE.md. The science, the reference algorithms, the targets and the
domain are unchanged; the v9 changes (no clock reads, the n = 4 fit, the inclusive reading at z_min, the 2-D and
empty layer arrays of the layered step) are kept. Each item gives the finding, its cause, the change and the
regression check.

1. Old task schema. Finding: problem.yaml used the pre-skeleton schema (problem_name, problem_id, id, task_id,
   title, subdomain, keywords, language, python_version, dependencies, test_dependencies, result_type,
   general_solution, main_problem, integration, top-level function_header/return_line, general_mutants,
   whole_task_mutants, main_mutants, target_provenance, problem_background_main; per step name, function,
   description, depends_on, inputs, outputs, valid_range, accuracy, errors, step_name, function_name).
   Cause: the task predates the current skeleton. Change: problem.yaml rewritten with the skeleton keys
   (domain, metadata, required_dependencies, problem_description_main, problem_io, general_tests,
   second_solution, mutants, edit_label, sub_steps with step_number, scaffold, solution, tests,
   step_description_prompt, function_header, return_line, step_background, n_test_cases, contract, mutants).
   The agent-visible content of the dropped fields (conventions, inputs and units, valid ranges, accuracy,
   errors) is now in problem_description_main, the step prompts and the function headers; the background is
   in background.md (new sections on the layered column and on the workflow); the target provenance is in
   the header comments of the test files and in this file. Regression: crown_check format OK.
2. Final step. Finding: the guide requires the final step to deliver the integrated result with the
   problem_io signature, but step 6 was layered_paleoclimate_perturbation while problem_io is
   surface_heat_flow (step 5). Change: the two steps swap numbers: step 5 is now
   layered_paleoclimate_perturbation (steps/, solution/, tests/, mutants/ renamed accordingly) and step 6 is
   surface_heat_flow, with step_dependencies [1, 2, 3, 4]. No function, prompt content or target changed.
   Regression: crown_check ref/second per step.
3. Test-case format. Finding: the tests used a pytest stand-in, _resolve()/importlib module lookup and
   _crown_run_all(). Change: every test function and every parametrized value is now a self-contained
   "# --- test case N ---" case (own imports, helpers and data generation; tools/selfcontain.py reported no
   problem) that calls the evaluated function by name. The ValueError parametrizations of each test function
   are merged into one case per file (a loop over the invalid inputs), so that computed values dominate each
   step: unmerged, step 5 would have had 15 and step 6 16 validation cases out of 22. The no-op _timed()
   wrapper left from v8 is gone. n_test_cases: 22, 11, 11, 7, 8, 7; tests/general.py has 6 cases. Every check
   of v9 is kept. Regression: every case run alone in a fresh process (crown_check).
4. Per-step solutions. Finding: solution/step_5.py (the workflow) redefined the functions of steps 1-4, and
   solution/step_1, 2, 4 carried unused copies of an erfc helper and SECONDS_PER_YEAR. Change: each
   solution/step_k.py defines only its own function and private helpers; the erfc helper is the private
   _erfc_array of step 3; solution.py defines every function once. Regression: crown_check ref (per-step
   context and solution.py on every step file).
5. Dependencies. Finding: second_solution.py imports scipy.integrate.quad, which was not declared. Change:
   required_dependencies = import numpy as np / import math / from scipy.integrate import quad.
6. Mutants. Finding: mutants were inline code; the inline step mutants were older copies that also lacked
   the v5 finite-value checks (two differences from the reference instead of one), and the workflow mutants
   redefined every function. Change: one file per mutant in mutants/, rebuilt from the current reference with
   exactly one scientific error; the step 6 mutants define only surface_heat_flow; the four whole-task
   mutants are complete solutions. Regression: crown_check mut (table below).
7. Exact float comparisons. Finding: assert_allclose(..., rtol=0) in step 1 and "== 0.0" branches in the
   target helpers of steps 1 and 5. Change: max-abs comparisons; the branches test "<= 0.0" on arguments that
   are non-negative (dogleg from atan2 of a norm, depths >= 0), same meaning.
8. Tolerances inconsistent with the stated accuracy (rule: prompt, docstring and tests state the same numbers;
   two computed outputs are compared with twice the single-output tolerance).
   - Step 1: assert_allclose(atol=1e-6) kept its default rtol = 1e-7, which allowed 6e-5 m at 600 m against
     the stated absolute 1e-6 m; now max-abs <= 1e-6 m. The array-call against scalar-call comparison used
     1e-9 m and now uses 2e-6 m. The 8-decimal three-dimensional targets were re-checked by adaptive
     quadrature of the arc tangent (agreement 5e-9 m).
   - Step 2: the harmonic-mean case compared 1000/R absolutely to 1e-9 (a relative 6e-10, tighter than the
     stated relative 1e-9); now R = 600 m^2 K W^-1 to a relative 1e-9. The absolute floor 1e-12 that the
     tests allow at z = 0 (where R = S = 0) is now stated in the prompt and the docstring.
   - Step 3: the 2-D case compares two computed outputs with 5e-6 K (twice 2.5e-6 K, largest |dT| = 2.5 K)
     instead of 2.5e-6 K; the other allclose checks are explicit max-abs checks with the same numbers.
   - Step 4: no accuracy was visible; the tests used 1e-8 K / 1e-11 W m^-2 for noise-free recovery and
     1e-9 K / 1e-12 W m^-2 against numpy.linalg.lstsq. Stated now: T0 and g within 1e-9 K, q0 within
     1e-12 W m^-2 of the exact least-squares solution, the standard errors within a relative 1e-6 and
     rms_residual within 1e-9 K; the noise-free checks use the same numbers. Normal equations, QR, SVD,
     pinv and lstsq agree to 1e-11 K and 1e-14 W m^-2 on these designs (condition number 9e3), so the
     stated numbers keep a margin above 100 for any stable method.
   - Step 6 / whole task: the accuracy (noise-free: 1e-5 K, 1e-8 W m^-2, 1e-4 K, rms below 1e-6 K; noisy:
     5e-4 K, 2e-6 W m^-2, 2e-3 K against the exact least-squares solution, standard errors 2 %, rms 1 %,
     n_used exact) was only in main_problem; it is now in the step 6 prompt, its function header and
     problem_io. No number changed.
9. Visible specification. Finding: the step 6 inputs, units, valid ranges and errors were only in hidden
   fields; the docstrings of steps 1-3 omitted the non-finite rejections stated in the prompts; step 4 did
   not state that n = 4 is valid or that every value is a Python float; step 4 declared dependencies on
   steps 1-3 although it calls no earlier function. Change: stated in the prompts and headers; step 4
   step_dependencies is [] (its prompt says it receives the arrays). Contracts with the seven keys were added
   to every step. The example amplitude in the step 6 input description is now g = 4 (it was 6.5, the
   generating amplitude of one test borehole).
10. Stated but untested. Finding: step 5 promised ValueError for a non-finite layer top, a non-finite history
    time and 2-D history arrays, and step 6 for a non-finite heat production, without a test. Change: these
    inputs were added to the merged ValueError cases (step 5 case 7, step 6 case 4); reference and second
    solution raise.
11. Controls. Finding: an all-zero stub passed the step 1 scalar case (TVD 0 at MD 0), and the step 3
    equal-steps case compared two outputs only (0 = 0 passes). Change: the step 1 case also checks a scalar
    query at MD 64 m (64 cos 30 deg), and the step 3 case checks the single step against
    -2 erfc(z / (2 sqrt(kappa t))). Regression: crown_check controls pass no case.
12. Metadata. subfield, tags, expert_time_estimate_hours (10: the v8 layered step made the former 4 h
    unrealistic), relevant_experience placeholder for the author, difficulty, solution and verification
    explanations, author, affiliation and edit_label grading_fix. The second-solution docstring now says that
    its covariance comes from the inverse triangular factor of QR (not from a pseudo-inverse) and describes its
    layered method.
13. Target-provenance text of this file (independent verification after the conformance pass). Finding: the
    Data section still said that tests/general.py alone regenerates the boreholes, that the least-squares
    targets are stored constants there, and that "the stored targets were then computed with both solutions";
    the evidence section cited the pytest function test_z_min_is_applied_in_vertical_depth and said the
    noiseless reference recovers q0 to 1e-16 W m^-2; the version 8 note located the layered targets in
    tests/step_6.py; the evidence for steps 1 and 2 named only the closed forms. Cause: these paragraphs were
    written for the v7-v9 test files (HDF5 era, pytest functions, the old step numbers) and were not updated
    with the test files in this version. In the current files nothing is stored for the workflow: every
    borehole case of tests/step_6.py and tests/general.py regenerates its hole and recomputes the noisy-hole
    targets with _independent_fit (numpy.linalg.lstsq, pseudo-inverse covariance) on the generator's own
    terms, and no target comes from solution.py or second_solution.py. Change (text only; no test, target,
    prompt or code changed): the Data paragraph names both test files and says the noisy-hole targets are
    recomputed inside each case; "No target is computed with solution.py or second_solution.py; both are
    compared with these targets" replaces the sentence on stored targets; the evidence section cites step 6
    case 5 / general case 5, adds the random-survey and random-column quadrature targets of steps 1 and 2,
    the step 4 noise-free and threshold constructions and a pointer to the step 5 evidence, gives the
    re-measured numbers, and restricts the 4-sigma statement to the noisy and deep holes (the noiseless
    standard errors are round-off, about 1e-16, and that hole is checked against the truth directly); the v5, v7, v8 and v9 notes mark their old step numbers (workflow = step 5 then;
    layered step = step 6 in v8-v9; tests/step_6.py of v8 is now tests/step_5.py). Regression: every number
    quoted was re-measured in a clean process: the embedded generator against the authoring generator above
    (scipy quad for R and S) differs by at most 1.4e-14 over every input array of the three boreholes;
    solution.py and second_solution.py agree with _independent_fit to 1e-14 W m^-2 (q0), 4e-13 K (T0) and
    3e-12 K (g), and recover the noiseless q0 = 0.041 to 4e-15 and 3e-16 W m^-2; the noisy hole lies 1.1
    sigma (q0) and 0.7 sigma (g) from the truth; 46 generator TVDs of the noisy hole lie in [150, 350) m
    against 40 MDs; the step 1 three-dimensional and the step 3 multistep 8-decimal targets agree with
    adaptive quadrature and with the Duhamel convolution to 5e-9 (their rounding). crown_check rerun in all
    modes (table below).

Additional verification run for this version: five stored 90-digit targets of the layered step (Abitibi
column at 75, 650 and 1500 m; contrasted column at 61 and 400 m) were recomputed with an independent forward
transfer-matrix propagation of (F, k dF/dz) from the surface in mpmath at 40 digits with mpmath's Talbot
inversion; they agree with the stored values to 5e-17 K. solution.py and second_solution.py were each run
twice in clean processes (and once with one BLAS/OpenMP thread) on the three synthetic boreholes, the
1000-depth / 50-layer / 20-interval layered case and a three-dimensional survey: identical outputs (SHA-256 of
the results) in every run.

crown_check (tools/crown_check.py, every case alone in a fresh process, -j 2), final run of revision 10a
(rerun in all modes after item 13, summary ALL OK; the clean-process repeat runs of both solutions above were
also repeated and gave identical SHA-256 digests):

| Mode | Result |
|---|---|
| format | OK |
| REF steps 1-6, general (solution.py) | 22/22, 11/11, 11/11, 7/7, 8/8, 7/7, 6/6 |
| SECOND steps 1-6, general | 22/22, 11/11, 11/11, 7/7, 8/8, 7/7, 6/6 |
| SHIFT +-1e-12 | all cases pass in every step |
| CONTROL None / 0.0 | 0 cases pass in every step and in general |

| Where | Mutant | Cases failed |
|---|---|---|
| step 1 | linear_tvd_between_stations | 15 of 22 |
| step 1 | balanced_tangential | 12 of 22 |
| step 2 | arithmetic_conductivity | 9 of 11 |
| step 2 | heat_production_integral_missing_half | 9 of 11 |
| step 3 | years_not_converted | 8 of 11 |
| step 3 | erfc_missing_factor_2 | 8 of 11 |
| step 3 | interval_sign_reversed | 9 of 11 |
| step 4 | heat_production_sign | 4 of 7 |
| step 4 | dof_n | 1 of 7 (the covariance case) |
| step 5 | mean_diffusivity_halfspace | 4 of 8 |
| step 5 | local_diffusivity_halfspace | 4 of 8 |
| step 5 | stehfest_double_precision | 6 of 8 |
| step 6 | measured_depth_as_depth | 6 of 7 |
| step 6 | no_paleoclimate_correction | 5 of 7 |
| step 6 | single_mean_conductivity | 6 of 7 |
| step 6 | z_min_on_measured_depth | 5 of 7 |
| whole task | whole_task_measured_depth_as_depth | 6 of 6 |
| whole task | whole_task_no_paleoclimate_correction | 4 of 6 |
| whole task | whole_task_single_mean_conductivity | 4 of 6 |
| whole task | whole_task_z_min_on_measured_depth | 5 of 6 |

The mutant table of "Registered mutants (problem.yaml)" above uses the v5-v7 test functions and step
numbers (workflow = step 5 then); this table replaced it in revision 10a and is itself superseded by the
revision 10b table below. Every section before "Internal revision 10a" uses the old step numbering (v5-v7: workflow =
step 5; v8-v9: workflow = step 5 and layered step = step 6); since revision 10a the layered step is step 5
and the workflow step 6.

## Internal revision 10b (grading_fix: authoring-guide conformance, pass 2; before the platform version 10 upload)

Second conformance pass on revision 10a (rules on whole-task tests, hidden inputs, truthfulness, independent
second solution, current-reference mutants, stated/tested consistency), with the findings of the independent
verifier of revision 10a. Revision 10a is kept (self-contained cases, no clock reads, no exact float equality,
contracts and metadata). Each item gives the finding, its cause, the change, the regression test and what was
rerun. Domains, tolerances and the science are unchanged; the visible text gains one accuracy statement
(item 6) and one changed sentence (items 7 and 11).

1. Second solution, half-space perturbation near the surface. Finding: second_solution.py integrated the
   Duhamel convolution in log-time from max(lo, 1e-3 yr), so the response to the latest interval was cut off
   for 0 < z <~ 1.5 m, inside the promised domain z >= 0 (at z = 0.05, 0.3 and 1.0 m under a 2-year interval
   of 1.2 K the error was 1.0 K, 0.28 K and 8e-5 K against 1.2e-6 K allowed). No step 3 case had a small
   positive depth. Cause: the lower limit of the log-time quadrature was a fixed time instead of scaling with
   z^2/kappa. Change: each interval is integrated from its own start (tau = 0 for the first) in the similarity
   variable eta = z/(2 sqrt(kappa tau)), in which the impulse response z/(2 sqrt(pi kappa tau^3))
   exp(-z^2/(4 kappa tau)) dtau becomes (2/sqrt(pi)) exp(-eta^2) deta, by adaptive quadrature truncated at
   eta = 40 (exp(-1600)). Regression: new step 3 case 11 (z = 0.05, 0.3, 1.0, 2.5 m; history 2 and 150
   years; targets from math.erfc): the revision 10a second solution fails it, the new one passes. Sweep against
   solution.py over 200 random histories (t from 1e-3 to 1e6 years, kappa from 3e-8 to 1e-5 m^2 s^-1, z from
   1e-300 to 3e4 m, including 0.01 to 1.5 m): largest difference 6.4e-16 K per K.
2. Second solution copied the strategy of test oracles (rule D). Finding: its TVD integrated the great-circle
   tangent with 48-point Gauss-Legendre quadrature, the strategy of the borehole generator embedded in
   tests/step_6.py and tests/general.py (64 points), and its R and S called scipy.integrate.quad on 1/k and z/k
   with the layer tops as break points, almost verbatim the step 2 oracle _quad_RS. Cause: the second solution
   predates the embedding of these generators in the test files. Change: TVD by circular-arc geometry: on a
   segment of length L and dogleg beta the path is a circle of radius L/beta in the plane of t1 and
   w = (t2 - t1 cos beta)/sin beta, so the rise after an arc length s is
   s [sinc(phi) t1z + (s/(2L)) sinc^2(phi/2) (t2z - t1z cos beta)/sinc(beta)], phi = beta s/L, a closed form
   that is exact for beta = 0 and free of cancellation for tiny doglegs (solution.py uses the ratio-factor
   formula; the step 1 oracle integrates the same tangent numerically). R and S by cumulative tables at the
   layer tops completed with the midpoint rule in the layer that holds z (exact for 1/k and z/k); this R and
   S route turned out to be the reference's own per-layer terms reorganized and was replaced in item 14.
   Regression: crown_check second (every case of every step and of general passes); sweeps against
   solution.py: TVD within 3.6e-12 m over 300 random surveys (segments of 1 m to 3 km, about 60 of them with
   doglegs of order 1e-9 rad), R and S (midpoint-rule version, superseded) within a relative 6.7e-16 over
   3000 random columns.
3. Reference S at shallow depth (found in this pass). Finding: solution.py and solution/step_2.py added
   (b**2 - a**2)/(2k) for every layer; for a layer below z, b = a = top, but numpy's scalar a**2 and array
   b**2 can differ by one ulp (on this platform for tops such as 995.3, 1358.1 or 1946.2 m), which left a
   residue of about 1e-10 m^3 K W^-1 in S: S(0) = 8.9e-11 against the stated absolute 1e-12, and a relative
   error 3.7e-6 at z = 0.01 m against the stated 1e-9. No case had a shallow depth under such a top. Cause:
   unfactored difference of squares. Change: (b - a)(b + a)/(2k), exactly zero for the layers below z, in
   solution.py, solution/step_2.py and every mutant that contains the reference layer_integrals (the two step
   2 mutants and the four whole-task mutants, so that each stays the current reference with one error).
   Regression: new step 2 case 11 (tops 0, 120, 995.3, 1358.1, 1946.2 m; z = 0, 0.01, 0.5, 2, 37 m and a
   scalar z = 0; R = z/k1 and S = z^2/(2 k1) to a relative 1e-9, R = S = 0 within 1e-12): the revision 10a
   reference fails it, the new one passes. The workflow numbers below were re-measured after the change.
4. Inclusive z_min boundary (task note; verifier advisory). Finding: step 6 case 6 kept a reading at TVD 200
   m = z_min of a vertical hole; a valid TVD within the stated 1e-6 m can be one ulp below (5-node
   Gauss-Legendre gives 199.99999999999997), so the case was a floating-point knife-edge. Cause: the case
   relied on TVD = MD being computed exactly. Change: the collar reading (MD 0, TVD 0 on every trajectory, the
   origin of the TVD) of a vertical noise-free hole with z_min = 0: n_used = 151 (every reading), and T0, q0,
   g within the noise-free limits of the generating values (6.0 deg C, 0.045 W m^-2, 5.5 K). The prompt
   statement "a reading exactly at z_min is kept" is unchanged. Regression: a step 1 written with 5-node
   Gauss-Legendre quadrature fails the old case and passes the new one; a new step 6 mutant
   z_min_cut_strict (TVD > z_min) fails case 6 (and only it); reference and second solution pass.
5. Independent target for the 50-layer column (task note; verifier advisory). Finding: step 5 case 6 checked
   the column with k rising from 1.5 to 5 W m^-1 K^-1 only by array-against-scalar self-consistency, so the
   mean- and local-diffusivity mutants passed it. Change: six depths (z[j] for j = 10, 137, 300, 500, 700,
   999 of the case's 1000-depth grid, 30 m to 3000 m, the last in the bottom half-space) are compared with
   values computed once in mpmath by a method shared with neither solution: forward propagation of
   (F, k dF/dz) from the surface with 2 x 2 transfer matrices [[cosh qh, sinh qh/(kq)], [kq sinh qh, cosh qh]],
   the unknown surface flux fixed by the decay condition k F' = -k q F at the top of the bottom half-space,
   and the working precision set from the growth of the exponentials (2 sum Re(q) h / ln 10 digits lost,
   plus a margin: 267, 334 and 392 digits); inversion by the fixed Talbot contour of degree 60 and 90 (the
   two agree to the 30 printed digits) and by Gaver-Stehfest with 60 terms (agreement 2e-16). Twenty
   significant digits are stored inline. The self-consistency check is kept. Regression: solution.py agrees
   with the targets to 3.0e-11 K, second_solution.py to 1.8e-10 K (tolerance 1e-8 x max|dT|, about 1e-8 K); the
   mean- and local-diffusivity mutants now fail case 6 (5 of 8 cases each). The script (about 120 lines:
   transform, growth estimate, Talbot and Stehfest sums, superposition of the 20 step responses) is not
   shipped with the task; its algorithm is fully described here.
6. Error budget of the workflow (task note; verifier advisory). Finding: the noise-free accuracy of step 6
   (T0 1e-5 K, q0 1e-8 W m^-2, g 1e-4 K) is tested with the candidate's own steps 1-4 in the whole-task
   cases, but step 3 promises only 1e-6 K per K. Worst-case propagation (L1 norms of the rows of the
   pseudo-inverse of [1, R, P] times the largest per-reading temperature error) on the noise-free holes:

   | Hole (readings used) | steps 1 + 2 at their limits: T0 / q0 / g | step 3 at 1e-6 K per K: T0 / q0 / g |
   |---|---|---|
   | general noiseless (263) | 1.8e-7 K / 2.6e-9 W m^-2 / 2.0e-6 K | 2.0e-5 K / 2.8e-7 W m^-2 / 2.2e-4 K |
   | step 6 vertical, z_min = 0 (151) | 8.1e-8 / 8.5e-10 / 7.7e-7 | 1.1e-5 / 1.1e-7 / 1.0e-4 |
   | step 6 vertical, 4 readings | 2.1e-7 / 7.8e-10 / 1.1e-6 | 2.9e-5 / 1.1e-7 / 1.5e-4 |

   (step 1: 1e-6 m of TVD times the largest |dT/dz| plus the shift of the layer tops; step 2: a relative 1e-9
   on q0 R and A S; step 4 adds 1e-9 K and 1e-12 W m^-2.) Steps 1, 2 and 4 stay inside the noise-free limits;
   step 3 at its limit does not (a smooth error 1e-6 exp(-z/500) K per K still gives 3.5e-9 W m^-2). With P
   accurate to 1e-8 K per K the step 3 column drops by 100 and every total is inside the limits with a
   margin of at least 1.8 on q0 (the erfc closed form gives about 1e-15). The noisy-hole limits hold even
   with step 3 at its limit (1.6e-7 W m^-2 against 2e-6). Cause: the step 3 accuracy was written for the step
   on its own. Change (no tolerance changed, rule G): the step 6 prompt, its function_header, problem_io, the
   scaffold docstring and the step 6 contract now state "The noise-free limits presume P accurate to 1e-8 K
   per kelvin of the largest |hist_dT_shape| or better (an error at the 1e-6 K per K limit of step 3 can move
   q0 by a few 1e-7 W m^-2); the accuracies of steps 1, 2 and 4 propagate within these limits." Regression:
   crown_check format (header = scaffold = problem_io checked by a script as well).
7. SECONDS_PER_YEAR in the step 3 header (verifier advisory). Finding: the visible function_header said
   "Times in years are converted with SECONDS_PER_YEAR", a name defined only in the scaffold module. Change:
   "Times in years are converted with 1 year = 365.25 x 86400 s." in the header and the scaffold docstring.
   Regression: crown_check format; header and scaffold identical.
8. difficulty_explanation numbers (verifier advisory). Finding: "about 3e-6 K per K" (Gaver-Stehfest) and
   "0.04 to 0.2 K per K" (erfc with mean or local diffusivity) did not match the shipped mutants. Cause: the
   numbers were estimated on the v8 columns before the case set was final. Change: re-measured per test
   column as max|mutant - reference| / max|dT| (mean / local / Stehfest): layer over half-space (2, 5, 400)
   0.097 / 0.129 / 2.1e-6; (6, 0.8, 400) 0.246 / 0.299 / 6.4e-6; (1.2, 4.4, 35) 0.071 / 0.064 / 2.6e-6;
   Abitibi 0.038 / 0.052 / 2.9e-6; contrasted 0.178 / 0.206 / 2.2e-6; 50 layers 0.073 / 0.039 / 6.6e-6;
   shape case 0.124 / 0.161 / 2.8e-6. The text now says "2e-6 to 7e-6 K per K" and "0.04 to 0.3 K per K".
9. Whole-task cases (rule A). Finding: general case 4 only checked ValueError (no computed result) and
   general case 0 only the output contract; the minimum-size boundary of the workflow was tested only with 0
   readings left (verifier advisory). Change: general case 4 (and step 6 case 4) add a vertical noise-free
   hole with six readings: z_min = 300 m keeps exactly 4 readings (the smallest valid fit; n_used = 4 and the
   generating T0, q0, g within the noise-free limits) and z_min = 500 m keeps 3 (ValueError). General case 0
   (and step 6 case 0) check that the rms residual of the noisy hole matches its 5 mK logging noise (expected
   sigma sqrt((n - 3)/n), relative scatter 1/sqrt(2(n - 3)) = 4 % for n = 326; band 15 %; measured ratio
   0.993). Every general case now calls surface_heat_flow and asserts a computed scientific result (crown_check
   format reports no general case without the final function). Regression: crown_check ref/second/mut/shift
   on general; the whole-task mutants fail 5 or 6 of the 6 general cases.
10. Sources (verifier advisory). Finding: Clauser (1984) was cited in the step 5 step_background without being
    checked or listed; the methods of the step 5 reference, second solution, targets and mutant were not in the
    sources table. Change: rows and full references for Carslaw & Jaeger (1959, ch. 12, image series), Abate &
    Valko (2004), Weideman & Trefethen (2007) and Stehfest (1970); "Clauser 1984" removed; the magnitude
    "up to tens of percent near strong contrasts" is no longer attributed to Beck (1977) in the step 5
    background and in background.md, and rests on the exact layered solutions of the tests (4 to 30 % of the
    largest surface departure in the test columns, item 8).
11. Visible text. Besides items 6 and 7, solution_explanation now gives the factored S term and
    verification_explanation describes the current second-solution methods, the new targets and cases and the
    strict-cut mutant. No prompt exposes a test input: the only sizes stated (1000 depths, 50 layers, 20
    intervals per call in step 5) are the stated domain limits, and the example amplitude of step 6 (g = 4) is
    used by no case.
12. Step 1 comments (verifier advisory): the tiny-dogleg cases 10 and 11 repeated the 8e-10 rad comment of
    case 9; each now states its own bend (sin(30 deg) dinc L / 2 = 4e-6, 1.5e-4 and 1e-2 m), and the helper
    _mp_build_tvd (no mpmath) is renamed _exact_build_tvd. No assertion changed.
13. Text of this file. The Data and evidence findings of the verifier (stored least-squares targets,
    "computed with both solutions", the pytest name, the tests/step_6.py location of the layered targets) were
    already corrected by item 13 of revision 10a; the evidence section now also lists the revision 10b cases and
    re-measured numbers. The tolerance-rationale, error-contract and registered-mutant sections are labelled as
    historical with the versions and step numbers they used, and the revision 10a crown_check table is marked as
    the revision 10a run. Completion of the revision 10a notes: item 5 (dependencies), cause: quad was imported
    by second_solution.py only and never declared; regression: crown_check second runs with exactly the
    declared imports. Item 7 (float comparisons), cause: pytest-era assert_allclose(rtol=0) and "== 0.0"
    shortcuts; regression: crown_check format (no exact-equality pattern) and ref/second/shift. Item 9
    (visible specification), cause: the old schema kept inputs, ranges and errors in fields the agent does
    not see; regression: crown_check format and the header = scaffold comparison. Item 10 (stated but
    untested), cause: the error lists grew in v5-v9 without matching invalid inputs; regression: the merged
    ValueError cases pass for the reference and the second solution. Item 12 (metadata), cause: the old
    schema had no such fields; regression: crown_check format (metadata keys).
14. Second solution R and S were the reference's arithmetic (rule D; blocking finding of the independent
    verifier after this pass). Finding: the cumulative tables of item 2 summed h/k and
    h (t_i + t_{i+1})/(2k) per layer and completed the layer that holds z with dz/k and
    dz (t_j + z)/(2k); in exact arithmetic these are term by term the reference's (b - a)/k and
    (b - a)(b + a)/(2k), so the "different method" for step 2 was the reference reorganized, and
    verification_explanation listed it among the different methods without a caveat. Cause: item 2 removed
    the copy of the quadrature oracle but replaced it with the reference's per-layer partition; a
    different exact route exists and was not used. Change: second_solution.py layer_integrals now writes
    the resistivity profile as a superposition of steps at the layer tops, 1/k(z') = sum_i c_i H(z' - t_i)
    with c_0 = 1/k_0 and c_i = 1/k_i - 1/k_{i-1}, so R(z) = sum_i c_i (z - t_i)_+ (a sum of ramps; no layer
    thickness and no per-layer partition), and S by integration by parts, S(z) = z R(z) - int_0^z R dz',
    with the integral of the continuous, piecewise-linear R by the trapezoid rule on the layer tops above z
    and on z (exact; no z'/k integrand is formed). The route shares no formula with solution.py (per-layer
    clipped sums), with the step 2 oracle (scipy quad of 1/k and z/k) or with the borehole generators of
    tests/step_6.py and tests/general.py (per-layer sums of (b - a)/k and (b^2 - a^2)/(2k)). Because R and
    S are elementary integrals, any exact route gives the same values up to rounding; the independent
    evidence for step 2 remains the hand-computed closed forms (cases 0-3 and 11) and the quadrature oracle
    (cases 4-9), and verification_explanation now says so. The module docstring of second_solution.py
    describes the new route. Regression: crown_check second (every case of every step and of general
    passes, step 2 12/12, including case 11 at z = 0.01 m under the deep stack, where R = c_0 z and
    S = z R - z R/2 exactly); a sweep over 1500 random columns (1 to 11 layers, thicknesses 1 cm to about
    3 km, conductivities 0.1 to 32 W m^-1 K^-1, depths 0, below 1 cm, inside the column, at every top and up
    to 3 km below the last top) against the exact rational layer sums of the float inputs (Python fractions):
    largest relative error 6.5e-14 (R) and 1.1e-13 (S) for the second solution (3.9e-16 and 5.7e-16 for
    solution.py), against the stated 1e-9. The whole-task numbers of the second solution were re-measured
    (below) and the clean-process repeat runs redone.
15. Re-examination of the item 14 finding (blocking finding of the independent verifier, rechecked after a
    restart of the working container; no file other than this one changed in item 15). Finding as reported:
    second_solution.py built S from S_top = cumsum(h (t_i + t_{i+1})/2 / k) and
    S = S_top[j] + dz (t_j + z)/(2 k_j), the reference's per-layer term reorganized, and
    verification_explanation presented it as a different method without a caveat. Confirmed: that code is
    the item 2 version, and in exact arithmetic its terms are, layer by layer, the reference's (b - a)/k and
    (b - a)(b + a)/(2k), so the finding was real for that version. Cause: as in item 14. Status: the item 14
    replacement is the code that ships, and it is the route the verifier proposes (R as the ramp sum
    sum_i c_i (z - t_i)_+ of the contrasts c_i of 1/k at the layer tops, S = z R - int_0^z R dz' with the
    integral of the piecewise-linear R by the trapezoid rule on the tops above z and on z);
    verification_explanation states that R and S are elementary integrals whose independent evidence is the
    hand-computed closed forms and the scipy quadrature oracle of step 2, and the module docstring of
    second_solution.py describes the route. Because the route is now different, the limitation is not kept
    as an open issue. Checks rerun for item 15:
    * crown_check, all modes (every case alone in a fresh process, one BLAS/OpenMP thread, -j 2): SUMMARY ALL
      OK, with exactly the counts of the table below (second solution 22/12/12/7/8/7 and general 6/6).
    * Exact-rational sweep, new random sample (seed 7, 1500 columns of 1 to 11 layers, thicknesses 1 cm to
      about 3 km, conductivities 0.1 to 32 W m^-1 K^-1, depths 0, 1e-4 to 1e-2 m, eight random depths in the
      column, every top and one depth up to 3 km below the last top), against the layer sums of the float
      inputs in Python fractions: largest relative error 3.9e-14 (R) and 1.3e-13 (S) for the second
      solution, 4.8e-16 and 4.7e-16 for solution.py, against the stated 1e-9 (the item 14 sample gave
      6.5e-14 and 1.1e-13); R = S = 0 within 1e-12 at z = 0 for both.
    * Conductivity contrast (found in this re-examination). The ramp sum and z R - int R subtract terms of
      the size of z max(1/k) to obtain R or S, so the rounding error of the second solution grows with the
      contrast kmax/kmin (the reference adds non-negative terms only and has no such growth). Measured over
      300 random columns of 2 to 50 layers per contrast, conductivities in two families a factor kmax/kmin
      apart, largest relative error R / S: 1e2: 3.7e-14 / 1.4e-13; 1e3: 3.1e-13 / 6.9e-13; 1e4:
      9.9e-13 / 5.0e-12; 1e5: 5.4e-12 / 3.0e-11; 1e6: 1.2e-11 / 1.1e-10. Rock conductivities span about 0.1 to
      10 W m^-1 K^-1 and the test columns 1 to 6, so the second solution meets the stated 1e-9 on every case
      and on every geological column; for contrasts far beyond 1e6, which no case uses, its 1e-9 is not
      established. This bounds only the second solution, not the task's domain or tolerances.
    * Repeatability: R and S of both solutions on 200 random columns (50 depths each) gave identical SHA-256
      digests in two runs with one BLAS/OpenMP thread and one run with two.
    * Whole-task numbers re-measured from the generators of general cases 1-3: they equal the "Re-measured
      numbers (after item 14)" below (noiseless q0 3.75e-15 W m^-2 for solution.py, below 1e-16 for the
      second solution; noisy and deep holes q0 9.8e-15 W m^-2, T0 3.2e-13 K, g 2.6e-12 K, standard errors
      1.8e-13 and rms 4.6e-14 relative at most; truth 1.10 / 0.68 sigma and 0.60 / 0.22 sigma).
    Regression test: crown_check second (step 2 12/12, including the closed-form cases 0-3 and 11 and the
    quadrature cases 4-9) and the sweeps above. No test, tolerance, prompt or code changed in item 15.

Not changed, with reasons (also reported to the lead): step 5 (layered perturbation) still does not feed
the workflow, which keeps the half-space correction of step 3 (making the workflow layered would change the
science and every whole-task target, so it belongs to a repair or difficulty_increase version); the noisy-hole
standard errors keep their 2 % tolerance, so the RSS/n mutant is caught only by step 4 case 2 (a scientifically
justified tighter tolerance is a decision for a later version); a stub that always raises ValueError passes
the merged validation case of each step (1 case per step, inherent to merging them); relevant_experience stays
the placeholder for the author.

Clean-process repeat runs: solution.py and second_solution.py were run twice with one BLAS/OpenMP thread and
once with two BLAS threads on the three synthetic boreholes, the 1000-depth / 50-layer / 20-interval column
and a three-dimensional survey: identical SHA-256 digests of the outputs in all three runs. After item 14 the
runs on the three boreholes and on 200 random layer columns (50 depths each) were repeated in the same way
(two runs with one thread, one with two): identical digests. Re-measured numbers (after item 14): noiseless
q0 recovered to 3.7e-15 (solution.py) and below 1e-16 W m^-2 (second_solution.py; 6.2e-17 before item 14);
noisy and deep holes against _independent_fit (largest over both solutions): q0 1e-14 W m^-2, T0 3.2e-13 K,
g 2.6e-12 K, standard errors 1.8e-13 relative, rms 4.6e-14 relative (1.9e-13 and 6e-14 before item 14);
generator truth 1.10 sigma (q0) and 0.68 sigma (g) from the noisy estimate, 0.60 and 0.22 sigma for the deep
hole; the two layered solvers agree to 4.1e-10 K over the 1000 depths of the 50-layer column.

crown_check (tools/crown_check.py, every case alone in a fresh process, one BLAS/OpenMP thread, -j 2), final
run of revision 10b (after item 14, repeated unchanged for item 15 on the shipped files), all modes, summary
ALL OK:

| Mode | Result |
|---|---|
| format | OK (no problem; every general case calls surface_heat_flow) |
| REF steps 1-6, general (solution.py) | 22/22, 12/12, 12/12, 7/7, 8/8, 7/7, 6/6 |
| SECOND steps 1-6, general | 22/22, 12/12, 12/12, 7/7, 8/8, 7/7, 6/6 |
| SHIFT +-1e-12 (step outputs; every evaluated function in general) | all cases pass |
| CONTROL None / 0.0 | 0 cases pass in every step and in general |

| Where | Mutant | Cases failed |
|---|---|---|
| step 1 | linear_tvd_between_stations | 15 of 22 |
| step 1 | balanced_tangential | 12 of 22 |
| step 2 | arithmetic_conductivity | 10 of 12 |
| step 2 | heat_production_integral_missing_half | 10 of 12 |
| step 3 | years_not_converted | 9 of 12 |
| step 3 | erfc_missing_factor_2 | 9 of 12 |
| step 3 | interval_sign_reversed | 10 of 12 |
| step 4 | heat_production_sign | 4 of 7 |
| step 4 | dof_n | 1 of 7 (the covariance case) |
| step 5 | mean_diffusivity_halfspace | 5 of 8 |
| step 5 | local_diffusivity_halfspace | 5 of 8 |
| step 5 | stehfest_double_precision | 6 of 8 |
| step 6 | measured_depth_as_depth | 6 of 7 |
| step 6 | no_paleoclimate_correction | 6 of 7 |
| step 6 | single_mean_conductivity | 7 of 7 |
| step 6 | z_min_on_measured_depth | 5 of 7 |
| step 6 | z_min_cut_strict | 1 of 7 (case 6) |
| whole task | whole_task_measured_depth_as_depth | 6 of 6 |
| whole task | whole_task_no_paleoclimate_correction | 5 of 6 |
| whole task | whole_task_single_mean_conductivity | 6 of 6 |
| whole task | whole_task_z_min_on_measured_depth | 5 of 6 |

## Version 10 (difficulty_increase) — platform version 10

This note is numbered after the platform versions: version 9 was the last upload. The two notes above,
internal revisions 10a and 10b (first headed "Version 10" and "Version 11"), are conformance revisions written
before this one; all three are part of the platform version 10 upload. Steps 1-5 (prompts,
headers, references, second solution, tests, mutants) are unchanged; items 1-11 describe the new step 6
(round 1), and item 12 the regional final step 7 added in a second round before the upload.

1. Finding. The platform reported version 9 "Too easy": 8 of 8 rollouts solved (strong model, maximum effort,
   4 h, mean score 1.0). The content check had classed step 6 (`surface_heat_flow`) as a transcription step.
   Cause: step 6 was a numbered recipe that chained steps 1-4 (TVD, R and S, the half-space perturbation of
   step 3, the linear three-parameter fit of step 4); it required no scientific decision, and the hardest step
   (step 5, the layered perturbation) did not feed the integrated result at all.

2. Change: a new final step, `deglaciation_heat_flow_inversion(survey_md, survey_inc, survey_azi, log_md,
   log_temp, layer_top_md, layer_k, heat_production, rho_c, hist_t_years, hist_dT_shape, t_end_bounds, z_min)`
   (problem_io, function_header, scaffold, solution/step_6.py, solution.py, second_solution.py). The time at
   which the last glaciation ended, t_end, becomes a fourth unknown next to T0, q0 and g, and the glacial
   signal is the layered-column perturbation of step 5: P(z; t_end) = layered_paleoclimate_perturbation(z,
   tops in TVD, layer_k, rho_c, [t_end, *hist_t_years], hist_dT_shape), with m >= 2 shape values and m - 1
   known interval ends (interval 0 from the present back to t_end, interval 1 from t_end back to
   hist_t_years[0], ...), 0 < t_lo < t_hi < hist_t_years[0]. Visible conventions: t_end is the global minimizer
   over the closed interval [t_lo, t_hi] of the profile RSS(t) (three-parameter least squares at fixed t);
   Wald standard errors s^2 (J^T J)^-1 with J = [1, R, P, g dP/dt_end] (t_end in years) and
   s^2 = RSS_min/(n_used - 4); profile-likelihood 95 % interval bounded by the crossings of
   RSS(t) = RSS_min (1 + F/(n_used - 4)), F = scipy.stats.f.ppf(0.95, 1, n_used - 4), nearest to t_end on each
   side; rms_residual = sqrt(RSS_min/n_used); the inclusive z_min cut on TVD; domain rms_residual >= 1e-4 K.
   Error rules: invalid logs or non-finite temperatures, invalid history (counts, finiteness, order),
   s_0 = s_1 (t_end without effect), invalid bounds, heat production, any earlier-step error, fewer than 5
   readings, an unresolvable design at t_end (criterion of step 4), a minimum at t_lo or t_hi, a nearest
   crossing outside [t_lo, t_hi]. The prompt names the functions of steps 1, 2, 4 and 5 (step_dependencies
   [1, 2, 4, 5]) and is not a numbered recipe: the reduction to a profile over t_end, the global search, the
   computation of dP/dt_end and the crossing search are left to the agent. Step 3 no longer feeds the final
   function (it remains the classical half-space step and the basis of the half-space mutants).
   required_dependencies gain `from scipy import stats` and `from scipy.optimize import brentq,
   minimize_scalar`. problem_description_main, the step 6 contract and background, the metadata
   (subfield, tags, expert time 14 h, difficulty, solution and verification explanations), background.md
   section 6 and edit_label (difficulty_increase) were rewritten for the new final step. background.md section 3
   said the glaciation left the upper 1-2 km "warmer" than a steady profile; with a cold glacial surface (the
   sign convention of the task, shape -1 during the glaciation) it is colder, which is also what makes the
   uncorrected heat flow too low (same paragraph); corrected to "colder".

3. Statistics (derivation, no external source needed). For Gaussian noise of unknown variance the profile
   log-likelihood of t is -(n/2) log RSS(t) + const. The linearized covariance at the estimate is
   s^2 (J^T J)^-1 with the unbiased s^2 = RSS_min/(n - 4) for four fitted parameters. For one parameter of
   interest the likelihood-ratio statistic, calibrated for the estimated variance, is
   (RSS(t) - RSS_min)/s^2, which is F(1, n - 4) distributed to first order, so the 95 % profile region is
   RSS(t) <= RSS_min (1 + F_0.95(1, n - 4)/(n - 4)). F_0.95(1, nu) is the square of the two-sided 95 % Student t
   quantile, which the second solution and the test oracle use.

4. Test oracle for P and dP/dt_end: branch-cut (real-axis) inversion (author's derivation). For the layered
   column with a bottom half-space the transform F(z, s) of the response to a unit surface step is analytic off
   the cut s <= 0 (the bottom half-space keeps only the decaying solution, and a negative-definite diffusion
   operator has no eigenvalue off that axis). Collapsing the Bromwich contour on the cut gives
   U(z, t) = 1 + (1/pi) int_0^inf exp(-u t) Im F+(z, u) / u du and dU/dt = -(1/pi) int_0^inf exp(-u t)
   Im F+(z, u) du, with F+ on the upper side, where q_j = i sqrt(u/kappa_j); with u = w^2/t these become
   1 + (2/pi) int exp(-w^2) Im F+ / w dw and -(2/(pi t)) int exp(-w^2) w Im F+ dw. On the cut every layer
   solution is a cos/sin combination: (F, k F') is propagated from the surface with real transfer matrices for
   the two surface states (1, 0) and (0, 1), and the complex surface flux follows from the radiation condition
   k F' + i k omega F = 0 at the top of the bottom half-space (so Im F = Im(flux) x F_b(z) inside the finite
   layers). The integrals use composite 24-point Gauss-Legendre panels on w in [0, 7] (exp(-49) = 5e-22), the
   panel width adapted to the oscillation rate of the layer stack. Nothing is shared with the reference
   (admittance recursion, fixed Talbot contour), the second solution (global linear system, hyperbolic
   contour) or the stored-target scripts (transfer matrices in the right half-plane, mpmath Talbot). Checks
   (run offline in a clean process): against the stored 90-digit mpmath values of tests/step_5.py, max error
   3.3e-16 K (five-layer Abitibi column, ten depths), 8.9e-16 K (contrasted column, nine depths) and 4.4e-16 K
   (50-layer column, six depths, 20 intervals); against erfc for a uniform column 6.7e-16 (2 years to 120 ka,
   0.5 m to 3 km); against the closed-form time derivative of erfc 2.7e-16 relative; against an independent
   parabolic-contour inversion (Weideman-Trefethen parabola, 36 nodes) 1.5e-14 for U and 7.5e-18 per year
   (1e-13 relative) for dU/dt on a seven-layer column; against the reference step 5 (fixed Talbot, 32 nodes)
   2.8e-11 to 5.3e-11, which is the round-off noise of the Talbot sum (it varies irregularly with t, by about
   2e-11 between times 0.1 year apart). A uniform column uses math.erfc and its closed-form derivative
   z/(2 sqrt(pi kappa) t^1.5) exp(-z^2/(4 kappa t)) directly.

5. Targets: orthogonal-residual construction. Each borehole is built forward (64-point Gauss-Legendre TVD of
   the great-circle tangent, exact layer sums). On the used readings the data are y = T + A S = T0 + q0 R +
   g P(t_end) + r at the generating values, with r = (I - Q Q^T) w for a seeded Gaussian w, Q an orthonormal
   basis of the four oracle Jacobian columns [1, R, P, g dP/dt_end], scaled to the stated rms. The generating
   values are then an exact stationary point with RSS_min = |r|^2, rms_residual = the stated rms, and the Wald
   standard errors follow from the oracle Jacobian. Readings shallower than z_min receive disturbed values
   (T0 + 1.5 + 0.02 z + 0.3 N(0, 1)) that follow no model, so a cut on MD or a strict cut changes n_used and the
   estimates. The interval targets are the oracle profile's nearest crossings (outward march in steps of
   0.05 to 0.3 sigma_t, then bisection with Illinois steps to 1e-12 relative); F comes from the Student t
   density integrated by Gauss-Legendre quadrature and bisection (agrees with scipy.stats.f.ppf to 6e-13
   relative for nu = 1 to 1000). That the generating t_end is the global minimum over the bounds of every case
   was verified offline by a dense scan of the oracle profile (6000 logarithmically spaced times per case; 1500
   for the two calls with bounds 9000-29900 years added in item 11): no point lies below RSS_min.

   | Case | Borehole | n_used | rms (K) | sigma_t (yr) | interval (yr) | Wald +- 1.96 sigma | special feature |
   |---|---|---|---|---|---|---|---|
   | step 0 | inclined 1800 m, 7 layers, t_end 11.5 ka, bounds 3-60 ka | 326 | 5e-3 | 192.9 | 11132.9-11891.8 (-367/+392) | 378 | output types |
   | step 1 | 1000 m, 5 strongly contrasted layers, 4-interval history, t_end 9 ka | 38 | 4e-3 | 722.5 | 7728.3-10729.2 (-1272/+1729) | 1416 | F(1, 34) = 4.13 vs 3.84 |
   | step 2 | vertical, uniform k = 3.1, z_min = 0 with the collar reading | 121 | 6e-3 | 728.1 | 9230.5-12162.4 | 1427 | erfc oracle, inclusive cut |
   | step 3 | step 0 geometry, glaciation 30 ka to t_end, bounds 3-23.96 ka, then 9-29.9 ka | 326 | 5e-3 | 305.9 | 10932.4-12141.2 | 599 | second basin, minimum 0.54 Delta above RSS_min at 23.97 ka (Delta = threshold - RSS_min); RSS(t_hi) below the threshold; with 9-29.9 ka both basins inside (ends 30 and 397 Delta), a local search ends at 23.97 ka |
   | step 4 | step 0 hole, bounds 3-9 ka and 14-60 ka (ValueError), 2.5-92 ka (valid) | 326 | | | | | end-point minima; ridge near 88.5 ka |
   | step 5 | step 1 hole, bounds 8.5-29.9 ka and 1.5-10 ka (ValueError), 7-29.9 ka (valid) | 38 | | | | | RSS(8500) and RSS(10000) below the threshold (0.13 and 0.38 Delta) |
   | step 6 | vertical, 7 readings, z_min 200 m (5 used) / 400 m (4: ValueError), 29 invalid inputs (4 added in item 11) | 5 | 2e-4 | 229.3 | 7717.8-14028.1 (-2282/+4028) | 450 | F(1, 1) = 161.4 |
   | general 0 | inclined 2400 m, 5 layers, 4-interval history, t_end 9.8 ka | 220 | 1e-2 | 67.8 | 9668.0-9935.2 | 133 | deep hole |
   | general 1 | step 1 geometry, other noise seed; log + 0.75 K | 38 | 4e-3 | 722.5 | 7723.7-10737.7 | 1416 | only T0 moves |
   | general 2 | step 3 geometry, t_end 12.2 ka, g 6.9, bounds 3.5-23.8 ka, then 9-29.9 ka | 326 | 5e-3 | 440.3 | 11400.2-13149.3 | 863 | second minimum (0.39 Delta) at 23.76 ka; with 9-29.9 ka both basins inside (ends 27.5 and 226 Delta), a local search ends at 23.76 ka |

6. Tolerances (stated in the prompt, header and contract) and error budget. Limits: T0 2e-3 K; q0, g, t_end
   relative 5e-4; the four standard errors relative 5e-3; t_end_low, t_end_high relative 5e-4; rms_residual
   relative 1e-3; n_used exact. Two outputs compared in general case 1 use twice these. They are set by the
   propagation of the stated accuracies of steps 1, 2 and 5 through the final step, measured by running the
   reference final step on every test borehole with perturbed earlier steps (the general cases use the
   candidate's own steps 1-5): (a) P in error by 1e-8 K per K, smooth in depth and in t_end (four patterns):
   t_end <= 2.8e-5, g <= 3.9e-5, q0 <= 5.4e-6 (relative), T0 <= 9.2e-5 K, standard errors <= 1.9e-4,
   interval ends <= 7.4e-5, rms <= 1.8e-4 (the five-reading case); (b) P in error by 1e-8 K per K varying
   irregularly with depth and with t_end (as the round-off of a contour inversion does): t_end <= 8.8e-5,
   g <= 1.6e-4, T0 <= 3.1e-4 K, standard errors <= 1.2e-3, interval ends <= 3.1e-5, rms <= 8.4e-5; (c) TVD in
   error by 1e-6 m or R and S by a relative 1e-9 (two patterns each): q0, g, t_end <= 1.3e-6, T0 <= 3.1e-6 K,
   standard errors and rms <= 3.2e-5, interval ends <= 1.3e-5 (the largest values on the five-reading case
   with its 0.2 mK residual; 3e-6 and below on the other boreholes). The largest ratio of any of these to its limit is 0.31. The profile is flat at its minimum,
   so the irregular errors of (b) dominate; with them a central difference of P with a 1-year step amplifies the
   error into dP/dt_end and exceeds the standard-error limit (3.2 times on the five-reading case), hence the
   proviso, visible until item 11, that dP/dt_end be formed without amplifying the error of P (item 11 removed
   it from the prompt as a method hint; the limits and the sentence that they include the propagated errors stay).
   At the domain floor rms_residual = 1e-4 K (item 11; the propagated P error does not shrink with the noise while
   the standard errors, the rms and the interval half-widths do) the same perturbations give at most 0.45 of a
   limit on the five-reading hole (sigma_q0, irregular P error), 0.26 on the short hole and 0.11 and 0.03 on the
   two inclined 326-reading holes (step cases 3 and 0 geometries). Linearized worst cases (L1 norms
   of the rows of the pseudo-inverse of J times 1e-7 K per reading) give at most 1.4e-5 relative for t_end,
   q0 and g and 3.4e-5 K for T0 (short hole). With exact earlier steps the largest error-to-limit ratio is
   3.2e-3 for the reference (general case 1; the Talbot round-off of about 3e-11 K per K limits the location of
   the flat minimum to about 1e-6 relative) and 3.2e-4 for the second solution, so the margins are at least 300
   and 3000. The scientific errors are far outside the limits: half-space perturbation 480 to 12000 times a
   limit, conditional standard errors 180 to 200 times, a t_end column without g 800 to 1400 times,
   chi-square quantile 3 to 507 times (the inclined holes with n_used = 326 alone would not see it: 0.26 and
   0.42 of the limit, hence the short, uniform-column and five-reading cases), n - 3 in the threshold 6 and
   468 times (short and five-reading holes; just above the limit on the uniform column), n - 3 in the noise
   variance 3 and 59 times (short and five-reading holes).

7. Second solution (second_solution.py, final step): Levenberg-Marquardt on all four parameters from every
   minimum of a 90-point logarithmic profile scan, with dP/dt_end from the inverse transform of F itself (the
   impulse response, U(z > 0, 0) = 0) on the hyperbolic contour of its own global-linear-system layered solver,
   P assembled from step responses at the known times, covariance from the QR factor of J, the threshold from
   scipy.stats.t.ppf(0.975, n - 4)^2, the interval ends by Illinois regula falsi after an outward march; its own
   steps 1-5 throughout. It shares neither the variable projection, the Talbot inversion, the finite-difference
   derivative, scipy.stats.f nor Brent's methods with the reference, and no code with the test oracle. It passes
   every case (largest error-to-limit ratio 3.2e-4), 0.5 to 1.1 s per call.

8. Mutants. Step 6 (each the current reference with one error): halfspace_mean_diffusivity,
   wald_without_t_end_column, chi_square_threshold, threshold_dof_n_minus_3, residual_variance_n_minus_3,
   jacobian_without_amplitude, interval_end_bracketed_to_bound, measured_depth_as_depth,
   z_min_on_measured_depth, z_min_cut_strict, single_mean_conductivity, local_minimizer_without_scan (item 11);
   the old step 6 mutants (surface_heat_flow) were removed or rebuilt for the new function. Whole task (each
   solution.py with one error in the final function): measured depth as depth, z_min on MD, one mean
   conductivity, half-space perturbation (replaces "no paleoclimate correction", which has no meaning when t_end
   is estimated), local minimizer without a scan (item 11). The brief also proposed a mutant without the
   end-point check; it was not registered because it cannot be detected: when the global minimum lies at t_lo
   (t_hi) the profile has no point below the threshold beyond it, so the crossing on that side does not exist
   inside the bounds and the uncontained-interval rule raises the same ValueError. The end-point rule stays
   stated and tested (step case 4). The resolvability rule at t_end is in the same position: if [1, R, P(t_end)]
   is degenerate, RSS(t_end) equals the residual of the fit on [1, R] alone, which bounds RSS(t) from above for
   every t, so t_end can be the minimizer only if the profile is constant, and then the end-point rule raises
   the same ValueError. A design degenerate only at the estimate therefore cannot be built; the rule stays
   stated and is exercised by step case 6 ("one_depth": every reading at one depth, a constant profile), where
   the rules cannot be told apart.

9. Regression tests and reruns. The seven step 6 cases and three general cases above replace the
   surface_heat_flow cases (every case self-contained, every general case calls the final function).
   tools/crown_check.py, all modes, every case alone in a fresh process, one BLAS/OpenMP thread, -j 2: summary
   ALL OK (table below; final run after the repairs of item 11, on the shipped files). Reference final call
   0.1 to 0.6 s for 5 to 326 readings; each case takes at most 3.3 s with the reference and 4.7 s with the
   second solution (single thread; step case 3, which has two calls since item 11). solution.py and
   second_solution.py were each run twice with one thread and once with two threads on six of the test
   boreholes (steps 0-3 and 6, general 0): identical SHA-256 digests of the outputs.

   | Mode | Result |
   |---|---|
   | format | OK (no problem; every general case calls deglaciation_heat_flow_inversion) |
   | REF steps 1-6, general (solution.py) | 22/22, 12/12, 12/12, 7/7, 8/8, 7/7, 3/3 |
   | SECOND steps 1-6, general | 22/22, 12/12, 12/12, 7/7, 8/8, 7/7, 3/3 |
   | SHIFT +-1e-12 (step outputs; every evaluated function in general) | all cases pass |
   | CONTROL None / 0.0 | 0 cases pass in every step and in general |

   | Where | Mutant | Cases failed |
   |---|---|---|
   | step 6 | halfspace_mean_diffusivity | 6 of 7 |
   | step 6 | wald_without_t_end_column | 7 of 7 |
   | step 6 | chi_square_threshold | 4 of 7 (1, 2, 5, 6) |
   | step 6 | threshold_dof_n_minus_3 | 4 of 7 (1, 2, 5, 6) |
   | step 6 | residual_variance_n_minus_3 | 3 of 7 (1, 5, 6) |
   | step 6 | jacobian_without_amplitude | 7 of 7 |
   | step 6 | interval_end_bracketed_to_bound | 1 of 7 (case 3) |
   | step 6 | measured_depth_as_depth | 6 of 7 |
   | step 6 | z_min_on_measured_depth | 3 of 7 (0, 3, 4) |
   | step 6 | z_min_cut_strict | 1 of 7 (case 2) |
   | step 6 | single_mean_conductivity | 6 of 7 |
   | step 6 | local_minimizer_without_scan (item 11) | 1 of 7 (case 3) |
   | whole task | whole_task_measured_depth_as_depth | 3 of 3 |
   | whole task | whole_task_z_min_on_measured_depth | 2 of 3 |
   | whole task | whole_task_single_mean_conductivity | 3 of 3 |
   | whole task | whole_task_halfspace_perturbation | 3 of 3 |
   | whole task | whole_task_local_minimizer_without_scan (item 11) | 1 of 3 (case 2) |

   Steps 1-5 mutants: unchanged results (revision 10b table).

10. Not changed or open. relevant_experience is still the author's placeholder; the author fills it in himself
    before upload (it is not written by the assistant). The difficulty band cannot be measured locally. The
    branch-cut oracle script lives inside the test cases (no separate file).

11. Final repair before upload (independent verification and blind-solver calibration). Each finding with its
    cause, change, regression test and reruns.
    a. Finding (blocking, independent verifier): the stated rule that t_end is the global minimizer over
       [t_lo, t_hi] was not discriminated. An alternative final step with one bounded Brent search of the
       profile over the whole interval and no scan passed all 7 step and 3 general cases. Cause: inadequate
       tests; in step case 3 and general case 2 the second basin sits at or beyond t_hi, and the first
       golden-section points of a local search over the interval already lie in the global basin. Change: step
       case 3 and general case 2 each gain a second call on the same borehole with t_end_bounds = (9000, 29900)
       years, which contains both basins. The second minimum is then interior (23.97 ka at 0.54 Delta above
       RSS_min; 23.76 ka at 0.39 Delta), both ends are far above the threshold (30 and 397 Delta; 27.5 and
       226 Delta), the smallest scaled singular value of [1, R, P(t)] over the bounds is 1.6e-3, and the nearest
       crossings (10.93-12.13 ka; 11.40-13.15 ka) lie inside the bounds. The targets are those of the first
       call (same borehole, generating values and nearest crossings). The RSS gap between the basins (5.3e-5
       and 3.8e-5 K^2) is more than 100 times the change in RSS that a 1e-8 K per K error of P can cause (at
       most about 3e-7 K^2). A 1500-point scan of the oracle profile over the new bounds finds no point below
       RSS_min. New mutants: local_minimizer_without_scan (step 6) and whole_task_local_minimizer_without_scan,
       one bounded Brent search over [t_lo, t_hi]; they converge to 23.97 ka and 23.76 ka and fail step case 3
       and general case 2. Brent searches started at the arithmetic midpoint of the interval also end in the
       second basin on both boreholes, and so does a bounded four-parameter least_squares from the midpoint on
       the general case 2 borehole (verifier's runs). The reference, solution.py, the second solution and a
       scan-and-refine alternative pass; on the new calls the reference reaches at most 1.8e-4 of a limit and
       the second solution 5.3e-5.
    b. Finding (advisory, verifier): two phrases of the step 6 prompt were method hints, not conventions:
       "(the profile can have several local minima)" and the explanation of the propagated-error budget
       ("the profile is flat at its minimum ... provided dP/dt_end is formed without amplifying that error").
       Change: both removed from the prompt. "The global minimizer of RSS(t) over the closed interval
       [t_lo, t_hi]" is the complete convention, and the accuracy paragraph keeps every limit and the sentence
       that the limits include the errors propagated from the stated accuracies of steps 1, 2 and 5 (as the
       function header already did). No convention, unit, limit or error rule changed; edit_label stays
       difficulty_increase. step_background keeps its physical remark on the amplitude-timing trade-off.
    c. Finding (advisory, verifier): the propagated-error budget was measured only on the test boreholes
       (rms_residual >= 2e-4 K), while the stated domain goes down to 1e-4 K. Change: re-measured at the domain
       floor (item 6); the worst value is 0.45 of a limit, so the visible sentence holds over the stated
       domain. verification_explanation now says so.
    d. Finding (advisory, verifier): stated boundaries of the error rules were untested. Change: step case 6
       gains t_hi equal to hist_t_years[0] (the strict upper limit), t_lo = t_hi, a 2-D hist_dT_shape and a 2-D
       hist_t_years; the reference and the second solution raise ValueError on each.
    e. Finding (advisory, verifier): the resolvability rule is exercised only together with a constant
       profile. No test change; item 8 explains why the rule cannot be separated with any input.
    f. Finding (advisory, verifier): two headings "Version 10" and an older "Version 11". Change: the two
       internal notes are renamed "Internal revision 10a" and "Internal revision 10b", with a naming note, and
       every reference to them is updated.
    g. Finding (blocking, metadata): relevant_experience is the author's placeholder (item 10, typo
       "relative_experience" corrected). It must be written by the author before upload.
    h. Blind-solver calibration (internal, not an independent review). Two solutions were written from the
       visible specification only (problem_description_main, required_dependencies, step prompts, function
       headers and return lines). Each was used as the whole candidate for every step file and for general,
       with every case alone in a fresh process. On the tests before this repair both passed all 71 cases;
       their largest error-to-limit ratios were 1.1e-7 and 1.8e-4 (both in step case 6), smaller than the
       reference's. Both scan the profile on a grid (401 and about 800 points) before refining, so the new
       calls of item a do not catch them: on the final test files both pass every case again:

       | Test file (cases) | blind solution 1 | blind solution 2 |
       |---|---|---|
       | step 1 (22) | 22/22 | 22/22 |
       | step 2 (12) | 12/12 | 12/12 |
       | step 3 (12) | 12/12 | 12/12 |
       | step 4 (7) | 7/7 | 7/7 |
       | step 5 (8) | 8/8 | 8/8 |
       | step 6 (7) | 7/7 | 7/7 |
       | general (3) | 3/3 | 3/3 |

       The ambiguities they noted are not at tested points (exactly 1e-6 rad in step 1, readings exactly at a
       non-zero z_min, a singular Wald matrix, an exact tie between an end value and an interior minimum, 2-D
       input to step 4, size-1 arrays as scalars). This calibration, and the verifier's estimate after these
       changes (about 5 to 7 of 8 rollouts), indicate that strong agents may complete the whole chain; the
       remaining difficulty is the layered Laplace inversion, the joint Wald covariance and the profile search,
       and the platform rollouts are the only measurement of the band.
    i. Reruns: tools/crown_check.py in all modes (item 9 tables, updated), the blind solutions on every case,
       the margins of the new calls for the reference and the second solution, the dense oracle scans of the
       new bounds and the domain-floor budget of item 6.

12. Second difficulty round before upload: a regional final step (step 7). Each finding with its cause, change,
    regression test and reruns.
    a. Finding (calibration, item 11h): two solutions written from the visible specification alone passed all
       71 cases of the round-1 task, with margins larger than the reference's, and the verifier estimated 5 to 7
       of 8 rollouts. Version 9 was "too easy" (8 of 8) and version 10 is the last version allowed, so the
       round-1 design would very likely miss the band (3 to 6 of 8) again.
       Cause: the single-hole inversion of step 6 is ordinary nonlinear least squares whose conventions (profile
       RSS, Wald covariance with n_used - 4, F threshold) are standard and completely stated; strong agents
       implement them reliably. The difficulty to add has to come from scientific and statistical reasoning, not
       from tolerances, hidden conventions or runtime.
    b. Change: a new final step 7, `regional_deglaciation_inversion(holes, hist_t_years, hist_dT_shape,
       t_end_bounds)` (problem_io, function_header, scaffold, solution/step_7.py, solution.py,
       second_solution.py, tests/step_7.py, tests/general.py, mutants). Several boreholes of one region share the
       glacial amplitude g and the end of the glaciation t_end; every hole keeps its own T0, q0, column, heat
       production, rho_c, z_min and an unknown noise level sigma_h (logs of different quality). Each hole is a
       dict with exactly the ten keys of the step 6 arguments it replaces. The prompt states the statistical
       model (independent normal errors, one unknown variance per hole) and the definitions of the results:
       maximum likelihood for all parameters with g on the whole real line and t_end the global maximizer over
       [t_lo, t_hi] of the profile log-likelihood (maximum over all other parameters); the log-likelihood as the
       natural logarithm of the joint normal density with its normalizing constants; standard errors from the
       inverse of the expected Fisher information of all parameters at the estimates; the 95 % likelihood-ratio
       interval with scipy.stats.chi2.ppf(0.95, 1) and the crossings nearest to t_end; at least 6 used readings
       per hole (the likelihood is unbounded when a hole can be fitted exactly); a resolvability rule for
       [1, R_h]; the end-point and containment rules of step 6; the domain of noise levels. It does not give the
       concentrated likelihood, the structure in g, the information matrix or an algorithm. Earlier functions
       named: true_vertical_depth, layer_integrals and layered_paleoclimate_perturbation (step_dependencies
       [1, 2, 5]). Steps 1-6 are unchanged (prompts, headers, references, second solution, tests, mutants; the
       round-1 whole-task tests and mutants of step 6 are replaced by step 7 ones, and step 6 keeps its own 7
       cases and 12 mutants). problem_description_main, problem_io, the subfield and tags, the expert time (18 h),
       the difficulty, solution and verification explanations and background.md (new section 7) were rewritten;
       edit_label stays difficulty_increase.
    c. Statistics (derivation, standard likelihood theory; no external source needed). For fixed (g, t) the
       maximum over T0_h and q0_h is the least-squares residual RSS_h(g, t) = a_h g^2 - 2 b_h g + c_h of hole h
       on [1, R_h], and the maximum over sigma_h is sigma_h^2 = RSS_h / n_h, so
       l(g, t) = -sum_h (n_h/2) (log(2 pi RSS_h(g, t)/n_h) + 1). In g this is a sum of Student-type log factors,
       not a quadratic: every stationary point lies between the holes' own optima b_h/a_h, and pooled least
       squares (one variance) or fixed weights are different estimators. The expected information of the
       normal model is block diagonal between the regression parameters and the sigma_h, so its inverse restricted
       to (T0_h, q0_h, g, t_end) is (sum_h J_h^T J_h / sigma_h^2)^-1, with J_h the Jacobian of hole h's mean;
       the observed information differs (curvature of the model in t_end, and per-hole scores of g and t_end that
       do not vanish separately). The likelihood-ratio statistic 2 (l_max - l_p(t)) is chi-square(1) to first
       order. With n_h - 2 <= 2 residual dimensions left after T0_h and q0_h, the two-parameter family
       g M_h P_h(t) (M_h the projection off [1, R_h]) can reach the data of the hole exactly and the likelihood
       is unbounded; with three the surface can still pass close to the data. The stated rule n_h >= 6 keeps at
       least four residual dimensions (tested at n_h = 6, valid, and 5, ValueError).
    d. Targets: weighted-score orthogonal construction (tests/step_7.py header; author's design). On the used
       readings of hole h, y_h = T + A S = T0_h + q0_h R_h + g P_h(t_end) + r_h with r_h = b_h + c_h: b_h = the
       seeded Gaussian vector projected off the four oracle Jacobian columns [1, R_h, P_h, g dP_h/dt_end] and scaled
       to the stated rms; c_h = noise_h (k1 e1 + k2 e2) with e1, e2 an orthonormal basis of the projections p_h, d_h
       of P_h and g dP_h/dt_end off [1, R_h] (stated coefficients) for every hole but one; for that hole c = -lambda
       X G^-1 v with X = [p, d], G = X^T X, v = sum over the other holes of (n_h/|r_h|^2) X_h^T c_h, and lambda the
       smaller root of s lambda^2 - n lambda + |b|^2 = 0 (s = v^T G^-1 v), so that the weighted scores sum_h
       (n_h/|r_h|^2) [P_h, g dP_h/dt_end]^T r_h vanish exactly. With sigma_h^2 = |r_h|^2/n_h every score of the
       log-likelihood is zero at the generating values: the weighted shared scores are below 4e-12 of the largest
       single-hole term, and the cosines between r_h and the columns 1 and R_h below 3e-13, on every region. The
       unweighted scores do not vanish, and pooled least squares lands elsewhere (table). Readings shallower than
       z_min get disturbed values (T0 + 1.5 + 0.02 z + 0.3 N(0, 1)). Offline checks in a clean process (scratch
       scripts, not part of the upload): a scan of the oracle profile log-likelihood at 2000 logarithmically spaced
       times over the bounds of every region and at 201 times within 4 sigma_t of t_end finds no value above l_max
       (largest excess 1e-11, at t_end itself, rounding); the profile in g at t_end (20001 points between and around
       the holes' optima) has a single maximum, at g, on the ten regions of this table (the region added in item 13c
       has two); the interval ends, the information and l_max follow from the oracle as in the test header. A first,
       unscaled version of the reference's polynomial in g was ill-conditioned for precise logs (the region of step
       case 0 rebuilt with 0.1 mK logs: polynomial roots rejected, log-likelihood 5.7 too low); the budget runs of
       item f found it, and the reference now writes the polynomial in a scaled variable and polishes the roots by
       Newton steps. A region with a discordant hole that would make the profile in g bimodal at t_end was explored
       and not used: with a precise hole of few readings the global maximum moved to another t_end (dense scan), so
       the generating values were not the maximum-likelihood solution. Item 13c adds such a region, with more
       readings in the precise hole, in which the dense scans confirm the generating values as the global maximum.

       | Case | Holes: n_used (noise, mK) | g, t_end | sigma_t (yr) | 95 % interval (yr) | pooled LS t_end (x t_end limit) | feature |
       |---|---|---|---|---|---|---|
       | step 0 | inclined 7 layers 163 (4.0); 4 layers 137 (12.3); 3 layers 136 (25.6) | 7.8, 11500 | 180.7 | 11147.9-11871.7 | 11401.8 (4.3) | output types |
       | step 1 | contrasted 1000 m 24 (2.1); deep 2400 m 220 (20.2) | 6.0, 9000 | 38.2 | 8925.1-9080.1 | 9168.9 (9.4) | RSS/(n - 2) changes the short hole's weight by 9 % |
       | step 2 | uniform k = 3.1, collar at z_min = 0, 121 (6.0); uniform k = 2.4, 57 (15.8) | 6.2, 10500 | 704.4 | 9271.8-12113.5 | 11938.8 (68.5) | erfc oracle, asymmetric interval |
       | step 3 | inclined (5 m) 326 (5.0); 137 (10.1); 136 (20.2); glaciation 24 ka | 7.8, 11500 | 338.4 | 10884.2-12227.7 | 11431.7 (3.0) | second maximum at 17.98 ka, 9.96 below l_max, valley at 15.7 ka (20.5 below) |
       | step 4 | step 0 region; bounds 3-11 and 12-60 ka (ValueError), 11-12 ka (valid) | | | | | single maximum over 3-60 ka |
       | step 5 | step 1 region; bounds 8.95-29.9 and 1.5-9.05 ka (ValueError), 8.9-9.1 ka (valid) | | | | | |
       | step 6 | vertical, 6 of 8 readings (4.6; z_min 400 m keeps 5: ValueError); 137 (8.0) | 7.0, 10000 | 376.5 | 9245.7-10793.1 | 9877.4 (6.1) | 34 invalid inputs |
       | general 0 | deep 220 (10.1); steep 1400 m 109 (3.1); 136 (18.2) | 5.0, 9800 | 46.8 | 9707.2-9895.7 | 9940.8 (7.2) | four-interval history |
       | general 1 | step 1 geometry, other seed and coefficients: 24 (2.1); 220 (20.3) | 6.0, 9000 | 38.3 | 8919.7-9076.2 | 8809.7 (10.6) | + 0.75 K on one log, reversed hole order |
       | general 2 | step 3 geometry, other seed: 326 (6.0); 137 (11.1); 136 (18.1); glaciation 24 ka | 7.4, 11800 | 466.8 | 10977.0-12843.5 | 11756.3 (1.9) | second maximum at 17.92 ka, 4.62 below l_max |

    e. Tests. tests/step_7.py (7 cases, 8 since item 13c, each self-contained with the generator, the branch-cut
       oracle and the checks): the table above; every output is compared with the exact values, the interval ends
       with the oracle's nearest crossings (outward march, Illinois bisection to 1e-12), the chi-square quantile
       comes from math.erf by bisection, and the maximum over g in the oracle profile is bracketed by the sign
       changes of its derivative on a 4001-point grid between the holes' own optima (neither solution's main method;
       since item 13d the reference adds a sign-change pass on a width-adapted grid with Brent's root finder as a
       backup). Oracle premises are asserted where a case depends on them (six used readings, the collar reading
       used, the asymmetric interval, the second basin of the short glaciation). Validation (step case 6): one hole,
       holes not a list, a hole that is not a dict, a missing and an extra key, five readings, unequal or 2-D log
       arrays, a NaN temperature, negative or NaN heat production, rho_c = 0, a zero or missing conductivity, a
       layer top or a reading beyond the survey, a survey not starting at 0, all readings at one depth ([1, R_h] not
       resolvable), and the history and bounds rules of step 6 (16 entries). tests/general.py (3 cases) calls
       regional_deglaciation_inversion with every earlier step from the candidate: a deep three-hole region, a
       two-hole region with the invariance checks (twice the single-output limits) and a second short-glaciation
       region.
    f. Tolerances and error budget (stated in the prompt, the header and the contract): g and t_end relative
       2e-3; every q0 relative 5e-4; every T0 2e-3 K; the four kinds of standard errors relative 1e-2; t_end_low
       and t_end_high relative 5e-4; noise_sd relative 3e-3; log_likelihood 0.1 (absolute); n_used exact; domain
       noise_sd from 1e-4 K to 0.03 K in every hole. Measured as for step 6 (item 6): the reference final step run
       on every test region with steps 1, 2 and 5 perturbed at their stated accuracies (P by 1e-8 K per K, four
       patterns smooth in depth and time and one irregular pattern that changes completely with t_end; TVD by
       1e-6 m; R and S by a relative 1e-9); worst error / limit:

       | Regions | smooth P | irregular P | TVD, R, S | exact inputs |
       |---|---|---|---|---|
       | 10 step 7 and whole-task regions | 0.061 | 0.373 (sigma_q0, general case 2) | 0.020 | 0.006 |
       | rebuilt at 0.1 mK: step 0, step 1, step 6 and step 2 regions | 0.319 (log-likelihood, 436 readings) | 0.520 (noise_sd, six-reading hole) | 0.020 | 0.011 |
       | rebuilt at 30 mK: step 6, short-glaciation and uniform regions | 0.118 | 0.282 | 0.061 | 0.021 |

       The first limits tried (g, t_end relative 5e-4 and standard errors 5e-3, as in step 6) were exceeded
       (1.32 times, step case 6, irregular P error): with irregular errors the flat maximum of the profile moves
       by up to 0.015 standard errors of t_end, which is more than 5e-4 t_end on regions with sigma_t / t_end of
       3 to 4 %; the interval ends lie on the steep flanks of the profile and stay within 0.016 of 5e-4. The limits
       on g, t_end and the standard errors were therefore set to 2e-3 and 1e-2 before any platform run (design
       stage, from the budget, not from pass rates). The log-likelihood limit covers the largest budget error
       (0.032 at the 0.1 mK floor with 436 readings); noise_sd 3e-3 covers 1.6e-3 (six-reading hole at 0.1 mK,
       irregular P error). The upper end of the domain (0.03 K) was added so that the stated limits hold over the
       whole domain: the absolute error of the flat maximum grows like the square root of the noise level. The
       scientific errors stay outside the limits (item i).
    g. Second solution (second_solution.py, step 7): Fisher scoring on all regression parameters (iteratively
       reweighted Gauss-Newton with the noise variances re-estimated at every iterate, Levenberg damping, the
       log-likelihood as merit function) from every maximum of an 80-point profile scan; at fixed t_end the
       amplitude by the iteratively reweighted fixed-point (minorize-maximize) ascent from every hole's own
       optimum and from the pooled value; dP/dt_end from the impulse response of its own hyperbolic-contour
       layered solver; covariance from the QR factor of the weighted Jacobian; the chi-square quantile as
       scipy.stats.norm.ppf(0.975)^2; interval ends by Illinois regula falsi; [1, R_h] resolvability from the
       singular values |u +- v|/sqrt(2) of the two unit columns. It shares neither the concentrated polynomial, the
       Talbot inversion, the finite difference, scipy.stats.chi2 nor Brent's methods with the reference, and no
       code with the test oracle. Largest error-to-limit ratio on the step 7 and general cases: 5.3e-4 (noise_sd, step case 1); 1.0 to 2.8 s per call.
    h. Mutants. Step 7 (each the current reference with one error): pooled_least_squares, two_stage_weights,
       information_variance_n_minus_2, observed_information, information_blocks_inverted_separately,
       profile_with_fixed_noise_levels, lr_threshold_without_factor_2, log_likelihood_without_constant,
       local_search_without_scan, first_hole_rho_c_for_all, halfspace_mean_diffusivity, measured_depth_as_depth,
       z_min_on_measured_depth, z_min_cut_strict. Whole task (solution.py with one error in the final function):
       pooled least squares, half-space perturbation, MD as depth, z_min on MD, local search without a scan, the
       first hole's rho_c for all holes; the five round-1 whole-task mutants (errors in step 6) were removed, since
       the general tests no longer call step 6. Not registered in this round: the local maximization over g from the
       pooled value (no test region had a bimodal profile in g, item d; registered in item 13c with the new region),
       and a mutant without the end-point rule (indistinguishable from the containment rule, as in item 8).

       | Where | Mutant | Cases failed |
       |---|---|---|
       | step 7 | pooled_least_squares | 7 of 7 |
       | step 7 | two_stage_weights | 7 of 7 |
       | step 7 | information_variance_n_minus_2 | 4 of 7 (1, 2, 5, 6) |
       | step 7 | observed_information | 5 of 7 (0, 1, 4, 5, 6) |
       | step 7 | information_blocks_inverted_separately | 7 of 7 |
       | step 7 | profile_with_fixed_noise_levels | 7 of 7 |
       | step 7 | lr_threshold_without_factor_2 | 7 of 7 |
       | step 7 | log_likelihood_without_constant | 7 of 7 |
       | step 7 | local_search_without_scan | 1 of 7 (case 3) |
       | step 7 | first_hole_rho_c_for_all | 7 of 7 |
       | step 7 | halfspace_mean_diffusivity | 6 of 7 |
       | step 7 | measured_depth_as_depth | 6 of 7 |
       | step 7 | z_min_on_measured_depth | 6 of 7 |
       | step 7 | z_min_cut_strict | 1 of 7 (case 2) |
       | whole task | whole_task_pooled_least_squares | 3 of 3 |
       | whole task | whole_task_halfspace_mean_diffusivity | 3 of 3 |
       | whole task | whole_task_measured_depth_as_depth | 3 of 3 |
       | whole task | whole_task_z_min_on_measured_depth | 3 of 3 |
       | whole task | whole_task_local_search_without_scan | 1 of 3 (case 2) |
       | whole task | whole_task_first_hole_rho_c_for_all | 3 of 3 |

       Steps 1-6 mutants: unchanged results (item 9 and the revision 10b table), rerun in this round.

    i. Regression tests and reruns. tools/crown_check.py, all modes, every case alone in a fresh process, one
       BLAS/OpenMP thread, -j 2, on the shipped files: summary ALL OK:

       | Mode | Result |
       |---|---|
       | format | OK (every general case calls regional_deglaciation_inversion) |
       | REF steps 1-7, general (solution.py) | 22/22, 12/12, 12/12, 7/7, 8/8, 7/7, 7/7, 3/3; solution.py passes every step file |
       | SECOND steps 1-7, general | 22/22, 12/12, 12/12, 7/7, 8/8, 7/7, 7/7, 3/3 |
       | MUT | every step and whole-task mutant fails at least one case of its target tests |
       | SHIFT +-1e-12 (step outputs; every evaluated function in general) | all cases pass |
       | CONTROL None / 0.0 | 0 cases pass in every step and in general |

       Reference final call 0.3 to 1.1 s; each step 7 or general case takes at most 4.1 s with the reference
       and 6.7 s with the second solution. The offline evidence scripts of items d and f were rerun on the final
       files. solution.py and second_solution.py were each run twice with one thread and once with two threads on
       four of the regions (step cases 0, 3 and 6, general case 2): identical SHA-256 digests of the outputs.
    j. Not changed or open. relevant_experience is still the author's placeholder. The difficulty band cannot be
       measured locally, and no blind-solver calibration of step 7 was run in this round (no independent solver was
       available in this session): whether strong agents solve the regional step reliably is not known (a
       calibration of step 7 was run in item 13f). The difficulty comes from the statistical model (per-hole
       variances estimated jointly, concentrated likelihood, global maximization in g and t_end, expected
       information, likelihood-ratio profile with the variances re-maximized); every convention needed for a unique
       answer is stated.

13. Final repair of the version 10 upload, round 2 (independent verifier and blind-solver scoring of step 7). This
    round belongs to the same platform version 10 upload, so edit_label stays difficulty_increase; the changes are a
    reference repair, a grading fix of the test harness and one added discriminating case. Each finding with its
    cause, change, regression test and reruns.
    a. Finding (blocking, independent verifier): the end-point rule of both solutions was wrong on part of the
       stated domain. Both refined only the interior maxima of their t_end scan, so when the maximum lay inside the
       first (or last) grid cell and the end value was higher than the neighbouring grid point, both raised "maximum
       at an end point" although the maximum was interior and the interval contained. The verifier's demonstration:
       the step case 0 region rebuilt with 0.4, 1.2 and 2.5 mK and bounds (11433, 99000): exact t_end 11500,
       interval 11463.8-11536.2 inside the bounds, l_p(t_lo) - l_max = -6.46, l_p(grid[1]) - l_max = -26.6;
       solution.py and second_solution.py both raised. Step 6 had the same pattern for the minimum of RSS(t). Cause:
       a reference bug (no test exercised a maximum in an end cell). Change: in solution/step_6.py,
       solution/step_7.py and solution.py, an end value not below (step 6: not above) its neighbour is now also
       refined by bounded Brent over the end cell, and the end point is declared the extremum only if no refined
       value beats it; the second solution starts its Levenberg-Marquardt (step 6) and Fisher-scoring (step 7)
       iterations also from such an end point. Every mutant that carries the reference scan was updated in the same
       way, so that each still differs from the reference by its one named error. A maximum closer to the bound than
       the threshold crossing cannot change the outcome: then l_p(t_lo) > l_max - chi2/2, the crossing on that side
       does not exist inside the bounds and the containment rule raises the same ValueError. After the change the
       verifier's demonstration returns t_end = 11499.989 (reference) and 11500.000 (second solution) with the
       interval 11463.76-11536.19.
       Regression tests (near-boundary valid controls of the stated end-point rule, added to the end-point cases):
       - step 6 case 4: the same inclined hole logged with 0.5 mK (326 readings used): single minimum at 11500
         years, sigma_t 19.29 years, interval 11462.18-11538.07. With bounds (11445, 60000) (RSS(t_lo) -
         RSS_min)/s^2 = 8.21, s^2 = RSS_min/(n_used - 4), while the first point of a 121-point logarithmic grid
         (11604.1 years) is at 28.6; with bounds (3000, 11555) the end value is 8.06 and the last interior grid
         point (11425.9 years) 14.96. Both calls must return the estimate.
       - step 7 case 4: the case 0 region with 0.4, 1.2 and 2.5 mK (noise_sd 0.402, 1.227, 2.557 mK): sigma_t 18.07
         years, interval 11463.76-11536.19. Bounds (11445, 60000): l_p(t_lo) - l_max = -4.386, -15.20 at 11604.1
         years; bounds (3000, 11555): -4.402 at t_hi, -7.87 at 11425.9 years.
       Evidence (offline, clean process): dense scans of the oracle profiles (2000 logarithmically spaced times over
       each pair of bounds and 201 within 4 sigma_t of t_end) find a single extremum, at the generating t_end (no
       RSS below RSS_min beyond rounding, -4.7e-14 relative; no l_p above l_max, the largest value 7.4e-9 below
       l_max); the profile in g at t_end has a single maximum. The verifier's verify_ml.py, run on both step 7
       calls: normalized scores 7.6e-12 (oracle), the reference step 5 P within 1.3e-10 of the oracle, its dense
       scan with the reference P and a brute-force maximization over g finds no value above l_max. The old reference
       and the old second solution both fail the two new case 4 calls (ValueError at an end point); the new ones
       pass. Error budget of the new calls (P perturbed by 1e-8 K per K as in items 6 and 12f): step 7 worst
       error/limit 0.009 (smooth), 0.059 (irregular), 0.003 (exact inputs); step 6 0.005, 0.019 and 0.000; second
       solution with exact inputs <= 0.001. An adversarial pattern aligned with the sign of the residual gives 0.717
       of the log-likelihood limit on the precise step 7 region and 0.245 of the rms_residual limit on the precise
       step 6 hole.
    b. Finding (blind-solver scoring, unfair failure): two solutions written from the visible specification failed
       all 7 step 6 cases with "too many values to unpack". Cause: a grading defect of the test harness, not a
       scientific error. Each test case is appended after the candidate's code; every step 6 case defined a
       module-level class _Hole (the synthetic-borehole oracle), and both blind solutions used the same name for a
       private helper class called inside deglaciation_heat_flow_inversion, so the test's class replaced theirs
       before the call. The reference, the second solution and the mutants never used that name, so crown_check
       could not see it. The same exposure existed for every module-level helper and constant of the tests (for
       example _check, _bc_U, _g_tvd, _YEAR, _GL_X, YEAR, KEYS, and exp, which step 3 and step 5 cases assigned
       before calling the candidate; a candidate with "from math import exp" would have failed). Change: in all
       eight test files every module-level function and class, every module-level name with a capital letter or a
       leading underscore and the names exp and e now carry a test-specific prefix (_t1_ to _t7_, _tg_ for general;
       e.g. _t6_Hole, _t7_Region, _tg_check, _t3_YEAR, _t3_exp). The renaming was done by a scope-aware script: only
       references that resolve to the module-level binding were changed (locals and parameters inside functions kept
       their names), it checked that no renamed name is a keyword argument, a class member or used inside an
       f-string, and it left no module-level reference to an old name (counts: step 1 4 names / 37 references, step
       2 8 / 79, step 3 4 / 39, step 4 12 / 134, step 5 8 / 45, step 6 16 / 238, step 7 25 / 301, general 23 / 136).
       No assertion, input or target changed. Regression: both blind solutions now pass every step 6 case (table in
       item f); crown_check results are unchanged in every mode.
    c. Finding (advisory, verifier): the stated maximization of g "on the whole real line" had no discriminating
       test: in all 10 regions the likelihood in g at t_end had a single maximum, so a local maximizer started from
       the pooled value passed every case. Cause: inadequate tests (item 12d had explored a bimodal region and
       rejected it because its generating values were not the global maximum). Change: new step 7 case 7, a region
       with a disturbed hole built so that the generating values are the global maximum. Three holes: the inclined
       seven-layer hole logged every 10 m with 0.3 mK (163 readings used, noise_sd 0.314 mK), the four-layer hole
       with 8 mK (137, 8.015 mK) and the deep 2400 m hole (220, 18.25 mK) whose residual carries a component along
       its own projected glacial column, c = noise (-89 e1) in the construction of item 12d (one hole, the precise
       one, solved for the zero weighted scores). Alone the deep hole would give g = 7.5525, the others 7.8006 and
       7.8198 (widths sqrt(m_h/a_h) 0.041, 0.010 and 0.463). At t_end = 11500 years the likelihood in g has two
       maxima: the global one at the generating g = 7.8 and a second one at g = 7.5588, 137.74 lower, separated by a
       minimum at 7.6867; the pooled least-squares amplitude 7.5880 lies on the side of the second maximum. g 7.8,
       t_end 11500 years, sigma_g 0.0123, sigma_t 16.39 years, interval 11460.03-11535.89, bounds (3000, 60000).
       Evidence (offline): a dense scan of the oracle profile (2000 logarithmically spaced times over the bounds and
       201 within 4 sigma_t) finds a single maximum, at the generating t_end (largest excess 1.0e-8, rounding); the
       end values are 1713.6 and 1363.7 below l_max; the likelihood in g is bimodal only for t_end between about
       11.1 and 16.3 ka (161 times from 9 to 20 ka), where its second maximum lies 83 to 152 below the first. The
       verifier's verify_ml.py: normalized scores 1.9e-12 (oracle) and 1.3e-7 with the reference P, a dense scan
       with the reference P and a brute-force maximization over g finds no value above l_max (excess -2.7e-7), and
       two local maxima in g at t_end. Pooled least squares gives t_end = 11133 years (16 times the t_end limit) and
       g = 7.397. New mutant amplitude_local_ascent_from_pooled (step 7): at fixed t_end the amplitude by the
       fixed-point (iteratively reweighted) ascent started at the pooled least-squares value; it returns g = 7.4912,
       t_end = 11090 years and log-likelihood 2077.5 instead of 7.8, 11500 and 2118.947, and fails case 7 (the only
       case where the likelihood in g is bimodal). The reference, solution.py, the second solution and both blind
       solutions pass case 7. Error budget of case 7: worst error/limit 0.017 (smooth P error), 0.037 (irregular),
       0.014 with exact inputs (noise_sd of the 0.3 mK hole), 0.690 of the log-likelihood limit for the adversarial
       pattern aligned with the residual sign; second solution 0.002 with exact inputs. The difficulty_explanation
       and verification_explanation now describe this case; item 12d, 12e and 12h statements that no region is
       bimodal are superseded.
    d. Finding (advisory, verifier): with H = 6 or 8 holes whose standard errors in g are about 1e-6 of the spread
       of their own optima, the reference's polynomial roots lost the global maximum over g in 2 of 1000 synthetic
       trials (85 and 1064 below; rerun here on the previous file: 2 trials above 0.01), far from t_end; no test
       affected. Change (robustness): _rg_best_g also brackets the sign changes of the derivative on a grid refined
       around every hole's optimum (offsets of 1e-3 to 1e3 widths, plus 201 uniform points) and adds their Brent
       roots as candidates (also in the step 7 and whole-task mutants, which carry the reference function). Rerun of
       the verifier's stress scripts (1000 trials, H = 2..8): no trial with a deficit above 0.01; the remaining
       deficits (at most 4.6e-3, with g equal to about 8 significant digits) are the rounding of a_h g^2 - 2 b_h g +
       c_h when the widths are 1e-6 of the spread. The second solution's fixed-point ascent misses by 0.12 in one
       trial (unchanged; it is the alternative method and is not used for any target).
    e. Finding (advisory, verifier): "The likelihood is unbounded when a hole can be fitted exactly, so every hole
       must keep at least 6 used readings" read as a non sequitur (with 5 readings the likelihood is already
       bounded; the margin argument was only in item 12c). Change (clarification of wording, same rule): "...; to
       keep it bounded with a margin, every hole must keep at least 6 used readings". The rule and its tests (6
       valid, 5 ValueError) are unchanged.
    f. Blind-solver calibration (internal calibration, not an independent review). The two blind solutions of this
       round (written from the visible specification only; each used as the whole candidate for every step file and
       for general, every case alone in a fresh process, one thread, 2 workers):

       | Test file (cases) | blind solution 3, before | blind solution 3, after | blind solution 4, before | blind solution 4, after |
       |---|---|---|---|---|
       | step 1 (22) | 22/22 | 22/22 | 22/22 | 22/22 |
       | step 2 (12) | 12/12 | 12/12 | 12/12 | 12/12 |
       | step 3 (12) | 12/12 | 12/12 | 12/12 | 12/12 |
       | step 4 (7) | 7/7 | 7/7 | 7/7 | 7/7 |
       | step 5 (8) | 8/8 | 8/8 | 8/8 | 8/8 |
       | step 6 (7) | 0/7 (name clash, item b) | 7/7 | 0/7 (name clash, item b) | 7/7 |
       | step 7 (7 before, 8 after) | 7/7 | 8/8 | 7/7 | 8/8 |
       | general (3) | 3/3 | 3/3 | 3/3 | 3/3 |

       "Before" is the scoring of the files before this round (the step 6 failures were the harness defect of item
       b; with the test class renamed in a scratch copy both passed step 6 7/7, at most 6.5e-4 of a limit). "After"
       is the final test files. Both blind solutions scan the profile in t_end and refine the end cells (a linear
       241-point grid with bounded Brent from every grid minimum including the ends; a 161-point logarithmic grid
       with derivative sign changes), and both maximize over g globally (sign changes of the derivative on a grid
       refined around every hole's optimum), so the new calls of items a and c do not catch them. Two of two
       spec-only solutions therefore pass the whole task; the discriminators added in this round (a maximum in an
       end cell, a bimodal likelihood in g) catch coarser implementations, not these two.

    g. Mutants. New: amplitude_local_ascent_from_pooled (step 7). Every step 6, step 7 and whole-task mutant was
       rebuilt for items a and d (same single error each). Results on the final tests:

       | Where | Mutant | Cases failed |
       |---|---|---|
       | step 6 | the 12 mutants of item 9 | unchanged: 6, 7, 4, 4, 3, 7, 1, 6, 3, 1, 6 and 1 of 7 (same cases as item 9) |
       | step 7 | pooled_least_squares | 8 of 8 |
       | step 7 | two_stage_weights | 8 of 8 |
       | step 7 | information_variance_n_minus_2 | 4 of 8 (1, 2, 5, 6) |
       | step 7 | observed_information | 6 of 8 (0, 1, 4, 5, 6, 7) |
       | step 7 | information_blocks_inverted_separately | 8 of 8 |
       | step 7 | profile_with_fixed_noise_levels | 8 of 8 |
       | step 7 | lr_threshold_without_factor_2 | 8 of 8 |
       | step 7 | log_likelihood_without_constant | 8 of 8 |
       | step 7 | local_search_without_scan | 1 of 8 (case 3) |
       | step 7 | amplitude_local_ascent_from_pooled (new) | 1 of 8 (case 7) |
       | step 7 | first_hole_rho_c_for_all | 8 of 8 |
       | step 7 | halfspace_mean_diffusivity | 7 of 8 |
       | step 7 | measured_depth_as_depth | 7 of 8 |
       | step 7 | z_min_on_measured_depth | 7 of 8 |
       | step 7 | z_min_cut_strict | 1 of 8 (case 2) |
       | whole task | the six whole-task mutants of item 12h | unchanged: 3, 3, 3, 3, 1 (case 2) and 3 of 3 |

       Steps 1-5 mutants: unchanged results.

    h. Regression tests and reruns. tools/crown_check.py, all modes, every case alone in a fresh process, one
       BLAS/OpenMP thread, -j 2, on the shipped files: summary ALL OK:

       | Mode | Result |
       |---|---|
       | format | OK |
       | REF steps 1-7, general (solution.py) | 22/22, 12/12, 12/12, 7/7, 8/8, 7/7, 8/8, 3/3 |
       | SECOND steps 1-7, general | 22/22, 12/12, 12/12, 7/7, 8/8, 7/7, 8/8, 3/3 |
       | MUT | every step and whole-task mutant fails at least one case of its target tests (table g) |
       | SHIFT +-1e-12 (step outputs; every evaluated function in general) | all cases pass |
       | CONTROL None / 0.0 | 0 cases pass in every step and in general |

       Each new call takes 1 to 4 s with the reference (step 7 case 4 now 8.4 s with its five calls, case 7 3.6 s)
       and up to 14 s with the second solution (step 7 case 4). solution.py and second_solution.py were each run
       twice with one thread and once with two threads on the five new calls: identical SHA-256 digests.

       The verifier's edge-cell demonstration, its verify_ml.py on the new regions, its stress scripts for the
       maximization in g, the dense oracle scans and the error-budget runs above were rerun on the final files. The
       agent-visible text changed only in the sentence of item e.
    i. Not changed or open. relevant_experience is still the author's placeholder; the author writes it before
       upload (it is not written by the assistant). The difficulty band cannot be measured locally (item f).
