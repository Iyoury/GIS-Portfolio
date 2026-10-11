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
- the conventions (reference temperature, history intervals, z_min cut in
  TVD, OLS covariance);
- all test cases.

No public code was copied.

## Data

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
in version 10). No target is computed with solution.py or
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
each case. Since version 11, case 11 checks shallow depths (0 to 37 m) under a
deep layer stack against R = z/k1 and S = z²/(2k1), and R = S = 0 at the
surface (absolute 1e-12). R and S are elementary integrals: the second
solution (ramp superposition of the conductivity contrasts and integration by
parts, version 11 item 14) and the reference give the same values up to
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
- Since version 11, case 11 checks depths of 0.05 to 2.5 m under a recent
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
of the stored 90-digit values were recomputed independently in version 10.
Since version 11 the 50-layer column of case 6 (k from 1.5 to 5 W m^-1 K^-1,
20 intervals) has independent targets at six depths (30 m to 3000 m, the last
one in the bottom half-space); their computation is described in the version 11
section. Before version 11 that column was only checked for self-consistency.

**Step 6 and whole task (`tests/step_6.py`, `tests/general.py`).**

- *noiseless*: the truth is the generator input. solution.py recovers
  q0 = 0.041 to 4e-15 W m^-2 and second_solution.py to 6e-17 W m^-2
  (re-measured in version 11).
- *noisy* and *deep*: **no value is taken from the reference solution.** The
  expected estimates are recomputed at test time inside each case of
  `tests/step_6.py` and `tests/general.py` (`_independent_fit`). They come
  from `numpy.linalg.lstsq`, with the covariance from the pseudo-inverse,
  applied to the generator's own model terms: Gauss–Legendre TVD, exact layer
  sums for R and S, and `math.erfc` for P. They agree with `solution.py` and
  `second_solution.py` to 1e-10 W m^-2 and 1e-7 K (re-measured in version 11:
  1e-14 W m^-2 on q0, 3e-13 K on T0, 3e-12 K on g, 2e-13 relative on the
  standard errors). These values are valid
  for the stated 1-D model: conduction only, horizontal layers, uniform κ,
  and OLS with n − 3 degrees of freedom.
- *n_used* is the count of readings whose generator TVD is ≥ z_min, by
  definition. The value 46 in step 6 case 5 / general case 5 (z_min applied
  in vertical depth) is the count of generator TVDs in [150, 350) m of the
  noisy hole (an MD cut would remove 40 readings).
- Since version 11: step 6 case 6 puts a reading at the collar (MD 0, hence
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
  `target_provenance` field of the old-format problem.yaml; since version 10
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
per-case mutant results are in the version 11 table at the end of this file. The table compares the
reference with each mutant. Δq is in mW m^-2 and Δg
in K.

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
- step 5 (the workflow; step 6 since version 10) / whole task: a survey not starting at MD 0, a non-increasing survey, a non-increasing
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

Version 7: the dogleg rejection band is inclusive (a dogleg exactly 1e-6 rad from 180 degrees is rejected, as the prompt says); the step 1 and step 2 tests assert the container types (ndarray with the query shape; tuple of two Python floats or of two ndarrays); step 5 (the workflow; step 6 since version 10) tests invalid layer conductivities (zero, negative, non-finite, wrong length) through the whole workflow; the overview no longer mentions how the task is graded.

## Version 8 (difficulty increase; v7 rollouts solved 8/8)

New step 6 (step 5 since version 10), `layered_paleoclimate_perturbation(z, layer_tops, layer_k, rho_c, t_years, dT)`: the
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
layered_paleoclimate_perturbation, swapped in version 10): step 6 no longer reads the clock; the prompt states the size it must
handle (1000 depths, 50 layers, 20 intervals in one call) and that case is now checked on values (50 equal
conductivities against the erfc half-space, varying conductivities against single-depth calls). Step 6 also
rejects two-dimensional and empty layer arrays in the tests. Step 4 fits a resolvable four-reading design
(smallest valid n). Step 5 has a vertical hole (TVD = MD exactly) with a reading exactly at z_min, which must
be kept (n_used = 131).

## Version 10 (grading_fix: authoring-guide conformance)

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

crown_check (tools/crown_check.py, every case alone in a fresh process, -j 2), final run of version 10
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
numbers (workflow = step 5 then); this table replaced it in version 10 and is itself superseded by the
version 11 table below. Every section before "Version 10" uses the old step numbering (v5-v7: workflow =
step 5; v8-v9: workflow = step 5 and layered step = step 6); since version 10 the layered step is step 5
and the workflow step 6.

## Version 11 (grading_fix: authoring-guide conformance, pass 2)

Second conformance pass on version 10 (rules on whole-task tests, hidden inputs, truthfulness, independent
second solution, current-reference mutants, stated/tested consistency), with the findings of the independent
verifier of version 10. Version 10 is kept (self-contained cases, no clock reads, no exact float equality,
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
   years; targets from math.erfc): the version 10 second solution fails it, the new one passes. Sweep against
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
   scalar z = 0; R = z/k1 and S = z^2/(2 k1) to a relative 1e-9, R = S = 0 within 1e-12): the version 10
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
    already corrected by item 13 of version 10; the evidence section now also lists the version 11 cases and
    re-measured numbers. The tolerance-rationale, error-contract and registered-mutant sections are labelled as
    historical with the versions and step numbers they used, and the version 10 crown_check table is marked as
    the version 10 run. Completion of the version 10 notes: item 5 (dependencies), cause: quad was imported
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
run of version 11 (after item 14), all modes, summary ALL OK:

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
