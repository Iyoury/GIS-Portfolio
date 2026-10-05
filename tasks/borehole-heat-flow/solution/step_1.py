"""Reference for step 1."""
import numpy as np
import math

_erfc = np.frompyfunc(math.erfc, 1, 1)


def erfc(x):
    """Element-wise complementary error function (numpy + math only)."""
    return np.asarray(_erfc(np.asarray(x, float)), dtype=float)

SECONDS_PER_YEAR = 365.25 * 86400.0


def _unit(inc, azi):
    return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])


def _sinc(x):
    return np.sinc(x / np.pi)          # sin(x) / x, exact at x = 0


def _ratio_factor(beta):
    h = 0.5 * beta                     # tan(h) / h, no straight-line shortcut
    if h < 1e-4:
        return 1.0 + h * h / 3.0 + 2.0 * h ** 4 / 15.0   # series, truncation < 1e-24
    return np.tan(h) / h


def true_vertical_depth(survey_md, inc_deg, azi_deg, md_query):
    """TVD (m) at md_query by the minimum-curvature method."""
    md = np.asarray(survey_md, float)
    inc = np.radians(np.asarray(inc_deg, float))
    azi = np.radians(np.asarray(azi_deg, float))
    q_in = np.asarray(md_query, float)
    q = np.atleast_1d(q_in).ravel()
    if not (md.shape == inc.shape == azi.shape) or md.ndim != 1 or md.size < 2:
        raise ValueError("survey arrays must be 1-D, equal length, >= 2 stations")
    if not (np.all(np.isfinite(md)) and np.all(np.isfinite(inc)) and np.all(np.isfinite(azi))):
        raise ValueError("survey values must be finite")
    if md[0] != 0.0 or np.any(np.diff(md) <= 0):
        raise ValueError("survey must start at MD 0 and increase strictly")
    if np.any(inc < 0) or np.any(inc >= np.pi):
        raise ValueError("inclination must be in [0, 180) degrees")
    if np.any(~np.isfinite(q)) or np.any(q < 0) or np.any(q > md[-1]):
        raise ValueError("query depths must lie within the survey")
    # cumulative TVD at stations
    tvd_st = np.zeros(md.size)
    units = [_unit(inc[i], azi[i]) for i in range(md.size)]
    betas = np.zeros(md.size - 1)
    for i in range(md.size - 1):
        # atan2 keeps full precision for tiny doglegs (arccos of a dot product does not)
        betas[i] = np.arctan2(np.linalg.norm(np.cross(units[i], units[i + 1])), units[i] @ units[i + 1])
        if betas[i] >= np.pi - 1e-6:
            raise ValueError("dogleg of 180 degrees between stations: path undefined")
        dmd = md[i + 1] - md[i]
        tvd_st[i + 1] = tvd_st[i] + 0.5 * dmd * (units[i][2] + units[i + 1][2]) * _ratio_factor(betas[i])
    out = np.empty(q.size)
    for j, m in enumerate(q):
        i = min(np.searchsorted(md, m, side="right") - 1, md.size - 2)
        dmd_seg = md[i + 1] - md[i]
        f = (m - md[i]) / dmd_seg
        b = betas[i]
        t1, t2 = units[i], units[i + 1]
        # tangent on the arc; the sinc form is exact for every dogleg, including 0
        tq = ((1 - f) * _sinc((1 - f) * b) * t1 + f * _sinc(f * b) * t2) / _sinc(b)
        dm = m - md[i]
        out[j] = tvd_st[i] + 0.5 * dm * (t1[2] + tq[2]) * _ratio_factor(f * b)
    return float(out[0]) if q_in.ndim == 0 else out.reshape(q_in.shape)
