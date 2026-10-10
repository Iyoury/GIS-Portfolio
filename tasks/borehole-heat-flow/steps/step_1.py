"""Step 1 - true vertical depth from a deviation survey (minimum curvature)."""
import numpy as np


def true_vertical_depth(survey_md, inc_deg, azi_deg, md_query):
    """True vertical depth (m, positive down) at measured depths md_query.

    Parameters
    ----------
    survey_md : 1-D array_like, measured depth of survey stations (m); first
        station at 0 (collar), strictly increasing, at least 2 stations.
    inc_deg : 1-D array_like, inclination from vertical at each station
        (degrees, 0 = straight down), 0 <= inc < 180.
    azi_deg : 1-D array_like, azimuth at each station (degrees clockwise from
        north).
    md_query : float or array_like, measured depths (m) with
        0 <= md_query <= survey_md[-1].

    Returns
    -------
    Python float for scalar md_query, otherwise a numpy float array with the
    shape of md_query. Absolute accuracy 1e-6 m (the minimum-curvature path
    is exact; no approximation between stations is permitted).

    Raises
    ------
    ValueError if the survey arrays are not 1-D of equal length, have fewer
    than 2 stations, contain a non-finite value, do not start at MD 0 or do
    not increase strictly, if any inclination is outside [0, 180), if the
    dogleg between two consecutive stations is within 1e-6 rad of 180 degrees
    (opposite directions: the arc is undefined), or if any query depth is
    non-finite or outside [0, survey_md[-1]].
    """
    raise NotImplementedError
