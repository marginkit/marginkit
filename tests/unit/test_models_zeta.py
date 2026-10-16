"""Expected ``fit_dose_response`` statuses on ``tests/data/zeta_matrix_counts.csv`` (Phase 4
plan's worked table; plan section 3.1's edge-case matrix).

Real-data regression coverage (plan section 6.5): every series here is a genuine edge case from
zeta-bench's committed matrix, not a synthetic construction, so a status change on this file is
either a real statistical finding or a real bug -- never adjust an assertion here to make it
pass (``CLAUDE.md``).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from robustmargin import Axis, Observations, Status, fit_dose_response

_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "zeta_matrix_counts.csv"


def _load_series() -> dict[str, tuple[list[float], list[int], list[int]]]:
    series: dict[str, tuple[list[float], list[int], list[int]]] = {}
    with _CSV_PATH.open(newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["series"]
            severity, successes, trials = series.setdefault(name, ([], [], []))
            severity.append(float(row["severity"]))
            successes.append(int(row["successes"]))
            trials.append(int(row["trials"]))
    return series


_SERIES = _load_series()


def _observations_for(
    series_name: str,
    *,
    axis_name: str = "sensor_noise",
    axis_unit: str = "m",
    axis_scale: str = "log",
) -> Observations:
    severity, successes, trials = _SERIES[series_name]
    axis = Axis(name=axis_name, unit=axis_unit, scale=axis_scale)
    return Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )


def test_pid_sensor_noise_u_1_l_0_is_ok_with_finite_params() -> None:
    obs = _observations_for("pid_sensor_noise")

    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

    assert fit.status is Status.OK
    assert fit.params is not None
    for name in ("mu", "s", "upper", "lower"):
        assert math.isfinite(fit.params[name].value)


def test_pid_wind_u_1_l_0_is_separation() -> None:
    obs = _observations_for("pid_wind", axis_name="wind", axis_unit="m/s", axis_scale="linear")

    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

    assert fit.status is Status.SEPARATION


def test_pid_wind_upper_estimate_is_separation() -> None:
    obs = _observations_for("pid_wind", axis_name="wind", axis_unit="m/s", axis_scale="linear")

    fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)

    assert fit.status is Status.SEPARATION


def test_sac_sensor_noise_u_1_is_control_incompatible() -> None:
    obs = _observations_for("sac_sensor_noise")

    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

    assert fit.status is Status.CONTROL_INCOMPATIBLE
    assert any("estimate" in warning.lower() for warning in fit.warnings)


def test_sac_sensor_noise_upper_estimate_is_separation() -> None:
    obs = _observations_for("sac_sensor_noise")

    fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)

    assert fit.status is Status.SEPARATION


def test_ppo_sensor_noise_u_1_is_control_incompatible() -> None:
    obs = _observations_for("ppo_sensor_noise")

    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

    assert fit.status is Status.CONTROL_INCOMPATIBLE


def test_ppo_sensor_noise_upper_estimate_is_separation_quasi_complete_at_0_01() -> None:
    """Plan's own note: quasi-complete separation at severity 0.01, where every success in the
    nonzero levels is concentrated (0.01: 143/300 success, but 0.05 and 0.1 are both 0/300).
    """
    obs = _observations_for("ppo_sensor_noise")

    fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)

    assert fit.status is Status.SEPARATION


def _ppo_mass_by_magnitude() -> tuple[Observations, Observations]:
    """Split the signed ``ppo_mass`` series into two magnitude series (R6: a signed axis is
    handled by fitting each side as its own axis), each sharing the severity=0.0 row, on a
    LINEAR axis.
    """
    severity, successes, trials = _SERIES["ppo_mass"]
    by_severity = dict(zip(severity, zip(successes, trials, strict=True), strict=True))

    control_successes, control_trials = by_severity[0.0]

    positive_axis = Axis(name="mass_positive", unit="kg", scale="linear")
    positive = Observations.from_counts(
        positive_axis,
        severity=[0.0, 0.1, 0.2],
        successes=[control_successes, *(by_severity[s][0] for s in (0.1, 0.2))],
        trials=[control_trials, *(by_severity[s][1] for s in (0.1, 0.2))],
        outcome="success",
        direction="decreasing",
    )

    negative_axis = Axis(name="mass_negative", unit="kg", scale="linear")
    negative = Observations.from_counts(
        negative_axis,
        severity=[0.0, 0.1, 0.2],
        successes=[control_successes, *(by_severity[-s][0] for s in (0.1, 0.2))],
        trials=[control_trials, *(by_severity[-s][1] for s in (0.1, 0.2))],
        outcome="success",
        direction="decreasing",
    )
    return positive, negative


def test_ppo_mass_magnitude_split_matches_the_documented_counts() -> None:
    """A guard on the fixture itself, not on ``fit_dose_response``: confirms the split this
    file performs reproduces the exact counts the Phase 4 plan states before any fit is run.
    """
    positive, negative = _ppo_mass_by_magnitude()

    assert positive.severity == (0.0, 0.1, 0.2)
    assert positive.successes == (99, 33, 0)
    assert positive.trials == (100, 100, 100)

    assert negative.severity == (0.0, 0.1, 0.2)
    assert negative.successes == (99, 77, 5)
    assert negative.trials == (100, 100, 100)


def test_ppo_mass_positive_magnitude_u_1_l_0_is_ok() -> None:
    positive, _negative = _ppo_mass_by_magnitude()

    fit = fit_dose_response(positive, model="binomial", link="probit", upper=1.0, lower=0.0)

    assert fit.status is Status.OK


def test_ppo_mass_negative_magnitude_u_1_l_0_is_ok() -> None:
    _positive, negative = _ppo_mass_by_magnitude()

    fit = fit_dose_response(negative, model="binomial", link="probit", upper=1.0, lower=0.0)

    assert fit.status is Status.OK
