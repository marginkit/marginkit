"""Real-data regression for ``Fit.monotonicity`` on ``tests/data/zeta_matrix_counts.csv`` (plan
section 6.5; plan section 5.7: "PPO x sensor_noise must trigger this";
``docs/decisions/0023-fit-diagnostics-and-schema-v3.md``'s "Amendment 1").

Every series here is a genuine edge case from zeta-bench's committed matrix, not a synthetic
construction (mirrors ``tests/unit/test_models_zeta.py``'s own convention): a status or
monotonicity-flag change on this file is either a real statistical finding or a real bug --
never adjust an assertion here to make it pass.

Amendment 1 to decisions/0023 exists *because* of this exact series: ``ppo_sensor_noise`` never
reaches ``Status.OK`` through :func:`~robustmargin.fit_dose_response` under fixed ``upper=1.0``
(``CONTROL_INCOMPATIBLE``, the zero-severity control has failures) or under ``upper="estimate"``
(``SEPARATION``, quasi-complete at severity 0.01) -- see ``tests/unit/test_models_zeta.py``.
``Fit.monotonicity`` is computed from the raw ``Observations`` *before* a fitting path is even
chosen and is attached on every return path, so its reversal is visible on both of those
now-non-``OK`` fits directly, with no placeholder curve needed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from robustmargin import (
    Axis,
    Definition,
    Fit,
    MonotonicityCheck,
    Observations,
    Status,
    fit_dose_response,
    threshold,
)

_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "zeta_matrix_counts.csv"
_LEVEL = 0.95


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
    series_name: str, *, axis_name: str = "sensor_noise", axis_unit: str = "m"
) -> Observations:
    severity, successes, trials = _SERIES[series_name]
    axis = Axis(name=axis_name, unit=axis_unit, scale="log")
    return Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )


def _flagged_pairs(fit: Fit) -> list[tuple[float, float]]:
    assert fit.monotonicity is not None
    return [(p.severity_low, p.severity_high) for p in fit.monotonicity.pairs if p.flagged]


class TestPpoSensorNoiseUnderFixedUpperOne:
    """The rate rises from 57/200 (0.285) at severity 0.0 to 143/300 (0.477) at severity 0.01 --
    decisions/0023's own worked example (raw one-sided Fisher exact p roughly 1.2e-5)."""

    def _fit(self) -> Fit:
        obs = _observations_for("ppo_sensor_noise")
        return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

    def test_status_is_control_incompatible(self) -> None:
        assert self._fit().status is Status.CONTROL_INCOMPATIBLE

    def test_diagnostics_is_none(self) -> None:
        assert self._fit().diagnostics is None

    def test_monotonicity_is_not_none_and_flags_the_reversal(self) -> None:
        fit = self._fit()

        assert fit.monotonicity is not None
        assert (0.0, 0.01) in _flagged_pairs(fit)


class TestPpoSensorNoiseUnderEstimatedUpper:
    def _fit(self) -> Fit:
        obs = _observations_for("ppo_sensor_noise")
        return fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)

    def test_status_is_separation(self) -> None:
        assert self._fit().status is Status.SEPARATION

    def test_the_same_pair_is_flagged_whatever_the_status(self) -> None:
        fit = self._fit()

        assert fit.monotonicity is not None
        assert (0.0, 0.01) in _flagged_pairs(fit)

    def test_only_the_genuine_reversal_is_flagged_not_the_later_collapse(self) -> None:
        fit = self._fit()

        assert _flagged_pairs(fit) == [(0.0, 0.01)]


class TestPpoSensorNoiseThresholdsCarryNonMonotoneDataWarning:
    """decisions/0023 Amendment 1 item 7: ``threshold()`` adds a ``NON_MONOTONE_DATA: `` warning
    whenever ``fit.monotonicity.any_flagged``, whatever the resulting censoring or status."""

    def test_under_control_incompatible(self) -> None:
        obs = _observations_for("ppo_sensor_noise")
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit.status is Status.CONTROL_INCOMPATIBLE

        t = threshold(
            fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
        )

        assert any(w.startswith("NON_MONOTONE_DATA: ") for w in t.warnings)

    def test_under_separation(self) -> None:
        obs = _observations_for("ppo_sensor_noise")
        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
        assert fit.status is Status.SEPARATION

        t = threshold(
            fit, definition=Definition.absolute(0.4), interval_method="profile", level=_LEVEL
        )

        assert any(w.startswith("NON_MONOTONE_DATA: ") for w in t.warnings)


class TestPidSensorNoiseHasNeitherFlagNorWarning:
    """Monotonically non-increasing throughout (200/200, 299/300, 122/300, 0/300): a genuine,
    converged ``Status.OK`` fit, exercised end to end."""

    def _fit(self) -> Fit:
        obs = _observations_for("pid_sensor_noise")
        return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

    def test_fit_reaches_ok_with_both_diagnostic_fields_present(self) -> None:
        fit = self._fit()

        assert fit.status is Status.OK
        assert fit.diagnostics is not None
        assert fit.monotonicity is not None

    def test_no_pair_is_flagged(self) -> None:
        fit = self._fit()

        assert fit.monotonicity is not None
        assert fit.monotonicity.any_flagged is False
        assert _flagged_pairs(fit) == []

    def test_threshold_has_no_non_monotone_data_warning(self) -> None:
        fit = self._fit()

        t = threshold(
            fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
        )

        assert not any(w.startswith("NON_MONOTONE_DATA: ") for w in t.warnings)


# ------------------------------------------------------------------------------------------
# Every fit_dose_response return path carries a monotonicity check, whatever the Status --
# parametrized over the zeta fixture's own natural statuses plus two constructed cases
# (decisions/0023 Amendment 1's own "Tests (Phase 7)" bullet).
# ------------------------------------------------------------------------------------------


class TestEveryReturnPathCarriesMonotonicity:
    @pytest.mark.parametrize(
        ("series", "upper", "lower", "expected_status"),
        [
            ("pid_sensor_noise", 1.0, 0.0, Status.OK),
            ("ppo_sensor_noise", 1.0, 0.0, Status.CONTROL_INCOMPATIBLE),
            ("ppo_sensor_noise", "estimate", 0.0, Status.SEPARATION),
            ("sac_sensor_noise", 1.0, 0.0, Status.CONTROL_INCOMPATIBLE),
        ],
    )
    def test_monotonicity_present_on_zeta_series(
        self,
        series: str,
        upper: float | str,
        lower: float,
        expected_status: Status,
    ) -> None:
        obs = _observations_for(series)

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=upper, lower=lower)

        assert fit.status is expected_status
        assert fit.monotonicity is not None
        assert isinstance(fit.monotonicity, MonotonicityCheck)

    def test_monotonicity_present_on_a_constructed_separation_fit(self) -> None:
        axis = Axis(name="x", unit="unit", scale="linear")
        obs = Observations.from_counts(
            axis,
            severity=[0.0, 1.0, 2.0, 3.0],
            successes=[10, 10, 10, 0],
            trials=[10, 10, 10, 10],
            outcome="success",
            direction="decreasing",
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.status is Status.SEPARATION
        assert fit.monotonicity is not None

    def test_monotonicity_present_on_a_constructed_not_converged_fit(self) -> None:
        # A *structurally* NOT_CONVERGED input (decisions/0005's no-interior-maximum rule,
        # "wrong-sign slope" case), not one pinned by search: success genuinely rises with
        # severity (2/10 -> 5/10 -> 9/10), so the fitted GLM slope is unambiguously the wrong
        # sign for direction="decreasing" -- a deterministic sign check, not an optimizer
        # tolerance, so it holds in every environment. An earlier version of this test used a
        # 3-cell/3-parameter fixture "found by search" whose convergence flipped with platform
        # and library versions (see tests/unit/test_validation.py::TestLinkComparison's own
        # comment on the same underlying issue).
        axis = Axis(name="severity", unit="unit", scale="log")
        obs = Observations.from_counts(
            axis,
            severity=[1.0, 2.0, 4.0],
            successes=[2, 5, 9],
            trials=[10, 10, 10],
            outcome="success",
            direction="decreasing",
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.status is Status.NOT_CONVERGED
        assert fit.monotonicity is not None
