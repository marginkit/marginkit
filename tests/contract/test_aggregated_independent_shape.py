"""Contract test: the aggregated, independent call shape zeta-bench uses.

Mirrors how zeta-bench's ``robustness/cards.py`` calls the ported grid rule: a degradation
curve as a list of ``(severity, success_rate)`` pairs, unzipped into two lists, run through
the public API with ``criterion=0.95, signed=True``, then read back via ``.value`` and
``.max_tested`` (see ``build_card_summary`` / ``_format_break`` in
``../zeta-bench/robustness/cards.py``). Only the public API surface is used here — no internal
robustmargin module is imported.

A failing test in this file means a breaking change to the public API that zeta-bench depends
on (plan section 3.3, ``docs/CONSUMERS.md``). **Do not edit this test to make it pass.** Stop
and ask; that failure is the signal a breaking change needs owner sign-off.

**Edit recorded (decisions/0023, item 2):** ``schema_version`` is now ``"3"`` on every result
type, including ``GridBreakPoint``. The two literal ``"2"`` assertions this file made against a
freshly-built ``GridBreakPoint``'s serialised ``schema_version`` were updated to ``"3"`` --
zeta-bench itself is unaffected (it unpacks ``grid_break_point``'s return to a plain
``(value, max_tested)`` tuple and never serialises the dataclass at all), but the two assertions
here are the contract for anyone who does.

**Edit recorded (decisions/0027):** the package is renamed ``robustmargin`` and
``schema_version`` is now ``"4"`` on every result type, including ``GridBreakPoint``. The two
literal ``"3"`` assertions this file made against a freshly-built ``GridBreakPoint``'s serialised
``schema_version`` were updated to ``"4"``, and the imports moved to ``robustmargin``. This is a
breaking change signed off in ``decisions/0027`` (Appendix B item 2) -- every import breaks and
every stored ``"3"`` card is rejected by the ``"4"``-only reader -- not an edit made to turn a
failing contract test green. zeta-bench is unaffected beyond its import (``from robustmargin
import grid_break_point``): it never serialises ``GridBreakPoint``, so no card it holds changes.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from inspect import Parameter

from robustmargin import Censoring, grid_break_point


def _load_matrix_counts() -> dict[str, tuple[list[float], list[int], list[int]]]:
    """Read ``tests/data/zeta_matrix_counts.csv`` into per-series ``(severity, successes,
    trials)`` arrays, as ``Observations.from_counts`` needs them (unlike the module-level
    ``_load_matrix_series`` below, which collapses each row to a rate for the grid rule).

    Imports locally rather than at module level, matching the Phase 3 additions further down:
    this file's original imports (above) stay untouched, so the diff against the pre-Phase-3
    file is additions only.
    """
    import csv
    from pathlib import Path

    matrix_csv = Path(__file__).resolve().parents[1] / "data" / "zeta_matrix_counts.csv"
    series: dict[str, tuple[list[float], list[int], list[int]]] = {}
    with matrix_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["series"]
            severities, successes, trials = series.setdefault(name, ([], [], []))
            severities.append(float(row["severity"]))
            successes.append(int(row["successes"]))
            trials.append(int(row["trials"]))
    return series


def test_zeta_bench_shaped_call_reads_value_and_max_tested() -> None:
    """Unzip a ``(severity, rate)`` curve, call with keyword ``criterion``, read the result."""
    curve: list[tuple[float, float]] = [
        (-0.2, 0.05),
        (-0.1, 0.77),
        (0.0, 0.99),
        (0.1, 0.33),
        (0.2, 0.0),
    ]
    severity, rate = zip(*curve)

    result = grid_break_point(list(severity), list(rate), criterion=0.95, signed=True)

    assert result.value == 0.1
    assert result.max_tested == 0.2


def test_zeta_bench_shaped_call_reports_right_censoring_as_none_value() -> None:
    """When the gate holds everywhere, ``.value`` is ``None`` and ``.max_tested`` is the bound.

    This is the shape zeta-bench's ``_format_break`` relies on: ``bp is None`` selects the
    "holds <= max tested" wording rather than a numeric break-point.
    """
    curve: list[tuple[float, float]] = [(0.0, 1.0), (2.0, 1.0), (5.0, 1.0), (10.0, 1.0)]
    severity, rate = zip(*curve)

    result = grid_break_point(list(severity), list(rate), criterion=0.95, signed=True)

    assert result.value is None
    assert result.max_tested == 10.0


def test_censoring_has_exactly_the_five_documented_members() -> None:
    """The tag freezes at ``v0.1.0a1``: no member added, removed, or renamed after that."""
    members = {member.name: member.value for member in Censoring}

    assert members == {
        "NONE": "NONE",
        "RIGHT": "RIGHT",
        "LEFT": "LEFT",
        "OPEN_UPPER": "OPEN_UPPER",
        "OPEN_LOWER": "OPEN_LOWER",
    }


def test_zeta_bench_shaped_call_reports_censoring_none_when_curve_fails() -> None:
    curve: list[tuple[float, float]] = [
        (-0.2, 0.05),
        (-0.1, 0.77),
        (0.0, 0.99),
        (0.1, 0.33),
        (0.2, 0.0),
    ]
    severity, rate = zip(*curve)

    result = grid_break_point(list(severity), list(rate), criterion=0.95, signed=True)

    assert result.censoring is Censoring.NONE


def test_zeta_bench_shaped_call_reports_censoring_right_with_none_value_when_curve_holds() -> None:
    curve: list[tuple[float, float]] = [(0.0, 1.0), (2.0, 1.0), (5.0, 1.0), (10.0, 1.0)]
    severity, rate = zip(*curve)

    result = grid_break_point(list(severity), list(rate), criterion=0.95, signed=True)

    assert result.censoring is Censoring.RIGHT
    assert result.value is None


def test_result_round_trips_through_json_with_string_censoring_and_schema_version() -> None:
    curve: list[tuple[float, float]] = [
        (-0.2, 0.05),
        (-0.1, 0.77),
        (0.0, 0.99),
        (0.1, 0.33),
        (0.2, 0.0),
    ]
    severity, rate = zip(*curve)
    result = grid_break_point(list(severity), list(rate), criterion=0.95, signed=True)

    dumped = json.dumps(dataclasses.asdict(result))
    reloaded = json.loads(dumped)

    assert reloaded["censoring"] in ("NONE", "RIGHT")
    assert reloaded["censoring"] == "NONE"
    # decisions/0022 bumped schema_version to "2" on every result type, GridBreakPoint
    # included: one schema document means one version. decisions/0023 bumped it again, to "3"
    # (Fit.diagnostics), and decisions/0027 to "4" (the rename), still package-wide. zeta-bench is
    # unaffected -- it unpacks grid_break_point to a plain (value, max_tested) tuple and never
    # serialises this dataclass -- but this assertion is the contract for anyone who does.
    assert reloaded["schema_version"] == "4"


def test_right_censored_result_round_trips_through_json_too() -> None:
    curve: list[tuple[float, float]] = [(0.0, 1.0), (2.0, 1.0), (5.0, 1.0), (10.0, 1.0)]
    severity, rate = zip(*curve)
    result = grid_break_point(list(severity), list(rate), criterion=0.95, signed=True)

    dumped = json.dumps(dataclasses.asdict(result))
    reloaded = json.loads(dumped)

    assert reloaded["censoring"] == "RIGHT"
    assert reloaded["schema_version"] == "4"  # decisions/0022, 0023, 0027, as above


def test_criterion_is_keyword_only_and_signed_defaults_to_false() -> None:
    signature = inspect.signature(grid_break_point)

    assert signature.parameters["criterion"].kind is Parameter.KEYWORD_ONLY
    assert signature.parameters["signed"].default is False


# ---------------------------------------------------------------------------------------------
# Phase 3 additions (plan section 7 Phase 3 DoD): the contract types, the JSON report module,
# and the fakes. Everything below is new; nothing above this line was touched (`git diff` shows
# additions only). Each new test imports the Phase 3 names it needs locally rather than at
# module level, for the same reason.
# ---------------------------------------------------------------------------------------------


def test_axis_and_from_counts_construction_on_pid_sensor_noise_counts() -> None:
    """The plan section 4.1 API sketch's own worked example: ``Axis`` plus
    ``Observations.from_counts`` on the PID x sensor_noise counts."""
    from robustmargin import Axis, Cell, Observations

    axis = Axis(name="sensor_noise", unit="m", scale="log")
    obs = Observations.from_counts(
        axis,
        severity=[0.0, 0.01, 0.05, 0.1],
        successes=[200, 299, 122, 0],
        trials=[200, 300, 300, 300],
        outcome="success",
        direction="decreasing",
    )

    assert obs.cells() == (
        Cell(severity=0.0, successes=200, trials=200),
        Cell(severity=0.01, successes=299, trials=300),
        Cell(severity=0.05, successes=122, trials=300),
        Cell(severity=0.1, successes=0, trials=300),
    )


def test_definition_absolute_0_95_stores_value() -> None:
    from robustmargin import Definition

    definition = Definition.absolute(0.95)

    assert definition.kind == "absolute"
    assert definition.value == 0.95


def test_scorecard_of_fakes_round_trips_through_json_text() -> None:
    """The consumer-shaped path: build a ``Scorecard``, serialise it to a JSON string, parse it
    back, and reload the result objects -- not just ``to_dict``/``from_dict`` in memory."""
    from robustmargin import Scorecard
    from robustmargin.report import from_dict, to_dict
    from robustmargin.testing import fake_fit, fake_ratio, fake_threshold

    card = Scorecard(
        results=(
            grid_break_point([0.0, 5.0, 10.0], [1.0, 0.95, 0.4], criterion=0.95),
            fake_fit(),
            fake_threshold(),
            fake_ratio(),
        ),
        provenance={},
    )

    text = json.dumps(to_dict(card))
    reloaded = from_dict(json.loads(text))

    assert reloaded == card


def test_pid_sensor_noise_absolute_0_95_gives_a_finite_bracketed_interval() -> None:
    """Worked case from plan section 5.5: the identifiable PID x sensor_noise transition."""
    from robustmargin import Axis, Definition, Observations, fit_dose_response, threshold

    severity, successes, trials = _load_matrix_counts()["pid_sensor_noise"]
    axis = Axis(name="sensor_noise", unit="m", scale="log")
    obs = Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
    t = threshold(fit, definition=Definition.absolute(0.95), interval_method="profile", level=0.95)

    assert t.censoring is Censoring.NONE
    assert t.lo is not None
    assert t.value is not None
    assert t.hi is not None
    assert t.lo <= t.value <= t.hi
    assert 0.01 < t.value < 0.1


def test_pid_wind_absolute_0_95_is_right_censored_at_10() -> None:
    """Worked case from plan section 5.5: 500/500 at every tested wind level."""
    from robustmargin import Axis, Definition, Observations, fit_dose_response, threshold

    severity, successes, trials = _load_matrix_counts()["pid_wind"]
    axis = Axis(name="wind", unit="m/s", scale="linear")
    obs = Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
    t = threshold(fit, definition=Definition.absolute(0.95), interval_method="profile", level=0.95)

    assert t.censoring is Censoring.RIGHT
    assert t.value is None
    assert t.lo == 10.0


def test_sac_sensor_noise_absolute_0_95_fails_at_baseline() -> None:
    """Worked case from plan section 5.5: the SAC control is already at 98/200."""
    from robustmargin import Axis, Definition, Observations, Status, fit_dose_response, threshold

    severity, successes, trials = _load_matrix_counts()["sac_sensor_noise"]
    axis = Axis(name="sensor_noise", unit="m", scale="log")
    obs = Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
    t = threshold(fit, definition=Definition.absolute(0.95), interval_method="profile", level=0.95)

    assert t.status is Status.FAILS_AT_BASELINE


def test_sac_sensor_noise_baseline_fraction_0_5_is_left_censored_at_0_01() -> None:
    """Worked case from plan section 5.5: the lowest nonzero level already fails."""
    from robustmargin import Axis, Definition, Observations, fit_dose_response, threshold

    severity, successes, trials = _load_matrix_counts()["sac_sensor_noise"]
    axis = Axis(name="sensor_noise", unit="m", scale="log")
    obs = Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
    t = threshold(
        fit, definition=Definition.baseline_fraction(0.5), interval_method="profile", level=0.95
    )

    assert t.censoring is Censoring.LEFT
    assert t.hi == 0.01


def test_fieller_ratio_on_pid_sensor_noise_halves_propagates_separation() -> None:
    """Two thresholds fit on *disjoint* halves of the PID x sensor_noise counts, on the same
    axis and the same ``absolute(0.95)`` definition, combined with a Fieller interval.

    **This fixture does not give two clean ``OK`` fits, and an earlier version of this docstring
    was wrong to say it did.** The two halves are constructed so that, summed cell by cell, they
    reproduce the fixture's own ``pid_sensor_noise`` counts (successes ``[200, 299, 122, 0]``
    over trials ``[200, 300, 300, 300]``): half A is ``[100, 150, 61, 0]`` over
    ``[100, 150, 150, 150]`` and half B is the complementary ``[100, 149, 61, 0]`` over
    ``[100, 150, 150, 150]``, so the two fits genuinely share no observations -- but half A is
    genuinely quasi-completely separated (confirmed: its likelihood is flat at its supremum from
    ``s = 0.06`` down to ``s = 1e-8``, so no finite MLE exists there, and
    ``decisions/0006``'s pre-fit separation rule is correct to say so). This is not repairable by
    re-splitting: ``pid_sensor_noise`` has exactly one failure below its mixed level
    (``299/300`` at ``0.01``), so *any* disjoint split leaves one half separated, and the only
    other candidate series in the fixture, ``ppo_sensor_noise``, is separated as a whole series.

    What this fixture is actually worth testing, then, is not "the OK/OK Fieller path" (see
    ``test_fieller_ratio_on_two_constructed_ok_thresholds_reports_an_interval_shape`` below for
    that) but that ``Status.SEPARATION`` on real, consumer-shaped zeta-bench data propagates
    through ``ratio_interval`` exactly as ``decisions/0015`` requires: a flag, never a number
    that merely happens to look plausible.

    ``dependence="independent"`` (plan section 5.6) is valid only when the two thresholds come
    from disjoint data, and must raise otherwise -- so this must not be two thresholds off one
    fit, which would share every observation.
    """
    from robustmargin import (
        Axis,
        Definition,
        Observations,
        Status,
        fit_dose_response,
        ratio_interval,
        threshold,
    )

    severity, _successes, _trials = _load_matrix_counts()["pid_sensor_noise"]
    axis = Axis(name="sensor_noise", unit="m", scale="log")

    half_a = Observations.from_counts(
        axis,
        severity=severity,
        successes=[100, 150, 61, 0],
        trials=[100, 150, 150, 150],
        outcome="success",
        direction="decreasing",
    )
    half_b = Observations.from_counts(
        axis,
        severity=severity,
        successes=[100, 149, 61, 0],
        trials=[100, 150, 150, 150],
        outcome="success",
        direction="decreasing",
    )

    fit_a = fit_dose_response(half_a, model="binomial", link="probit", upper=1.0, lower=0.0)
    fit_b = fit_dose_response(half_b, model="binomial", link="probit", upper=1.0, lower=0.0)
    assert fit_a.status is Status.SEPARATION
    assert fit_b.status is Status.OK
    t_a = threshold(
        fit_a, definition=Definition.absolute(0.95), interval_method="profile", level=0.95
    )
    t_b = threshold(
        fit_b, definition=Definition.absolute(0.95), interval_method="profile", level=0.95
    )
    assert t_a.status is Status.SEPARATION

    r = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

    assert r.status is Status.SEPARATION
    assert r.shape is None
    assert r.estimate is None
    assert r.lo is None
    assert r.hi is None


def test_fieller_ratio_on_two_constructed_ok_thresholds_reports_an_interval_shape() -> None:
    """The OK/OK Fieller contract path: since ``pid_sensor_noise`` cannot supply two clean
    ``OK`` fits from disjoint halves (see the test above), this covers the same shape-reporting
    contract -- ``ratio_interval`` on two independent, converged ``Status.OK`` thresholds returns
    some :class:`~robustmargin.IntervalShape` -- from two small but genuinely non-separated,
    non-shared datasets on the public API surface.
    """
    from robustmargin import (
        Axis,
        Definition,
        IntervalShape,
        Observations,
        Status,
        fit_dose_response,
        ratio_interval,
        threshold,
    )

    axis = Axis(name="sensor_noise", unit="m", scale="log")
    obs_a = Observations.from_counts(
        axis,
        severity=[0.0, 0.01, 0.05, 0.1],
        successes=[100, 90, 60, 20],
        trials=[100, 100, 100, 100],
        outcome="success",
        direction="decreasing",
    )
    obs_b = Observations.from_counts(
        axis,
        severity=[0.0, 0.02, 0.1, 0.2],
        successes=[100, 85, 45, 10],
        trials=[100, 100, 100, 100],
        outcome="success",
        direction="decreasing",
    )
    fit_a = fit_dose_response(obs_a, model="binomial", link="probit", upper=1.0, lower=0.0)
    fit_b = fit_dose_response(obs_b, model="binomial", link="probit", upper=1.0, lower=0.0)
    assert fit_a.status is Status.OK
    assert fit_b.status is Status.OK
    t_a = threshold(
        fit_a, definition=Definition.absolute(0.5), interval_method="profile", level=0.95
    )
    t_b = threshold(
        fit_b, definition=Definition.absolute(0.5), interval_method="profile", level=0.95
    )
    assert t_a.status is Status.OK and t_b.status is Status.OK

    r = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

    assert isinstance(r.shape, IntervalShape)
