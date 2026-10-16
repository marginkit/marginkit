"""Unit tests for ``robustmargin.empirical.grid_break_point``.

The first test class, ``TestPortedFromZetaBench``, ports ``TestBreakPoint`` from
``../zeta-bench/tests/test_cards.py`` (lines 91-106), which exercises
``robustness/cards.py::break_point`` (lines 124-139), at zeta-bench commit
``435fc2793b2afdb8404099c83b0c36a71d9ffba9`` (PR #17), per
``docs/decisions/0002-provenance-by-copy.md``. Case names are domain-neutral; the signed-axis
case reuses the literal curve values from ``test_cards.py::_synthetic_rows``'s ``mass`` fixture
(lines 56-60) rather than re-deriving ``degradation_curve``'s pooling, since that pooling
itself is not part of the ported API.

Everything after ``TestPortedFromZetaBench`` is new to robustmargin: the deliberate divergences
from zeta-bench (``ValueError`` instead of a silent ``(None, None)``, keyword-only ``criterion``,
a frozen result, built-in ``float`` fields) and the real-data regression over
``tests/data/zeta_matrix_counts.csv``.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pytest

from robustmargin import Censoring, GridBreakPoint, grid_break_point

_DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "zeta_matrix_counts.csv"


class TestPortedFromZetaBench:
    """Direct ports of ``TestBreakPoint`` (test_cards.py lines 91-106)."""

    def test_first_failing_magnitude(self) -> None:
        """Smallest severity whose performance is strictly below the criterion wins.

        Ports ``test_first_failing_magnitude``: the point at 5.0 sits exactly on the
        criterion (0.95) and must pass, since the rule is strict ``<``.
        """
        result = grid_break_point([0.0, 5.0, 10.0], [1.0, 0.95, 0.4], criterion=0.95)

        assert result.value == 10.0
        assert result.max_tested == 10.0
        assert result.censoring is Censoring.NONE

    def test_signed_axis_breaks_on_failing_side(self) -> None:
        """A signed axis breaks on whichever side fails first, by magnitude.

        Ports ``test_signed_family_breaks_on_failing_side``, using the literal curve from
        ``_synthetic_rows``'s ``mass`` fixture: the -0.1 side fails (0.9 < 0.95) before the
        +/-0.2 points do.
        """
        severity = [-0.2, -0.1, 0.0, 0.1, 0.2]
        performance = [0.2, 0.9, 1.0, 1.0, 1.0]

        result = grid_break_point(severity, performance, criterion=0.95, signed=True)

        assert result.value == 0.1
        assert result.max_tested == 0.2
        assert result.censoring is Censoring.NONE

    def test_right_censored_when_criterion_holds_everywhere(self) -> None:
        """Ports ``test_censored_when_gate_holds_everywhere``: no failing level at all."""
        result = grid_break_point([0.0, 5.0, 10.0], [1.0, 1.0, 0.99], criterion=0.95)

        assert result.value is None
        assert result.max_tested == 10.0
        assert result.censoring is Censoring.RIGHT

    def test_empty_input_raises(self) -> None:
        """Diverges from zeta-bench's ``break_point([], gate=...) == (None, None)``.

        robustmargin has no silent fallback (CLAUDE.md hard constraint 3): empty input raises
        rather than returning a plausible-looking result.
        """
        with pytest.raises(ValueError):
            grid_break_point([], [], criterion=0.95)


class TestValidation:
    """Every failure path the API contract lists, one behaviour per test."""

    def test_mismatched_lengths_raise_value_error(self) -> None:
        with pytest.raises(ValueError):
            grid_break_point([0.0, 1.0, 2.0], [1.0, 0.5], criterion=0.95)

    def test_non_1d_severity_raises_value_error(self) -> None:
        severity = [[0.0, 1.0], [2.0, 3.0]]
        performance = [1.0, 0.5]

        with pytest.raises(ValueError):
            grid_break_point(severity, performance, criterion=0.95)

    def test_non_1d_performance_raises_value_error(self) -> None:
        severity = [0.0, 2.0]
        performance = [[1.0, 1.0], [0.5, 0.5]]

        with pytest.raises(ValueError):
            grid_break_point(severity, performance, criterion=0.95)

    @pytest.mark.parametrize(
        ("severity", "performance", "criterion"),
        [
            pytest.param([0.0, float("nan"), 5.0], [1.0, 0.9, 0.4], 0.95, id="nan-severity"),
            pytest.param([0.0, 2.0, 5.0], [1.0, float("nan"), 0.4], 0.95, id="nan-performance"),
            pytest.param([0.0, 2.0, 5.0], [1.0, 0.9, 0.4], float("nan"), id="nan-criterion"),
            pytest.param([0.0, float("inf"), 5.0], [1.0, 0.9, 0.4], 0.95, id="inf-severity"),
            pytest.param([0.0, -float("inf"), 5.0], [1.0, 0.9, 0.4], 0.95, id="neg-inf-severity"),
            pytest.param([0.0, 2.0, 5.0], [1.0, float("inf"), 0.4], 0.95, id="inf-performance"),
            pytest.param([0.0, 2.0, 5.0], [1.0, 0.9, 0.4], float("inf"), id="inf-criterion"),
        ],
    )
    def test_non_finite_values_raise_value_error(
        self, severity: list[float], performance: list[float], criterion: float
    ) -> None:
        with pytest.raises(ValueError):
            grid_break_point(severity, performance, criterion=criterion)

    def test_negative_severity_unsigned_raises_value_error(self) -> None:
        """``signed=False`` is the default: a negative severity is a caller bug, not data."""
        with pytest.raises(ValueError):
            grid_break_point([-1.0, 0.0, 5.0], [1.0, 1.0, 0.4], criterion=0.95)

    def test_negative_severity_signed_is_accepted(self) -> None:
        """The mirror of the previous case: ``signed=True`` allows negative severities."""
        result = grid_break_point([-1.0, 0.0, 5.0], [0.4, 1.0, 1.0], criterion=0.95, signed=True)

        assert result.value == 1.0

    def test_criterion_is_keyword_only(self) -> None:
        """Passing ``criterion`` positionally is a ``TypeError``, not a lenient accept."""
        bad_call = cast(Callable[..., GridBreakPoint], grid_break_point)
        with pytest.raises(TypeError):
            bad_call([0.0, 5.0], [1.0, 0.4], 0.95)

    def test_result_is_frozen(self) -> None:
        result = GridBreakPoint(value=5.0, max_tested=10.0, censoring=Censoring.NONE)

        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, result).value = 1.0


class TestStrictInequalityAndTypes:
    def test_performance_equal_to_criterion_does_not_fail(self) -> None:
        """Equality passes: the rule is strict ``<``, not ``<=``."""
        result = grid_break_point([0.0, 5.0], [1.0, 0.95], criterion=0.95)

        assert result.value is None
        assert result.censoring is Censoring.RIGHT

    def test_value_and_max_tested_are_builtin_float(self) -> None:
        """Fields must be ``float``, not ``np.float64``, even when inputs are numpy arrays."""
        severity = np.array([0.0, 5.0, 10.0], dtype=np.float64)
        performance = np.array([1.0, 0.95, 0.4], dtype=np.float64)

        result = grid_break_point(severity, performance, criterion=0.95)

        assert type(result.value) is float
        assert type(result.max_tested) is float

    def test_max_tested_is_builtin_float_when_right_censored(self) -> None:
        """``value`` is ``None`` in the right-censored case; ``max_tested`` is still a float."""
        severity = np.array([0.0, 5.0, 10.0])
        performance = np.array([1.0, 1.0, 0.99])

        result = grid_break_point(severity, performance, criterion=0.95)

        assert result.value is None
        assert type(result.max_tested) is float

    def test_accepts_numpy_array_input(self) -> None:
        """Numpy arrays and plain lists must agree, since both are documented input shapes."""
        severity = [0.0, 5.0, 10.0]
        performance = [1.0, 0.95, 0.4]

        from_lists = grid_break_point(severity, performance, criterion=0.95)
        from_arrays = grid_break_point(np.array(severity), np.array(performance), criterion=0.95)

        assert from_lists == from_arrays

    def test_float32_performance_at_criterion_fails_on_precision(self) -> None:
        """Documented divergence from zeta-bench (``docs/PROVENANCE.md``): everything is cast
        up to float64, and the comparison is then exact in float64 -- there is no tolerance.

        ``np.float32(0.95)`` is not exactly ``0.95``; its nearest float32 value widens to
        ``0.949999988079071`` once cast to float64, which is strictly below a float64
        ``criterion=0.95``. zeta-bench's numpy would instead narrow the Python-float criterion
        down to float32 before comparing, where the two are equal and the level passes. This
        test asserts robustmargin's float64 behaviour is deliberate, not a bug to fix.
        """
        severity = [0.0, 5.0]
        performance = np.array([1.0, 0.95], dtype=np.float32)

        result = grid_break_point(severity, performance, criterion=0.95)

        assert result.value == 5.0
        assert result.censoring is Censoring.NONE

    def test_unsigned_negative_zero_severity_is_accepted(self) -> None:
        """``-0.0`` is not ``< 0.0``, so ``signed=False``'s non-negativity check accepts it.

        A failing level at ``-0.0`` reports ``value == 0.0`` with the sign cleared, matching
        ``abs()``'s convention (used on ``severity`` in the source) rather than propagating the
        input's negative sign bit.
        """
        result = grid_break_point([-0.0, 1.0], [0.4, 1.0], criterion=0.95, signed=False)

        assert result.value == 0.0
        assert math.copysign(1.0, result.value) == 1.0


def _load_matrix_series() -> dict[str, tuple[list[float], list[float]]]:
    """Read ``tests/data/zeta_matrix_counts.csv`` into per-series (severity, rate) arrays."""
    series: dict[str, tuple[list[float], list[float]]] = {}
    with _DATA_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["series"]
            severities, rates = series.setdefault(name, ([], []))
            severities.append(float(row["severity"]))
            successes = float(row["successes"])
            trials = float(row["trials"])
            rates.append(successes / trials)
    return series


# (series, signed, expected value, expected max_tested, expected censoring). Values are cross-
# checked against ../zeta-bench/results/cards/{pid,sac,ppo}.json, keys
# variants.<label>.families.<family>.{break_point,max_tested}, copied in as literals per the
# validation strategy (plan section 6.1): tests never read ../zeta-bench at runtime. All five
# series agree with the committed card JSON and with plan section 3.1's table; no disagreement
# to report.
_EXPECTED_MATRIX_RESULTS: tuple[tuple[str, bool, float | None, float, Censoring], ...] = (
    ("pid_sensor_noise", False, 0.05, 0.1, Censoring.NONE),
    ("pid_wind", False, None, 10.0, Censoring.RIGHT),
    ("sac_sensor_noise", False, 0.0, 0.1, Censoring.NONE),
    ("ppo_sensor_noise", False, 0.0, 0.1, Censoring.NONE),
    ("ppo_mass", True, 0.1, 0.2, Censoring.NONE),
)


@pytest.mark.parametrize(
    ("series_name", "signed", "expected_value", "expected_max_tested", "expected_censoring"),
    _EXPECTED_MATRIX_RESULTS,
    ids=[case[0] for case in _EXPECTED_MATRIX_RESULTS],
)
def test_matrix_series_break_point_at_criterion_0_95(
    series_name: str,
    signed: bool,
    expected_value: float | None,
    expected_max_tested: float,
    expected_censoring: Censoring,
) -> None:
    """Real-data regression: every zeta-bench matrix series at the 0.95 gate."""
    matrix = _load_matrix_series()
    severity, rate = matrix[series_name]

    result = grid_break_point(severity, rate, criterion=0.95, signed=signed)

    assert result.value == expected_value
    assert result.max_tested == pytest.approx(expected_max_tested)
    assert result.censoring is expected_censoring


def test_matrix_series_names_match_fixture() -> None:
    """The parametrised cases above must cover every series the fixture actually has."""
    matrix = _load_matrix_series()
    expected_names = {case[0] for case in _EXPECTED_MATRIX_RESULTS}

    assert set(matrix) == expected_names


class TestProvenanceAndSchemaVersion:
    """``GridBreakPoint`` gains ``schema_version`` and ``provenance``, per the 2026-09-13 owner
    decision: both appended after the existing fields, both with defaults, so the three-field
    positional form stays valid.
    """

    def test_field_order_is_value_max_tested_censoring_schema_version_provenance(self) -> None:
        field_names = tuple(f.name for f in dataclasses.fields(GridBreakPoint))

        assert field_names == (
            "value",
            "max_tested",
            "censoring",
            "schema_version",
            "provenance",
        )

    def test_default_schema_version_is_the_current_package_version(self) -> None:
        """``GridBreakPoint``'s default tracks the single, package-wide ``SCHEMA_VERSION``
        (``"4"`` as of ``decisions/0027``), since ``report.py`` versions the whole card format,
        not each type independently. This test name does not hardcode which version that is."""
        result = grid_break_point([0.0, 5.0, 10.0], [1.0, 0.95, 0.4], criterion=0.95)

        assert result.schema_version == "4"

    def test_default_provenance_is_an_empty_mapping(self) -> None:
        import collections.abc

        result = grid_break_point([0.0, 5.0, 10.0], [1.0, 0.95, 0.4], criterion=0.95)

        assert isinstance(result.provenance, collections.abc.Mapping)
        assert len(result.provenance) == 0

    def test_replace_sets_provenance_without_disturbing_other_fields(self) -> None:
        result = grid_break_point([0.0, 5.0, 10.0], [1.0, 0.95, 0.4], criterion=0.95)
        provenance = {"run": "abc", "nested": {"k": [1, 2]}}

        replaced = dataclasses.replace(result, provenance=provenance)

        assert replaced.provenance == provenance
        assert replaced.value == result.value
        assert replaced.max_tested == result.max_tested
        assert replaced.censoring == result.censoring
        assert replaced.schema_version == result.schema_version

    def test_positional_three_field_construction_fills_in_defaults(self) -> None:
        """The pre-existing three-field positional form must keep working (backward compat)."""
        result = GridBreakPoint(0.1, 0.2, Censoring.NONE)

        assert result.value == 0.1
        assert result.max_tested == 0.2
        assert result.censoring is Censoring.NONE
        # decisions/0027: package-wide SCHEMA_VERSION
        assert result.schema_version == "4"
        assert len(result.provenance) == 0
