"""Reference-value parity for the zeta PPO x sensor_noise monotonicity reversal
(``docs/decisions/0023-fit-diagnostics-and-schema-v3.md``'s "Amendment 1"; plan section 6.1).

Two independent cross-checks, so this file has "teeth" even before
``tools/drc_reference/zeta_ppo_sensor_noise_monotonicity.json`` is regenerated (no R here):

1. **``scipy.stats.fisher_exact`` directly** (``TestAgainstScipyDirectly``) -- runs today, no
   fixture needed, and pins the exact table/alternative convention
   (``docs/decisions/0023``'s Amendment 1, item 4) robustmargin's own ``check_monotonicity`` must
   reproduce.
2. **R's own ``fisher.test(..., alternative="greater")``**
   (``TestAgainstRsFisherTest``), read from the fixture ``generate.R`` now emits. This fails with
   a clear ``FileNotFoundError`` until the owner re-runs ``generate.R`` in the pinned container --
   that failure means "regenerate the fixture", not a statistical finding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scipy.stats import fisher_exact

from robustmargin import Axis, Observations, check_monotonicity

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tools" / "drc_reference"
_REL_TOL = 1e-4  # plan section 6.1

# tests/data/zeta_matrix_counts.csv, series ppo_sensor_noise, the two lowest severities.
_SEVERITY_LOW = 0.0
_SEVERITY_HIGH = 0.01
_SUCCESSES_LOW, _TRIALS_LOW = 57, 200
_SUCCESSES_HIGH, _TRIALS_HIGH = 143, 300


def _load(name: str) -> dict[str, Any]:
    with (_FIXTURE_DIR / name).open() as fh:
        result: dict[str, Any] = json.load(fh)
    return result


def _ppo_check() -> Any:
    axis = Axis(name="sensor_noise", unit="m", scale="log")
    obs = Observations.from_counts(
        axis,
        severity=[_SEVERITY_LOW, _SEVERITY_HIGH, 0.05, 0.1],
        successes=[_SUCCESSES_LOW, _SUCCESSES_HIGH, 0, 0],
        trials=[_TRIALS_LOW, _TRIALS_HIGH, 300, 300],
        outcome="success",
        direction="decreasing",
    )
    return check_monotonicity(obs)


def _first_pair_p_value() -> float:
    check = _ppo_check()
    first = check.pairs[0]
    assert (first.severity_low, first.severity_high) == (_SEVERITY_LOW, _SEVERITY_HIGH)
    return float(first.p_value)


class TestAgainstScipyDirectly:
    def test_raw_p_value_matches_scipy_fisher_exact(self) -> None:
        failures_low = _TRIALS_LOW - _SUCCESSES_LOW
        failures_high = _TRIALS_HIGH - _SUCCESSES_HIGH
        # decisions/0023 Amendment 1, item 4's exact convention.
        table = [[_SUCCESSES_HIGH, failures_high], [_SUCCESSES_LOW, failures_low]]
        _, expected_p = fisher_exact(table, alternative="greater")

        actual_p = _first_pair_p_value()

        assert actual_p == pytest.approx(float(expected_p), rel=1e-9)

    def test_raw_p_value_is_roughly_the_decision_records_figure(self) -> None:
        """decisions/0023's own worked example: "raw p ~ 1.2e-5"."""
        actual_p = _first_pair_p_value()

        assert actual_p == pytest.approx(1.2e-5, rel=0.1)


class TestAgainstRsFisherTest:
    """Reads ``tools/drc_reference/zeta_ppo_sensor_noise_monotonicity.json``, which does not
    exist in this environment (no R) -- ``generate.R`` has been extended to emit it
    (this phase), but regenerating it requires the pinned R container. Until then this class
    fails with ``FileNotFoundError``, not a wrong number."""

    def test_r_native_p_value_matches_check_monotonicity(self) -> None:
        fixture = _load("zeta_ppo_sensor_noise_monotonicity.json")  # FileNotFoundError until regen
        expected_p = fixture["fisher_test"]["p_value"]

        actual_p = _first_pair_p_value()

        assert actual_p == pytest.approx(expected_p, rel=_REL_TOL)

    def test_r_fixture_dataset_matches_the_csv_counts(self) -> None:
        fixture = _load("zeta_ppo_sensor_noise_monotonicity.json")  # FileNotFoundError until regen
        dataset = fixture["dataset"]

        assert dataset["successes_low"] == _SUCCESSES_LOW
        assert dataset["trials_low"] == _TRIALS_LOW
        assert dataset["successes_high"] == _SUCCESSES_HIGH
        assert dataset["trials_high"] == _TRIALS_HIGH
