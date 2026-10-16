"""``decisions/0027``: score-card schema ``"4"``, and a reader that reads ``"4"`` only.

0027 renames the package to ``robustmargin`` and spends one schema bump on it. What it fixes, and
what this file pins:

* ``SCHEMA_VERSION`` is ``"4"``, and so is every result type's ``schema_version`` default,
  including everything built by ``robustmargin.testing.fake_*``.
* ``load_schema()`` returns the v4 document, ``$id == "urn:robustmargin:schema:scorecard-v4"``.
  v4 is v3 with **exactly two changes**: that ``$id``, and ``Scorecard``'s ``marginkit_version``
  property and ``required`` entry renamed to ``library_version``. The structural digest below
  is how "nothing else moves" is pinned without keeping the retired v3 document in the tree.
* ``load_schema("1")`` / ``("2")`` / ``("3")`` raise ``FileNotFoundError``: those files left the
  package. This is a recorded exception to Appendix B item 4 (read the previous schema).
* ``Scorecard.library_version`` replaces ``marginkit_version`` in Python and in JSON.
* ``from_dict`` on any payload whose ``schema_version`` is ``"1"``, ``"2"`` or ``"3"`` raises
  ``ValueError`` and never mis-reads it -- a whole ``Scorecard``, a bare result, or one nested
  inside a ``"4"`` card. The message names the version found (its ``repr``), says it is the
  marginkit card format, and says this reader reads schema ``'4'`` only. It does not cite
  ``docs/`` or ``decisions/`` (``test_warning_strings_avoid_private_doc_citations.py``). Any
  other unrecognised version keeps the existing "unknown schema_version" error, which does
  **not** say ``marginkit``.

The word ``marginkit`` appears in this file only where it is the thing under test: the retired
card format's name, the retired field name, and the retired URN. Dataclass constructors do not
validate ``schema_version``, and this change adds no such validation, so nothing here depends on
what a constructor does with a ``schema_version`` it is handed.

Replaces ``test_schema_v3.py``, ``test_schema_rewrite_fidelity.py`` and
``test_ratio_interval_schema_v2.py``, whose subject was reading and writing schemas ``"1"`` to
``"3"``; what they pinned about the *current* document (``Fit.diagnostics``/``.monotonicity``,
``HALF_OPEN``, JSON round trips, schema validation of fakes) is kept here, against v4.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import pytest

import robustmargin
from robustmargin import (
    Axis,
    Censoring,
    Definition,
    Fit,
    GridBreakPoint,
    IntervalShape,
    Observations,
    Ratio,
    Scorecard,
    Status,
    Threshold,
    diagnose,
    fit_dose_response,
    grid_break_point,
    ratio_interval,
    threshold,
)
from robustmargin.report import SCHEMA_VERSION, from_dict, load_schema, to_dict
from robustmargin.testing import fake_fit, fake_ratio, fake_threshold

_RETIRED_VERSIONS: tuple[str, ...] = ("1", "2", "3")
_LEVEL = 0.95
_AXIS = Axis(name="severity", unit="unit", scale="log")

# ------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------


def _dump(obj: object) -> dict[str, Any]:
    """``to_dict`` with the payload typed as plain JSON-shaped ``Any`` so tests can index it."""
    payload: dict[str, Any] = to_dict(obj)
    return payload


def _schema() -> dict[str, Any]:
    """``load_schema`` typed the same way, for the same reason."""
    schema: dict[str, Any] = load_schema()
    return schema


def _grid_result() -> GridBreakPoint:
    return grid_break_point([0.0, 5.0, 10.0], [1.0, 0.95, 0.4], criterion=0.95)


def _card(*results: Any) -> Scorecard:
    return Scorecard(results=tuple(results), provenance={})


def _schema_versions_in(obj: Any) -> list[Any]:
    """Every value stored under a ``schema_version`` key, at any depth of a ``to_dict`` payload."""
    found: list[Any] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "schema_version":
                found.append(value)
            else:
                found.extend(_schema_versions_in(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_schema_versions_in(item))
    return found


def _with_schema_version_everywhere(obj: Any, version: str) -> Any:
    """A copy of a ``to_dict`` payload with every ``schema_version`` value replaced."""
    if isinstance(obj, dict):
        return {
            key: (
                version if key == "schema_version" else _with_schema_version_everywhere(v, version)
            )
            for key, v in obj.items()
        }
    if isinstance(obj, list):
        return [_with_schema_version_everywhere(item, version) for item in obj]
    return obj


def _json_round_trip(payload: Any) -> Any:
    return json.loads(json.dumps(payload, allow_nan=False))


def _navigate(payload: Any, path: tuple[str | int, ...]) -> Any:
    node = payload
    for step in path:
        node = node[step]
    return node


def _threshold_with_grid() -> Threshold:
    """A ``SEPARATION`` threshold carrying its ``GridBreakPoint`` fallback in ``.grid``."""
    fit = fake_fit(status=Status.SEPARATION)
    return Threshold(
        axis=fit.axis,
        definition=Definition.absolute(0.95),
        direction=fit.direction,
        value=None,
        lo=None,
        hi=None,
        level=_LEVEL,
        interval_method="exact_bound",
        censoring=Censoring.NONE,
        status=Status.SEPARATION,
        dependence="independent",
        fit=fit,
        grid=GridBreakPoint(value=0.05, max_tested=0.1, censoring=Censoring.NONE),
    )


def _assert_old_version_error(excinfo: pytest.ExceptionInfo[ValueError], version: str) -> None:
    message = str(excinfo.value)
    assert repr(version) in message
    assert "marginkit" in message
    assert "reads schema_version '4' only" in message
    assert "unknown field" not in message
    assert "docs/" not in message
    assert "decisions/" not in message


# ------------------------------------------------------------------------------------------
# schema_version is "4" everywhere
# ------------------------------------------------------------------------------------------

# `UNREACHABLE` and `FAILS_AT_BASELINE` describe whether a *threshold* can be solved, so a `Fit`
# is never built with either (mirrors tests/unit/test_testing.py).
_FIT_STATUSES: tuple[Status, ...] = tuple(
    s for s in Status if s not in (Status.UNREACHABLE, Status.FAILS_AT_BASELINE)
)

_DEFAULT_BUILDERS: dict[str, Callable[[], Any]] = {
    "Fit": fake_fit,
    "Threshold": fake_threshold,
    "Ratio": fake_ratio,
    "GridBreakPoint": _grid_result,
    "Scorecard": lambda: _card(fake_fit()),
}


class TestSchemaVersionIsFour:
    def test_schema_version_constant_is_4(self) -> None:
        assert SCHEMA_VERSION == "4"

    @pytest.mark.parametrize("type_name", list(_DEFAULT_BUILDERS))
    def test_a_freshly_built_result_defaults_to_schema_version_4(self, type_name: str) -> None:
        assert _DEFAULT_BUILDERS[type_name]().schema_version == "4"

    def test_directly_constructed_grid_break_point_defaults_to_schema_version_4(self) -> None:
        assert GridBreakPoint(0.1, 0.2, Censoring.NONE).schema_version == "4"

    def test_every_schema_version_in_a_serialised_card_of_fakes_is_4(self) -> None:
        card = _card(_grid_result(), fake_fit(), fake_threshold(), fake_ratio())

        versions = _schema_versions_in(_dump(card))

        # The card itself, four results, and the fit / thresholds / fits nested inside them:
        # a recursion that missed the nested ones would see fewer than ten.
        assert len(versions) >= 10
        assert set(versions) == {"4"}

    @pytest.mark.parametrize("status", _FIT_STATUSES, ids=lambda s: s.name)
    def test_a_fake_fit_of_every_status_is_tagged_4_throughout(self, status: Status) -> None:
        fit = fake_fit(status=status)

        assert fit.schema_version == "4"
        assert set(_schema_versions_in(_dump(fit))) == {"4"}

    @pytest.mark.parametrize("status", list(Status), ids=lambda s: s.name)
    def test_a_fake_threshold_of_every_status_is_tagged_4_throughout(self, status: Status) -> None:
        t = fake_threshold(status=status, censoring=Censoring.NONE)

        assert t.schema_version == "4"
        assert set(_schema_versions_in(_dump(t))) == {"4"}


# ------------------------------------------------------------------------------------------
# load_schema
# ------------------------------------------------------------------------------------------


class TestLoadSchema:
    def test_default_schema_id_is_the_robustmargin_v4_urn(self) -> None:
        assert _schema()["$id"] == "urn:robustmargin:schema:scorecard-v4"

    def test_load_schema_4_is_the_default(self) -> None:
        assert load_schema("4") == _schema()

    @pytest.mark.parametrize("version", [*_RETIRED_VERSIONS, "5"])
    def test_load_schema_for_a_version_that_is_not_packaged_raises_file_not_found(
        self, version: str
    ) -> None:
        with pytest.raises(FileNotFoundError):
            load_schema(version)

    def test_only_the_v4_document_is_packaged(self) -> None:
        schema_dir = Path(robustmargin.__file__).resolve().parent / "schema"

        names = sorted(p.name for p in schema_dir.glob("scorecard-v*.json"))

        assert names == ["scorecard-v4.json"]

    def test_the_root_still_describes_a_scorecard(self) -> None:
        assert _schema()["properties"]["type"]["const"] == "Scorecard"


# ------------------------------------------------------------------------------------------
# v4 is v3 with exactly two changes
# ------------------------------------------------------------------------------------------


def _structural_digest(schema: dict[str, Any]) -> str:
    """SHA-256 of the schema's *validation content*: the root ``$id`` (pinned on its own above),
    every ``title`` / ``description`` annotation (prose, which a rename legitimately rewrites)
    and the order of every ``required`` list (a set, semantically) are left out; everything
    else -- properties, types, enums, ``$ref``s, ``additionalProperties`` -- is in."""

    def strip(node: Any, *, root: bool) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, value in node.items():
                if root and key == "$id":
                    continue
                if key in ("title", "description") and isinstance(value, str):
                    continue
                if key == "required" and isinstance(value, list):
                    out[key] = sorted(value)
                    continue
                out[key] = strip(value, root=False)
            return out
        if isinstance(node, list):
            return [strip(item, root=False) for item in node]
        return node

    canonical = json.dumps(strip(schema, root=True), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Computed once from the retired v3 document -- `git show ef0d719:src/marginkit/schema/
# scorecard-v3.json`, its last committed form -- with the two changes of decisions/0027 applied
# (`marginkit_version` -> `library_version` in the Scorecard `properties` and `required`), then
# passed through `_structural_digest`. If this fails, the v4 document differs from "v3 plus those
# two changes" in some field, type, enum member or invariant, which 0027 says does not happen:
# diff scorecard-v4.json against that commit's v3 to find it.
_V4_STRUCTURAL_DIGEST = "6f7a70d8e037caefe6ae88b615443b180eab240971e7f8c8250568b232f86829"


class TestSchemaV4IsV3WithTwoChanges:
    def test_v4_structure_is_v3_structure_with_only_library_version_renamed(self) -> None:
        assert _structural_digest(_schema()) == _V4_STRUCTURAL_DIGEST, (
            "scorecard-v4.json differs from the retired v3 document in more than its $id and "
            "the Scorecard marginkit_version -> library_version rename (decisions/0027 item 3)"
        )

    def test_digest_helper_is_sensitive_to_a_validation_change(self) -> None:
        """A guard on the digest itself: it must move when a validation keyword moves, and must
        not when only prose or ``required`` ordering does."""
        schema = _schema()
        baseline = _structural_digest(schema)

        widened = copy.deepcopy(schema)
        widened["$defs"]["Status"]["enum"].append("NEW_MEMBER")
        reworded = copy.deepcopy(schema)
        reworded["description"] = "different prose"
        reworded["title"] = "different title"
        reordered = copy.deepcopy(schema)
        reordered["required"] = list(reversed(reordered["required"]))

        assert _structural_digest(widened) != baseline
        assert _structural_digest(reworded) == baseline
        assert _structural_digest(reordered) == baseline

    def test_scorecard_root_requires_library_version_and_not_marginkit_version(self) -> None:
        schema = _schema()

        assert set(schema["required"]) == {
            "type",
            "results",
            "created_at",
            "library_version",
            "schema_version",
        }
        assert schema["properties"]["library_version"] == {"type": "string"}
        assert "marginkit_version" not in schema["properties"]

    @pytest.mark.parametrize(
        ("def_name", "enum_type"),
        [("Censoring", Censoring), ("Status", Status), ("IntervalShape", IntervalShape)],
    )
    def test_schema_enums_match_the_python_enums_member_for_member(
        self, def_name: str, enum_type: type[enum.Enum]
    ) -> None:
        schema = _schema()

        assert set(schema["$defs"][def_name]["enum"]) == {m.value for m in enum_type}

    def test_half_open_is_in_the_intervalshape_enum(self) -> None:
        assert "HALF_OPEN" in _schema()["$defs"]["IntervalShape"]["enum"]


# ------------------------------------------------------------------------------------------
# Scorecard.library_version
# ------------------------------------------------------------------------------------------


class TestScorecardLibraryVersion:
    def test_library_version_is_a_field_and_marginkit_version_is_not(self) -> None:
        names = {f.name for f in dataclasses.fields(Scorecard)}

        assert "library_version" in names
        assert "marginkit_version" not in names

    def test_library_version_defaults_to_the_package_version(self) -> None:
        card = _card(fake_fit())

        assert card.library_version == robustmargin.__version__

    def test_there_is_no_marginkit_version_attribute(self) -> None:
        assert not hasattr(_card(fake_fit()), "marginkit_version")

    def test_the_constructor_rejects_a_marginkit_version_keyword(self) -> None:
        retired_keyword: dict[str, Any] = {"marginkit_version": "0.1.1"}

        with pytest.raises(TypeError):
            Scorecard(results=(fake_fit(),), provenance={}, **retired_keyword)

    def test_to_dict_writes_library_version_and_no_marginkit_version(self) -> None:
        payload = _dump(Scorecard(results=(fake_fit(),), provenance={}, library_version="9.9.9"))

        assert payload["library_version"] == "9.9.9"
        assert "marginkit_version" not in payload

    def test_library_version_survives_a_json_round_trip(self) -> None:
        card = Scorecard(results=(fake_fit(),), provenance={}, library_version="9.9.9")

        reloaded = from_dict(_json_round_trip(_dump(card)))

        assert isinstance(reloaded, Scorecard)
        assert reloaded.library_version == "9.9.9"
        assert reloaded == card

    def test_a_v4_card_carrying_a_marginkit_version_key_is_rejected_as_unknown(self) -> None:
        payload = _dump(_card(fake_fit()))
        payload["marginkit_version"] = "0.1.1"

        with pytest.raises(ValueError) as excinfo:
            from_dict(payload)

        assert "marginkit_version" in str(excinfo.value)

    def test_a_v4_card_with_library_version_renamed_back_is_rejected_as_unknown(self) -> None:
        payload = _dump(_card(fake_fit()))
        payload["marginkit_version"] = payload.pop("library_version")

        with pytest.raises(ValueError) as excinfo:
            from_dict(payload)

        assert "marginkit_version" in str(excinfo.value)


# ------------------------------------------------------------------------------------------
# Reading a retired schema version raises, and never mis-reads
# ------------------------------------------------------------------------------------------


def _old_shaped_card_payload(version: str) -> dict[str, Any]:
    """A payload shaped like a card the retired format actually wrote: every ``schema_version``
    set to ``version``, the ``library_version`` key under its old name ``marginkit_version``,
    and -- for ``"1"``/``"2"`` -- the ``Fit`` keys that only ``"3"`` introduced removed outright."""
    payload: dict[str, Any] = _with_schema_version_everywhere(
        _dump(_card(fake_threshold())), version
    )
    payload["marginkit_version"] = payload.pop("library_version")
    if version in ("1", "2"):
        fit = payload["results"][0]["fit"]
        del fit["diagnostics"]
        del fit["monotonicity"]
    return payload


# (builder, path from the result down to the object whose schema_version is set). `()` is the
# result itself; the others are objects nested inside it.
_NESTING_CASES: list[tuple[str, Callable[[], Any], tuple[str, ...]]] = [
    ("fit", fake_fit, ()),
    ("threshold", fake_threshold, ()),
    ("threshold.fit", fake_threshold, ("fit",)),
    ("threshold.grid", _threshold_with_grid, ("grid",)),
    ("ratio", fake_ratio, ()),
    ("ratio.threshold_a", fake_ratio, ("threshold_a",)),
    ("ratio.threshold_a.fit", fake_ratio, ("threshold_a", "fit")),
    ("ratio.threshold_b", fake_ratio, ("threshold_b",)),
    ("ratio.threshold_b.fit", fake_ratio, ("threshold_b", "fit")),
    ("grid", _grid_result, ()),
]
_NESTING_IDS = [case[0] for case in _NESTING_CASES]


class TestOldSchemaVersionsAreRejected:
    @pytest.mark.parametrize("version", _RETIRED_VERSIONS)
    def test_a_whole_old_card_raises_naming_the_version(self, version: str) -> None:
        payload = _with_schema_version_everywhere(_dump(_card(fake_threshold())), version)

        with pytest.raises(ValueError) as excinfo:
            from_dict(payload)

        _assert_old_version_error(excinfo, version)

    @pytest.mark.parametrize("version", _RETIRED_VERSIONS)
    def test_a_genuinely_old_shaped_card_is_caught_at_its_version_not_its_old_field_name(
        self, version: str
    ) -> None:
        """The old card's ``marginkit_version`` key would be an unknown field to the v4 reader;
        the version check runs first, so the caller is told what the card *is*."""
        payload = _old_shaped_card_payload(version)
        assert "marginkit_version" in payload

        with pytest.raises(ValueError) as excinfo:
            from_dict(_json_round_trip(payload))

        _assert_old_version_error(excinfo, version)

    @pytest.mark.parametrize("version", _RETIRED_VERSIONS)
    def test_an_old_card_root_with_current_children_still_raises(self, version: str) -> None:
        payload = _dump(_card(fake_threshold()))
        payload["schema_version"] = version

        with pytest.raises(ValueError) as excinfo:
            from_dict(payload)

        _assert_old_version_error(excinfo, version)

    @pytest.mark.parametrize("version", _RETIRED_VERSIONS)
    @pytest.mark.parametrize(("name", "build", "path"), _NESTING_CASES, ids=_NESTING_IDS)
    def test_an_old_object_nested_inside_a_current_card_raises(
        self, name: str, build: Callable[[], Any], path: tuple[str, ...], version: str
    ) -> None:
        payload = _dump(_card(build()))
        assert set(_schema_versions_in(payload)) == {"4"}
        _navigate(payload, ("results", 0, *path))["schema_version"] = version

        with pytest.raises(ValueError) as excinfo:
            from_dict(_json_round_trip(payload))

        _assert_old_version_error(excinfo, version)

    @pytest.mark.parametrize("version", _RETIRED_VERSIONS)
    @pytest.mark.parametrize(("name", "build", "path"), _NESTING_CASES, ids=_NESTING_IDS)
    def test_an_old_bare_result_or_part_of_one_raises(
        self, name: str, build: Callable[[], Any], path: tuple[str, ...], version: str
    ) -> None:
        payload = _dump(build())
        assert set(_schema_versions_in(payload)) == {"4"}
        _navigate(payload, path)["schema_version"] = version

        with pytest.raises(ValueError) as excinfo:
            from_dict(_json_round_trip(payload))

        _assert_old_version_error(excinfo, version)


class TestUnrecognisedSchemaVersionKeepsTheExistingError:
    """A version outside 1-4 is not "an old marginkit card": it keeps the plain
    "unknown schema_version" error, and that error does not name marginkit."""

    @pytest.mark.parametrize("version", ["5", "999", "0", "10", ""])
    def test_a_bare_result_with_an_unrecognised_version_says_unknown_schema_version(
        self, version: str
    ) -> None:
        payload = _dump(fake_fit())
        payload["schema_version"] = version

        with pytest.raises(ValueError) as excinfo:
            from_dict(payload)

        message = str(excinfo.value)
        assert "unknown schema_version" in message
        assert repr(version) in message
        assert "marginkit" not in message

    @pytest.mark.parametrize("version", ["5", "999"])
    def test_a_card_with_an_unrecognised_root_version_says_unknown_schema_version(
        self, version: str
    ) -> None:
        payload = _dump(_card(fake_fit()))
        payload["schema_version"] = version

        with pytest.raises(ValueError) as excinfo:
            from_dict(payload)

        message = str(excinfo.value)
        assert "unknown schema_version" in message
        assert "marginkit" not in message

    @pytest.mark.parametrize("version", ["5", "999"])
    def test_an_unrecognised_version_nested_in_a_current_card_says_unknown_schema_version(
        self, version: str
    ) -> None:
        payload = _dump(_card(fake_threshold()))
        payload["results"][0]["fit"]["schema_version"] = version

        with pytest.raises(ValueError) as excinfo:
            from_dict(payload)

        message = str(excinfo.value)
        assert "unknown schema_version" in message
        assert repr(version) in message
        assert "marginkit" not in message


# ------------------------------------------------------------------------------------------
# What v4 does carry: round trips and validation, kept from the v2/v3 files
# ------------------------------------------------------------------------------------------


def _validator_for(type_name: str) -> jsonschema.protocols.Validator:
    schema = _schema()
    return jsonschema.Draft202012Validator(
        {"$defs": schema["$defs"], "$ref": f"#/$defs/{type_name}"}
    )


_FAKE_BUILDERS: dict[str, Callable[[], Any]] = {
    "fake_fit": fake_fit,
    "fake_threshold": fake_threshold,
    "fake_ratio": fake_ratio,
    "fake_ratio_half_open": lambda: fake_ratio(shape=IntervalShape.HALF_OPEN, status=Status.OK),
    "fake_threshold_with_grid": _threshold_with_grid,
    "grid_break_point": _grid_result,
}


class TestV4RoundTripsAndValidates:
    @pytest.mark.parametrize("name", list(_FAKE_BUILDERS))
    def test_every_fake_round_trips_through_json_text(self, name: str) -> None:
        result = _FAKE_BUILDERS[name]()

        reloaded = from_dict(json.loads(json.dumps(_dump(result), allow_nan=False)))

        assert reloaded == result

    @pytest.mark.parametrize("name", list(_FAKE_BUILDERS))
    def test_a_card_of_every_fake_validates_against_the_full_packaged_schema(
        self, name: str
    ) -> None:
        card = _card(_FAKE_BUILDERS[name]())

        jsonschema.Draft202012Validator(_schema()).validate(_dump(card))

    def test_one_card_holding_every_fake_round_trips_and_validates(self) -> None:
        card = _card(*(build() for build in _FAKE_BUILDERS.values()))
        payload = _dump(card)

        jsonschema.Draft202012Validator(_schema()).validate(payload)
        reloaded = from_dict(_json_round_trip(payload))

        assert reloaded == card

    @pytest.mark.parametrize("name", list(_FAKE_BUILDERS))
    def test_reading_and_rewriting_a_v4_payload_is_byte_identical(self, name: str) -> None:
        payload = _json_round_trip(_dump(_card(_FAKE_BUILDERS[name]())))

        rewritten = _dump(from_dict(payload))

        assert json.dumps(rewritten, sort_keys=True) == json.dumps(payload, sort_keys=True)

    def test_half_open_fake_has_hi_none_and_lo_at_or_below_estimate(self) -> None:
        ratio = fake_ratio(shape=IntervalShape.HALF_OPEN, status=Status.OK)

        assert ratio.shape is IntervalShape.HALF_OPEN
        assert ratio.lo is not None
        assert ratio.hi is None
        assert ratio.estimate is not None
        assert ratio.lo <= ratio.estimate

    def test_half_open_ratio_validates_against_the_ratio_definition(self) -> None:
        ratio = fake_ratio(shape=IntervalShape.HALF_OPEN, status=Status.OK)

        errors = list(_validator_for("Ratio").iter_errors(_dump(ratio)))

        assert errors == []

    def test_a_fit_with_diagnostics_and_monotonicity_validates_and_round_trips(self) -> None:
        fit = fake_fit(status=Status.OK)
        assert fit.diagnostics is not None
        assert fit.monotonicity is not None

        _validator_for("Fit").validate(_dump(fit))
        assert from_dict(_json_round_trip(_dump(fit))) == fit

    def test_diagnostics_and_monotonicity_are_optional_on_fit(self) -> None:
        fit_def = _schema()["$defs"]["Fit"]

        assert "diagnostics" in fit_def["properties"]
        assert "monotonicity" in fit_def["properties"]
        assert "diagnostics" not in fit_def["required"]
        assert "monotonicity" not in fit_def["required"]

    def test_diagnostics_has_no_monotonicity_entry(self) -> None:
        assert "monotonicity" not in _schema()["$defs"]["Diagnostics"]["properties"]

    def test_monotonicity_check_and_adjacent_pair_reject_an_extra_key(self) -> None:
        defs = _schema()["$defs"]

        for type_name in ("MonotonicityCheck", "AdjacentPair", "Diagnostics", "BaselineRate"):
            assert defs[type_name].get("additionalProperties") is False, type_name


# ------------------------------------------------------------------------------------------
# One card built from real fits, not fakes
# ------------------------------------------------------------------------------------------


def _real_ok_fit() -> Fit:
    obs = Observations.from_counts(
        _AXIS,
        severity=[0.5, 1.0, 2.0, 4.0],
        successes=[190, 150, 60, 5],
        trials=[200, 200, 200, 200],
        outcome="success",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
    assert fit.status is Status.OK
    assert fit.monotonicity is not None
    return dataclasses.replace(fit, diagnostics=diagnose(fit))


def _real_threshold() -> Threshold:
    t = threshold(
        _real_ok_fit(),
        definition=Definition.absolute(0.5),
        interval_method="profile",
        level=_LEVEL,
    )
    assert t.status is Status.OK
    return t


def _real_half_open_ratio() -> Ratio:
    """A tightly-determined numerator over a barely-constrained denominator: the Fieller region
    is ``[lo, inf)``. The same construction ``test_ratio_interval_shape.py`` reaches
    ``HALF_OPEN`` with; reproduced here so this file does not rest on a fake for the shape."""
    numerator = _real_threshold()
    obs = Observations.from_counts(
        _AXIS,
        severity=[1.0, 2.0],
        successes=[5, 4],
        trials=[20, 20],
        outcome="success",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
    assert fit.status is Status.OK
    denominator = threshold(
        fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
    )
    ratio = ratio_interval(
        numerator, denominator, method="fieller", dependence="independent", level=_LEVEL
    )
    assert ratio.shape is IntervalShape.HALF_OPEN
    return ratio


def _real_fallback_threshold_with_grid() -> Threshold:
    """Success rises with severity under ``direction="decreasing"``: no finite slope exists, so
    the fit is ``NOT_CONVERGED`` and the threshold falls back to the exact bounds plus a grid
    break point."""
    obs = Observations.from_counts(
        _AXIS,
        severity=[1.0, 2.0, 4.0, 8.0],
        successes=[20, 35, 50, 65],
        trials=[100, 100, 100, 100],
        outcome="success",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
    assert fit.status is Status.NOT_CONVERGED
    t = threshold(fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL)
    assert t.grid is not None
    return t


def _real_fails_at_baseline_threshold() -> Threshold:
    """A control cell (severity 0) whose exact upper bound, 365/400, is below the 0.95 target."""
    obs = Observations.from_counts(
        _AXIS,
        severity=[0.0, 1.0, 2.0, 4.0, 8.0],
        successes=[365, 330, 250, 120, 30],
        trials=[400, 400, 400, 400, 400],
        outcome="success",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
    t = threshold(
        fit, definition=Definition.absolute(0.95), interval_method="profile", level=_LEVEL
    )
    assert t.status is Status.FAILS_AT_BASELINE
    assert t.baseline is not None
    return t


class TestARealCardIsSchemaFourThroughout:
    def _real_card(self) -> Scorecard:
        return Scorecard(
            results=(
                _real_ok_fit(),
                _real_threshold(),
                _real_half_open_ratio(),
                _real_fallback_threshold_with_grid(),
                _real_fails_at_baseline_threshold(),
                _grid_result(),
            ),
            provenance={"run": "schema-v4", "nested": {"k": [1, 2, None, True, 3.5]}},
        )

    def test_every_schema_version_in_the_serialised_card_is_4(self) -> None:
        versions = _schema_versions_in(_dump(self._real_card()))

        assert len(versions) >= 12
        assert set(versions) == {"4"}

    def test_the_card_validates_against_the_packaged_schema(self) -> None:
        jsonschema.Draft202012Validator(_schema()).validate(_dump(self._real_card()))

    def test_the_card_round_trips_through_json_text_unchanged(self) -> None:
        card = self._real_card()

        reloaded = from_dict(json.loads(json.dumps(_dump(card), allow_nan=False)))

        assert reloaded == card

    def test_the_card_carries_diagnostics_monotonicity_baseline_grid_and_half_open(self) -> None:
        card = self._real_card()
        fit, threshold_ok, ratio, fallback, baseline_threshold, _ = card.results

        assert isinstance(fit, Fit) and fit.diagnostics is not None
        assert fit.monotonicity is not None
        assert isinstance(ratio, Ratio) and ratio.shape is IntervalShape.HALF_OPEN
        assert isinstance(fallback, Threshold) and fallback.grid is not None
        assert isinstance(baseline_threshold, Threshold) and baseline_threshold.baseline is not None
        assert isinstance(threshold_ok, Threshold)

    def test_serialising_the_card_twice_gives_identical_json(self) -> None:
        card = self._real_card()

        first = json.dumps(_dump(card), sort_keys=True)
        second = json.dumps(_dump(from_dict(json.loads(first))), sort_keys=True)

        assert first == second
