"""Everything that ships must make sense to a reader who has only the package.

``robustmargin`` is published to PyPI, and a published release is permanent. The project's own
design documents (the decision records, the implementation plan, the roadmap, the review log, the
consumer contract, the architecture and statistics notes, the validation notes, and the
agent-instruction file) are gitignored and never published. A docstring, comment, schema
description or README sentence that cites one of them -- ``decisions/0023``, "plan section 5.4",
a section sign, "hard constraint 3", "owner decision", ``D5``, "Phase 7", "(R2)", a review-finding
label, an agent's name -- sends the reader to a document they cannot open.

`test_warning_strings_avoid_private_doc_citations.py` already covers *runtime* warning and
exception strings, and deliberately exempts docstrings and comments. This file closes that gap by
scanning the **raw text** of every file that ships, so docstrings and comments are included:

* every ``src/robustmargin/**/*.py``;
* ``src/robustmargin/schema/*.json`` (the packaged score-card schema, whose ``description`` fields
  are read by consumers);
* ``README.md`` (the PyPI project page);
* ``pyproject.toml`` (it ships in the sdist).

Two groups of patterns, reported separately so a failure says which kind of leak it is:

``_CITATION_PATTERNS``
    The agreed list of private-document references.
``_DANGLING_PATTERNS``
    References that only mean something *inside* a private document -- "Amendment 1", "item 6",
    a bare "section 5.6", "stats-review fix", "design choice 2". They are what is left behind when
    the document's name is removed but its internal numbering is not, so they would still send a
    reader looking for something that does not exist.

The scan is per file and each failure lists *every* offending ``file:line`` with the matched text,
so one run is a complete worklist.

A scanner that matches nothing would pass vacuously, so the matcher itself is pinned: known-bad
strings are caught, known-good strings (R-language references to ``drc`` and ``glm``, "planned",
"planning", numeric literals such as ``0.0025``, literature citations, the one public PROVENANCE
URL) are not, and the file list is checked to be non-trivial.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src" / "robustmargin"

# The one `docs/` path that is allowed to appear: the committed, public attribution file, as an
# absolute URL. A relative `docs/PROVENANCE.md` (or any other `docs/` path) is not allowed: it
# means nothing to a reader who installed from PyPI.
_PROVENANCE_URL = "https://github.com/robustmargin/robustmargin/blob/main/docs/PROVENANCE.md"
_PROVENANCE_PLACEHOLDER = "<public-provenance-url>"

# (label, pattern). The label is what a failure prints next to the matched text.
_Pattern = tuple[str, re.Pattern[str]]

_CITATION_PATTERNS: tuple[_Pattern, ...] = (
    ("decision-record path", re.compile(r"decisions/")),
    # A bare decision-record number: four digits with a leading zero, used as a citation
    # (`0023`, `0023 Amendment 2`, "0008, 0009"). It must not match a digit run that is part of a
    # number: `0.0025` (preceded by "."), `1e-0025` (preceded by an exponent sign), `10025` or
    # `1_0025` (preceded by a word character), `0.00250` or `0023.5` (followed by a word
    # character or by "." and a digit).
    (
        "decision-record number",
        re.compile(r"(?<![\w.])(?<![eE][-+])0\d{3}(?!\w|\.\d)"),
    ),
    ("section sign", re.compile("§")),
    # The word "plan" and its plural; `\b` keeps "planned", "planning", "planar", "explanation".
    ("plan", re.compile(r"\bplans?\b", re.IGNORECASE)),
    (
        "private document name",
        re.compile(
            r"IMPLEMENTATION_PLAN|\bROADMAP\b|\bPITCH\b"
            r"|\b(?:REVIEWS|CONSUMERS|ARCHITECTURE|STATISTICS|CLAUDE)\b(?:\.md)?"
            r"|validation\.md"
        ),
    ),
    ("hard constraint", re.compile(r"\bhard[- ]constraints?\b", re.IGNORECASE)),
    ("owner", re.compile(r"\bowners?\b", re.IGNORECASE)),
    ("decision label D1-D7", re.compile(r"\bD[1-7]\b")),
    ("phase number", re.compile(r"\bphases? \d", re.IGNORECASE)),
    ("Appendix B", re.compile(r"\bAppendix B\b")),
    # R1-R10 requirement labels and "R6 #14" review references. "R `drc`", "R's glm" and
    # "R-squared" carry no digit directly after the R, so they do not match.
    ("requirement label R1-R10", re.compile(r"\bR(?:10|[1-9])\b")),
    # Review-finding labels ("finding C2", "W1", "(S3)"). Checked against every letter-digit token
    # in the shipped text first: the only ones present are review labels, never mathematics
    # (no H0/H1, L1/L2 or matrix-entry names). Restricted to the letters actually used for
    # findings so that a future "H0" or "L2" in a statistical statement does not trip it. The
    # trailing look-ahead keeps chemical-style tokens ("C2H5") and decimals ("S3.5") out.
    ("review-finding label", re.compile(r"(?<![\w.])[ACSW]\d{1,2}(?![\w.])")),
    ("agent name", re.compile(r"\b(?:test-author|api-compat|stats-reviewer|implementer)\b")),
    # Any `docs/` path. The PROVENANCE URL is removed from the line before matching (see
    # `_scan_text`), so it is the only form that survives. `readthedocs/`-style host or path
    # segments ending in "docs" are not `docs/` directories.
    ("docs/ path", re.compile(r"(?<![\w-])docs/")),
)

_DANGLING_PATTERNS: tuple[_Pattern, ...] = (
    ("Amendment", re.compile(r"\bamendments?\b", re.IGNORECASE)),
    ("bare section number", re.compile(r"\bsections? \d", re.IGNORECASE)),
    ("bare item number", re.compile(r"\bitems? \d", re.IGNORECASE)),
    ("stats review", re.compile(r"\bstats[- ]review", re.IGNORECASE)),
    ("design choice number", re.compile(r"\bdesign choices? \d", re.IGNORECASE)),
)


class _Hit(NamedTuple):
    lineno: int
    labels: tuple[str, ...]
    matched: tuple[str, ...]
    line: str


def _scan_text(text: str, patterns: tuple[_Pattern, ...]) -> list[_Hit]:
    """Every line of ``text`` matching any of ``patterns``, with 1-based line numbers.

    The public PROVENANCE URL is blanked out first, so the ``docs/`` inside it is not a hit but a
    relative ``docs/PROVENANCE.md`` on the same line still is.
    """
    hits: list[_Hit] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.replace(_PROVENANCE_URL, _PROVENANCE_PLACEHOLDER)
        labels: list[str] = []
        matched: list[str] = []
        for label, pattern in patterns:
            for match in pattern.finditer(line):
                labels.append(label)
                matched.append(match.group(0))
        if labels:
            hits.append(_Hit(lineno, tuple(labels), tuple(matched), raw_line.strip()))
    return hits


def _format_hits(relpath: str, hits: list[_Hit]) -> str:
    rows = []
    for hit in hits:
        found = ", ".join(f"{label}={text!r}" for label, text in zip(hit.labels, hit.matched))
        rows.append(f"  {relpath}:{hit.lineno}: {found}\n      {hit.line[:150]}")
    return "\n".join(rows)


def _shipped_files() -> list[Path]:
    """The files that ship: the package sources and schema, the PyPI page, the sdist metadata."""
    sources = sorted(_SRC.rglob("*.py"))
    schemas = sorted((_SRC / "schema").glob("*.json"))
    return [*sources, *schemas, _ROOT / "README.md", _ROOT / "pyproject.toml"]


def _relative(path: Path) -> str:
    return path.relative_to(_ROOT).as_posix()


_SHIPPED = _shipped_files()


# ------------------------------------------------------------------------------------------
# 1. The scan of the shipped files.
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", _SHIPPED, ids=_relative)
def test_shipped_file_cites_no_unpublished_document(path: Path) -> None:
    hits = _scan_text(path.read_text(encoding="utf-8"), _CITATION_PATTERNS)
    if hits:
        pytest.fail(
            f"{_relative(path)} ships to PyPI but cites material that is not published "
            f"({len(hits)} line(s)); rewrite each so it stands on its own:\n"
            + _format_hits(_relative(path), hits),
            pytrace=False,
        )


@pytest.mark.parametrize("path", _SHIPPED, ids=_relative)
def test_shipped_file_has_no_dangling_reference_into_an_unpublished_document(path: Path) -> None:
    hits = _scan_text(path.read_text(encoding="utf-8"), _DANGLING_PATTERNS)
    if hits:
        pytest.fail(
            f"{_relative(path)} ships to PyPI but numbers things that only exist in an unpublished "
            f"document ({len(hits)} line(s)); state the rule itself instead of its number:\n"
            + _format_hits(_relative(path), hits),
            pytrace=False,
        )


# ------------------------------------------------------------------------------------------
# 2. Guards on the scan itself: it must read the files it claims to, and it must be able to fail.
# ------------------------------------------------------------------------------------------


def test_scan_covers_every_module_the_schema_the_readme_and_the_sdist_metadata() -> None:
    relative = {_relative(path) for path in _SHIPPED}

    assert _SRC.name == "robustmargin"
    # Not vacuous: a package rename that left `_SRC` pointing nowhere would make `rglob` yield
    # nothing, and every parametrised test above would silently disappear.
    assert len(_SHIPPED) >= 14
    for module in (
        "__init__",
        "censoring",
        "empirical",
        "intervals",
        "models",
        "ratio",
        "report",
        "testing",
        "threshold",
        "types",
        "validation",
    ):
        assert f"src/robustmargin/{module}.py" in relative
    assert "src/robustmargin/schema/scorecard-v4.json" in relative
    assert "README.md" in relative
    assert "pyproject.toml" in relative


def test_scan_reads_nonempty_text_from_every_shipped_file() -> None:
    for path in _SHIPPED:
        assert path.is_file(), f"{_relative(path)} is listed as shipped but does not exist"
        assert path.read_text(encoding="utf-8").strip(), f"{_relative(path)} is empty"


def test_scan_reports_the_line_number_and_matched_text_of_every_offence() -> None:
    text = (
        '"""A fine docstring."""\n'
        "x = 1  # see decisions/0023 and plan section 5.4\n"
        "\n"
        "y = 2  # nothing wrong here\n"
        "z = 3  # hard constraint 3\n"
    )

    hits = _scan_text(text, _CITATION_PATTERNS)

    assert [hit.lineno for hit in hits] == [2, 5]
    assert "decisions/" in hits[0].matched
    assert "0023" in hits[0].matched
    assert "plan" in hits[0].matched
    assert hits[1].matched == ("hard constraint",)
    report = _format_hits("pkg/mod.py", hits)
    assert "pkg/mod.py:2:" in report
    assert "pkg/mod.py:5:" in report
    assert "'decisions/'" in report


def test_a_comment_or_docstring_is_scanned_not_only_string_literals() -> None:
    """The runtime-string test skips docstrings and comments; this one must not."""
    text = '"""Module docstring citing decisions/0023."""\n# a comment citing the plan\n'

    assert [hit.lineno for hit in _scan_text(text, _CITATION_PATTERNS)] == [1, 2]


# ------------------------------------------------------------------------------------------
# 3. Pinning the matcher: known-bad strings are caught.
# ------------------------------------------------------------------------------------------


def _labels(text: str) -> set[str]:
    hits = _scan_text(text, (*_CITATION_PATTERNS, *_DANGLING_PATTERNS))
    return {label for hit in hits for label in hit.labels}


_KNOWN_BAD: tuple[tuple[str, str], ...] = (
    # decision records
    ("(`decisions/0009`)", "decision-record path"),
    ("see decisions/0023 for the rule", "decision-record number"),
    ("``docs/decisions/0023-fit-diagnostics``", "decision-record path"),
    ("per `decisions/0008`, `0009`, `0011`", "decision-record number"),
    ("what `0011` forecloses", "decision-record number"),
    ("`0023` Amendment 2", "decision-record number"),
    ("0023 Amendment 2, item 4", "decision-record number"),
    ("decision 0011 says so", "decision-record number"),
    ("(0008, 0009)", "decision-record number"),
    ("0010.", "decision-record number"),
    # section sign
    ("label §5.3-§5.5", "section sign"),
    ("(§3.1)", "section sign"),
    # the plan
    ("(plan section 5.4)", "plan"),
    ("the plan says so", "plan"),
    ("Plan §5.1", "plan"),
    ("loose floors (plan R9)", "plan"),
    ("the Phase 4 plan's answer", "plan"),
    ("the plans", "plan"),
    # private document names
    ("see IMPLEMENTATION_PLAN.md", "private document name"),
    ("per ROADMAP section 3.1", "private document name"),
    ("`REVIEWS.md` R8 #18", "private document name"),
    ("`CONSUMERS.md`", "private document name"),
    ("`ARCHITECTURE.md` A9", "private document name"),
    ("``STATISTICS.md``", "private document name"),
    ("docs/validation.md", "private document name"),
    ("CLAUDE.md hard constraint 6", "private document name"),
    ("PITCH.md", "private document name"),
    ("see ARCHITECTURE for why", "private document name"),
    # hard constraints
    ("(hard constraint 3)", "hard constraint"),
    ("Hard Constraint 2 forbids it", "hard constraint"),
    ("hard-constraint 1", "hard constraint"),
    # owner
    ("pending owner decision", "owner"),
    ("the owner's answer", "owner"),
    ("Owner decision 2026-10-15", "owner"),
    ("which the owner chose", "owner"),
    # decision labels
    ("pending D5", "decision label D1-D7"),
    ("(D7)", "decision label D1-D7"),
    ("D1, D2 and D3", "decision label D1-D7"),
    ("owner decision D5's restatement", "decision label D1-D7"),
    # phases
    ("Phase 4", "phase number"),
    ("(Phase 7)", "phase number"),
    ("phase 5 stats", "phase number"),
    ("in Phases 3 and 4", "phase number"),
    # appendix
    ("a recorded exception to Appendix B item 4", "Appendix B"),
    # requirement labels and review references
    ("(R2)", "requirement label R1-R10"),
    ("identical fits (R1): everything", "requirement label R1-R10"),
    ('R10 "Still open" item 1', "requirement label R1-R10"),
    ("`REVIEWS.md` R6 #14", "requirement label R1-R10"),
    ("R8 #18", "requirement label R1-R10"),
    ("(R3; R5)", "requirement label R1-R10"),
    # review-finding labels
    ("finding C2", "review-finding label"),
    ("(C1), so predict(0)", "review-finding label"),
    ("likelihood-path convergence (W3)", "review-finding label"),
    ("amendment, W1)", "review-finding label"),
    ("# S3, Phase 6 stats review", "review-finding label"),
    ("`ARCHITECTURE.md` A9", "review-finding label"),
    ("W2 / the observed information", "review-finding label"),
    # agent names
    ("which `api-compat` caught", "agent name"),
    ("written by test-author", "agent name"),
    ("a stats-reviewer finding", "agent name"),
    ("ask the implementer", "agent name"),
    # docs/ paths
    ("``docs/PROVENANCE.md``", "docs/ path"),
    ("see docs/STATISTICS.md", "docs/ path"),
    ("../docs/anything", "docs/ path"),
    # the exception is the one exact URL, not "any docs URL" or "any host"
    ("https://github.com/robustmargin/robustmargin/blob/main/docs/ROADMAP.md", "docs/ path"),
    ("https://github.com/other/other/blob/main/docs/PROVENANCE.md", "docs/ path"),
    # dangling references into an unpublished document
    ('(decision\'s "Amendment 1")', "Amendment"),
    ("a later amendment", "Amendment"),
    ("(Amendments 2-4)", "Amendment"),
    ("(section 5.6)", "bare section number"),
    ("Section 7 of the document", "bare section number"),
    ("(round 4 sections 1-2)", "bare section number"),
    ("item 6", "bare item number"),
    ("Item 6: this call", "bare item number"),
    ("stats-review fix: an absolute tolerance", "stats review"),
    ("Stats review added the rule", "stats review"),
    ("the stats review", "stats review"),
    ("(design choice 2)", "design choice number"),
)


@pytest.mark.parametrize(
    ("text", "label"), _KNOWN_BAD, ids=[f"{label}: {t}" for t, label in _KNOWN_BAD]
)
def test_matcher_catches_known_bad_string(text: str, label: str) -> None:
    assert label in _labels(text), f"{text!r} should be caught as {label!r}; got {_labels(text)}"


def test_relative_provenance_path_is_caught_even_when_the_public_url_is_on_the_same_line() -> None:
    line = f"[`docs/PROVENANCE.md`]({_PROVENANCE_URL})"

    hits = _scan_text(line, _CITATION_PATTERNS)

    assert len(hits) == 1
    assert hits[0].matched == ("docs/",)


# ------------------------------------------------------------------------------------------
# 4. Pinning the matcher: known-good strings pass.
# ------------------------------------------------------------------------------------------


_KNOWN_GOOD: tuple[str, ...] = (
    # The R language is a public reference, not a private document.
    "R `drc`",
    "R ``drc``'s ``logLik()``",
    'matches R ``drc``\'s ``type = "relative"``',
    "R's glm",
    "R's own ``binomial()$linkinv`` clamp",
    "a `glm` fit in R",
    "tools/drc_reference/README.md",
    # "planned", "planning" and kin are ordinary words.
    "planned for 0.2",
    "CI-width planning",
    "a planned release; planning a design; planar; explanation",
    "implemented, implementation, an implementable rule",
    # The one public file, as an absolute URL, alone and inside a markdown link.
    _PROVENANCE_URL,
    f"[PROVENANCE.md]({_PROVENANCE_URL})",
    f"Attribution: {_PROVENANCE_URL}.",
    "https://github.com/badkoubeh/zeta-bench",
    # Numeric literals that contain four digits with a leading zero.
    "0.0025",
    "alpha = 0.0025",
    "1e-0025",
    "1.5E-0025",
    "x = -0.0025,",
    "10025",
    "1_0025",
    "0x0025",
    "0.00250",
    "version 0.1.1rc1 on 2026-09-12",
    "tolerance 1e-10, ratio 0.0100",
    # Literature and method names.
    "Clopper-Pearson",
    "Albert & Anderson (1984)",
    "Fieller (1954), Wilson (1927), Dunn & Smyth (1996)",
    "Pearson dispersion, Newton's method, the Wald interval",
    "a decision rule on the p-value",
    "D-optimal design; Cook's D",
    "a phase-type distribution; out of phase; Phase of the moon",
    "this section describes the interface; each item of the list",
    "an amended schema; the amended text",
    "ownership of the file handle",
    "Cartesian product A x B; S_3; H0 versus H1",
    "chemical C2H5OH and C1.5",
    "the pitch of the thread; the roadmap of a trip",
)


@pytest.mark.parametrize("text", _KNOWN_GOOD)
def test_matcher_passes_known_good_string(text: str) -> None:
    hits = _scan_text(text, (*_CITATION_PATTERNS, *_DANGLING_PATTERNS))

    assert not hits, f"{text!r} is legitimate public text but was flagged: {hits}"


def test_every_pattern_has_a_known_bad_example_so_none_can_rot_into_matching_nothing() -> None:
    covered = {label for _text, label in _KNOWN_BAD}
    declared = {label for label, _ in (*_CITATION_PATTERNS, *_DANGLING_PATTERNS)}

    assert declared <= covered, f"patterns with no known-bad example: {sorted(declared - covered)}"
