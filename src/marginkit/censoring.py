"""Censoring classification (plan section 5.5) and exact one-sided bounds (`decisions/0009`).

Public: :func:`exact_one_sided_bound`, the one piece of numeric machinery every rule in this
module (and :func:`marginkit.empirical.per_cell_clopper_pearson`) is built from -- the exact
arithmetic lives in exactly one place, here. Internal: :func:`classify`, which turns a
:class:`~marginkit.Fit` plus an already-resolved target performance into a
:class:`Classification` describing which of plan section 5.5's outcomes applies.

Classification order (plan section 5.5's table is a list of conditions, not a sequence; this is
the sequence, per `decisions/0008`, `0009`, `0011`, `0012`):

1. ``fit.status is CONTROL_INCOMPATIBLE`` -- terminal mirror.
2. ``FAILS_AT_BASELINE`` -- the zero-severity control's exact one-sided bound at ``level``
   already fails the target, and no other tested level passes it (`decisions/0008`, `0011`: a
   control failure contradicted by a later passing level is a monotonicity contradiction, not a
   clean baseline failure, and belongs to step 7 instead). Short-circuits before ``UNREACHABLE``.
3. ``UNREACHABLE`` -- the target is not strictly inside ``(l, u)``. Only checked when
   ``fit.status is OK``, since reachability needs a fitted ``l``/``u`` (`decisions/0008`).
4. ``RIGHT`` -- the highest tested level's one-sided *lower* bound at ``level`` still clears the
   target.
5. ``LEFT`` -- the lowest *nonzero* tested level's one-sided *upper* bound at ``level`` fails the
   target, and the control's one-sided lower bound clears it (`decisions/0012`: evaluated on the
   data, so this and ``RIGHT`` are reachable on a converged fit too).
6. ``fit.status is OK`` and neither ``RIGHT`` nor ``LEFT`` fired -- solve (the caller runs the
   requested interval method); ``Classification.solve`` is ``True``.
7. Otherwise (``SEPARATION``/``NOT_CONVERGED``) -- the exact-bound fallback: attach the grid
   value, then build a two-sided bracket from every tested cell (control included), each side at
   ``(1 + level) / 2`` (`decisions/0009`, so joint coverage is at least ``level``). A bracket
   whose largest confidently-passing severity is not below its smallest confidently-failing one
   contradicts monotonicity; report no bounds at all with both levels named in a warning
   (`decisions/0011`), keeping the fit's own status.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from scipy.stats import binomtest

from marginkit.empirical import BaselineRate, GridBreakPoint, grid_break_point
from marginkit.models import Fit
from marginkit.types import Cell, Censoring, Status, _check_finite_number

__all__ = ["Classification", "classify", "exact_one_sided_bound"]

_VALID_SIDES = frozenset({"lower", "upper"})

# The two statuses that mean "the fit failed but the data still supports an exact
# statement" -- plan section 5.2's fallback, and the only two `Threshold` allows `grid` on.
_NONCONVERGENT_STATUSES = frozenset({Status.SEPARATION, Status.NOT_CONVERGED})


def exact_one_sided_bound(
    successes: int, trials: int, *, level: float, side: Literal["lower", "upper"]
) -> float:
    """The exact (Clopper-Pearson) one-sided confidence bound on a binomial proportion at
    confidence ``level`` (`decisions/0009`).

    ``scipy.stats.binomtest(successes, trials).proportion_ci`` is two-sided: at
    ``confidence_level = c`` it puts ``(1 - c) / 2`` in each tail, so each endpoint is itself a
    one-sided bound at confidence ``(1 + c) / 2``. To get a one-sided bound at confidence
    ``level``, this passes ``confidence_level = 2 * level - 1`` and reads the requested side.
    This is the *only* place in marginkit this conversion is written down; every other exact
    bound in the package (per-cell rates, the censoring rules below) calls this function rather
    than repeating it.

    Parameters
    ----------
    successes, trials
        The observed count and the number of trials, as taken by ``scipy.stats.binomtest``.
    level
        The one-sided confidence level, strictly between 0 and 1.
    side
        ``"lower"`` for a one-sided lower bound (how confident a *lower* limit on the true
        proportion is), ``"upper"`` for a one-sided upper bound.

    Returns
    -------
    float
        The requested bound, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``side`` is not ``"lower"`` or ``"upper"``, or if ``level`` is not a finite number
        strictly between 0 and 1.
    """
    if side not in _VALID_SIDES:
        raise ValueError(
            f"exact_one_sided_bound: side must be one of {sorted(_VALID_SIDES)}, got {side!r}"
        )
    _check_finite_number(level, label="exact_one_sided_bound level")
    # The `2 * level - 1` conversion below is defined only above 0.5: at or under it scipy is
    # handed a non-positive `confidence_level` and raises naming a number the caller never
    # supplied. A one-sided bound at 0.5 or less is not a meaningful request anyway.
    if not (0.5 < level < 1.0):
        raise ValueError(
            f"exact_one_sided_bound: level must satisfy 0.5 < level < 1, got {level!r}"
        )

    confidence_level = 2.0 * level - 1.0
    result = binomtest(successes, trials).proportion_ci(
        confidence_level=confidence_level, method="exact"
    )
    return float(result.low) if side == "lower" else float(result.high)


@dataclass(frozen=True, kw_only=True)
class Classification:
    """The outcome of :func:`classify`: everything :func:`marginkit.threshold` needs to build a
    :class:`~marginkit.Threshold`, other than the solved ``value`` itself (which, when
    ``solve`` is ``True``, is the caller's job -- computed from ``fit.params`` directly, then
    refined by the requested interval method).

    Attributes
    ----------
    status
        The :class:`~marginkit.Status` the resulting threshold should report.
    censoring
        The :class:`~marginkit.Censoring` the resulting threshold should report. ``NONE`` unless
        ``RIGHT``/``LEFT`` fired.
    lo, hi
        Severity-scale exact bounds, when this classification determined one directly (``RIGHT``,
        ``LEFT``, or a two-sided ``SEPARATION``/``NOT_CONVERGED`` bracket). ``None`` otherwise.
    baseline
        Set exactly when ``status is Status.FAILS_AT_BASELINE``.
    grid
        Set exactly on the ``SEPARATION``/``NOT_CONVERGED`` exact-bound fallback (step 7).
    warnings
        Free-text notes to append to the resulting threshold's own ``warnings``.
    solve
        ``True`` only for the ``fit.status is OK``, non-``RIGHT``/``LEFT`` case: the caller must
        solve for ``value`` and run the requested interval method.
    """

    status: Status
    censoring: Censoring
    lo: float | None
    hi: float | None
    baseline: BaselineRate | None
    grid: GridBreakPoint | None
    warnings: tuple[str, ...] = ()
    solve: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))


def _control_cell(cells: tuple[Cell, ...]) -> Cell | None:
    for cell in cells:
        if cell.severity == 0.0:
            return cell
    return None


def _baseline_rate(control: Cell, *, level: float) -> BaselineRate:
    """The control cell's rate with its exact interval, for a `FAILS_AT_BASELINE` result.

    This is a **two-sided** Clopper-Pearson interval at `level`, which is what plan section 5.4
    specifies for a reported per-cell rate: "Use `binomtest(k, n).proportion_ci(
    confidence_level=..., method='exact')`. Label these `per_cell_clopper_pearson`."

    It is deliberately *not* the one-sided bound that decided the `FAILS_AT_BASELINE`
    classification. Those are two different jobs: the classification asks "does this cell
    provably fail the target", a one-sided question answered at `level` by
    `exact_one_sided_bound`; the reported interval describes where the control rate lies, a
    two-sided question. Reporting the one-sided pair under the label
    `per_cell_clopper_pearson` would give a roughly `2*level - 1` region wearing a `level`
    name.
    """
    lo, hi = binomtest(control.successes, control.trials).proportion_ci(
        confidence_level=level, method="exact"
    )
    return BaselineRate(
        severity=control.severity,
        successes=control.successes,
        trials=control.trials,
        rate=control.successes / control.trials,
        lo=float(lo),
        hi=float(hi),
        level=level,
        method="per_cell_clopper_pearson",
    )


def _fails_at_baseline(
    cells: tuple[Cell, ...], *, target: float, level: float
) -> Classification | None:
    control = _control_cell(cells)
    if control is None:
        return None
    control_upper = exact_one_sided_bound(
        control.successes, control.trials, level=level, side="upper"
    )
    if not control_upper < target:
        return None
    # decisions/0011: a control that fails is only a *clean* baseline failure if nothing else
    # tested ever confidently passes the same target -- otherwise this is a monotonicity
    # contradiction (a later level passing while the control fails), which step 7 reports
    # instead, naming both levels.
    other_passes = any(
        exact_one_sided_bound(c.successes, c.trials, level=level, side="lower") >= target
        for c in cells
        if c.severity > 0.0
    )
    if other_passes:
        return None
    return Classification(
        status=Status.FAILS_AT_BASELINE,
        censoring=Censoring.NONE,
        lo=None,
        hi=None,
        baseline=_baseline_rate(control, level=level),
        grid=None,
    )


def _unreachable(fit: Fit, *, target: float) -> Classification | None:
    if fit.status is not Status.OK:
        return None
    assert fit.params is not None
    lower = fit.params["lower"].value
    upper = fit.params["upper"].value
    if lower < target < upper:
        return None
    return Classification(
        status=Status.UNREACHABLE,
        censoring=Censoring.NONE,
        lo=None,
        hi=None,
        baseline=None,
        grid=None,
    )


def _right(
    fit: Fit, cells: tuple[Cell, ...], *, target: float, level: float
) -> Classification | None:
    highest = cells[-1]
    if exact_one_sided_bound(highest.successes, highest.trials, level=level, side="lower") < target:
        return None
    return Classification(
        status=fit.status,
        censoring=Censoring.RIGHT,
        lo=highest.severity,
        hi=None,
        baseline=None,
        grid=None,
    )


def _left(
    fit: Fit, cells: tuple[Cell, ...], *, target: float, level: float
) -> Classification | None:
    nonzero = [c for c in cells if c.severity > 0.0]
    if not nonzero:
        return None
    lowest_nonzero = nonzero[0]
    lowest_fails = (
        exact_one_sided_bound(
            lowest_nonzero.successes, lowest_nonzero.trials, level=level, side="upper"
        )
        < target
    )
    if not lowest_fails:
        return None
    control = _control_cell(cells)
    if control is None:
        return None
    control_passes = (
        exact_one_sided_bound(control.successes, control.trials, level=level, side="lower")
        >= target
    )
    if not control_passes:
        return None
    return Classification(
        status=fit.status,
        censoring=Censoring.LEFT,
        lo=None,
        hi=lowest_nonzero.severity,
        baseline=None,
        grid=None,
    )


def _grid_for(cells: tuple[Cell, ...], *, target: float) -> GridBreakPoint:
    severity = [c.severity for c in cells]
    performance = [c.successes / c.trials for c in cells]
    return grid_break_point(severity, performance, criterion=target, signed=False)


def _contradiction_warning(
    *,
    lo_severity: float,
    lo_bound: float,
    hi_severity: float,
    hi_bound: float,
    target: float,
    level: float,
    curve_exists: bool,
) -> str:
    """``curve_exists`` picks the one sentence that differs between this function's two
    callers, so the rest of the wording -- and the frozen ``NON_MONOTONE: `` token -- is shared.

    ``True`` (``_contradiction_or_none``'s own ``Status.OK`` branch): the fit converged, so
    ``threshold()`` still solves for ``value`` and still runs the requested interval method --
    only the *exact* one-sided bounds (the suppressed ``RIGHT``/``LEFT`` claim) go unreported,
    not the profile/delta ``lo``/``hi`` this ``Threshold`` actually carries.

    ``False`` (``_contradiction_or_none``'s non-``OK`` branch, and ``_exact_bound_fallback``,
    which is reached only for a failed fit): there is no fitted curve at all, and this
    ``Threshold``'s ``value``/``lo``/``hi`` are all ``None`` -- "no bounds are reported" is
    simply true here, with nothing to qualify.
    """
    exact_bounds_clause = (
        "the exact one-sided bounds are not reported here -- any lo/hi this Threshold carries "
        "comes from the fitted curve instead, not from these exact tests"
        if curve_exists
        else "no bounds are reported"
    )
    return (
        # `decisions/0011`'s second amendment: the leading `NON_MONOTONE: ` token is the
        # supported way to detect this condition -- distinct from `Fit.monotonicity`'s own
        # structured, data-only adjacent-level check (`decisions/0023` "Amendment 1"), which
        # compares raw counts against each other rather than exact bounds against the *fitted*
        # curve; the two can disagree, and both are reported when they fire (see
        # `threshold.py`'s `"NON_MONOTONE_DATA: "` token). On a converged fit this warning is the
        # *only* signal of *this* kind of contradiction -- `grid` is legal only for
        # SEPARATION/NOT_CONVERGED (`decisions/0010`), so a consumer filtering on `status is OK
        # and censoring is NONE` would otherwise see a clean bounded interval from data the exact
        # tests just proved inconsistent with any monotone curve. Changing or removing this token
        # is a behaviourally breaking change.
        "NON_MONOTONE: censoring.classify: exact one-sided tests at confidence "
        f"{level} contradict monotonicity: severity {lo_severity} confidently "
        f"passes (lower bound {lo_bound!r} >= target {target!r}) while the lower severity "
        f"{hi_severity} confidently fails (upper bound {hi_bound!r} < target {target!r}); "
        f"{exact_bounds_clause}"
    )


def _scan_bounds(
    cells: tuple[Cell, ...], *, target: float, level: float
) -> tuple[float | None, float, float | None, float]:
    """The highest severity that confidently passes and the lowest that confidently fails.

    The passing scan skips severity 0: "the threshold is above 0" is true of every threshold by
    construction, so a control that passes is not a bound. The failing scan keeps it, because a
    control that *fails* is genuinely informative and is what a monotonicity contradiction is
    usually built from.
    """
    lo_severity: float | None = None
    lo_bound = 0.0
    hi_severity: float | None = None
    hi_bound = 0.0
    for cell in cells:
        if cell.severity > 0.0:
            bound = exact_one_sided_bound(cell.successes, cell.trials, level=level, side="lower")
            if bound >= target and (lo_severity is None or cell.severity > lo_severity):
                lo_severity, lo_bound = cell.severity, bound
        bound_up = exact_one_sided_bound(cell.successes, cell.trials, level=level, side="upper")
        if bound_up < target and (hi_severity is None or cell.severity < hi_severity):
            hi_severity, hi_bound = cell.severity, bound_up
    return lo_severity, lo_bound, hi_severity, hi_bound


def _contradiction_or_none(
    fit: Fit, cells: tuple[Cell, ...], *, target: float, level: float
) -> Classification | None:
    """`decisions/0011`, applied *before* the `RIGHT`/`LEFT` branches rather than only inside
    the bracket.

    Both short-circuits return a directional claim about where the threshold lies -- "sigma* >
    x_max", "sigma* < x_min" -- and every such claim rests on the monotonicity assumption stated
    in `Threshold`'s docstring. When the exact tests themselves contradict that assumption, the
    premise of the claim has failed, so the claim is not a weaker true statement: it is a
    statement whose premise is false. Reporting it with no diagnostic is what `0011` forecloses.

    The confidence is `level`, not the bracket's `(1 + level) / 2`, because these are the
    one-sided bounds the `RIGHT`/`LEFT` rules are themselves evaluated at (`decisions/0009`).
    """
    lo_severity, lo_bound, hi_severity, hi_bound = _scan_bounds(cells, target=target, level=level)
    if lo_severity is None or hi_severity is None or lo_severity < hi_severity:
        return None
    warning = _contradiction_warning(
        lo_severity=lo_severity,
        lo_bound=lo_bound,
        hi_severity=hi_severity,
        hi_bound=hi_bound,
        target=target,
        level=level,
        curve_exists=fit.status is Status.OK,
    )
    if fit.status is Status.OK:
        # A converged fit still has a model-based estimate, and `Threshold` requires `value`
        # whenever status is OK and censoring is NONE -- `decisions/0011`'s all-`None` encoding
        # is only constructible on a failed fit, which is the case it was written for. So the
        # contradiction suppresses the `RIGHT`/`LEFT` branches (whose directional claim rests on
        # the monotonicity the data just contradicted) and falls through to the solve path,
        # carrying the warning. The caller gets the model's answer plus an explicit statement
        # that the exact tests disagree with the model's monotonicity assumption, which is what
        # plan section 5.7 means by a diagnostic that is reported and never acted on.
        return Classification(
            status=Status.OK,
            censoring=Censoring.NONE,
            lo=None,
            hi=None,
            baseline=None,
            grid=None,
            warnings=(warning,),
            solve=True,
        )
    return Classification(
        status=fit.status,
        censoring=Censoring.NONE,
        lo=None,
        hi=None,
        baseline=None,
        grid=_grid_for(cells, target=target),
        warnings=(warning,),
    )


def _exact_bound_fallback(
    fit: Fit, cells: tuple[Cell, ...], *, target: float, level: float
) -> Classification:
    grid = _grid_for(cells, target=target)
    bracket_level = (1.0 + level) / 2.0

    # The lower side scans only *nonzero* severities. A control at severity 0 that passes says
    # "the threshold is above 0", which every threshold satisfies by construction -- `Threshold`
    # requires `>= 0` on a linear axis and `> 0` on a log one -- so it is not a bound at all. On
    # a log axis it is also illegal: `lo = 0.0` is rejected outright. The upper side keeps the
    # control, because a control that *fails* is genuinely informative (and is what the
    # monotonicity contradiction of `decisions/0011` is usually built from).
    lo_severity: float | None = None
    lo_bound = 0.0
    for cell in cells:
        if cell.severity <= 0.0:
            continue
        bound = exact_one_sided_bound(
            cell.successes, cell.trials, level=bracket_level, side="lower"
        )
        if bound >= target and (lo_severity is None or cell.severity > lo_severity):
            lo_severity = cell.severity
            lo_bound = bound

    hi_severity: float | None = None
    hi_bound = 0.0
    for cell in cells:
        bound = exact_one_sided_bound(
            cell.successes, cell.trials, level=bracket_level, side="upper"
        )
        if bound < target and (hi_severity is None or cell.severity < hi_severity):
            hi_severity = cell.severity
            hi_bound = bound

    if lo_severity is not None and hi_severity is not None and lo_severity >= hi_severity:
        # `_exact_bound_fallback` is reached only for a failed fit (`classify` returns directly
        # on `Status.OK` before ever calling it) -- `curve_exists=False` unconditionally, not
        # `fit.status is Status.OK`, so that invariant is visible here rather than merely relied
        # on.
        warning = _contradiction_warning(
            lo_severity=lo_severity,
            lo_bound=lo_bound,
            hi_severity=hi_severity,
            hi_bound=hi_bound,
            target=target,
            level=bracket_level,
            curve_exists=False,
        )
        return Classification(
            status=fit.status,
            censoring=Censoring.NONE,
            lo=None,
            hi=None,
            baseline=None,
            grid=grid,
            warnings=(warning,),
        )

    warnings: tuple[str, ...] = ()
    if lo_severity is None and hi_severity is None:
        warnings = (
            "censoring.classify: no tested cell's exact one-sided bound at confidence "
            f"{bracket_level} confidently passed or failed the target {target!r}; no bounds "
            "are reported",
        )
    elif lo_severity is None or hi_severity is None:
        # Only one side is determined, so this is not a bracket and `decisions/0009`'s
        # `(1 + level) / 2` split does not apply: a single one-sided statement belongs at
        # `level`, like every other one-sided bound in the package.
        #
        # Which side that is must be decided from the *bracket-level* scan above, before the
        # rescan, and the other side must be forced back to `None`. The looser `level` can
        # determine a side that `(1 + level) / 2` did not -- a `0/5` cell has an upper bound of
        # 0.5218 at 0.975 but 0.4507 at 0.95, so against a target of 0.5 it is undetermined at
        # the bracket level and failing at `level`. Taking both sides from the rescan therefore
        # manufactures a two-sided bracket with each side at `level`, whose joint coverage is
        # about `2 * level - 1` while `Threshold.level` still reports `level`. That is exactly
        # the mislabelling this branch exists to prevent, and an earlier version of it did
        # precisely that (`REVIEWS.md` R8 #18).
        determined = "lower" if hi_severity is None else "upper"
        rescanned_lo, _, rescanned_hi, _ = _scan_bounds(cells, target=target, level=level)
        if determined == "lower":
            lo_severity, hi_severity = rescanned_lo, None
        else:
            lo_severity, hi_severity = None, rescanned_hi
        warnings = (
            "censoring.classify: the exact-bound fallback determined only the "
            f"{determined} side at confidence {level} -- no tested cell confidently "
            f"{'failed' if determined == 'lower' else 'passed'} the target {target!r} -- so "
            "this is a one-sided exact bound, not a bracket. Censoring stays NONE because the "
            "bound sits at a tested level inside the range, which is not what RIGHT/LEFT "
            "describe",
        )
    return Classification(
        status=fit.status,
        censoring=Censoring.NONE,
        lo=lo_severity,
        hi=hi_severity,
        baseline=None,
        grid=grid,
        warnings=warnings,
    )


def _with_grid(
    classification: Classification, fit: Fit, cells: tuple[Cell, ...], *, target: float
) -> Classification:
    """Attach the grid break-point when, and only when, the fit failed (`decisions/0010`).

    ``Threshold`` permits ``grid`` only for ``SEPARATION`` and ``NOT_CONVERGED``, so a
    converged fit's one-sided bound is returned unchanged.
    """
    if fit.status not in _NONCONVERGENT_STATUSES:
        return classification
    return replace(classification, grid=_grid_for(cells, target=target))


def classify(fit: Fit, *, target: float, level: float) -> Classification:
    """Classify a fit against a resolved target performance, per plan section 5.5 (order per the
    module docstring).

    Parameters
    ----------
    fit
        The :class:`~marginkit.Fit` to classify. Its ``cells`` (sorted ascending by
        :meth:`~marginkit.Observations.cells`) are what every exact test here is run against.
    target
        The already-resolved target performance ``P*`` (plan section 5.3). Resolving it from a
        :class:`~marginkit.Definition` -- including which ``u`` a failed-fit fallback uses
        (`decisions/0008`) -- is the caller's job.
    level
        The confidence level, strictly between 0 and 1.

    Returns
    -------
    Classification
    """
    if fit.status is Status.CONTROL_INCOMPATIBLE:
        return Classification(
            status=Status.CONTROL_INCOMPATIBLE,
            censoring=Censoring.NONE,
            lo=None,
            hi=None,
            baseline=None,
            grid=None,
        )

    cells = fit.cells

    fails_at_baseline = _fails_at_baseline(cells, target=target, level=level)
    if fails_at_baseline is not None:
        return fails_at_baseline

    unreachable = _unreachable(fit, target=target)
    if unreachable is not None:
        return unreachable

    # `RIGHT`/`LEFT` are reached on a converged fit as well as a failed one (`decisions/0012`),
    # so the grid statistic is attached here rather than inside `_right`/`_left`: plan section
    # 5.5 pairs the grid rule with the exact bounds for separation and non-convergence, and a
    # one-sided bound on a failed fit is that same fallback. On an `OK` fit `Threshold` forbids
    # `grid`, so the rule is stated once, in one place, and cannot leak into the `OK` branch.
    # `decisions/0011` before the directional branches: a `RIGHT`/`LEFT` bound asserts where
    # the threshold lies, and that assertion rests on monotonicity. If the exact tests
    # contradict monotonicity, no such bound may be reported (C2, Phase 5 stats review).
    contradiction = _contradiction_or_none(fit, cells, target=target, level=level)
    if contradiction is not None:
        return contradiction

    right = _right(fit, cells, target=target, level=level)
    if right is not None:
        return _with_grid(right, fit, cells, target=target)

    left = _left(fit, cells, target=target, level=level)
    if left is not None:
        return _with_grid(left, fit, cells, target=target)

    if fit.status is Status.OK:
        return Classification(
            status=Status.OK,
            censoring=Censoring.NONE,
            lo=None,
            hi=None,
            baseline=None,
            grid=None,
            solve=True,
        )

    return _exact_bound_fallback(fit, cells, target=target, level=level)
