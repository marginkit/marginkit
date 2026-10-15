"""Fit diagnostics (model-only) and the data-only adjacent-level monotonicity check (plan
section 5.7; ``docs/decisions/0023-fit-diagnostics-and-schema-v3.md``, as amended by that
decision's "Amendment 1" (monotonicity moves onto ``Fit`` directly), "Amendment 2" (the
structural goodness-of-fit exclusion and the ``SPARSE_CELLS: `` warning), "Amendment 3" (that
warning fires only on an observed event, not a bare small expectation) and "Amendment 4" (the
warning's expected-count cutoff, tightened to below ``0.1``)).

Two independent things live here, deliberately kept apart (Amendment 1's own rationale):

- :class:`Diagnostics` / :func:`diagnose` need a converged fit -- goodness of fit, dispersion,
  and link comparison are all statements about how well a fitted curve matches the data, so they
  are ``None`` whenever :attr:`~marginkit.Fit.status` is not :data:`~marginkit.Status.OK`
  (hard constraint 3, read as "no model-derived numbers on a failed fit").
- :class:`MonotonicityCheck` / :func:`check_monotonicity` need only raw per-level counts -- an
  exact one-sided Fisher test on adjacent severity levels, Holm-corrected across the family. It
  therefore runs on *every* :func:`~marginkit.fit_dose_response` return path, whatever the
  ``Status``, because non-monotone data is a common cause of a failed fit and disappearing the
  check exactly then would hide the finding it exists to surface.

Both are reported and never acted on automatically (plan section 5.7): nothing here changes a
reported number, chooses a link, or corrects for dispersion. ``Fit.link`` stays whatever the
caller passed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import xlogy
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from marginkit.types import Cell, Observations, Status, _check_finite_number

if TYPE_CHECKING:
    # Only for the `diagnose(fit: Fit)` annotation. `from __future__ import annotations` (above)
    # makes every annotation in this module a string, so this import is never evaluated at
    # runtime -- it exists purely for mypy, and keeps this module from ever importing
    # marginkit.models at import time (models.py imports *this* module at its own top level, to
    # build Fit.diagnostics/Fit.monotonicity and to call diagnose()/check_monotonicity() from
    # fit_dose_response; the reverse, real import would be a cycle).
    from marginkit.models import Fit

__all__ = [
    "AdjacentPair",
    "Diagnostics",
    "MonotonicityCheck",
    "check_monotonicity",
    "diagnose",
]

# plan section 5.7: "a warning when phi > 1.5" -- strict, an exact 1.5 does not warn.
_DISPERSION_WARNING_THRESHOLD = 1.5
_OVERDISPERSED_WARNING_PREFIX = "OVERDISPERSED: "
_SPARSE_CELLS_WARNING_PREFIX = "SPARSE_CELLS: "
# decisions/0023 Amendment 4 (tightening Amendment 3, itself correcting Amendment 2 item 3): a
# counted cell is flagged on one side (successes or failures) only when *both* hold -- the
# expected count on that side is below this cutoff, and the *observed* count on that side is at
# least 1. A small expected count with nothing observed there is a clean plateau, not a finding
# (Amendment 3); an observed event against a merely-moderate expectation (down to about 0.43 near
# a design's transition) is ordinary too (Amendment 4) -- only a genuine event against an
# expectation this small is the "dominated by a single event" pathology (Amendment 2's own "W1"
# finding) the warning exists for. Report-only: nothing here is recomputed. A fixed constant, not
# a parameter -- changing it needs a new amendment (Amendment 4's own "Consequences").
_SPARSE_EXPECTED_CUTOFF = 0.1
# decisions/0023 Amendment 2, item 2: R's own `binomial()$linkinv` clamp (`.Machine$double.eps`).
_PROBABILITY_EPS = float(np.finfo(float).eps)

# decisions/0023 item 3: link comparison covers exactly these three v1 binomial links.
_COMPARISON_LINKS: tuple[str, ...] = ("probit", "logit", "cloglog")

_MONOTONICITY_METHOD = "fisher_exact_one_sided_holm"
_VALID_DIRECTIONS = frozenset({"decreasing", "increasing"})
_VALID_DEPENDENCE_INPUTS = frozenset({"independent"})


@dataclass(frozen=True, kw_only=True)
class Diagnostics:
    """Model-only fit diagnostics: goodness of fit, dispersion, and link comparison
    (plan section 5.7, ``decisions/0023`` as amended by that decision's "Amendment 1", which
    moved the monotonicity check off this type entirely -- see :class:`MonotonicityCheck`).

    Built only from a converged fit (:func:`diagnose` requires ``fit.status is Status.OK``), and
    attached to :attr:`~marginkit.Fit.diagnostics` only in that case: ``None`` whenever the fit
    failed (hard constraint 3, read as "no model-derived numbers on a failed fit"). Contrast
    :class:`MonotonicityCheck`, attached to :attr:`~marginkit.Fit.monotonicity` on *every*
    ``fit_dose_response`` return path regardless of ``Status`` -- the two appended fields on
    ``Fit`` have different nullability for exactly this reason.

    Attributes
    ----------
    deviance
        The binomial deviance on the per-severity-level cells, or ``None`` when ``df <= 0``
        (see ``df`` below) -- not a failed fit, just no goodness-of-fit statistic to report. A
        cell is excluded from this sum only when it is *structurally* deterministic under the
        model -- a log-axis cell at severity exactly ``0.0`` when ``upper`` is *fixed* at exactly
        ``1.0`` (``direction="decreasing"``); that cell has zero variance and zero deviance by
        construction (`decisions/0023` Amendment 2, item 1). Every other cell counts, including
        one whose fitted probability happens to round to exactly ``0`` or ``1`` in double
        precision: probabilities are computed to avoid needless cancellation and clamped to
        ``[eps, 1 - eps]`` (``eps`` the machine epsilon, matching R's own
        ``binomial()$linkinv``), so such a cell contributes approximately zero rather than a
        ``0/0``. An earlier implementation excluded by *computed* value instead of by structure,
        which made ``df`` depend on the link (Amendment 2's own "C1" finding) -- corrected here.
    pearson_chi2
        The Pearson chi-square statistic on the same cells, with the same exclusion and the
        same ``None``-on-``df <= 0`` rule as ``deviance``.
    df
        Degrees of freedom: the number of cells counted towards ``deviance``/``pearson_chi2``
        (after the structural exclusion above) minus the number of estimated parameters. Always
        an integer, even when it is zero or negative.
    dispersion
        ``pearson_chi2 / df`` (plan section 5.7's ``phi``), or ``None`` exactly when
        ``pearson_chi2`` is ``None`` (``df <= 0``). A value strictly greater than ``1.5`` adds a
        warning starting with ``"OVERDISPERSED: "``; nothing is rescaled and no interval
        changes as a result (plan section 5.7: reported, never acted on). ``phi`` can be
        dominated by a single observed event against a tiny expected count (`decisions/0023`
        Amendment 2's own "W1" finding) -- see ``warnings``' ``"SPARSE_CELLS: "`` entry, which
        flags exactly that case without changing ``dispersion`` itself.
    link_aic
        AIC for each of ``"probit"``, ``"logit"`` and ``"cloglog"``, refit under the same
        ``upper``/``lower`` specification as the diagnosed fit. The diagnosed fit's own link
        reuses its own ``log_likelihood`` and parameter count directly (never refit, so it is
        exact by construction); the other two links are refit through the existing statsmodels
        fitting paths (hard constraint 2). ``None`` for a link whose refit does not reach
        :data:`~marginkit.Status.OK` -- see ``link_status``.
    link_status
        The :class:`~marginkit.Status` of each link's own fit (the diagnosed fit's own status,
        which is always ``OK``, for its own link; the refit's status for the other two).
        ``link_aic[link] is None`` if and only if ``link_status[link] is not Status.OK``.
    warnings
        Free-text notes: the ``df <= 0`` reason, an ``"OVERDISPERSED: "`` token, or a
        ``"SPARSE_CELLS: "`` token naming a counted cell flagged on one side (successes or
        failures) because that side's expected count is below ``0.1`` *and* at least one event
        was actually observed there (`decisions/0023` Amendments 2-4) -- report-only, like
        ``"OVERDISPERSED: "``: it changes no number, only says that ``pearson_chi2`` and
        ``dispersion`` are unreliable at that cell. Empty by default.
    """

    deviance: float | None
    pearson_chi2: float | None
    df: int
    dispersion: float | None
    link_aic: Mapping[str, float | None]
    link_status: Mapping[str, Status]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "link_aic", dict(self.link_aic))
        object.__setattr__(self, "link_status", dict(self.link_status))

        if isinstance(self.df, bool) or not isinstance(self.df, int):
            raise ValueError(f"Diagnostics.df must be a genuine int, got {self.df!r}")

        gof_fields = (self.deviance, self.pearson_chi2, self.dispersion)
        gof_all_none = all(v is None for v in gof_fields)
        gof_all_set = all(v is not None for v in gof_fields)
        if not (gof_all_none or gof_all_set):
            raise ValueError(
                "Diagnostics.deviance, .pearson_chi2 and .dispersion must be either all None "
                f"(df <= 0) or all set, got deviance={self.deviance!r}, "
                f"pearson_chi2={self.pearson_chi2!r}, dispersion={self.dispersion!r}"
            )
        if gof_all_none and self.df > 0:
            raise ValueError(
                "Diagnostics.deviance/.pearson_chi2/.dispersion are None, but df "
                f"({self.df!r}) is > 0 -- they are only None when df <= 0"
            )
        if gof_all_set:
            assert self.deviance is not None and self.pearson_chi2 is not None
            assert self.dispersion is not None
            _check_finite_number(self.deviance, label="Diagnostics.deviance")
            _check_finite_number(self.pearson_chi2, label="Diagnostics.pearson_chi2")
            _check_finite_number(self.dispersion, label="Diagnostics.dispersion")
            if self.df <= 0:
                raise ValueError(
                    f"Diagnostics.df is {self.df!r} (<= 0), so deviance/pearson_chi2/dispersion "
                    "must be None"
                )

        if set(self.link_aic) != set(_COMPARISON_LINKS):
            raise ValueError(
                f"Diagnostics.link_aic must have exactly the keys {sorted(_COMPARISON_LINKS)}, "
                f"got {sorted(self.link_aic)}"
            )
        if set(self.link_status) != set(_COMPARISON_LINKS):
            raise ValueError(
                "Diagnostics.link_status must have exactly the keys "
                f"{sorted(_COMPARISON_LINKS)}, got {sorted(self.link_status)}"
            )
        for link in _COMPARISON_LINKS:
            aic = self.link_aic[link]
            status = self.link_status[link]
            if not isinstance(status, Status):
                raise ValueError(
                    f"Diagnostics.link_status[{link!r}] must be a Status, got {status!r}"
                )
            if aic is None:
                if status is Status.OK:
                    raise ValueError(
                        f"Diagnostics.link_aic[{link!r}] is None, but "
                        f"link_status[{link!r}] is Status.OK -- an OK refit always has an AIC"
                    )
            else:
                _check_finite_number(aic, label=f"Diagnostics.link_aic[{link!r}]")
                if status is not Status.OK:
                    raise ValueError(
                        f"Diagnostics.link_aic[{link!r}] is set, but link_status[{link!r}] is "
                        f"{status!r}, not Status.OK"
                    )


@dataclass(frozen=True, kw_only=True)
class AdjacentPair:
    """One adjacent-severity-level pair from a :class:`MonotonicityCheck`: the raw counts at
    both levels, the one-sided Fisher exact test between them, and its Holm-adjusted p-value
    (``docs/decisions/0023-fit-diagnostics-and-schema-v3.md``'s "Amendment 1", item 4).

    Attributes
    ----------
    severity_low, severity_high
        The two adjacent severity levels, ``severity_low < severity_high``.
    successes_low, trials_low, successes_high, trials_high
        The raw pooled counts at each level (:meth:`~marginkit.Observations.cells`).
    p_value
        The raw one-sided Fisher exact p-value for a reversal against ``direction`` (that is, the
        higher severity performing *better* than the lower one, which is evidence against
        ``direction="decreasing"``; the mirror image against ``"increasing"``) --
        :func:`scipy.stats.fisher_exact` on
        ``[[successes_high, failures_high], [successes_low, failures_low]]``,
        ``alternative="greater"`` for ``direction="decreasing"``, ``"less"`` for
        ``"increasing"``.
    holm_adjusted_p
        ``p_value`` after a Holm step-down correction across every pair in the same
        :class:`MonotonicityCheck` (:func:`statsmodels.stats.multitest.multipletests`,
        ``method="holm"``).
    flagged
        ``holm_adjusted_p < alpha`` (``alpha`` recorded on the enclosing
        :class:`MonotonicityCheck`).
    """

    severity_low: float
    severity_high: float
    successes_low: int
    trials_low: int
    successes_high: int
    trials_high: int
    p_value: float
    holm_adjusted_p: float
    flagged: bool

    def __post_init__(self) -> None:
        _check_finite_number(self.severity_low, label="AdjacentPair.severity_low")
        _check_finite_number(self.severity_high, label="AdjacentPair.severity_high")
        if not self.severity_low < self.severity_high:
            raise ValueError(
                "AdjacentPair.severity_low must be strictly less than .severity_high, got "
                f"severity_low={self.severity_low!r}, severity_high={self.severity_high!r}"
            )
        for label, value in (
            ("successes_low", self.successes_low),
            ("trials_low", self.trials_low),
            ("successes_high", self.successes_high),
            ("trials_high", self.trials_high),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"AdjacentPair.{label} must be a genuine int, got {value!r}")
        if not (0 <= self.successes_low <= self.trials_low):
            raise ValueError(
                "AdjacentPair requires 0 <= successes_low <= trials_low, got "
                f"successes_low={self.successes_low!r}, trials_low={self.trials_low!r}"
            )
        if not (0 <= self.successes_high <= self.trials_high):
            raise ValueError(
                "AdjacentPair requires 0 <= successes_high <= trials_high, got "
                f"successes_high={self.successes_high!r}, trials_high={self.trials_high!r}"
            )
        _check_finite_number(self.p_value, label="AdjacentPair.p_value")
        _check_finite_number(self.holm_adjusted_p, label="AdjacentPair.holm_adjusted_p")
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError(f"AdjacentPair.p_value must lie in [0, 1], got {self.p_value!r}")
        if not (0.0 <= self.holm_adjusted_p <= 1.0):
            raise ValueError(
                f"AdjacentPair.holm_adjusted_p must lie in [0, 1], got {self.holm_adjusted_p!r}"
            )
        if not isinstance(self.flagged, bool):
            raise ValueError(f"AdjacentPair.flagged must be a genuine bool, got {self.flagged!r}")


@dataclass(frozen=True, kw_only=True)
class MonotonicityCheck:
    """A data-only adjacent-level monotonicity check (``decisions/0023``'s "Amendment 1"): for
    each pair of adjacent severity levels in the data, a one-sided Fisher exact test for "the
    higher severity performs better" under ``direction``, Holm-corrected across the whole
    family.

    Needs only raw counts, never a fitted curve, so it is attached to
    :attr:`~marginkit.Fit.monotonicity` on *every* :func:`~marginkit.fit_dose_response` return
    path, whatever the resulting :class:`~marginkit.Status` -- contrast :class:`Diagnostics`,
    which needs a converged fit and is ``None`` otherwise. See :func:`check_monotonicity`.

    Attributes
    ----------
    direction
        ``"decreasing"`` or ``"increasing"``, read from the ``Observations`` the check was built
        from (never passed separately, so the check cannot disagree with its own data).
    alpha
        The significance level ``flagged`` is decided at, strictly between 0 and 1.
    dependence
        ``"independent"`` in this release: the check assumes independent trials. Recorded so a
        serialised result states the assumption it relies on.
    method
        Names the statistical method. Always ``"fisher_exact_one_sided_holm"`` in this release.
    pairs
        One :class:`AdjacentPair` per adjacent pair of severity levels, ascending. Empty when
        fewer than two distinct severity levels exist (``warnings`` then names the reason).
    any_flagged
        ``True`` exactly when at least one entry of ``pairs`` is ``flagged``.
    warnings
        Free-text notes (for example, naming a degenerate single-level input). Empty by default.
    """

    direction: str
    alpha: float
    dependence: str
    pairs: tuple[AdjacentPair, ...]
    any_flagged: bool
    method: str = _MONOTONICITY_METHOD
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", tuple(self.pairs))
        object.__setattr__(self, "warnings", tuple(self.warnings))

        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"MonotonicityCheck.direction must be one of {sorted(_VALID_DIRECTIONS)}, "
                f"got {self.direction!r}"
            )
        _check_finite_number(self.alpha, label="MonotonicityCheck.alpha")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(
                f"MonotonicityCheck.alpha must satisfy 0 < alpha < 1, got {self.alpha!r}"
            )
        if self.dependence not in _VALID_DEPENDENCE_INPUTS:
            raise ValueError(
                "MonotonicityCheck.dependence must be one of "
                f"{sorted(_VALID_DEPENDENCE_INPUTS)}, got {self.dependence!r}"
            )
        if self.method != _MONOTONICITY_METHOD:
            raise ValueError(
                f"MonotonicityCheck.method must be {_MONOTONICITY_METHOD!r}, got {self.method!r}"
            )
        if not isinstance(self.any_flagged, bool):
            raise ValueError(
                f"MonotonicityCheck.any_flagged must be a genuine bool, got {self.any_flagged!r}"
            )
        for pair in self.pairs:
            if not isinstance(pair, AdjacentPair):
                raise ValueError(
                    f"MonotonicityCheck.pairs entries must each be an AdjacentPair, got {pair!r}"
                )
        expected_any_flagged = any(pair.flagged for pair in self.pairs)
        if self.any_flagged != expected_any_flagged:
            raise ValueError(
                f"MonotonicityCheck.any_flagged ({self.any_flagged!r}) must equal "
                f"any(pair.flagged for pair in pairs) ({expected_any_flagged!r})"
            )
        for pair in self.pairs:
            expected_flagged = pair.holm_adjusted_p < self.alpha
            if pair.flagged != expected_flagged:
                raise ValueError(
                    f"AdjacentPair.flagged ({pair.flagged!r}) must equal "
                    f"holm_adjusted_p < alpha ({pair.holm_adjusted_p!r} < {self.alpha!r} = "
                    f"{expected_flagged!r})"
                )


def check_monotonicity(
    obs: Observations, *, dependence: str | None = None, alpha: float = 0.05
) -> MonotonicityCheck:
    """Test every pair of adjacent severity levels in ``obs`` for a reversal (``decisions/0023``'s
    "Amendment 1"): a one-sided Fisher exact test for "the higher severity performs better" under
    ``obs.direction``, Holm-corrected across the whole family.

    Levels are built from ``obs.cells()`` -- successes/trials pooled per exact severity, sorted
    ascending, exactly like every other cell-based computation in this package (R1: aggregated
    and per-observation input give identical results). The zero-severity control is a level like
    any other, and the control-to-first-level pair is always in the family, whether or not the
    likelihood a fit built from the same data used the control row (plan section 5.1 drops an
    all-success control under a fixed ``upper=1.0``; this function never reads a fit at all).

    Parameters
    ----------
    obs
        The severity/outcome data to test.
    dependence
        ``None`` (the default) resolves to ``"independent"`` unless ``obs.cluster`` is set, in
        which case omitting it raises (R2: a method assuming independence must refuse clustered
        input unless the caller explicitly opts in). ``"independent"`` may always be passed
        explicitly; the resulting ``MonotonicityCheck.dependence`` is always ``"independent"`` in
        this release.
    alpha
        The significance level ``flagged`` is decided at (``holm_adjusted_p < alpha``), strictly
        between 0 and 1. Defaults to ``0.05``.

    Returns
    -------
    MonotonicityCheck
        ``pairs`` is empty (with a warning naming the reason) when ``obs`` has fewer than two
        distinct severity levels.

    Raises
    ------
    ValueError
        If ``dependence`` is not ``None`` or ``"independent"``; if ``obs.cluster`` is set and
        ``dependence`` is ``None``; or if ``alpha`` is not a finite number strictly between 0
        and 1.
    """
    if dependence is not None and dependence not in _VALID_DEPENDENCE_INPUTS:
        raise ValueError(
            "check_monotonicity: dependence must be None or 'independent' in this release, got "
            f"{dependence!r} (the check assumes independent trials; a paired/clustered design "
            "needs its own method, deferred to a future decision)"
        )
    if obs.cluster is not None and dependence is None:
        raise ValueError(
            "check_monotonicity: obs carries cluster ids (obs.cluster is set); a method that "
            "assumes independence must not run on clustered data unless the caller passes "
            "dependence='independent' explicitly (R2)"
        )
    _check_finite_number(alpha, label="check_monotonicity alpha")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"check_monotonicity: alpha must satisfy 0 < alpha < 1, got {alpha!r}")

    resolved_dependence = "independent"
    direction = obs.direction
    alternative = "greater" if direction == "decreasing" else "less"
    cells = obs.cells()

    if len(cells) < 2:
        return MonotonicityCheck(
            direction=direction,
            alpha=alpha,
            dependence=resolved_dependence,
            pairs=(),
            any_flagged=False,
            warnings=(
                "check_monotonicity: fewer than two distinct severity levels "
                f"({len(cells)}) -- no adjacent pairs to test",
            ),
        )

    level_pairs = list(zip(cells, cells[1:]))  # deliberately not strict: cells[1:] is one shorter
    raw_p_values: list[float] = []
    for low, high in level_pairs:
        failures_low = low.trials - low.successes
        failures_high = high.trials - high.successes
        table = [[high.successes, failures_high], [low.successes, failures_low]]
        _, p_value = fisher_exact(table, alternative=alternative)
        raw_p_values.append(float(p_value))

    _, holm_adjusted, _, _ = multipletests(raw_p_values, method="holm")

    pairs = tuple(
        AdjacentPair(
            severity_low=low.severity,
            severity_high=high.severity,
            successes_low=low.successes,
            trials_low=low.trials,
            successes_high=high.successes,
            trials_high=high.trials,
            p_value=p_value,
            holm_adjusted_p=float(holm_p),
            flagged=bool(holm_p < alpha),
        )
        for (low, high), p_value, holm_p in zip(
            level_pairs, raw_p_values, holm_adjusted, strict=True
        )
    )
    any_flagged = any(pair.flagged for pair in pairs)

    return MonotonicityCheck(
        direction=direction,
        alpha=alpha,
        dependence=resolved_dependence,
        pairs=pairs,
        any_flagged=any_flagged,
        warnings=(),
    )


def _success_and_failure_probabilities(fit: Fit) -> tuple[np.ndarray, np.ndarray]:
    """``(p, 1 - p)`` per cell of ``fit.cells``, from the one shared curve formula in
    ``marginkit.models`` (S1, Phase 7 stats review) -- never a second copy of it here. Imported
    inside this function, not at module scope, since ``marginkit.models`` imports this module at
    *its* top level (see this module's own docstring on the import cycle); by the time any
    ``Fit`` exists to diagnose, ``marginkit.models`` has already finished importing.
    """
    from marginkit.models import _curve_success_and_failure_probability  # local: see docstring

    assert fit.params is not None  # diagnose() only calls this for an OK fit
    mu = fit.params["mu"].value
    s = fit.params["s"].value
    upper = fit.params["upper"].value
    lower = fit.params["lower"].value
    severities = np.array([cell.severity for cell in fit.cells], dtype=np.float64)
    return _curve_success_and_failure_probability(
        severities, mu=mu, s=s, upper=upper, lower=lower, scale=fit.axis.scale, link=fit.link
    )


def _refit_link(fit: Fit, link: str) -> Fit:
    """Refit ``fit``'s data under ``link``, holding the same ``upper``/``lower`` specification
    (both fixed/estimate flags and, when fixed, the same value). Goes through the existing
    private statsmodels fitting paths in ``marginkit.models`` (hard constraint 2: no new
    optimizer or IRLS) -- imported here, inside the function, rather than at module scope, since
    by the time any ``Fit`` exists to diagnose, ``marginkit.models`` has already finished
    importing this module (see this module's own docstring).

    Skips the separation and control-incompatibility pre-checks ``fit_dose_response`` itself
    runs before calling either fitting path: both are properties of the data and the
    ``upper``/``lower`` configuration, not of the link, and ``fit`` already reached
    :data:`~marginkit.Status.OK` under the very same data and configuration, so a different link
    cannot newly trigger either one.
    """
    from marginkit.models import _fit_generic, _fit_glm  # local: see docstring above

    assert fit.params is not None  # diagnose() only calls this for an OK fit
    upper_param = fit.params["upper"]
    lower_param = fit.params["lower"]
    upper_is_estimate = not upper_param.fixed
    lower_is_estimate = not lower_param.fixed
    upper_value = math.nan if upper_is_estimate else upper_param.value
    lower_value = math.nan if lower_is_estimate else lower_param.value

    obs = Observations.from_counts(
        fit.axis,
        severity=[c.severity for c in fit.cells],
        successes=[c.successes for c in fit.cells],
        trials=[c.trials for c in fit.cells],
        outcome=fit.outcome,
        direction=fit.direction,
    )

    if (
        not upper_is_estimate
        and not lower_is_estimate
        and upper_value == 1.0
        and lower_value == 0.0
    ):
        return _fit_glm(obs, cells=fit.cells, cluster_ids=None, link=link)
    return _fit_generic(
        obs,
        cells=fit.cells,
        cluster_ids=None,
        link=link,
        upper_is_estimate=upper_is_estimate,
        upper_value=upper_value,
        lower_is_estimate=lower_is_estimate,
        lower_value=lower_value,
    )


def _link_comparison(fit: Fit) -> tuple[dict[str, float | None], dict[str, Status]]:
    """AIC and status for each of ``_COMPARISON_LINKS``: the diagnosed fit's own link reuses its
    own ``log_likelihood``/parameter count directly (exact, no refit); the other two are refit
    via :func:`_refit_link` (``decisions/0023``: "does two extra fits ... on each call").
    """
    assert fit.params is not None and fit.log_likelihood is not None  # fit.status is OK

    link_aic: dict[str, float | None] = {}
    link_status: dict[str, Status] = {}
    for link in _COMPARISON_LINKS:
        if link == fit.link:
            k = sum(1 for p in fit.params.values() if not p.fixed)
            link_aic[link] = -2.0 * fit.log_likelihood + 2.0 * k
            link_status[link] = Status.OK
            continue
        refit = _refit_link(fit, link)
        if refit.status is Status.OK:
            assert refit.params is not None and refit.log_likelihood is not None
            k = sum(1 for p in refit.params.values() if not p.fixed)
            link_aic[link] = -2.0 * refit.log_likelihood + 2.0 * k
        else:
            link_aic[link] = None
        link_status[link] = refit.status
    return link_aic, link_status


def _is_structural_zero_control(fit: Fit, cell: Cell) -> bool:
    """``decisions/0023`` Amendment 2, item 1: a cell is excluded from goodness of fit **only**
    when it is deterministic *by construction* under the model, never merely because its
    *computed* fitted probability happens to round to a boundary (Amendment 2's own "C1" finding:
    the earlier, value-based rule made ``df`` depend on the link).

    The only case that is structurally deterministic in this release: a log-axis cell at severity
    exactly ``0.0``, under ``direction="decreasing"`` (the only direction ``Fit`` accepts for
    ``model="binomial"``), when ``upper`` is *fixed* at exactly ``1.0`` -- R3's own
    ``P(0) = upper`` rule then pins that cell's fitted probability to exactly ``1.0`` with zero
    variance, for every possible ``mu``/``s``. An *estimated* ``upper`` is never exempt, even when
    its fitted value happens to equal ``1.0``: the control still identifies ``u`` in that case, so
    it is not deterministic *by construction* and is counted
    (``TestControlCountedWhenUpperIsEstimated``).
    """
    if cell.severity != 0.0 or fit.axis.scale != "log" or fit.direction != "decreasing":
        return False
    assert fit.params is not None
    upper_param = fit.params["upper"]
    return upper_param.fixed and upper_param.value == 1.0


def diagnose(fit: Fit) -> Diagnostics:
    """Model-only diagnostics for a converged fit: goodness of fit, dispersion, and link
    comparison (plan section 5.7, ``decisions/0023`` as amended). Never changes ``fit`` itself
    (plan section 5.7: reported, never acted on) -- in particular, ``fit.link`` is never
    consulted to pick a "better" link.

    Parameters
    ----------
    fit
        The fit to diagnose. Must have ``status is Status.OK`` (hard constraint 3: a failed fit
        carries no model-derived numbers -- :func:`~marginkit.fit_dose_response` itself only
        calls this for an ``OK`` fit and leaves :attr:`~marginkit.Fit.diagnostics` ``None``
        otherwise).

    Returns
    -------
    Diagnostics

    Raises
    ------
    ValueError
        If ``fit.status`` is not :data:`~marginkit.Status.OK`.
    """
    if fit.status is not Status.OK:
        raise ValueError(
            f"diagnose: fit.status must be Status.OK (a failed fit carries no model-derived "
            f"diagnostics -- hard constraint 3), got {fit.status!r}"
        )
    assert fit.params is not None and fit.log_likelihood is not None

    k = sum(1 for p in fit.params.values() if not p.fixed)
    p_raw, q_raw = _success_and_failure_probabilities(fit)
    # decisions/0023 Amendment 2, item 2: R's own binomial()$linkinv clamp, applied to each of
    # p and (1 - p) independently -- never derived one from the other's clamped value, which
    # would reintroduce the cancellation _curve_success_and_failure_probability exists to avoid.
    p_clamped = np.clip(p_raw, _PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)
    q_clamped = np.clip(q_raw, _PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)

    deviance_sum = 0.0
    pearson_sum = 0.0
    counted_cells = 0
    sparse_entries: list[tuple[float, float]] = []
    for cell, p, q in zip(fit.cells, p_clamped, q_clamped, strict=True):
        if _is_structural_zero_control(fit, cell):
            # decisions/0023 Amendment 2, item 1: deterministic under the model (P(0) = upper
            # exactly, fixed at 1.0) -- zero variance and zero deviance by construction, not
            # excluded because its *computed* probability happens to round to a boundary.
            continue
        counted_cells += 1
        y = float(cell.successes)
        n = float(cell.trials)
        p_f = float(p)
        q_f = float(q)
        expected_successes = n * p_f
        expected_failures = n * q_f
        deviance_sum += 2.0 * (
            float(xlogy(y, y / expected_successes))
            + float(xlogy(n - y, (n - y) / expected_failures))
        )
        pearson_sum += (y - expected_successes) ** 2 / (n * p_f * q_f)
        # decisions/0023 Amendments 3 and 4: a side is flagged only when *both* its expected
        # count is below the cutoff and something was actually observed there -- a small
        # expectation with nothing observed on that side is a clean plateau, not a finding.
        flagged_side_expecteds: list[float] = []
        if expected_successes < _SPARSE_EXPECTED_CUTOFF and y >= 1.0:
            flagged_side_expecteds.append(expected_successes)
        if expected_failures < _SPARSE_EXPECTED_CUTOFF and (n - y) >= 1.0:
            flagged_side_expecteds.append(expected_failures)
        if flagged_side_expecteds:
            sparse_entries.append((cell.severity, min(flagged_side_expecteds)))

    df = counted_cells - k
    warnings: list[str] = []
    deviance: float | None
    pearson_chi2: float | None
    dispersion: float | None
    if df <= 0:
        deviance, pearson_chi2, dispersion = None, None, None
        warnings.append(
            f"diagnose: degrees of freedom is {df} (<= 0: {counted_cells} cell(s) counted "
            f"minus {k} estimated parameter(s)) -- no goodness-of-fit statistics are reported"
        )
    else:
        deviance = float(deviance_sum)
        pearson_chi2 = float(pearson_sum)
        dispersion = pearson_chi2 / df
        if dispersion > _DISPERSION_WARNING_THRESHOLD:
            warnings.append(
                f"{_OVERDISPERSED_WARNING_PREFIX}dispersion (Pearson chi-square / df) is "
                f"{dispersion!r}, exceeding {_DISPERSION_WARNING_THRESHOLD}; nothing is "
                "rescaled and no interval changes as a result"
            )

    if sparse_entries:
        # decisions/0023 Amendments 2-4: report-only -- names only the flagged severities (an
        # observed event against a below-cutoff expectation on that side), and the single
        # smallest expected count among those *flagged sides only*; changes no number above.
        severities_repr = ", ".join(repr(severity) for severity, _ in sparse_entries)
        minimum_expected = min(count for _, count in sparse_entries)
        warnings.append(
            f"{_SPARSE_CELLS_WARNING_PREFIX}severity/severities {severities_repr} have an "
            f"expected count of successes or failures below "
            f"{_SPARSE_EXPECTED_CUTOFF!r} with at least one observed there (minimum "
            f"{minimum_expected:.6f}); Pearson chi-square and dispersion (phi) are unreliable "
            "there; no number here is changed as a result"
        )

    link_aic, link_status = _link_comparison(fit)

    return Diagnostics(
        deviance=deviance,
        pearson_chi2=pearson_chi2,
        df=df,
        dispersion=dispersion,
        link_aic=link_aic,
        link_status=link_status,
        warnings=tuple(warnings),
    )
