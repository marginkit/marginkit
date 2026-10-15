# Changelog

All notable changes to marginkit are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html). While the version is below 1.0 a
minor bump may contain breaking changes; `docs/decisions/0003-consumer-version-policy.md` is
what consumers pin against.

Every entry that adds, renames, removes, or changes the behaviour of a public symbol must also
satisfy the public-API change checklist in `docs/IMPLEMENTATION_PLAN.md` Appendix B.

## [Unreleased]

### Changed
- Warning and exception messages no longer cite internal design documents (decision records,
  plan sections, decision labels). Only the prose changed: the leading tokens `NON_MONOTONE: `,
  `NON_MONOTONE_DATA: `, `OVERDISPERSED: ` and `SPARSE_CELLS: ` are byte-identical, and every
  exception type and raise condition is unchanged except as noted under Fixed. Match warnings on
  the token (`w.startswith("NON_MONOTONE: ")`), not on the text after it.
- The `NON_MONOTONE: ` warning now says precisely which bounds are not reported: the exact
  one-sided bounds. On a converged fit any `lo`/`hi` shown come from the fitted curve; on a failed
  fit no bounds are reported.

### Fixed
- `fit_dose_response` on a log axis with `upper=1.0` fixed, an estimated `lower`, and a
  zero-severity control: the control, whose likelihood contribution is exactly zero, no longer
  enters the optimizer or its start values, so the fit is judged exactly as for the same data
  without the control. Estimates can differ within optimizer tolerance, and a fit at the edge of
  convergence can change `status` between `OK` and `NOT_CONVERGED`. The reported `log_likelihood`
  still covers every cell. Data without such a control is unaffected.
- `fit_dose_response` on a log axis whose only severity is the zero control now raises a
  `ValueError` saying a dose-response curve needs at least one severity level besides the control.
  It previously raised an unexplained numpy error, or, when an asymptote was estimated, returned
  `NOT_CONVERGED`.

## [0.1.1] -- unreleased

`0.1.1rc1` is a TestPyPI-only dry run of the release pipeline (build, Trusted Publishing upload,
install check); pre-release tags never reach PyPI. It carries the same content as 0.1.1.

Packaging only, prepared as the first PyPI release (`decisions/0024`), to be tagged `v0.1.1` once publishing to PyPI is possible.
No code, public API, schema or numeric change from 0.1.0; upgrading needs nothing.

### Changed
- The README's links to `docs/PROVENANCE.md`, `CHANGELOG.md` and `LICENSE` are absolute, so they
  work on the PyPI project page, where relative links break. They point at the `marginkit`
  GitHub organization, which now owns the repository.
- Not yet on PyPI: the `marginkit` name there is held by another account, and a PEP 541 request
  is pending (`decisions/0024` Amendment 1). Until it is granted, install from the git tag; the
  README switches to `pip install marginkit` in the release that first reaches PyPI.
- `[project.urls]` adds the source repository and the issue tracker, next to the existing
  homepage and changelog links.
- Releases are built and published by `.github/workflows/release.yml` through PyPI Trusted
  Publishing: TestPyPI first, an install check, then PyPI after the owner's approval.
  `tools/check_dist.py` holds the artifact checks that CI and the release both run.

## [0.1.0] — 2026-09-29

Phase 7, tagged `v0.1.0` at `6da49ab`. It completes v0.1: binomial dose-response fits, thresholds
with profile or delta intervals, censoring, ratios with Fieller or log-delta intervals, and now fit
diagnostics. Published as a git tag only; 0.1.1 is the first PyPI release.

**This release is breaking.** Upgrade every reader before any writer.

### Phase 7 -- diagnostics

#### Added
- `Diagnostics` and `diagnose(fit)`: deviance and Pearson χ² on grouped cells with their degrees of
  freedom, dispersion `φ = Pearson χ² / df`, and the AIC of each of `probit`, `logit` and `cloglog`
  refitted under the same `upper`/`lower` specification, with each refit's `Status`. Computed for
  successful fits only (`decisions/0023`). They are reported and never acted on: no link is
  selected and nothing is rescaled.
  - Warnings: `OVERDISPERSED: ` when `φ > 1.5`; `SPARSE_CELLS: ` when a counted cell has an observed
    success or failure where fewer than 0.1 were expected, because Pearson χ² and φ are unreliable
    there.
  - Degrees of freedom exclude only a cell that is deterministic under the model: the log-axis
    severity-0 control when `upper` is fixed at 1. Every other cell counts, with fitted
    probabilities clamped as R's `glm` does (`decisions/0023` Amendment 2).
- `MonotonicityCheck`, `AdjacentPair` and `check_monotonicity(obs, *, dependence=None,
  alpha=0.05)`: a one-sided Fisher exact test on each pair of adjacent severity levels with a Holm
  correction across pairs, computed from the counts alone. The direction comes from `obs`
  (`decisions/0023` Amendment 1).
- `Fit.diagnostics` (set only when `status is OK`) and `Fit.monotonicity` (set on every result from
  `fit_dose_response`, whatever the status), appended fields.
- `fit_dose_response(..., dependence=None)`, a keyword-only argument.
- `marginkit.testing.fake_fit` fills `monotonicity` and, for `OK`, `diagnostics` through the real
  code, so a fake looks like a real fit. New keywords `diagnostics=` and `monotonicity=` override.
- `schema/scorecard-v3.json`.

#### Changed -- breaking
- **`schema_version` is `"3"` on every result type.** Every card written by 0.1.0 is rejected by a
  0.1.0a4 or older reader. 0.1.0 reads v1 and v2 cards, decodes `diagnostics` and `monotonicity` as
  `None`, and writes an old card back under its original tag without the new fields.
  `load_schema()` now returns v3; pass `"2"` or `"1"` for the older documents.
- **`fit_dose_response` raises `ValueError` on clustered observations unless
  `dependence="independent"` is passed.** *Migration:* add `dependence="independent"` to the call
  and keep passing it to `threshold()`. That records an independence assumption; clustered
  inference is planned for v0.2. Callers without cluster ids change nothing.

#### Changed -- behavioural (same signatures, no number changes)
- `threshold()` adds a warning starting `NON_MONOTONE_DATA: ` whenever `fit.monotonicity` flags a
  pair, on every status. **It is a different test from `NON_MONOTONE: `** (exact bounds against the
  fitted curve), which is unchanged; the two can disagree. *Migration:* match the full prefix,
  including the colon and space. `startswith("NON_MONOTONE")` now matches both.
- `ratio_interval()` puts both input thresholds' `warnings` (`t_a`'s, then `t_b`'s) at the front of
  `Ratio.warnings`. In 0.1.0a4 it carried none of them. *Migration:* an empty `Ratio.warnings` no
  longer means "nothing to note" in 0.1.0a4's sense; read `ratio.threshold_a.warnings` and
  `ratio.threshold_b.warnings` to tell which input a warning came from.
- `fit_dose_response` runs two extra fits per call, for the link comparison.

#### Unchanged
- Every fitted parameter, threshold, interval, status and censoring value. Checked on the zeta
  matrix: 48 fit configurations and 36 ratios, no numeric difference.
- `grid_break_point`, and the `NON_MONOTONE: ` token.

#### Open items
- Can clustered fitting land in v0.2 without either a cluster-aware monotonicity test or relaxing
  "`monotonicity` is never `None` on a fresh fit"? `MonotonicityCheck.dependence` accepts only
  `"independent"`, and widening it is a schema bump.
- On the estimated-asymptote path, adding an all-success control row can change convergence.

## [0.1.0a4] — 2026-09-27

Phase 6, tagged `v0.1.0a4` at `f785f58`.

**This release is breaking, unlike 0.1.0a3.** `IntervalShape` gains a member and `schema_version`
goes to `"2"` on every result type. Upgrading code needs no change to call existing functions, but
**a reader must be upgraded before a writer**: every card written by 0.1.0a4 carries
`schema_version="2"` and is rejected by a 0.1.0a2/0.1.0a3 reader, including a card whose contents
are byte-identical to a v1 one. In the other direction 0.1.0a4 reads v1 cards unchanged (Appendix
B item 4). `decisions/0003` is what consumers pin against; a breaking change inside an unreleased
`0.1.0` pre-release series is what the alpha series exists for -- see **Open items**.

### Phase 6 -- ratio intervals

#### Added
- `ratio_interval(t_a, t_b, /, *, method, dependence, level=None) -> Ratio`. The ratio of two
  thresholds' severities with a Fieller interval on the natural severity scale
  (`method="fieller"`) or a log-scale delta interval (`method="log_delta"`, which additionally
  requires both axes `scale="log"` and both values `> 0`). `t_a`/`t_b` are positional-only.
  `dependence` must be `"independent"`; `"paired"` raises, naming it v0.2 Phase 9 work. Both
  methods use the normal quantile, never Student-t -- R `drc`'s `EDcomp(interval="fieller")` uses
  `qt`, which is why its interval endpoints, but not its point estimates, differ from marginkit's
  (`tests/reference/test_ratio_drc_edcomp.py`).
- `level=None`, the default, inherits the confidence level from the two input thresholds, which
  must agree; an explicit `level` that disagrees with either raises (`decisions/0016`). There is
  still no defaulted confidence level anywhere in the public API.
- Under `dependence="independent"`, `ratio_interval` raises when the two thresholds come from the
  very same `Fit` object (identity, not value equality: two disjoint samples may legitimately
  produce an equal `Fit` by value). Without it the ratio of a threshold with itself was reported
  as `1.0 [0.888, 1.126]`, when it is identically 1 with zero variance -- wrong, not merely wide
  (`decisions/0018` amendment). `Ratio` itself is **unchanged** and still constructs from two
  thresholds sharing a `Fit`, exactly as in `0.1.0a2`.
- `marginkit.testing.fake_ratio` can build every `IntervalShape` member, including `HALF_OPEN` and
  the now-unreachable `EXCLUSIVE`.
- `report.load_schema(version="2")` takes an optional version; `load_schema("1")` reads the schema
  a pre-`0022` card was written under. `schema/scorecard-v2.json` is packaged alongside
  `scorecard-v1.json`, which is unchanged.
- Decisions `0015` (a censored or failed input threshold propagates a direction-aware
  status/censoring flag and no numbers), `0016` (the level is inherited, not defaulted), `0017`
  (the `A == 0` half-line, now partly superseded by `0022`), `0018` and its amendment (the
  independence guard stays cluster-ids-only, plus the shared-`Fit` identity check), `0019` (a
  Fieller root below zero is intersected with the positive parameter space), `0020` (a `log_delta`
  interval too wide to exponentiate raises), `0021` (both roots negative is `UNBOUNDED`, narrowed
  by `0022`) and `0022` (`HALF_OPEN` and the schema bump).

#### Changed -- breaking
- **`IntervalShape` gains `HALF_OPEN`**: `lo` set, `hi` is `None`, meaning `[lo, inf)`. **Code that
  branches exhaustively on `shape` must handle it.** It occupies an **open region** of the parameter
  space -- reached whenever the denominator threshold is poorly determined (`z^2*Vb > theta_b^2`) and
  the numerator is not -- rather than the measure-zero boundary `decisions/0017` took it for. How
  often it occurs is a property of a design's power, not a fixed rate: `decisions/0022` records the
  `p(1-p)` relationship and why an earlier "~24%" figure was withdrawn as a justification. It
  replaces two previously-reported shapes: `A < 0` with `K > 0` reported
  `EXCLUSIVE` with an impossible negative `lo`, and `A == 0` with `K > 0` reported `UNBOUNDED`,
  discarding a real finite bound. `decisions/0017` had justified the latter on the case being
  "measure-zero", which is true of `g == 1` exactly but not of the open region beside it; that
  argument is withdrawn in the record.
- **A set bounded above but not below stays `BOUNDED`**, reported as `[0, hi]`, because `0` is a
  true lower bound for a severity ratio (`decisions/0019`). `HALF_OPEN` is therefore always
  lower-bounded and never upper-bounded; the asymmetry is deliberate.
- **`schema_version` is `"2"` on `GridBreakPoint`, `Fit`, `Threshold`, `Ratio` and `Scorecard`, and
  `report.SCHEMA_VERSION` is `"2"`.** One schema document covers all five types, so one version
  applies to all five; a `Fit` card tagged `"2"` may be byte-identical to a `"1"` one and is still
  tagged `"2"`, because a reader of that card must understand v2 anyway -- a v2 `Ratio` can sit
  beside it in the same `Scorecard`. **Migration: upgrade every reader to 0.1.0a4 before any writer
  starts emitting.** `from_dict` accepts both `"1"` and `"2"` at every nesting level, preserves
  whichever the card carried, and round-trips a v1 card unchanged.
  - `GridBreakPoint` is the one bumped type with a live consumer. zeta-bench is unaffected in fact:
    `robustness/cards.py` unpacks `grid_break_point` to a plain `(value, max_tested)` tuple and
    never serialises the dataclass, so its regenerated card diff stays empty (verified against
    `origin/main` = `975456a`). Any consumer that serialises a `GridBreakPoint` directly does see
    `"2"` where it saw `"1"`.
- **`report.load_schema()` with no argument now returns the v2 document** (`$id`
  `urn:marginkit:schema:scorecard-v2`), not v1. v2 is a strict superset of v1 -- the only
  substantive difference is the widened `IntervalShape` enum, and `schema_version` is unconstrained
  in both -- so a consumer validating **stored v1 cards** with `load_schema()` is unaffected. A
  consumer asserting on `$id` or `title` is not. Pass `load_schema("1")` for the old document.
- `Ratio`'s `method="log_delta"` invariant is loosened from `shape=BOUNDED` to
  `BOUNDED | HALF_OPEN`. Safe for code that constructs a `Ratio`; a card relying on it is readable
  only by a marginkit at least as new as the writer.

#### Unchanged
- `Status` and `Censoring`, every result field name, every signature of an existing function, and
  `schema/scorecard-v1.json`, which stays packaged for the reader path and is untouched *by this
  release*. It is not a frozen historical artifact, though: Phase 5 added `Threshold.baseline` and
  `.grid` to it while `schema_version` was still `"1"` (`REVIEWS.md` R9), so a consumer pinned to
  `0.1.0a2` cannot validate a `0.1.0a3`-written card against the v1 schema it shipped with.
- `IntervalShape.EXCLUSIVE` keeps its member and its `Ratio` invariants, but is **unreachable
  through `ratio_interval` in v0.1**: it needs same-signed roots with `A < 0`, hence `B <= 0`, hence
  a nonzero covariance term, which only `dependence="paired"` supplies in v0.2. Verified both
  algebraically and over 200,000 draws (0 occurrences). It is retained rather than removed because
  removing an enum member would need a deprecation shim, and because keeping it is what makes v2 a
  strict superset of v1.

#### Limitations
- **`ratio_interval` does not verify independence.** Under `dependence="independent"` it raises on
  input fits that share a `cluster_id`, and on two thresholds sharing one `Fit` object, and on
  nothing else. Nothing in the package records an observation id, so two fits built from
  overlapping rows that carry no cluster ids are accepted and produce a ratio whose stated coverage
  is wrong. The check is necessary, not sufficient; establishing independence is the caller's
  responsibility (`decisions/0018`, `REVIEWS.md` R6 #14, deferred to v0.2).
- A consumer cannot distinguish `(0, inf)` from a half-line by `shape` alone in the `A == 0`,
  `K <= 0` sub-case, which is still `UNBOUNDED` with the reason in `warnings` (`decisions/0017`,
  `0021`).

#### Open items
- `CONSUMERS.md`'s change table says a breaking change takes a "minor bump while in 0.x", which
  would read as `0.2.0`. `0.1.0` has never been released -- only the `a1` and `a2` pre-releases are
  tagged -- so the whole `0.1.0aN` series is one unreleased minor, and a break between `a3` and
  `a4` is what a pre-release series is for. Recorded as an open amendment to `decisions/0003`,
  alongside the PEP 440 pre-release range question already open against it (`REVIEWS.md` R5 #5).
- `EXCLUSIVE` is dead surface until `dependence="paired"` lands. Should Phase 9 be the point at
  which its reachability is re-asserted by a test, rather than left to be noticed?

## [0.1.0a3] — 2026-09-27

Phases 4 and 5, tagged `v0.1.0a3` at Phase 5's merge commit `34b5a69`, which carries
`__version__ = "0.1.0a3"`. Additive; upgrading from 0.1.0a2 needs no consumer change. The two
phases share one version because `v0.1.0a3` was never tagged for Phase 4 on its own -- Phase 4
merged with `__version__` still at `0.1.0a2`, so retro-tagging that commit would ship a package
reporting the wrong version.

### Phase 4 -- dose-response fitting

### Added
- `fit_dose_response(obs, *, model, link, upper, lower) -> Fit`. All four keywords are required,
  with no defaults: `model="binomial"`; `link` one of `"probit"`, `"logit"`, `"cloglog"`;
  `upper`/`lower` a float or `"estimate"`. Asymptotes fixed at 1 and 0 fit with statsmodels
  `GLM`; anything else with `GenericLikelihoodModel` (`decisions/0004`). A failed fit returns a
  `Fit` whose `status` says so (`SEPARATION`, `NOT_CONVERGED`, `CONTROL_INCOMPATIBLE`), with
  `params`, `covariance` and `log_likelihood` set to `None` and the reason in `warnings`.
  `direction="increasing"` raises `ValueError`. `Fit.cluster_ids` is copied from `obs`.
- `Fit.predict(severity, *, level=0.95) -> Prediction`, with pointwise delta-method bands.
  Raises `ValueError` unless `status` is `OK`.
- `Prediction`: frozen, keyword-only, compared by identity (`eq=False`); read-only numpy array
  fields `severity`, `estimate`, `lo`, `hi` and a float `level`. Not serialisable:
  `report.to_dict` and `Scorecard` reject it.
- R `drc`/`MASS` reference fixtures and their generator in `tools/drc_reference/`.
- Decisions `0005` (no interior maximum → `NOT_CONVERGED`), `0006` (separation from the exact
  overlap check, amending plan §5.2) and `0007` (observed-information covariance).

### Changed
- `Covariance` positive-definiteness is now checked on the correlation matrix: every variance
  `> 0` and minimum correlation eigenvalue `> 1e-12`, replacing minimum eigenvalue
  `> 1e-12·max(1, max|entry|)`. A fit's status no longer depends on severity units. This only
  loosens the rule: every matrix 0.1.0a2 accepted is still accepted. Stored cards: a `Fit`
  written by this version may be rejected by a 0.1.0a2 reader; read cards with a marginkit at
  least as new as the writer.

### Unchanged
- JSON schema v1 and every existing field and signature.

### Notice (now in effect, as of Phase 5)
- Exporting `threshold()` has made the attribute `marginkit.threshold` the function, not the
  module. `import marginkit.threshold as m; m.Threshold` no longer works. Use
  `from marginkit import Threshold`; `from marginkit.threshold import Threshold` still resolves
  through `sys.modules` but is not API.

### Phase 5 -- thresholds, intervals, censoring

#### Added
- `threshold(fit, *, definition, interval_method, level, dependence=None) -> Threshold`. Solves
  for the severity at which the fitted curve crosses a target performance, for all three
  definitions (`absolute`, `baseline_fraction`, `relative`); `definition` has no default,
  pending D5. `interval_method` is `"profile"` (primary) or `"delta"`, and is a *request*: on
  any censored or failed path the result records `"exact_bound"`. Omitting `dependence` on
  clustered input raises (R2).
- `Threshold.baseline: BaselineRate | None` and `Threshold.grid: GridBreakPoint | None`, both
  defaulting to `None`. They carry what plan section 5.5 says a `FAILS_AT_BASELINE` and a
  `SEPARATION`/`NOT_CONVERGED` result must report, which previously had nowhere to live
  (`decisions/0010`). Appended fields: code upgrading from 0.1.0a2 is unaffected, but a stored
  card carrying them must be read by a marginkit at least as new as the writer.
- `BaselineRate`: the control cell's rate with its exact interval, labelled
  `per_cell_clopper_pearson`. Serialisable, as a `Threshold` field.
- `per_cell_clopper_pearson(cells, *, level=0.95) -> ExactRates`. `ExactRates` is deliberately
  **not** serialisable and has no schema entry, which is what enforces plan section 5.4's rule
  that per-cell exact rates never attach to a threshold.
- Internal modules `marginkit.censoring` (section 5.5's rules and the exact one-sided bounds)
  and `marginkit.intervals` (profile and delta). Neither import path is API.
- Decisions `0008` (`baseline_fraction` reachability, the fallback `u`, and `FAILS_AT_BASELINE`
  before `UNREACHABLE`), `0009` (each side of a two-sided exact bracket at `(1 + level)/2`),
  `0010` (the two appended fields), `0011` (exact bounds contradicting monotonicity report no
  bounds), `0012` (section 5.5's `RIGHT`/`LEFT` conditions outrank a converged fit), `0013` (an
  interval unbounded on both sides reports the estimate with no bounds) and `0014` (a crossing at
  or below zero severity on a linear axis is `UNREACHABLE`).

#### Changed
- **`Threshold`'s `(status=OK, censoring=NONE)` invariant is loosened.** `value` is still
  required, but `lo` and `hi` may now both be `None` (they must be both set or both `None`). A
  converged fit on a design too coarse to constrain the parameter has a profile that crosses the
  chi-square target on neither side, and v1 has no `Censoring` member for "open on both sides";
  the estimate is reported, the bounds are not, and `warnings` names the search limits that were
  reached (`decisions/0013`). Safe for code that *constructs* a `Threshold`. **Code that reads
  `Threshold.lo`/`.hi` must now handle `None` on the `OK` path, not only on the censored ones.**
- `threshold()` returns `Status.UNREACHABLE` with `censoring=NONE` and no numbers when the
  closed-form solve lands at or below zero severity on a linear axis, rather than raising from
  `Threshold`'s non-negativity rule. `UNREACHABLE` now has two causes and `warnings` tells them
  apart (`decisions/0014`). Only linear axes can reach it.
- `Threshold`'s docstring no longer claims a `SEPARATION`/`NOT_CONVERGED` bracket has joint
  coverage `2 * level - 1`. Each side is now computed at `(1 + level)/2`, so joint coverage is
  at least `level` and `Threshold.level` means the same thing on every path (`decisions/0009`).
  Behavioural for brackets, which are wider than the superseded wording implied; one-sided
  `RIGHT`/`LEFT` bounds are unchanged at `level`.
- The default `pytest` run excludes the `slow` marker, and a separate CI job runs plan section
  6.3's coverage simulations once rather than on each of four matrix legs.

## [0.1.0a2] — 2026-09-14

The v1 contract: vocabulary, result objects, the JSON score-card schema, and test fakes, published
before any estimation exists so that consumers can build against them. To be tagged `v0.1.0a2`
once the owner has approved the public API surface and this release's PR is merged. Not uploaded
to PyPI: D3 is still open.

### Added

- Vocabulary in `marginkit`:
  - `Axis(name, unit, scale, citation)`, with `scale` either `"log"` or `"linear"`.
  - `Observations.from_counts(...)` and `Observations.from_observations(..., cluster=...)`.
    `outcome` and `direction` are required. Both forms pool into the same `cells()`.
  - `Definition.absolute(a)`, `.baseline_fraction(p)` and `.relative(p)`.
  - `Cell`, plus the `Status` and `IntervalShape` enums. Their values equal their member names,
    as in `Censoring`.
- Result objects `Parameter`, `Covariance`, `Fit`, `Threshold` and `Ratio`. These are frozen,
  keyword-only dataclasses that check their invariants when constructed:
  - A failed fit carries no parameters, covariance or log-likelihood.
  - A threshold follows the censoring rules: a censored or failed threshold never carries a
    point estimate, and its status must agree with its fit's status.
  - A ratio requires the same axis, unit and definition on both sides. With
    `dependence="independent"`, it refuses thresholds whose fits share cluster ids.
- Dependence is recorded in results: `Fit.cluster_ids` holds the fit's cluster ids, and
  `Threshold.dependence` is required. A stored result therefore shows when independence was
  assumed.
- `marginkit.types.JSONValue`, the type of the values allowed in `provenance`.
- `Scorecard`, and `marginkit.report` with `SCHEMA_VERSION`, `to_dict`, `from_dict` and
  `load_schema`. The schema is `schema/scorecard-v1.json` (JSON Schema draft 2020-12), shipped in
  the wheel. `GridBreakPoint` serialises under the same schema and may appear in a `Scorecard`.
- `marginkit.testing.fake_fit`, `fake_threshold` and `fake_ratio`: schema-valid results for
  consumers' own tests. Each is labelled as a fake and is not an estimate.

### Notes

- **No estimation yet.** Nothing in this release produces a `Fit`, `Threshold` or `Ratio` except
  `marginkit.testing`. `fit_dose_response`, `threshold` and `ratio_interval` arrive in later
  releases.
- **Schema v1 is binomial-only and strict.** `from_dict` rejects unknown fields rather than
  dropping them, so reading a stored card needs a marginkit at least as new as the one that wrote
  it. A new result family or enum value (for example continuous fits or paired ratios) will mean
  schema version `"2"`.
- **Binomial results accept only `direction="decreasing"`**, until owner decision D5 defines the
  increasing case. `Observations` accepts both directions.
- **A failed or censored `Ratio` carries no shape, estimate or bounds**, until one-sided ratio
  bounds are specified.
- `grid_break_point`, the `GridBreakPoint` fields and `Censoring` are unchanged.

## [0.1.0a1] — 2026-09-13

Tagged `v0.1.0a1` on GitHub at `cdd0410`. Not uploaded to PyPI: package naming and publication
(D3) is still an open owner decision.

### Added

- `marginkit.grid_break_point(severity, performance, *, criterion, signed=False)`, returning a
  frozen `GridBreakPoint(value, max_tested, censoring)`: the smallest tested severity magnitude
  whose performance is strictly below `criterion`, or `value=None` with `Censoring.RIGHT` when
  no tested level fails. It reports an observed grid statistic, not an estimate of a threshold.
  The rule's conventions are copied from zeta-bench's `robustness/cards.py::break_point`
  (`badkoubeh/zeta-bench@435fc27`); see `docs/PROVENANCE.md`.
- `marginkit.Censoring`, with all five members specified for the v0.1 contract:
  `NONE`, `RIGHT`, `LEFT`, `OPEN_UPPER`, `OPEN_LOWER`. `grid_break_point` uses only `NONE` and
  `RIGHT`. The rest are declared now so that the model-based thresholds in later phases do not
  add enum members a consumer must handle.
- `tests/data/zeta_matrix_counts.csv`: pooled per-level counts from zeta-bench's committed
  matrix, covering the edge cases the package must handle, with a provenance note.
- `docs/PROVENANCE.md`, committed (the rest of `docs/` stays private).

### Differences from the zeta-bench source

- Empty input, mismatched lengths, non-finite values, and negative severities without
  `signed=True` raise `ValueError`. zeta-bench returned `(None, None)` for an empty curve and
  treated a NaN rate as passing.

### Notes

- First release with public API. `GridBreakPoint` also carries `schema_version` (default `"1"`)
  and an opaque `provenance` mapping (default empty), which marginkit stores and never
  interprets. Set it with `dataclasses.replace`. JSON serialisation arrives with `report.py`
  in Phase 3.
- `GridBreakPoint.censoring` is a point-estimate label for the grid: `RIGHT` means no tested
  level fell below the criterion. It carries no exact one-sided test and no monotonicity
  assumption, so it is not the model-based classification that later thresholds will report.
- Version ranges: decision 0003 specifies `marginkit>=0.X,<0.(X+1)`. Under PEP 440 the
  pre-release `0.1.0a1` sorts below `0.1`, so `>=0.1,<0.2` excludes it. An owner amendment to
  0003 is pending. Until marginkit is on PyPI, the only consumer depends on the git tag, so
  nothing is affected yet.

## [0.1.0.dev0] — Phase 1 bootstrap, untagged

### Added

- Packaging and tooling skeleton: setuptools with an `src/` layout, an SPDX `license` field,
  `py.typed`, `requires-python >= 3.11`, and the runtime dependency set `numpy`, `scipy`,
  `statsmodels`.
- `ruff` (line length 100), `mypy --strict` over `src/`, and `pytest` with a 90% branch-coverage
  gate, all wired into pre-commit and GitHub Actions.
- A CI matrix covering Python 3.11, 3.12 and 3.13 against current dependencies, plus a
  dependency-floor job on Python 3.11 (`ci/constraints-min.txt`), so the declared floors are
  tested at the floor rather than merely declared.
- A packaging job that asserts `py.typed` is present inside the built wheel. A missing marker
  fails silently and disables type checking for every consumer.
- A test asserting that importing `marginkit` pulls in no GPU, ML-framework, or plotting module.
- Owner decision records 0001 through 0004, covering parallel milestone scheduling, the
  provenance method, the consumer version policy, and models with non-trivial asymptotes.

### Notes

- No public API yet. The contract is specified in `docs/IMPLEMENTATION_PLAN.md` sections 4.1 and
  4.2 and is frozen in Phase 3, under owner review.
