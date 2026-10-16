# Provenance

What robustmargin took from other projects, where from, and what it did not take. Required by
owner decision 0002 (provenance by copy), which names this file as one of three places the
attribution is recorded; the other two are the import commit's message and its `Provenance:`
trailer.

## zeta-bench

| | |
|---|---|
| Source repository | [`badkoubeh/zeta-bench`](https://github.com/badkoubeh/zeta-bench) |
| Source commit | `435fc2793b2afdb8404099c83b0c36a71d9ffba9` (PR #17, 2026-07-16), the squash commit that introduced `robustness/cards.py` |
| Source symbol | `robustness/cards.py::break_point` (sixteen lines including its docstring) |
| Licence | Apache-2.0, the same as robustmargin. zeta-bench ships no `NOTICE` file, so there is no notice text to carry forward beyond this attribution. |
| Imported as | `robustmargin.empirical.grid_break_point`, returning `GridBreakPoint` |
| Imported in | `0.1.0a1`, released under the package's earlier name `marginkit` (IMPLEMENTATION_PLAN §7 Phase 2) |
| Method | Copied and rewritten under neutral names. Not `git filter-repo`: the source file entered zeta-bench history as one squash commit, so there was no lineage worth carrying (decision 0002). |

### What was taken: conventions, not estimation

The source function fits no model and computes no interval. It is a grid rule: the smallest
tested severity whose observed performance falls below a criterion. **robustmargin's break-point
and censoring conventions come from zeta-bench. Its dose-response estimation is new.**

The conventions carried over, against which the copy can be judged:

1. **Strict comparison.** A level fails when its performance is strictly below the criterion.
   Equality passes.
2. **Right-censoring is reported, never hidden.** When no tested level fails, the result has no
   value, carries the largest tested magnitude, and is labelled right-censored, so that "held
   to X" is never read as "unbreakable".
3. **Signed axes break on magnitude.** A signed axis breaks on whichever side fails first, and
   both the break-point and the largest tested value are magnitudes.
4. **Severity 0 is anchored on the nominal cell.** This lives in how zeta-bench builds a curve
   (`degradation_curve`), not in the break-point rule. robustmargin takes the rule and leaves
   curve construction to the caller.

### Deliberate differences from the source

| Input | zeta-bench `break_point` | robustmargin `grid_break_point` |
|---|---|---|
| Empty curve | returns `(None, None)` | raises `ValueError` |
| Mismatched lengths, non-1-D input | not representable (one list of pairs) | raises `ValueError` |
| NaN performance | compares as passing | raises `ValueError` |
| NaN or ±inf severity, NaN or ±inf criterion, ±inf performance | passed through into the comparison | raises `ValueError` |
| Negative severity | always taken as a magnitude | raises `ValueError` unless `signed=True` |
| float32 performance at the criterion | numpy casts a Python-float criterion down to float32, so `np.float32(0.95)` against `0.95` passes | everything is cast up to float64 and compared exactly, so the same input fails (`0.949999988 < 0.95`) |
| Integers above 2^53 | compared exactly | lose precision in the float64 cast |
| Return type | `(break_severity, max_tested)` tuple | frozen `GridBreakPoint(value, max_tested, censoring)`, holding built-in `float`s |

The first two follow robustmargin's rule that a degenerate input raises rather than returning a
plausible-looking result. zeta-bench's wrapper keeps its own empty-curve behaviour, so its
public behaviour on every input it actually produces is unchanged. That is verified by
regenerating its card JSON before and after the swap and requiring an empty diff
(IMPLEMENTATION_PLAN §6.6).

### Test data

`tests/data/zeta_matrix_counts.csv` holds pooled per-level success counts derived from
zeta-bench's `results/robustness_matrix.csv`. Its provenance note is `tests/data/README.md`.

## What this file does not claim

zeta-bench consumes robustmargin's grid rule. It does not use, and has never used, a model-based
threshold or a confidence interval from this package; that would be the follow-up in
IMPLEMENTATION_PLAN §9. Both projects have the same author, so zeta-bench's use demonstrates
that the API is domain-general. It is not third-party adoption.
