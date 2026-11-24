# FlexFL targets

Both commands also fit FlexFL's per-run meta-dataset, one row per federated learning (FL) run,
assembled by FlexFL's `scripts/assemble_meta_dataset.py`. The abbreviations used below are
coefficient of determination (**R²**), mean absolute error (**MAE**), leave-one-dataset-out
(**LODO**), leave-one-model-out (**LOMO**) and doubly held out (**DHO**).

## Targets

Use `--target performance`, `--target total_time_s`, `--target comm_bytes_total`,
`--target n_epochs`, `--target compute_time_total_s`, `--target compute_time_max_s`,
`--target comm_time_total_s`, or `--target validation_time_s` with a FlexFL CSV supplied
by `--data`. Performance also requires
`--task-type classification` or `--task-type regression`. A task type on any other target
filters the input rows as well. `n_epochs` is the last epoch a run logged before it
stopped, at most the run's global epoch cap, the `epoch_cap` column.
`comm_bytes_sent` and `comm_bytes_recv` are not targets:
each is half of `comm_bytes_total` to within 1%, so a fit on either repeats the
`comm_bytes_total` fit.

FlexFL's assembler produces the decomposition columns, clipped to the master's first `start`
and last `end`. `compute_time_total_s` sums worker work time, while
`compute_time_max_s` is the work time of the busiest worker. `comm_time_total_s` sums
worker communication time. Its timings pair a master and a worker clock, so positive
clock skew is not detectable. `validation_time_s` sums validation time and mostly follows
validation-set size and the number of validations. The assembler leaves the decomposition
columns empty for a run with no worker logs or no work inside that window. Loading one of
these four targets drops the rows whose target is empty, after any task-type filter, and
prints the number dropped to stderr, with counts by dataset and by `fl_algo`; it fails if
dropping removes every row. An empty feature, or an empty value in any other selected
target, still fails the load. Constant-feature removal runs after the drop, so for these
targets it sees only the kept rows.

A corpus that mixes early-stopping rules is rejected: fit the runs of one rule at a time.

## Log scale

`--log-target` with `total_time_s`, `comm_bytes_total`, `compute_time_total_s`,
`compute_time_max_s`, `comm_time_total_s`, or `validation_time_s` fits `log1p` of the target,
so R² and MAE are on the log scale. It is rejected for `mcc`, `performance` and `n_epochs`.
Equations are labelled `log1p(<target>)` and named `E3_log1p_k<n>`.

## What is fitted

A FlexFL target runs E3 only, over every FlexFL dataset and model feature, under the
configuration in `config/study.json` and the study's length rule. There is no E1, E2 or E3-MAX,
and none of the MCC study's baselines, practices, figures or chapters; `--baselines` adds the
FlexFL baselines below. Features constant in the loaded frame are removed before the term
library is built.

Unbounded targets, the cost targets and `n_epochs`, prune terms whose contribution spans
less than 0.002 times the target's 1st-to-99th percentile spread divided by 2, while
bounded targets keep 0.002.

`epoch_cap` is a FlexFL model feature: the global epoch cap each run trained under. Loading a
FlexFL CSV without the column fails with a stale-corpus error; re-assemble it with FlexFL's
`scripts/assemble_meta_dataset.py`. In a corpus where every run shares one cap it is constant
and removed before the term library is built.

LOMO holds out one `fl_algo` value at a time, while the four `fl_algo_*` indicators are
model features. In each held-out fold the held-out algorithm's indicator is zero on every
training row, so its terms get no weight there. FlexFL LOMO R² therefore measures transfer
to an unseen algorithm without per-algorithm offsets, and it is not comparable with the MCC
study's LOMO, whose model features are descriptors shared across models.

## Reselection and feature evidence

Each fit writes `term_frequency.csv` and `feature_evidence.csv` beside the equation. The term
frequency table has one row per equation term, with its feature group, weight, coefficient,
10th-to-90th percentile swing (`effect`), direction, number of dataset exclusions that
reselected it (`folds`), number of datasets searched, and `frequency = folds / searches`. A term
never reselected has frequency zero.

The term `effect` is the 10th-to-90th percentile swing in that term's contribution. The feature
evidence table reports each feature's signed upper-minus-lower decile difference,
Spearman correlation, mean term frequency, the configured margin, and each filter result. A
feature passes the margin, direction, and frequency filters when the absolute decile difference
is at least the target's margin, the absolute Spearman correlation is at least 0.15, and its
mean term frequency is at least 0.5. It passes the sign filter when both values are nonzero and
their signs match. `passes_all` is true only when all four filters pass. Equality passes each
numeric filter, matching the MCC study's `>=` rule.

The upper tail includes rows where the feature is at or above its 90th percentile, and the
lower tail includes rows at or below its 10th percentile. Tied values are all included. Features
that share a term each receive that term's whole contribution. Reselection runs at the pre-prune
equation length.

Margins are in the fitted target scale and come from the `effects.decile_margin` section of
`config/study.json`. Performance has a separate margin for classification and regression. Cost
targets use the raw target key or the `-log1p` key according to the fitted scale. A target with
no configured margin needs an explicit `--effects` configuration.

## Baselines

`--baselines` scores these beside the E3 fit, in-sample and under LODO and LOMO, and writes
`baseline_comparison.csv` next to the equation: one row per model and protocol with
`target`, `model`, `protocol`, `r2`, `mae`, `n` and `corpus_sha256`, the first twelve hex
digits of the CSV's SHA-256. It also writes `baseline_notes.txt`, empty unless a baseline was
skipped. Every model is scored on the fitted scale, `log1p` with `--log-target`. In-sample
predictions are bounded only by the target's bounds, as E3's are. Held-out predictions of
every fitted model are clipped to the training fold's range, as E3's are.

- `E3 (fixed form)`: the E3 fit's own scores in-sample and under LODO and LOMO; under any
  other protocol its published terms refit per fold, as [Extrapolation](#extrapolation)
  describes. Its terms are chosen once, on every row, so its held-out scores are an upper
  estimate of transfer.
- `E3 (form per fold)`: LODO and LOMO with E3's term library, built once from every row's
  features, and its search re-run on each training fold's targets, so no held-out outcome
  influences which terms are chosen. The length is the E3 fit's, which was chosen on every row.
- `mean`: the training fold's mean of the target. For `performance` the task type already
  restricts the rows.
- `GradientBoosting (<n> stages)`: scikit-learn's regressor on the E3 fit's features, at the
  `opaque` sizes in `config/study.json`, untuned.
- `analytic oracle (observed n_epochs)` and `analytic (E3 n_epochs)`: `comm_bytes_total` only,
  with no fitted constant and never clipped. A model-sized message carries its pickled size, the
  `{"type", "data"}` dictionary FlexFL's Raw message layer pickles at protocol 4 with the weights
  as one float32 array: `4 * total_parameters` bytes plus a pickle header, 171 to 183 bytes for
  the models of 73 to 96,278 parameters in the tier-10 corpus. Control messages are left out. A
  synchronous run exchanges one message each way per pool worker per round. A DecentralizedSync
  round is one epoch. A CentralizedSync round covers `min_workers` of the `pool_batches` batches
  of an epoch, so a run takes `ceil(epochs * pool_batches / min_workers)` rounds. A synchronous
  run that stopped early, below `epoch_cap`, runs one more round. An asynchronous run exchanges
  one message each way per completed task, `pool_batches` (Centralized) or `min_workers`
  (Decentralized) tasks per epoch, plus the `min_workers - 1` tasks still in flight when it
  stops. The oracle row uses each run's own `n_epochs`, an outcome no other model sees. The E3
  row uses an E3 `n_epochs` fit, on the raw scale: its equation in-sample and its search re-run
  per fold held out, floored at one epoch and capped at the run's `epoch_cap`. Both are skipped,
  with a note on stderr and in `baseline_notes.txt`, when the corpus lacks `min_workers`,
  `pool_batches` or `n_epochs`, when a row does not mark exactly one `fl_algo_*` column, when a
  row has no `min_workers` or no `n_epochs`, or when a Centralized row has no `pool_batches`.
  Runs made before the sweep recorded `min_workers` have neither.

## Extrapolation

`--extrapolation` does what `--baselines` does and adds six protocols to the same table. Each is
scored on its held-out rows only, so its `n` is the number of held-out runs.

- `extrapolate_workers`: one fold, fitted on the runs with at most 34 workers (`num_workers`)
  and scored on the runs with 36 to 56. `extrapolate_workers_unclipped` is the same fold with
  the fitted models' predictions not clipped to the training range, since that clip caps the
  cost a larger cluster can reach; for `performance` they stay within the target's bounds, as
  E3's equation keeps its own. The analytic rows are never clipped, so they skip the
  unclipped twin, which would repeat them, and appear under every other protocol.
- `loo_distribution`: three folds, each fitted on two of iid, non_iid and dirichlet (the
  `strategy` column) and scored on the third, pooled into one score. `holdout_iid`,
  `holdout_non_iid` and `holdout_dirichlet` score each fold alone. Runs with any other
  strategy are in no fold.

Under these protocols, which E3 does not report itself, `E3 (fixed form)` keeps the published
equation's terms and refits their weights on each training fold. The published equation is
the pruned one, while E3's own LODO and LOMO scores refit the form before pruning. The term
library and the equation length still come from every row, as for `E3 (form per fold)`. The
published terms were chosen with the held-out runs' targets in view, the 36 to 56 worker runs
included, so these `E3 (fixed form)` scores are an upper estimate too; `E3 (form per fold)` is
the row whose terms no held-out target chose. A protocol whose fold has an empty side, or a
corpus missing one of the three strategies, is skipped with a note on stderr and in
`baseline_notes.txt`.

## Practice diagnostics

`ml-meta-perf-practices` tests four FL practices against a FlexFL corpus and the per-target tables of an
earlier run (`feature_evidence.csv`, `term_frequency.csv` and `equation.json` under each `<slug>/`). It fits
nothing. Every threshold is in the `flexfl_practices` section of `config/study.json`, and the run records them.

- P1, asynchronous algorithms shorten wall time but need more epochs or bytes. Async minus sync, paired on
  dataset, distribution and worker combination within the Centralized and the Decentralized family.
- P2, more workers raise communication volume and time faster than they cut compute time. Elasticity here is the OLS slope of `log1p(y)` on
  `log(num_workers)`, a log1p-scale slope and not a log-log elasticity, of
  `comm_bytes_total`, `comm_time_total_s` and `compute_time_max_s` within each dataset,
  algorithm and distribution, pooled and per algorithm.
- P3, non-IID and Dirichlet splits need more epochs and reach lower performance than IID. Paired on dataset,
  algorithm and worker combination; MCC is higher-better, mape lower-better.
- P4, parameter count drives communication bytes and time more than dataset size. Spearman correlations over
  the per-dataset medians; capped by `size_max_outcome`, since each dataset has one network. The equation comparison contrasts decile extremes of dataset-constant features, so each side is
  one dataset, and the two features' extremes are different datasets.

A practice is `not tested` when a deciding comparison has fewer matched units than `min_pairs`, the
corpus lacks a column it needs, or no relevant equation contains its feature (or that target's table is
missing). It is `met` when its observed criterion holds and every required feature passes all four
`feature_evidence.csv` filters with the expected sign and sits in at least `flexfl_practices.min_stable_terms` term(s) reselected in at least
`effects.min_frequency` of the exclusions (P4 also needs a larger |decile difference| for `total_parameters`
than for `n_samples`), and `qualified` otherwise, including a contradicted practice. A not-tested reason also
names any observed criterion that failed. Spearman comparisons need at least `flexfl_practices.min_datasets` datasets to be computable.
The run stops when a table's `baseline_comparison.csv` names another corpus, and records tables without one.

`practices.csv` has one row per practice with the criterion, the observed and equation evidence as text,
`observed_holds`, `equation_holds` (empty when not computable), the outcome and its reason, the practice's
thresholds, the corpus digest and the commit. `practices_detail.csv` has one row per number, with its family, algorithm, split,
task or target, unit, comparison direction, threshold, whether it holds and decides, and the source file
(`corpus`, or the target's `feature_evidence.csv` or `term_frequency.csv`). `decides` marks the rows the
outcome rests on; per-algorithm and descriptive rows are reported only. On equation rows, `decile_difference`
holds when its sign is the expected one and `supported` when the feature also passes all four filters in a
stable term; a feature shared by several parts appears once, with the parts in `note`. Time and byte
differences are on the log1p scale; mape is a fraction.

## Commands

```bash
# one E3 fit; writes equation.json, equation.txt, curve.csv, term_effects.csv,
# group_shares.csv, term_frequency.csv and feature_evidence.csv under results/flexfl/<slug>/
# term_effects.csv reports each term's block span and weighted-contribution effect range
# group_shares.csv has five rows of covariance shares
ml-meta-perf --target total_time_s --log-target --data meta_dataset.csv

# the same fit plus the baselines; adds baseline_comparison.csv
ml-meta-perf --target comm_bytes_total --log-target --data meta_dataset.csv --baselines

# the baselines plus the worker-count and distribution protocols, in the same files
ml-meta-perf --target comm_bytes_total --log-target --data meta_dataset.csv --extrapolation

# the configuration sweep on the same target
ml-meta-perf-search --target total_time_s --log-target --data meta_dataset.csv --output results/sweep_time

# the practice diagnostics over the tables of earlier fits; writes practices.csv (one row per
# practice) and practices_detail.csv (every number with its source) under results/flexfl/
ml-meta-perf-practices --data meta_dataset.csv --tables results/flexfl
```

`term_effects.csv` has a `group` column (`dataset`, `architecture`, `fl_config`, `federation` or `mixed`),
and `blocks` lists the blocks a term spans. Its `effect` is the 10th-to-90th percentile range of the weighted
contribution. `group_shares.csv` has five rows, whose covariance shares sum to 1 when the equation output varies.

The sweep fits the same grid as for MCC. FlexFL targets have no go/no-go threshold and no
model ranking to score, so the DHO task-metric columns of `candidates.csv` are empty, and
among the configurations tied on R² at a readable length the proposal goes to fewer terms,
then lower arity, then the heavier penalty.
