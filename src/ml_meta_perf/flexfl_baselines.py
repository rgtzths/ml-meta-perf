"""Baselines for a FlexFL target, scored beside its E3 equation under the same protocols.

* **E3 (fixed form)**: the published fit's own scores. Its terms are chosen once, on every
  row, so its held-out scores are an upper estimate.
* **E3 (form per fold)**: E3's term library, built once from every row's features, with the
  search re-run inside every held-out fold on the training rows' targets only, so no held-out
  outcome influences which terms are used. The length is the published fit's, chosen on every row.
* **mean**: the training fold's mean of the target.
* **GradientBoosting**: scikit-learn's regressor on the E3 fit's features, at the sizes in
  ``config/study.json``'s ``opaque`` block, untuned, with the MCC study's seed.
* **analytic**: ``comm_bytes_total`` only, with no fitted constant. A model-sized message is
  taken to carry ``4 * total_parameters`` bytes; control messages and serialization overhead
  are left out. The master exchanges one message each way per pool worker per round. A
  DecentralizedSync round is one epoch; a CentralizedSync round covers ``min_workers`` of the
  ``pool_batches`` batches in an epoch, so a run needs ``ceil(epochs * pool_batches /
  min_workers)`` rounds; a synchronous run stopped early, below ``epoch_cap``, runs one more
  round before it stops. The asynchronous algorithms exchange one message each way per
  completed task, ``pool_batches`` (Centralized) or ``min_workers`` (Decentralized) tasks per
  epoch, plus the ``min_workers - 1`` tasks still in flight when the run stops. The *oracle*
  row uses the run's own ``n_epochs``, an outcome no other model sees; the *E3* row uses the
  ``n_epochs`` predictions of an E3 fit on the same protocol (fixed form in-sample, form per
  fold held out), on the raw scale and floored at one epoch.

Every model is scored with `ml_meta_perf.validate.score` on the target's scale (``log1p`` for
a log target). In-sample predictions are bounded only by the schema's bounds, as E3's are.
Held-out predictions of the fitted models are clipped to their training fold's range, as E3's
are; the analytic predictions, which fit nothing, are never clipped.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

from ml_meta_perf.config import Configuration, OpaqueConfig, SelectionConfig, load_config
from ml_meta_perf.data import (
    DEFAULT_PATH,
    FLEXFL_EPOCH_CAP_COLUMN,
    Schema,
    columns_as_arrays,
    flexfl_schema,
    groups,
    load,
    target,
)
from ml_meta_perf.experiment import FlexFLReport, run_flexfl
from ml_meta_perf.opaque import SEED
from ml_meta_perf.search import search
from ml_meta_perf.terms import build_library
from ml_meta_perf.validate import leave_one_group_out, library_rows, score

ANALYTIC_TARGET = "comm_bytes_total"
ANALYTIC_COLUMNS: tuple[str, ...] = ("min_workers", "pool_batches", "n_epochs")
BYTES_PER_PARAMETER = 4.0
FL_ALGOS: tuple[str, ...] = ("CentralizedSync", "CentralizedAsync", "DecentralizedSync", "DecentralizedAsync")
IN_SAMPLE = "in_sample"
DEFAULT_SPLITS: dict[str, str] = {"loo_dataset": "dataset", "loo_model": "model"}
ORACLE = "analytic oracle (observed n_epochs)"
PREDICTED = "analytic (E3 n_epochs)"

FitPredict = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BaselineComparison:
    """One row per model and protocol: ``target, model, protocol, r2, mae, n, corpus_sha256``."""

    table: pl.DataFrame
    notes: tuple[str, ...]


def analytic_messages(
    fl_algo: np.ndarray, n_epochs: np.ndarray, epoch_cap: np.ndarray, workers: np.ndarray, pool_batches: np.ndarray
) -> np.ndarray:
    """Model-sized messages the master sends and receives in one run."""
    unknown = sorted(set(fl_algo.tolist()) - set(FL_ALGOS))
    if unknown:
        raise ValueError(f"no message model for fl_algo {unknown}")
    early = (n_epochs < epoch_cap).astype(np.float64)
    centralized_rounds = np.ceil(n_epochs * pool_batches / workers)
    return np.select(
        [
            fl_algo == "CentralizedSync",
            fl_algo == "DecentralizedSync",
            fl_algo == "CentralizedAsync",
            fl_algo == "DecentralizedAsync",
        ],
        [
            2.0 * workers * (centralized_rounds + early),
            2.0 * workers * (n_epochs + early),
            2.0 * n_epochs * pool_batches + 2.0 * (workers - 1.0),
            2.0 * n_epochs * workers + 2.0 * (workers - 1.0),
        ],
    )


def analytic_comm_bytes(
    fl_algo: np.ndarray,
    n_epochs: np.ndarray,
    epoch_cap: np.ndarray,
    workers: np.ndarray,
    pool_batches: np.ndarray,
    total_parameters: np.ndarray,
) -> np.ndarray:
    """``comm_bytes_total`` predicted from the message count and the model size."""
    messages = analytic_messages(fl_algo, n_epochs, epoch_cap, workers, pool_batches)
    return BYTES_PER_PARAMETER * total_parameters * messages


def corpus_digest(path: str | Path) -> str:
    """The first twelve hex digits of the file's SHA-256."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def in_sample(fit_predict: FitPredict, rows: int, bounds: tuple[float, float] | None) -> np.ndarray:
    every = np.ones(rows, dtype=bool)
    predictions = fit_predict(every, every)
    return predictions if bounds is None else np.clip(predictions, *bounds)


def out_of_fold(fit_predict: FitPredict, truth: np.ndarray, labels: np.ndarray, clip: bool = True) -> np.ndarray:
    """Each group's predictions from a fit on the other groups, clipped to that fold's range when ``clip``."""
    predictions = np.empty_like(truth)
    for _, train, test in leave_one_group_out(labels):
        held = fit_predict(train, test)
        predictions[test] = np.clip(held, float(truth[train].min()), float(truth[train].max())) if clip else held
    return predictions


def fold_mean(truth: np.ndarray) -> FitPredict:
    return lambda train, test: np.full(int(test.sum()), float(truth[train].mean()))


def fold_boosting(design: np.ndarray, truth: np.ndarray, opaque: OpaqueConfig) -> FitPredict:
    def fit_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
        model = GradientBoostingRegressor(
            n_estimators=opaque.boosting_stages,
            learning_rate=opaque.boosting_learning_rate,
            max_depth=opaque.boosting_max_depth,
            random_state=SEED,
        )
        model.fit(design[train], truth[train])
        return np.asarray(model.predict(design[test]), dtype=np.float64)

    return fit_predict


def fold_search(
    frame: pl.DataFrame, schema: Schema, truth: np.ndarray, n_terms: int, config: Configuration
) -> FitPredict:
    """E3's term library, built once from every row's features, with its search re-run on the
    training rows' targets only, at most ``n_terms`` long, predicting the test rows. ``n_terms``
    itself is the caller's, usually the all-row fit's length."""
    columns = columns_as_arrays(frame, schema.features)
    library = build_library(
        schema.dataset_features,
        schema.model_features,
        columns,
        max_arity=config.max_arity,
        max_abs_zscore=config.max_abs_zscore,
    )

    def fit_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
        result = search(
            library_rows(library, train),
            truth[train],
            max_terms=n_terms,
            penalty=config.penalty,
            pool_size=config.pool_size,
            beam_width=config.beam_width,
            bounds=schema.bounds,
        )
        equation = result.equations[max(size for size in result.equations if size <= n_terms)]
        return equation.predict({name: values[test] for name, values in columns.items()})

    return fit_predict


def fixed(predictions: np.ndarray) -> FitPredict:
    return lambda _train, test: predictions[test]


def resolve_splits(
    frame: pl.DataFrame, schema: Schema, splits: Mapping[str, np.ndarray] | None
) -> dict[str, np.ndarray]:
    """The default dataset and ``fl_algo`` splits plus ``splits``, which may not redefine them."""
    columns = {"dataset": schema.dataset_column, "model": schema.model_column}
    resolved = {label: groups(frame, columns[kind]) for label, kind in DEFAULT_SPLITS.items()}
    for label, labels in (splits or {}).items():
        if label == IN_SAMPLE:
            raise ValueError(f"{IN_SAMPLE!r} is not a held-out split")
        if label in resolved and not np.array_equal(np.asarray(labels), resolved[label]):
            raise ValueError(f"split {label!r} is reserved for the {DEFAULT_SPLITS[label]} grouping")
        values = np.asarray(labels)
        if values.shape != (frame.height,):
            raise ValueError(f"split {label!r} needs one label per loaded row ({frame.height})")
        if np.unique(values).size < 2:
            raise ValueError(f"split {label!r} needs at least two groups")
        resolved[label] = values
    return resolved


def _analytic_inputs(frame: pl.DataFrame, schema: Schema) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """The analytic model's columns, or why this corpus cannot feed it."""
    missing = [column for column in ANALYTIC_COLUMNS if column not in frame.columns]
    if missing:
        return None, (
            f"analytic baseline skipped: the corpus has no {', '.join(missing)}; "
            "re-assemble it with FlexFL's scripts/assemble_meta_dataset.py"
        )
    needed = ("min_workers", "pool_batches", "n_epochs", "total_parameters", FLEXFL_EPOCH_CAP_COLUMN)
    columns = columns_as_arrays(frame, needed)
    indicators = columns_as_arrays(frame, tuple(f"{schema.model_column}_{name}" for name in FL_ALGOS))
    onehot = np.column_stack([indicators[f"{schema.model_column}_{name}"] for name in FL_ALGOS])
    if not np.all(onehot.sum(axis=1) == 1):
        return None, "analytic baseline skipped: a row does not mark exactly one fl_algo_* column"
    columns["fl_algo"] = np.asarray(FL_ALGOS)[onehot.argmax(axis=1)]
    unrecorded = int(np.sum(np.isnan(columns["min_workers"])))
    if unrecorded:
        return None, f"analytic baseline skipped: {unrecorded} rows have no min_workers"
    unmeasured = int(np.sum(np.isnan(columns["n_epochs"])))
    if unmeasured:
        return None, f"analytic baseline skipped: {unmeasured} rows have no n_epochs"
    centralized = np.isin(columns["fl_algo"], ("CentralizedSync", "CentralizedAsync"))
    unreported = int(np.sum(centralized & np.isnan(columns["pool_batches"])))
    if unreported:
        return None, f"analytic baseline skipped: {unreported} Centralized rows have no pool_batches"
    return columns, None


def analytic_predictions(columns: dict[str, np.ndarray], epochs: np.ndarray, log_target: bool) -> np.ndarray:
    predicted = analytic_comm_bytes(
        columns["fl_algo"],
        epochs,
        columns[FLEXFL_EPOCH_CAP_COLUMN],
        columns["min_workers"],
        columns["pool_batches"],
        columns["total_parameters"],
    )
    return np.log1p(predicted) if log_target else predicted


def epoch_predictions(
    epochs_report: FlexFLReport,
    frame: pl.DataFrame,
    epochs_truth: np.ndarray,
    protocols: Mapping[str, np.ndarray],
    config: Configuration,
) -> dict[str, np.ndarray]:
    """The E3 ``n_epochs`` fit's predictions per protocol, floored at one epoch.

    In-sample uses the fit's own equation; each held-out protocol re-runs its search per fold.
    """
    fitted = epochs_report.schema
    columns = columns_as_arrays(frame, fitted.features)
    per_fold = fold_search(frame, fitted, epochs_truth, epochs_report.equation.n_terms, config)
    predicted = {IN_SAMPLE: np.asarray(epochs_report.equation.equation.predict(columns), dtype=np.float64)} | {
        label: out_of_fold(per_fold, epochs_truth, labels) for label, labels in protocols.items()
    }
    return {label: np.maximum(values, 1.0) for label, values in predicted.items()}


def _scores(model: str, protocol: str, truth: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    scores = score(truth, predicted)
    return {"model": model, "protocol": protocol, "r2": scores.r2, "mae": scores.mae, "n": scores.n}


def run_baselines(
    path: str | Path | None,
    schema: Schema,
    config: Configuration | None = None,
    selection: SelectionConfig | None = None,
    opaque: OpaqueConfig | None = None,
    *,
    report: FlexFLReport | None = None,
    splits: Mapping[str, np.ndarray] | None = None,
) -> BaselineComparison:
    """Score E3 and every baseline that applies to ``schema``'s target on the CSV at ``path``.

    ``report`` is the E3 fit on the same ``path`` and ``schema``, fitted here when not given.
    ``splits`` adds held-out protocols, one group label per loaded row; ``loo_dataset`` and
    ``loo_model`` are the dataset and ``fl_algo`` groupings E3 reports, and may not be redefined.
    E3 (fixed form) has rows only for the protocols E3 reports itself.
    """
    resolved = Path(path) if path is not None else DEFAULT_PATH
    study = load_config()
    config, opaque = config or study.search, opaque or study.opaque
    report = report or run_flexfl(resolved, schema, config, selection)
    frame = load(resolved, schema, extra_columns=ANALYTIC_COLUMNS)
    truth = target(frame, schema)
    if truth.shape[0] != report.equation.in_sample["n"]:
        raise ValueError(
            f"the E3 fit scored {report.equation.in_sample['n']} rows, the baselines load {truth.shape[0]}"
        )
    protocols = resolve_splits(frame, schema, splits)
    fitted = report.schema
    features = columns_as_arrays(frame, fitted.features)
    design = np.column_stack([features[name] for name in fitted.features])
    e3 = report.equation

    rows: list[dict[str, object]] = [
        {"model": "E3 (fixed form)", "protocol": IN_SAMPLE, **{k: e3.in_sample[k] for k in ("r2", "mae", "n")}}
    ]
    rows += [
        {"model": "E3 (fixed form)", "protocol": label, **{k: e3.cross_validated[label][k] for k in ("r2", "mae", "n")}}
        for label in protocols
        if label in e3.cross_validated
    ]
    per_fold = fold_search(frame, fitted, truth, e3.n_terms, config)
    rows += [
        _scores("E3 (form per fold)", label, truth, out_of_fold(per_fold, truth, labels))
        for label, labels in protocols.items()
    ]
    learned: dict[str, FitPredict] = {
        "mean": fold_mean(truth),
        f"GradientBoosting ({opaque.boosting_stages} stages)": fold_boosting(design, truth, opaque),
    }
    for name, fit_predict in learned.items():
        rows.append(_scores(name, IN_SAMPLE, truth, in_sample(fit_predict, truth.shape[0], schema.bounds)))
        rows += [
            _scores(name, label, truth, out_of_fold(fit_predict, truth, labels)) for label, labels in protocols.items()
        ]

    notes: list[str] = []
    if schema.target_column == ANALYTIC_TARGET:
        columns, reason = _analytic_inputs(frame, schema)
        if columns is None:
            notes.append(str(reason))
        else:
            oracle = analytic_predictions(columns, columns["n_epochs"], schema.log_target)
            rows += [_scores(ORACLE, label, truth, oracle) for label in (IN_SAMPLE, *protocols)]
            epochs_schema = flexfl_schema("n_epochs", schema.task_type)
            epochs_report = run_flexfl(resolved, epochs_schema, config, selection)
            epochs = epoch_predictions(epochs_report, frame, target(frame, epochs_schema), protocols, config)
            rows += [
                _scores(PREDICTED, label, truth, analytic_predictions(columns, predicted, schema.log_target))
                for label, predicted in epochs.items()
            ]

    digest = corpus_digest(resolved)
    table = pl.DataFrame(
        [{"target": schema.label, **row, "corpus_sha256": digest} for row in rows],
        schema={
            "target": pl.String,
            "model": pl.String,
            "protocol": pl.String,
            "r2": pl.Float64,
            "mae": pl.Float64,
            "n": pl.Int64,
            "corpus_sha256": pl.String,
        },
    )
    return BaselineComparison(table, tuple(notes))
