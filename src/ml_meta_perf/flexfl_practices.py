"""FlexFL practice diagnostics: four literature-derived FL practices against the corpus and the equations.

Each practice pairs an observed comparison on the corpus with the equation evidence of an earlier
FlexFL run (`feature_evidence.csv`, `term_frequency.csv` and `equation.json` per target), read
rather than refitted. The outcome is ``not tested`` when a deciding comparison has too few matched
units, the corpus lacks a needed column, or no relevant equation contains the feature (or its table
is missing); ``met`` when the observed criterion holds and every required feature passes all four
evidence filters with the expected sign and sits in at least one stable term; and ``qualified``
otherwise, including a contradicted practice. Every threshold comes from the ``flexfl_practices``
section of ``config/study.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from jsonargparse import ActionConfigFile, ArgumentParser  # pyright: ignore[reportPrivateImportUsage]

from ml_meta_perf.config import (
    DEFAULT_CONFIG_PATH,
    MET,
    NOT_TESTED,
    QUALIFIED,
    Configuration,
    EffectsConfig,
    OpaqueConfig,
    PracticesConfig,
    SelectionConfig,
    SweepConfig,
)
from ml_meta_perf.data import FLEXFL_BLOCKS
from ml_meta_perf.flexfl_baselines import DISTRIBUTIONS, FL_ALGOS, corpus_digest
from ml_meta_perf.model import Equation
from ml_meta_perf.stats import spearman

PRESENT = "present"
ABSENT = "absent from the equation"
MISSING = "table missing"

FAMILIES = {
    "Centralized": ("CentralizedSync", "CentralizedAsync"),
    "Decentralized": ("DecentralizedSync", "DecentralizedAsync"),
}
ALGORITHMS = FL_ALGOS
SKEWED = tuple(name for name in DISTRIBUTIONS if name != "iid")
UNITS = {
    "total_time_s-log1p": "log1p(s)",
    "comm_bytes_total-log1p": "log1p(bytes)",
    "comm_time_total_s-log1p": "log1p(s)",
    "compute_time_max_s-log1p": "log1p(s)",
    "n_epochs": "epochs",
    "performance-classification": "MCC",
    "performance-regression": "mape (fraction)",
}
SLUGS = tuple(UNITS)
COLUMNS = (
    "dataset",
    "fl_algo",
    "strategy",
    "combo",
    "num_workers",
    "main_metric",
    "performance",
    "total_time_s",
    "n_epochs",
    "comm_bytes_total",
    "comm_time_total_s",
    "compute_time_max_s",
    "total_parameters",
    "n_samples",
)

SUMMARY_SCHEMA = {
    "practice": pl.String,
    "statement": pl.String,
    "criterion": pl.String,
    "observed": pl.String,
    "equation_evidence": pl.String,
    "confounds": pl.String,
    "observed_holds": pl.Boolean,
    "equation_holds": pl.Boolean,
    "outcome": pl.String,
    "reason": pl.String,
    "thresholds": pl.String,
    "tables_corpus_check": pl.String,
    "corpus_sha256": pl.String,
    "commit": pl.String,
}
DETAIL_SCHEMA = {
    "practice": pl.String,
    "part": pl.String,
    "role": pl.String,
    "family": pl.String,
    "algorithm": pl.String,
    "split": pl.String,
    "task": pl.String,
    "target": pl.String,
    "feature": pl.String,
    "term": pl.String,
    "quantity": pl.String,
    "unit": pl.String,
    "value": pl.Float64,
    "n": pl.Int64,
    "direction": pl.String,
    "threshold": pl.Float64,
    "holds": pl.Boolean,
    "decides": pl.Boolean,
    "source": pl.String,
    "note": pl.String,
}


@dataclass(frozen=True)
class Check:
    quantity: str
    unit: str
    value: float
    n: int
    direction: str
    threshold: float
    holds: bool | None
    decides: bool
    tested: bool
    scope: dict[str, str] = field(default_factory=dict)
    dropped: int = 0
    part: str = "observed"
    source: str = "corpus"


@dataclass(frozen=True)
class Stability:
    min_frequency: float
    min_terms: int


@dataclass(frozen=True)
class Support:
    slug: str
    feature: str
    expected: int
    role: str
    state: str
    row: dict[str, float]
    terms: tuple[tuple[str, str, float, float], ...]
    min_frequency: float
    min_terms: int

    @property
    def present(self) -> bool:
        return self.state == PRESENT

    @property
    def stable_terms(self) -> int:
        return sum(frequency >= self.min_frequency for _, _, _, frequency in self.terms)

    @property
    def supported(self) -> bool:
        return (
            self.present
            and bool(self.row["passes_all"])
            and float(np.sign(self.row["decile_difference"])) == self.expected
            and self.stable_terms >= self.min_terms
        )


@dataclass(frozen=True)
class Part:
    name: str
    alternatives: tuple[Support, ...]

    @property
    def available(self) -> bool:
        return any(item.present for item in self.alternatives)

    @property
    def supported(self) -> bool:
        return any(item.supported for item in self.alternatives)

    def describe(self) -> str:
        states = {item.state for item in self.alternatives}
        cause = MISSING if states == {MISSING} else ABSENT
        return f"{self.name}: {' / '.join(item.feature for item in self.alternatives)} {cause}"


@dataclass(frozen=True)
class Evidence:
    features: pl.DataFrame
    terms: pl.DataFrame
    term_features: dict[str, tuple[str, ...]]
    corpus_sha256: str | None


@dataclass(frozen=True)
class Outcome:
    practice: str
    statement: str
    criterion: str
    confounds: str
    thresholds: dict[str, object]
    checks: tuple[Check, ...]
    parts: tuple[Part, ...]
    context: tuple[Support, ...]
    observed_holds: bool | None
    equation_holds: bool | None
    outcome: str
    reason: str


@dataclass(frozen=True)
class Assessment:
    checks: list[Check]
    parts: list[Part]
    observed: bool
    failures: list[str]
    context: list[Support] = field(default_factory=list)
    equation_extra: bool = True
    gate: str | None = None


def read_evidence(tables: Path, slug: str) -> Evidence | None:
    folder = tables / slug
    paths = [folder / name for name in ("feature_evidence.csv", "term_frequency.csv", "equation.json")]
    if not all(path.is_file() for path in paths):
        return None
    equation = Equation.load(paths[2])
    baseline = folder / "baseline_comparison.csv"
    digest = None
    if baseline.is_file():
        digests = pl.read_csv(baseline, schema_overrides={"corpus_sha256": pl.String})["corpus_sha256"].unique()
        digest = ",".join(sorted(str(value) for value in digests.to_list()))
    return Evidence(
        features=pl.read_csv(paths[0]),
        terms=pl.read_csv(paths[1]),
        term_features={term.name: term.features for term in equation.terms},
        corpus_sha256=digest,
    )


def support(
    evidence: dict[str, Evidence | None], slug: str, feature: str, expected: int, stability: Stability, role: str
) -> Support:
    found = evidence.get(slug)
    if found is None:
        return Support(slug, feature, expected, role, MISSING, {}, (), stability.min_frequency, stability.min_terms)
    names = [name for name, features in found.term_features.items() if feature in features]
    spans = tuple(
        (str(row["term"]), str(row["blocks"]), float(row["effect"]), float(row["frequency"]))
        for row in found.terms.filter(pl.col("term").is_in(names)).iter_rows(named=True)
    )
    rows = found.features.filter(pl.col("feature") == feature)
    if rows.height == 0:
        return Support(slug, feature, expected, role, ABSENT, {}, spans, stability.min_frequency, stability.min_terms)
    row = rows.row(0, named=True)
    numbers = {
        name: float(row[name])
        for name in (
            "decile_difference",
            "spearman",
            "mean_frequency",
            "margin",
            "n_terms",
            "pass_margin",
            "pass_direction",
            "pass_sign",
            "pass_frequency",
            "passes_all",
        )
    }
    return Support(slug, feature, expected, role, PRESENT, numbers, spans, stability.min_frequency, stability.min_terms)


def holds(value: float, direction: str, threshold: float) -> bool:
    match direction:
        case ">=":
            return value >= threshold
        case "<=":
            return value <= threshold
        case ">":
            return value > threshold
        case "<":
            return value < threshold
        case "|x| <":
            return abs(value) < threshold
        case _:
            raise ValueError(f"unknown direction {direction!r}")


def check(
    quantity: str,
    unit: str,
    value: float,
    n: int,
    direction: str,
    threshold: float,
    minimum: int,
    *,
    decides: bool = True,
    dropped: int = 0,
    part: str = "observed",
    source: str = "corpus",
    **scope: str,
) -> Check:
    tested = n >= (minimum if decides else 1) and bool(np.isfinite(value))
    verdict = holds(value, direction, threshold) if tested else None
    return Check(
        quantity,
        unit,
        value if tested else float("nan"),
        n,
        direction,
        threshold,
        verdict,
        decides,
        tested,
        scope,
        dropped,
        part,
        source,
    )


def describe(quantity: str, unit: str, value: float, n: int, **scope: str) -> Check:
    return Check(quantity, unit, value, n, "", float("nan"), None, False, True, scope)


def median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else float("nan")


def paired(
    frame: pl.DataFrame, key: list[str], column: str, base: str, other: str, value: pl.Expr
) -> tuple[np.ndarray, np.ndarray, int]:
    sides = []
    for level, name in ((base, "base"), (other, "other")):
        side = frame.filter(pl.col(column) == level).select(*key, value.alias(name))
        if side.select(key).is_duplicated().any():
            raise ValueError(f"{column}={level} has more than one row per {', '.join(key)}")
        sides.append(side)
    joined = sides[0].join(sides[1], on=key, how="inner").drop_nulls()
    first = joined["base"].to_numpy().astype(float)
    second = joined["other"].to_numpy().astype(float)
    finite = np.isfinite(first) & np.isfinite(second)
    return first[finite], second[finite], int((~finite).sum())


def difference(
    frame: pl.DataFrame, key: list[str], column: str, base: str, other: str, value: pl.Expr
) -> tuple[np.ndarray, int]:
    first, second, dropped = paired(frame, key, column, base, other, value)
    return second - first, dropped


def share(values: np.ndarray, *, negative: bool) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values < 0 if negative else values > 0))


def lacking(frame: pl.DataFrame, needed: tuple[str, ...]) -> list[str]:
    return [column for column in needed if column not in frame.columns]


def async_vs_sync(
    frame: pl.DataFrame, evidence: dict[str, Evidence | None], config: PracticesConfig, stability: Stability
) -> Assessment:
    key = ["dataset", "strategy", "combo"]
    checks: list[Check] = []
    parts: list[Part] = []
    failures: list[str] = []
    observed_all = True
    for family, (sync, asynchronous) in FAMILIES.items():
        subset = frame.filter(pl.col("fl_algo").is_in([sync, asynchronous]))
        time, time_dropped = difference(subset, key, "fl_algo", sync, asynchronous, pl.col("total_time_s").log1p())
        epochs, epochs_dropped = difference(
            subset, key, "fl_algo", sync, asynchronous, pl.col("n_epochs").cast(pl.Float64)
        )
        sent, sent_dropped = difference(
            subset, key, "fl_algo", sync, asynchronous, pl.col("comm_bytes_total").log1p()
        )
        slow, fast, _ = paired(subset, key, "fl_algo", sync, asynchronous, pl.col("total_time_s").cast(pl.Float64))
        group = [
            check(
                "median async-sync difference, total_time_s",
                "log1p(s)",
                median(time),
                time.size,
                "<=",
                config.async_time_diff,
                config.min_pairs,
                dropped=time_dropped,
                family=family,
            ),
            check(
                "median async-sync difference, n_epochs",
                "epochs",
                median(epochs),
                epochs.size,
                ">=",
                config.async_epoch_diff,
                config.min_pairs,
                dropped=epochs_dropped,
                family=family,
            ),
            check(
                "median async-sync difference, comm_bytes_total",
                "log1p(bytes)",
                median(sent),
                sent.size,
                ">=",
                config.async_bytes_diff,
                config.min_pairs,
                dropped=sent_dropped,
                family=family,
            ),
        ]
        checks += group
        checks += [
            describe("median async/sync ratio, total_time_s", "ratio", median(fast / slow), fast.size, family=family),
            describe(
                "share of pairs with shorter async wall time",
                "share",
                share(time, negative=True),
                time.size,
                family=family,
            ),
            describe(
                "share of pairs with more async epochs",
                "share",
                share(epochs, negative=False),
                epochs.size,
                family=family,
            ),
            describe(
                "share of pairs with more async bytes", "share", share(sent, negative=False), sent.size, family=family
            ),
        ]
        observed = bool(group[0].holds) and (bool(group[1].holds) or bool(group[2].holds))
        if not observed:
            failures.append(
                f"{family} observed: wall time {group[0].value:.3f}, epochs {group[1].value:.1f}, "
                f"bytes {group[2].value:.3f} against <= {config.async_time_diff} and (>= {config.async_epoch_diff} "
                f"or >= {config.async_bytes_diff})"
            )
        observed_all &= observed
        on, off = f"fl_algo_{asynchronous}", f"fl_algo_{sync}"
        parts += [
            Part(
                f"{family} wall time",
                (
                    support(evidence, "total_time_s-log1p", on, -1, stability, "alternative"),
                    support(evidence, "total_time_s-log1p", off, 1, stability, "alternative"),
                ),
            ),
            Part(
                f"{family} epochs or bytes",
                (
                    support(evidence, "n_epochs", on, 1, stability, "alternative"),
                    support(evidence, "n_epochs", off, -1, stability, "alternative"),
                    support(evidence, "comm_bytes_total-log1p", on, 1, stability, "alternative"),
                    support(evidence, "comm_bytes_total-log1p", off, -1, stability, "alternative"),
                ),
            ),
        ]
    return Assessment(checks, parts, observed_all, failures)


def elasticity(workers: np.ndarray, values: np.ndarray) -> float:
    if np.unique(workers).size < 2:
        return float("nan")
    return float(np.polyfit(np.log(workers), np.log1p(values), 1)[0])


def cell_elasticities(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, algorithm, strategy), cell in frame.group_by(["dataset", "fl_algo", "strategy"], maintain_order=True):
        workers = cell["num_workers"].to_numpy().astype(float)
        rows.append(
            {
                "dataset": dataset,
                "fl_algo": algorithm,
                "strategy": strategy,
                "bytes": elasticity(workers, cell["comm_bytes_total"].to_numpy().astype(float)),
                "comm_time": elasticity(workers, cell["comm_time_total_s"].to_numpy().astype(float)),
                "compute_max": elasticity(workers, cell["compute_time_max_s"].to_numpy().astype(float)),
            }
        )
    schema = {"dataset": pl.String, "fl_algo": pl.String, "strategy": pl.String}
    schema |= {name: pl.Float64 for name in ("bytes", "comm_time", "compute_max")}
    return pl.DataFrame(rows, schema=schema).drop_nans()


def workers_vs_communication(
    frame: pl.DataFrame, evidence: dict[str, Evidence | None], config: PracticesConfig, stability: Stability
) -> Assessment:
    cells = cell_elasticities(frame)
    checks: list[Check] = []
    gap = config.workers_elasticity_gap
    for algorithm in ("all", *ALGORITHMS):
        decides = algorithm == "all"
        subset = cells if decides else cells.filter(pl.col("fl_algo") == algorithm)
        n = subset.height
        sent, timed, compute = (median(subset[name].to_numpy()) for name in ("bytes", "comm_time", "compute_max"))
        checks += [
            check(
                "median elasticity, comm_bytes_total",
                "elasticity",
                sent,
                n,
                ">",
                0.0,
                config.min_pairs,
                decides=decides,
                algorithm=algorithm,
            ),
            check(
                "median elasticity, comm_time_total_s",
                "elasticity",
                timed,
                n,
                ">",
                0.0,
                config.min_pairs,
                decides=decides,
                algorithm=algorithm,
            ),
            check(
                "median elasticity, compute_time_max_s",
                "elasticity",
                compute,
                n,
                "<",
                0.0,
                config.min_pairs,
                decides=decides,
                algorithm=algorithm,
            ),
            check(
                "median elasticity comm_bytes_total minus |median elasticity compute_time_max_s|",
                "elasticity",
                sent - abs(compute),
                n,
                ">=",
                gap,
                config.min_pairs,
                decides=decides,
                algorithm=algorithm,
            ),
            check(
                "median elasticity comm_time_total_s minus |median elasticity compute_time_max_s|",
                "elasticity",
                timed - abs(compute),
                n,
                ">=",
                gap,
                config.min_pairs,
                decides=decides,
                algorithm=algorithm,
            ),
        ]
    pooled = [item for item in checks if item.decides]
    failures = [
        f"pooled {item.quantity} {item.value:.4f} not {item.direction} {item.threshold}"
        for item in pooled
        if item.holds is False
    ]
    parts = [
        Part(
            "num_workers raises comm_bytes_total",
            (support(evidence, "comm_bytes_total-log1p", "num_workers", 1, stability, "required"),),
        ),
        Part(
            "num_workers raises comm_time_total_s",
            (support(evidence, "comm_time_total_s-log1p", "num_workers", 1, stability, "required"),),
        ),
        Part(
            "num_workers lowers compute_time_max_s",
            (support(evidence, "compute_time_max_s-log1p", "num_workers", -1, stability, "required"),),
        ),
    ]
    return Assessment(checks, parts, all(bool(item.holds) for item in pooled), failures)


def non_iid_vs_iid(
    frame: pl.DataFrame, evidence: dict[str, Evidence | None], config: PracticesConfig, stability: Stability
) -> Assessment:
    key = ["dataset", "fl_algo", "combo"]
    checks: list[Check] = []
    parts: list[Part] = []
    failures: list[str] = []
    observed_all = True
    for split in SKEWED:
        for algorithm in ("all", *ALGORITHMS):
            decides = algorithm == "all"
            subset = frame if decides else frame.filter(pl.col("fl_algo") == algorithm)
            epochs, epochs_dropped = difference(
                subset, key, "strategy", "iid", split, pl.col("n_epochs").cast(pl.Float64)
            )
            mcc, mcc_dropped = difference(
                subset.filter(pl.col("main_metric") == "mcc"), key, "strategy", "iid", split, pl.col("performance")
            )
            mape, mape_dropped = difference(
                subset.filter(pl.col("main_metric") == "mape"), key, "strategy", "iid", split, pl.col("performance")
            )
            scope = {"split": split, "algorithm": algorithm}
            group = [
                check(
                    f"median {split}-iid difference, n_epochs",
                    "epochs",
                    median(epochs),
                    epochs.size,
                    ">=",
                    config.non_iid_epoch_diff,
                    config.min_pairs,
                    decides=decides,
                    dropped=epochs_dropped,
                    task="all",
                    **scope,
                ),
                check(
                    f"median {split}-iid difference, MCC",
                    "MCC",
                    median(mcc),
                    mcc.size,
                    "<=",
                    config.non_iid_mcc_diff,
                    config.min_pairs,
                    decides=decides,
                    dropped=mcc_dropped,
                    task="classification",
                    **scope,
                ),
                check(
                    f"median {split}-iid difference, mape",
                    "mape (fraction)",
                    median(mape),
                    mape.size,
                    ">=",
                    config.non_iid_mape_diff,
                    config.min_pairs,
                    decides=decides,
                    dropped=mape_dropped,
                    task="regression",
                    **scope,
                ),
            ]
            checks += group
            if decides and not all(bool(item.holds) for item in group):
                observed_all = False
                failures += [
                    f"{item.quantity} {item.value:.4f} not {item.direction} {item.threshold}"
                    for item in group
                    if item.holds is False
                ]
        for slug, sign, verb in (
            ("n_epochs", 1, "raises n_epochs"),
            ("performance-classification", -1, "lowers MCC"),
            ("performance-regression", 1, "raises mape"),
        ):
            parts.append(
                Part(
                    f"{split} {verb}",
                    (
                        support(evidence, slug, f"strategy_{split}", sign, stability, "alternative"),
                        support(evidence, slug, "strategy_iid", -sign, stability, "alternative"),
                    ),
                )
            )
    return Assessment(checks, parts, observed_all, failures)


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.unique(first).size < 2 or np.unique(second).size < 2:
        return float("nan")
    return spearman(first, second)


def model_vs_dataset_size(
    frame: pl.DataFrame, evidence: dict[str, Evidence | None], config: PracticesConfig, stability: Stability
) -> Assessment:
    varying = (
        frame.group_by("dataset")
        .agg(pl.col("total_parameters").n_unique().alias("p"), pl.col("n_samples").n_unique().alias("s"))
        .filter((pl.col("p") > 1) | (pl.col("s") > 1))
    )
    if varying.height:
        raise ValueError(f"total_parameters or n_samples varies within dataset {varying['dataset'].to_list()}")
    per_dataset = frame.group_by("dataset").agg(
        pl.col("comm_bytes_total").log1p().median().alias("comm_bytes_total"),
        pl.col("comm_time_total_s").log1p().median().alias("comm_time_total_s"),
        pl.col("total_parameters").first().cast(pl.Float64),
        pl.col("n_samples").first().cast(pl.Float64),
    )
    parameters = per_dataset["total_parameters"].to_numpy()
    samples = per_dataset["n_samples"].to_numpy()
    datasets = per_dataset.height
    together = rank_correlation(parameters, samples) if datasets >= config.min_datasets else float("nan")
    gate = check(
        "Spearman(total_parameters, n_samples) across datasets",
        "Spearman",
        together,
        datasets,
        "|x| <",
        config.size_collinear_rho,
        config.min_datasets,
        target="datasets",
    )
    checks = [gate]
    observed = True
    failures: list[str] = []
    for name in ("comm_bytes_total", "comm_time_total_s"):
        values = per_dataset[name].to_numpy()
        pair = [
            check(
                "Spearman(per-dataset median log1p, total_parameters)",
                "Spearman",
                rank_correlation(values, parameters),
                datasets,
                ">=",
                config.size_rho,
                config.min_datasets,
                target=name,
            ),
            check(
                "Spearman(per-dataset median log1p, n_samples)",
                "Spearman",
                rank_correlation(values, samples),
                datasets,
                "|x| <",
                config.size_rho,
                config.min_datasets,
                target=name,
            ),
        ]
        checks += pair
        if not all(bool(item.holds) for item in pair):
            observed = False
            failures.append(
                f"{name} observed: Spearman with total_parameters {pair[0].value:.3f} (>= {config.size_rho} needed), "
                f"with n_samples {pair[1].value:.3f} (|x| < {config.size_rho} needed)"
            )
    parts: list[Part] = []
    context: list[Support] = []
    larger_all = True
    architecture = [feature for feature in FLEXFL_BLOCKS["architecture"] if feature != "total_parameters"]
    for slug in ("comm_bytes_total-log1p", "comm_time_total_s-log1p"):
        model = support(evidence, slug, "total_parameters", 1, stability, "required")
        data = support(evidence, slug, "n_samples", 1, stability, "competitor")
        carriers = [support(evidence, slug, feature, 1, stability, "carrier") for feature in architecture]
        parts.append(Part(f"total_parameters raises {slug}", (model,)))
        context += [data, *[item for item in carriers if item.present]]
        if model.present:
            rival = abs(data.row["decile_difference"]) if data.present else 0.0
            comparison = check(
                "|decile difference total_parameters| minus |decile difference n_samples|",
                UNITS[slug],
                abs(model.row["decile_difference"]) - rival,
                1,
                ">",
                0.0,
                1,
                part="equation",
                source=f"{slug}/feature_evidence.csv",
                target=slug,
            )
            checks.append(comparison)
            larger = bool(comparison.holds)
            larger_all &= larger
            if not larger:
                failures.append(f"{slug}: n_samples has the larger decile difference")
        named = [item.feature for item in carriers if item.supported]
        if named:
            failures.append(f"{slug}: architecture effect carried by {', '.join(named)}")
    return Assessment(
        checks,
        parts,
        observed,
        failures,
        context,
        equation_extra=larger_all,
        gate=None
        if gate.holds is not False
        else (
            f"total_parameters and n_samples are collinear across the {datasets} datasets "
            f"(Spearman {together:.3f}, |x| >= {config.size_collinear_rho})"
        ),
    )


@dataclass(frozen=True)
class Practice:
    name: str
    statement: str
    needed: tuple[str, ...]
    thresholds: tuple[str, ...]
    assess: Callable[[pl.DataFrame, dict[str, Evidence | None], PracticesConfig, Stability], Assessment]
    criterion: Callable[[PracticesConfig], str]
    confounds: str
    cap_field: str | None = None


PRACTICES = (
    Practice(
        "P1",
        "Asynchronous algorithms shorten wall time but need more epochs or bytes than synchronous ones.",
        ("dataset", "strategy", "combo", "fl_algo", "total_time_s", "n_epochs", "comm_bytes_total"),
        ("min_pairs", "min_stable_terms", "async_time_diff", "async_epoch_diff", "async_bytes_diff"),
        async_vs_sync,
        lambda c: (
            "Per family (Centralized, Decentralized), async minus sync paired on dataset, distribution and worker "
            f"combination, at least {c.min_pairs} pairs: median log1p(total_time_s) difference <= {c.async_time_diff} "
            f"and (median n_epochs difference >= {c.async_epoch_diff} or median log1p(comm_bytes_total) difference >= "
            f"{c.async_bytes_diff}). Equation: the family's async indicator lowers total_time_s and raises n_epochs or "
            "comm_bytes_total (or its sync indicator the opposite), passing all four filters. Met needs both families."
        ),
        (
            "Pairs match dataset, distribution and worker combination within one aggregation family. Learning rate, "
            "batch size and local epochs are sampled per algorithm, so they differ within a pair (CentralizedAsync "
            "classification uses a learning-rate range scaled by 8/n). Wall time is total_time_s, bytes "
            "comm_bytes_total."
        ),
    ),
    Practice(
        "P2",
        "More workers raise communication volume and time faster than they cut compute time.",
        (
            "dataset",
            "strategy",
            "fl_algo",
            "num_workers",
            "comm_bytes_total",
            "comm_time_total_s",
            "compute_time_max_s",
        ),
        ("min_pairs", "min_stable_terms", "workers_elasticity_gap"),
        workers_vs_communication,
        lambda c: (
            "Elasticity here is the OLS slope of log1p(y) on log(num_workers), a log1p-scale slope and not a log-log "
            "elasticity, within each (dataset, algorithm, distribution) "
            f"cell. Over at least {c.min_pairs} cells, the pooled median elasticity is > 0 for comm_bytes_total and "
            "comm_time_total_s and < 0 for compute_time_max_s, and each communication median exceeds |median "
            f"compute_time_max_s elasticity| by at least {c.workers_elasticity_gap}. Per-algorithm medians are "
            "reported and do not decide. Equation: num_workers raises comm_bytes_total and comm_time_total_s and "
            "lowers compute_time_max_s, passing all four filters."
        ),
        (
            "Compared across worker totals within dataset, algorithm and distribution. Compute is compute_time_max_s, "
            "the busiest worker that sets wall time, not compute_time_total_s, which sums over workers and grows with "
            "the worker count by construction. Block shares are not used: worker features enter only mixed terms."
        ),
    ),
    Practice(
        "P3",
        "Non-IID and Dirichlet splits need more epochs and reach lower performance than IID.",
        ("dataset", "strategy", "combo", "fl_algo", "n_epochs", "performance", "main_metric"),
        ("min_pairs", "min_stable_terms", "non_iid_epoch_diff", "non_iid_mcc_diff", "non_iid_mape_diff"),
        non_iid_vs_iid,
        lambda c: (
            "For non_iid and for dirichlet, minus iid, paired on dataset, algorithm and worker combination, at least "
            f"{c.min_pairs} pairs each: median n_epochs difference >= {c.non_iid_epoch_diff}, median MCC difference <= "
            f"{c.non_iid_mcc_diff} (classification) and median mape difference >= {c.non_iid_mape_diff} (regression, "
            "lower is better). Per-algorithm medians are reported and do not decide. Equation: strategy_<split> "
            "raises n_epochs, lowers MCC and raises mape (or strategy_iid the opposite), passing all four filters. "
            "Met needs both splits."
        ),
        (
            "The three distributions of a cell share hyperparameters (the seed derives from worker combination, "
            "dataset, algorithm and repeat), so the pairs differ only in the split. Performance is MCC (higher is "
            "better) for classification and mape, a fraction (lower is better), for regression."
        ),
    ),
    Practice(
        "P4",
        "Model size (parameter count) drives communication bytes and time more than dataset size.",
        ("dataset", "comm_bytes_total", "comm_time_total_s", "total_parameters", "n_samples"),
        ("size_rho", "size_collinear_rho", "size_max_outcome", "min_datasets", "min_stable_terms"),
        model_vs_dataset_size,
        lambda c: (
            f"Across at least {c.min_datasets} datasets, per-dataset median log1p of comm_bytes_total and of "
            f"comm_time_total_s: Spearman with total_parameters >= {c.size_rho} and |Spearman| with n_samples < "
            f"{c.size_rho}. Not tested when |Spearman(total_parameters, n_samples)| >= {c.size_collinear_rho}. "
            "Equation: total_parameters raises both targets, passes all four filters and has a larger "
            "|decile difference| than n_samples. Outcome capped at "
            f"{c.size_max_outcome}."
        ),
        (
            "Each dataset has one fixed network, so parameter count and dataset size vary only across the datasets, "
            "never within one, and cannot be separated from dataset identity. Bytes are parameters times messages "
            "by construction (the analytic message model), so 'model size drives bytes' is partly true by definition. "
            "The equation comparison contrasts decile extremes of dataset-constant features, so each side is one "
            "dataset, and the two features' extremes are different datasets."
        ),
        cap_field="size_max_outcome",
    ),
)


def judge(
    practice: Practice,
    frame: pl.DataFrame,
    evidence: dict[str, Evidence | None],
    config: PracticesConfig,
    stability: Stability,
) -> Outcome:
    thresholds = {name: getattr(config, name) for name in practice.thresholds}
    common = (practice.name, practice.statement, practice.criterion(config), practice.confounds, thresholds)
    missing = lacking(frame, practice.needed)
    if missing:
        return Outcome(*common, (), (), (), None, None, NOT_TESTED, f"corpus lacks {', '.join(missing)}")
    found = practice.assess(frame, evidence, config, stability)
    untested = [
        f"{item.quantity}, {scope_text(item)} (n={item.n}{dropped_text(item)})"
        for item in found.checks
        if item.decides and not item.tested
    ]
    unavailable = [part.describe() for part in found.parts if not part.available]
    observed = None if untested or found.gate else found.observed
    equation = None if unavailable else all(part.supported for part in found.parts) and found.equation_extra
    if found.gate or untested or unavailable:
        reasons = ([found.gate] if found.gate else []) + [f"too few matched units: {text}" for text in untested]
        reasons += unavailable
        if observed is False:
            reasons += found.failures
        label, reason = NOT_TESTED, "; ".join(reasons)
    else:
        failures = list(found.failures)
        failures += [
            f"{part.name}: no alternative passes the filters with the expected sign in a stable term"
            for part in found.parts
            if not part.supported
        ]
        if observed and equation:
            label, reason = MET, "observed criterion and equation evidence both hold"
            if practice.cap_field and getattr(config, practice.cap_field) == QUALIFIED:
                label, reason = QUALIFIED, reason + f"; capped by {practice.cap_field}"
        else:
            label, reason = QUALIFIED, "; ".join(failures) if failures else "criterion not fully met"
    return Outcome(
        *common, tuple(found.checks), tuple(found.parts), tuple(found.context), observed, equation, label, reason
    )


def load_corpus(path: str | Path) -> pl.DataFrame:
    frame = pl.read_csv(path, infer_schema_length=None)
    frame = frame.select([column for column in COLUMNS if column in frame.columns])
    if "combo" in frame.columns:
        frame = frame.with_columns(pl.col("combo").cast(pl.String))
    return frame


def dropped_text(item: Check) -> str:
    return f", {item.dropped} non-finite dropped" if item.dropped else ""


def scope_text(item: Check) -> str:
    return ", ".join(f"{key} {value}" for key, value in item.scope.items())


def describe_observed(outcome: Outcome) -> str:
    return "; ".join(
        f"{scope_text(item)}: {item.quantity} {item.value:.4f} {item.unit} (n={item.n})"
        for item in outcome.checks
        if item.part == "observed" and (item.decides or item.unit == "ratio")
    )


def describe_support(item: Support) -> str:
    if not item.present:
        return f"{item.slug}/{item.feature}: {item.state}"
    terms = ", ".join(
        f"{term} [{blocks}] effect {effect:.3f} freq {frequency:.3f}" for term, blocks, effect, frequency in item.terms
    )
    return (
        f"{item.slug}/{item.feature} ({item.role}"
        + (f", expected sign {item.expected:+d}" if item.role in ("required", "alternative") else "")
        + "): decile difference "
        f"{item.row['decile_difference']:.4f}, Spearman {item.row['spearman']:.3f}, mean frequency "
        f"{item.row['mean_frequency']:.3f}, passes_all {bool(item.row['passes_all'])}"
        + (f"; terms {terms}" if terms else "")
    )


def unique_supports(outcome: Outcome) -> list[tuple[str, Support]]:
    named: dict[tuple[str, str], tuple[list[str], Support]] = {}
    pairs = [(part.name, item) for part in outcome.parts for item in part.alternatives]
    pairs += [("context", item) for item in outcome.context]
    for name, item in pairs:
        names, _ = named.setdefault((item.slug, item.feature), ([], item))
        names.append(name)
    return [("; ".join(names), item) for names, item in named.values()]


def describe_equation(outcome: Outcome) -> str:
    support_text = "; ".join(describe_support(item) for _, item in unique_supports(outcome))
    comparison_text = "; ".join(
        f"{item.quantity} {item.value:.4f} {item.unit} (n={item.n}, {scope_text(item)})"
        for item in outcome.checks
        if item.part == "equation"
    )
    return "; ".join(text for text in (support_text, comparison_text) if text)


def detail_rows(outcome: Outcome) -> list[dict[str, object]]:
    blank: dict[str, object] = {name: None for name in DETAIL_SCHEMA}
    rows: list[dict[str, object]] = []
    for item in outcome.checks:
        rows.append(
            blank
            | item.scope
            | {
                "practice": outcome.practice,
                "part": item.part,
                "role": "criterion" if item.decides else "descriptive",
                "quantity": item.quantity,
                "unit": item.unit,
                "value": item.value,
                "n": item.n,
                "direction": item.direction or None,
                "threshold": None if item.direction == "" else item.threshold,
                "holds": item.holds,
                "decides": item.decides,
                "source": item.source,
                "note": None if item.tested and not item.dropped else (
                    "not computable" if not item.tested else f"{item.dropped} non-finite pairs dropped"
                ) + (f"; {item.dropped} non-finite pairs dropped" if not item.tested and item.dropped else ""),
            }
        )
    for name, item in unique_supports(outcome):
        base = blank | {
            "practice": outcome.practice,
            "part": "equation",
            "role": item.role,
            "target": item.slug,
            "feature": item.feature,
            "decides": item.role in ("required", "alternative"),
            "note": name,
        }
        if not item.present:
            rows.append(
                base
                | {
                    "quantity": "present",
                    "unit": "flag",
                    "value": 0.0,
                    "source": f"{item.slug}/feature_evidence.csv",
                    "note": f"{name}: {item.state}",
                }
            )
            continue
        source = f"{item.slug}/feature_evidence.csv"
        unit = UNITS.get(item.slug, "")
        for quantity in item.row:
            row = base | {
                "quantity": quantity,
                "unit": unit if quantity == "decile_difference" else ("flag" if quantity.startswith("pass") else ""),
                "value": item.row[quantity],
                "source": source,
            }
            if quantity == "decile_difference" and item.role in ("required", "alternative"):
                sign = float(np.sign(item.row[quantity])) == item.expected
                row |= {"direction": ">" if item.expected > 0 else "<", "threshold": 0.0, "holds": sign}
            rows.append(row)
        if item.role in ("required", "alternative"):
            rows.append(
                base
                | {
                    "quantity": "supported",
                    "unit": "flag",
                    "value": float(item.supported),
                    "holds": item.supported,
                    "source": source,
                    "note": f"{name}: passes_all, expected sign and at least one stable term",
                }
            )
        effects = [effect for _, _, effect, _ in item.terms]
        if effects:
            for quantity, value in (("term_effect_min", min(effects)), ("term_effect_max", max(effects))):
                rows.append(
                    base
                    | {"quantity": quantity, "unit": unit, "value": value, "source": f"{item.slug}/term_frequency.csv"}
                )
            rows.append(
                base
                | {
                    "quantity": "stable_terms",
                    "unit": "count",
                    "value": float(item.stable_terms),
                    "direction": ">=",
                    "threshold": float(item.min_terms),
                    "holds": item.stable_terms >= item.min_terms,
                    "source": f"{item.slug}/term_frequency.csv",
                }
            )
        for term, blocks, effect, frequency in item.terms:
            for quantity, value in (("term_effect", effect), ("term_frequency", frequency)):
                rows.append(
                    base
                    | {
                        "term": term,
                        "quantity": quantity,
                        "unit": unit if quantity == "term_effect" else "fraction of exclusions",
                        "value": value,
                        "source": f"{item.slug}/term_frequency.csv",
                        "note": f"{name}; blocks {blocks}",
                    }
                )
    return rows


def corpus_check(evidence: dict[str, Evidence | None], corpus_sha256: str) -> str:
    present = {slug: found for slug, found in evidence.items() if found is not None}
    stale = {
        slug: found.corpus_sha256 for slug, found in present.items() if found.corpus_sha256 not in (None, corpus_sha256)
    }
    if stale:
        raise ValueError(f"tables were fitted on another corpus than {corpus_sha256}: {stale}")
    unchecked = sorted(slug for slug, found in present.items() if found.corpus_sha256 is None)
    checked = len(present) - len(unchecked)
    text = f"{checked} of {len(present)} tables carry corpus {corpus_sha256}"
    return text + (f"; no baseline_comparison.csv in {', '.join(unchecked)}" if unchecked else "")


def flag_check(evidence: dict[str, Evidence | None], min_frequency: float) -> str:
    margins = []
    for slug, found in evidence.items():
        if found is None or found.features.height == 0:
            continue
        expected = found.features["mean_frequency"] >= min_frequency
        stale = found.features.filter(found.features["pass_frequency"] != expected)["feature"].to_list()
        if stale:
            raise ValueError(f"{slug}: pass_frequency disagrees with min_frequency {min_frequency} for {stale}")
        margins.append(f"{slug}={','.join(str(v) for v in sorted(found.features['margin'].unique().to_list()))}")
    return "fit margins " + "; ".join(margins) if margins else ""


def diagnose(
    frame: pl.DataFrame,
    tables: Path,
    config: PracticesConfig,
    *,
    min_frequency: float,
    corpus_sha256: str,
    commit: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    evidence = {slug: read_evidence(tables, slug) for slug in SLUGS}
    guarded = corpus_check(evidence, corpus_sha256)
    flags = flag_check(evidence, min_frequency)
    guarded = f"{guarded}; {flags}" if flags else guarded
    stability = Stability(min_frequency, config.min_stable_terms)
    outcomes = [judge(practice, frame, evidence, config, stability) for practice in PRACTICES]
    summary = pl.DataFrame(
        [
            {
                "practice": outcome.practice,
                "statement": outcome.statement,
                "criterion": outcome.criterion,
                "observed": describe_observed(outcome),
                "equation_evidence": describe_equation(outcome),
                "confounds": outcome.confounds,
                "observed_holds": outcome.observed_holds,
                "equation_holds": outcome.equation_holds,
                "outcome": outcome.outcome,
                "reason": outcome.reason,
                "thresholds": json.dumps(outcome.thresholds | {"min_frequency": min_frequency}, sort_keys=True),
                "tables_corpus_check": guarded,
                "corpus_sha256": corpus_sha256,
                "commit": commit,
            }
            for outcome in outcomes
        ],
        schema=SUMMARY_SCHEMA,
    )
    detail = pl.DataFrame([row for outcome in outcomes for row in detail_rows(outcome)], schema=DETAIL_SCHEMA)
    return summary, detail


def current_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="ml-meta-perf-practices",
        description="Test four FL practices against a FlexFL corpus and the tables of an earlier FlexFL run.",
        default_config_files=[str(DEFAULT_CONFIG_PATH)],
    )
    parser.add_argument("--config", action=ActionConfigFile, help="study configuration JSON")
    parser.add_argument("--search", type=Configuration, help="unused here")
    parser.add_argument("--selection", type=SelectionConfig, help="unused here")
    parser.add_argument("--opaque", type=OpaqueConfig, help="unused here")
    parser.add_argument("--effects", type=EffectsConfig, help="min_frequency marks a stable term")
    parser.add_argument("--sweep", type=SweepConfig, help="unused here")
    parser.add_argument("--flexfl_practices", type=PracticesConfig, help="the practice criteria and thresholds")
    parser.add_argument("--data", type=str, required=True, help="FlexFL meta-dataset CSV")
    parser.add_argument(
        "--tables", type=str, required=True, help="the flexfl/ folder of an earlier run, one subfolder per target"
    )
    parser.add_argument("--output", type=str, default="results", help="writes <output>/flexfl/practices*.csv")
    parser.add_argument("--commit", type=str | None, default=None, help="commit to record (default: git HEAD)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.instantiate(parser.parse_args(argv))
    digest = corpus_digest(arguments.data)
    commit = arguments.commit or current_commit()
    summary, detail = diagnose(
        load_corpus(arguments.data),
        Path(arguments.tables),
        arguments.flexfl_practices,
        min_frequency=arguments.effects.min_frequency,
        corpus_sha256=digest,
        commit=commit,
    )
    folder = Path(arguments.output) / "flexfl"
    folder.mkdir(parents=True, exist_ok=True)
    summary.write_csv(folder / "practices.csv")
    detail.write_csv(folder / "practices_detail.csv")
    outcomes = ", ".join(f"{row['practice']} {row['outcome']}" for row in summary.iter_rows(named=True))
    print(f"commit {commit} corpus {digest}: {outcomes}")
    print(f"written to {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
