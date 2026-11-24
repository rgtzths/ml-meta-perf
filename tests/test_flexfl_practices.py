"""FlexFL practice diagnostics: each outcome label from synthetic corpora and evidence tables."""

import contextlib
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ml_meta_perf import flexfl_practices as practices
from ml_meta_perf.config import PracticesConfig, load_config, parse_config
from ml_meta_perf.flexfl_baselines import corpus_digest
from ml_meta_perf.flexfl_practices import (
    DETAIL_SCHEMA,
    MET,
    NOT_TESTED,
    QUALIFIED,
    SUMMARY_SCHEMA,
    Evidence,
    Stability,
    diagnose,
    holds,
    load_corpus,
    main,
    support,
)
from ml_meta_perf.model import Equation
from ml_meta_perf.terms import Atom, Term
from tests.test_flexfl_schema import FIXTURE

ALGORITHMS = ("CentralizedSync", "CentralizedAsync", "DecentralizedSync", "DecentralizedAsync")
STRATEGIES = ("iid", "non_iid", "dirichlet")
PARAMETERS = (100, 200, 300, 400, 500, 600, 700, 800)
SAMPLES = (500, 100, 800, 300, 700, 200, 600, 400)
MIN_FREQUENCY = 0.5
SHARE_WEIGHTS = {"n_samples": 0.1}

SUPPORTING: dict[str, list[tuple[str, float, bool]]] = {
    "total_time_s-log1p": [("fl_algo_CentralizedAsync", -1.0, True), ("fl_algo_DecentralizedAsync", -1.0, True)],
    "n_epochs": [
        ("fl_algo_CentralizedAsync", 5.0, True),
        ("fl_algo_DecentralizedAsync", 5.0, True),
        ("strategy_non_iid", 5.0, True),
        ("strategy_dirichlet", 5.0, True),
    ],
    "comm_bytes_total-log1p": [
        ("num_workers", 1.0, True),
        ("total_parameters", 2.0, True),
        ("n_samples", 0.1, False),
    ],
    "comm_time_total_s-log1p": [("num_workers", 1.0, True), ("total_parameters", 2.0, True)],
    "compute_time_max_s-log1p": [("num_workers", -0.3, True)],
    "performance-classification": [("strategy_non_iid", -0.05, True), ("strategy_dirichlet", -0.05, True)],
    "performance-regression": [("strategy_non_iid", 0.05, True), ("strategy_dirichlet", 0.05, True)],
}


def practices_config(**changes: object) -> PracticesConfig:
    return dataclasses.replace(load_config().flexfl_practices, **changes)  # pyright: ignore[reportArgumentType]


def corpus(
    *,
    async_time: float = 0.5,
    async_epochs: float = 5.0,
    skew_epochs: float = 5.0,
    skew_mcc: float = -0.05,
    skew_mape: float = 0.05,
    compute_exponent: float = -0.3,
    samples: tuple[int, ...] = SAMPLES,
    strategies: tuple[str, ...] = STRATEGIES,
    slow_family: str | None = None,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index, (parameters, size) in enumerate(zip(PARAMETERS, samples, strict=True)):
        metric = "mcc" if index % 2 == 0 else "mape"
        for algorithm in ALGORITHMS:
            asynchronous = algorithm.endswith("Async")
            slowed = slow_family is not None and algorithm.startswith(slow_family)
            for strategy in strategies:
                skewed = strategy != "iid"
                for combo in range(5):
                    workers = 6 + 4 * combo
                    rows.append(
                        {
                            "dataset": f"d{index}",
                            "fl_algo": algorithm,
                            "strategy": strategy,
                            "combo": f"c{combo}",
                            "num_workers": workers,
                            "main_metric": metric,
                            "performance": (0.6 if metric == "mcc" else 0.1)
                            + (0.001 * combo)
                            + ((skew_mcc if metric == "mcc" else skew_mape) if skewed else 0.0),
                            "total_time_s": 1000.0 * (1 + combo) * (async_time if asynchronous and not slowed else 1.0),
                            "n_epochs": 20
                            + combo
                            + (async_epochs if asynchronous else 0)
                            + (skew_epochs if skewed else 0),
                            "comm_bytes_total": 1000.0 * parameters * workers,
                            "comm_time_total_s": 0.1 * parameters * workers,
                            "compute_time_max_s": 100.0 * workers**compute_exponent,
                            "total_parameters": parameters,
                            "n_samples": size,
                        }
                    )
    return pl.DataFrame(rows)


def write_tables(
    folder: Path,
    evidence: dict[str, list[tuple[str, float, bool]]],
    *,
    digest: str | None = None,
    frequency: float = 1.0,
    weights: dict[str, float] | None = None,
) -> Path:
    for slug, features in evidence.items():
        target = folder / slug
        target.mkdir(parents=True)
        names = [feature for feature, _, _ in features]
        pl.DataFrame(
            {
                "feature": names,
                "meaning": names,
                "n_terms": [1] * len(names),
                "decile_difference": [difference for _, difference, _ in features],
                "spearman": [float(np.sign(difference)) for _, difference, _ in features],
                "mean_frequency": [frequency] * len(names),
                "margin": [0.1] * len(names),
                "pass_margin": [passes for _, _, passes in features],
                "pass_direction": [passes for _, _, passes in features],
                "pass_sign": [passes for _, _, passes in features],
                "pass_frequency": [frequency >= MIN_FREQUENCY] * len(names),
                "passes_all": [passes for _, _, passes in features],
            }
        ).write_csv(target / "feature_evidence.csv")
        pl.DataFrame(
            {
                "term": names,
                "group": ["mixed"] * len(names),
                "blocks": ["fl_config"] * len(names),
                "weight": [(weights or {}).get(name, 1.0) for name in names],
                "beta": [1.0] * len(names),
                "effect": [abs(difference) for _, difference, _ in features],
                "direction": ["raises"] * len(names),
                "folds": [8] * len(names),
                "searches": [8] * len(names),
                "frequency": [frequency] * len(names),
            }
        ).write_csv(target / "term_frequency.csv")
        Equation(
            intercept=0.0,
            terms=tuple(Term("atom", (Atom(name),)) for name in names),
            weights=tuple((weights or {}).get(name, 1.0) for name in names),
            standardized_weights=tuple(1.0 for _ in names),
        ).save(target / "equation.json")
        if digest is not None:
            pl.DataFrame({"target": [slug], "corpus_sha256": [digest]}).write_csv(target / "baseline_comparison.csv")
    return folder


def run(
    frame: pl.DataFrame, tables: Path, config: PracticesConfig | None = None, **options: Any
) -> tuple[pl.DataFrame, pl.DataFrame]:
    settings = {"min_frequency": MIN_FREQUENCY, "corpus_sha256": "abc", "commit": "test"} | options
    return diagnose(frame, tables, practices_config() if config is None else config, **settings)


def summary_row(frame: pl.DataFrame, tables: Path, config: PracticesConfig, practice: str) -> dict[str, Any]:
    summary, _ = run(frame, tables, config)
    return summary.filter(pl.col("practice") == practice).row(0, named=True)


def outcomes(frame: pl.DataFrame, tables: Path, config: PracticesConfig) -> dict[str, str]:
    summary, _ = run(frame, tables, config)
    return dict(zip(summary["practice"].to_list(), summary["outcome"].to_list(), strict=True))


def set_feature(tables: Path, slug: str, feature: str, **values: object) -> None:
    path = tables / slug / "feature_evidence.csv"
    target = pl.col("feature") == feature
    pl.read_csv(path).with_columns(
        *[pl.when(target).then(pl.lit(value)).otherwise(pl.col(name)).alias(name) for name, value in values.items()]
    ).write_csv(path)


class PracticeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def tables(self, evidence: dict[str, list[tuple[str, float, bool]]] | None = None, **options: object) -> Path:
        folder = self.root / f"tables{len(list(self.root.iterdir()))}"
        options.setdefault("weights", SHARE_WEIGHTS)
        return write_tables(folder, SUPPORTING if evidence is None else evidence, **options)  # pyright: ignore[reportArgumentType]


class TestEveryPracticeMet(PracticeTestCase):
    def test_supporting_corpus_and_evidence_meet_every_practice_when_uncapped(self) -> None:
        labels = outcomes(corpus(), self.tables(), practices_config(size_max_outcome=MET))
        self.assertEqual(labels, {"P1": MET, "P2": MET, "P3": MET, "P4": MET})

    def test_the_configured_cap_holds_model_size_at_qualified(self) -> None:
        summary, _ = run(corpus(), self.tables())
        row = summary.filter(pl.col("practice") == "P4").row(0, named=True)
        self.assertEqual(row["outcome"], QUALIFIED)
        self.assertIn("capped by size_max_outcome", row["reason"])

    def test_a_threshold_read_from_config_decides_the_label(self) -> None:
        labels = outcomes(corpus(), self.tables(), practices_config(async_time_diff=-2.0))
        self.assertEqual(labels["P1"], QUALIFIED)


class TestAsyncVersusSync(PracticeTestCase):
    def test_a_contradicted_practice_is_qualified_not_untested(self) -> None:
        self.assertEqual(outcomes(corpus(async_time=2.0), self.tables(), practices_config())["P1"], QUALIFIED)

    def test_one_family_missing_the_criterion_is_qualified(self) -> None:
        frame = corpus(slow_family="Decentralized")
        self.assertEqual(outcomes(frame, self.tables(), practices_config())["P1"], QUALIFIED)

    def test_bytes_can_carry_the_cost_side_when_epochs_do_not(self) -> None:
        frame = corpus(async_epochs=0.0).with_columns(
            pl.when(pl.col("fl_algo").str.ends_with("Async"))
            .then(pl.col("comm_bytes_total") * 2.0)
            .otherwise(pl.col("comm_bytes_total"))
            .alias("comm_bytes_total")
        )
        evidence = dict(SUPPORTING)
        evidence["comm_bytes_total-log1p"] = [
            *SUPPORTING["comm_bytes_total-log1p"],
            ("fl_algo_CentralizedAsync", 0.7, True),
            ("fl_algo_DecentralizedAsync", 0.7, True),
        ]
        evidence["n_epochs"] = [row for row in SUPPORTING["n_epochs"] if row[0].startswith("strategy")]
        self.assertEqual(outcomes(frame, self.tables(evidence), practices_config())["P1"], MET)

    def test_the_sync_indicator_with_the_opposite_sign_counts_as_evidence(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["total_time_s-log1p"] = [
            ("fl_algo_CentralizedSync", 1.0, True),
            ("fl_algo_DecentralizedSync", 1.0, True),
        ]
        self.assertEqual(outcomes(corpus(), self.tables(evidence), practices_config())["P1"], MET)

    def test_evidence_failing_the_filters_is_qualified(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["total_time_s-log1p"] = [
            (feature, value, False) for feature, value, _ in SUPPORTING["total_time_s-log1p"]
        ]
        self.assertEqual(outcomes(corpus(), self.tables(evidence), practices_config())["P1"], QUALIFIED)

    def test_evidence_with_the_wrong_sign_is_qualified(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["total_time_s-log1p"] = [
            (feature, -value, True) for feature, value, _ in SUPPORTING["total_time_s-log1p"]
        ]
        self.assertEqual(outcomes(corpus(), self.tables(evidence), practices_config())["P1"], QUALIFIED)

    def test_too_few_pairs_is_not_tested(self) -> None:
        frame = corpus()
        pairs = frame.filter(pl.col("fl_algo") == "CentralizedSync").height
        self.assertEqual(outcomes(frame, self.tables(), practices_config(min_pairs=pairs))["P1"], MET)
        self.assertEqual(outcomes(frame, self.tables(), practices_config(min_pairs=pairs + 1))["P1"], NOT_TESTED)


class TestWorkersVersusCommunication(PracticeTestCase):
    def test_compute_rising_with_workers_is_qualified(self) -> None:
        self.assertEqual(outcomes(corpus(compute_exponent=0.3), self.tables(), practices_config())["P2"], QUALIFIED)

    def test_a_missing_compute_table_is_not_tested(self) -> None:
        evidence = {slug: rows for slug, rows in SUPPORTING.items() if slug != "compute_time_max_s-log1p"}
        row = summary_row(corpus(), self.tables(evidence), practices_config(), "P2")
        self.assertEqual(row["outcome"], NOT_TESTED)
        self.assertIn("num_workers table missing", row["reason"])

    def test_a_feature_absent_from_the_equation_is_not_tested(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["compute_time_max_s-log1p"] = [("n_samples", 1.0, True)]
        row = summary_row(corpus(), self.tables(evidence), practices_config(), "P2")
        self.assertEqual(row["outcome"], NOT_TESTED)
        self.assertIn("num_workers absent from the equation", row["reason"])
        self.assertTrue(row["observed_holds"])

    def test_a_feature_only_in_unstable_terms_is_qualified(self) -> None:
        row = summary_row(corpus(), self.tables(frequency=0.25), practices_config(), "P2")
        self.assertEqual(row["outcome"], QUALIFIED)
        self.assertTrue(row["observed_holds"])
        self.assertFalse(row["equation_holds"])

    def test_evidence_with_the_wrong_sign_is_qualified(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["compute_time_max_s-log1p"] = [("num_workers", 0.3, True)]
        self.assertEqual(outcomes(corpus(), self.tables(evidence), practices_config())["P2"], QUALIFIED)

    def test_a_gap_below_the_margin_is_qualified(self) -> None:
        row = summary_row(corpus(compute_exponent=-1.5), self.tables(), practices_config(), "P2")
        self.assertEqual(row["outcome"], QUALIFIED)
        self.assertIn("minus |median elasticity compute_time_max_s|", row["reason"])

    def test_per_algorithm_rows_are_reported_but_do_not_decide(self) -> None:
        _, detail = run(corpus(), self.tables())
        rows = detail.filter((pl.col("practice") == "P2") & (pl.col("algorithm") == "CentralizedAsync"))
        self.assertGreater(rows.height, 0)
        self.assertFalse(any(rows["decides"].to_list()))
        self.assertTrue(all(np.isfinite(rows["value"].to_numpy())))

    def test_cells_without_a_worker_spread_are_not_tested(self) -> None:
        frame = corpus().with_columns(pl.lit(8).alias("num_workers"))
        self.assertEqual(outcomes(frame, self.tables(), practices_config())["P2"], NOT_TESTED)


class TestNonIidVersusIid(PracticeTestCase):
    def test_regression_error_unchanged_is_qualified(self) -> None:
        self.assertEqual(outcomes(corpus(skew_mape=0.0), self.tables(), practices_config())["P3"], QUALIFIED)

    def test_mcc_and_mape_take_opposite_directions(self) -> None:
        self.assertEqual(outcomes(corpus(skew_mcc=0.05), self.tables(), practices_config())["P3"], QUALIFIED)

    def test_evidence_with_the_wrong_sign_is_qualified(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["performance-classification"] = [("strategy_non_iid", 0.05, True), ("strategy_dirichlet", 0.05, True)]
        self.assertEqual(outcomes(corpus(), self.tables(evidence), practices_config())["P3"], QUALIFIED)

    def test_no_strategy_feature_in_the_epoch_equation_is_not_tested(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["n_epochs"] = [row for row in SUPPORTING["n_epochs"] if row[0].startswith("fl_algo")]
        self.assertEqual(outcomes(corpus(), self.tables(evidence), practices_config())["P3"], NOT_TESTED)

    def test_a_missing_split_is_not_tested(self) -> None:
        frame = corpus(strategies=("iid", "non_iid"))
        self.assertEqual(outcomes(frame, self.tables(), practices_config())["P3"], NOT_TESTED)


class TestModelVersusDatasetSize(PracticeTestCase):
    def test_parameter_shares_met_with_detail_for_both_targets(self) -> None:
        summary, detail = run(corpus(), self.tables(), practices_config(size_max_outcome=MET))
        self.assertEqual(summary.filter(pl.col("practice") == "P4")["outcome"].to_list(), [MET])
        quantity = "share(total_parameters) minus share(n_samples) of output variance"
        rows = detail.filter((pl.col("practice") == "P4") & (pl.col("quantity") == quantity))
        self.assertEqual(set(rows["target"].to_list()), {"comm_bytes_total-log1p", "comm_time_total_s-log1p"})
        self.assertEqual(set(rows["part"].to_list()), {"equation"})
        self.assertEqual(
            set(rows["source"].to_list()),
            {
                "comm_bytes_total-log1p/equation.json and corpus",
                "comm_time_total_s-log1p/equation.json and corpus",
            },
        )
        self.assertTrue((rows["value"] >= 0.10).all())
        self.assertTrue(rows["holds"].all())
        self.assertEqual(set(rows["threshold"].to_list()), {0.10})

    def test_dataset_size_carries_more_share_and_names_the_target(self) -> None:
        summary, detail = run(
            corpus(), self.tables(weights={"n_samples": 3.0}), practices_config(size_max_outcome=MET)
        )
        row = summary.filter(pl.col("practice") == "P4").row(0, named=True)
        self.assertEqual(row["outcome"], QUALIFIED)
        self.assertIn("comm_bytes_total-log1p: share(total_parameters) minus share(n_samples) is", row["reason"])
        comparison = detail.filter(
            (pl.col("practice") == "P4")
            & (pl.col("target") == "comm_bytes_total-log1p")
            & (pl.col("quantity") == "share(total_parameters) minus share(n_samples) of output variance")
        ).row(0, named=True)
        self.assertFalse(comparison["holds"])

    def test_equal_shares_do_not_pass_a_positive_gap(self) -> None:
        config = practices_config(size_max_outcome=MET, size_share_gap=0.01)
        tables = self.tables(weights={"total_parameters": 1.0, "n_samples": 1.0})
        summary, detail = run(corpus(), tables, config)
        self.assertEqual(
            detail.filter(
                (pl.col("practice") == "P4")
                & (pl.col("target") == "comm_bytes_total-log1p")
                & (pl.col("quantity") == "share(total_parameters) minus share(n_samples) of output variance")
            ).height,
            1,
        )
        comparison = detail.filter(
            (pl.col("practice") == "P4")
            & (pl.col("target") == "comm_bytes_total-log1p")
            & (pl.col("quantity") == "share(total_parameters) minus share(n_samples) of output variance")
        ).row(0, named=True)
        self.assertAlmostEqual(comparison["value"], 0.0, delta=1e-9)
        self.assertFalse(comparison["holds"])
        self.assertEqual(summary.filter(pl.col("practice") == "P4")["outcome"].to_list(), [QUALIFIED])

    def test_a_negative_dataset_size_share_adds_to_the_gap(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["comm_time_total_s-log1p"] = [
            ("num_workers", 1.0, True),
            ("total_parameters", 2.0, True),
            ("n_samples", 0.1, False),
        ]
        frame = corpus(samples=(700, 300, 800, 500, 600, 100, 400, 200))
        summary, detail = run(
            frame,
            self.tables(evidence, weights={"n_samples": 0.3}),
            practices_config(size_max_outcome=MET, size_share_gap=1.0),
        )
        shares = detail.filter(
            (pl.col("practice") == "P4")
            & (pl.col("quantity") == "share of output variance in terms containing the feature")
            & (pl.col("feature") == "n_samples")
        )
        self.assertEqual(shares.height, 2)
        self.assertTrue((shares["value"] < 0.0).all())
        self.assertEqual(summary.filter(pl.col("practice") == "P4")["outcome"].to_list(), [MET])

    def test_configured_share_gap_decides_the_label(self) -> None:
        weight = 0.92
        columns = corpus()
        parameter = columns["total_parameters"].to_numpy().astype(float)
        samples = columns["n_samples"].to_numpy().astype(float) * weight
        total = parameter + samples
        expected = float(
            np.mean((parameter - parameter.mean()) * (total - total.mean())) / np.var(total)
            - np.mean((samples - samples.mean()) * (total - total.mean())) / np.var(total)
        )
        self.assertGreater(expected, 0.05)
        self.assertLess(expected, 0.10)
        self.assertEqual(
            outcomes(
                corpus(), self.tables(weights={"n_samples": weight}), practices_config(size_max_outcome=MET)
            )["P4"],
            QUALIFIED,
        )
        self.assertEqual(
            outcomes(
                corpus(),
                self.tables(weights={"n_samples": weight}),
                practices_config(size_max_outcome=MET, size_share_gap=0.05),
            )["P4"],
            MET,
        )

    def test_share_gap_equal_to_threshold_holds_but_next_float_does_not(self) -> None:
        quantity = "share(total_parameters) minus share(n_samples) of output variance"
        _, detail = run(corpus(), self.tables(), practices_config(size_max_outcome=MET))
        rows = detail.filter((pl.col("practice") == "P4") & (pl.col("quantity") == quantity))
        self.assertEqual(rows.height, 2)
        gap = float(np.min(rows["value"].to_numpy()))
        self.assertGreater(gap, 0.0)
        above = float(np.nextafter(gap, np.inf))
        self.assertLessEqual(above, 1.0)
        self.assertEqual(
            outcomes(corpus(), self.tables(), practices_config(size_max_outcome=MET, size_share_gap=gap))["P4"],
            MET,
        )
        summary, above_detail = run(
            corpus(), self.tables(), practices_config(size_max_outcome=MET, size_share_gap=above)
        )
        self.assertEqual(summary.filter(pl.col("practice") == "P4")["outcome"].to_list(), [QUALIFIED])
        above_rows = above_detail.filter((pl.col("practice") == "P4") & (pl.col("quantity") == quantity))
        smaller_target = rows.filter(pl.col("value") == gap)["target"][0]
        self.assertFalse(above_rows.filter(pl.col("target") == smaller_target)["holds"][0])

    def test_missing_equation_column_makes_share_not_computable(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["comm_bytes_total-log1p"] = [*SUPPORTING["comm_bytes_total-log1p"], ("missing_column", 1.0, True)]
        summary, detail = run(corpus(), self.tables(evidence), practices_config(size_max_outcome=MET))
        row = summary.filter(pl.col("practice") == "P4").row(0, named=True)
        self.assertEqual(row["outcome"], NOT_TESTED)
        self.assertIn("not computable", row["reason"])
        comparison = detail.filter(
            (pl.col("practice") == "P4")
            & (pl.col("target") == "comm_bytes_total-log1p")
            & (pl.col("quantity") == "share(total_parameters) minus share(n_samples) of output variance")
        ).row(0, named=True)
        self.assertGreater(comparison["n"], 0)
        self.assertIsNone(comparison["holds"])

    def test_absent_n_samples_term_has_zero_share(self) -> None:
        _, detail = run(corpus(), self.tables(), practices_config(size_max_outcome=MET))
        quantity = "share(total_parameters) minus share(n_samples) of output variance"
        self.assertEqual(
            detail.filter(
                (pl.col("practice") == "P4")
                & (pl.col("target") == "comm_time_total_s-log1p")
                & (pl.col("quantity") == quantity)
            ).height,
            1,
        )
        comparison = detail.filter(
            (pl.col("practice") == "P4")
            & (pl.col("target") == "comm_time_total_s-log1p")
            & (pl.col("quantity") == quantity)
        ).row(0, named=True)
        self.assertTrue(comparison["holds"])
        self.assertEqual(
            detail.filter(
                (pl.col("practice") == "P4")
                & (pl.col("target") == "comm_time_total_s-log1p")
                & (pl.col("quantity") == "share of output variance in terms containing the feature")
                & (pl.col("feature") == "n_samples")
            ).height,
            1,
        )
        share = detail.filter(
            (pl.col("practice") == "P4")
            & (pl.col("target") == "comm_time_total_s-log1p")
            & (pl.col("quantity") == "share of output variance in terms containing the feature")
            & (pl.col("feature") == "n_samples")
        ).row(0, named=True)
        self.assertEqual(share["value"], 0.0)

    def test_decile_extremes_do_not_decide_the_equation_comparison(self) -> None:
        evidence = dict(SUPPORTING)
        for slug in ("comm_bytes_total-log1p", "comm_time_total_s-log1p"):
            evidence[slug] = [
                (
                    feature,
                    2.0 if feature == "total_parameters" else 9.0 if feature == "n_samples" else difference,
                    passes,
                )
                for feature, difference, passes in SUPPORTING[slug]
            ]
        self.assertEqual(
            outcomes(corpus(), self.tables(evidence), practices_config(size_max_outcome=MET))["P4"],
            MET,
        )
        for slug in ("comm_bytes_total-log1p", "comm_time_total_s-log1p"):
            evidence[slug] = [
                (
                    feature,
                    9.0 if feature == "total_parameters" else 2.0 if feature == "n_samples" else difference,
                    passes,
                )
                for feature, difference, passes in SUPPORTING[slug]
            ]
        self.assertEqual(
            outcomes(
                corpus(), self.tables(evidence, weights={"n_samples": 3.0}), practices_config(size_max_outcome=MET)
            )["P4"],
            QUALIFIED,
        )

    def test_collinear_sizes_are_not_tested(self) -> None:
        frame = corpus(samples=PARAMETERS)
        self.assertEqual(outcomes(frame, self.tables(), practices_config(size_max_outcome=MET))["P4"], NOT_TESTED)

    def test_dataset_size_larger_in_the_equation_is_qualified(self) -> None:
        labels = outcomes(
            corpus(), self.tables(weights={"n_samples": 3.0}), practices_config(size_max_outcome=MET)
        )
        self.assertEqual(labels["P4"], QUALIFIED)

    def test_parameters_absent_from_an_equation_are_not_tested(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["comm_time_total_s-log1p"] = [("num_workers", 1.0, True), ("max_layer_width", 3.0, True)]
        row = summary_row(corpus(), self.tables(evidence), practices_config(size_max_outcome=MET), "P4")
        self.assertEqual(row["outcome"], NOT_TESTED)
        self.assertIn("total_parameters absent from the equation", row["reason"])

    def test_other_architecture_features_carrying_the_effect_are_named(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["comm_time_total_s-log1p"] = [
            ("num_workers", 1.0, True),
            ("total_parameters", 2.0, False),
            ("max_layer_width", 3.0, True),
            ("n_layers", -3.0, True),
        ]
        frame = corpus().with_columns(pl.lit(1.0).alias("max_layer_width"), pl.lit(1.0).alias("n_layers"))
        row = summary_row(frame, self.tables(evidence), practices_config(size_max_outcome=MET), "P4")
        self.assertEqual(row["outcome"], QUALIFIED)
        self.assertIn("architecture effect carried by max_layer_width", row["reason"])
        self.assertNotIn("n_layers", row["reason"])

    def test_dataset_size_tracking_communication_is_qualified(self) -> None:
        frame = corpus().with_columns(
            (pl.col("n_samples") * 1000.0 * pl.col("num_workers")).alias("comm_bytes_total"),
            (pl.col("n_samples") * 0.1 * pl.col("num_workers")).alias("comm_time_total_s"),
        )
        row = summary_row(frame, self.tables(), practices_config(size_max_outcome=MET), "P4")
        self.assertEqual(row["outcome"], QUALIFIED)
        self.assertFalse(row["observed_holds"])


class TestOutputs(PracticeTestCase):
    def test_tables_fitted_on_another_corpus_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "another corpus"):
            run(corpus(), self.tables(digest="000000000000"), commit="t")

    def test_tables_carry_their_schemas_thresholds_and_sources(self) -> None:
        summary, detail = run(corpus(), self.tables(), practices_config(async_time_diff=-0.37), commit="c0ffee")
        self.assertEqual(summary.schema, pl.Schema(SUMMARY_SCHEMA))
        self.assertEqual(detail.schema, pl.Schema(DETAIL_SCHEMA))
        self.assertEqual(set(summary["corpus_sha256"].to_list()), {"abc"})
        self.assertEqual(set(summary["commit"].to_list()), {"c0ffee"})
        self.assertIn('"min_pairs": 30', summary["thresholds"][0])
        rows = {row["practice"]: row for row in summary.iter_rows(named=True)}
        self.assertIn("-0.37", rows["P1"]["criterion"])
        for row in rows.values():
            with self.subTest(practice=row["practice"]):
                self.assertTrue(row["observed"])
                self.assertTrue(row["equation_evidence"])
        self.assertIn("median async-sync difference, total_time_s", rows["P1"]["observed"])
        self.assertIn("fl_algo_CentralizedAsync", rows["P1"]["equation_evidence"])
        self.assertNotIn("each side is one dataset", rows["P4"]["confounds"])
        self.assertIn("features that share a term overlap", rows["P4"]["confounds"])
        self.assertIn("log1p-scale slope", rows["P2"]["criterion"])
        keys = {
            "P1": {"min_pairs", "async_time_diff", "async_epoch_diff", "async_bytes_diff"},
            "P2": {"min_pairs", "workers_elasticity_gap"},
            "P3": {"min_pairs", "non_iid_epoch_diff", "non_iid_mcc_diff", "non_iid_mape_diff"},
            "P4": {"size_rho", "size_collinear_rho", "size_share_gap", "size_max_outcome", "min_datasets"},
        }
        for practice, names in keys.items():
            with self.subTest(practice=practice):
                recorded = set(json.loads(rows[practice]["thresholds"]))
                self.assertEqual(recorded, names | {"min_frequency", "min_stable_terms"})
        sources = set(detail["source"].to_list())
        self.assertIn("corpus", sources)
        self.assertIn("total_time_s-log1p/feature_evidence.csv", sources)
        self.assertIn("n_epochs/term_frequency.csv", sources)
        self.assertIn("comm_bytes_total-log1p/equation.json and corpus", sources)
        self.assertIn("comm_time_total_s-log1p/equation.json and corpus", sources)

    def test_tables_without_a_corpus_digest_are_recorded_as_unchecked(self) -> None:
        row = summary_row(corpus(), self.tables(), practices_config(), "P1")
        self.assertIn("0 of 7 tables carry corpus abc", row["tables_corpus_check"])

    def test_a_missing_corpus_column_leaves_only_its_practices_untested(self) -> None:
        labels = outcomes(corpus().drop("compute_time_max_s"), self.tables(), practices_config(size_max_outcome=MET))
        self.assertEqual(labels, {"P1": MET, "P2": NOT_TESTED, "P3": MET, "P4": MET})

    def test_duplicate_pair_keys_are_rejected(self) -> None:
        frame = corpus()
        with self.assertRaisesRegex(ValueError, "more than one row"):
            outcomes(pl.concat([frame, frame.head(1)]), self.tables(), practices_config())

    def test_evidence_shared_by_two_parts_is_reported_once(self) -> None:
        _, detail = run(corpus(), self.tables(), commit="t")
        rows = detail.filter(
            (pl.col("practice") == "P3") & (pl.col("feature") == "strategy_iid") & (pl.col("quantity") == "present")
        )
        self.assertEqual(rows["target"].to_list(), ["n_epochs", "performance-classification", "performance-regression"])
        self.assertIn("non_iid raises n_epochs; dirichlet raises n_epochs", rows["note"][0])

    def test_a_contradicted_observation_is_named_in_a_not_tested_reason(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["n_epochs"] = [row for row in SUPPORTING["n_epochs"] if row[0].startswith("fl_algo")]
        row = summary_row(corpus(skew_epochs=0.0), self.tables(evidence), practices_config(), "P3")
        self.assertEqual(row["outcome"], NOT_TESTED)
        self.assertIn("median non_iid-iid difference, n_epochs 0.0000 not >= 2.0", row["reason"])

    def test_the_sign_flag_and_the_support_flag_are_separate_rows(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["compute_time_max_s-log1p"] = [("num_workers", -0.3, False)]
        _, detail = run(corpus(), self.tables(evidence), commit="t")
        rows = detail.filter((pl.col("target") == "compute_time_max_s-log1p") & (pl.col("feature") == "num_workers"))
        flags = dict(zip(rows["quantity"].to_list(), rows["holds"].to_list(), strict=True))
        self.assertTrue(flags["decile_difference"])
        self.assertFalse(flags["supported"])

    def test_the_wall_time_ratio_is_on_raw_seconds(self) -> None:
        _, detail = run(corpus(), self.tables(), commit="t")
        ratio = detail.filter(pl.col("quantity") == "median async/sync ratio, total_time_s")["value"].to_list()
        self.assertEqual(ratio, [0.5, 0.5])

    def test_a_mixed_term_is_found_for_each_of_its_features(self) -> None:
        name = "[log(total_parameters)] * [log(num_workers)]"
        evidence = Evidence(
            features=pl.DataFrame(
                {
                    "feature": ["num_workers"],
                    "decile_difference": [1.0],
                    "spearman": [1.0],
                    "mean_frequency": [1.0],
                    "margin": [0.1],
                    "n_terms": [1],
                    "pass_margin": [True],
                    "pass_direction": [True],
                    "pass_sign": [True],
                    "pass_frequency": [True],
                    "passes_all": [True],
                }
            ),
            terms=pl.DataFrame(
                {"term": [name], "blocks": ["architecture+federation"], "effect": [2.0], "frequency": [1.0]}
            ),
            term_features={name: ("total_parameters", "num_workers")},
            corpus_sha256=None,
            equation=Equation(0.0, (), (), ()),
        )
        found = support(
            {"comm_time_total_s-log1p": evidence},
            "comm_time_total_s-log1p",
            "num_workers",
            1,
            Stability(0.5, 1),
            "required",
        )
        self.assertTrue(found.supported)
        self.assertEqual(found.terms, ((name, "architecture+federation", 2.0, 1.0),))

    def test_the_shipped_fixture_has_no_matched_units(self) -> None:
        summary, _ = run(load_corpus(FIXTURE), self.tables(), commit="t")
        for row in summary.head(3).iter_rows(named=True):
            with self.subTest(practice=row["practice"]):
                self.assertEqual(row["outcome"], NOT_TESTED)
                self.assertIn("too few matched units", row["reason"])
                self.assertNotIn("absent from the equation", row["reason"])

    def test_main_writes_both_tables(self) -> None:
        data = self.root / "corpus.csv"
        corpus().write_csv(data)
        tables = self.tables()
        output = self.root / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(["--data", str(data), "--tables", str(tables), "--output", str(output), "--commit", "c0ffee"])
        self.assertEqual(status, 0)
        summary = pl.read_csv(output / "flexfl" / "practices.csv")
        self.assertEqual(summary.height, 4)
        self.assertEqual(set(summary["commit"].to_list()), {"c0ffee"})
        self.assertTrue((output / "flexfl" / "practices_detail.csv").is_file())


class TestGuards(PracticeTestCase):
    def test_a_non_finite_metric_row_is_dropped_and_counted(self) -> None:
        for task, dataset, bad in (("classification", "d0", float("nan")), ("regression", "d1", float("inf"))):
            with self.subTest(task=task):
                target = (
                    (pl.col("dataset") == dataset)
                    & (pl.col("strategy") == "iid")
                    & (pl.col("fl_algo") == "CentralizedSync")
                    & (pl.col("combo") == "c0")
                )
                frame = corpus().with_columns(
                    pl.when(target).then(bad).otherwise(pl.col("performance")).alias("performance")
                )
                summary, detail = run(frame, self.tables(), commit="t")
                row = summary.filter(pl.col("practice") == "P3").row(0, named=True)
                self.assertEqual(row["outcome"], MET)
                rows = detail.filter(
                    (pl.col("practice") == "P3")
                    & (pl.col("task") == task)
                    & (pl.col("algorithm") == "all")
                    & pl.col("decides")
                )
                self.assertEqual(rows["note"].to_list(), ["1 non-finite units dropped"] * 2)

    def test_a_constant_dataset_size_is_not_computable(self) -> None:
        frame = corpus(samples=(500,) * 8)
        row = summary_row(frame, self.tables(), practices_config(size_max_outcome=MET), "P4")
        self.assertEqual(row["outcome"], NOT_TESTED)
        self.assertIn("not computable", row["reason"])
        self.assertNotIn("too few matched units", row["reason"])

    def test_a_dataset_with_two_parameter_counts_is_rejected(self) -> None:
        frame = corpus().with_columns(
            pl.when((pl.col("dataset") == "d0") & (pl.col("combo") == "c0"))
            .then(150)
            .otherwise(pl.col("total_parameters"))
            .alias("total_parameters")
        )
        with self.assertRaisesRegex(ValueError, "varies within dataset"):
            outcomes(frame, self.tables(), practices_config())

    def test_a_flag_that_disagrees_with_the_current_min_frequency_is_rejected(self) -> None:
        tables = self.tables()
        path = tables / "n_epochs" / "feature_evidence.csv"
        table = pl.read_csv(path)
        table.with_columns(
            pl.when(pl.col("feature") == "strategy_non_iid").then(False).otherwise(pl.col("pass_frequency")).alias(
                "pass_frequency"
            )
        ).write_csv(path)
        with self.assertRaisesRegex(ValueError, "pass_frequency disagrees"):
            outcomes(corpus(), tables, practices_config())

    def test_a_non_finite_mean_frequency_with_a_false_flag_is_accepted(self) -> None:
        tables = self.tables()
        set_feature(tables, "n_epochs", "strategy_non_iid", mean_frequency=float("nan"), pass_frequency=False)
        summary, _ = run(corpus(), tables)
        self.assertEqual(summary.height, 4)
        self.assertIn("fit margins", summary["tables_corpus_check"][0])

    def test_a_true_flag_below_the_current_min_frequency_is_rejected(self) -> None:
        tables = self.tables()
        set_feature(tables, "n_epochs", "strategy_non_iid", mean_frequency=MIN_FREQUENCY / 2, pass_frequency=True)
        with self.assertRaisesRegex(ValueError, "pass_frequency disagrees"):
            run(corpus(), tables)

    def test_the_fit_margins_are_listed_for_matching_tables(self) -> None:
        row = summary_row(corpus(), self.tables(), practices_config(), "P1")
        self.assertIn("fit margins", row["tables_corpus_check"])

    def test_the_equation_comparison_is_reported_under_equation_evidence(self) -> None:
        summary, detail = run(corpus(), self.tables(), practices_config(size_max_outcome=MET), commit="t")
        row = summary.filter(pl.col("practice") == "P4").row(0, named=True)
        quantity = "share(total_parameters) minus share(n_samples) of output variance"
        self.assertNotIn(quantity, row["observed"])
        self.assertIn(quantity, row["equation_evidence"])
        rows = detail.filter(pl.col("quantity") == quantity)
        self.assertEqual(set(rows["part"].to_list()), {"equation"})
        self.assertEqual(
            set(rows["source"].to_list()),
            {"comm_bytes_total-log1p/equation.json and corpus", "comm_time_total_s-log1p/equation.json and corpus"},
        )

    def test_the_fit_margins_name_each_table(self) -> None:
        row = summary_row(corpus(), self.tables(digest="abc"), practices_config(), "P1")
        self.assertIn("total_time_s-log1p=0.1", row["tables_corpus_check"])

    def test_the_stable_term_minimum_is_stated_in_the_criterion_and_the_support_rows(self) -> None:
        summary, detail = run(corpus(), self.tables(), practices_config(min_stable_terms=2))
        criterion = summary.filter(pl.col("practice") == "P1")["criterion"][0]
        self.assertIn("at least 2 stable term", criterion)
        notes = detail.filter((pl.col("practice") == "P1") & (pl.col("quantity") == "supported"))["note"].to_list()
        self.assertGreater(len(notes), 0)
        for note in notes:
            self.assertIn("at least 2 stable term", note)

    def test_the_stable_term_threshold_row_carries_the_configured_minimum(self) -> None:
        _, detail = run(corpus(), self.tables(), practices_config(min_stable_terms=2))
        thresholds = detail.filter(pl.col("quantity") == "stable_terms")["threshold"].unique().to_list()
        self.assertEqual(thresholds, [2.0])

    def test_cells_with_a_non_finite_elasticity_are_dropped_and_counted(self) -> None:
        pooled = (
            (pl.col("practice") == "P2")
            & (pl.col("algorithm") == "all")
            & (pl.col("quantity") == "median elasticity, compute_time_max_s")
        )
        _, clean = run(corpus(), self.tables())
        cell = (pl.col("dataset") == "d0") & (pl.col("fl_algo") == "CentralizedSync") & (pl.col("strategy") == "iid")
        frame = corpus().with_columns(
            pl.when(cell).then(float("nan")).otherwise(pl.col("compute_time_max_s")).alias("compute_time_max_s")
        )
        summary, detail = run(frame, self.tables())
        row = detail.filter(pooled).row(0, named=True)
        self.assertIn("1 non-finite units dropped", row["note"] or "")
        self.assertEqual(row["n"], clean.filter(pooled)["n"][0] - 1)
        self.assertEqual(summary.filter(pl.col("practice") == "P2")["outcome"][0], MET)

    def test_a_non_finite_wall_time_is_dropped_and_counted(self) -> None:
        row = (
            (pl.col("dataset") == "d0")
            & (pl.col("fl_algo") == "CentralizedSync")
            & (pl.col("strategy") == "iid")
            & (pl.col("combo") == "c0")
        )
        frame = corpus().with_columns(
            pl.when(row).then(float("nan")).otherwise(pl.col("total_time_s")).alias("total_time_s")
        )
        summary, detail = run(frame, self.tables())
        self.assertEqual(summary.filter(pl.col("practice") == "P1")["outcome"][0], MET)
        rows = detail.filter(
            (pl.col("practice") == "P1")
            & (pl.col("family") == "Centralized")
            & (pl.col("quantity") == "median async-sync difference, total_time_s")
        )
        self.assertEqual(rows["note"].to_list(), ["1 non-finite units dropped"])

    def test_a_dataset_with_two_sample_counts_is_rejected(self) -> None:
        frame = corpus().with_columns(
            pl.when((pl.col("dataset") == "d0") & (pl.col("combo") == "c0"))
            .then(999)
            .otherwise(pl.col("n_samples"))
            .alias("n_samples")
        )
        with self.assertRaisesRegex(ValueError, "varies within dataset"):
            outcomes(frame, self.tables(), practices_config())

    def test_constant_communication_bytes_leave_the_size_practice_untested(self) -> None:
        frame = corpus().with_columns(pl.lit(1000.0).alias("comm_bytes_total"))
        row = summary_row(frame, self.tables(), practices_config(size_max_outcome=MET), "P4")
        self.assertEqual(row["outcome"], NOT_TESTED)
        self.assertIn("not computable", row["reason"])

    def test_non_finite_iid_rows_leave_a_split_without_enough_pairs(self) -> None:
        frame = corpus().with_columns(
            pl.when(
                (pl.col("strategy") == "iid")
                & (pl.col("main_metric") == "mcc")
                & pl.col("dataset").is_in(["d0", "d2", "d4"])
            )
            .then(float("nan"))
            .otherwise(pl.col("performance"))
            .alias("performance")
        )
        summary, detail = run(frame, self.tables())
        reason = summary.filter(pl.col("practice") == "P3")["reason"][0]
        self.assertEqual(summary.filter(pl.col("practice") == "P3")["outcome"][0], NOT_TESTED)
        self.assertIn("non-finite units dropped", reason)
        notes = detail.filter(
            (pl.col("practice") == "P3")
            & (pl.col("task") == "classification")
            & (pl.col("algorithm") == "all")
            & pl.col("decides")
        )["note"].to_list()
        self.assertEqual(len(notes), 2)
        for note in notes:
            self.assertTrue(note.startswith("not computable; "))

    def test_the_dataset_minimum_comes_from_the_config(self) -> None:
        labels = outcomes(corpus(), self.tables(), practices_config(min_datasets=9, size_max_outcome=MET))
        self.assertEqual(labels["P4"], NOT_TESTED)

    def test_the_stable_term_minimum_comes_from_the_config(self) -> None:
        labels = outcomes(corpus(), self.tables(), practices_config(min_stable_terms=2))
        self.assertEqual(labels["P2"], QUALIFIED)

    def test_the_recorded_thresholds_name_the_new_keys(self) -> None:
        summary, _ = run(corpus(), self.tables(), commit="t")
        recorded = {row["practice"]: json.loads(row["thresholds"]) for row in summary.iter_rows(named=True)}
        self.assertIn("min_datasets", recorded["P4"])
        for thresholds in recorded.values():
            self.assertIn("min_stable_terms", thresholds)


class TestCoverage(PracticeTestCase):
    def test_matching_digests_are_counted_as_checked(self) -> None:
        row = summary_row(corpus(), self.tables(digest="abc"), practices_config(), "P1")
        self.assertTrue(row["tables_corpus_check"].startswith("7 of 7 tables carry corpus abc"))

    def test_a_table_without_a_baseline_file_is_named(self) -> None:
        tables = self.tables(digest="abc")
        (tables / "n_epochs" / "baseline_comparison.csv").unlink()
        text = summary_row(corpus(), tables, practices_config(), "P1")["tables_corpus_check"]
        self.assertTrue(text.startswith("6 of 7 tables carry corpus abc"))
        self.assertIn("no baseline_comparison.csv in n_epochs", text)

    def test_a_baseline_file_naming_two_corpora_is_rejected(self) -> None:
        tables = self.tables(digest="abc")
        pl.DataFrame({"target": ["n_epochs"] * 2, "corpus_sha256": ["abc", "def"]}).write_csv(
            tables / "n_epochs" / "baseline_comparison.csv"
        )
        with self.assertRaisesRegex(ValueError, "another corpus"):
            outcomes(corpus(), tables, practices_config())

    def test_main_accepts_tables_fitted_on_the_same_corpus(self) -> None:
        data = self.root / "corpus.csv"
        corpus().write_csv(data)
        digest = corpus_digest(data)
        tables = self.tables(digest=digest)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main(["--data", str(data), "--tables", str(tables), "--output", str(self.root / "out")])
        self.assertEqual(status, 0)
        self.assertIn(digest, stdout.getvalue())
        summary = pl.read_csv(self.root / "out" / "flexfl" / "practices.csv")
        self.assertTrue(summary["tables_corpus_check"][0].startswith(f"7 of 7 tables carry corpus {digest}"))

    def test_the_iid_indicator_alone_can_carry_the_skew_evidence(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["n_epochs"] = [("strategy_iid", -5.0, True)]
        evidence["performance-classification"] = [("strategy_iid", 0.05, True)]
        evidence["performance-regression"] = [("strategy_iid", -0.05, True)]
        self.assertEqual(outcomes(corpus(), self.tables(evidence), practices_config())["P3"], MET)

    def test_one_wrong_signed_split_indicator_is_qualified(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["n_epochs"] = [("strategy_non_iid", 5.0, True), ("strategy_dirichlet", -5.0, True)]
        self.assertEqual(outcomes(corpus(), self.tables(evidence), practices_config())["P3"], QUALIFIED)

    def test_the_sync_indicator_alone_can_carry_the_epoch_side(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["n_epochs"] = [("fl_algo_CentralizedSync", -5.0, True), ("fl_algo_DecentralizedSync", -5.0, True)]
        self.assertEqual(outcomes(corpus(), self.tables(evidence), practices_config())["P1"], MET)

    def test_a_task_without_enough_pairs_leaves_the_split_untested(self) -> None:
        frame = corpus().with_columns(pl.lit("mcc").alias("main_metric"))
        row = summary_row(frame, self.tables(), practices_config(), "P3")
        self.assertEqual(row["outcome"], NOT_TESTED)
        self.assertIn("mape", row["reason"])

    def test_min_pairs_is_inclusive_for_the_pooled_cells(self) -> None:
        _, detail = run(corpus(), self.tables(), commit="t")
        cells = detail.filter((pl.col("practice") == "P2") & pl.col("decides") & (pl.col("part") == "observed"))[
            "n"
        ].unique().to_list()
        self.assertEqual(len(cells), 1)
        self.assertEqual(outcomes(corpus(), self.tables(), practices_config(min_pairs=cells[0]))["P2"], MET)
        self.assertEqual(outcomes(corpus(), self.tables(), practices_config(min_pairs=cells[0] + 1))["P2"], NOT_TESTED)

    def test_every_comparison_direction_is_exact_at_its_threshold(self) -> None:
        cases = (
            (1.0, ">=", True),
            (1.0, "<=", True),
            (1.0, ">", False),
            (1.0, "<", False),
            (-1.0, "|x| <", False),
            (0.9, "|x| <", True),
        )
        for value, direction, expected in cases:
            with self.subTest(direction=direction, value=value):
                self.assertEqual(holds(value, direction, 1.0), expected)
        with self.assertRaises(ValueError):
            holds(1.0, "==", 1.0)

    def test_the_algorithm_and_split_lists_come_from_the_baselines(self) -> None:
        self.assertEqual(
            practices.ALGORITHMS, ("CentralizedSync", "CentralizedAsync", "DecentralizedSync", "DecentralizedAsync")
        )
        self.assertEqual(practices.SKEWED, ("non_iid", "dirichlet"))

    def test_the_pooled_median_decides_over_one_rising_algorithm(self) -> None:
        frame = corpus().with_columns(
            pl.when(pl.col("fl_algo") == "CentralizedSync")
            .then(100.0 * pl.col("num_workers").cast(pl.Float64) ** 0.3)
            .otherwise(pl.col("compute_time_max_s"))
            .alias("compute_time_max_s")
        )
        _, detail = run(frame, self.tables(), commit="t")
        own = detail.filter(
            (pl.col("practice") == "P2")
            & (pl.col("algorithm") == "CentralizedSync")
            & (pl.col("quantity") == "median elasticity, compute_time_max_s")
        )
        self.assertGreater(own["value"][0], 0.0)
        self.assertEqual(outcomes(frame, self.tables(), practices_config())["P2"], MET)

    def test_a_communication_time_flat_in_workers_is_qualified(self) -> None:
        frame = corpus().with_columns((pl.col("total_parameters") * 0.1).alias("comm_time_total_s"))
        row = summary_row(frame, self.tables(), practices_config(), "P2")
        self.assertEqual(row["outcome"], QUALIFIED)
        self.assertIn("comm_time_total_s", row["reason"])

    def test_main_applies_a_threshold_override(self) -> None:
        data = self.root / "corpus.csv"
        corpus().write_csv(data)
        output = self.root / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(
                [
                    "--data",
                    str(data),
                    "--tables",
                    str(self.tables()),
                    "--output",
                    str(output),
                    "--commit",
                    "c0ffee",
                    "--flexfl_practices.min_pairs",
                    "100000",
                ]
            )
        self.assertEqual(status, 0)
        first = pl.read_csv(output / "flexfl" / "practices.csv").row(0, named=True)
        self.assertEqual(first["outcome"], NOT_TESTED)
        self.assertEqual(json.loads(first["thresholds"])["min_pairs"], 100000)

    def test_two_datasets_are_too_few_for_the_size_comparison(self) -> None:
        frame = corpus().filter(pl.col("dataset").is_in(["d0", "d1"]))
        row = summary_row(frame, self.tables(), practices_config(size_max_outcome=MET), "P4")
        self.assertEqual(row["outcome"], NOT_TESTED)
        self.assertIn("too few matched units", row["reason"])
        self.assertNotIn("collinear", row["reason"])

    def test_a_stable_term_needs_the_configured_frequency_and_count(self) -> None:
        name = "[log(num_workers)]"
        other = "[num_workers] * [total_parameters]"

        def evidence(frequencies: tuple[float, ...]) -> dict[str, Evidence | None]:
            names = (name, other)[: len(frequencies)]
            found = Evidence(
                features=pl.DataFrame(
                    {
                        "feature": ["num_workers"],
                        "decile_difference": [1.0],
                        "spearman": [1.0],
                        "mean_frequency": [1.0],
                        "margin": [0.1],
                        "n_terms": [len(names)],
                        "pass_margin": [True],
                        "pass_direction": [True],
                        "pass_sign": [True],
                        "pass_frequency": [True],
                        "passes_all": [True],
                    }
                ),
                terms=pl.DataFrame(
                    {
                        "term": list(names),
                        "blocks": ["federation"] * len(names),
                        "effect": [1.0] * len(names),
                        "frequency": list(frequencies),
                    }
                ),
                term_features={term: ("num_workers",) for term in names},
                corpus_sha256=None,
                equation=Equation(0.0, (), (), ()),
            )
            return {"n_epochs": found}

        cases = (
            ((0.9, 0.1), Stability(0.5, 1), True),
            ((0.5,), Stability(0.5, 1), True),
            ((0.49,), Stability(0.5, 1), False),
            ((0.9, 0.1), Stability(0.5, 2), False),
            ((0.9, 0.8), Stability(0.5, 2), True),
        )
        for frequencies, stability, expected in cases:
            with self.subTest(frequencies=frequencies, stability=stability):
                found = support(evidence(frequencies), "n_epochs", "num_workers", 1, stability, "required")
                self.assertEqual(found.supported, expected)


class TestPracticesConfig(unittest.TestCase):
    def test_the_shipped_file_holds_the_locked_thresholds(self) -> None:
        config = load_config().flexfl_practices
        self.assertEqual(config.min_pairs, 30)
        self.assertEqual(config.size_max_outcome, QUALIFIED)
        self.assertEqual(config.size_rho, 0.74)
        self.assertEqual(config.min_datasets, 3)
        self.assertEqual(config.min_stable_terms, 1)
        self.assertEqual(config.async_time_diff, -0.10)
        self.assertEqual(config.async_epoch_diff, 2.0)
        self.assertEqual(config.async_bytes_diff, 0.10)
        self.assertEqual(config.workers_elasticity_gap, 0.10)
        self.assertEqual(config.non_iid_epoch_diff, 2.0)
        self.assertEqual(config.non_iid_mcc_diff, -0.02)
        self.assertEqual(config.non_iid_mape_diff, 0.02)
        self.assertEqual(config.size_collinear_rho, 0.74)
        self.assertEqual(config.size_share_gap, 0.10)

    def test_invalid_values_are_rejected(self) -> None:
        self.assertIn("size_share_gap", PracticesConfig.__dataclass_fields__)
        for change in (
            {"min_pairs": 1},
            {"size_rho": 0.0},
            {"size_share_gap": 0.0},
            {"size_share_gap": 1.5},
            {"size_share_gap": float("nan")},
            {"size_max_outcome": "not tested"},
            {"async_time_diff": 0.1},
            {"non_iid_mcc_diff": 0.02},
            {"async_epoch_diff": -1.0},
            {"workers_elasticity_gap": float("nan")},
            {"min_datasets": 2},
            {"min_stable_terms": 0},
            {"async_bytes_diff": -0.1},
            {"non_iid_epoch_diff": -1.0},
            {"non_iid_mape_diff": -0.02},
            {"async_time_diff": float("inf")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                practices_config(**change)

    def test_a_missing_section_is_an_error(self) -> None:
        payload = dataclasses.asdict(load_config())
        del payload["flexfl_practices"]
        with self.assertRaises(KeyError):
            parse_config(payload)


if __name__ == "__main__":
    unittest.main()
