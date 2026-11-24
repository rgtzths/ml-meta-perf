"""FlexFL practice diagnostics: each outcome label from synthetic corpora and evidence tables."""

import contextlib
import dataclasses
import io
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ml_meta_perf.config import PracticesConfig, load_config, parse_config
from ml_meta_perf.flexfl_practices import (
    DETAIL_SCHEMA,
    MET,
    NOT_TESTED,
    QUALIFIED,
    SUMMARY_SCHEMA,
    Evidence,
    diagnose,
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
                "pass_frequency": [passes for _, _, passes in features],
                "passes_all": [passes for _, _, passes in features],
            }
        ).write_csv(target / "feature_evidence.csv")
        pl.DataFrame(
            {
                "term": names,
                "group": ["mixed"] * len(names),
                "blocks": ["fl_config"] * len(names),
                "weight": [1.0] * len(names),
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
            weights=tuple(1.0 for _ in names),
            standardized_weights=tuple(1.0 for _ in names),
        ).save(target / "equation.json")
        if digest is not None:
            pl.DataFrame({"target": [slug], "corpus_sha256": [digest]}).write_csv(target / "baseline_comparison.csv")
    return folder


def summary_row(frame: pl.DataFrame, tables: Path, config: PracticesConfig, practice: str) -> dict[str, Any]:
    summary, _ = diagnose(frame, tables, config, min_frequency=0.5, corpus_sha256="abc", commit="test")
    return summary.filter(pl.col("practice") == practice).row(0, named=True)


def outcomes(frame: pl.DataFrame, tables: Path, config: PracticesConfig) -> dict[str, str]:
    summary, _ = diagnose(frame, tables, config, min_frequency=0.5, corpus_sha256="abc", commit="test")
    return dict(zip(summary["practice"].to_list(), summary["outcome"].to_list(), strict=True))


class PracticeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def tables(self, evidence: dict[str, list[tuple[str, float, bool]]] | None = None, **options: object) -> Path:
        folder = self.root / f"tables{len(list(self.root.iterdir()))}"
        return write_tables(folder, SUPPORTING if evidence is None else evidence, **options)  # pyright: ignore[reportArgumentType]


class TestEveryPracticeMet(PracticeTestCase):
    def test_supporting_corpus_and_evidence_meet_every_practice_when_uncapped(self) -> None:
        labels = outcomes(corpus(), self.tables(), practices_config(size_max_outcome=MET))
        self.assertEqual(labels, {"P1": MET, "P2": MET, "P3": MET, "P4": MET})

    def test_the_configured_cap_holds_model_size_at_qualified(self) -> None:
        summary, _ = diagnose(
            corpus(), self.tables(), practices_config(), min_frequency=0.5, corpus_sha256="abc", commit="test"
        )
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
        _, detail = diagnose(
            corpus(), self.tables(), practices_config(), min_frequency=0.5, corpus_sha256="abc", commit="test"
        )
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
    def test_collinear_sizes_are_not_tested(self) -> None:
        frame = corpus(samples=PARAMETERS)
        self.assertEqual(outcomes(frame, self.tables(), practices_config(size_max_outcome=MET))["P4"], NOT_TESTED)

    def test_dataset_size_larger_in_the_equation_is_qualified(self) -> None:
        evidence = dict(SUPPORTING)
        evidence["comm_bytes_total-log1p"] = [
            ("num_workers", 1.0, True),
            ("total_parameters", 2.0, True),
            ("n_samples", 3.0, True),
        ]
        labels = outcomes(corpus(), self.tables(evidence), practices_config(size_max_outcome=MET))
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
        row = summary_row(corpus(), self.tables(evidence), practices_config(size_max_outcome=MET), "P4")
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
            diagnose(
                corpus(),
                self.tables(digest="000000000000"),
                practices_config(),
                min_frequency=0.5,
                corpus_sha256="abc",
                commit="t",
            )

    def test_tables_carry_their_schemas_thresholds_and_sources(self) -> None:
        summary, detail = diagnose(
            corpus(), self.tables(), practices_config(), min_frequency=0.5, corpus_sha256="abc", commit="c0ffee"
        )
        self.assertEqual(summary.schema, pl.Schema(SUMMARY_SCHEMA))
        self.assertEqual(detail.schema, pl.Schema(DETAIL_SCHEMA))
        self.assertEqual(set(summary["corpus_sha256"].to_list()), {"abc"})
        self.assertEqual(set(summary["commit"].to_list()), {"c0ffee"})
        self.assertIn('"min_pairs": 30', summary["thresholds"][0])
        sources = set(detail["source"].to_list())
        self.assertIn("corpus", sources)
        self.assertIn("total_time_s-log1p/feature_evidence.csv", sources)
        self.assertIn("n_epochs/term_frequency.csv", sources)

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
        _, detail = diagnose(
            corpus(), self.tables(), practices_config(), min_frequency=0.5, corpus_sha256="abc", commit="t"
        )
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
        _, detail = diagnose(
            corpus(), self.tables(evidence), practices_config(), min_frequency=0.5, corpus_sha256="abc", commit="t"
        )
        rows = detail.filter((pl.col("target") == "compute_time_max_s-log1p") & (pl.col("feature") == "num_workers"))
        flags = dict(zip(rows["quantity"].to_list(), rows["holds"].to_list(), strict=True))
        self.assertTrue(flags["decile_difference"])
        self.assertFalse(flags["supported"])

    def test_the_wall_time_ratio_is_on_raw_seconds(self) -> None:
        _, detail = diagnose(
            corpus(), self.tables(), practices_config(), min_frequency=0.5, corpus_sha256="abc", commit="t"
        )
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
        )
        found = support(
            {"comm_time_total_s-log1p": evidence}, "comm_time_total_s-log1p", "num_workers", 1, 0.5, "required"
        )
        self.assertTrue(found.supported)
        self.assertEqual(found.terms, ((name, "architecture+federation", 2.0, 1.0),))

    def test_the_shipped_fixture_has_no_matched_units(self) -> None:
        summary, _ = diagnose(
            load_corpus(FIXTURE), self.tables(), practices_config(), min_frequency=0.5, corpus_sha256="abc", commit="t"
        )
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


class TestPracticesConfig(unittest.TestCase):
    def test_the_shipped_file_holds_the_locked_thresholds(self) -> None:
        config = load_config().flexfl_practices
        self.assertEqual(config.min_pairs, 30)
        self.assertEqual(config.size_max_outcome, QUALIFIED)
        self.assertEqual(config.size_rho, 0.74)

    def test_invalid_values_are_rejected(self) -> None:
        for change in ({"min_pairs": 1}, {"size_rho": 0.0}, {"size_max_outcome": "not tested"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                practices_config(**change)

    def test_a_missing_section_is_an_error(self) -> None:
        payload = dataclasses.asdict(load_config())
        del payload["flexfl_practices"]
        with self.assertRaises(KeyError):
            parse_config(payload)


if __name__ == "__main__":
    unittest.main()
