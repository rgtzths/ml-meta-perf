"""The FlexFL baselines: the analytic message model, the fold helpers and the comparison table."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

from ml_meta_perf import cli, experiment
from ml_meta_perf import flexfl_baselines as baselines
from ml_meta_perf.config import load_config
from ml_meta_perf.data import drop_constant_features, flexfl_schema, load, target
from ml_meta_perf.flexfl_baselines import (
    ORACLE,
    PREDICTED,
    BaselineComparison,
    analytic_comm_bytes,
    corpus_digest,
    epoch_predictions,
    fold_boosting,
    fold_mean,
    fold_search,
    in_sample,
    out_of_fold,
    resolve_splits,
    run_baselines,
)
from ml_meta_perf.validate import leave_one_group_out, score
from tests.test_flexfl_schema import FIXTURE, TINY, TINY_FLAGS

ALGOS = np.array(["CentralizedSync", "CentralizedAsync", "DecentralizedSync", "DecentralizedAsync"])
PROTOCOLS = {"in_sample", "loo_dataset", "loo_model"}
LEARNED = {"mean", "GradientBoosting (50 stages)"}
ANALYTIC = {ORACLE, PREDICTED}
E3 = {"E3 (fixed form)", "E3 (form per fold)"}
SKIPPED = (
    "analytic baseline skipped: the corpus has no min_workers, pool_batches; "
    "re-assemble it with FlexFL's scripts/assemble_meta_dataset.py"
)


def with_analytic_columns(directory: Path, blank: str | None = None) -> Path:
    frame = pl.read_csv(FIXTURE)
    centralized = pl.col("fl_algo").is_in(["CentralizedSync", "CentralizedAsync"])
    frame = frame.with_columns(
        (pl.col("num_workers") // 2 + 1).alias("min_workers"),
        pl.when(centralized).then(pl.col("num_workers") * 3).otherwise(None).alias("pool_batches"),
    )
    if blank is not None:
        first = frame["fl_algo"].to_list().index("CentralizedSync")
        cleared = pl.when(pl.int_range(pl.len()) == first).then(None).otherwise(pl.col(blank))
        frame = frame.with_columns(cleared.alias(blank))
    path = directory / "with_analytic.csv"
    frame.write_csv(path)
    return path


def models(table: pl.DataFrame) -> set[str]:
    return set(table["model"].to_list())


def cell(table: pl.DataFrame, model: str, protocol: str, column: str) -> float:
    return table.filter((pl.col("model") == model) & (pl.col("protocol") == protocol))[column].item()


class AnalyticModelTest(unittest.TestCase):
    def bytes(self, n_epochs: float, epoch_cap: float, workers: float, pool: float) -> np.ndarray:
        ones = np.ones(4)
        return analytic_comm_bytes(ALGOS, n_epochs * ones, epoch_cap * ones, workers * ones, pool * ones, 10 * ones)

    def test_bytes_per_algorithm_at_the_epoch_cap(self) -> None:
        np.testing.assert_array_equal(self.bytes(3, 3, 4, 7), [1920.0, 1920.0, 960.0, 1200.0])

    def test_an_early_stop_adds_one_synchronous_round(self) -> None:
        np.testing.assert_array_equal(self.bytes(3, 5, 4, 7), [2240.0, 1920.0, 1280.0, 1200.0])

    def test_centralized_sync_rounds_up_to_whole_pools(self) -> None:
        self.assertEqual(self.bytes(1, 1, 2, 5)[0], 480.0)

    def test_decentralized_rows_ignore_pool_batches(self) -> None:
        twos = np.full(2, 2.0)
        predicted = analytic_comm_bytes(ALGOS[2:], twos, twos, np.full(2, 4.0), np.full(2, np.nan), np.full(2, 10.0))
        np.testing.assert_array_equal(predicted, [640.0, 880.0])

    def test_unknown_algorithm_is_rejected(self) -> None:
        ones = np.ones(1)
        with self.assertRaisesRegex(ValueError, "FedAvg"):
            analytic_comm_bytes(np.array(["FedAvg"]), ones, ones, ones, ones, ones)


class FoldHelpersTest(unittest.TestCase):
    truth = np.array([1.0, 2.0, 3.0, 10.0])
    labels = np.array(["a", "a", "b", "b"])

    def test_out_of_fold_mean_predicts_the_other_groups(self) -> None:
        predicted = out_of_fold(fold_mean(self.truth), self.truth, self.labels)
        np.testing.assert_array_equal(predicted, [6.5, 6.5, 1.5, 1.5])

    def test_out_of_fold_clips_to_the_training_range_unless_told_not_to(self) -> None:
        def constant(_train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return np.full(int(test.sum()), 100.0)

        np.testing.assert_array_equal(out_of_fold(constant, self.truth, self.labels), [10.0, 10.0, 2.0, 2.0])
        np.testing.assert_array_equal(out_of_fold(constant, self.truth, self.labels, clip=False), [100.0] * 4)

    def test_in_sample_clips_only_to_the_schema_bounds(self) -> None:
        def wide(_train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return np.array([-5.0, 2.0, 9.0])[test]

        np.testing.assert_array_equal(in_sample(wide, 3, None), [-5.0, 2.0, 9.0])
        np.testing.assert_array_equal(in_sample(wide, 3, (-1.0, 1.0)), [-1.0, 1.0, 1.0])

    def test_boosting_uses_the_given_sizes(self) -> None:
        design = np.arange(40, dtype=np.float64).reshape(20, 2)
        truth = np.sin(design[:, 0]) + design[:, 1]
        opaque = dataclasses.replace(load_config().opaque, boosting_stages=7, boosting_max_depth=2)
        every = np.ones(20, dtype=bool)
        expected = GradientBoostingRegressor(n_estimators=7, learning_rate=0.1, max_depth=2, random_state=0)
        expected.fit(design, truth)
        np.testing.assert_array_equal(fold_boosting(design, truth, opaque)(every, every), expected.predict(design))

    def test_fold_search_never_sees_the_held_out_rows(self) -> None:
        schema = flexfl_schema("comm_bytes_total", log_target=True)
        frame = load(FIXTURE, schema)
        schema = drop_constant_features(frame, schema)
        truth = target(frame, schema)
        test = frame["dataset"].to_numpy() == "clf_a"
        changed = truth.copy()
        changed[test] = changed[test] * 3.0 + 7.0
        original = fold_search(frame, schema, truth, 2, TINY)(~test, test)
        perturbed = fold_search(frame, schema, changed, 2, TINY)(~test, test)
        np.testing.assert_array_equal(original, perturbed)

    def test_fold_search_builds_its_library_from_the_training_rows(self) -> None:
        schema = flexfl_schema("comm_bytes_total", log_target=True)
        frame = load(FIXTURE, schema)
        schema = drop_constant_features(frame, schema)
        test = frame["dataset"].to_numpy() == "clf_a"
        seen: list[int] = []
        real = baselines.build_library

        def recording(*args: object, **kwargs: object) -> object:
            seen.append(len(next(iter(args[2].values()))))  # pyright: ignore[reportAttributeAccessIssue]
            return real(*args, **kwargs)  # pyright: ignore[reportArgumentType]

        with mock.patch.object(baselines, "build_library", side_effect=recording):
            fold_search(frame, schema, target(frame, schema), 2, TINY)(~test, test)
        self.assertEqual(seen, [int((~test).sum())])


class SplitsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = flexfl_schema("comm_bytes_total")
        self.frame = load(FIXTURE, self.schema)

    def test_defaults_are_the_dataset_and_algorithm(self) -> None:
        resolved = resolve_splits(self.frame, self.schema, None)
        self.assertEqual(list(resolved), ["loo_dataset", "loo_model"])
        np.testing.assert_array_equal(resolved["loo_model"], self.frame["fl_algo"].to_numpy())

    def test_an_extra_split_is_added_and_an_identical_default_is_accepted(self) -> None:
        extra = np.where(self.frame["num_workers"].to_numpy() > 4, "big", "small")
        splits = {"loo_dataset": self.frame["dataset"].to_numpy(), "by_size": extra}
        self.assertEqual(list(resolve_splits(self.frame, self.schema, splits)), ["loo_dataset", "loo_model", "by_size"])

    def test_reserved_names_cannot_be_redefined(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved for the dataset grouping"):
            resolve_splits(self.frame, self.schema, {"loo_dataset": self.frame["fl_algo"].to_numpy()})
        with self.assertRaisesRegex(ValueError, "not a held-out split"):
            resolve_splits(self.frame, self.schema, {"in_sample": self.frame["dataset"].to_numpy()})


class LoadExtraColumnsTest(unittest.TestCase):
    def test_extra_columns_are_kept_in_order_without_a_null_check(self) -> None:
        schema = flexfl_schema("comm_bytes_total")
        requested = ("pool_batches", "absent", "min_workers", "pool_batches", "comm_bytes_total")
        with tempfile.TemporaryDirectory() as directory:
            path = with_analytic_columns(Path(directory))
            frame = load(path, schema, extra_columns=requested)
            plain = load(path, schema)
        self.assertEqual(frame.columns, [*plain.columns, "pool_batches", "min_workers"])
        self.assertEqual(frame["pool_batches"].dtype, pl.Float64)
        self.assertEqual(frame["pool_batches"].null_count(), 12)
        self.assertTrue(frame.select(plain.columns).equals(plain))


class EpochPredictionsTest(unittest.TestCase):
    def test_predictions_are_floored_at_one_epoch_and_clipped_per_fold(self) -> None:
        schema = flexfl_schema("n_epochs")
        frame = load(FIXTURE, schema)
        truth = target(frame, schema)
        rows = frame.height
        report = SimpleNamespace(
            schema=drop_constant_features(frame, schema),
            equation=SimpleNamespace(n_terms=1, equation=SimpleNamespace(predict=lambda _columns: np.full(rows, -2.0))),
        )
        labels = frame["dataset"].to_numpy()
        expected = np.empty(rows)
        for _, train, test in leave_one_group_out(labels):
            expected[test] = max(1.0, float(truth[train].min()))

        def negative(_train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return np.full(int(test.sum()), -5.0)

        with mock.patch.object(baselines, "fold_search", return_value=negative):
            predicted = epoch_predictions(report, frame, truth, {"loo_dataset": labels}, TINY)  # pyright: ignore[reportArgumentType]
        self.assertEqual(list(predicted), ["in_sample", "loo_dataset"])
        np.testing.assert_array_equal(predicted["in_sample"], np.ones(rows))
        np.testing.assert_array_equal(predicted["loo_dataset"], expected)


class RunBaselinesTest(unittest.TestCase):
    def test_comm_bytes_without_the_analytic_columns(self) -> None:
        comparison = run_baselines(FIXTURE, flexfl_schema("comm_bytes_total"), TINY)
        table = comparison.table
        self.assertEqual(table.columns, ["target", "model", "protocol", "r2", "mae", "n", "corpus_sha256"])
        self.assertEqual(models(table), E3 | LEARNED)
        self.assertEqual(table.height, 11)
        per_fold = table.filter(pl.col("model") == "E3 (form per fold)")
        self.assertEqual(set(per_fold["protocol"]), PROTOCOLS - {"in_sample"})
        self.assertEqual(set(table["corpus_sha256"]), {hashlib.sha256(FIXTURE.read_bytes()).hexdigest()[:12]})
        self.assertEqual(set(table["n"]), {24})
        self.assertAlmostEqual(cell(table, "mean", "in_sample", "r2"), 0.0, places=12)
        self.assertGreater(cell(table, "GradientBoosting (50 stages)", "in_sample", "r2"), 0.5)
        self.assertEqual(comparison.notes, (SKIPPED,))

    def test_comm_bytes_with_the_analytic_columns(self) -> None:
        schema = flexfl_schema("comm_bytes_total", log_target=True)
        with tempfile.TemporaryDirectory() as directory:
            path = with_analytic_columns(Path(directory))
            comparison = run_baselines(path, schema, TINY)
            frame = load(path, schema, extra_columns=("min_workers", "pool_batches", "n_epochs"))
        table = comparison.table
        self.assertEqual(comparison.notes, ())
        self.assertEqual(models(table), E3 | LEARNED | ANALYTIC)
        self.assertEqual(table.height, 17)
        self.assertEqual(set(table["target"]), {"log1p(comm_bytes_total)"})
        expected = np.log1p(
            analytic_comm_bytes(
                frame["fl_algo"].to_numpy().astype(str),
                frame["n_epochs"].to_numpy(),
                frame["epoch_cap"].to_numpy(),
                frame["min_workers"].to_numpy(),
                frame["pool_batches"].to_numpy(),
                frame["total_parameters"].to_numpy(),
            )
        )
        truth = target(frame, schema)
        for protocol in PROTOCOLS:
            self.assertAlmostEqual(cell(table, ORACLE, protocol, "r2"), score(truth, expected).r2, places=12)

    def test_a_row_without_an_analytic_input_skips_the_analytic_rows(self) -> None:
        for blank, note in (
            ("pool_batches", "analytic baseline skipped: 1 Centralized rows have no pool_batches"),
            ("min_workers", "analytic baseline skipped: 1 rows have no min_workers"),
        ):
            with self.subTest(blank=blank), tempfile.TemporaryDirectory() as directory:
                path = with_analytic_columns(Path(directory), blank=blank)
                comparison = run_baselines(path, flexfl_schema("comm_bytes_total"), TINY)
                self.assertFalse(ANALYTIC & models(comparison.table))
                self.assertEqual(comparison.notes, (note,))

    def test_performance_has_no_analytic_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = with_analytic_columns(Path(directory))
            comparison = run_baselines(path, flexfl_schema("performance", "classification"), TINY)
        self.assertEqual(comparison.notes, ())
        self.assertEqual(models(comparison.table), E3 | LEARNED)
        self.assertEqual(set(comparison.table["n"]), {12})

    def test_an_extra_split_adds_rows_for_every_refitted_model(self) -> None:
        schema = flexfl_schema("comm_bytes_total")
        frame = load(FIXTURE, schema)
        workers = frame["num_workers"].to_numpy()
        halves = np.where(workers > np.median(workers), "big", "small")
        table = run_baselines(FIXTURE, schema, TINY, splits={"by_size": halves}).table
        self.assertEqual(models(table.filter(pl.col("protocol") == "by_size")), {"E3 (form per fold)"} | LEARNED)

    def test_a_report_for_other_rows_is_rejected(self) -> None:
        other = experiment.run_flexfl(FIXTURE, flexfl_schema("performance", "classification"), TINY)
        with self.assertRaisesRegex(ValueError, "the E3 fit scored 12 rows, the baselines load 24"):
            run_baselines(FIXTURE, flexfl_schema("comm_bytes_total"), TINY, report=other)

    def test_corpus_digest_is_the_sha256_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "abc.csv"
            path.write_bytes(b"abc")
            self.assertEqual(corpus_digest(path), "ba7816bf8f01")
            path.write_bytes(b"abd")
            self.assertNotEqual(corpus_digest(path), "ba7816bf8f01")


class BaselinesCliTest(unittest.TestCase):
    def run_cli(self, out: Path, *flags: str) -> str:
        stdout = io.StringIO()
        argv = ["--target", "comm_bytes_total", "--data", str(FIXTURE), "--output", str(out), "--quiet"]
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = cli.main([*argv, *TINY_FLAGS, *flags])
        self.assertEqual(code, 0)
        return stdout.getvalue()

    def test_writes_the_comparison_and_the_notes_beside_the_equation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            stdout = self.run_cli(out, "--baselines")
            folder = out / "flexfl" / "comm_bytes_total"
            table = pl.read_csv(folder / "baseline_comparison.csv")
            notes = (folder / "baseline_notes.txt").read_text(encoding="utf-8")
        self.assertEqual(models(table), E3 | LEARNED)
        self.assertEqual(notes, SKIPPED + "\n")
        self.assertIn("baselines written to", stdout)

    def test_forwards_the_configuration_and_the_fitted_report(self) -> None:
        comparison = BaselineComparison(pl.DataFrame({"model": ["mean"]}), ())
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            cli, "run_baselines", return_value=comparison
        ) as called:
            self.run_cli(Path(directory), "--baselines", "--opaque.boosting_stages", "7")
        called.assert_called_once()
        args, kwargs = called.call_args
        self.assertEqual(args[0], str(FIXTURE))
        self.assertEqual(args[1].target_column, "comm_bytes_total")
        self.assertEqual(args[2].max_terms, 2)
        self.assertEqual(getattr(args[4], "boosting_stages", None), 7)
        self.assertIsInstance(kwargs["report"], experiment.FlexFLReport)

    def test_the_flag_is_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(cli, "run_baselines") as called:
            out = Path(directory)
            self.run_cli(out)
            self.assertFalse((out / "flexfl" / "comm_bytes_total" / "baseline_comparison.csv").exists())
        called.assert_not_called()

    def test_the_mcc_target_rejects_the_flag(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            cli.main(["--baselines"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
