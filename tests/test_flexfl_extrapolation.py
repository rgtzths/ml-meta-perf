"""The extrapolation protocols: worker-count extrapolation and leave one distribution out."""

from __future__ import annotations

import dataclasses
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import polars as pl

from ml_meta_perf import cli, experiment
from ml_meta_perf import flexfl_baselines as baselines
from ml_meta_perf.config import load_config
from ml_meta_perf.data import columns_as_arrays, flexfl_schema, load, target
from ml_meta_perf.fit import Standardizer
from ml_meta_perf.flexfl_baselines import (
    ANALYTIC_COLUMNS,
    ORACLE,
    PREDICTED,
    BaselineComparison,
    Fold,
    Protocol,
    analytic_predictions,
    epoch_predictions,
    extrapolation_protocols,
    fold_boosting,
    fold_fixed_form,
    fold_mean,
    fold_search,
    grouped,
    held_out,
    out_of_fold,
    run_baselines,
)
from ml_meta_perf.validate import score
from tests.test_flexfl_baselines import cell, models, with_analytic_columns
from tests.test_flexfl_schema import FIXTURE, TINY, TINY_FLAGS

NEW = {
    "extrapolate_workers",
    "extrapolate_workers_unclipped",
    "loo_distribution",
    "holdout_iid",
    "holdout_non_iid",
    "holdout_dirichlet",
}
DEFAULT = {"in_sample", "loo_dataset", "loo_model"}
LEARNED = {"mean", "GradientBoosting (50 stages)"}
E3 = {"E3 (fixed form)", "E3 (form per fold)"}
STRATEGIES = ("iid", "non_iid", "dirichlet")


def with_extrapolation_rows(source: Path, directory: Path) -> Path:
    """The CSV at ``source`` plus a copy of its rows at three times the workers, iid relabelled dirichlet."""
    frame = pl.read_csv(source)
    larger = frame.with_columns(
        pl.col("num_workers") * 3,
        pl.col("n_workers_joined") * 3,
        pl.col("comm_bytes_total") * 1.7,
        pl.col("total_time_s") * 1.3,
        pl.when(pl.col("strategy") == "iid").then(pl.lit("dirichlet")).otherwise(pl.col("strategy")).alias("strategy"),
    )
    combined = pl.concat([frame, larger.select(frame.columns)], how="vertical_relaxed").with_columns(
        [(pl.col("strategy") == name).cast(pl.Int64).alias(f"strategy_{name}") for name in STRATEGIES]
    )
    path = directory / "with_extrapolation.csv"
    combined.write_csv(path)
    return path


class ProtocolsTest(unittest.TestCase):
    def test_the_worker_fold_trains_up_to_34_and_tests_36_to_56(self) -> None:
        frame = pl.DataFrame({"num_workers": [6.0, 32.0, 34.0, 35.0, 36.0, 56.0, 58.0]})
        protocols, notes = extrapolation_protocols(frame)
        clipped, unclipped = protocols["extrapolate_workers"], protocols["extrapolate_workers_unclipped"]
        np.testing.assert_array_equal(clipped.folds[0].train, [True, True, True, False, False, False, False])
        np.testing.assert_array_equal(clipped.folds[0].test, [False, False, False, False, True, True, False])
        self.assertEqual((clipped.clip, unclipped.clip), (True, False))
        self.assertEqual(unclipped.folds, clipped.folds)
        self.assertEqual(notes, ["distribution protocols skipped: the corpus has no strategy column"])

    def test_each_distribution_is_held_out_from_the_other_two(self) -> None:
        strategy = ["iid", "non_iid", "dirichlet", "iid", None, "other"]
        frame = pl.DataFrame({"num_workers": [6.0] * 6, "strategy": strategy})
        protocols, notes = extrapolation_protocols(frame)
        self.assertEqual(set(protocols), {"loo_distribution", "holdout_iid", "holdout_non_iid", "holdout_dirichlet"})
        iid = protocols["holdout_iid"].folds[0]
        np.testing.assert_array_equal(iid.train, [False, True, True, False, False, False])
        np.testing.assert_array_equal(iid.test, [True, False, False, True, False, False])
        pooled = protocols["loo_distribution"]
        self.assertEqual(len(pooled.folds), 3)
        np.testing.assert_array_equal(pooled.scored, [True, True, True, True, False, False])
        self.assertEqual(
            notes,
            ["extrapolation protocol extrapolate_workers skipped: 6 runs with at most 34 workers and 0 with 36 to 56"],
        )

    def test_a_missing_distribution_skips_every_distribution_protocol(self) -> None:
        frame = pl.DataFrame({"num_workers": [6.0, 40.0], "strategy": ["iid", "non_iid"]})
        protocols, notes = extrapolation_protocols(frame)
        self.assertEqual(set(protocols), {"extrapolate_workers", "extrapolate_workers_unclipped"})
        self.assertEqual(notes, ["distribution protocols skipped: no runs with strategy dirichlet"])

    def test_an_empty_training_side_skips_the_worker_protocols(self) -> None:
        protocols, notes = extrapolation_protocols(pl.DataFrame({"num_workers": [36.0, 56.0]}))
        self.assertEqual(protocols, {})
        self.assertEqual(
            notes[0],
            "extrapolation protocol extrapolate_workers skipped: 0 runs with at most 34 workers and 2 with 36 to 56",
        )


class ProtocolValidationTest(unittest.TestCase):
    def test_a_protocol_rejects_unusable_folds(self) -> None:
        yes, no = np.array([True, False]), np.array([False, True])
        cases = {
            "at least one fold": (),
            "training and test rows": (Fold(np.zeros(2, dtype=bool), yes),),
            "one train and one test flag per row": (Fold(np.ones(3, dtype=bool), yes),),
            "more than one fold": (Fold(no, yes), Fold(no, yes)),
            "trains on a row it holds out": (Fold(np.ones(2, dtype=bool), yes),),
        }
        for message, folds in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                Protocol(folds)
        self.assertEqual(Protocol((Fold(no, yes),), clip=False).analytic, True)


class HeldOutTest(unittest.TestCase):
    truth = np.array([1.0, 2.0, 3.0, 10.0, 20.0])

    def test_a_directed_fold_scores_only_its_test_rows(self) -> None:
        fold = Fold(np.array([True, True, True, False, False]), np.array([False, False, False, True, False]))

        def constant(_train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return np.full(int(test.sum()), 100.0)

        clipped = held_out(constant, self.truth, Protocol((fold,)))
        unclipped = held_out(constant, self.truth, Protocol((fold,), clip=False))
        np.testing.assert_array_equal(clipped, [np.nan, np.nan, np.nan, 3.0, np.nan])
        np.testing.assert_array_equal(unclipped, [np.nan, np.nan, np.nan, 100.0, np.nan])

    def test_grouped_matches_leave_one_group_out(self) -> None:
        labels = np.array(["a", "a", "b", "b", "c"])
        np.testing.assert_array_equal(
            held_out(fold_mean(self.truth), self.truth, grouped(labels)),
            out_of_fold(fold_mean(self.truth), self.truth, labels),
        )
        np.testing.assert_array_equal(grouped(labels).scored, np.ones(5, dtype=bool))


class FixedFormTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = flexfl_schema("comm_bytes_total", log_target=True)
        self.report = experiment.run_flexfl(FIXTURE, self.schema, TINY)
        frame = load(FIXTURE, self.schema)
        self.truth = target(frame, self.schema)
        self.columns = columns_as_arrays(frame, self.report.schema.features)
        self.train = frame["dataset"].to_numpy() != "clf_a"

    def test_refits_the_published_terms_on_the_training_rows(self) -> None:
        equation = self.report.equation.equation
        self.assertTrue(equation.terms)
        matrix = np.column_stack([term.evaluate(self.columns) for term in equation.terms])
        train, test = self.train, ~self.train
        standardizer = Standardizer.fit(matrix[train])
        design = standardizer.apply(matrix[train])
        offset = self.truth[train].mean()
        gram = design.T @ design + TINY.penalty * np.eye(len(equation.terms))
        weights = np.linalg.solve(gram, design.T @ (self.truth[train] - offset))
        expected = standardizer.apply(matrix[test]) @ weights + offset
        fit = fold_fixed_form(equation, self.columns, self.truth, TINY.penalty)
        np.testing.assert_allclose(fit(train, test), expected, rtol=1e-12)
        moved = self.truth.copy()
        moved[test] += 50.0
        np.testing.assert_array_equal(
            fold_fixed_form(equation, self.columns, moved, TINY.penalty)(train, test), fit(train, test)
        )

    def test_an_equation_without_terms_predicts_the_training_mean(self) -> None:
        empty = dataclasses.replace(self.report.equation.equation, terms=(), weights=(), standardized_weights=())
        fit = fold_fixed_form(empty, self.columns, self.truth, TINY.penalty)
        np.testing.assert_array_equal(
            fit(self.train, ~self.train), np.full(int((~self.train).sum()), self.truth[self.train].mean())
        )


class LoadLabelsTest(unittest.TestCase):
    def test_label_columns_are_strings_kept_after_the_extras_without_a_null_check(self) -> None:
        schema = flexfl_schema("performance", "classification")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.csv"
            raw = pl.read_csv(FIXTURE).with_columns(
                pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("strategy")).alias("strategy")
            )
            raw.write_csv(path)
            frame = load(
                path,
                schema,
                extra_columns=("n_epochs",),
                label_columns=("strategy", "absent", "strategy", "n_workers_joined"),
            )
            plain = load(path, schema)
        self.assertEqual(frame.columns, [*plain.columns, "n_epochs", "strategy", "n_workers_joined"])
        self.assertEqual(frame["strategy"].dtype, pl.String)
        self.assertEqual(frame["n_workers_joined"].dtype, pl.String)
        joined = raw.filter(pl.col("is_classification").cast(pl.Boolean))["n_workers_joined"]
        self.assertEqual(frame["n_workers_joined"].to_list(), [str(value) for value in joined.to_list()])
        expected = raw.filter(pl.col("is_classification").cast(pl.Boolean))["strategy"]
        self.assertEqual(frame["strategy"].to_list(), expected.to_list())
        self.assertTrue(frame.select(plain.columns).equals(plain))


class EpochProtocolTest(unittest.TestCase):
    def test_a_directed_protocol_leaves_unscored_rows_empty(self) -> None:
        schema = flexfl_schema("n_epochs")
        frame = load(FIXTURE, schema)
        truth = target(frame, schema)
        rows = frame.height
        report = SimpleNamespace(
            schema=schema,
            equation=SimpleNamespace(n_terms=1, equation=SimpleNamespace(predict=lambda _c: np.full(rows, 3.0))),
        )
        fold = Fold(np.arange(rows) < 12, np.arange(rows) >= 20)

        def seven(_train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return np.full(int(test.sum()), 7.0)

        with mock.patch.object(baselines, "fold_search", return_value=seven):
            predicted = epoch_predictions(report, frame, truth, {"workers": Protocol((fold,))}, TINY)  # pyright: ignore[reportArgumentType]
        expected = np.full(rows, np.nan)
        expected[20:] = np.clip(7.0, truth[:12].min(), truth[:12].max())
        np.testing.assert_array_equal(predicted["workers"], np.minimum(expected, frame["epoch_cap"].to_numpy()))


class RunBaselinesExtrapolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = with_extrapolation_rows(FIXTURE, Path(self.directory.name))
        self.schema = flexfl_schema("comm_bytes_total", log_target=True)
        self.frame = load(self.path, self.schema, extra_columns=ANALYTIC_COLUMNS, label_columns=("strategy",))
        self.truth = target(self.frame, self.schema)

    def test_every_model_is_scored_on_each_protocol_held_out_rows(self) -> None:
        report = experiment.run_flexfl(self.path, self.schema, TINY)
        comparison = run_baselines(self.path, self.schema, TINY, report=report, extrapolation=True)
        table = comparison.table
        self.assertEqual(set(table["protocol"]), DEFAULT | NEW)
        for protocol in NEW:
            self.assertEqual(models(table.filter(pl.col("protocol") == protocol)), E3 | LEARNED)
        counts = {
            "extrapolate_workers": 12,
            "holdout_iid": 12,
            "holdout_non_iid": 24,
            "holdout_dirichlet": 12,
            "loo_distribution": 48,
        }
        for protocol, n in counts.items():
            self.assertEqual(set(table.filter(pl.col("protocol") == protocol)["n"]), {n})
        workers = self.frame["num_workers"].to_numpy()
        train, test = workers <= 34, workers >= 36
        truth = self.truth
        bounds = (truth[train].min(), truth[train].max())
        per_fold = fold_search(self.frame, report.schema, truth, report.equation.n_terms, TINY)(train, test)
        self.assertAlmostEqual(
            cell(table, "E3 (form per fold)", "extrapolate_workers", "r2"),
            score(truth[test], np.clip(per_fold, *bounds)).r2,
            places=12,
        )
        self.assertAlmostEqual(
            cell(table, "E3 (form per fold)", "extrapolate_workers_unclipped", "r2"),
            score(truth[test], per_fold).r2,
            places=12,
        )
        self.assertAlmostEqual(
            cell(table, "mean", "extrapolate_workers", "mae"),
            score(truth[test], np.full(int(test.sum()), truth[train].mean())).mae,
            places=12,
        )
        strategy = self.frame["strategy"].to_numpy()
        held = strategy == "dirichlet"
        features = columns_as_arrays(self.frame, report.schema.features)
        fixed = fold_fixed_form(report.equation.equation, features, truth, TINY.penalty)(~held, held)
        self.assertAlmostEqual(
            cell(table, "E3 (fixed form)", "holdout_dirichlet", "r2"),
            score(truth[held], np.clip(fixed, truth[~held].min(), truth[~held].max())).r2,
            places=12,
        )
        design = np.column_stack([features[name] for name in report.schema.features])
        pooled = held_out(fold_boosting(design, truth, load_config().opaque), truth, grouped(strategy))
        self.assertAlmostEqual(
            cell(table, "GradientBoosting (50 stages)", "loo_distribution", "r2"), score(truth, pooled).r2, places=12
        )
        self.assertEqual(
            cell(table, "E3 (fixed form)", "loo_dataset", "r2"), report.equation.cross_validated["loo_dataset"]["r2"]
        )
        self.assertEqual(
            comparison.notes,
            (
                "analytic baseline skipped: the corpus has no min_workers, pool_batches; "
                "re-assemble it with FlexFL's scripts/assemble_meta_dataset.py",
            ),
        )

    def test_the_unclipped_twin_scores_raw_predictions(self) -> None:
        def far(_train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return np.full(int(test.sum()), 1000.0)

        with mock.patch.object(baselines, "fold_search", return_value=far):
            table = run_baselines(self.path, self.schema, TINY, extrapolation=True).table
        workers = self.frame["num_workers"].to_numpy()
        train, test = workers <= 34, workers >= 36
        truth = self.truth
        clipped = np.full(int(test.sum()), truth[train].max())
        self.assertAlmostEqual(
            cell(table, "E3 (form per fold)", "extrapolate_workers", "mae"), score(truth[test], clipped).mae, places=12
        )
        self.assertAlmostEqual(
            cell(table, "E3 (form per fold)", "extrapolate_workers_unclipped", "mae"),
            score(truth[test], np.full(int(test.sum()), 1000.0)).mae,
            places=12,
        )

    def test_the_protocols_e3_reports_copy_its_own_scores(self) -> None:
        report = experiment.run_flexfl(self.path, self.schema, TINY)
        sentinel = {label: {"r2": 0.25, "mae": 0.5, "n": 48} for label in report.equation.cross_validated}
        report = dataclasses.replace(report, equation=dataclasses.replace(report.equation, cross_validated=sentinel))
        table = run_baselines(self.path, self.schema, TINY, report=report, extrapolation=True).table
        for protocol in ("loo_dataset", "loo_model"):
            self.assertEqual(cell(table, "E3 (fixed form)", protocol, "r2"), 0.25)
        self.assertNotEqual(cell(table, "E3 (fixed form)", "loo_distribution", "r2"), 0.25)

    def test_the_predicted_analytic_rows_use_each_protocol_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = with_extrapolation_rows(with_analytic_columns(Path(directory)), Path(directory))
            frame = load(path, self.schema, extra_columns=ANALYTIC_COLUMNS, label_columns=("strategy",))
            rows = frame.height
            epochs = {"in_sample": np.full(rows, 2.0)} | {
                label: np.full(rows, 3.0 + index)
                for index, label in enumerate(
                    sorted(NEW - {"extrapolate_workers_unclipped"} | {"loo_dataset", "loo_model"})
                )
            }
            with mock.patch.object(baselines, "epoch_predictions", return_value=epochs):
                table = run_baselines(path, self.schema, TINY, extrapolation=True).table
        columns, _ = baselines._analytic_inputs(frame, self.schema)  # pyright: ignore[reportPrivateUsage]
        assert columns is not None
        truth = target(frame, self.schema)
        protocols, _ = extrapolation_protocols(frame)
        for label, values in epochs.items():
            scored = protocols[label].scored if label in protocols else np.ones(rows, dtype=bool)
            predicted = analytic_predictions(columns, values, True)
            self.assertAlmostEqual(
                cell(table, PREDICTED, label, "r2"), score(truth[scored], predicted[scored]).r2, places=12
            )

    def test_an_extra_split_gets_the_refit_fixed_form(self) -> None:
        schema = flexfl_schema("comm_bytes_total")
        report = experiment.run_flexfl(FIXTURE, schema, TINY)
        frame = load(FIXTURE, schema)
        workers = frame["num_workers"].to_numpy()
        halves = np.where(workers > np.median(workers), "big", "small")
        table = run_baselines(FIXTURE, schema, TINY, report=report, splits={"by_size": halves}).table
        truth = target(frame, schema)
        features = columns_as_arrays(frame, report.schema.features)
        fixed = fold_fixed_form(report.equation.equation, features, truth, TINY.penalty)
        expected = held_out(fixed, truth, grouped(halves))
        self.assertAlmostEqual(cell(table, "E3 (fixed form)", "by_size", "r2"), score(truth, expected).r2, places=12)
        self.assertNotAlmostEqual(cell(table, "E3 (fixed form)", "by_size", "r2"), cell(table, "mean", "by_size", "r2"))

    def test_the_unclipped_twin_stays_within_a_bounded_target(self) -> None:
        schema = flexfl_schema("performance", "classification")
        bounds = schema.bounds
        assert bounds is not None

        def far(_train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return np.full(int(test.sum()), 5.0)

        with mock.patch.object(baselines, "fold_search", return_value=far):
            table = run_baselines(self.path, schema, TINY, extrapolation=True).table
        frame = load(self.path, schema)
        truth = target(frame, schema)
        test = frame["num_workers"].to_numpy() >= 36
        expected = score(truth[test], np.full(int(test.sum()), bounds[1])).mae
        self.assertAlmostEqual(
            cell(table, "E3 (form per fold)", "extrapolate_workers_unclipped", "mae"), expected, places=12
        )

    def test_reserved_names_hold_when_their_protocols_are_skipped(self) -> None:
        frame = load(FIXTURE, flexfl_schema("comm_bytes_total"))
        halves = np.where(frame["num_workers"].to_numpy() > 10, "a", "b")
        for name in ("holdout_dirichlet", "extrapolate_workers"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "reserved for the extrapolation protocols"),
            ):
                run_baselines(
                    FIXTURE, flexfl_schema("comm_bytes_total"), TINY, splits={name: halves}, extrapolation=True
                )

    def test_without_the_flag_the_table_is_unchanged(self) -> None:
        table = run_baselines(self.path, self.schema, TINY).table
        self.assertEqual(set(table["protocol"]), DEFAULT)

    def test_a_split_may_not_take_an_extrapolation_name(self) -> None:
        halves = np.where(self.frame["num_workers"].to_numpy() > 20, "a", "b")
        with self.assertRaisesRegex(ValueError, r"\['holdout_iid'\] are reserved for the extrapolation protocols"):
            run_baselines(self.path, self.schema, TINY, splits={"holdout_iid": halves}, extrapolation=True)

    def test_a_corpus_without_the_protocols_skips_them_with_notes(self) -> None:
        comparison = run_baselines(FIXTURE, flexfl_schema("comm_bytes_total"), TINY, extrapolation=True)
        self.assertEqual(set(comparison.table["protocol"]), DEFAULT)
        self.assertEqual(
            comparison.notes[:2],
            (
                "extrapolation protocol extrapolate_workers skipped: "
                "24 runs with at most 34 workers and 0 with 36 to 56",
                "distribution protocols skipped: no runs with strategy dirichlet",
            ),
        )

    def test_the_analytic_rows_cover_the_clipped_protocols_on_their_held_out_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            analytic = with_analytic_columns(Path(directory))
            path = with_extrapolation_rows(analytic, Path(directory))
            comparison = run_baselines(path, self.schema, TINY, extrapolation=True)
            frame = load(path, self.schema, extra_columns=ANALYTIC_COLUMNS, label_columns=("strategy",))
        table = comparison.table
        self.assertEqual(comparison.notes, ())
        for model in (ORACLE, PREDICTED):
            self.assertEqual(
                set(table.filter(pl.col("model") == model)["protocol"]),
                DEFAULT | NEW - {"extrapolate_workers_unclipped"},
            )
        columns, _ = baselines._analytic_inputs(frame, self.schema)  # pyright: ignore[reportPrivateUsage]
        assert columns is not None
        oracle = analytic_predictions(columns, columns["n_epochs"], True)
        test = frame["num_workers"].to_numpy() >= 36
        truth = target(frame, self.schema)
        self.assertAlmostEqual(
            cell(table, ORACLE, "extrapolate_workers", "r2"), score(truth[test], oracle[test]).r2, places=12
        )
        self.assertEqual(set(table.filter(pl.col("protocol") == "extrapolate_workers")["n"]), {int(test.sum())})


class ExtrapolationCliTest(unittest.TestCase):
    def run_cli(self, *flags: str, data: Path = FIXTURE, out: Path | None = None) -> tuple[str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = ["--target", "comm_bytes_total", "--data", str(data), "--quiet"]
        argv += ["--output", str(out)] if out is not None else ["--output", ""]
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main([*argv, *TINY_FLAGS, *flags])
        self.assertEqual(code, 0)
        return stdout.getvalue(), stderr.getvalue()

    def test_the_flag_implies_the_baselines_and_writes_the_new_protocols(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = with_extrapolation_rows(FIXTURE, Path(directory))
            out = Path(directory) / "out"
            self.run_cli("--extrapolation", data=data, out=out)
            table = pl.read_csv(out / "flexfl" / "comm_bytes_total" / "baseline_comparison.csv")
        self.assertEqual(set(table["protocol"]), DEFAULT | NEW)

    def test_forwards_the_flag(self) -> None:
        comparison = BaselineComparison(pl.DataFrame({"model": ["mean"]}), ())
        for flags, expected in ((("--extrapolation",), True), (("--baselines",), False)):
            with self.subTest(flags=flags), mock.patch.object(cli, "run_baselines", return_value=comparison) as called:
                self.run_cli(*flags)
            self.assertIs(called.call_args.kwargs["extrapolation"], expected)

    def test_the_mcc_target_rejects_the_flag(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            cli.main(["--extrapolation"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("--extrapolation applies only to a FlexFL --target", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
