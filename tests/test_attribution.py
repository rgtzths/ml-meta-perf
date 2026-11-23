"""Term effects, group shares and the identity-based variance decomposition."""

import unittest

import numpy as np
import polars as pl

from ml_meta_perf.attribution import (
    DATASET_ONLY,
    MIXED,
    MODEL_ONLY,
    block_effects,
    block_shares,
    classify,
    classify_blocks,
    contributions,
    group_shares,
    term_effects,
    variance_decomposition,
)
from ml_meta_perf.data import FLEXFL_BLOCKS
from ml_meta_perf.model import LOWERS, RAISES, Equation
from ml_meta_perf.terms import Atom, Term

DATASET = ("d1", "d2")
MODEL = ("m1", "m2")


def equation() -> Equation:
    return Equation(
        intercept=0.5,
        terms=(
            Term("atom", (Atom("d1"),)),
            Term("atom", (Atom("m1"),)),
            Term("product", (Atom("d2"), Atom("m2"))),
        ),
        weights=(0.10, -0.20, 0.05),
        standardized_weights=(0.3, -0.5, 0.2),
    )


def columns(n: int = 60, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.uniform(1.0, 5.0, n) for name in (*DATASET, *MODEL)}


class TestClassify(unittest.TestCase):
    def test_labels_each_group(self) -> None:
        self.assertEqual(classify(("d1",), DATASET, MODEL), DATASET_ONLY)
        self.assertEqual(classify(("m1",), DATASET, MODEL), MODEL_ONLY)
        self.assertEqual(classify(("d1", "m2"), DATASET, MODEL), MIXED)

    def test_unknown_feature_defaults_to_dataset(self) -> None:
        self.assertEqual(classify(("other",), DATASET, MODEL), DATASET_ONLY)


class TestContributions(unittest.TestCase):
    def test_shape_matches_terms(self) -> None:
        data = columns()
        self.assertEqual(contributions(equation(), data).shape, (60, 3))

    def test_rows_sum_to_the_prediction_minus_intercept(self) -> None:
        data = columns()
        eq = equation()
        np.testing.assert_allclose(
            contributions(eq, data).sum(axis=1) + eq.intercept, eq.evaluate(data), rtol=1e-9
        )

    def test_empty_equation_gives_an_empty_matrix(self) -> None:
        empty = Equation(intercept=0.2, terms=(), weights=(), standardized_weights=())
        self.assertEqual(contributions(empty, columns()).shape, (60, 0))


class TestTermEffects(unittest.TestCase):
    def setUp(self) -> None:
        self.table = term_effects(equation(), columns(), DATASET, MODEL)

    def test_one_row_per_term(self) -> None:
        self.assertEqual(self.table.height, 3)

    def test_sorted_by_absolute_effect(self) -> None:
        effects = np.abs(self.table["effect"].to_numpy())
        self.assertTrue((np.diff(effects) <= 1e-12).all())

    def test_effects_are_non_negative_spans(self) -> None:
        self.assertTrue((self.table["effect"].to_numpy() >= 0.0).all())

    def test_direction_follows_the_standardised_weight(self) -> None:
        for row in self.table.iter_rows(named=True):
            expected = RAISES if row["beta"] > 0 else LOWERS
            self.assertEqual(row["direction"], expected)

    def test_groups_are_labelled(self) -> None:
        self.assertEqual(set(self.table["group"].to_list()), {DATASET_ONLY, MODEL_ONLY, MIXED})


class TestGroupShares(unittest.TestCase):
    def test_shares_sum_to_one(self) -> None:
        table = group_shares(equation(), columns(), DATASET, MODEL)
        self.assertAlmostEqual(float(table["share"].sum()), 1.0, places=6)

    def test_every_group_is_present_even_when_unused(self) -> None:
        only_dataset = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("d1"),)),),
            weights=(1.0,),
            standardized_weights=(1.0,),
        )
        table = group_shares(only_dataset, columns(), DATASET, MODEL)
        self.assertEqual(table.height, 3)
        counts = dict(zip(table["group"].to_list(), table["n_terms"].to_list(), strict=True))
        self.assertEqual(counts[MODEL_ONLY], 0)
        self.assertEqual(counts[DATASET_ONLY], 1)

    def test_a_single_group_takes_the_whole_share(self) -> None:
        only_dataset = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("d1"),)),),
            weights=(1.0,),
            standardized_weights=(1.0,),
        )
        table = group_shares(only_dataset, columns(), DATASET, MODEL)
        shares = dict(zip(table["group"].to_list(), table["share"].to_list(), strict=True))
        self.assertAlmostEqual(shares[DATASET_ONLY], 1.0, places=6)


def flexfl_equation() -> Equation:
    return Equation(
        intercept=0.5,
        terms=(
            Term("atom", (Atom("n_samples"),)),
            Term("atom", (Atom("total_parameters"),)),
            Term("atom", (Atom("learning_rate"),)),
            Term("atom", (Atom("num_workers"),)),
            Term("product", (Atom("n_samples"), Atom("num_workers"))),
        ),
        weights=(0.10, -0.20, 0.05, 0.15, -0.03),
        standardized_weights=(0.3, -0.5, 0.2, 0.4, -0.1),
    )


def flexfl_columns(n: int = 60, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "n_samples": rng.uniform(1.0, 5.0, n),
        "total_parameters": rng.uniform(1.0, 5.0, n),
        "learning_rate": rng.uniform(1.0, 5.0, n),
        "num_workers": rng.uniform(1.0, 5.0, n),
    }


class TestFlexFLBlocks(unittest.TestCase):
    def test_classifies_single_two_and_three_blocks_in_plan_order(self) -> None:
        self.assertEqual(tuple(FLEXFL_BLOCKS), ("dataset", "architecture", "fl_config", "federation"))
        self.assertEqual(classify_blocks(("n_samples",)), ("dataset", ("dataset",)))
        self.assertEqual(
            classify_blocks(("num_workers", "n_samples")),
            (MIXED, ("dataset", "federation")),
        )
        self.assertEqual(
            classify_blocks(("num_workers", "learning_rate", "n_samples")),
            (MIXED, ("dataset", "fl_config", "federation")),
        )

    def test_empty_and_unknown_features_raise(self) -> None:
        with self.assertRaises(ValueError):
            classify_blocks(())
        with self.assertRaisesRegex(ValueError, "unknown"):
            classify_blocks(("nope",))

    def test_block_effects_report_spans_and_weighted_percentile_ranges(self) -> None:
        equation = flexfl_equation()
        data = flexfl_columns()
        table = block_effects(equation, data)
        self.assertEqual(
            table.columns,
            ["term", "group", "blocks", "weight", "beta", "effect", "direction"],
        )
        by_term = {row["term"]: row for row in table.iter_rows(named=True)}
        expected_groups = {
            "n_samples": ("dataset", "dataset"),
            "total_parameters": ("architecture", "architecture"),
            "learning_rate": ("fl_config", "fl_config"),
            "num_workers": ("federation", "federation"),
            "[n_samples] * [num_workers]": (MIXED, "dataset+federation"),
        }
        self.assertEqual(
            {term: (row["group"], row["blocks"]) for term, row in by_term.items()},
            expected_groups,
        )
        matrix = contributions(equation, data)
        product_index = next(
            index
            for index, term in enumerate(equation.terms)
            if set(term.features) == {"n_samples", "num_workers"}
        )
        weighted = matrix[:, product_index]
        low, high = np.percentile(weighted, [10.0, 90.0])
        product_name = equation.terms[product_index].name
        self.assertAlmostEqual(by_term[product_name]["effect"], float(high - low))
        effects = np.abs(table["effect"].to_numpy())
        self.assertTrue((np.diff(effects) <= 1e-12).all())

    def test_block_shares_are_partitioned_and_sum_to_one(self) -> None:
        table = block_shares(flexfl_equation(), flexfl_columns())
        self.assertEqual(
            table["group"].to_list(),
            ["dataset", "architecture", "fl_config", "federation", MIXED],
        )
        self.assertAlmostEqual(float(table["share"].sum()), 1.0, delta=1e-9)
        self.assertEqual(table["n_terms"].to_list(), [1, 1, 1, 1, 1])
        only_dataset = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("n_samples"),)),),
            weights=(1.0,),
            standardized_weights=(1.0,),
        )
        unused = block_shares(only_dataset, flexfl_columns())
        self.assertEqual(unused.filter(pl.col("group") == "architecture")["share"].item(), 0.0)

    def test_single_block_takes_all_share_and_empty_equation_has_explicit_shapes(self) -> None:
        only_dataset = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("n_samples"),)),),
            weights=(1.0,),
            standardized_weights=(1.0,),
        )
        single = block_shares(only_dataset, flexfl_columns())
        self.assertEqual(single.filter(pl.col("group") == "dataset")["share"].item(), 1.0)

        empty = Equation(intercept=0.2, terms=(), weights=(), standardized_weights=())
        effects = block_effects(empty, flexfl_columns())
        self.assertEqual(
            effects.columns,
            ["term", "group", "blocks", "weight", "beta", "effect", "direction"],
        )
        self.assertEqual(effects.height, 0)
        shares = block_shares(empty, flexfl_columns())
        self.assertEqual(shares["group"].to_list(), ["dataset", "architecture", "fl_config", "federation", MIXED])
        self.assertEqual(shares["share"].to_list(), [0.0] * 5)
        self.assertEqual(shares["n_terms"].to_list(), [0] * 5)
        self.assertEqual(float(shares["share"].sum()), 0.0)


class TestVarianceDecomposition(unittest.TestCase):
    def test_perfectly_separated_groups_explain_everything(self) -> None:
        target = np.array([0.1, 0.1, 0.9, 0.9])
        datasets = np.array(["a", "a", "b", "b"])
        models = np.array(["x", "y", "x", "y"])
        table = variance_decomposition(target, datasets, models)
        scores = dict(zip(table["knowing only"].to_list(), table["variance_explained"].to_list(), strict=True))
        self.assertAlmostEqual(scores["dataset identity"], 1.0)
        self.assertAlmostEqual(scores["model identity"], 0.0)

    def test_counts_the_groups(self) -> None:
        table = variance_decomposition(
            np.array([0.1, 0.2, 0.3]), np.array(["a", "b", "c"]), np.array(["x", "x", "y"])
        )
        counts = dict(zip(table["knowing only"].to_list(), table["n_groups"].to_list(), strict=True))
        self.assertEqual(counts["dataset identity"], 3)
        self.assertEqual(counts["model identity"], 2)

    def test_constant_target_is_zero_not_nan(self) -> None:
        table = variance_decomposition(np.ones(4), np.array(["a", "a", "b", "b"]), np.array(["x", "y", "x", "y"]))
        self.assertTrue(np.all(np.isfinite(table["variance_explained"].to_numpy())))

    def test_real_data_puts_dataset_above_model(self) -> None:
        # The empirical answer to "does model choice matter more than the data".
        from ml_meta_perf.data import DATASET_COLUMN, MODEL_COLUMN, groups, load, target

        frame = load()
        table = variance_decomposition(
            target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)
        )
        scores = dict(zip(table["knowing only"].to_list(), table["variance_explained"].to_list(), strict=True))
        self.assertGreater(scores["dataset identity"], scores["model identity"])


if __name__ == "__main__":
    unittest.main()
