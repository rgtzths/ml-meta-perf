"""Extraction of written practices, and the filters that keep unsupported ones out."""

import unittest
from unittest import mock

import numpy as np
import polars as pl

from ml_meta_perf.model import Equation
from ml_meta_perf.practices import (
    best_practices,
    concordance,
    concordance_summary,
    feature_evidence,
    feature_practices,
    render,
    term_frequencies,
)
from ml_meta_perf.terms import Atom, Term


def rising_equation() -> Equation:
    """MCC increases with ``up`` and decreases with ``down``."""
    return Equation(
        intercept=0.5,
        terms=(Term("atom", (Atom("up"),)), Term("atom", (Atom("down"),))),
        weights=(0.10, -0.08),
        standardized_weights=(0.6, -0.4),
    )


def columns(n: int = 100) -> dict[str, np.ndarray]:
    return {"up": np.linspace(1.0, 5.0, n), "down": np.linspace(1.0, 5.0, n)[::-1].copy()}


def stability_table(**frequencies: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "term": list(frequencies),
            "folds": [round(20 * value) for value in frequencies.values()],
            "frequency": list(frequencies.values()),
        }
    )


class TestFeaturePractices(unittest.TestCase):
    def setUp(self) -> None:
        self.table = feature_practices(rising_equation(), columns())

    def test_one_row_per_feature(self) -> None:
        self.assertEqual(sorted(self.table["feature"].to_list()), ["down", "up"])

    def test_direction_matches_the_relationship(self) -> None:
        rows = {row["feature"]: row for row in self.table.iter_rows(named=True)}
        self.assertGreater(rows["up"]["direction"], 0.99)
        self.assertLess(rows["down"]["direction"], -0.99)

    def test_effect_is_signed_and_follows_the_feature(self) -> None:
        rows = {row["feature"]: row for row in self.table.iter_rows(named=True)}
        # up rises from 1 to 5 with weight 0.10, so its top decile contributes more.
        self.assertGreater(rows["up"]["effect"], 0.0)
        self.assertLess(rows["down"]["effect"], 0.0)

    def test_effect_measures_response_not_spread(self) -> None:
        # A quantile of the contributions alone would be order-invariant and identical
        # for both features; ordering by the feature is what makes the sign meaningful.
        effects = self.table["effect"].to_numpy()
        self.assertNotAlmostEqual(float(effects[0]), float(effects[1]))

    def test_glossary_is_used_when_available(self) -> None:
        equation = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("nr_class"),)),),
            weights=(1.0,),
            standardized_weights=(1.0,),
        )
        table = feature_practices(equation, {"nr_class": np.linspace(2.0, 9.0, 30)})
        self.assertIn("classes", table["meaning"][0])

    def test_stability_is_nan_without_a_table(self) -> None:
        self.assertTrue(np.isnan(self.table["stability"].to_numpy()).all())

    def test_stability_is_read_from_the_table(self) -> None:
        table = feature_practices(rising_equation(), columns(), stability_table(up=0.9, down=0.4))
        rows = {row["feature"]: row for row in table.iter_rows(named=True)}
        self.assertAlmostEqual(rows["up"]["stability"], 0.9)
        self.assertAlmostEqual(rows["down"]["stability"], 0.4)

    def test_missing_term_in_a_non_empty_stability_table_counts_as_zero(self) -> None:
        equation = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("up"),)), Term("atom", (Atom("up", "sq"),))),
            weights=(0.1, 0.1),
            standardized_weights=(0.5, 0.5),
        )
        table = feature_practices(equation, {"up": columns()["up"]}, stability_table(up=0.8))
        self.assertAlmostEqual(table["stability"][0], 0.4)


class TestFeatureEvidence(unittest.TestCase):
    def test_term_frequencies_use_dataset_count_and_keep_never_reselected_terms(self) -> None:
        effects = pl.DataFrame(
            {
                "term": ["selected", "never"],
                "weight": [0.4, 0.2],
                "beta": [0.3, 0.1],
                "effect": [0.08, 0.04],
                "direction": [0.9, 0.8],
            }
        )
        stability = pl.DataFrame({"term": ["selected"], "folds": [7], "frequency": [1.0]})

        table = term_frequencies(effects, stability, searches=8)
        rows = {row["term"]: row for row in table.iter_rows(named=True)}
        self.assertEqual(
            set(table.columns),
            {"term", "weight", "beta", "effect", "direction", "folds", "searches", "frequency"},
        )
        self.assertEqual((rows["selected"]["folds"], rows["selected"]["searches"]), (7, 8))
        self.assertEqual(rows["selected"]["frequency"], 0.875)
        self.assertEqual((rows["never"]["folds"], rows["never"]["frequency"]), (0, 0.0))

    def test_a_decreasing_feature_passes_with_a_negative_difference(self) -> None:
        table = feature_evidence(
            rising_equation(),
            columns(),
            stability_table(up=1.0, down=1.0),
            margin=0.05,
            min_direction=0.15,
            min_frequency=0.5,
        )
        rows = {row["feature"]: row for row in table.iter_rows(named=True)}
        self.assertAlmostEqual(rows["up"]["decile_difference"], 4 / 11)
        self.assertAlmostEqual(rows["down"]["decile_difference"], -3.2 / 11)
        self.assertLess(rows["down"]["spearman"], -0.99)
        for feature in ("up", "down"):
            self.assertTrue(rows[feature]["pass_margin"])
            self.assertTrue(rows[feature]["pass_direction"])
            self.assertTrue(rows[feature]["pass_sign"])
            self.assertTrue(rows[feature]["pass_frequency"])
            self.assertTrue(rows[feature]["passes_all"])

    def test_feature_filters_report_each_evidence_gate(self) -> None:
        equation = Equation(
            intercept=0.0,
            terms=(
                Term("atom", (Atom("up"),)),
                Term("atom", (Atom("small"),)),
                Term("atom", (Atom("v", "sq"),)),
                Term("atom", (Atom("f"),)),
                Term("atom", (Atom("f", "sq"),)),
                Term("atom", (Atom("weak"),)),
                Term("atom", (Atom("never"),)),
            ),
            weights=(0.1, 0.001, 0.2, -1.0, 0.3, 0.1, 0.1),
            standardized_weights=(0.6, 0.1, 0.8, -0.5, 0.5, 0.6, 0.6),
        )
        data = columns()
        data.update(
            {
                "small": np.linspace(1.0, 5.0, 100),
                "v": np.linspace(-3.0, 3.0, 100),
                "f": np.concatenate([np.linspace(0.0, 0.9, 90), np.linspace(4.0, 5.0, 10)]),
                "weak": np.linspace(1.0, 5.0, 100),
                # The shipped fixture fit does not produce a zero-frequency term naturally, so this case is synthetic.
                "never": np.linspace(1.0, 5.0, 100),
            }
        )
        stability = pl.DataFrame(
            {
                "term": ["up", "small", "v^2", "f", "f^2", "weak"],
                "folds": [8, 8, 8, 8, 8, 3],
                "frequency": [1.0, 1.0, 1.0, 1.0, 1.0, 0.375],
            }
        )

        table = feature_evidence(
            equation,
            data,
            stability,
            margin=0.05,
            min_direction=0.15,
            min_frequency=0.5,
        )
        rows = {row["feature"]: row for row in table.iter_rows(named=True)}
        self.assertEqual(
            table.columns,
            [
                "feature",
                "meaning",
                "n_terms",
                "decile_difference",
                "spearman",
                "mean_frequency",
                "margin",
                "pass_margin",
                "pass_direction",
                "pass_sign",
                "pass_frequency",
                "passes_all",
            ],
        )
        self.assertEqual(
            (
                rows["up"]["pass_margin"],
                rows["up"]["pass_direction"],
                rows["up"]["pass_sign"],
                rows["up"]["pass_frequency"],
                rows["up"]["passes_all"],
            ),
            (True, True, True, True, True),
        )
        self.assertAlmostEqual(rows["up"]["decile_difference"], 0.36363636363636365)
        self.assertAlmostEqual(rows["up"]["spearman"], 1.0)
        self.assertAlmostEqual(rows["up"]["mean_frequency"], 1.0)
        self.assertFalse(rows["small"]["pass_margin"])
        self.assertFalse(rows["small"]["passes_all"])
        self.assertFalse(rows["v"]["pass_direction"])
        self.assertFalse(rows["v"]["passes_all"])
        self.assertFalse(rows["f"]["pass_sign"])
        self.assertFalse(rows["f"]["passes_all"])
        self.assertFalse(rows["weak"]["pass_frequency"])
        self.assertFalse(rows["weak"]["passes_all"])
        self.assertFalse(rows["never"]["pass_frequency"])
        self.assertFalse(rows["never"]["passes_all"])
        self.assertEqual(set(table["margin"].to_list()), {0.05})
        self.assertEqual(
            table["decile_difference"].abs().to_list(),
            sorted(table["decile_difference"].abs().to_list(), reverse=True),
        )

    def test_decile_difference_is_invariant_to_row_order_with_ties(self) -> None:
        values = np.repeat(np.arange(8.0), 25)
        other = np.tile(np.arange(25.0), 8)
        equation = Equation(
            intercept=0.0,
            terms=(Term("product", (Atom("x"), Atom("other"))),),
            weights=(1.0,),
            standardized_weights=(1.0,),
        )
        stability = pl.DataFrame({"term": ["[x] * [other]"], "folds": [1], "frequency": [1.0]})
        original = feature_evidence(
            equation, {"x": values, "other": other}, stability, margin=0.0, min_direction=0.0, min_frequency=0.0
        )
        permutation = np.random.default_rng(0).permutation(values.size)
        shuffled = feature_evidence(
            equation,
            {"x": values[permutation], "other": other[permutation]},
            stability,
            margin=0.0,
            min_direction=0.0,
            min_frequency=0.0,
        )
        original_x = original.filter(pl.col("feature") == "x")["decile_difference"][0]
        shuffled_x = shuffled.filter(pl.col("feature") == "x")["decile_difference"][0]
        self.assertAlmostEqual(original_x, shuffled_x)

    def test_constant_feature_has_zero_decile_difference(self) -> None:
        equation = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("constant"),)),),
            weights=(1.0,),
            standardized_weights=(1.0,),
        )
        table = feature_evidence(
            equation,
            {"constant": np.ones(20)},
            pl.DataFrame({"term": ["constant"], "folds": [1], "frequency": [1.0]}),
            margin=0.0,
            min_direction=0.0,
            min_frequency=0.0,
        )
        self.assertEqual(table["decile_difference"][0], 0.0)

    def test_margin_filter_includes_cutoff_and_rejects_value_below_it(self) -> None:
        difference = 0.3636363636363637
        equation = rising_equation()
        data = columns()
        stability = stability_table(up=0.5, down=0.5)
        at_cutoff = feature_evidence(
            equation, data, stability, margin=difference, min_direction=0.0, min_frequency=0.0
        ).filter(pl.col("feature") == "up")
        below_cutoff = feature_evidence(
            equation,
            data,
            stability,
            margin=float(np.nextafter(difference, np.inf)),
            min_direction=0.0,
            min_frequency=0.0,
        ).filter(pl.col("feature") == "up")
        self.assertTrue(at_cutoff["pass_margin"][0])
        self.assertFalse(below_cutoff["pass_margin"][0])

    def test_direction_filter_includes_cutoff_and_rejects_value_below_it(self) -> None:
        equation = rising_equation()
        data = columns()
        stability = stability_table(up=0.5, down=0.5)
        at_cutoff = feature_evidence(
            equation, data, stability, margin=0.0, min_direction=1.0, min_frequency=0.0
        ).filter(pl.col("feature") == "up")
        below_cutoff = feature_evidence(
            equation,
            data,
            stability,
            margin=0.0,
            min_direction=float(np.nextafter(1.0, np.inf)),
            min_frequency=0.0,
        ).filter(pl.col("feature") == "up")
        self.assertTrue(at_cutoff["pass_direction"][0])
        self.assertFalse(below_cutoff["pass_direction"][0])

    def test_frequency_filter_includes_cutoff_and_rejects_value_below_it(self) -> None:
        equation = rising_equation()
        data = columns()
        stability = stability_table(up=0.5, down=0.5)
        at_cutoff = feature_evidence(
            equation, data, stability, margin=0.0, min_direction=0.0, min_frequency=0.5
        ).filter(pl.col("feature") == "up")
        below_cutoff = feature_evidence(
            equation,
            data,
            stability,
            margin=0.0,
            min_direction=0.0,
            min_frequency=float(np.nextafter(0.5, np.inf)),
        ).filter(pl.col("feature") == "up")
        self.assertTrue(at_cutoff["pass_frequency"][0])
        self.assertFalse(below_cutoff["pass_frequency"][0])

    def test_weak_spearman_is_the_only_failed_evidence_gate(self) -> None:
        equation = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("x", "sq"),)), Term("atom", (Atom("x"),))),
            weights=(1.0, 0.01),
            standardized_weights=(1.0, 0.01),
        )
        stability = pl.DataFrame(
            {"term": ["x^2", "x"], "folds": [1, 1], "frequency": [1.0, 1.0]}
        )
        table = feature_evidence(
            equation,
            {"x": np.linspace(-1.0, 1.0, 100)},
            stability,
            margin=0.001,
            min_direction=0.15,
            min_frequency=0.5,
        )
        row = table.row(0, named=True)
        self.assertGreater(row["decile_difference"], 0.001)
        self.assertGreater(row["spearman"], 0.0)
        self.assertLess(abs(row["spearman"]), 0.15)
        self.assertEqual(
            (row["pass_margin"], row["pass_direction"], row["pass_sign"], row["pass_frequency"], row["passes_all"]),
            (True, False, True, True, False),
        )

    def test_nonfinite_decile_difference_fails_margin(self) -> None:
        equation = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("x"),)),),
            weights=(float("nan"),),
            standardized_weights=(1.0,),
        )
        table = feature_evidence(
            equation,
            {"x": np.linspace(1.0, 5.0, 100)},
            pl.DataFrame({"term": ["x"], "folds": [1], "frequency": [1.0]}),
            margin=0.05,
            min_direction=0.15,
            min_frequency=0.5,
        )
        self.assertFalse(table["pass_direction"][0])
        self.assertFalse(table["pass_margin"][0])
        self.assertFalse(table["passes_all"][0])

    def test_nan_spearman_fails_direction(self) -> None:
        equation = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("x"),)),),
            weights=(1.0,),
            standardized_weights=(1.0,),
        )
        practices = pl.DataFrame(
            {
                "feature": ["x"],
                "meaning": ["x"],
                "n_terms": [1],
                "direction": [float("nan")],
                "effect": [0.1],
                "stability": [1.0],
            }
        )
        with mock.patch("ml_meta_perf.practices.feature_practices", return_value=practices):
            table = feature_evidence(
                equation,
                {"x": np.linspace(1.0, 5.0, 100)},
                pl.DataFrame({"term": ["x"], "folds": [1], "frequency": [1.0]}),
                margin=0.05,
                min_direction=0.15,
                min_frequency=0.5,
            )
        self.assertTrue(np.isnan(table["spearman"][0]))
        self.assertFalse(table["pass_direction"][0])

    def test_empty_equation_has_the_feature_evidence_schema(self) -> None:
        empty = Equation(intercept=0.0, terms=(), weights=(), standardized_weights=())
        table = feature_evidence(empty, columns(), pl.DataFrame(), margin=0.05, min_direction=0.15, min_frequency=0.5)
        self.assertEqual(table.height, 0)
        self.assertEqual(
            table.columns,
            [
                "feature",
                "meaning",
                "n_terms",
                "decile_difference",
                "spearman",
                "mean_frequency",
                "margin",
                "pass_margin",
                "pass_direction",
                "pass_sign",
                "pass_frequency",
                "passes_all",
            ],
        )


class TestBestPractices(unittest.TestCase):
    def test_statements_state_the_direction(self) -> None:
        table = best_practices(rising_equation(), columns())
        statements = {row["feature"]: row["practice"] for row in table.iter_rows(named=True)}
        self.assertIn("higher MCC", statements["up"])
        self.assertIn("lower MCC", statements["down"])

    def test_unstable_terms_are_dropped(self) -> None:
        table = best_practices(rising_equation(), columns(), stability_table(up=0.95, down=0.10))
        self.assertEqual(table["feature"].to_list(), ["up"])

    def test_tiny_effects_are_dropped(self) -> None:
        tiny = Equation(
            intercept=0.5,
            terms=(Term("atom", (Atom("up"),)),),
            weights=(1e-6,),
            standardized_weights=(0.9,),
        )
        self.assertEqual(best_practices(tiny, columns()).height, 0)

    def test_non_monotone_features_are_dropped(self) -> None:
        # A symmetric V: large contribution range, no direction anyone could state.
        values = np.concatenate([np.linspace(-3.0, 3.0, 60)])
        equation = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("v", "sq"),)),),
            weights=(0.2,),
            standardized_weights=(0.8,),
        )
        table = best_practices(equation, {"v": values})
        self.assertEqual(table.height, 0)

    def test_features_whose_two_measurements_disagree_are_dropped(self) -> None:
        # ``direction`` is a rank correlation over the whole range and ``effect`` compares
        # the top decile against the bottom, so they can disagree in sign. When they do,
        # the sentence would state one direction while quoting the other as its evidence,
        # which is worse than saying nothing.
        #
        # Ninety rows over a narrow low range where the contribution falls, then ten far
        # above it where the quadratic takes over: monotonically falling by rank, rising
        # by decile.
        values = np.concatenate([np.linspace(0.0, 0.9, 90), np.linspace(4.0, 5.0, 10)])
        equation = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("f"),)), Term("atom", (Atom("f", "sq"),))),
            weights=(-1.0, 0.3),
            standardized_weights=(-0.5, 0.5),
        )
        measured = feature_practices(equation, {"f": values})
        self.assertLess(float(measured["direction"][0]), 0.0)
        self.assertGreater(float(measured["effect"][0]), 0.0)

        self.assertEqual(best_practices(equation, {"f": values}).height, 0)

    def test_agreeing_measurements_are_kept(self) -> None:
        table = best_practices(rising_equation(), columns())
        self.assertEqual(table.height, 2)
        for row in table.iter_rows(named=True):
            self.assertGreater(float(row["direction"]) * float(row["effect"]), 0.0)

    def test_confidence_rises_with_evidence(self) -> None:
        strong = best_practices(rising_equation(), columns(), stability_table(up=0.95, down=0.95))
        levels = set(strong["confidence"].to_list())
        self.assertTrue(levels <= {"strong", "moderate", "weak"})
        self.assertIn("strong", levels)

    def test_confidence_is_unrated_without_stability(self) -> None:
        table = best_practices(rising_equation(), columns())
        self.assertEqual(set(table["confidence"].to_list()), {"unrated"})

    def test_empty_equation_yields_no_practices(self) -> None:
        empty = Equation(intercept=0.5, terms=(), weights=(), standardized_weights=())
        table = best_practices(empty, columns())
        self.assertEqual(table.height, 0)
        self.assertIn("practice", table.columns)


class TestConcordance(unittest.TestCase):
    """Whether two independently-discovered equations give the same advice."""

    def setUp(self) -> None:
        self.columns = columns()
        self.a = rising_equation()
        # A different equation over the same features, agreeing in direction but not in
        # magnitude: this is the case the analysis is designed to detect.
        self.b = Equation(
            intercept=0.4,
            terms=(Term("atom", (Atom("up"),)), Term("atom", (Atom("down"),))),
            weights=(0.02, -0.30),
            standardized_weights=(0.2, -0.8),
        )

    def test_agrees_when_directions_match(self) -> None:
        table = concordance(self.a, self.b, self.columns)
        self.assertEqual(sorted(table["feature"].to_list()), ["down", "up"])
        self.assertTrue(table["agrees"].all())

    def test_disagrees_when_a_sign_flips(self) -> None:
        flipped = Equation(
            intercept=0.4,
            terms=(Term("atom", (Atom("up"),)),),
            weights=(-0.10,),
            standardized_weights=(-0.6,),
        )
        table = concordance(self.a, flipped, self.columns)
        self.assertFalse(table["agrees"].all())

    def test_only_shared_features_are_compared(self) -> None:
        other = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("up"),)),),
            weights=(0.1,),
            standardized_weights=(0.5,),
        )
        self.assertEqual(concordance(self.a, other, self.columns)["feature"].to_list(), ["up"])

    def test_summary_reports_full_agreement(self) -> None:
        summary = concordance_summary(concordance(self.a, self.b, self.columns))
        self.assertEqual(summary["shared"], 2.0)
        self.assertAlmostEqual(summary["direction_agreement"], 1.0)

    def test_summary_of_an_empty_table(self) -> None:
        empty = Equation(intercept=0.0, terms=(), weights=(), standardized_weights=())
        summary = concordance_summary(concordance(empty, empty, self.columns))
        self.assertEqual(summary["shared"], 0.0)

    def test_custom_names_appear_in_the_columns(self) -> None:
        table = concordance(self.a, self.b, self.columns, names=("enum", "cut"))
        self.assertIn("direction_enum", table.columns)
        self.assertIn("effect_cut", table.columns)


class TestRender(unittest.TestCase):
    def test_numbered_output(self) -> None:
        text = render(best_practices(rising_equation(), columns()))
        self.assertIn(" 1. ", text)
        self.assertEqual(len(text.splitlines()), 2)

    def test_empty_table_says_so(self) -> None:
        empty = Equation(intercept=0.5, terms=(), weights=(), standardized_weights=())
        self.assertIn("No practice", render(best_practices(empty, columns())))


if __name__ == "__main__":
    unittest.main()
