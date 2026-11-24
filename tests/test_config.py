"""Tests for the study configuration file and its schema."""

import json
import tempfile
import unittest
from pathlib import Path

from ml_meta_perf.config import DEFAULT_CONFIG_PATH, load_config, margin_key, parse_config
from ml_meta_perf.data import flexfl_schema


class TestStudyConfiguration(unittest.TestCase):
    """``config/study.json`` is the only source of the tuned hyperparameters."""

    def setUp(self) -> None:
        self.payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.payload["effects"] = {
            "decile_margin": {
                "total_time_s-log1p": 0.1,
                "comm_bytes_total-log1p": 0.1,
                "compute_time_total_s-log1p": 0.1,
                "compute_time_max_s-log1p": 0.1,
                "comm_time_total_s-log1p": 0.1,
                "validation_time_s-log1p": 0.1,
                "n_epochs": 2.0,
                "performance-classification": 0.02,
                "performance-regression": 0.02,
            },
            "min_direction": 0.15,
            "min_frequency": 0.5,
        }

    def test_the_shipped_file_parses_into_every_section(self) -> None:
        study = load_config()
        self.assertEqual(study, parse_config(self.payload))
        self.assertLessEqual(study.search.max_arity, study.selection.capability_arity)
        self.assertGreater(study.search.max_terms, study.sweep.readable_terms)
        self.assertTrue(hasattr(study, "effects"), "StudyConfig must expose effects")
        self.assertEqual(study.effects.min_direction, 0.15)
        self.assertEqual(study.effects.min_frequency, 0.5)
        slugs = (
            "total_time_s-log1p",
            "comm_bytes_total-log1p",
            "compute_time_total_s-log1p",
            "compute_time_max_s-log1p",
            "comm_time_total_s-log1p",
            "validation_time_s-log1p",
            "n_epochs",
            "performance-classification",
            "performance-regression",
        )
        self.assertEqual([study.effects.margin_for(slug) for slug in slugs], [0.1] * 6 + [2.0, 0.02, 0.02])

    def test_an_unconfigured_effect_margin_names_the_slug(self) -> None:
        effects = load_config().effects
        with self.assertRaisesRegex(KeyError, "unknown-target"):
            effects.margin_for("unknown-target")

    def test_margin_key_uses_configured_fitted_scales(self) -> None:
        study = load_config()
        schemas = (
            (flexfl_schema("total_time_s", log_target=True), "total_time_s-log1p"),
            (flexfl_schema("comm_bytes_total", log_target=True), "comm_bytes_total-log1p"),
            (flexfl_schema("compute_time_total_s", log_target=True), "compute_time_total_s-log1p"),
            (flexfl_schema("compute_time_max_s", log_target=True), "compute_time_max_s-log1p"),
            (flexfl_schema("comm_time_total_s", log_target=True), "comm_time_total_s-log1p"),
            (flexfl_schema("validation_time_s", log_target=True), "validation_time_s-log1p"),
            (flexfl_schema("n_epochs"), "n_epochs"),
            (flexfl_schema("performance", "classification"), "performance-classification"),
            (flexfl_schema("performance", "regression"), "performance-regression"),
            (flexfl_schema("total_time_s", "classification", log_target=True), "total_time_s-log1p"),
        )
        for schema, expected in schemas:
            with self.subTest(schema=schema.slug):
                key = margin_key(schema)
                self.assertEqual(key, expected)
                self.assertGreater(study.effects.margin_for(key), 0.0)

    def test_a_missing_effects_section_is_an_error(self) -> None:
        del self.payload["effects"]
        with self.assertRaises(KeyError):
            parse_config(self.payload)

    def test_unknown_or_missing_effects_key_is_an_error(self) -> None:
        self.payload["effects"]["unexpected"] = 1
        with self.assertRaises(TypeError):
            parse_config(self.payload)

    def test_a_missing_effects_key_is_an_error(self) -> None:
        del self.payload["effects"]["min_frequency"]
        with self.assertRaises(TypeError):
            parse_config(self.payload)

    def test_an_unknown_key_is_an_error_not_a_silent_default(self) -> None:
        self.payload["search"]["pentalty"] = 1.0
        with self.assertRaises(TypeError):
            parse_config(self.payload)

    def test_a_missing_key_is_an_error(self) -> None:
        del self.payload["selection"]["delta"]
        with self.assertRaises(TypeError):
            parse_config(self.payload)

    def test_a_missing_file_names_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.json"
            with self.assertRaisesRegex(FileNotFoundError, "absent.json"):
                load_config(missing)


if __name__ == "__main__":
    unittest.main()
