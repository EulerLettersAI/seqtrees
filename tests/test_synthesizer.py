import importlib.util
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seqtree import SequentialTreeSynthesizer


DATA = [
    {"age": 24, "sex_code": 0, "risk_code": 0},
    {"age": 31, "sex_code": 1, "risk_code": 0},
    {"age": 45, "sex_code": 0, "risk_code": 1},
    {"age": 52, "sex_code": 1, "risk_code": 1},
    {"age": 67, "sex_code": 0, "risk_code": 2},
    {"age": 73, "sex_code": 1, "risk_code": 2},
]


class SequentialTreeSynthesizerTest(unittest.TestCase):
    def test_fit_sample_uses_requested_column_order(self):
        model = SequentialTreeSynthesizer(
            variable_order=["age", "sex_code", "risk_code"],
            min_samples_leaf=1,
            random_state=1,
        )

        model.fit(DATA)
        rows = model.sample(20)

        self.assertEqual(model.get_variable_order(), ["age", "sex_code", "risk_code"])
        self.assertEqual(len(rows), 20)
        self.assertEqual(list(rows[0].keys()), ["age", "sex_code", "risk_code"])
        self.assertTrue(all(row["risk_code"] in {0, 1, 2} for row in rows))

    def test_optimize_order_includes_every_column(self):
        model = SequentialTreeSynthesizer(optimize_order=True, min_samples_leaf=1, n_jobs=2)

        model.fit(DATA)

        self.assertEqual(set(model.get_variable_order()), {"age", "sex_code", "risk_code"})
        self.assertEqual(len(model.get_variable_order()), 3)
        self.assertEqual(model.n_jobs_, 2)

    def test_reproducible_sampling_with_random_state(self):
        model = SequentialTreeSynthesizer(min_samples_leaf=1, random_state=123).fit(DATA)

        first = model.sample(5, random_state=9, n_jobs=2)
        second = model.sample(5, random_state=9, n_jobs=2)

        self.assertEqual(first, second)

    def test_row_sequences_are_supported(self):
        rows = [
            [0, 0, "A"],
            [0, 1, "A"],
            [1, 0, "B"],
            [1, 1, "B"],
        ]
        model = SequentialTreeSynthesizer(variable_order=[0, 1, 2], min_samples_leaf=1)

        model.fit(rows)
        synthetic = model.sample(3)

        self.assertEqual(model.feature_names_in_, ["x0", "x1", "x2"])
        self.assertEqual(len(synthetic), 3)
        self.assertEqual(list(synthetic[0].keys()), ["x0", "x1", "x2"])

    def test_invalid_variable_order_raises(self):
        model = SequentialTreeSynthesizer(variable_order=["age", "risk_code"])

        with self.assertRaises(ValueError):
            model.fit(DATA)

    def test_invalid_parallelism_raises(self):
        model = SequentialTreeSynthesizer(n_jobs=0)

        with self.assertRaises(ValueError):
            model.fit(DATA)

    def test_native_backend_can_be_selected(self):
        model = SequentialTreeSynthesizer(tree_backend="native", min_samples_leaf=1)

        model.fit(DATA)

        self.assertEqual(model.sample(2)[0].keys(), DATA[0].keys())

    def test_float_columns_are_empirical_by_default(self):
        data = [
            {"score": 0.1, "risk_code": 0},
            {"score": 0.4, "risk_code": 0},
            {"score": 0.8, "risk_code": 1},
        ]
        model = SequentialTreeSynthesizer(variable_order=["score", "risk_code"], min_samples_leaf=1).fit(data)

        rows = model.sample(30, random_state=5)

        self.assertEqual(model.continuous_columns_, {"score"})
        self.assertTrue(all(row["score"] in {0.1, 0.4, 0.8} for row in rows))

    def test_interpolation_can_generate_new_float_values(self):
        data = [
            {"score": 0.0, "risk_code": 0},
            {"score": 10.0, "risk_code": 1},
        ]
        model = SequentialTreeSynthesizer(
            variable_order=["score", "risk_code"],
            continuous_strategy="interpolate",
            continuous_columns=["score"],
            min_samples_leaf=1,
        ).fit(data)

        rows = model.sample(50, random_state=11)
        scores = [row["score"] for row in rows]

        self.assertTrue(any(score not in {0.0, 10.0} for score in scores))
        self.assertTrue(all(0.0 <= score <= 10.0 for score in scores))

    def test_integer_code_columns_are_not_inferred_continuous(self):
        model = SequentialTreeSynthesizer(continuous_strategy="interpolate", min_samples_leaf=1).fit(DATA)

        self.assertEqual(model.continuous_columns_, set())

    def test_lightgbm_backend_is_optional(self):
        model = SequentialTreeSynthesizer(tree_backend="lightgbm", min_samples_leaf=1, n_jobs=1)

        if importlib.util.find_spec("lightgbm") is None:
            with self.assertRaises(ImportError):
                model.fit(DATA)
        else:
            model.fit(DATA)
            self.assertEqual(len(model.sample(2)), 2)

    def test_to_puml_contains_learned_order(self):
        model = SequentialTreeSynthesizer(variable_order=["age", "sex_code", "risk_code"], min_samples_leaf=1).fit(DATA)

        puml = model.to_puml()

        self.assertIn("@startuml", puml)
        self.assertIn('"age"', puml)
        self.assertIn("v_age --> v_sex_code", puml)


if __name__ == "__main__":
    unittest.main()
