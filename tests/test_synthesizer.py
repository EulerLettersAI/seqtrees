import importlib.util
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seqtree import SequentialTreeSynthesizer


DATA = [
    {"age": "young", "sex": "F", "risk": "low"},
    {"age": "young", "sex": "M", "risk": "low"},
    {"age": "middle", "sex": "F", "risk": "medium"},
    {"age": "middle", "sex": "M", "risk": "medium"},
    {"age": "older", "sex": "F", "risk": "high"},
    {"age": "older", "sex": "M", "risk": "high"},
]


class SequentialTreeSynthesizerTest(unittest.TestCase):
    def test_fit_sample_uses_requested_column_order(self):
        model = SequentialTreeSynthesizer(
            variable_order=["age", "sex", "risk"],
            min_samples_leaf=1,
            random_state=1,
        )

        model.fit(DATA)
        rows = model.sample(20)

        self.assertEqual(model.get_variable_order(), ["age", "sex", "risk"])
        self.assertEqual(len(rows), 20)
        self.assertEqual(list(rows[0].keys()), ["age", "sex", "risk"])
        self.assertTrue(all(row["risk"] in {"low", "medium", "high"} for row in rows))

    def test_optimize_order_includes_every_column(self):
        model = SequentialTreeSynthesizer(optimize_order=True, min_samples_leaf=1, n_jobs=2)

        model.fit(DATA)

        self.assertEqual(set(model.get_variable_order()), {"age", "sex", "risk"})
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
        model = SequentialTreeSynthesizer(variable_order=["age", "risk"])

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

    def test_lightgbm_backend_is_optional(self):
        model = SequentialTreeSynthesizer(tree_backend="lightgbm", min_samples_leaf=1, n_jobs=1)

        if importlib.util.find_spec("lightgbm") is None:
            with self.assertRaises(ImportError):
                model.fit(DATA)
        else:
            model.fit(DATA)
            self.assertEqual(len(model.sample(2)), 2)

    def test_to_puml_contains_learned_order(self):
        model = SequentialTreeSynthesizer(variable_order=["age", "sex", "risk"], min_samples_leaf=1).fit(DATA)

        puml = model.to_puml()

        self.assertIn("@startuml", puml)
        self.assertIn('"age"', puml)
        self.assertIn("v_age --> v_sex", puml)


if __name__ == "__main__":
    unittest.main()
