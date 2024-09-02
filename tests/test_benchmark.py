import unittest
import pandas as pd
from bat import Benchmark  # Replace your_module with the actual module name


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            "model": ["model_a", "model_b", "model_a", "model_b"],
            "scenario": ["scenario_1", "scenario_1", "scenario_2", "scenario_2"],
            "score": [0.8, 0.7, 0.9, 0.6],
        }
        self.df = pd.DataFrame(data)
        self.benchmark = Benchmark(self.df, "test_source")

    def test_assign_df(self):
        # Check if DataFrame is assigned correctly
        self.assertEqual(self.benchmark.df.shape, (4, 5))
        self.assertEqual(self.benchmark.df["source"].unique()[0], "test_source")

    def test_normalize_scores_per_scenario(self):
        # Test score normalization
        normalized_df = self.benchmark.normalize_scores_per_scenario()
        scenario_1_scores = normalized_df[normalized_df["scenario"] == "scenario_1"][
            "score"
        ]
        scenario_2_scores = normalized_df[normalized_df["scenario"] == "scenario_2"][
            "score"
        ]
        self.assertEqual(scenario_1_scores.min(), 0.0)
        self.assertEqual(scenario_1_scores.max(), 1.0)
        self.assertEqual(scenario_2_scores.min(), 0.0)
        self.assertEqual(scenario_2_scores.max(), 1.0)

    def test_add_aggragete(self):
        # Test aggregate column addition
        self.benchmark.add_aggragete(
            new_col_name="aggregate", agg_source_name="aggregated_source"
        )
        self.assertIn("aggregate", self.benchmark.df["scenario"].unique())
        aggregate_rows = self.benchmark.df[self.benchmark.df["scenario"] == "aggregate"]
        self.assertEqual(len(aggregate_rows), 2)  # Two models, so two aggregate rows

    def test_validate_dataframe(self):
        # Test DataFrame validation (should pass with the sample DataFrame)
        self.benchmark.validate_dataframe_post_formatting()

    def test_extend(self):
        # Test extending the Benchmark object
        new_data = {
            "model": ["model_c"],
            "scenario": ["scenario_3"],
            "score": [0.5],
            "source": ["new_source"],
            "aggragated_from": [[]],
        }
        new_df = pd.DataFrame(new_data)
        new_benchmark = Benchmark(new_df, "new_source")
        self.benchmark.extend(new_benchmark)
        self.assertEqual(len(self.benchmark.df), 5)  # Original 4 rows + 1 new row

    def test_get_models(self):
        # Test getting unique model names
        models = self.benchmark.get_models()
        self.assertEqual(set(models), {"model_a", "model_b"})

    def test_get_scenarios(self):
        # Test getting unique scenario names
        scenarios = self.benchmark.get_scenarios()
        self.assertEqual(set(scenarios), {"scenario_1", "scenario_2"})

    def test_get_model_appearences_count(self):
        # Test counting model appearances
        counts = self.benchmark.get_model_appearences_count()
        self.assertEqual(counts["model_a"], 2)
        self.assertEqual(counts["model_b"], 2)

    def test_get_scenario_appearences_count(self):
        # Test counting scenario appearances
        counts = self.benchmark.get_scenario_appearences_count()
        self.assertEqual(counts["scenario_1"], 2)
        self.assertEqual(counts["scenario_2"], 2)

    # Tests for show_overlapping_model_counts and clear_repeated_scenarios are more
    # complex and might require mocking or specific data setups to test effectively.
    # Consider adding these tests based on your specific needs and how you
    # handle plotting and data cleaning in those methods.

    def test_validate_df_pre_formatting_unnamed_0(self):
        # Test DataFrame validation with 'Unnamed: 0' column
        bad_data = {
            "Unnamed: 0": [0, 1],
            "model": ["model_a", "model_b"],
            "scenario_1": [0.8, 0.7],
            "scenario_2": [0.9, 0.6],
        }
        bad_df = pd.DataFrame(bad_data)
        with self.assertRaises(ValueError) as context:
            Benchmark(bad_df, "test_source")
        self.assertIn(
            "DataFrame should not contain 'Unnamed: 0' column", str(context.exception)
        )

    def test_validate_df_pre_formatting_missing_model(self):
        # Test DataFrame validation with missing 'model' column
        bad_data = {
            "scenario": ["scenario_1", "scenario_1", "scenario_2", "scenario_2"],
            "score": [0.8, 0.7, 0.9, 0.6],
        }
        bad_df = pd.DataFrame(bad_data)
        with self.assertRaises(ValueError) as context:
            Benchmark(bad_df, "test_source")
        self.assertIn("DataFrame must contain a 'model' column", str(context.exception))

    def test_validate_df_pre_formatting_missing_scenario(self):
        # Test DataFrame validation with missing 'scenario' (and only 'model')
        bad_data = {
            "model": ["model_a", "model_b"],
        }
        bad_df = pd.DataFrame(bad_data)
        with self.assertRaises(ValueError) as context:
            Benchmark(bad_df, "test_source")
        self.assertIn(
            "DataFrame must contain at least 'model' and one scenario column",
            str(context.exception),
        )

    def test_validate_df_pre_formatting_duplicate_model_scenario(self):
        # Test DataFrame validation with duplicate model-scenario pairs
        bad_data = {
            "model": ["model_a", "model_a", "model_b"],
            "scenario_1": [0.8, 0.9, 0.7],  # Duplicate model_a for scenario_1
            "scenario_2": [0.7, 0.6, 0.8],
        }
        bad_df = pd.DataFrame(bad_data)
        with self.assertRaises(ValueError) as context:
            Benchmark(bad_df, "test_source")
        self.assertIn(
            "DataFrame contains duplicate model-scenario pairs", str(context.exception)
        )

    def test_validate_df_pre_formatting_non_numeric_score(self):
        # Test DataFrame validation with non-numeric score
        bad_data = {
            "model": ["model_a", "model_b"],
            "scenario": ["scenario_1", "scenario_2"],
            "score": ["not_a_number", "also_not_a_number"],
        }
        bad_df = pd.DataFrame(bad_data)
        with self.assertRaises(ValueError) as context:
            Benchmark(bad_df, "test_source")
        self.assertIn("score must be numeric", str(context.exception))

    def test_validate_dataframe_post_formatting_missing_columns(self):
        # Test with missing required columns after formatting
        data = {"model": ["model_a"], "scenario": ["scenario_1"], "score": [0.8]}
        df = pd.DataFrame(data)
        benchmark = Benchmark(df, "test_source")

        # Remove required columns and check if ValueError is raised
        benchmark.df.drop(columns=["source", "aggragated_from"], inplace=True)
        with self.assertRaises(ValueError) as context:
            benchmark.validate_dataframe_post_formatting()
        self.assertIn(
            "DataFrame must contain the following columns", str(context.exception)
        )

    def test_validate_dataframe_post_formatting_non_numeric_score_after_formatting(
        self,
    ):
        # Test with non-numeric score after formatting
        data = {
            "model": ["model_a"],
            "scenario": ["scenario_1"],
            "score": [0.8],
            "source": ["test_source"],
            "aggragated_from": [[]],
        }
        df = pd.DataFrame(data)
        benchmark = Benchmark(df, "test_source")

        # Change score to non-numeric and check if ValueError is raised
        benchmark.df["score"] = "not_a_number"
        with self.assertRaises(ValueError) as context:
            benchmark.validate_dataframe_post_formatting()
        self.assertIn("score must be numeric", str(context.exception))


if __name__ == "__main__":
    unittest.main()
