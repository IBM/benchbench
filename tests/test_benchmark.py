import unittest
import pandas as pd
from benchmark import Benchmark  # assuming the class is in a file named benchmark.py


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        data = {
            "model": ["Model A", "Model B", "Model C"],
            "scenario": ["Scenario 1", "Scenario 1", "Scenario 2"],
            "score": [0.9, 0.8, 0.95],
            "source": ["Source 1", "Source 1", "Source 2"],
        }
        self.df = pd.DataFrame(data)
        self.benchmark = Benchmark(self.df.copy())

    def test_normalize_scores_per_scenario(self):
        normalized_df = self.benchmark.normalize_scores_per_scenario(self.df.copy())
        scenario1 = normalized_df[normalized_df["scenario"] == "Scenario 1"]
        scenario2 = normalized_df[normalized_df["scenario"] == "Scenario 2"]

        self.assertEqual(scenario1["score"].max(), 1.0)
        self.assertEqual(scenario1["score"].min(), 0.0)
        self.assertEqual(scenario2["score"].max(), 1.0)
        self.assertEqual(scenario2["score"].min(), 1.0)

    def test_add_aggragete(self):
        self.benchmark.add_aggragete("aggregated_scenario")
        self.assertIn("aggregated_scenario", self.benchmark.df["scenario"].values)
        self.assertIn("aggragated_from", self.benchmark.df.columns)

    def test_validate_dataframe(self):
        with self.assertRaises(ValueError):
            df_invalid = self.df.drop(columns=["score"])
            Benchmark.validate_dataframe(df_invalid)

        with self.assertRaises(ValueError):
            df_invalid = self.df.copy()
            df_invalid.loc[1, "model"] = df_invalid.loc[0, "model"]
            Benchmark.validate_dataframe(df_invalid)

        with self.assertRaises(ValueError):
            df_invalid = self.df.copy()
            df_invalid["score"] = df_invalid["score"].astype(str)
            Benchmark.validate_dataframe(df_invalid)

    def test_standardize_scenario_name(self):
        standardized_name = self.benchmark.standardize_scenario_name(
            "Scenario (GSM-8K)"
        )
        self.assertEqual(standardized_name, "scenario-gsm8k")

    def test_standardize_model_name(self):
        standardized_name = self.benchmark.standardize_model_name("Model (Command-R+)")
        self.assertEqual(standardized_name, "model-command-r-plus")

    def test_extend(self):
        other_data = {
            "model": ["Model D", "Model E"],
            "scenario": ["Scenario 3", "Scenario 4"],
            "score": [0.85, 0.75],
            "source": ["Source 3", "Source 4"],
        }
        other_df = pd.DataFrame(other_data)
        other_benchmark = Benchmark(other_df)
        extended_benchmark = self.benchmark.extend(other_benchmark)

        self.assertIn("Model D", extended_benchmark.get_models())
        self.assertIn("Scenario 3", extended_benchmark.get_scenarios())

    def test_get_models(self):
        models = self.benchmark.get_models()
        self.assertListEqual(list(models), ["model-a", "model-b", "model-c"])

    def test_get_scenarios(self):
        scenarios = self.benchmark.get_scenarios()
        self.assertListEqual(list(scenarios), ["scenario-1", "scenario-2"])

    def test_get_model_appearences_count(self):
        counts = self.benchmark.get_model_appearences_count()
        self.assertEqual(counts["model-a"], 1)
        self.assertEqual(counts["model-b"], 1)
        self.assertEqual(counts["model-c"], 1)

    def test_get_scenario_appearences_count(self):
        counts = self.benchmark.get_scenario_appearences_count()
        self.assertEqual(counts["scenario-1"], 2)
        self.assertEqual(counts["scenario-2"], 1)

    def test_clear_repeated_scenarios(self):
        self.benchmark.clear_repeated_scenarios()
        self.assertEqual(len(self.benchmark.df), 3)

    def test_show_overlapping_model_counts(self):
        try:
            self.benchmark.show_overlapping_model_counts()
        except Exception as e:
            self.fail(f"show_overlapping_model_counts method failed: {e}")


if __name__ == "__main__":
    unittest.main()
