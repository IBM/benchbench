import unittest
import pandas as pd
from bat.benchmark import Benchmark  # Assuming the class is in benchmark.py


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame(
            {
                "model": ["model1", "model2", "model3", "model1", "model2", "model3"],
                "scenario": [
                    "arena_hard",
                    "arena_hard",
                    "arena_hard",
                    "triviaqa_mixed",
                    "triviaqa_mixed",
                    "triviaqa_mixed",
                ],
                "score": [0.8, 0.6, 0.9, 0.7, 0.5, 0.6],
                "source": [
                    "source1",
                    "source1",
                    "source1",
                    "source2",
                    "source2",
                    "source2",
                ],
                "aggragated_from": [[] for _ in range(6)],  # Added missing column
            }
        )

    def test_initialization(self):
        benchmark = Benchmark(
            self.sample_data, data_source="test_source"
        )  # Added data_source
        self.assertIn("model", benchmark.df.columns)
        self.assertIn("scenario", benchmark.df.columns)
        self.assertIn("score", benchmark.df.columns)
        self.assertIn("aggragated_from", benchmark.df.columns)
        self.assertIn("source", benchmark.df.columns)  # Added missing assertion

    def test_normalize_scores_per_scenario(self):
        benchmark = Benchmark(self.sample_data, data_source="test_source")
        normalized_df = benchmark.normalize_scores_per_scenario()
        self.assertTrue((normalized_df.groupby("scenario")["score"].max() == 1).all())
        self.assertTrue((normalized_df.groupby("scenario")["score"].min() == 0).all())

    def test_add_aggragete(self):
        benchmark = Benchmark(self.sample_data, data_source="test_source")
        benchmark.add_aggragete(
            "aggregated_scenario", [], "mwr", agg_source_name="test_source"
        )
        self.assertIn("aggregated_scenario", benchmark.df["scenario"].values)

    def test_standardize_scenario_name(self):
        name = "GSM 8k (Open-Book)"
        standardized_name = Benchmark.standardize_scenario_name(name)
        self.assertEqual(standardized_name, "gsm8k_open")

    def test_standardize_model_name(self):
        name = "Command-R+ (HF)"
        standardized_name = Benchmark.standardize_model_name(name)
        self.assertEqual(standardized_name, "command_r_plus")

    def test_extend(self):
        benchmark1 = Benchmark(self.sample_data, data_source="source_1")
        benchmark2 = Benchmark(self.sample_data, data_source="source_2")
        benchmark1.extend(benchmark2)
        self.assertEqual(len(benchmark1.df), 12)

    def test_get_models(self):
        benchmark = Benchmark(self.sample_data, data_source="source_1")
        models = benchmark.get_models()
        self.assertEqual(len(models), 3)

    def test_get_scenarios(self):
        benchmark = Benchmark(self.sample_data, data_source="test_source")
        scenarios = benchmark.get_scenarios()
        self.assertEqual(len(scenarios), 2)

    def test_clear_repeated_scenarios(self):
        data = {
            "model": ["model1", "model2", "model3", "model1", "model2", "model3"],
            "scenario": [
                "arena_hard",
                "arena_hard",
                "arena_hard",
                "arena_hard",
                "arena_hard",
                "arena_hard",
            ],
            "score": [0.8, 0.6, 0.9, 0.85, 0.55, 0.95],
            "source": [
                "source1",
                "source1",
                "source1",
                "source2",
                "source2",
                "source2",
            ],
            "aggragated_from": [[] for _ in range(6)],  # Added missing column
        }
        df = pd.DataFrame(data)
        benchmark = Benchmark(df)
        benchmark.clear_repeated_scenarios()
        self.assertEqual(len(benchmark.df), 3)

    def test_add_tags(self):
        benchmark = Benchmark(self.sample_data, data_source="test_source")
        benchmark.add_tags()
        self.assertIn("tag", benchmark.df.columns)
        self.assertEqual(benchmark.df["tag"].iloc[0], "holistic")
        self.assertEqual(benchmark.df["tag"].iloc[3], "knowledge")


if __name__ == "__main__":
    unittest.main()
