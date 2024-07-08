from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    exp_to_run: str
    n_models_taken_list: List[int] = field(default_factory=lambda: [])
    model_select_strategy_list: List[str] = field(default_factory=lambda: [])
    n_exps: int = 10
    corr_types: List[str] = field(default_factory=lambda: ["kendall"])
    include_aggregate_as_scenario: bool = False
    scenario_blacklist: List[str] = field(default_factory=lambda: [])
    aggregate_scenarios: List[str] = field(default_factory=lambda: [])
    reference_data_path: str = "src/bat/assets/combined_holistic.csv"
    external_benchmarks_tested: List[str] = field(default_factory=lambda: [])
    min_n_models_intersect: int = 5

    def __post_init__(self):
        self.validate_n_models_taken_list()
        self.validate_model_select_strategy_list()
        self.validate_corr_types()

    def validate_n_models_taken_list(self):
        if not all(isinstance(x, int) for x in self.n_models_taken_list):
            raise ValueError("All items in n_models_taken_list must be integers")

    def validate_model_select_strategy_list(self):
        valid_strategies = {
            "somewhere_aggregate",
            "middle_aggregate",
            "top_aggregate",
            "bottom_aggregate",
            "random",
        }
        if not all(
            item in valid_strategies for item in self.model_select_strategy_list
        ):
            raise ValueError(
                f"Invalid strategy in model_select_strategy_list. Valid options are: {valid_strategies}"
            )

    def validate_corr_types(self):
        valid_types = {"kendall", "pearson"}
        if not all(item in valid_types for item in self.corr_types):
            raise ValueError(
                f"Invalid correlation type. Valid options are: {valid_types}"
            )

    def update_or_add_fields(self, **kwargs):
        """
        Add or update fields dynamically. All new fields are validated.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Re-validate the fields if necessary
        if "n_models_taken_list" in kwargs:
            self.validate_n_models_taken_list()
        if "model_select_strategy_list" in kwargs:
            self.validate_model_select_strategy_list()
        if "corr_types" in kwargs:
            self.validate_corr_types()


class ConfigurationManager:
    def __init__(self):
        self.configs = {
            "recommended": Config(
                exp_to_run="recommended",
                n_models_taken_list=[5, 7, 10, 15, 20],
                model_select_strategy_list=["somewhere_aggregate"],
                include_aggregate_as_scenario=True,
                n_exps=10,
            ),
            "test": Config(
                exp_to_run="test",
                n_models_taken_list=[20, 4],
                model_select_strategy_list=["random"],
                include_aggregate_as_scenario=True,
                n_exps=3,
            ),
            "resolution_matters": Config(
                exp_to_run="resolution_matters",
                n_models_taken_list=[5, 6, 7, 8, 9, 10, 12, 15, 20],
                model_select_strategy_list=["somewhere_aggregate", "random"],
                include_aggregate_as_scenario=False,
                n_exps=20,
            ),
            "location_matters": Config(
                exp_to_run="location_matters",
                n_models_taken_list=[4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 20],
                model_select_strategy_list=[
                    "somewhere_aggregate",
                    "middle_aggregate",
                    "top_aggregate",
                    "bottom_aggregate",
                    "random",
                ],
                corr_types=["kendall", "pearson"],
                include_aggregate_as_scenario=True,
            ),
            "figure_1": Config(
                exp_to_run="figure_1",
                n_models_taken_list=[5, 10, 15, 20, 40],
                model_select_strategy_list=[
                    "top_aggregate",
                ],
                include_aggregate_as_scenario=True,
            ),
            "corr_metric_matters": Config(
                exp_to_run="corr_metric_matters",
                n_models_taken_list=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                model_select_strategy_list=["random"],
                corr_types=["kendall", "pearson"],
                n_exps=10,
                scenario_blacklist=[
                    "MBPP",
                    "Length(ch, Alpaca)",
                    # "WinoGrande",
                    # "TruthfulQA",
                    # "HellaSwag",
                    # "ARC-C",
                    # "GSM-8K",
                    "HELM  Lite",
                    # "LLMonitor",
                    # "Alpaca(v1)",
                    "OpenComp.",
                    "EQ+MAGI",
                    "GPT4ALL",
                ],
            ),
            "reference_benchmark_matters": Config(
                exp_to_run="reference_benchmark_matters",
                n_models_taken_list=[10],
                model_select_strategy_list=["random"],
                include_aggregate_as_scenario=True,
                corr_types=["kendall"],
                scenario_blacklist=[
                    "MBPP",
                    "Length(ch, Alpaca)",
                    "HELM  Lite",
                    "OpenComp.",
                    "EQ+MAGI",
                ],
                aggregate_scenarios=[
                    # "AGI Eval ",
                    # "Alpaca(v2)",
                    "Alpaca(v2, len adj)",
                    "Arena Elo",
                    # "BBH",
                    # "GPT4All",
                    # "Hugging-6",
                    # "MAGI",
                    "MMLU",
                    "MT-bench",
                ],
                n_exps=20,
            ),
            "bench_bench": Config(
                exp_to_run="bench_bench",
                n_models_taken_list=[5, 10, 20],
                model_select_strategy_list=[
                    "somewhere_aggregate",
                    # "middle_aggregate",
                    # "top_aggregate",
                    # "bottom_aggregate",
                    # "random",
                ],
                corr_types=["kendall"],
                include_aggregate_as_scenario=True,
                scenario_blacklist=[
                    "MBPP",
                    "Length(ch, Alpaca)",
                    "OpenComp.",
                    "HELM  Lite",
                ],
                external_benchmarks_tested=["arena_hard"],
                aggregate_scenarios=[
                    "AGI Eval ",
                    "Alpaca(v2)",
                    "Alpaca(v2, len adj)",
                    "Arena Elo",
                    "BBH",
                    "GPT4All",
                    "Hugging-6",
                    "MAGI",
                    "MMLU",
                    "MT-bench",
                ],
                # , "oc"],
            ),
            "n_models_matters": Config(
                exp_to_run="n_models_matters",
                n_models_taken_list=[
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                ],
                model_select_strategy_list=["random"],
                include_aggregate_as_scenario=False,
                scenario_blacklist=["MBPP", "Length(ch, Alpaca)", "OpenComp."],
                n_exps=20,
            ),
            "ablation": Config(
                exp_to_run="ablation",
                n_models_taken_list=[
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                ],
                model_select_strategy_list=[
                    "somewhere_aggregate",
                    "middle_aggregate",
                    "top_aggregate",
                    "bottom_aggregate",
                    "random",
                ],
                include_aggregate_as_scenario=False,
                n_exps=1,
                reference_data_path="src/bat/assets/combined_holistic.csv",
                corr_types=["pearson", "kendall"],
            ),
        }

    def get_recommended_config(self):
        return self.configs["recommended"]


if __name__ == "__main__":
    manager = ConfigurationManager()
    # Example access to configurations:
    print(manager.configs["resolution_matters"].n_exps)
