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


# if __name__ == "__main__":
#     manager = ConfigurationManager()
#     # Example access to configurations:
#     print(manager.configs["resolution_matters"].n_exps)
