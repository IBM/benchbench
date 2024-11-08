import pandas as pd
from bat import Config, Tester, Benchmark, Reporter
from datetime import datetime


def load_scenarios(filepath, comment_char="#"):
    """Loads scenarios from a text file, ignoring commented lines.
    Allows specifying the comment character.
    """
    with open(filepath, "r") as f:
        scenarios = [
            line.strip() for line in f if not line.strip().startswith(comment_char)
        ]
    return scenarios


scenarios_for_aggregate = load_scenarios("examples/scenarios_for_aggregate.txt")
scenarios_of_intereset = load_scenarios("examples/scenarios_of_intereset.txt")

# Configuration for agreement testing
n_models_taken = 10  # Number of models to sample for each comparison. 0 means all intersecting models.
model_select_strategy = (
    "random"  # How to select models: "top", "bottom", "random", "somewhere"
)
corr_type = "kendall"  # Correlation types: "kendall", "pearson"
n_exps = 3  # Number of experiments for random sampling. Set to 1 for deterministic strategies.

# --- Load your benchmark ---
my_bench_df = pd.read_csv("examples/my_bench.csv")
my_bench_source_name = f"uploaded_benchmark_{datetime.now().strftime('%y%m%d')}"
my_bench = Benchmark(
    my_bench_df, data_source=my_bench_source_name, normalized_names=False
)

# --- Load the existing benchbench benchmark catalog ---
allbench = Benchmark()
allbench.load_local_catalog()

# --- Create an aggregate benchmark ---
allbench.add_aggregate(
    new_col_name="aggregate",
    agg_source_name="aggregate",
    scenario_whitelist=scenarios_for_aggregate,
    min_scenario_for_models_to_appear_in_agg=max(1, len(scenarios_for_aggregate) // 3),
)

# --- Combine your benchmark with the existing benchmarks ---
allbench.extend(my_bench)

# --- Analyze model overlap for insights ---
uploaded_models = my_bench.get_models()
aggregate_models = allbench.df[allbench.df["source"] == "aggregate"]["model"].unique()
n_overlap_models = len(set(aggregate_models).intersection(uploaded_models))
print(f"Number of models overlapping: {n_overlap_models}")

# --- Remove duplicate scenarios before analysis ---
allbench.clear_repeated_scenarios(
    source_to_keep=my_bench_source_name
)  # Prioritize keeping your benchmark's scenarios


# --- Select specific scenarios for analysis ---
my_scenario_name = allbench.df.query(f'source=="{my_bench_source_name}"')[
    "scenario"
].iloc[0]
scenarios_to_analyze = (
    scenarios_of_intereset + ["aggregate"] + [my_scenario_name]
)  # Use my_bench_name for consistency
allbench.df = allbench.df[allbench.df["scenario"].isin(scenarios_to_analyze)]

# --- Configure and run the agreement tester ---
cfg = Config(
    exp_to_run="example",
    n_models_taken_list=[
        n_models_taken
    ],  # Use lists for consistency with Config definition
    model_select_strategy_list=[model_select_strategy],  # Use lists for consistency
    corr_types=[corr_type],  # Use lists for consistency
    n_exps=n_exps if n_models_taken != 0 else 1,
)

tester = Tester(cfg=cfg)
agreements = tester.all_vs_all_agreement_testing(
    allbench
)  # No need for single_source_scenario here, as we've already filtered

# --- Report the results ---
reporter = Reporter()

reporter.draw_agreements_for_one_source(
    agreements,
    source_of_interest=my_bench_source_name,
)
reporter.draw_agreement_matrix(agreements)
z_score_df = reporter.get_all_z_scores(agreements, aggragate_name="aggregate")
print(z_score_df[["scenario", "z_score"]])
