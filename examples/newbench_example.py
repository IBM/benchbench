import pandas as pd
from bat import Config, Tester, Benchmark, Reporter
from datetime import datetime


holistic_scenarios = [
    "Helm Lite",
    "HF OpenLLM v2",
    "OpenCompass Academic",
    "LMSys Arena",
    "Helm Classic",
    "AlphacaEval v2lc",
    "LiveBench 240829",
    "WildBench Elo LC",
]

n_models_taken_list = [10]  # Number of top models to consider
model_select_strategy_list = ["random"]  # How to select models (e.g., "top", "random")
corr_types = ["kendall"]  # Correlation types to calculate
n_exps = 3  # Number of experiments to run


my_bench_df = pd.read_csv("examples/my_bench.csv")

my_bench = Benchmark()
my_bench.assign_df(
    my_bench_df,
    data_source=f"uploaded_benchmark_{datetime.now().strftime('%y%m%d')}.csv",
    normalized_names=False,
)

allbench = Benchmark()
allbench.load_local_catalog()

allbench.add_aggregate(
    new_col_name="aggregate",
    agg_source_name="aggregate",
    scenario_whitelist=holistic_scenarios,
    min_scenario_for_models_to_appear_in_agg=1
    if len(holistic_scenarios) == 1
    else len(holistic_scenarios) // 3,
)

allbench.extend(my_bench)

# Get unique models for each scenario
uploaded_models = allbench.df[allbench.df["source"].str.contains("uploaded")][
    "model"
].unique()
aggregate_models = allbench.df[allbench.df["source"].str.contains("aggregate")][
    "model"
].unique()

# Find the intersection (overlap) of models
n_overlap_models = len(set(aggregate_models).intersection(uploaded_models))

allbench.clear_repeated_scenarios()

scenarios_i_want = holistic_scenarios + ["aggregate"] + ["my_bench"]
allbench.df = allbench.df[allbench.df["scenario"].isin(scenarios_i_want)]

cfg = Config(
    exp_to_run="example",
    n_models_taken_list=n_models_taken_list,
    model_select_strategy_list=model_select_strategy_list,
    corr_types=corr_types,
    n_exps=n_exps if n_models_taken_list != [0] else 1,
)

tester = Tester(cfg=cfg)

agreements = tester.all_vs_all_agreement_testing(
    allbench,
    single_source_scenario="aggregate",  # olny measuring all with the aggragate
)

reporter = Reporter()
z_scores = reporter.get_all_z_scores(agreements=agreements, aggragate_name="aggregate")
