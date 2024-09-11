import pandas as pd
from bat.utils import get_holistic_benchmark
from bat import Tester, Benchmark, Config, Reporter

reporter = Reporter()

cfg = Config(
    exp_to_run="example",
    n_models_taken_list=[0],
    model_select_strategy_list=["random"],
    n_exps=10,
)

newbench_name = "livebench"
new_bench_agg_name = f"{newbench_name}_mwr"

tester = Tester(cfg=cfg)

models_for_benchmark_scoring = tester.fetch_reference_models_names(
    reference_benchmark=get_holistic_benchmark(), n_models=20
)

newbench = Benchmark(
    pd.read_csv(f"src/bat/assets/{newbench_name}.csv"),
    data_source=newbench_name,
)

inside_agreements = tester.all_vs_all_agreement_testing(newbench)
reporter.draw_agreements(inside_agreements)

newbench.df = newbench.df.query('scenario=="livebench_lb"')

holistic = get_holistic_benchmark()
holistic.add_aggregate(new_col_name="aggregate", agg_source_name="holistic")

allbench = newbench.extend(holistic)
allbench.clear_repeated_scenarios(source_to_keep=newbench_name)

all_agreements = tester.all_vs_all_agreement_testing(allbench)

# observed_scenario = "arena_elo"  # "livebench_lb"
# blacklist_sources = []  # "livebench"

reporter.draw_agreements(all_agreements, scenario_sources=[newbench_name])

z_scores = reporter.get_all_z_scores(all_agreements)
