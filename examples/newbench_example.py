import pandas as pd
from bat import Tester, Config, Benchmark, Reporter
from bat.utils import get_holistic_benchmark


cfg = Config(
    exp_to_run="example",
    n_models_taken_list=[0],
    model_select_strategy_list=["random"],
    n_exps=10,
    # reference_data_path="data/combined_holistic.csv",
)


newbench_name = "fakebench"
tester = Tester(cfg=cfg)

models_for_benchmark_scoring = tester.fetch_reference_models_names(
    reference_benchmark=get_holistic_benchmark(), n_models=20
)

print(models_for_benchmark_scoring)

newbench = Benchmark(
    pd.read_csv(f"src/bat/assets/{newbench_name}.csv"),
    data_source=newbench_name,
)

newbench.add_aggragete(new_col_name=f"{newbench_name}_mwr")

newbench_agreements = tester.all_vs_all_agreement_testing(newbench)

reporter = Reporter()
reporter.draw_agreements(newbench_agreements)

allbench = newbench.extend(get_holistic_benchmark())
allbench.clear_repeated_scenarios(source_to_keep=newbench_name)

all_agreements = tester.all_vs_all_agreement_testing(allbench)

reporter.draw_agreements(all_agreements)
