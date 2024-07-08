import itertools
import random
import pandas as pd

from bat.configs import ConfigurationManager
from bat.reporting import plot_experiments_results
from bat.logic import get_pair_agreement
from bat.benchmark import Benchmark


def run_exp():
    # exp_to_run = "corr_vs_dates"
    # exp_to_run = "resolution_matters"
    # exp_to_run = "corr_metric_matters"
    # exp_to_run = "location_matters"
    # exp_to_run = "test"
    # exp_to_run = "figure_1"
    # exp_to_run = "reference_benchmark_matters"
    # exp_to_run = "bench_bench"
    # exp_to_run = "n_models_matters"
    # exp_to_run = "external"
    # exp_to_run = "ref_bench_quantitative"
    exp_to_run = "ablation"
    # exp_to_run = "bluebench"

    cfg = ConfigurationManager().configs[exp_to_run]
    cfg.exp_to_run = exp_to_run

    tester = Tester(cfg=cfg)

    all_bench_res = pd.read_csv(cfg.reference_data_path)

    all_agreements = tester.all_vs_all_agreement_testing(Benchmark(all_bench_res))

    plot_experiments_results(agreement_df=all_agreements, cfg=tester.cfg)


class Tester:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def fetch_reference_models_names(
        reference_benchmark,
        n_models,
    ):
        return list(reference_benchmark.get_model_appearences_count().keys())[:n_models]

    def all_vs_all_agreement_testing(self, benchmark):
        assert all(
            benchmark.df.drop_duplicates(subset=["scenario", "source"])
            .groupby("scenario")["source"]
            .count()
            == 1
        ), "duplicated scenarios exist, consider running benchmark.clear_repeated_scenarios()"

        all_bench_res = benchmark.df

        # List of all scenarios
        pair_agreements = []

        used_scenarios = all_bench_res["scenario"].unique().tolist()
        if not self.cfg.include_aggregate_as_scenario:
            used_scenarios = [s for s in used_scenarios if s != "Aggregate"]

        scenario_pairs = [
            (a, b) for a, b in itertools.combinations(used_scenarios, 2) if a != b
        ]

        # Iterate over each pair of scenarios
        for corr_type in self.cfg.corr_types:
            for model_select_strategy in self.cfg.model_select_strategy_list:
                for model_subset_size_requested in self.cfg.n_models_taken_list:
                    for scenario1, scenario2 in scenario_pairs:
                        cur_scen_res = all_bench_res.query(
                            f'scenario == "{scenario1}" or scenario == "{scenario2}"'
                        )

                        for exp_n in range(self.cfg.n_exps):
                            # for date_threshold in date_thresholds:
                            pair_agreements_cfg = {
                                "scenario": scenario1,
                                "source": cur_scen_res.query(
                                    "scenario==@scenario1"
                                ).iloc[0]["source"],
                                "ref_scenario": scenario2,
                                "ref_source": cur_scen_res.query(
                                    "scenario==@scenario2"
                                ).iloc[0]["source"],
                                "corr_type": corr_type,
                                "model_select_strategy": model_select_strategy,
                                "model_subset_size_requested": model_subset_size_requested,
                                "exp_n": exp_n,
                            }

                            # sorting according to one of the benchmarks
                            res_to_sort_by = all_bench_res.query(
                                f"scenario=='{random.choice([scenario1, scenario2])}'"
                            )

                            models_intersect = (
                                cur_scen_res["model"]
                                .value_counts()[
                                    cur_scen_res["model"].value_counts() == 2
                                ]
                                .index.tolist()
                            )

                            if len(models_intersect) < max(
                                model_subset_size_requested,
                                self.cfg.min_n_models_intersect,
                            ):
                                continue

                            pair_agreement, models_taken = get_pair_agreement(
                                cur_scen_res,
                                res_to_sort_by,
                                pair_agreements_cfg,
                                models_intersect,
                            )

                            if pair_agreement is not None:
                                pair_agreement_reported = pair_agreements_cfg.copy()
                                pair_agreement_reported.update(
                                    {
                                        "correlation": pair_agreement,
                                        # "models_selected": models_taken,
                                    }
                                )
                                pair_agreements.append(pair_agreement_reported)

        all_agreements = pd.DataFrame(pair_agreements)

        # add the the reversed scenario pairs
        all_agreements_reversed_scenarios = all_agreements.rename(
            columns={"scenario": "ref_scenario", "ref_scenario": "scenario"}
        )
        all_agreements = pd.concat(
            [all_agreements, all_agreements_reversed_scenarios]
        ).reset_index(drop=True)

        return all_agreements


if __name__ == "__main__":
    # import line_profiler

    # profile = line_profiler.LineProfiler()
    # profile.add_function(main)
    # profile.enable()

    # main()

    # profile.print_stats()

    run_exp()
