import itertools
import random
import pandas as pd

from bat.configs import ConfigurationManager

# from .configs import ConfigurationManager
from bat.reporting import plot_experiments_results
from bat.logic import get_pair_agreement


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

    # all_bench_res = load_data(
    #     "data/BLZ_data_240313.csv", scenario_blacklist=tester.cfg.scenario_blacklist
    # )

    # all_bench_res = add_aggragete_with_mwr(
    #     all_bench_res, scenarios_for_aggragate=tester.cfg.aggregate_scenarios
    # )
    all_bench_res = pd.read_csv(cfg.reference_data_path)

    # if "arena_hard" in cfg.external_benchmarks_tested:
    #     from benchmark_agreement.examples.utils import (
    #         get_arena_hard_benchmark,
    #     )

    #     arena_hard = get_arena_hard_benchmark()
    #     arena_hard["scenario"] = "Arena Hard"
    #     assert (
    #         cfg.n_dates == None
    #     ), "adding external_benchmark is not supported with date thresholding"
    #     all_bench_res = pd.concat([all_bench_res, arena_hard])

    # if "oc" in cfg.external_benchmarks_tested:
    #     renaming_dict = {
    #         "Claude-1": "Claude-1",
    #         "LLaMA-13B": "Llama-13B",
    #         "LLaMA-2-13B-Chat": "Llama-2-13b-chat",
    #         "LLaMA-2-7B-Chat": "Llama-2-7b-chat",
    #         "LLaMA-2-70B-Chat": "Llama-2-70b-chat",
    #         "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7b-Instruct-v0.1",
    #         "Mistral-7B-Instruct-v0.1": "Mistral-7B-Instruct-v0.1",
    #         "Qwen-14B-Chat": "Qwen-14B-Chat",
    #         "Qwen-7B-Chat": "qwen1.5-7b-chat",
    #         "Yi-34B": "Yi-34B-Chat",
    #         "ChatGLM3-6B": "ChatGLM3-6B",
    #         "ChatGLM2-6B": "ChatGLM2-6B",
    #         "Dolphin-2.2.1-Mistral-7B": "Dolphin-2.2.1-Mistral-7B",
    #         "Vicuna-13B-v1.3": "Vicuna-13B",
    #         "Vicuna-7B-v1.3": "Vicuna-7B",
    #         # Additional mappings based on revised inspection
    #         "InternLM2-Chat-20B": "InternLM-20B-Chat",  # Assuming based on structure
    #         "Vicuna-33B-v1.3": "Vicuna-33B",  # Minor versioning detail
    #         "InternLM2-20B": "InternLM-20B",  # Close naming
    #         "WeMix-LLaMa2-70B": "WeMix-LLaMa2-13B",  # Family of models
    #         "InternLM2-7B": "InternLM-7B",  # Direct correlation in naming
    #         "GPT-4": "GPT-4-0314",  # Speculative based on common versioning
    #     }

    #     oc = pd.read_csv("data/oc/oc1_combined_en_240206_v1.csv")
    #     models_in_oc = renaming_dict.keys()
    #     oc = oc.query("Model in @models_in_oc")
    #     oc = (
    #         oc.drop(columns=["Release"])
    #         .rename(columns={"Model": "model"})
    #         .melt(id_vars="model", var_name="scenario", value_name="score")
    #     )
    #     oc["scenario"] = oc["scenario"].apply(lambda x: "OC_" + x)
    #     all_bench_res = pd.concat([all_bench_res, oc])

    # if "holmes" in cfg.external_benchmarks_tested:
    #     holmes = (
    #         (
    #             pd.read_csv("data/holmes_results_v1.0.csv")
    #             .rename(
    #                 columns={
    #                     "train portion": "train_portion",
    #                     "probing dataset": "subscenario",
    #                 }
    #             )
    #             .query("train_portion==1")
    #             .drop(
    #                 columns=[
    #                     "linguistic subfield",
    #                     "probe type",
    #                     "Unnamed: 0",
    #                     "linguistic phenomena",
    #                     "train_portion",
    #                     "probe",
    #                 ]
    #             )
    #         )
    #         .melt(id_vars=["subscenario"], var_name="model", value_name="score")
    #         .groupby("model")["score"]
    #         .mean()
    #         .reset_index()
    #         .sort_values("score", ascending=False)
    #     )

    # holmes["model"] = holmes["model"].apply(
    #     lambda x: x.lower().replace(" ", "-").split("/")[-1]
    # )

    # [model for model in holmes.model.unique() if model in all_bench_res.model.unique()]

    # if cfg.n_dates != None:
    #     all_bench_res = all_bench_res.merge(get_release_date_df(), on="model")

    # all_bench_res.query('scenario=="Arena Hard"').model.unique()

    # all_bench_res = pd.read_csv(cfg.reference_data_path)
    all_agreements = tester.all_vs_all_agreement_testing(all_bench_res)

    plot_experiments_results(agreement_df=all_agreements, cfg=tester.cfg)


class Tester:
    def __init__(self, cfg):
        self.cfg = cfg

    def fetch_reference_models_names(
        self, n_models, only_opensource=False, scenario_ratio_threshold=1 / 2
    ):
        from benchmark_agreement.examples.utils import (
            fetch_reference_models_names as _fetch_reference_models_names,
        )

        return _fetch_reference_models_names(
            cfg=self.cfg,
            n_models=n_models,
            only_opensource=only_opensource,
            scenario_ratio_threshold=scenario_ratio_threshold,
        )

    def run_bench_agreement_testing(self, new_bench_res, tested_bench_name="tested"):
        # turn results to a df
        new_bench_res = pd.DataFrame(new_bench_res, columns=["model", "score"])
        # add scenario columns
        new_bench_res["scenario"] = tested_bench_name

        # combine tested benchmark and reference benchmarks
        ref_bench_res = pd.read_csv(self.cfg.reference_data_path)
        all_bench_res = pd.concat([ref_bench_res, new_bench_res])

        return self.all_vs_all_agreement_testing(all_bench_res)

    def all_vs_all_agreement_testing(self, all_bench_res):
        # all_bench_res["model"] = all_bench_res["model"].apply(
        #     lambda model_name: model_name.lower().replace(" ", "-")
        # )

        if "date_random" in self.cfg.model_select_strategy_list:
            date_thresholds = all_bench_res.date.sort_values().unique().tolist()[6:-1]
        else:
            date_thresholds = [
                None
                # all_bench_res.date.sort_values().unique().tolist()[-2]
            ]  # last date

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
                                "ref_scenario": scenario2,
                                "corr_type": corr_type,
                                "model_select_strategy": model_select_strategy,
                                "model_subset_size_requested": model_subset_size_requested,
                                # "date_threshold": date_threshold,
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

                            if len(models_intersect) < model_subset_size_requested:
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
                                        "models_selected": models_taken,
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

    def generate_benchmark_agreement_testing_report(
        self, agreements, tested_name="tested"
    ):
        agreements_no_agg = agreements.query(
            'scenario!="Aggregate" and ref_scenario!="Aggregate"'
        )

        agreements_no_agg_test_scen = agreements_no_agg.query(
            f'model_select_strategy=="somewhere_aggregate" and scenario=="{tested_name}"'
        ).rename(columns={"model_subset_size_requested": "Resolution"})

        agreements_no_tested = agreements.query(
            'scenario!="@tested_name" and ref_scenario!="@tested_name"'
        )

        assert (
            len(agreements["corr_type"].unique()) == 1
        ), "only one corr_type is allowed"

        assert (
            len(agreements["somewhere_aggregate"].unique()) == 1
        ), "only one model_select_strategy is allowed"

        threshold_df = (
            agreements_no_tested.groupby(
                ["model_subset_size_requested", "corr_type", "ref_scenario", "exp_n"]
            )
            .agg({"correlation": ["mean", "std"]})["correlation"]
            .reset_index()
        )
        threshold_df["threshold"] = threshold_df["mean"] - threshold_df["std"]

        agreements_tested = agreements.query('scenario=="tested"')
        agreements_tested

        threshold_df

        print(
            threshold_df.query('ref_scenario=="Aggregate"').sort_values(
                ["corr_type", "model_subset_size_requested"]
            )[["model_subset_size_requested", "corr_type", "threshold"]]
        )

        from tabulate import tabulate

        def pprint_df(dframe):
            print(tabulate(dframe, headers="keys", tablefmt="psql", showindex=False))

        pprint_df(
            agreements_no_agg_test_scen.groupby(["Resolution"])["correlation"]
            .agg({"mean"})
            .reset_index()
        )

        print(
            agreements_no_agg_test_scen.groupby(["Resolution", "ref_scenario"])
            .agg({"correlation": ["mean", "std"]})
            .to_markdown()
        )

        # sns.set(font_scale=1.2)

        # import plotly.express as px

        # fig = px.line(
        #     agreements_no_agg.query(
        #         f'model_select_strategy=="somewhere_aggregate" and scenario=="{tested_name}"'
        #     ),
        #     x="model_subset_size_requested",
        #     y="correlation",
        #     # markersize=10,
        #     # linewidth=4,
        # )
        # fig.show()

        # fig, ax = plt.subplots()

        # # correlation as a function of model_subset_size_requested
        # sns.pointplot(
        #     # ax=ax,
        #     # kind="point",
        #     data=agreements_no_agg.query(
        #         f'model_select_strategy=="somewhere_aggregate" and scenario=="{tested_name}"'
        #     ),
        #     y="correlation",
        #     x="model_subset_size_requested",
        #     # hue="model_select_strategy",
        #     markersize=10,
        #     linewidth=4,
        #     # errorbar="se",
        #     # linestyle="",
        #     # col="corr_type",
        #     # sharey=False,
        # )
        # # scneario-wise agreement (lines)
        # sns.pointplot(
        #     ax=ax,
        #     # kind="point",
        #     data=agreements_no_agg.query(
        #         f'corr_type=="kendall" and model_select_strategy=="somewhere_aggregate" and scenario=="{tested_name}"'
        #     ),
        #     y="correlation",
        #     x="model_subset_size_requested",
        #     hue="ref_scenario",
        #     errorbar=None,
        #     alpha=0.2,
        #     # aspect=1.5,
        #     # col="corr_type",
        # )
        # plt.xlabel("Resolution")
        # plt.ylabel("Mean Benchamark Pairs Correlation")
        # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        # plt.tight_layout()


if __name__ == "__main__":
    # import line_profiler

    # profile = line_profiler.LineProfiler()
    # profile.add_function(main)
    # profile.enable()

    # main()

    # profile.print_stats()

    run_exp()
