import itertools
import random
import pandas as pd

from bat.logic import get_pair_agreement


class Tester:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def fetch_reference_models_names(
        reference_benchmark,
        n_models,
    ):
        return list(reference_benchmark.get_model_appearences_count().keys())[:n_models]

    def all_vs_all_agreement_testing(self, benchmark, single_source_scenario=None):
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

        scenario_pairs = [
            (a, b) for a, b in itertools.combinations(used_scenarios, 2) if a != b
        ]

        if single_source_scenario:
            assert (
                single_source_scenario in used_scenarios
            ), f"single_source_scenario requested {single_source_scenario} does not appear as a scenario in the benchmar"
            scenario_pairs = [
                (a, b) for a, b in scenario_pairs if single_source_scenario in [a, b]
            ]

        # Iterate over each pair of scenarios
        for corr_type in self.cfg.corr_types:
            for model_select_strategy in self.cfg.model_select_strategy_list:
                for model_subset_size_requested in self.cfg.n_models_taken_list:
                    for scenario1, scenario2 in scenario_pairs:
                        cur_scen_res = all_bench_res.query(
                            f'scenario == "{scenario1}" or scenario == "{scenario2}"'
                        )

                        scenario_source = cur_scen_res.query(
                            "scenario==@scenario1"
                        ).iloc[0]["source"]
                        ref_source = cur_scen_res.query("scenario==@scenario2").iloc[0][
                            "source"
                        ]

                        for exp_n in range(self.cfg.n_exps):
                            # for date_threshold in date_thresholds:
                            pair_agreements_cfg = {
                                "scenario": scenario1,
                                "scenario_source": scenario_source,
                                "ref_scenario": scenario2,
                                "ref_source": ref_source,
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

                            pair_agreement, p_value = get_pair_agreement(
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
                                        "p_value": p_value,
                                    }
                                )
                                pair_agreements.append(pair_agreement_reported)

        all_agreements = pd.DataFrame(pair_agreements)

        # add the the reversed scenario pairs
        all_agreements_reversed_scenarios = all_agreements.rename(
            columns={
                "scenario": "ref_scenario",
                "ref_scenario": "scenario",
                "scenario_source": "ref_source",
                "ref_source": "scenario_source",
            }
        )
        all_agreements = pd.concat(
            [all_agreements, all_agreements_reversed_scenarios]
        ).reset_index(drop=True)

        return all_agreements
