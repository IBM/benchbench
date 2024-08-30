import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np

benchmark2tag = {
    "arena_hard": "holistic",
    "mixeval_hard": "holistic",
    "mixeval_hard_mixed": "holistic",
    "mixeval": "holistic",
    "arena_elo": "holistic",
    "arena_elo_mixed": "holistic",
    "agieval": "holistic",
    "bbh": "holistic",
    "oc1_mwr": "holistic",
    "oc2_mwr": "holistic",
    "alpacav1": "holistic",
    "alpacav2": "holistic",
    "alpacaeval2_lc": "holistic",
    "eq_benchv2": "holistic",
    "gpt4all": "holistic",
    "hugging_6": "holistic",
    "llmonitor": "holistic",
    "magi": "holistic",
    "mt_bench": "holistic",
    "helm_lite_mwr": "holistic",
    "helm_mwr": "holistic",
    "biggen_mwr": "holistic",
    "wildbench_mix": "holistic",
    "wildbench_gpt4t": "holistic",
    "wildbench_haiku": "holistic",
    "wildbench_llama2": "holistic",
    "wb_score": "holistic",
    "olmes_average": "holistic",
    "livebench_average": "holistic",
    #
    "triviaqa_mixed": "knowledge",
    "mmlu_mixed": "knowledge",
    "triviaqa_hard_mixed": "knowledge",
    "triviaqa_hard": "knowledge",
    "mmlu_hard_mixed": "knowledge",
    "boolq_mixed": "knowledge",
    "boolq": "knowledge",
    "triviaqa": "knowledge",
    "naturalquestions": "knowledge",
    "mmlu": "knowledge",
    "mmlu_hard": "knowledge",
    "record": "knowledge",
    "openbookqa": "knowledge",
    "truthfulqa": "knowledge",
    "narrativeqa": "knowledge",
    "naturalquestions_open": "knowledge",
    "naturalquestions_closed": "knowledge",
    "legalbench": "knowledge",
    "medqa": "knowledge",
    "csqa": "knowledge",
    "mmlu_pro": "knowledge",
    "data_analysis_average": "knowledge",
    "high_en": "knowledge",
    "middle_en": "knowledge",
    "primary_en": "knowledge",
    #
    "drop_mixed": "reasoning",
    "hellaswag_mixed": "reasoning",
    "commonsenseqa_mixed": "reasoning",
    "drop_hard_mixed": "reasoning",
    "drop_hard": "reasoning",
    "commonsenseqa": "reasoning",
    "hellaswag": "reasoning",
    "piqa": "reasoning",
    "drop": "reasoning",
    "copa": "reasoning",
    "wic": "reasoning",
    "wsc": "reasoning",
    "winogrande": "reasoning",
    "theory_of_mind": "reasoning",
    "arc_c": "reasoning",
    "arc_e": "reasoning",
    "intention_recognition_en": "reasoning",
    "reasoning_average": "reasoning",
    "reasoning": "reasoning",
    #
    "math": "math",
    "gsm8k": "math",
    "mathematics_average": "math",
    #
    "humaneval": "code",
    "mbpp": "code",
    "humaneval_plus_en": "code",
    "sanitized_mbpp_en": "code",
    "humaneval_x": "code",
    "coding_average": "code",
    #
    "tydiqa": "mt",
    "flores": "mt",
    "translation": "mt",
    "wmt_2014": "mt",
    #
    "agentbench_overall": "agent",
    "agentbench_os": "agent",
    "agentbench_db": "agent",
    "agentbench_kg": "agent",
    "agentbench_dcg": "agent",
    "agentbench_ltp": "agent",
    "agentbench_hh": "agent",
    "agentbench_ws": "agent",
    "agentbench_wb": "agent",
    #
    "ax_b": "other",
    "ax_g": "other",
    "rte": "other",
    "siqa": "other",
    "racemiddle": "other",
    "racehigh": "other",
    "xsum": "other",
    "lambada": "other",
    "teval_en": "other",
    "sentiment_analysis_en": "other",
    "content_summarization_en": "other",
    "quac": "other",
    "ms_marco_regular": "other",
    "ms_marco_trec": "other",
    "cnn/dailymail": "other",
    "imdb": "other",
    "civilcomments": "other",
    "raft": "other",
    "grounding": "other",
    "instruction_following": "other",
    "planning": "other",
    "refinement": "other",
    "safety": "other",
    "tool_usage": "other",
    "language_average": "other",
    "if_average": "other",
}


def get_nice_benchmark_name(bench_name):
    prettified_names = {
        "holmes": "Holmes",
        "helm_lite_narrativeqa": "Helm Lite NarrativeQA",
        "helm_lite_naturalquestionsopen": "Helm Lite NaturalQuestionsOpen",
        "helm_lite_naturalquestionsclosed": "Helm Lite NaturalQuestionsClosed",
        "helm_lite_openbookqa": "Helm Lite OpenBookQA",
        "helm_lite_mmlu": "Helm Lite MMLU",
        "helm_lite_math_equivalentcot": "Helm Lite MathEquivalentCOT",
        "helm_lite_gsm8k": "Helm Lite GSM8K",
        "helm_lite_legalbench": "Helm Lite LegalBench",
        "helm_lite_medqa": "Helm Lite MedQA",
        "helm_lite_wmt2014": "Helm Lite WMT2014",
        "hfv2_bbh": "HFv2 BBH",
        "hfv2_bbh_raw": "HFv2 BBH Raw",
        "hfv2_gpqa": "HFv2 GPQA",
        "hfv2_ifeval": "HFv2 IFEval",
        "hfv2_math_lvl_5": "HFv2 Math Level 5",
        "hfv2_mmlu_pro": "HFv2 MMLU Pro",
        "hfv2_musr": "HFv2 MuSR",
        "oc_mmlu": "OpenCompass MMLU",
        "oc_mmlu_pro": "OpenCompass MMLU Pro",
        "oc_cmmlu": "OpenCompass CMMLU",
        "oc_bbh": "OpenCompass BBH",
        "oc_gqpa_dimand": "OpenCompass GQPA-Dimand",
        "oc_humaneval": "OpenCompass HumanEval",
        "oc_ifeval": "OpenCompass IFEval",
        "helm_mmlu": "Helm MMLU",
        "helm_boolq": "Helm BoolQ",
        "helm_narrativeqa": "Helm NarrativeQA",
        "helm_naturalquestionsclosed": "Helm NaturalQuestionsClosed",
        "helm_naturalquestionsopen": "Helm NaturalQuestionsOpen",
        "helm_quac": "Helm QuAC",
        "helm_openbookqa": "Helm OpenBookQA",
        "helm_imdb": "Helm IMDB",
        "helm_civilcomments": "Helm CivilComments",
        "helm_raft": "Helm RAFT",
        "mmlu_pro": "MMLU Pro",
        "mixeval_triviaqa": "MixEval TriviaQA",
        "mixeval_mmlu": "MixEval MMLU",
        "mixeval_drop": "MixEval DROP",
        "mixeval_hellaswag": "MixEval HellaSwag",
        "mixeval_commonsenseqa": "MixEval CommonsenseQA",
        "mixeval_triviaqa_hard": "MixEval TriviaQA Hard",
        "mixeval_mmlu_hard": "MixEval MMLU Hard",
        "mixeval_drop_hard": "MixEval DROP Hard",
        "oc_language": "OpenCompass Language",
        "oc_knowledge": "OpenCompass Knowledge",
        "oc_reasoning": "OpenCompass Reasoning",
        "oc_math": "OpenCompass Math",
        "oc_code": "OpenCompass Code",
        "oc_instruct": "OpenCompass Instruction",
        "oc_agent": "OpenCompass Agent",
        "oc_arena": "OpenCompass Arena",
        "lb_reasoning": "LiveBench Reasoning",
        "lb_coding": "LiveBench Coding",
        "lb_mathematics": "LiveBench Mathematics",
        "lb_data_analysis": "LiveBench Data Analysis",
        "lb_language": "LiveBench Language",
        "lb_if": "LiveBench Instruction Following",
        "wb_info_seek": "WildBench Information Seeking",
        "wb_creative": "WildBench Creative",
        "wb_code_debug": "WildBench Code Debugging",
        "wb_math_data": "WildBench Math & Data",
        "wb_reason_plan": "WildBench Reasoning & Planning",
        "wb_score": "WildBench Score",
        "hfv1_arc": "HFv1 ARC",
        "hfv1_gsm8k": "HFv1 GSM8K",
        "hfv1_hellaswag": "HFv1 HellaSwag",
        "hfv1_mmlu": "HFv1 MMLU",
        "hfv1_truthfulqa": "HFv1 TruthfulQA",
        "hfv1_winogrande": "HFv1 Winogrande",
        "biggen_grounding": "BigBench Grounding",
        "biggen_instruction_following": "BigBench Instruction Following",
        "biggen_planning": "BigBench Planning",
        "biggen_reasoning": "BigBench Reasoning",
        "biggen_refinement": "BigBench Refinement",
        "biggen_safety": "BigBench Safety",
        "biggen_theory_of_mind": "BigBench Theory of Mind",
        "biggen_tool_usage": "BigBench Tool Usage",
        "biggen_multilingual": "BigBench Multilingual",
        "lb_reasoning_average": "LiveBench Reasoning Average",
        "lb_coding_average": "LiveBench Coding Average",
        "lb_mathematics_average": "LiveBench Mathematics Average",
        "lb_data_analysis_average": "LiveBench Data Analysis Average",
        "lb_language_average": "LiveBench Language Average",
        "lb_if_average": "LiveBench Instruction Following Average",
        "helm_lite": "Helm Lite",
        "hf_open_llm_v2": "HF OpenLLM v2",
        "opencompass_academic": "OpenCompass Academic",
        "arena_elo": "Arena Elo",
        "helm_classic": "Helm Classic",
        "mixeval": "MixEval",
        "mixeval_hard": "MixEval Hard",
        "opencompass": "OpenCompass",
        "alphacaeval_v2lc": "AlphacaEval v2lc",
        "livebench_240725": "LiveBench 240725",
        "wb_elo_lc": "WildBench Elo LC",
        "arena_hard": "Arena Hard",
        "agentbench": "AgentBench",
        "hf_open_llm_v1": "HF OpenLLM v1",
        "biggen": "BigBench",
        "livebench_240624": "LiveBench 240624",
        "mt_bench": "MT-Bench",
    }

    if bench_name in prettified_names:
        return prettified_names[bench_name]
    else:
        return bench_name


class Benchmark:
    def __init__(self, df=pd.DataFrame(), data_source=None):
        self.is_empty = True
        self.df = None
        if len(df) > 0:
            assert data_source, "A datasource must be inputted with a df"
            self.assign_df(df, data_source)

    def load_local_catalog(self, catalog_rel_path="assets/benchmarks"):
        catalog_path = os.path.join(Path(__file__).parent, catalog_rel_path)

        for file_name in os.listdir(catalog_path):
            self.extend(
                Benchmark(
                    pd.read_csv(os.path.join(catalog_path, file_name)),
                    data_source=file_name,
                )
            )

        self.df = self.df.query('scenario != "arena_elo0527"')

    def assign_df(self, df, data_source):
        assert (
            df.columns[0] == "model"
        ), f'the zeroth df column mush be "model", instead, got {df.columns[0]}'

        if "scenario" not in df.columns:
            # Assuming the first column is 'model' and the rest are scenarios
            df = pd.melt(df, id_vars=["model"], var_name="scenario", value_name="score")

        df.replace("-", np.nan, inplace=True)
        df.dropna()
        df["score"] = df["score"].astype(float, errors="ignore")

        df["model"] = df["model"].apply(self.standardize_model_name)
        df["scenario"] = df["scenario"].apply(self.standardize_scenario_name)
        df["aggragated_from"] = [[] for _ in range(len(df))]
        if data_source:
            df["source"] = data_source
        self.df = df
        # self.add_tags()
        self.validate_dataframe()
        self.df.dropna(inplace=True)
        self.is_empty = False

    def normalize_scores_per_scenario(self):
        """
        Normalize the 'score' column in the DataFrame to a 0-1 range within each scenario.

        Parameters:
        df (pd.DataFrame): DataFrame containing 'scenario', 'model', and 'score' columns.

        Returns:
        pd.DataFrame: DataFrame with the 'score' column normalized within each scenario.
        """
        if "score" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'score' column")

        # Apply normalization within each group defined by 'scenario'
        def normalize(group):
            min_score = group["score"].min()
            max_score = group["score"].max()
            # Avoid division by zero in case all scores in a group are the same
            if max_score == min_score:
                group["score"] = (
                    1  # or 0, depending on how you want to handle this case
                )
            else:
                group["score"] = (group["score"] - min_score) / (max_score - min_score)
            return group

        return self.df.groupby("scenario", as_index=False, group_keys=False).apply(
            normalize
        )

    def add_aggragete(
        self,
        new_col_name,
        scenario_blacklist=[],
        mean_or_mwr="mwr",
        agg_source_name=None,
        min_scenario_for_models_to_appear_in_agg=0,
    ):
        def calculate_win_rate(series):
            assert (
                len(series) > 1
            ), "Error: tryting to get the mean win rate with only one column"

            def win_rate(x):
                win_count = sum(1 for value in series if x > value)
                return win_count / (len(series) - 1)

            return series.transform(win_rate)

        df_for_agg = self.df.query("scenario not in @scenario_blacklist")

        n_scenario_for_aggregate = len(df_for_agg["scenario"].unique())
        min_scenario_for_models_to_appear_in_agg = min(
            min_scenario_for_models_to_appear_in_agg, n_scenario_for_aggregate
        )

        # remove models that appears in less then
        models_to_consider = (  # noqa: F841
            df_for_agg.groupby(["model"])["scenario"]
            .count()
            .to_frame()
            .query("scenario>=@min_scenario_for_models_to_appear_in_agg")
            .index.to_list()
        )

        df_for_agg = df_for_agg.query("model in @models_to_consider")

        df_for_agg["wr"] = df_for_agg.groupby(["scenario"])["score"].transform(
            calculate_win_rate
        )

        mean_df = (
            df_for_agg.groupby(["model"])
            .agg({"score": "mean", "wr": "mean"})
            .reset_index()
        )
        mean_df["score"] = mean_df["wr"] if mean_or_mwr == "mwr" else mean_df["score"]
        mean_df["scenario"] = new_col_name
        mean_df["aggragated_from"] = mean_df["scenario"].apply(
            lambda x: [
                scenario
                for scenario in self.df["scenario"].unique()
                if scenario not in scenario_blacklist
            ]
        )

        if agg_source_name:
            mean_df["source"] = agg_source_name
        elif len(self.df["source"].unique()) == 1:
            mean_df["source"] = self.df["source"].unique()[0]
        else:
            raise IOError(
                "more that one source for aggrageted column, in this case, you must specify a agg_source_name"
            )

        self.df = pd.concat([self.df, mean_df.drop(columns=["wr"])])

    def validate_dataframe(self):
        if "Unnamed: 0" in self.df.columns:
            self.df.drop(columns=["Unnamed: 0"], inplace=True)

        required_columns = [
            "model",
            "scenario",
            "score",
            "source",
            "aggragated_from",
        ]

        relevant_columns = [
            col_name for col_name in self.df.columns.tolist() if col_name != "tag"
        ]
        if sorted(relevant_columns) != sorted(required_columns):
            raise ValueError(
                f"DataFrame must contain the following columns: {sorted(required_columns)}\n"
                f"Instead, it contains {sorted(relevant_columns)}"
            )

        if (
            not len(
                self.df[
                    self.df.duplicated(
                        subset=["model", "scenario", "source"], keep=False
                    )
                ]
            )
            == 0
        ):
            # raise ValueError("a model appears more than once for a single scenario")
            # Group by the columns you want to check for duplicates and keep the row with the highest score
            self.df = self.df.groupby(["model", "scenario", "source"], as_index=False)[
                "score"
            ].max()
            print(
                "Warning: Duplicate entries found. Keeping rows with the best scores."
            )

        if not pd.api.types.is_numeric_dtype(self.df["score"]):
            raise ValueError("score must be numeric")

    @staticmethod
    def standardize_scenario_name(name):
        name = (
            name.strip()
            .lower()
            .replace("   ", "-")
            .replace("  ", "-")
            .replace(" ", "-")
            .replace("(", "")
            .replace(")", "")
            .replace("gsm-8k", "gsm8k")
            .replace("open-book", "open")
            .replace("closed-book", "closed")
            .replace("agi-eval", "agieval")
            .replace("alpacaeval2-wr", "alpacav2")
            .replace("alpacav2,-len-adj", "alpacaeval2-lc")
            .replace("hswag", "hellaswag")
            .replace("obqa", "openbookqa")
            .replace("winogrande", "winog")
            .replace("winog", "winogrande")
            .replace("-", "_")
        )

        return get_nice_benchmark_name(name)

    @staticmethod
    def standardize_model_name(name):
        name = (
            name.strip()
            .lower()
            .replace("   ", "-")
            .replace("  ", "-")
            .replace(" ", "-")
            .replace("(", "")
            .replace(")", "")
            .replace("β", "beta")
            .replace("command-r+", "command-r-plus")
            .replace("dbrx-inst", "dbrx-instruct")
            .replace("-hf", "")
            .replace("-", "_")
            .replace("llama_3", "llama3")
            .replace("ul2", "flan-ul2")
            .split("/")[-1]
        )
        return name

    def extend(self, other):
        if not isinstance(other, Benchmark):
            raise TypeError("The added object must be an instance of Benchmark")

        if self.df is not None:
            self.df = pd.concat([self.df, other.df])
        else:
            self.df = other.df

        return self

    def get_models(self):
        return self.df["model"].unique()

    def get_scenarios(self):
        return self.df["scenario"].unique()

    def get_model_appearences_count(self):
        return (
            self.df.groupby("model")["scenario"]
            .count()
            .sort_values(ascending=False)
            .to_dict()
        )

    def get_scenario_appearences_count(self):
        return (
            self.df.groupby("scenario")["model"]
            .count()
            .sort_values(ascending=False)
            .to_dict()
        )

    def show_overlapping_model_counts(self):
        # Counting the occurrences of models for each scenario pair
        cross_tab = pd.crosstab(self.df["scenario"], self.df["model"])

        # Compute the number of models shared between each pair of scenarios
        scenario_combinations = cross_tab.dot(cross_tab.T)

        # Sorting the scenarios based on total models
        sorted_scenarios = (
            scenario_combinations.sum(axis=1).sort_values(ascending=False).index
        )
        scenario_combinations = scenario_combinations.loc[
            sorted_scenarios, sorted_scenarios
        ]

        # Plotting the heatmap
        # plt.figure(figsize=(10, 8))
        sns.clustermap(
            scenario_combinations,
            cmap="coolwarm",
            vmax=20,
            linewidths=0.002,
            xticklabels=True,
            yticklabels=True,
            fmt="d",
            annot=True,
        )

        plt.title("Heatmap of Model Count for Each Pair of Scenarios")
        plt.tight_layout()
        save_path = "figures/show_overlapping_model_counts.png"
        plt.savefig(save_path)
        plt.clf()
        print(f"saved to: {save_path}")

    def clear_repeated_scenarios(self, source_to_keep=None):
        self.df["scenario__source"] = self.df["scenario"] + "__" + self.df["source"]
        # Counting the occurrences of models for each scenario pair
        cross_tab = pd.crosstab(self.df["scenario__source"], self.df["model"])

        # Compute the number of models shared between each pair of scenarios
        scenario_combinations = cross_tab.dot(cross_tab.T)

        self.df["scenario__source_counts"] = self.df["scenario__source"].apply(
            lambda x: scenario_combinations.sum(axis=1)[x]
        )

        # scenario_counts = self.df.drop_duplicates(['scenario','source']).groupby(['scenario'])['source'].count()
        scenarios_already_delt_with = []
        scenarios_source_to_drop = []
        for scenario, scenario_df in self.df.drop_duplicates(
            ["scenario", "source"]
        ).groupby("scenario"):
            if scenario in scenarios_already_delt_with:
                continue
            # scenario = scenario[0]

            if len(scenario_df) > 1:
                if source_to_keep and source_to_keep in scenario_df["source"]:
                    scenario_source_to_keep = scenario_df.query(
                        "source!=@source_to_keep"
                    )["scenario__source"]
                else:
                    scenario_source_to_keep = scenario_df.iloc[
                        scenario_df["scenario__source_counts"].argmax()
                    ]["scenario__source"]

                cur_scenarios_source_to_drop = [
                    scen_source
                    for scen_source in scenario_df["scenario__source"].unique().tolist()
                    if scen_source not in scenario_source_to_keep
                ]
                scenarios_source_to_drop.extend(cur_scenarios_source_to_drop)
                print(
                    f"kept: {scenario_source_to_keep}, dropped: {cur_scenarios_source_to_drop}"
                )
                scenarios_already_delt_with.append(scenario)

        self.df = self.df.query("scenario__source not in @scenarios_source_to_drop")
        self.df.drop(
            columns=["scenario__source", "scenario__source_counts"], inplace=True
        )

    def add_tags(self):
        self.df["tag"] = self.df["scenario"].apply(lambda x: benchmark2tag[x])


if __name__ == "__main__":
    b = Benchmark()
    b.load_local_catalog()
    print()
