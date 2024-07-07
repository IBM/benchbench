from fuzzywuzzy import process, fuzz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


class Benchmark:
    def __init__(self, df, data_source=None):
        df["model"] = df["model"].apply(self.standardize_model_name)
        df["scenario"] = df["scenario"].apply(self.standardize_scenario_name)
        df["aggragated_from"] = [[] for _ in range(len(df))]
        if data_source:
            df["source"] = data_source
        self.df = df
        self.add_tags()
        self.validate_dataframe()
        self.df.dropna(inplace=True)

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
                group[
                    "score"
                ] = 1  # or 0, depending on how you want to handle this case
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
        specified_source="",
    ):
        def calculate_win_rate(series):
            assert (
                len(series) > 1
            ), "Error: tryting to get the mean win rate with only one column"

            def win_rate(x):
                win_count = sum(1 for value in series if x > value)
                return win_count / (len(series) - 1)

            return series.transform(win_rate)

        self.df["wr"] = self.df.groupby(["scenario"])["score"].transform(
            calculate_win_rate
        )

        mean_df = (
            self.df.query("scenario not in @scenario_blacklist")
            .groupby(["model"])
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

        if specified_source:
            mean_df["source"] = specified_source
        elif len(self.df["source"].unique()) == 1:
            mean_df["source"] = self.df["source"].unique()[0]
        else:
            raise IOError(
                "more that one source for aggrageted column, in this case, you must specify a source"
            )

        self.df = pd.concat([self.df, mean_df]).drop(columns=["wr"])

    def validate_dataframe(self):
        if "Unnamed: 0" in self.df.columns:
            self.df.drop(columns=["Unnamed: 0"], inplace=True)

        required_columns = [
            "model",
            "scenario",
            "score",
            "source",
            "aggragated_from",
            "tag",
        ]
        if sorted(self.df.columns.tolist()) != sorted(required_columns):
            raise ValueError(
                f"DataFrame must contain the following columns: {sorted(required_columns)}\n"
                f"Instead, it contains {sorted(self.df.columns.tolist())}"
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
            raise ValueError("a model appears more than once for a single scenario")

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

        return name

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
        )
        return name

    def extend(self, other, model_match_thresh=1.0):
        if not isinstance(other, Benchmark):
            raise TypeError("The added object must be an instance of Benchmark")

        # Fuzzy match and replace model names from other to fit self
        model_map = {}
        n_matches = 0
        for model_name_in_other in other.df["model"].unique():
            best_match, score = process.extractOne(
                model_name_in_other, self.get_models(), scorer=fuzz.token_sort_ratio
            )
            if (
                score / 100 >= model_match_thresh
            ):  # Adjust the threshold based on your matching criteria
                model_map[model_name_in_other] = best_match
                n_matches += 1
                # print(f"{score} - {model_name_in_other}:{best_match}")
            else:
                model_map[
                    model_name_in_other
                ] = model_name_in_other  # Use the new model name as

        # print(f"matched {n_matches}/{len(other.df["model"].unique())}")
        other.df["model"] = other.df["model"].replace(model_map)

        self.df = pd.concat([self.df, other.df])

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
        scenarios_source_to_drop = []
        for scenario, scenario_df in self.df.drop_duplicates(
            ["scenario", "source"]
        ).groupby("scenario"):
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

        self.df = self.df.query("scenario__source not in @scenarios_source_to_drop")
        self.df.drop(
            columns=["scenario__source", "scenario__source_counts"], inplace=True
        )

    def add_tags(self):
        self.df["tag"] = self.df["scenario"].apply(lambda x: benchmark2tag[x])
