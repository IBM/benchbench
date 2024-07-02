from fuzzywuzzy import process, fuzz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Benchmark:
    def __init__(self, df):
        df["model"] = df["model"].apply(self.standardize_model_name)
        df["scenario"] = df["scenario"].apply(self.standardize_scenario_name)
        df["aggragated_from"] = [[] for _ in range(len(df))]
        self.validate_dataframe(df)
        # df = self.normalize_scores_per_scenario(df)
        df.dropna(inplace=True)
        self.df = df

    @staticmethod
    def normalize_scores_per_scenario(df):
        """
        Normalize the 'score' column in the DataFrame to a 0-1 range within each scenario.

        Parameters:
        df (pd.DataFrame): DataFrame containing 'scenario', 'model', and 'score' columns.

        Returns:
        pd.DataFrame: DataFrame with the 'score' column normalized within each scenario.
        """
        if "score" not in df.columns:
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

        return df.groupby("scenario", as_index=False, group_keys=False).apply(normalize)

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

    @staticmethod
    def validate_dataframe(df):
        required_columns = ["model", "scenario", "score", "source", "aggragated_from"]
        if sorted(df.columns.tolist()) != sorted(required_columns):
            raise ValueError(
                f"DataFrame must contain the following columns: {required_columns}"
            )

        if (
            not len(
                df[df.duplicated(subset=["model", "scenario", "source"], keep=False)]
            )
            == 0
        ):
            raise ValueError("a model appears more than once for a single scenario")

        if not pd.api.types.is_numeric_dtype(df["score"]):
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
        sns.heatmap(
            scenario_combinations,
            cmap="coolwarm",
            vmax=20,
            linewidths=0.002,
            xticklabels=True,
            yticklabels=True,
        )  # , fmt="d", annot=True)

        plt.title("Heatmap of Model Count for Each Pair of Scenarios")
        plt.show()

    def clear_repeated_scenarios(self):
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
                scenarios_source_to_drop.extend(
                    list(
                        scenario_df.sort_values(
                            "scenario__source_counts", ascending=False
                        )["scenario__source"][1:]
                    )
                )

        print("Dropped scenarios: " + "\n".join(scenarios_source_to_drop))
        self.df = self.df.query("scenario__source not in @scenarios_source_to_drop")
        self.df.drop(
            columns=["scenario__source", "scenario__source_counts"], inplace=True
        )
