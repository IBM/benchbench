import pandas as pd
import yaml
import numpy as np
import random
import json

model_family2release = {
    "BIG-G sparse": "2022/06/01",
    "BIG-G T=0": "2022/06/01",
    "BIG-G T=1": "2022/06/01",
    "PaLM": "2022/04/01",
    "GPT": "2020/05/28",
    "Gopher": "2021/12/01",
}


def sample_sublists_for_list(
    all_models_sorted, sublists_size=1, n_sublists=0, drop_from_top=False
):
    # assert not (
    #     drop_from_top and sublists_size != len(all_models_sorted)
    # ), "drop from top defines the length of resulting"

    if drop_from_top:
        sublists = []
        top_models_to_remove = []
        for window_num, model in enumerate(all_models_sorted):
            sublists.append(
                [
                    model
                    for model in all_models_sorted[: sublists_size + window_num]
                    if model not in top_models_to_remove
                ]
            )
            top_models_to_remove.append(sublists[-1][0])  # drop the first model
            if len(sublists) == n_sublists:
                break

    else:
        random.seed(0)
        sublists = []
        for _ in range(n_sublists):
            sublists.append(random.sample(all_models_sorted, sublists_size))

    return sublists


def calculate_win_rate(series):
    assert len(series) > 1, "no meaning for a win rate with only one object"

    def win_rate(x):
        win_count = sum(1 for value in series if x > value)
        return win_count / (len(series) - 1)

    return series.transform(win_rate)


def is_a_helm_benchmark(data_path):
    return any(
        [
            bench_name in data_path
            for bench_name in ["heim", "helm_classic", "helm_lite"]
        ]
    )


def load_data(
    data_path,
    melt=True,
    allowed_n_model_deviation_percent_for_scenario=100,
    max_num_scenarios=9999,
    drop_aggragated_scores=True,
    metadata_to_add=[],
    scenario_blacklist=[],
):
    bench_data = pd.read_csv(data_path)
    identifier_fields = ["model"]

    if "oc" in data_path:
        bench_data.rename(
            columns={"Model": "model", "Release": "release"}, inplace=True
        )
        identifier_fields.append("release")

    elif "bigbench" in data_path:
        bench_data["model"] = (
            bench_data["model_family"] + ":" + bench_data["model_name"]
        )
        bench_data.rename(columns={"task": "scenario"}, inplace=True)

        bench_data = (
            bench_data.groupby(["scenario", "model"])
            .agg({"score": "mean"})
            .reset_index()
        )  # average over n_shot

        bench_data = bench_data[
            ~bench_data["model"].str.contains("sparse")
        ]  # dropping the sparse models
        bench_data = bench_data[
            ~bench_data["model"].str.contains("T=1")
        ]  # dropping T=1 models (worst according the paper)

        bench_data["wr"] = bench_data.groupby(["scenario"])["score"].transform(
            calculate_win_rate
        )
        mwr = (
            bench_data.groupby(["model"])["wr"]
            .mean()
            .to_frame()
            .reset_index()
            .rename(columns={"wr": "Mean win rate"})
        )

        bench_data = bench_data.pivot(
            index=["model"], columns="scenario", values="score"
        ).reset_index()
        bench_data = bench_data.merge(mwr, on=["model"])

        # import seaborn as sns
        # import matplotlib.pyplot as plt

        # cdf = df.drop(columns=["model"]).corr(method="kendall")
        # category_ordered = cdf['Mean win rate'].sort_values(ascending=False).index.tolist()

        # category_ordered.remove("Mean win rate")
        # category_ordered.insert(0, "Mean win rate")

        # cdf = cdf[category_ordered]
        # cdf = cdf.loc[category_ordered].dropna()

        # sns.clustermap(cdf, cmap="viridis")
        # plt.show()

    elif "openllm" in data_path:
        bench_data = bench_data.replace(
            {col: col.replace("__", "/") for col in bench_data.columns}
        )
        bench_data = bench_data.rename(
            columns={
                "Average ⬆️": "Average",
                "model_name_for_query": "model",
                "Type": "tune_type",
            },
        )

        dataset_names = [
            "Average",
            "ARC",
            "HellaSwag",
            "MMLU",
            "TruthfulQA",
            "Winogrande",
            "GSM8K",
            # "DROP", # drop was removed from the benchmark
        ]

        identifier_fields.append("tune_type")

        bench_data = bench_data[dataset_names + identifier_fields]
        bench_data.drop_duplicates(inplace=True)

        bench_data = bench_data.sort_values(by="Average", ascending=False)

    elif "BLZ_data" in data_path:
        bench_data.rename(columns={"Model": "model"}, inplace=True)
        dataset_names = [
            col_name for col_name in bench_data.columns if col_name != "model"
        ]

    elif "yall" in data_path:
        bench_data.drop(columns=["URL", "Likes", "Tags"], inplace=True)
        bench_data.rename(columns={"Model": "model"}, inplace=True)

    elif is_a_helm_benchmark(data_path):
        bench_data = normalize_helm_models_and_scenarios(bench_data)

    else:
        raise NotImplementedError(f"current data_path={data_path} is not supported")

    # by now all that is left should be datasets
    dataset_names = [col_name for col_name in bench_data.columns if col_name != "model"]

    # expand along the datasets
    bench_data = pd.melt(
        bench_data,
        id_vars=identifier_fields,
        var_name="scenario",
        value_vars=dataset_names,
        value_name="score",
    ).reset_index(drop=True)

    # duplicated submission in openllm
    if "openllm" in data_path:
        bench_data.sort_values(["model", "scenario", "tune_type"]).drop_duplicates(
            subset=["model", "scenario"], keep="first"
        )  # dropping the models that has more than one tune_type

    # add category column to oc2
    elif "oc" in data_path:
        if "category" in metadata_to_add:
            scen2cat = json.load(
                open("/Users/yotamp/repos/eval-by-proxy/data/oc/scen2cat", "r")
            )
            bench_data["category"] = bench_data["scenario"].apply(
                lambda scen: scen2cat[scen]
            )
        model_blacklist = ["InternLM2-Chat-20B", "InternLM2-Chat-7B"]
        bench_data = bench_data[~bench_data["model"].isin(model_blacklist)]

    elif "bigbench" in data_path:
        bench_data = bench_data.query("scenario!='Mean win rate'").copy()
        if "category" in metadata_to_add:
            scen2cat = json.load(
                open("/Users/yotamp/repos/eval-by-proxy/data/bigbench/scen2cat", "r")
            )
            bench_data["category"] = bench_data["scenario"].apply(
                lambda scen: scen2cat[scen]
            )
        if "desc_cluster" in metadata_to_add:
            scen2desc_cluster = json.load(
                open(
                    "/Users/yotamp/repos/eval-by-proxy/data/bigbench/scen2desc_cluster",
                    "r",
                )
            )
            bench_data["desc_cluster"] = bench_data["scenario"].apply(
                lambda scen: str(scen2desc_cluster[scen])
            )
        if "release" in metadata_to_add:
            bench_data["release"] = bench_data["model"].apply(
                lambda x: model_family2release[x.split(":")[0]]
            )

    bench_data.scenario = bench_data.scenario.apply(
        lambda x: x.replace("Average", "agg_score").replace(
            "Mean win rate", "agg_score"
        )
    )

    # drop the aggragation score column
    if drop_aggragated_scores:
        bench_data = bench_data[~bench_data["scenario"].str.contains("agg_score")]

    # normalization
    bench_data = normalize_scores_by_benchmark(data_path, bench_data)

    # make scores numeric
    bench_data["score"] = bench_data["score"].apply(
        lambda x: pd.to_numeric(x, errors="coerce")
    )

    # only keep scenarios that has more than X% of the models
    if allowed_n_model_deviation_percent_for_scenario < 100:
        bench_data = drop_scenarios_with_less_model_than_dev(
            allowed_n_model_deviation_percent_for_scenario, bench_data
        )

    # restrict the number of scenarios (specifically for bigbench)
    if max_num_scenarios < len(bench_data["scenario"].unique()):
        selected_scenarios = random.sample(
            bench_data["scenario"].unique().tolist(), k=max_num_scenarios
        )

        bench_data = bench_data[bench_data["scenario"].isin(selected_scenarios)]

    bench_data.loc[:, "wr"] = bench_data.groupby("scenario")["score"].transform(
        calculate_win_rate
    )

    if not melt:
        raise NotImplementedError("need to implement a pivot here")

    bench_data = bench_data[~bench_data["scenario"].isin(scenario_blacklist)]

    return bench_data


def drop_scenarios_with_less_model_than_dev(
    allowed_n_model_deviation_percent_for_scenario, df
):
    n_models = len(df["model"].unique())

    scenarios_to_include = []
    # run over scenarios and check how many models there are
    for scenario, group in df.groupby("scenario"):
        n_models_in_scen = len(group.dropna()["model"].unique())

        if (
            n_models_in_scen
            + round(n_models * (allowed_n_model_deviation_percent_for_scenario / 100))
            >= n_models
        ):
            scenarios_to_include.append(scenario)

    df = df[df["scenario"].isin(scenarios_to_include)]
    return df


def normalize_scores_by_benchmark(data_path, df):
    if any([bench in data_path for bench in ["openllm", "oc", "yall"]]):
        df.score = df.score.apply(lambda x: x / 100)

    elif "heim" in data_path:
        df.score = df.apply(
            lambda row: (
                row["score"] / 5 if row.scenario != "agg_score" else row["score"]
            ),
            axis=1,
        )

    # these are already normalized
    elif any([bench in data_path for bench in ["helm_classic", "helm_lite"]]):
        pass  # already between 0 and 1

    # these require normalization by row
    elif any(
        [
            bench in data_path
            for bench in [
                "BLZ_data",
                "bigbench",
            ]
        ]
    ):
        df["score"] = df["score"].astype("float")

        normalized_dfs = []
        for scenario, scenario_df in df.groupby("scenario"):
            max_score = scenario_df["score"].max()
            if 0 <= max_score <= 1:
                normalize_by = 1  # for completeness
            elif 0 <= max_score <= 100:
                normalize_by = 100
            elif max_score > 100:
                normalize_by = max_score
            else:
                raise NotImplementedError("Cant normalize this")

            scenario_df["score"] = scenario_df["score"].apply(
                lambda x: x / normalize_by if isinstance(x, float) else np.nan
            )

            scenario_df["scenario"] = scenario
            normalized_dfs.append(scenario_df)

        df = pd.concat(normalized_dfs)
        df = df.dropna(subset=["score"])

    else:
        raise NotImplementedError(
            f"{data_path} is not supported, specifically, proper normalization was not defined"
        )

    return df


def normalize_helm_models_and_scenarios(df):
    display_real_names_dict = {
        model_dict["display_name"]: model_dict["name"]
        for model_dict in yaml.safe_load(
            open("data/metadata/helm_model_mapping.yaml", "r")
        )["models"]
    }

    df = df.rename(columns={"Model": "model"})
    # df = df.apply(lambda x: pd.to_numeric(x, errors="coerce"))
    df = df.rename(
        columns={
            col: col.replace(" - Image text alignment (human)", "")
            for col in df.columns
        }
    )

    def replace_display_and_real_names_for_helm(model_name):
        if "GPT-3.5" not in model_name:
            if "text-davinci" in model_name:
                model_name = f"GPT-3.5 ({model_name})"
            elif "gpt-3.5-turbo-" in model_name:
                model_name = f'GPT-3.5 Turbo ({model_name.split("gpt-3.5-turbo-")[-1]})'
        if model_name == "Openjourney v1 (1B)":
            model_name = "Openjourney (1B)"

        if model_name in display_real_names_dict.keys():
            return display_real_names_dict[model_name]
        else:
            print(f"could not find {model_name} in dictionary")
            return model_name.lower().replace(" ", "-")

    df["model"] = df["model"].apply(replace_display_and_real_names_for_helm)
    return df


if __name__ == "__main__":
    supported_data_paths = [
        # "data/helm_classic_240130.csv",
        # "data/bigbench_filtered_fv2.csv", https://github.com/INK-USC/predicting-big-bench/blob/main/data/bigbench/filtered_preferred_only_v2.csv
        # "data/openllm_231128.csv",
        # "data/helm_lite_240130.csv",
        # "data/heim_240130.csv",
        # "data/BLZ_data_240130.csv",
        # "data/yall_240206.csv",  # https://huggingface.co/spaces/mlabonne/Yet_Another_LLM_Leaderboard
        "data/oc_combined_en_240206.csv",
        # safty https://huggingface.co/spaces/AI-Secure/llm-trustworthy-leaderboard
        # code https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard
    ]

    all_dfs = []
    for data_path in supported_data_paths:
        all_dfs.append(
            load_data(
                data_path=data_path,
                melt=True,
                allowed_n_model_deviation_percent_for_scenario=20,
            )
        )
        all_dfs[-1]["benchmark"] = data_path.split("data/")[-1].split(".csv")[0]

    all_data = pd.concat(all_dfs)
    # all_data.to_csv("data/combined_240204.csv")

    # print()


def get_release_date_df():
    cdf = pd.read_csv("data/arena_direct/leaderboard_table_20240326.csv")[
        ["Knowledge cutoff date", "Model"]
    ]
    cdf["date"] = pd.to_datetime(
        cdf["Knowledge cutoff date"], format="%Y/%m", errors="coerce"
    ).tolist()

    cdf = cdf[["Model", "date"]].rename(columns={"Model": "model"})

    return cdf


def add_aggragete_with_mwr(df, scenarios_for_aggragate):
    if "wr" not in df.columns:
        df["wr"] = df.groupby(["scenario"])["score"].transform(calculate_win_rate)

    mean_df = pd.DataFrame(columns=df.columns)
    mean_df = (
        df.query("scenario in @scenarios_for_aggragate")
        .groupby(["model"])
        .agg({"score": "mean", "wr": "mean"})
        .reset_index()
    )
    mean_df["score"] = mean_df["wr"]
    mean_df["scenario"] = "Aggregate"
    df = pd.concat([df, mean_df]).drop(columns=["wr"])
    return df


def get_agreement(df, corr_type):
    corr_matrix = df.pivot_table(
        index="model",
        columns="scenario",
        values="score",
    ).corr(corr_type)

    similarity = corr_matrix.iloc[0, 1]

    return similarity


from fuzzywuzzy import process


# Fuzzy search function
def find_model_matches(list1, list2, threshold=90):
    matches = []
    for model1 in list1:
        match, score = process.extractOne(model1, list2)
        if score >= threshold:
            matches.append((model1, match, score))
    return matches
