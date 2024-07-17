import random
import pandas as pd
from scipy.stats import pearsonr, kendalltau
import numpy as np


def get_pair_agreement(pair_scen_res, res_to_sort_by, cfg, models_intersect):
    # how many models occur in both

    model_subset_size_taken = (
        min(cfg["model_subset_size_requested"], len(models_intersect))
        if cfg["model_subset_size_requested"] != 0
        else len(models_intersect)
    )

    if any(
        [
            x in cfg["model_select_strategy"]
            for x in ["top", "bottom", "middle", "somewhere"]
        ]
    ):
        if cfg["exp_n"] != 0 and "somewhere" not in cfg["model_select_strategy"]:
            return None, None  # skipping experimentation since deterministic

        models_taken = sample_models_directed(
            res_to_sort_by,
            cfg["model_select_strategy"],
            models_intersect,
            model_subset_size_taken,
        )

    elif "random" in cfg["model_select_strategy"]:
        random.seed(cfg["exp_n"])
        models_taken = random.sample(
            models_intersect,
            k=model_subset_size_taken,
        )

    else:
        raise NotImplementedError

    agreement, p_value = get_agreement(
        pair_scen_res[pair_scen_res["model"].isin(models_taken)][
            ["model", "scenario", "score"]
        ],
        cfg["corr_type"],
    )

    return agreement, p_value


def get_df_of_scenario_to_order_by(df, model_select_strategy):
    if "aggregate" in model_select_strategy:
        order_by = "Aggregate"

    elif "arena" in model_select_strategy:
        order_by = "Arena Elo"

    else:
        raise NotImplementedError

    return df[df["scenario"] == order_by]


def sample_models_directed(
    res_to_sort_by,
    model_select_strategy,
    models_intersect,
    n_models_really_taken,
):
    df_of_scenario_to_order_by = res_to_sort_by.query("model in @models_intersect")
    # get_df_of_scenario_to_order_by(
    # bench_res, model_select_strategy
    # )

    if "top" in model_select_strategy:
        models_taken = df_of_scenario_to_order_by.nlargest(
            n_models_really_taken,
            "score",
        )["model"].tolist()
    elif "bottom" in model_select_strategy:
        models_taken = df_of_scenario_to_order_by.nsmallest(
            n_models_really_taken,
            "score",
        )["model"].tolist()

    elif "middle" in model_select_strategy:
        df_sorted = df_of_scenario_to_order_by.sort_values("score", ascending=False)
        middle_idx = len(df_sorted) // 2
        half_n = n_models_really_taken // 2

        if n_models_really_taken % 2 == 0:
            sampled_df = df_sorted.iloc[middle_idx - half_n : middle_idx + half_n]
        else:
            sampled_df = df_sorted.iloc[middle_idx - half_n : middle_idx + half_n + 1]

        models_taken = sampled_df["model"].unique().tolist()

    elif "somewhere":
        df_sorted = df_of_scenario_to_order_by.sort_values("score", ascending=False)

        idx = random.randrange(len(df_sorted) - n_models_really_taken + 1)
        models_taken = (
            df_sorted.iloc[idx : idx + n_models_really_taken]["model"].unique().tolist()
        )

    else:
        raise NotImplementedError

    return models_taken


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
    if corr_type == "pearson":
        corr_func = pearsonr
    elif corr_type == "kendall":
        corr_func = kendalltau
    else:
        raise IOError(f"corr_type {corr_type} is not supported")

    pivot_df = df.pivot(
        index="model",
        columns="scenario",
        values="score",
    )

    similarity = pivot_df.corr(method=lambda x, y: corr_func(x, y)[0]).iloc[0, 1]
    p_value = (
        pivot_df.corr(method=lambda x, y: corr_func(x, y)[1])
        - np.eye(len(pivot_df.columns))
    ).iloc[0, 1]

    return similarity, p_value
