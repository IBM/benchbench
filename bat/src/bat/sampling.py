import random


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
