import random
from .sampling import sample_models_directed
from .utils import get_agreement


def get_pair_agreement(pair_scen_res, res_to_sort_by, cfg, models_intersect):
    # how many models occur in both

    model_subset_size_taken = min(
        cfg["model_subset_size_requested"], len(models_intersect)
    )

    if any(
        [
            x in cfg["model_select_strategy"]
            for x in ["top", "bottom", "middle", "somewhere"]
        ]
    ):
        if cfg["exp_n"] != 0 and "somewhere" not in cfg["model_select_strategy"]:
            return None, None  # skipping experimentation since determinstic

        models_taken = sample_models_directed(
            res_to_sort_by,
            cfg["model_select_strategy"],
            models_intersect,
            model_subset_size_taken,
        )

    elif "random" in cfg["model_select_strategy"]:
        # models_intersect_date_subset = models_intersect.copy()
        # if cfg["model_select_strategy"] == "date_random":
        #     # remove models after the threshold

        #     date_threshold = cfg["date_threshold"]

        #     models_intersect_date_subset = (
        #         pair_scen_res.query(
        #             f'date<="{date_threshold}" and model in @models_intersect'
        #         )["model"]
        #         .unique()
        #         .tolist()
        #     )

        #     model_subset_size_taken = min(
        #         model_subset_size_taken,
        #         len(models_intersect_date_subset),
        #     )

        random.seed(cfg["exp_n"])
        models_taken = random.sample(
            models_intersect,
            k=model_subset_size_taken,
        )

    else:
        raise NotImplementedError

    agreement = get_agreement(
        pair_scen_res[pair_scen_res["model"].isin(models_taken)][
            ["model", "scenario", "score"]
        ],
        cfg["corr_type"],
    )

    # a = pair_scen_res[pair_scen_res["model"].isin(models_taken)][
    #     ["model", "score", "source"]
    # ]
    # a.pivot(index="model", columns="source").corr()
    return agreement, models_taken
