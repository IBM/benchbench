import seaborn as sns
import matplotlib.pyplot as plt

import os
import pandas as pd


# def plot_experiments_results(agreement_df, cfg):
#     sns.set()

#     exp_to_run = cfg.exp_to_run

#     if exp_to_run == "resolution_matters":
#         sns.set(font_scale=1.2, style="white")

#         fig, ax = plt.subplots(width_ratios=[1.5])

#         # correlation as a function of model_subset_size_requested
#         sns.pointplot(
#             ax=ax,
#             # kind="point",
#             data=agreement_df.query('corr_type=="kendall"').replace(
#                 {
#                     "somewhere_aggregate": "Adjacent sampling",
#                     "random": "Random sampling",
#                 }
#             ),
#             y="correlation",
#             x="model_subset_size_requested",
#             hue="model_select_strategy",
#             markersize=10,
#             linewidth=4,
#             # legend=False,
#             # errorbar="se",
#             # linestyle="",
#             # col="corr_type",
#             # sharey=False,
#             # aspect=1.5,
#         )
#         # scneario-wise agreement (lines)
#         sns.pointplot(
#             ax=ax,
#             # kind="point",
#             data=agreement_df.query(
#                 'corr_type=="kendall"'
#                 # " and scenario not in @scenarios_not_to_show and ref_scenario not in @scenarios_not_to_show"
#                 " and model_select_strategy=='somewhere_aggregate'"
#             ),
#             y="correlation",
#             x="model_subset_size_requested",
#             hue="scenario",
#             errorbar=None,
#             alpha=0.2,
#             legend=False,
#             # aspect=1.5,
#             # col="corr_type",
#             # aspect=1.5,
#         )
#         plt.xlabel("Granularity (Number of models)")
#         plt.ylabel("Mean Benchmark Agreement\n(Kendall-tau correlation)")
#         ax.invert_xaxis()
#         handles, labels = ax.get_legend_handles_labels()
#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(handles=handles, labels=labels, frameon=False)
#         # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#         plt.tight_layout()
#         plt.savefig("figures/final_for_paper/pointplot_granularity_matters.pdf")


class Reporter:
    def __init__(self) -> None:
        os.makedirs("figures", exist_ok=True)

    @staticmethod
    def draw_agreements(agreements, ref_sources=None, scenario_sources=None):
        sns.set_theme(font_scale=1.2)

        filtered_agreements = Reporter.filter_with_sources(
            agreements, ref_sources, scenario_sources
        )

        # Grouping and calculating mean for 'correlation' and 'p_value'
        grouped = (
            filtered_agreements.groupby(["scenario", "ref_scenario"])
            .agg(
                correlation_mean=("correlation", "mean"),
                p_value_mean=("p_value", "mean"),
            )
            .reset_index()
        ).dropna()

        sns.barplot(
            data=grouped.sort_values("correlation_mean"),
            x="ref_scenario",
            y="correlation_mean",
        )
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("figures/temp.png")
        plt.clf()

        # Pivoting the data
        # correlation_pivot = grouped[
        #     ["scenario", "ref_scenario", "correlation_mean"]
        # ].pivot(index="scenario", columns="ref_scenario")
        # p_value_pivot = grouped[["scenario", "ref_scenario", "p_value_mean"]].pivot(
        #     index="scenario", columns="ref_scenario"
        # )

        # # Creating a combined annotation DataFrame
        # combined_annotations = (
        #     correlation_pivot["correlation_mean"].round(2).astype(str)
        #     + " | "
        #     + p_value_pivot["p_value_mean"].round(2).astype(str)
        # )

        # # plt.figure(figsize=(28, 20))
        # sns.heatmap(
        #     correlation_pivot,
        #     annot=True,  # combined_annotations,
        #     fmt="",  # Treat annotations as strings
        #     cmap="coolwarm",  # Adjust color map as needed
        #     xticklabels=True,
        #     yticklabels=True,
        # )
        # plt.show()

    @staticmethod
    def filter_with_sources(agreements, ref_sources, scenario_sources):
        if not scenario_sources and not ref_sources:  # use all
            scenario_sources = agreements["scenario_source"].unique().tolist()
            ref_sources = agreements["ref_source"].unique().tolist()

        elif scenario_sources and not ref_sources:
            ref_sources = [
                scen
                for scen in agreements["ref_source"].unique().tolist()
                if scen not in scenario_sources
            ]

        elif scenario_sources and ref_sources:
            pass

        else:
            raise NotImplementedError

        filtered_agreements = agreements.query(
            "scenario_source in @scenario_sources and ref_source in @ref_sources"
        )

        return filtered_agreements
        # plt.tight_layout()
        # plt.savefig("figures/newbench_cluster_within.png")
        # print("figure saved to figures/newbench_heatmap_within.png")
        # plt.clf()

    @staticmethod
    def get_all_z_scores(agreements, aggragate_name="aggregate"):
        z_scores = []
        for observed_scenario in agreements["scenario"].unique():
            if (
                observed_scenario == aggragate_name
                or len(
                    agreements.dropna().query(
                        "scenario==@observed_scenario"
                        " and ref_scenario==@aggragate_name"
                    )
                )
                == 0
            ):
                continue

            z_score, corr_with_agg, p_value_of_corr_with_agg = Reporter.get_z_score(
                agreements=agreements,
                observed_scenario=observed_scenario,
                aggragate_name="aggregate",
            )

            z_scores.append(
                {
                    "scenario": observed_scenario,
                    "z_score": z_score,
                    "corr_with_agg": corr_with_agg,
                    "p_value_of_corr_with_agg": p_value_of_corr_with_agg,
                    "source": agreements.query("scenario==@observed_scenario")[
                        "scenario_source"
                    ].iloc[0],
                }
            )

        return pd.DataFrame(z_scores)

    @staticmethod
    def get_z_score(
        agreements,
        observed_scenario,
        aggragate_name="aggregate",
        blacklist_sources=[],
    ):
        if (
            not len(
                agreements.dropna().query(
                    "scenario==@observed_scenario" " and ref_scenario==@aggragate_name"
                )
            )
            > 0
        ):
            raise IOError

        ref_agreements_with_agg = (
            agreements.dropna()
            .query(
                "scenario_source not in @blacklist_sources"
                " and ref_scenario==@aggragate_name"
            )
            .groupby(["scenario"])
            .agg(
                correlation_mean=("correlation", "mean"),
                p_value_mean=("p_value", "mean"),
            )
        )

        obs_with_agg = agreements.query(
            "scenario==@observed_scenario" " and ref_scenario==@aggragate_name"
        ).agg(
            correlation_mean=("correlation", "mean"),
            p_value_mean=("p_value", "mean"),
        )

        obs_agreements_with_agg = float(obs_with_agg.iloc[0, 0])
        obs_agreements_with_agg_p_value = float(obs_with_agg.iloc[1, 1])

        ref_mean = ref_agreements_with_agg["correlation_mean"].mean()
        ref_std = ref_agreements_with_agg["correlation_mean"].std()
        z_score = float((obs_agreements_with_agg - ref_mean) / ref_std)

        # # Create the plot
        # plt.figure(figsize=(10, 6))
        # sns.kdeplot(
        #     ref_agreements_with_agg,
        #     x="correlation_mean",
        #     color="blue",
        #     alpha=0.7,
        #     label="Reference Values (KDE)",
        # )
        # plt.axvline(
        #     x=ref_mean,
        #     color="red",
        #     linestyle="dashed",
        #     linewidth=2,
        #     label=f"Mean = {ref_mean:.2f}",
        # )

        # # Mark the observation
        # plt.axvline(
        #     x=obs_agreements_with_agg,
        #     color="green",
        #     linestyle="dashed",
        #     linewidth=2,
        #     label=f"Observation (z={z_score:.2f})",
        # )

        # # Add legend and labels
        # plt.legend()
        # plt.xlabel("Values")
        # plt.ylabel("Density")
        # plt.title("Distribution of Reference Values and Observation with KDE")

        # # Show plot
        # plt.savefig("figures/temp.png")

        return z_score, obs_agreements_with_agg, obs_agreements_with_agg_p_value
