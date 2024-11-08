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
    def draw_agreements_for_one_source(
        agreements, source_of_interest, ref_sources=None
    ):
        filtered_agreements = Reporter.filter_with_sources(
            agreements, ref_sources, source_of_interest
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

        sns.set_theme(font_scale=1.2)

        g = sns.catplot(
            kind="bar",
            data=grouped.sort_values("correlation_mean"),
            x="ref_scenario",
            y="correlation_mean",
            # palette="viridis",  # or any other Seaborn palette
            edgecolor=".2",  # Add edge color for better visibility
            linewidth=1,  # Adjust line width
            # width=2,
            aspect=1.8,
            # legend=True,
        )
        plt.xticks(rotation=90, fontsize=10)  # Adjust fontsize
        plt.xlabel("Reference Scenario", fontsize=12)  # Add labels with fontsize
        plt.ylabel("Mean Correlation", fontsize=12)
        plt.title(
            f"Mean Agreement Between {source_of_interest} and All other Benchmark",
            fontsize=14,
        )  # Add title

        plt.tight_layout()
        plt.show(block=True)
        # plt.savefig("figures/temp.png")

    @staticmethod
    def draw_agreement_matrix(agreements, sources_hide=None):
        filtered_agreements = Reporter.filter_with_sources(
            agreements, sources_hide, sources_hide
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

        # Pivoting the data
        correlation_pivot = grouped[
            ["scenario", "ref_scenario", "correlation_mean"]
        ].pivot(index="scenario", columns="ref_scenario")
        p_value_pivot = grouped[["scenario", "ref_scenario", "p_value_mean"]].pivot(
            index="scenario", columns="ref_scenario"
        )

        plt.figure(figsize=(10, 8))  # Increase figure size for better visualization

        sns.heatmap(
            correlation_pivot["correlation_mean"].round(2),
            annot=True,  # combined_annotations,
            fmt=".2f",  # Format annotations to two decimal places
            cmap="coolwarm",  # Adjust color map as needed
            center=0,  # Center the colormap around 0 for better contrast
            linewidths=0.5,  # Add lines between cells for better separation
            linecolor="lightgray",  # Set line color to light gray
        )
        plt.xticks(
            rotation=90, fontsize=10
        )  # Rotate x-axis labels for better readability
        plt.yticks(fontsize=10)  # Adjust y-axis label font size
        plt.xlabel("Reference Scenario", fontsize=12)  # Add labels with fontsize
        plt.ylabel("Scenario", fontsize=12)  # Add y-axis label
        plt.title("Mean Benchmark Agreement Across Scenarios", fontsize=14)  # Add title
        plt.tight_layout()
        plt.show(block=True)

    @staticmethod
    def filter_with_sources(agreements, ref_sources_to_keep, scenario_sources_to_keep):
        if not scenario_sources_to_keep and not ref_sources_to_keep:  # use all
            scenario_sources_to_keep = agreements["scenario_source"].unique().tolist()
            ref_sources_to_keep = agreements["ref_source"].unique().tolist()

        elif scenario_sources_to_keep and not ref_sources_to_keep:
            ref_sources_to_keep = [
                scen
                for scen in agreements["ref_source"].unique().tolist()
                if scen not in scenario_sources_to_keep
            ]

        elif scenario_sources_to_keep and ref_sources_to_keep:
            pass

        else:
            raise NotImplementedError

        filtered_agreements = agreements.query(
            "scenario_source in @scenario_sources_to_keep and ref_source in @ref_sources_to_keep"
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

            (
                z_score,
                corr_with_agg,
                p_value_of_corr_with_agg,
                n_models_of_corr_with_agg,
            ) = Reporter.get_z_score(
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
                    "n_models_of_corr_with_agg": n_models_of_corr_with_agg,
                    "source": agreements.query("scenario==@observed_scenario")[
                        "scenario_source"
                    ].iloc[0],
                }
            )

        return pd.DataFrame(z_scores).sort_values('z_score')

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
                n_models_mean=("model_subset_size_requested", "mean"),
            )
        )

        obs_with_agg = agreements.query(
            "scenario==@observed_scenario" " and ref_scenario==@aggragate_name"
        ).agg(
            correlation_mean=("correlation", "mean"),
            p_value_mean=("p_value", "mean"),
            n_models_mean=("model_subset_size_requested", "mean"),
        )

        obs_agreements_with_agg = float(obs_with_agg.iloc[0, 0])
        obs_agreements_with_agg_p_value = float(obs_with_agg.iloc[1, 1])
        obs_agreements_with_agg_n_models = float(obs_with_agg.iloc[2, 2])

        ref_mean = ref_agreements_with_agg["correlation_mean"].mean()
        ref_std = ref_agreements_with_agg["correlation_mean"].std()
        z_score = float((obs_agreements_with_agg - ref_mean) / ref_std)

        return (
            z_score,
            obs_agreements_with_agg,
            obs_agreements_with_agg_p_value,
            obs_agreements_with_agg_n_models,
        )
