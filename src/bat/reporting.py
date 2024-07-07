import seaborn as sns
import matplotlib.pyplot as plt


def plot_experiments_results(agreement_df, cfg):
    sns.set()

    exp_to_run = cfg.exp_to_run

    if exp_to_run == "resolution_matters":
        sns.set(font_scale=1.2, style="white")

        fig, ax = plt.subplots(width_ratios=[1.5])

        # correlation as a function of model_subset_size_requested
        sns.pointplot(
            ax=ax,
            # kind="point",
            data=agreement_df.query('corr_type=="kendall"').replace(
                {
                    "somewhere_aggregate": "Adjacent sampling",
                    "random": "Random sampling",
                }
            ),
            y="correlation",
            x="model_subset_size_requested",
            hue="model_select_strategy",
            markersize=10,
            linewidth=4,
            # legend=False,
            # errorbar="se",
            # linestyle="",
            # col="corr_type",
            # sharey=False,
            # aspect=1.5,
        )
        # scneario-wise agreement (lines)
        sns.pointplot(
            ax=ax,
            # kind="point",
            data=agreement_df.query(
                'corr_type=="kendall"'
                # " and scenario not in @scenarios_not_to_show and ref_scenario not in @scenarios_not_to_show"
                " and model_select_strategy=='somewhere_aggregate'"
            ),
            y="correlation",
            x="model_subset_size_requested",
            hue="scenario",
            errorbar=None,
            alpha=0.2,
            legend=False,
            # aspect=1.5,
            # col="corr_type",
            # aspect=1.5,
        )
        plt.xlabel("Granularity (Number of models)")
        plt.ylabel("Mean Benchmark Agreement\n(Kendall-tau correlation)")
        ax.invert_xaxis()
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, frameon=False)
        # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig("figures/final_for_paper/pointplot_granularity_matters.pdf")
