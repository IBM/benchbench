import seaborn as sns
import matplotlib.pyplot as plt


def plot_experiments_results(agreement_df, cfg):
    sns.set()

    exp_to_run = cfg.exp_to_run
    agreement_df.replace(
        {
            "Arena Elo": "LMSys Arena",
            "Hugging-6": "HF OpenLLM",
            "Alpaca(v2)": "Alpaca v2",
        },
        inplace=True,
    )

    if exp_to_run == "bluebench":
        bluebench_scenarios = [
            "20_newsgroups",
            "cfpb",
            "attaq_500",
            "billsum",
            "financebench",
            "rag",
            "safety",
            "tldr",
            "universal_ner",
            "mmlu_pro",
        ]
        df = agreement_df.query(
            "scenario in @bluebench_scenarios and ref_scenario not in @bluebench_scenarios"
        )[["scenario", "ref_scenario", "correlation", "exp_n"]]

        sns.pairplot(
            data=df[["scenario", "correlation"]].pivot(columns="scenario"),
            # .iloc[5:25],
            # plot_kws={"sharex": False, "sharey": False},
            # diag_kws={"sharex": False, "sharey": False},
            kind="reg",
            corner=True,
        )

        # .query(
        # "ref_scenario not in @bluebench_scenarios"
        # )

    if exp_to_run == "ablation":
        # show the different results one can get with the different strategies for the same pairs
        # lets start with having just one pair

        # df = agreement_df.query(
        #     'scenario=="arena-hard" and ref_scenario=="mixeval-hard"'
        # )

        ###### Models ##########

        df = agreement_df.copy()
        # df.drop(columns=["corr_type", "models_selected"], inplace=True)

        # only keep pairs that happen more than 10 times
        passed_n_models = df.query("model_subset_size_requested>=10").drop_duplicates(
            subset=["scenario", "ref_scenario"]
        )[["scenario", "ref_scenario"]]
        passed_n_models["passed"] = True
        df = df.merge(passed_n_models)

        # only keep benchmarks have more than 5 references
        how_many_is_common = 5
        common_occuring_scenarios = (
            df.groupby(["scenario"])["ref_scenario"]
            .nunique()[
                df.groupby(["scenario"])["ref_scenario"].nunique() >= how_many_is_common
            ]
            .index.tolist()
        )
        df = df.query("scenario in @common_occuring_scenarios")

        models_df = df.copy()
        models_df["selected_randomly"] = df["model_select_strategy"].apply(
            lambda x: True if x == "random" else False
        )

        # std over: ref_scenario, exp_n, n_models, ~model_select_strategy
        print("models")
        print(
            models_df.groupby(["selected_randomly", "scenario"])["correlation"]
            .std()
            .reset_index()
            .groupby(["selected_randomly"])["correlation"]
            .mean()
        )

        # a negligent author would have one scenario, ref_scenario,
        # we have many and we average

        # one experiment is scenario

        ######## ref benchmarks ##########
        refb_df = df.query('model_select_strategy!="random"')

        print(f'before:{refb_df.groupby(["scenario"])["correlation"].std().mean()}')

        # std over: exp_n, n_models
        refb_df = (
            refb_df.groupby(["scenario", "ref_scenario"])["correlation"]
            .std()
            .reset_index()
        )
        refb_df.rename(columns={"correlation": "corr_std"}, inplace=True)

        print(f'after: {refb_df.groupby(["scenario"])["corr_std"].mean().mean()}')

        # a negligent author would have one scenario, ref_scenario,
        # we have many and we average

        ######### correlation function ############

        cor_df = df.query('model_select_strategy!="random"')

        cor_df = (
            cor_df.query('corr_type=="kendall"')
            .groupby(["scenario"])["correlation"]
            .std()
            .reset_index()
        )
        cor_df.rename(columns={"correlation": "corr_std"}, inplace=True)
        print(f'after: {cor_df.groupby(["scenario"])["corr_std"].mean().mean()}')

        ########## all ########

        # keep only the random and kendall configurations (model_selection and corr_type matters)
        all_df = df.query('model_select_strategy=="random" and corr_type=="kendall"')

        # get STDs
        all_df = (
            all_df.groupby(["scenario", "ref_scenario"])["correlation"]
            .std()
            .reset_index()
        )
        all_df.rename(columns={"correlation": "corr_std"}, inplace=True)

        # average std over ref_scenarios (ref_benchmark_matters)
        add_df = all_df.groupby(["scenario"])["corr_std"].mean()

        # average over scenarios
        print(f"after: {add_df.mean()}")

        ########### min max game ########

        # scenario = "arena-elo"

        df.columns
        # n_models = 12

        for scenario in df.scenario.unique():
            print(f"\n\nScenario: {scenario}")

            print(
                f"min before"
                + df.query('scenario==@scenario and model_select_strategy!="random"')[
                    "correlation"
                ].min()
            )
            print(
                'max before:'+df.query('scenario==@scenario and model_select_strategy!="random"')["correlation"].max()
            )

            # # std over: exp_n, n_models
            # refb_df = (
            #     refb_df.groupby(["scenario", "ref_scenario"])["correlation"]
            #     .std()
            #     .reset_index()
            # )
            # refb_df.rename(columns={"correlation": "corr_std"}, inplace=True)

            # print(
            #     f'min refrence: {refb_df.query('scenario==@scenario')["correlation"].min()}'
            # )
            # print(
            #     f'max refrence: {refb_df.query('scenario==@scenario')["correlation"].max()}'
            # )

            # models_df = df.copy().query('')
            df["selected_randomly"] = df["model_select_strategy"].apply(
                lambda x: True if x == "random" else False
            )

            print(
                'min models:'+
                    df.query('scenario==@scenario and selected_randomly==True').groupby(["scenario"])["correlation"].min().values[0]
                    
            )
            print(
                'max models:'+
                    df.query('scenario==@scenario and selected_randomly==True').groupby(["scenario"])["correlation"].max().values[0]
             
            )

            print(
                'min corr:'+
                    df.query('scenario==@scenario and selected_randomly==False and corr_type=="kendall"').groupby(["scenario"])["correlation"].min().values[0]
                    
            )
            print(
                'max corr:'+
                    df.query('scenario==@scenario and selected_randomly==False and corr_type=="kendall"').groupby(["scenario"])["correlation"].max().values[0]
                    
            )

            print(
                'min all:'+
                    df.query('scenario==@scenario and selected_randomly==True and corr_type=="kendall"').groupby(["scenario"])["correlation"].min().values[0]
                    
            )
            print(
                'max all:'+
                    df.query('scenario==@scenario and selected_randomly==True and corr_type=="kendall" and model_subset_size_requested==@n_models').groupby(["scenario"])["correlation"].max().values[0]
                    
            )

        # how do we reduce the variance with multiple references? we average right?

        # g = sns.catplot(
        #     kind="bar",
        #     data=df,
        #     # .query(
        #     # "scenario in @scenarios_to_show"  # and ref_scenario in @scenarios_to_show"
        #     # ),
        #     col="scenario",
        #     col_wrap=3,
        #     hue="ref_scenario",
        #     y="correlation",
        # )

    if exp_to_run == "ref_bench_quantitative":
        # how does the STD change when we average more and more scenarios
        scenarios_to_show = [
            "arena-hard",
            "mixeval",
            "arena-elo",
            "oc1_mwr",
            "helm_lite_mwr",
            "helm_mwr",
        ]

        sns.move_legend(
            g,
            "upper left",
            bbox_to_anchor=(0.65, 0.3),
            frameon=True,
            title="Reference",
        )

        plt.show()

        # calculate the STDs across ref_scenario for each scenario
        # average over exps
        aggr_df = (
            agreement_df.groupby(["scenario", "ref_scenario"])["correlation"]
            .mean()
            .reset_index()
        )
        # calculate the STDs across ref_scenario for each scenario
        aggr_df = (
            aggr_df.groupby(["scenario"]).agg({"correlation": ["std"]}).reset_index()
        )
        aggr_df.columns = ["scenario", "correlation_std"]

        sns.barplot(
            data=aggr_df.query(
                "scenario in @scenarios_to_show"
            ),  # and ref_scenario in @scenarios_to_show"),
            x="scenario",
            y="correlation_std",
        )

        plt.xticks(rotation=45)
        plt.tight_layout()

        threshold = 0.7

        ##################
        ##################

        print("just keep the scenarios that has more results out of the sources\n" * 5)

        # agreement_df = agreement_df.query(
        #     '(scenario=="arena-hard : arena_hard_2404" or scenario=="mixeval : mixeval_240601")'
        #     ' and (ref_scenario=="mixeval-hard : mixeval_240601" or ref_scenario=="mixeval : mixeval_240601" or ref_scenario== "mmlu-mixed : mixeval_240601")'
        # )
        # agreement_df = agreement_df[
        #     ["scenario", "ref_scenario", "exp_n", "correlation"]
        # ]

        # agreement_df["scenario"] = agreement_df["scenario"].apply(
        #     lambda x: x.split(" : ")[0]
        # )
        # agreement_df["ref_scenario"] = agreement_df["ref_scenario"].apply(
        #     lambda x: x.split(" : ")[0]
        # )

        ##################
        ##################

        agreement_df = agreement_df[
            ["scenario", "ref_scenario", "exp_n", "correlation"]
        ]

        # # get mean correlation by averaging over exps
        # agreement_df["mean_correlation"] = agreement_df.groupby(
        #     ["scenario", "ref_scenario"]
        # )["correlation"].transform("mean")
        # and remove rows of exps
        agreement_df.drop_duplicates(
            ["scenario", "ref_scenario", "exp_n"], inplace=True
        )
        agreement_df = agreement_df[
            ["scenario", "ref_scenario", "exp_n", "correlation"]
        ]

        # check which couple passed the threshold
        agreement_df["pair_passed_threshold"] = agreement_df["correlation"].apply(
            lambda x: 1 if x >= threshold else 0
        )

        # get the majority vote
        agreement_df["passed_mean_over_ref"] = agreement_df.groupby(
            ["scenario", "exp_n"]
        )["pair_passed_threshold"].transform("mean")
        # pass the majority vote through a threshold (0.5)
        agreement_df["passed_majority"] = agreement_df["passed_mean_over_ref"].apply(
            lambda x: 1 if x >= 0.5 else 0
        )

        # # measure agreement to majority
        # agreement_df["error"] = 1 * (
        #     agreement_df["pair_passed_threshold"] != agreement_df["passed_majority"]
        # )
        # # get error rate as the average of the error
        # agreement_df["error_rate_from_ref"] = agreement_df.groupby(
        #     ["scenario", "exp_n"]
        # )["error"].transform("mean")

        agreement_df = agreement_df[
            [
                "scenario",
                "ref_scenario",
                "exp_n",
                "pair_passed_threshold",
                "passed_majority",
                # "error_rate_from_ref",
            ]
        ]

        import pandas as pd
        from tqdm import tqdm

        boot_agreement_df = []

        n_boot = 100
        for n_benchmarks_to_use in tqdm([1, 3, 5, 100]):
            for _, scenario_agg_df in agreement_df.groupby(["scenario", "exp_n"]):
                n_benchmarks_to_use_here = min(
                    n_benchmarks_to_use, len(scenario_agg_df.ref_scenario.unique())
                )
                if n_benchmarks_to_use_here % 2 == 0:
                    n_benchmarks_to_use_here -= 1  # majority vote does not work for two
                for i in range(1, n_boot + 1):
                    bootstraped_scenario_agg_df = scenario_agg_df.sample(
                        n=n_benchmarks_to_use_here, replace=True
                    )
                    bootstraped_scenario_agg_df["boot_exp_n"] = i
                    bootstraped_scenario_agg_df["n_benchmarks_sampled"] = (
                        n_benchmarks_to_use_here
                    )
                    boot_agreement_df.append(bootstraped_scenario_agg_df)

        boot_agreement_df = pd.concat(boot_agreement_df)

        # # check which couple passed the threshold
        # boot_agreement_df["passed"] = boot_agreement_df["mean_correlation"].apply(
        #     lambda x: 1 if x >= threshold else 0
        # )

        # get the majority vote, the mean of the passing score
        boot_agreement_df["passed_mean_over_ref"] = boot_agreement_df.groupby(
            ["scenario", "boot_exp_n", "exp_n", "n_benchmarks_sampled"]
        )["pair_passed_threshold"].transform("mean")
        # pass the majority vote through a threshold (0.5)
        boot_agreement_df["passed_majority_boot"] = boot_agreement_df[
            "passed_mean_over_ref"
        ].apply(lambda x: 1 if x >= 0.5 else 0)

        boot_agreement_df["error_from_boot"] = boot_agreement_df.apply(
            lambda row: 1
            if row["passed_majority_boot"] != row["passed_majority"]
            else 0,
            axis=1,
        )

        # mean error over the bootstrap samples
        boot_agreement_df["error_rate_from_boot"] = boot_agreement_df.groupby(
            ["scenario", "exp_n", "n_benchmarks_sampled"]
        )["error_from_boot"].transform("mean")

        boot_agreement_df.drop_duplicates(
            subset=["scenario", "exp_n", "n_benchmarks_sampled"], inplace=True
        )

        scenarios_not_to_show = [
            "alpacav2",
            "alpacaeval2-lc",
            "helm_mwr",
            "oc2_mwr",
            "mmlu-hard-mixed",
        ]
        x_title = "# of Reference Benchmarks"
        y_title = "Error Rate"

        g = sns.catplot(
            kind="bar",
            data=boot_agreement_df[
                [
                    "scenario",
                    "error_rate_from_boot",
                    "exp_n",
                    "n_benchmarks_sampled",
                ]
            ]
            .query("scenario not in @scenarios_not_to_show")
            .rename(
                columns={
                    "n_benchmarks_sampled": x_title,
                    "error_rate_from_boot": y_title,
                }
            ),
            col="scenario",
            col_wrap=4,
            x=x_title,
            y=y_title,
            # sharey=False,
        )
        sns.move_legend(g, "upper left", bbox_to_anchor=(0.15, 0.95), frameon=True)
        plt.tight_layout()

        # df = df.melt(id_vars="model", var_name="scenario", value_name="score")

        # theres error rate that stems from bootstraping
        # and theres error rate from single dataset selectio

        sns.catplot(
            kind="bar",
            data=a.drop_duplicates(subset=["scenario", "boot_exp_n"]).query(
                "scenario in @scenarios_to_show"
            ),
            x="scenario",
            hue="method",
            # col_wrap=3,
            y="error_rate",
            # hue="ref_scenario",
        )
        plt.xticks(rotation=45)
        plt.tight_layout()

        # agreement_df["passed_mean"] = agreement_df["passed_mean"].apply()

    if exp_to_run == "test":
        # correlation as a function of model_subset_size_requested
        sns.pointplot(
            # kind="point",
            data=agreement_df.query(
                'corr_type=="kendall" and scenario!="Aggregate" and ref_scenario!="Aggregate"'
            ).replace(
                {
                    "somewhere_aggregate": "Contiguous sampling",
                    "random": "Random sampling",
                }
            ),
            y="correlation",
            x="model_subset_size_requested",
            hue="model_select_strategy",
            markersize=10,
            linewidth=4,
            # errorbar="se",
            # linestyle="",
            # col="corr_type",
            # sharey=False,
        )

    if exp_to_run == "corr_metric_matters":
        pivot_df = agreement_df.pivot_table(
            index=[
                "scenario",
                "ref_scenario",
                "model_subset_size_requested",
                "model_select_strategy",
                # "exp_n",
            ],  # , "exp_n"
            columns="corr_type",
            values="correlation",
            aggfunc="mean",
        ).reset_index()

        scenarios_not_to_show = [
            "MAGI",
            "GPT4All",
        ]

        ax = plt.subplot()
        sns.set(
            style="white",
            font_scale=1.5,
        )

        plot_df = pivot_df.query(
            'model_select_strategy=="random"'
            " and model_subset_size_requested==15"
            " and scenario not in @scenarios_not_to_show and ref_scenario not in @scenarios_not_to_show"
        )

        from scipy.stats import linregress

        slope, intercept, r_value, p_value, std_err = linregress(
            plot_df["kendall"], plot_df["pearson"]
        )

        equation = f"Y={round(slope,2)}*X+{round(intercept,2)} (r²={round(r_value,2)})"

        # what are the general corrleations
        g = sns.scatterplot(
            ax=ax,
            data=plot_df,
            x="kendall",
            y="pearson",
            # kind="reg",
            hue="scenario",
            # hue='exp_n',
            # col="scenario",
            # col_wrap=4,
            legend=False,
        )  # , col_w

        sns.regplot(
            ax=ax,
            x="kendall",
            y="pearson",
            data=plot_df,
            scatter=False,
            # legend=False,
            # ax=g.axes[0, 0],
        )

        # Annotating the regression coefficients
        ax.text(0.53, 0.31, equation, fontsize=15)

        ax.set_xlabel("Kendall-tau (rank) correlation")
        ax.set_ylabel("Pearson (score) correlation")

        # sns.move_legend(
        #     g,
        #     "upper left",
        #     bbox_to_anchor=(1.0, 0.9),
        #     frameon=True,
        #     title="Benchmark",
        # )

        plt.tight_layout()
        plt.savefig("figures/final_for_paper/regplot_for_correlation_matters.pdf")

        # correlations for different model_subset_size_requested
        g = sns.lmplot(
            pivot_df.query('model_select_strategy=="random" and scenario!="Aggregate"'),
            x="correlation_kendall",
            y="correlation_pearson",
            col="model_subset_size_requested",
            # col_wrap=4,
            fit_reg=True,
            # hue='exp_n',
            # col="scenario",
            # col_wrap=4,
        )  # , col_w

        # how do the dists look like for all pairs
        sns.set(font_scale=1.2)
        g = sns.displot(
            pivot_df.query(
                'model_select_strategy=="somewhere_aggregate" '
                'and ref_scenario!="Aggregate" and scenario!="Aggregate" '
                "and (model_subset_size_requested==5 or model_subset_size_requested==10 or model_subset_size_requested==15 or model_subset_size_requested==20)"
            ),
            x="kendall",
            hue="model_subset_size_requested",
            kind="kde",
            aspect=1.3,
            fill=True,
            alpha=0.2,
            # bw_adjust=1, cut=0
        )
        plt.xlim(-0.5, 1)
        sns.move_legend(
            g,
            "upper left",
            bbox_to_anchor=(0.15, 0.9),
            frameon=True,
            title="Resolution",
        )
        g.set_xlabels("Correlation (Kendall tau)")

        # how do the dists look like for pairs with aggragate
        sns.set(font_scale=1.2)
        g = sns.displot(
            pivot_df.query(
                'model_select_strategy=="top_aggregate" and (ref_scenario=="Aggregate" or ref_scenario=="LMSys Arena") '
            ),
            x="correlation_kendall",
            hue="model_subset_size_requested",
            kind="kde",
            aspect=1.3,
            fill=True,
            alpha=0.2,
            common_norm=True,
            col="ref_scenario",
            # bw_adjust=1, cut=0
        )
        plt.xlim(-1, 1)
        sns.move_legend(g, "upper left", bbox_to_anchor=(0.15, 0.8), frameon=True)

        threshold_df = (
            agreement_df.query(
                'model_select_strategy=="somewhere_aggregate" and (ref_scenario=="Aggregate" or ref_scenario=="LMSys Arena")'
            )
            .groupby(["model_subset_size_requested", "corr_type", "ref_scenario"])
            .agg({"correlation": ["mean", "std"]})["correlation"]
            .reset_index()
        )
        threshold_df["threshold"] = threshold_df["mean"] - threshold_df["std"]

        print(
            threshold_df.query('ref_scenario=="Aggregate"').sort_values(
                ["corr_type", "model_subset_size_requested"]
            )[["model_subset_size_requested", "corr_type", "threshold"]]
        )
        # compare distributions of kendall and pearson

        # sns.pointplot(
        #     data=corr_matrix.query(
        #         'model_select_strategy=="top_aggregate" and corr_type=="pearson" and ref_scenario=="LMSys Arena"'
        #     ),
        #     x="model_subset_size_requested",
        #     y="correlation",
        #     hue="scenario",
        # )

    if exp_to_run == "regret":
        sns.catplot(
            kind="box",
            data=agreement_df,
            y="correlation",
            x="corr_type",
            col="ref_scenario",
            col_wrap=4,
        )

        pivot_df = agreement_df.pivot_table(
            index=["scenario", "ref_scenario"],  # , "exp_n"
            columns="corr_type",
            values="correlation",
            aggfunc="mean",
        ).reset_index()

        g = sns.jointplot(
            pivot_df,
            x="regret",
            y="pearson",
            kind="reg",
            # hue='exp_n',
            # col="scenario",
            # col_wrap=4,
        )  # , col_w

        pivot_df.groupby(["scenario"])[["regret", "pearson"]].corr().iloc[
            0::2, -1
        ].mean()

        import numpy as np
        from scipy.stats import pearsonr

        corr = pivot_df[["regret", "pearson"]].corr(
            method=lambda x, y: pearsonr(x, y)[0]
        )["pearson"]["regret"]
        pvalues = (
            pivot_df[["regret", "pearson"]].corr(method=lambda x, y: pearsonr(x, y)[1])
            - np.eye(len(pivot_df[["regret", "pearson"]].columns))
        )["pearson"]["regret"]

        print(f"corr: {corr}, pvalue: {pvalues}")

    if exp_to_run == "resolution_matters":
        sns.set(font_scale=1.2, style="white")

        scenarios_not_to_show = [
            "Alpaca(v2, len adj)",
            "Aggregate",
            # "GAlpaca(v2, len adj)PT4All",
        ]

        fig, ax = plt.subplots(width_ratios=[1.5])

        # correlation as a function of model_subset_size_requested
        sns.pointplot(
            ax=ax,
            # kind="point",
            data=agreement_df.query(
                'corr_type=="kendall"'
                " and scenario not in @scenarios_not_to_show and ref_scenario not in @scenarios_not_to_show"
            ).replace(
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
                " and scenario not in @scenarios_not_to_show and ref_scenario not in @scenarios_not_to_show"
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
        plt.ylabel("Mean Benchamark Agreement\n(Kendall-tau correlation)")
        ax.invert_xaxis()
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, frameon=False)
        # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig("figures/final_for_paper/pointplot_granularity_matters.pdf")

        # scenario wise agreement (grid)
        sns.catplot(
            kind="point",
            data=agreement_df.query(
                'model_select_strategy=="top_aggregate" and ref_scenario!="Aggregate"'
            ),
            y="correlation",
            x="model_subset_size_requested",
            hue="corr_type",
            col="scenario",
            col_wrap=4,
            errorbar=None,
            sharey=False,
        )

        # scenario wise n_model_used_in_corr (grid)
        ax = sns.lineplot(
            data=agreement_df.query('model_select_strategy=="top_aggregate"').rename(
                columns={
                    "model_subset_size_requested": "# models (max allowed)",
                    "n_models_in_corr": "# models (possible)",
                }
            ),
            y="# models (possible)",
            x="# models (max allowed)",
            # hue="model_select_strategy",
            hue="scenario",
            # col_wrap=4,
            errorbar=None,
            marker="o",
        )
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()

        # aggragat agreement with rest
        g = sns.catplot(
            kind="point",
            data=agreement_df.query(
                'ref_scenario=="Aggregate" and corr_type=="kendall"'
            ),
            y="correlation",
            x="model_subset_size_requested",
            hue="model_select_strategy",
            col="scenario",
            col_wrap=4,
            errorbar="se",
            sharey=False,
        )
        # plt.xlabel("Resolution")
        g.set_ylabels("Mean Benchamark Pairs Correlation")
        g.set_xlabels("Resolution")

    if exp_to_run == "location_matters":
        sns.set(font_scale=1.2)
        # correlation as a function of model_subset_size_requested
        ax = sns.catplot(
            kind="point",
            data=agreement_df.query(
                'corr_type=="kendall" and scenario!="Aggregate" and ref_scenario!="Aggregate"'
            ),
            y="correlation",
            x="model_subset_size_requested",
            hue="model_select_strategy",
            # errorbar="se",
            # linestyle="",
            # col="corr_type",
            # sharey=False,
        )
        plt.xlabel("Resolution")
        plt.ylabel("Mean Benchamark Pairs Correlation")
        plt.tight_layout()

        # same as above with bars
        sns.catplot(
            kind="bar",
            data=agreement_df.query(
                'corr_type=="kendall"'
                'and scenario!="Aggregate" '
                'and ref_scenario!="Aggregate" '
                "and (model_subset_size_requested==5 or model_subset_size_requested==20)"
            ),
            y="correlation",
            x="model_subset_size_requested",
            hue="model_select_strategy",
            hue_order=[
                # "random",
                "bottom_aggregate",
                "top_aggregate",
                # "somewhere_aggregate",
                "middle_aggregate",
            ][::-1],
            errorbar="se",
            aspect=1.2,
        )
        plt.xlabel("Resolution")

    if exp_to_run == "figure_1":
        import pandas as pd

        sns.set(style="whitegrid")

        ks_to_take = [15, 10, 5]

        agreement_df["model_subset_size_requested"] = pd.Categorical(
            agreement_df["model_subset_size_requested"], categories=[15, 10, 5]
        )

        benchmarks_to_take = [
            "BBH <-> Arena",
            "MMLU <-> Arena",
            "Alpaca v2* <-> Arena",
        ]

        sns.catplot(
            data=agreement_df.replace(
                {
                    "BBH": "BBH <-> Arena",
                    "MMLU": "MMLU <-> Arena",
                    "Alpaca(v2, len adj)": "Alpaca v2* <-> Arena",
                }
            ).query(
                'scenario!="Aggregate" and ref_scenario!="Aggregate"'
                ' and scenario=="LMSys Arena"'
                ' and model_select_strategy=="top_aggregate"'
                " and model_subset_size_requested in @ks_to_take"
                " and ref_scenario in @benchmarks_to_take"
                ' and corr_type=="kendall"'
            ),
            kind="bar",
            y="correlation",
            hue="model_subset_size_requested",
            # hue_order=[15, 10, 5],
            x="ref_scenario",
            # col_wrap=2,
        )

    if exp_to_run == "corr_vs_dates":
        fig, ax = plt.subplots()

        # do benchmark agreement degrade?
        sns.pointplot(
            ax=ax,
            data=agreement_df.query(
                'corr_type=="kendall"',  # and model_select_strategy!="random"'
            ),
            y="correlation",
            x="date_threshold",
            hue="model_select_strategy",
            # col="exp_n",
            # hue='ref_scenario',
            # errorbar="se",
            markersize=10,
            linewidth=4,
        )

        # without random, only
        sns.pointplot(
            ax=ax,
            data=agreement_df.query(
                'model_select_strategy!="random" and corr_type=="kendall"'
            ),
            y="correlation",
            x="date_threshold",
            hue="scenario",
            errorbar=None,
            alpha=0.2,
        )
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.ylim(0.45, 0.85)
        plt.tight_layout()

        ax = sns.catplot(
            kind="point",
            data=agreement_df,  # .query(
            # 'corr_type=="pearson"',  # and model_select_strategy!="random"'
            # ),
            # .query(
            # 'model_select_strategy!="random" and scenario=="LMSys Arena"'
            # ),
            y="correlation",
            x="date_threshold",
            hue="corr_type",
            col="model_select_strategy",
            # hue='ref_scenario',
            errorbar="se",
        )

    if exp_to_run == "reference_benchmark_matters":
        agreement_df["model_subset_size"] = agreement_df["models_selected"].apply(
            lambda x: len(x)
        )

        # scenarios_to_show = [
        #     "AGI Eval ",
        #     "Alpaca(v2)",
        #     # "Alpaca(v2, len adj)",
        #     "LMSys Arena",
        #     "BBH",
        #     # "EQ-Bench(v2)",
        #     "GPT4All",
        #     "Hugging-6",
        #     # "HumanEval",
        #     # "MAGI",
        #     "MMLU",
        #     "MT-bench",
        #     # "Aggregate",
        # ]

        # sns.set_style("white")
        # g = sns.pointplot(
        #     # kind="point",
        #     data=agreement_df.query(
        #         "corr_type=='pearson'"
        #         " and model_subset_size_requested==20"
        #         " and scenario in @scenarios_to_show"
        #         " and ref_scenario in @scenarios_to_show"
        #     )
        #     .groupby(["scenario", "ref_scenario"])["correlation"]
        #     .mean()
        #     .reset_index(),
        #     y="correlation",
        #     x="scenario",
        #     hue="ref_scenario",
        #     linestyles="",
        # )
        # plt.xticks(rotation=45)
        # sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        # plt.tight_layout()

        # print()

        # # find the aggragate

        # # scenarios_to_drop = [
        # #     "Aggregate",
        # #     "MBPP",
        # #     "Length(ch, Alpaca)",
        # #     "OpenComp.",
        # #     "HELM  Lite",
        # #     "GPT4All",
        # # ]

        # print(
        #     agreement_df.query(
        #         "scenario not in @scenarios_to_drop"
        #         " and ref_scenario not in @scenarios_to_drop"
        #         " and model_subset_size_requested==20"
        #     )
        #     .groupby(["scenario"])["correlation"]
        #     .mean()
        #     .sort_values(ascending=False)
        # )

        # from .configs import default_aggregate_scenarios

        scenarios_to_show = [
            "AGI Eval ",
            "Alpaca v2",
            # "Alpaca(v2, len adj)",
            "LMSys Arena",
            "BBH",
            # "EQ-Bench(v2)",
            # "GPT4All",
            "HF OpenLLM",
            # "HumanEval",
            # "MAGI",
            "MMLU",
            "MT-bench",
            # "Aggregate",
        ]

        fig, ax = plt.subplots()
        sns.heatmap(
            ax=ax,
            data=agreement_df.query(
                "corr_type=='kendall'"
                " and model_subset_size_requested==10"
                " and scenario in @scenarios_to_show"
                " and ref_scenario in @scenarios_to_show"
            )[["scenario", "ref_scenario", "correlation", "exp_n"]]
            .reset_index()
            .pivot_table(
                # index=["exp_n", "scenario"],
                index="scenario",
                columns="ref_scenario",
                values="correlation",
            )
            .fillna(1),
            annot=True,
            cmap="viridis",
            fmt=".2f",
            linewidths=0.5,
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        plt.tight_layout()

        plt.savefig(
            "figures/final_for_paper/heatmap_for_reference_benchmark_matters.pdf"
        )

        # find the aggregate
        arena_agreements = (
            agreement_df.query(
                "corr_type=='kendall'"
                " and model_subset_size_requested==10"
                " and scenario=='LMSys Arena'"
            )
            .groupby(["ref_scenario"])["correlation"]
            .mean()
            .reset_index()
            .sort_values("correlation", ascending=False)
        )
        arena_agreements["correlation"] = arena_agreements["correlation"].round(2)
        arena_agreements.to_csv("agreements_to_arena_10_random.csv", index=False)
        # " and scenario in @scenarios_to_show"
        # " and ref_scenario in @scenarios_to_show"

        # # Threshold for color change
        # threshold = 0.75

        # # Create a custom colormap
        # colors = [
        #     "red",
        #     "white",
        #     "green",
        # ]  # Red for below or equal to threshold, green for above threshold
        # from matplotlib.colors import LinearSegmentedColormap

        # cmap = LinearSegmentedColormap.from_list("Custom", colors, N=256)

        # data = (
        #     agreement_df.query(
        #         "corr_type=='pearson'"
        #         " and model_subset_size_requested==20"
        #         " and scenario in @scenarios_to_show"
        #         " and ref_scenario in @scenarios_to_show"
        #     )[["scenario", "ref_scenario", "correlation", "exp_n"]]
        #     .reset_index()
        #     .pivot_table(
        #         # index=["exp_n", "scenario"],
        #         index="scenario",
        #         columns="ref_scenario",
        #         values="correlation",
        #     )
        # )

        # norm = plt.Normalize(
        #     0.5, 1
        # )  # Normalize the color scale based on possible correlation values (0 to 1)

        # sns.clustermap(
        #     data.fillna(1),
        #     annot=True,
        #     fmt=".3f",
        #     cmap=cmap,
        #     norm=norm,
        #     cbar_kws={"ticks": [0, threshold, 1]},
        # )
        # plt.tight_layout()

        # n_models = 20

        # sns.kdeplot(
        #     data=agreement_df.query("ref_scenario=='Aggregate'").query(
        #         "model_subset_size_requested==@n_models"
        #     ),
        #     hue="scenario",
        #     x="correlation",
        #     common_norm=True,
        # )

        # sns.kdeplot(
        #     data=agreement_df.query("ref_scenario=='Aggregate'").query(
        #         "model_subset_size_requested==@n_models"
        #     ),
        #     # hue="scenario",
        #     x="correlation",
        #     common_norm=True,
        # )

        # # each ref_benchmark should have his own threshold

        # scenarios_to_show = [
        #     # "AGI Eval ",
        #     # "Alpaca(v2)",
        #     "Alpaca(v2, len adj)",
        #     # "LMSys Arena",
        #     # "BBH",
        #     # "EQ-Bench(v2)",
        #     "GPT4All",
        #     # "Hugging-6",
        #     # "HumanEval",
        #     # "MAGI",
        #     "MMLU",
        #     "MT-bench",
        #     "Aggregate",
        # ]
        # # ref_scenarios_to_show = [
        # #     "AGI Eval ",
        # #     # "Alpaca(v2)",
        # #     "Alpaca(v2, len adj)",
        # #     "LMSys Arena",
        # #     "BBH",
        # #     # "EQ-Bench(v2)",
        # #     "GPT4All",
        # #     "Hugging-6",
        # #     "HumanEval",
        # #     "MAGI",
        # #     "MMLU",
        # #     "MT-bench",
        # #     "Aggregate",
        # # ]

        # sns.set_style("white")
        # g = sns.pointplot(
        #     # kind="point",
        #     data=agreement_df.query(
        #         "corr_type=='pearson'"
        #         " and model_subset_size_requested==10"
        #         " and scenario in @scenarios_to_show"
        #         " and ref_scenario in @scenarios_to_show"
        #     )
        #     .groupby(["scenario", "ref_scenario"])["correlation"]
        #     .mean()
        #     .reset_index(),
        #     y="correlation",
        #     x="scenario",
        #     hue="ref_scenario",
        #     linestyles="",
        # )
        # plt.xticks(rotation=45)
        # sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        # plt.tight_layout()

        # ### significance testing:
        # agreement_df.to_csv(
        #     "/Users/yotamperlitz/repos/eval-by-proxy/agreement_testing_playground.csv"
        # )
        # threshhold = (
        #     agreement_df.groupby(["ref_scenario", "model_subset_size_requested"])
        #     .agg(
        #         {
        #             "correlation": [
        #                 "mean",
        #                 "std",
        #             ]
        #         }
        #     )
        #     .reset_index()
        # )

        # threshhold.columns = [
        #     col[0] if col[1] == "" else f"{col[0]}_{col[1]}" for col in threshhold.columns
        # ]

        # threshhold["threshold"] = (
        #     threshhold["correlation_mean"] - threshhold["correlation_std"]
        # )

        # threshhold.groupby("model_subset_size_requested").agg({"correlation_mean": "mean"})

        # results = []
        # for benchmark in agreement_df.scenario.unique().tolist():
        #     if benchmark == "Aggregate":
        #         continue
        #     threshold_no_mt = threshhold.query("ref_scenario=='Aggregate'")
        #     agreement_only_mt = agreement_df[
        #         ["scenario", "ref_scenario", "model_subset_size_requested", "correlation"]
        #     ].query("scenario==@benchmark")

        #     import pandas as pd

        #     merged_df = pd.merge(
        #         agreement_only_mt,
        #         threshold_no_mt,
        #         on=["ref_scenario", "model_subset_size_requested"],
        #     ).query("model_subset_size_requested==20")

        #     # Perform the paired t-test
        #     from scipy.stats import ttest_rel

        #     t_stat, p_value = ttest_rel(merged_df["correlation"], merged_df["threshold"])
        #     results.append(
        #         {
        #             "benchmark": benchmark,
        #             "t_stat": t_stat,
        #             "p_value": p_value,
        #         }
        #     )

    if exp_to_run == "bench_bench":
        # agreement_df.to_csv("bench_bench_cache.csv")

        import pandas as pd

        # print(default_aggregate_scenarios)

        # with p_value
        main_reference_benchmarks = (
            "AGI Eval ",
            "Alpaca(v2)",
            "Alpaca(v2, len adj)",
            "Arena Elo",
            "BBH",
            "GPT4All",
            "Hugging-6",
            "MAGI",
            "MMLU",
            "MT-bench",  # ["LMSys Arena", "Arena Hard", "MT-Bench"]
        )

        results = []
        for n_models in agreement_df.model_subset_size_requested.unique().tolist():
            for benchmark in agreement_df.scenario.unique().tolist():
                if benchmark in main_reference_benchmarks:
                    continue

                agreement_only_one_bench = agreement_df.query(
                    "scenario==@benchmark"
                    " and ref_scenario in @main_reference_benchmarks"
                    " and model_subset_size_requested==@n_models"
                    " and model_select_strategy!='random'"
                )

                agreement_aggregate = agreement_df.query(
                    # "scenario in @default_aggregate_scenarios"
                    "scenario!=@benchmark"
                    " and ref_scenario in @main_reference_benchmarks"
                    " and model_subset_size_requested==@n_models"
                    " and model_select_strategy!='random'"
                )

                mean_corr = agreement_only_one_bench["correlation"].mean()
                ref_corrs = (
                    agreement_aggregate.groupby(["scenario"])["correlation"]
                    .mean()
                    .reset_index()
                )
                ref_corr_thresh = (
                    ref_corrs["correlation"].mean() - ref_corrs["correlation"].std()
                )

                # sns.displot(kind="kde", data=ref_corrs, x="correlation")
                # plt.axvline(mean_corr, 0, 10, linewidth=4, color="r")
                # plt.axvline(ref_corr_thresh, 0, 10, linewidth=4, color="b")

                z_score = (mean_corr - ref_corrs["correlation"].mean()) / ref_corrs[
                    "correlation"
                ].std()

                results.append(
                    {
                        "benchmark": benchmark,
                        "correlation": mean_corr,
                        # "t_stat": t_stat,
                        # "p_value": p_value,
                        "z_score": z_score,
                        "n_models": n_models,
                    }
                )

        res = pd.DataFrame(results)
        res.pivot(columns=["benchmark", "n_models"], values=["z_score", "correlation"])

        # Pivot the DataFrame
        df_pivot = res.pivot(
            index="benchmark",
            columns="n_models",
            values=["correlation", "z_score"],
            # aggfunc="first",
        )

        df_pivot.to_csv("benchbench-leaderboard_for_streamlit.csv")

        df = pd.read_csv("benchbench-leaderboard_for_streamlit.csv")
        df = (
            df.rename(
                columns={
                    "correlation": "correlation (5)",
                    "correlation.1": "correlation (10)",
                    "correlation.2": "correlation (20)",
                    "z_score.2": "Z-Score",
                    "Unnamed: 0": "Benchmark",
                }
            )
            .drop(
                columns=[
                    "z_score",
                    "z_score.1",
                ]
            )
            .iloc[2:, :]
        ).sort_values("Z-Score", ascending=False)[
            [
                "Benchmark",
                "Z-Score",
                "correlation (5)",
                "correlation (10)",
                "correlation (20)",
            ]
        ]

        # Rename the columns for clarity
        df_pivot.columns = [f"{val}_{int(col)}_models" for val, col in df_pivot.columns]
        df_pivot = df_pivot.reset_index()

        def pass_z_test(row):
            return 1 if row["z_score"] >= -1 else 0

        res["z_test_pass"] = res.apply(pass_z_test, axis=1)

        print(
            res.query("n_models==10").sort_values("z_score", ascending=False)[
                ["benchmark", "z_score", "z_test_pass"]
            ]
        )
        saved = res.query("n_models==10").sort_values("z_score", ascending=False)[
            ["benchmark", "z_score", "z_test_pass"]
        ]

        saved["z_score"] = saved["z_score"].round(2)

        saved.to_csv("BAT_w_aggregate_10_random.csv")

        sns.clustermap(
            data=agreement_df.query(
                "corr_type=='kendall'"
                " and model_subset_size_requested==20"
                " and model_select_strategy=='random'"
                # " and scenario in @scenarios_to_show"
                # " and ref_scenario in @scenarios_to_show"
            )[["scenario", "ref_scenario", "correlation", "exp_n"]]
            .reset_index()
            .pivot_table(
                # index=["exp_n", "scenario"],
                index="scenario",
                columns="ref_scenario",
                values="correlation",
            )
            .fillna(1),
            annot=True,
        )
        plt.tight_layout()

        print()

    if exp_to_run == "n_models_matters":
        print()

        print(
            agreement_df.query(
                'scenario=="LMSys Arena" and ref_scenario=="Alpaca(v2, len adj)"'
            )["correlation"].tolist()
        )

        scenarios_not_to_show = [
            "EQ+MAGI",
            "EQ-Bench(v2)",
            "GPT4All",
            "GSM-8K",
            "HELM  Lite",
            "MAGI",
        ]

        sns.set(style="white", font_scale=1.5)
        fig, ax = plt.subplots()

        plot_df = (
            agreement_df.groupby(
                [
                    "scenario",
                    "model_subset_size_requested",
                    "ref_scenario",
                ]  # averaging over "n_exp",
            )["correlation"]
            .std()
            .reset_index()
            .rename(columns={"correlation": "correlation_std"})
            .query(
                "scenario not in @scenarios_not_to_show and ref_scenario not in @scenarios_not_to_show"
            )
        )

        sns.lineplot(
            ax=ax,
            data=plot_df,
            x="model_subset_size_requested",
            # hue="scenario",
            y="correlation_std",
            legend=False,
            markersize=10,
            linewidth=4,
        )

        # ax.invert_yaxis()

        sns.lineplot(
            ax=ax,
            data=plot_df,
            x="model_subset_size_requested",
            hue="scenario",
            y="correlation_std",
            alpha=0.3,
            legend=False,
            errorbar=None,
        )

        ax.invert_yaxis()
        ax.set_xlabel("Model Subset Size")
        ax.set_ylabel("Mean Agreement STD\n(across model subsets)")
        ax.set_xlim(5, 16)
        ax.set_ylim(0.25, 0)
        plt.tight_layout()

        plt.savefig("figures/final_for_paper/lineplot_n_models_matters.pdf")
        plt.show()

    if exp_to_run == "external":
        agreement_df.query(
            'scenario=="Arena Hard" and model_select_strategy=="random"'
        ).groupby(["ref_scenario"])["correlation"].mean().reset_index().sort_values(
            "correlation",
            ascending=False,
        )

        agreement_df.query(
            'model_select_strategy=="random" and scenario=="MMLU" and ref_scenario=="OC_MMLU"'
        )["correlation"].mean()
