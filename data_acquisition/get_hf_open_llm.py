import requests
from bs4 import BeautifulSoup
import json
import pandas as pd


def get_json_format_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    script_elements = soup.find_all("script")
    json_format_data = json.loads(str(script_elements[1])[31:-10])
    return json_format_data


def get_datas(data):
    for component_index in range(
        0, 50, 1
    ):  # component_index sometimes changes when they update the space, we can use this "for" loop to avoid changing component index manually
        try:
            result_list = []
            i = 0
            while True:
                try:
                    results = data["components"][component_index]["props"]["value"][
                        "data"
                    ][i]
                    columns = data["components"][component_index]["props"]["headers"]
                    try:
                        results_json = {"T": results[0], "Model": results[-1]}

                        if (
                            len(columns) < 13
                        ):  # If there are less than 15 columns (this number can definetly change), we know that we are trying wrong component index, so breaking loop to try next component index.
                            break

                        for col_index, col_name in enumerate(columns[2:-1], start=2):
                            results_json[col_name] = results[col_index]

                    except IndexError:  # Wrong component index, so breaking loop to try next component index. (NOTE: More than one component index can give you some results but we must find the right component index to get all results we want.)
                        break
                    result_list.append(results_json)
                    i += 1
                except IndexError:  # No rows to extract so return the list (We know it is the right component index because we didn't break out of loop on the other exception.)
                    return result_list
        except (KeyError, TypeError):
            continue

    return result_list


if __name__ == "__main__":
    # for V2
    data = get_json_format_data(
        url="https://open-llm-leaderboard-open-llm-leaderboard.hf.space/"
    )
    finished_models = get_datas(data)
    df = pd.DataFrame(finished_models)
    df = df.query("T=='🟢' or T=='💬'")
    cols_to_use = [
        "Model",
        "Average ⬆️",
        "IFEval",
        "BBH",
        "BBH Raw",
        "MATH Lvl 5",
        "GPQA",
        "MUSR",
        "MMLU-PRO",
    ]
    df = df[cols_to_use]
    df.rename(
        columns={
            "Average ⬆️": "hf_open_llm_v2",
            "Model": "model",
            "MATH Lvl 5": "MATH_Lvl_5",
        },
        inplace=True,
    )

    df.to_csv("src/bat/assets/benchmarks/hf_open_llm_v2_240829.csv", index=False)

    # for V1

    data = get_json_format_data(
        url="https://open-llm-leaderboard-old-open-llm-leaderboard.hf.space/"
    )
    finished_models = get_datas(data)
    df = pd.DataFrame(finished_models)
    df = df.query("T=='🟢' or T=='💬'")

    cols_to_use = [
        "Model",
        "Average ⬆️",
        "ARC",
        "HellaSwag",
        "MMLU",
        "TruthfulQA",
        "Winogrande",
        "GSM8K",
    ]

    df = df[cols_to_use]
    df.rename(
        columns={
            "Average ⬆️": "hf_open_llm_v1",
            "Model": "model",
        },
        inplace=True,
    )

    df.to_csv("src/bat/assets/benchmarks/hf_open_llm_v1_240829_frozen.csv", index=False)
