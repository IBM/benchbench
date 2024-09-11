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
                        results_json = {"Model": results[0]}

                        if (
                            len(columns) < 13
                        ):  # If there are less than 15 columns (this number can definetly change), we know that we are trying wrong component index, so breaking loop to try next component index.
                            break

                        for col_index, col_name in enumerate(columns[1:-1], start=1):
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
    # for biggen

    data = get_json_format_data(
        url="https://prometheus-eval-BiGGen-Bench-Leaderboard.hf.space/"
    )
    finished_models = get_datas(data)
    df = pd.DataFrame(finished_models)

    # df["Model"]

    df["Model"] = df["Model"].apply(lambda x: x.split('">')[-1].split("</a>")[0])

    df.rename(
        columns={
            "Average": "biggen",
            "Model": "model",
        },
        inplace=True,
    )

    import pandas as pd
    import re

    # Function to clean column names
    def clean_column(col):
        col = re.sub(r"[^\w\s]", "", col)  # Remove emojis
        col = (
            col.strip().lower().replace(" ", "_")
        )  # Lowercase and replace spaces with _
        if col != "model" and col != "biggen":
            col = "biggen_" + col
        return col

    # Apply the cleaning function to the columns
    cleaned_columns = [clean_column(col) for col in df.columns.tolist()]
    df.columns = cleaned_columns
    df.drop(columns=["biggen_model_type"], inplace=True)

    df.to_csv("src/bat/assets/benchmarks_to_add/biggen_240829.csv", index=False)
