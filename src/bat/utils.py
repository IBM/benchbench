import pandas as pd
import os
from bat.benchmark import Benchmark


def get_holistic_benchmark(file_name="assets/combined_holistic_20240708.csv"):
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.read_csv(f"src/bat/{file_name}")

    return Benchmark(df)
