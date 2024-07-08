from bat.benchmark import Benchmark
import pandas as pd
import os


def get_holistic_benchmark():
    if os.path.exists("assets/combined_holistic.csv"):
        return Benchmark(pd.read_csv())
    else:
        return Benchmark(pd.read_csv("src/bat/assets/combined_holistic.csv"))
