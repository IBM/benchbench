import pandas as pd
import os
from bat import benchmark
from bat.benchmark import Benchmark


def get_holistic_benchmark(file_name="assets/combined_holistic_20240708.csv"):
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.read_csv(f"src/bat/{file_name}")

    return Benchmark(df)

if __name__ == "__main__":

    csv_path = 'src/bat/assets/combined_20240704.csv'
    from bat.benchmark import Benchmark
    benchmark = Benchmark(pd.read_csv(csv_path))
    for source, df in benchmark.df.groupby('source'):
        df.to_csv(f'src/bat/assets/benchmarks/{source}.csv', index=False)
    
