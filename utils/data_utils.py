import pandas as pd 


def load_data_txt(path: str) -> list:
    """Load data from a file."""
    with open(path, "r") as f:
        data = f.readlines()[0]
        data = [int(x.strip()) for x in data.split(",")]
    return data


def load_data_csv(path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(path)