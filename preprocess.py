import pandas as pd

def prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df["headline"].tolist()
    y = df["label"].tolist()
    return X, y