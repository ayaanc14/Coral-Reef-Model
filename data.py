from pathlib import Path
import pandas as pd


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    return df
