
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/sample.csv")

def inspect_data():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    # Read only the columns needed for this check
    df = pd.read_csv(
        DATA_PATH,
        usecols=[" Source IP", " Destination IP", " Label"]
    )

    df.columns = df.columns.str.strip()

    print("\n--- Total Rows ---")
    print(len(df))

    print("\n--- Full Dataset Attack Labels ---")
    print(df["Label"].value_counts(dropna=False))

    print("\n--- Unique Labels ---")
    print(df["Label"].unique())

if __name__ == "__main__":
    inspect_data()
