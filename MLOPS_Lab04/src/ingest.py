import pandas as pd
import json
from pathlib import Path
from src.config import DATA_PATHS

def load_data():
    df = pd.read_csv(DATA_PATHS['raw'])
    print("Loaded:", df.shape)
    return df

def clean_data(df):
    df = df.drop_duplicates()
    df = df[(df['rating'] >= 0.5) & (df['rating'] <= 5.0)]
    print("Cleaned:", df.shape)
    return df

def save_data(df):
    Path("data/processed").mkdir(exist_ok=True)
    df.to_csv(DATA_PATHS['processed'], index=False)
    print("Saved cleaned data")

def save_report(df):
    report = {"rows": len(df)}
    Path("evaluations").mkdir(exist_ok=True)
    with open(DATA_PATHS['validation_report'], "w") as f:
        json.dump(report, f)

def main():
    df = load_data()
    df = clean_data(df)
    save_data(df)
    save_report(df)
    print("Pipeline complete")

if __name__ == "__main__":
    main()