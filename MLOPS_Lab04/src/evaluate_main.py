import pandas as pd
import json
from src.train import KNNModel
from src.evaluate import evaluate_predictions

def main():
    print("Loading model...")
    model = KNNModel.load("models/model.pkl")

    print("Loading data...")
    df = pd.read_csv("data/processed/ratings_clean.csv")

    # Split data
    test_df = df.sample(frac=0.2, random_state=42)

    y_true = []
    y_pred = []

    print("Generating predictions...")

    for _, row in test_df.iterrows():
        actual = row["rating"]
        pred = model.predict(int(row["user_id"]), int(row["movie_id"]))

        y_true.append(actual)
        y_pred.append(pred)

    # Evaluate
    results = evaluate_predictions(y_true, y_pred)

    # Save report
    with open("evaluations/evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation complete ✅")

if __name__ == "__main__":
    main()