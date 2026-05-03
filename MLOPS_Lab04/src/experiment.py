import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from src.train import KNNModel
from src.features import RatingFeatures

# Set experiment name
mlflow.set_experiment("KNN-Recommender")

def train_and_evaluate(k):

    with mlflow.start_run():

        print(f"\nRunning experiment with K={k}")

        # Load data
        df = pd.read_csv("data/processed/ratings_clean.csv")

        # Split
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)

        # Load features
        features = RatingFeatures.load("models/rating_features.pkl")

        # Train model
        model = KNNModel(k=k)
        model.fit(features, train_df)

        # Evaluate
        errors = []
        for _, row in val_df.iterrows():
            pred = model.predict(int(row['user_id']), int(row['movie_id']))
            errors.append((row['rating'] - pred) ** 2)

        rmse = np.sqrt(np.mean(errors))

        print("RMSE:", rmse)

        # 🔥 MLflow logging
        mlflow.log_param("k", k)
        mlflow.log_metric("rmse", rmse)

        # Save model
        mlflow.sklearn.log_model(model, "model")

        return rmse


def main():

    ks = [3, 5, 10, 15, 20]
    results = {}

    for k in ks:
        rmse = train_and_evaluate(k)
        results[k] = rmse

    print("\nFinal Results:")
    for k, v in results.items():
        print(f"K={k} → RMSE={v:.4f}")


if __name__ == "__main__":
    main()