import pandas as pd
import numpy as np
import json
from src.train import KNNModel
from src.features import RatingFeatures

def main():

    print("Loading features...")
    features = RatingFeatures.load("models/rating_features.pkl")

    print("Loading data...")
    df = pd.read_csv("data/processed/ratings_clean.csv")

    # Split data (80/20)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    print("Training size:", len(train_df))
    print("Validation size:", len(val_df))

    # Train model
    model = KNNModel(k=5)
    model.fit(features, train_df)

    # Evaluate
    print("Evaluating...")
    errors = []

    for _, row in val_df.iterrows():
        pred = model.predict(int(row['user_id']), int(row['movie_id']))
        errors.append((row['rating'] - pred) ** 2)

    rmse = np.sqrt(np.mean(errors))

    print("RMSE:", rmse)

    # Save model
    model.save("models/model.pkl")

    # Save metadata
    metadata = {
        "k": 5,
        "rmse": float(rmse)
    }

    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete")

if __name__ == "__main__":
    main()
    