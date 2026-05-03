from src.train import KNNModel

model = KNNModel.load("models/model.pkl")

print("Testing predictions...\n")

for user, movie in [(1, 10), (2, 20), (3, 30)]:
    pred = model.predict(user, movie)
    print(f"User {user}, Movie {movie} → {pred:.2f}")

print("\nDone ✅")