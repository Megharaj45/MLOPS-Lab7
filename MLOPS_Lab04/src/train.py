import pandas as pd
import numpy as np
import joblib

class KNNModel:

    def __init__(self, k=5):
        self.k = k
        self.features = None
        self.data = None

    def fit(self, features, df):
        self.features = features
        self.data = df
        print("Model fitted with K =", self.k)

    def predict(self, user_id, movie_id):

        similar_users = self.features.get_similar_users(user_id, self.k)

        if not similar_users:
            return 3.0

        user_ids = [u for u, s in similar_users]

        ratings = self.data[
            (self.data['user_id'].isin(user_ids)) &
            (self.data['movie_id'] == movie_id)
        ]['rating']

        if len(ratings) == 0:
            return 3.0

        return float(np.mean(ratings))

    def save(self, path):
        joblib.dump(self, path)
        print("Model saved:", path)

    @staticmethod
    def load(path):
        return joblib.load(path)