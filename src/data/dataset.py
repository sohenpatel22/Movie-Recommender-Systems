from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RatingsDataset(Dataset):
    """
    Dataset for recommendation models.
    Returns:
        user_idx, movie_idx, rating, user_features, movie_features
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        users_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        ratings_threshold: float = 4.0,
    ):
        self.ratings_df = ratings_df.reset_index(drop=True)

        self.user2idx = {u: i for i, u in enumerate(users_df["user_id"].unique())}
        self.movie2idx = {m: i for i, m in enumerate(movies_df["movie_id"].unique())}

        self.idx2user = {v: k for k, v in self.user2idx.items()}
        self.idx2movie = {v: k for k, v in self.movie2idx.items()}

        self.users_df = users_df.set_index("user_id")
        self.movies_df = movies_df.set_index("movie_id")

        self.user_feature_cols = [c for c in users_df.columns if c != "user_id"]
        self.movie_feature_cols = [
            c for c in movies_df.columns if c not in ["movie_id", "title"]
        ]

        self.num_user_features = len(self.user_feature_cols)
        self.num_item_features = len(self.movie_feature_cols)

        self.user_idxs = torch.tensor(
            self.ratings_df["user_id"].map(self.user2idx).values,
            dtype=torch.long,
        )
        self.movie_idxs = torch.tensor(
            self.ratings_df["movie_id"].map(self.movie2idx).values,
            dtype=torch.long,
        )
        self.ratings = torch.tensor(
            self.ratings_df["rating"].values,
            dtype=torch.float32,
        )

        self.relevant_ratings = defaultdict(set)
        for _, row in self.ratings_df.iterrows():
            if row["rating"] >= ratings_threshold:
                self.relevant_ratings[row["user_id"]].add(row["movie_id"])

        self.unique_users = users_df["user_id"].unique()
        self.unique_movies = movies_df["movie_id"].unique()
        self.num_users = len(self.unique_users)
        self.num_items = len(self.unique_movies)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_idx = self.user_idxs[idx].item()
        movie_idx = self.movie_idxs[idx].item()
        rating = self.ratings[idx].item()

        original_user_id = self.idx2user[user_idx]
        original_movie_id = self.idx2movie[movie_idx]

        user_features = (
            self.users_df.loc[original_user_id][self.user_feature_cols]
            .values.astype(np.float32)
        )
        movie_features = (
            self.movies_df.loc[original_movie_id][self.movie_feature_cols]
            .values.astype(np.float32)
        )

        return (
            user_idx,
            movie_idx,
            rating,
            torch.tensor(user_features, dtype=torch.float32),
            torch.tensor(movie_features, dtype=torch.float32),
        )


def get_unique_users_from_dataset(dataset):
    """Return original user IDs present in a dataset."""
    user_idx_list = dataset.user_idxs
    return [dataset.idx2user[int(idx)] for idx in user_idx_list.unique()]