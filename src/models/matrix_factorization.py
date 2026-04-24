from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn


class MatrixFactorisation(nn.Module):
    """Basic MF model for recommendation."""

    def __init__(
        self,
        dataset,
        num_factors: int = 40,
        use_bias: bool = True,
        global_mean: Optional[float] = None,
    ):
        super().__init__()

        self.user2idx = dataset.user2idx
        self.idx2user = dataset.idx2user
        self.movie2idx = dataset.movie2idx
        self.idx2movie = dataset.idx2movie

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.user_embedding = nn.Embedding(self.num_users, num_factors)
        self.item_embedding = nn.Embedding(self.num_items, num_factors)

        self.use_bias = use_bias
        if self.use_bias:
            self.user_bias = nn.Embedding(self.num_users, 1)
            self.item_bias = nn.Embedding(self.num_items, 1)
            nn.init.constant_(self.user_bias.weight, 0.0)
            nn.init.constant_(self.item_bias.weight, 0.0)
        else:
            self.user_bias = None
            self.item_bias = None

        gm = 0.0 if global_mean is None else float(global_mean)
        self.global_mean = torch.tensor(gm, dtype=torch.float32)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users, items, *args, **kwargs):
        user_vec = self.user_embedding(users)
        item_vec = self.item_embedding(items)

        scores = (user_vec * item_vec).sum(dim=1) + self.global_mean

        if self.use_bias:
            scores = (
                scores
                + self.user_bias(users).squeeze(-1)
                + self.item_bias(items).squeeze(-1)
            )

        return scores.float()

    def predict_all_items_for_user(self, user_id: int, device="cpu") -> np.ndarray:
        """Return scores for all items for one user."""
        if user_id not in self.user2idx:
            raise ValueError(f"User ID {user_id} not found.")

        self.eval()

        user_idx = self.user2idx[user_id]
        user_tensor = torch.tensor(user_idx, dtype=torch.long, device=device)

        with torch.no_grad():
            user_vec = self.user_embedding(user_tensor)
            item_vecs = self.item_embedding.weight

            scores = item_vecs @ user_vec + self.global_mean

            if self.use_bias:
                scores = (
                    scores
                    + self.item_bias.weight.squeeze(-1)
                    + self.user_bias(user_tensor).squeeze(-1)
                )

        return scores.detach().cpu().numpy()

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        user_rate_map: Optional[dict] = None,
        exclude_items: Optional[Iterable[int]] = None,
        device="cpu",
    ) -> list:
        """Return top-k recommended movie ids."""
        scores = self.predict_all_items_for_user(user_id=user_id, device=device)

        excluded = set()

        if user_rate_map is not None and user_id in user_rate_map:
            excluded.update(user_rate_map[user_id])

        if exclude_items is not None:
            excluded.update(exclude_items)

        candidate_indices = []
        for idx in range(len(scores)):
            movie_id = self.idx2movie[idx]
            if movie_id not in excluded:
                candidate_indices.append(idx)

        if len(candidate_indices) == 0:
            return []

        actual_k = min(k, len(candidate_indices))
        candidate_indices = np.array(candidate_indices)

        candidate_scores = scores[candidate_indices]

        topk_partition_idx = np.argpartition(candidate_scores, -actual_k)[-actual_k:]
        topk_internal = candidate_indices[topk_partition_idx]
        topk_internal = topk_internal[np.argsort(scores[topk_internal])[::-1]]

        return [self.idx2movie[idx] for idx in topk_internal]


class MatrixFactorisationCBF(MatrixFactorisation):
    """MF model with user + item features."""

    def __init__(
        self,
        dataset,
        num_factors: int = 40,
        use_bias: bool = True,
        global_mean: Optional[float] = None,
    ):
        super().__init__(
            dataset=dataset,
            num_factors=num_factors,
            use_bias=use_bias,
            global_mean=global_mean,
        )

        self.u_feat_proj = nn.Linear(dataset.num_user_features, num_factors)
        self.i_feat_proj = nn.Linear(dataset.num_item_features, num_factors)

        nn.init.xavier_uniform_(self.u_feat_proj.weight)
        nn.init.xavier_uniform_(self.i_feat_proj.weight)

    def forward(self, users, items, user_features, item_features, *args, **kwargs):
        user_vec = self.user_embedding(users) + self.u_feat_proj(user_features)
        item_vec = self.item_embedding(items) + self.i_feat_proj(item_features)

        scores = (user_vec * item_vec).sum(dim=1) + self.global_mean

        if self.use_bias:
            scores = (
                scores
                + self.user_bias(users).squeeze(-1)
                + self.item_bias(items).squeeze(-1)
            )

        return scores.float()