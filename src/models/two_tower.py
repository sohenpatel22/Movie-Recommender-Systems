import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, num_users, user_feat_dim, embed_dim=128, dropout=0.1):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        hidden = max(128, embed_dim + user_feat_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + user_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, embed_dim),
        )

    def forward(self, user_idx, user_feats):
        base_emb = self.user_embedding(user_idx)
        x = torch.cat([base_emb, user_feats], dim=-1)
        out = base_emb + self.mlp(x)
        return F.normalize(out, p=2, dim=-1)


class MovieTower(nn.Module):
    def __init__(self, num_movies, movie_feat_dim, embed_dim=128, dropout=0.1):
        super().__init__()
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        nn.init.xavier_uniform_(self.movie_embedding.weight)

        hidden = max(128, embed_dim + movie_feat_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + movie_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, embed_dim),
        )

    def forward(self, movie_idx, movie_feats):
        base_emb = self.movie_embedding(movie_idx)
        x = torch.cat([base_emb, movie_feats], dim=-1)
        out = base_emb + self.mlp(x)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        user_feat_dim,
        movie_feat_dim,
        embed_dim=128,
        init_temperature=0.07,
    ):
        super().__init__()
        self.user_tower = UserTower(num_users, user_feat_dim, embed_dim)
        self.movie_tower = MovieTower(num_movies, movie_feat_dim, embed_dim)

        # Learnable temperature — log scale keeps it positive and well-conditioned
        self.log_temperature = nn.Parameter(
            torch.tensor(init_temperature).log()
        )

    @property
    def temperature(self):
        # Clamp to avoid collapse (too cold) or uniform (too hot)
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def forward(self, user_idx, movie_idx, user_feats, movie_feats):
        user_vec = self.user_tower(user_idx, user_feats)
        movie_vec = self.movie_tower(movie_idx, movie_feats)
        return user_vec, movie_vec
