import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, num_users, user_feat_dim, embed_dim=64, dropout=0.1):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + user_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, user_idx, user_feats):
        base_emb = self.user_embedding(user_idx)
        x = torch.cat([base_emb, user_feats], dim=-1)
        out = base_emb + self.mlp(x)
        return F.normalize(out, p=2, dim=-1)


class MovieTower(nn.Module):
    def __init__(self, num_movies, movie_feat_dim, embed_dim=64, dropout=0.1):
        super().__init__()
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        nn.init.xavier_uniform_(self.movie_embedding.weight)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + movie_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, movie_idx, movie_feats):
        base_emb = self.movie_embedding(movie_idx)
        x = torch.cat([base_emb, movie_feats], dim=-1)
        out = base_emb + self.mlp(x)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies, user_feat_dim, movie_feat_dim, embed_dim=64):
        super().__init__()
        self.user_tower = UserTower(num_users, user_feat_dim, embed_dim)
        self.movie_tower = MovieTower(num_movies, movie_feat_dim, embed_dim)

    def forward(self, user_idx, movie_idx, user_feats, movie_feats):
        user_vec = self.user_tower(user_idx, user_feats)
        movie_vec = self.movie_tower(movie_idx, movie_feats)
        return user_vec, movie_vec