import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def calculate_in_batch_recall(scores: torch.Tensor, k: int = 10) -> float:
    """Compute in-batch Recall@K."""
    batch_size = scores.size(0)
    actual_k = min(k, scores.size(1))

    _, topk_indices = torch.topk(scores, k=actual_k, dim=1)

    positive_indices = torch.arange(batch_size, device=scores.device).unsqueeze(1)
    hits = (topk_indices == positive_indices).any(dim=1).float()

    return hits.mean().item()


def train_one_epoch_retrieval(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    top_k: int = 100,
) -> tuple[float, float]:
    """Train the retrieval model for one epoch."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_recall = 0.0

    for batch in tqdm(loader, desc="Retrieval train", leave=False):
        user_idx, movie_idx, ratings, user_feats, movie_feats = batch

        user_idx = user_idx.to(device)
        movie_idx = movie_idx.to(device)
        user_feats = user_feats.to(device)
        movie_feats = movie_feats.to(device)

        user_vec, movie_vec = model(user_idx, movie_idx, user_feats, movie_feats)
        scores = torch.matmul(user_vec, movie_vec.T)

        labels = torch.arange(scores.size(0), device=device)
        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recall += calculate_in_batch_recall(scores, k=top_k)

    avg_loss = total_loss / len(loader)
    avg_recall = total_recall / len(loader)

    return avg_loss, avg_recall


def evaluate_retrieval(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    top_k: int = 100,
) -> tuple[float, float]:
    """Evaluate the retrieval model."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_recall = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Retrieval eval", leave=False):
            user_idx, movie_idx, ratings, user_feats, movie_feats = batch

            user_idx = user_idx.to(device)
            movie_idx = movie_idx.to(device)
            user_feats = user_feats.to(device)
            movie_feats = movie_feats.to(device)

            user_vec, movie_vec = model(user_idx, movie_idx, user_feats, movie_feats)
            scores = torch.matmul(user_vec, movie_vec.T)

            labels = torch.arange(scores.size(0), device=device)
            loss = criterion(scores, labels)

            total_loss += loss.item()
            total_recall += calculate_in_batch_recall(scores, k=top_k)

    avg_loss = total_loss / len(loader)
    avg_recall = total_recall / len(loader)

    return avg_loss, avg_recall


def get_top_k_candidates(
    retrieval_model: torch.nn.Module,
    dataset,
    device: torch.device,
    top_k: int = 50,
    batch_size: int = 512,
    user_idx_list: Optional[list[int]] = None,
    seen_items_map: Optional[dict] = None,
) -> dict:
    """Get top-k candidate items for each user."""
    retrieval_model.eval()

    all_movie_indices = torch.arange(len(dataset.movie2idx), dtype=torch.long, device=device)

    all_movie_features = torch.tensor(
        np.vstack([
            dataset.movies_df.loc[dataset.idx2movie[i]][dataset.movie_feature_cols]
            .values.astype(np.float32)
            for i in range(len(dataset.idx2movie))
        ]),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        all_movie_vectors = retrieval_model.movie_tower(all_movie_indices, all_movie_features)

    if user_idx_list is None:
        user_idx_list = list(range(len(dataset.idx2user)))

    user_candidates = {}
    user_feature_cols = dataset.user_feature_cols

    for start in range(0, len(user_idx_list), batch_size):
        batch_user_indices = user_idx_list[start:start + batch_size]
        batch_user_ids = [dataset.idx2user[user_idx] for user_idx in batch_user_indices]

        user_idx_tensor = torch.tensor(batch_user_indices, dtype=torch.long, device=device)

        user_features = torch.tensor(
            np.vstack([
                dataset.users_df.loc[user_id][user_feature_cols].values.astype(np.float32)
                for user_id in batch_user_ids
            ]),
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            user_vectors = retrieval_model.user_tower(user_idx_tensor, user_features)
            scores = torch.matmul(user_vectors, all_movie_vectors.T)

        for i, user_idx in enumerate(batch_user_indices):
            original_user_id = dataset.idx2user[user_idx]
            user_scores = scores[i].clone()

            if seen_items_map is not None and original_user_id in seen_items_map:
                seen_movie_ids = seen_items_map[original_user_id]
                seen_internal = [
                    dataset.movie2idx[movie_id]
                    for movie_id in seen_movie_ids
                    if movie_id in dataset.movie2idx
                ]
                if len(seen_internal) > 0:
                    user_scores[seen_internal] = -1e9

            actual_k = min(top_k, user_scores.shape[0])
            _, topk_indices = torch.topk(user_scores, k=actual_k)

            user_candidates[user_idx] = topk_indices.cpu().tolist()

    return user_candidates


def run_retrieval_training(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    dataset,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    retrieval_topk: int = 100,
    candidate_topk: int = 100,
    candidate_user_idx_list: Optional[list[int]] = None,
    seen_items_map: Optional[dict] = None,
) -> dict:
    """Train retrieval model and generate candidates."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "train_recall": [],
        "val_loss": [],
        "val_recall": [],
        "epoch_time": [],
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        start_time = time.time()

        train_loss, train_recall = train_one_epoch_retrieval(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            top_k=retrieval_topk,
        )

        val_loss, val_recall = evaluate_retrieval(
            model=model,
            loader=val_loader,
            device=device,
            top_k=retrieval_topk,
        )

        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_recall"].append(train_recall)
        history["val_loss"].append(val_loss)
        history["val_recall"].append(val_recall)
        history["epoch_time"].append(elapsed)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Recall@{retrieval_topk}: {train_recall:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Recall@{retrieval_topk}: {val_recall:.4f}"
        )

    for param in model.parameters():
        param.requires_grad = False

    candidates = get_top_k_candidates(
        retrieval_model=model,
        dataset=dataset,
        device=device,
        top_k=candidate_topk,
        user_idx_list=candidate_user_idx_list,
        seen_items_map=seen_items_map,
    )

    return {
        "model": model,
        "history": history,
        "candidates": candidates,
    }