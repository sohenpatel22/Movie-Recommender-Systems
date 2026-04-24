import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Hard Negative Mining
# ---------------------------------------------------------------------------

def mine_hard_negatives_from_mf(
    mf_model: torch.nn.Module,
    user_ids: torch.Tensor,
    pos_movie_indices: torch.Tensor,
    k_hard: int = 30,
    device: torch.device = "cpu",
) -> torch.Tensor:
    """
    For each (user, positive_movie) pair, find a hard negative:
    a movie scored highly by MF that is NOT the positive item.

    Returns a tensor of hard negative internal movie indices,
    shape (batch_size,).
    """
    hard_negs = []
    catalog_size = len(mf_model.idx2movie)

    # Set eval mode once outside the loop — more efficient than per-user inside
    mf_model.eval()

    with torch.no_grad():
        for user_tensor, pos_idx in zip(user_ids, pos_movie_indices):
            user_id = mf_model.idx2user[user_tensor.item()]

            try:
                scores = mf_model.predict_all_items_for_user(
                    user_id=user_id, device=device
                )
            except ValueError:
                # Fall back to random negative using catalog size — safe and correct
                hard_negs.append(np.random.randint(0, catalog_size))
                continue

            # Mask out the positive
            scores[pos_idx.item()] = -1e9

            # Take top-k hard candidates, sample one randomly
            # (pure argmax causes the same negatives every batch)
            top_hard_indices = np.argpartition(scores, -k_hard)[-k_hard:]
            chosen = int(np.random.choice(top_hard_indices))
            hard_negs.append(chosen)

    return torch.tensor(hard_negs, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Loss with Hard Negatives + Temperature
# ---------------------------------------------------------------------------

def contrastive_loss_with_hard_negatives(
    user_vecs: torch.Tensor,
    pos_movie_vecs: torch.Tensor,
    hard_neg_vecs: torch.Tensor,
    temperature: torch.Tensor,
    hard_neg_weight: float = 1.0,
) -> torch.Tensor:
    """
    InfoNCE loss combining in-batch negatives with hard negatives.

    For each user i:
      - positive: pos_movie_vecs[i]
      - in-batch negatives: all pos_movie_vecs[j], j != i
      - hard negatives: hard_neg_vecs[i]  (concatenated into the denominator)

    Args:
        user_vecs:       (B, D) L2-normalized user embeddings
        pos_movie_vecs:  (B, D) L2-normalized positive movie embeddings
        hard_neg_vecs:   (B, D) L2-normalized hard negative movie embeddings
        temperature:     scalar tensor (learnable)
        hard_neg_weight: multiplier on hard negative logits (>1 = penalize more)
    """
    B = user_vecs.size(0)

    # In-batch scores: (B, B)
    in_batch_scores = torch.matmul(user_vecs, pos_movie_vecs.T) / temperature

    # Hard negative scores: (B, 1)
    hard_neg_scores = (
        (user_vecs * hard_neg_vecs).sum(dim=-1, keepdim=True)
        / temperature
        * hard_neg_weight
    )

    # Concatenate: for each user, columns are [all in-batch movies | hard neg]
    # Shape: (B, B+1)
    all_scores = torch.cat([in_batch_scores, hard_neg_scores], dim=1)

    # Labels: the positive for user i is column i (diagonal)
    labels = torch.arange(B, device=user_vecs.device)

    return nn.CrossEntropyLoss()(all_scores, labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_in_batch_recall(scores: torch.Tensor, k: int = 10) -> float:
    """Compute in-batch Recall@K."""
    batch_size = scores.size(0)
    actual_k = min(k, scores.size(1))
    _, topk_indices = torch.topk(scores, k=actual_k, dim=1)
    positive_indices = torch.arange(batch_size, device=scores.device).unsqueeze(1)
    hits = (topk_indices == positive_indices).any(dim=1).float()
    return hits.mean().item()


# ---------------------------------------------------------------------------
# One Epoch
# ---------------------------------------------------------------------------

def train_one_epoch_retrieval(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    top_k: int = 100,
    mf_model: Optional[torch.nn.Module] = None,
    hard_neg_k: int = 30,
    hard_neg_weight: float = 1.5,
) -> tuple[float, float]:
    """Train the retrieval model for one epoch."""
    model.train()

    total_loss = 0.0
    total_recall = 0.0

    for batch in tqdm(loader, desc="Retrieval train", leave=False):
        user_idx, movie_idx, ratings, user_feats, movie_feats = batch

        user_idx    = user_idx.to(device)
        movie_idx   = movie_idx.to(device)
        user_feats  = user_feats.to(device)
        movie_feats = movie_feats.to(device)

        user_vec, movie_vec = model(user_idx, movie_idx, user_feats, movie_feats)

        if mf_model is not None:
            # --- Hard negative path ---
            hard_neg_indices = mine_hard_negatives_from_mf(
                mf_model=mf_model,
                user_ids=user_idx,
                pos_movie_indices=movie_idx,
                k_hard=hard_neg_k,
                device=device,
            )

            # Get features for hard negatives
            hard_neg_ids = [mf_model.idx2movie[i.item()] for i in hard_neg_indices]
            hard_neg_feats = torch.tensor(
                np.vstack([
                    loader.dataset.movies_df.loc[mid][loader.dataset.movie_feature_cols]
                    .values.astype(np.float32)
                    for mid in hard_neg_ids
                ]),
                dtype=torch.float32,
                device=device,
            )

            hard_neg_vecs = model.movie_tower(hard_neg_indices, hard_neg_feats)

            loss = contrastive_loss_with_hard_negatives(
                user_vecs=user_vec,
                pos_movie_vecs=movie_vec,
                hard_neg_vecs=hard_neg_vecs,
                temperature=model.temperature,
                hard_neg_weight=hard_neg_weight,
            )

        else:
            # --- Fallback: standard in-batch with temperature ---
            scores = torch.matmul(user_vec, movie_vec.T) / model.temperature
            labels = torch.arange(scores.size(0), device=device)
            loss = nn.CrossEntropyLoss()(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clip — temperature can cause spiky gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Recall metric uses raw (untempered) cosine scores for fairness
        with torch.no_grad():
            raw_scores = torch.matmul(user_vec, movie_vec.T)
        total_loss   += loss.item()
        total_recall += calculate_in_batch_recall(raw_scores, k=top_k)

    avg_loss   = total_loss   / len(loader)
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

    total_loss   = 0.0
    total_recall = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Retrieval eval", leave=False):
            user_idx, movie_idx, ratings, user_feats, movie_feats = batch

            user_idx    = user_idx.to(device)
            movie_idx   = movie_idx.to(device)
            user_feats  = user_feats.to(device)
            movie_feats = movie_feats.to(device)

            user_vec, movie_vec = model(user_idx, movie_idx, user_feats, movie_feats)

            scores = torch.matmul(user_vec, movie_vec.T) / model.temperature
            labels = torch.arange(scores.size(0), device=device)
            loss   = nn.CrossEntropyLoss()(scores, labels)

            # Raw cosine for recall metric
            raw_scores = torch.matmul(user_vec, movie_vec.T)

            total_loss   += loss.item()
            total_recall += calculate_in_batch_recall(raw_scores, k=top_k)

    avg_loss   = total_loss   / len(loader)
    avg_recall = total_recall / len(loader)
    return avg_loss, avg_recall


# ---------------------------------------------------------------------------
# Candidate Generation
# ---------------------------------------------------------------------------

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

    all_movie_indices = torch.arange(
        len(dataset.movie2idx), dtype=torch.long, device=device
    )

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
        all_movie_vectors = retrieval_model.movie_tower(
            all_movie_indices, all_movie_features
        )

    if user_idx_list is None:
        user_idx_list = list(range(len(dataset.idx2user)))

    user_candidates   = {}
    user_feature_cols = dataset.user_feature_cols

    for start in range(0, len(user_idx_list), batch_size):
        batch_user_indices = user_idx_list[start : start + batch_size]
        batch_user_ids     = [dataset.idx2user[i] for i in batch_user_indices]

        user_idx_tensor = torch.tensor(
            batch_user_indices, dtype=torch.long, device=device
        )
        user_features = torch.tensor(
            np.vstack([
                dataset.users_df.loc[uid][user_feature_cols]
                .values.astype(np.float32)
                for uid in batch_user_ids
            ]),
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            user_vectors = retrieval_model.user_tower(user_idx_tensor, user_features)
            scores       = torch.matmul(user_vectors, all_movie_vectors.T)

        for i, user_idx in enumerate(batch_user_indices):
            original_user_id = dataset.idx2user[user_idx]
            user_scores      = scores[i].clone()

            if seen_items_map is not None and original_user_id in seen_items_map:
                seen_internal = [
                    dataset.movie2idx[mid]
                    for mid in seen_items_map[original_user_id]
                    if mid in dataset.movie2idx
                ]
                if seen_internal:
                    user_scores[seen_internal] = -1e9

            actual_k = min(top_k, user_scores.shape[0])
            _, topk_indices = torch.topk(user_scores, k=actual_k)
            user_candidates[user_idx] = topk_indices.cpu().tolist()

    return user_candidates


# ---------------------------------------------------------------------------
# Full Training Run
# ---------------------------------------------------------------------------

def run_retrieval_training(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    dataset,
    device: torch.device,
    epochs: int = 15,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    retrieval_topk: int = 100,
    candidate_topk: int = 100,
    candidate_user_idx_list: Optional[list[int]] = None,
    seen_items_map: Optional[dict] = None,
    mf_model: Optional[torch.nn.Module] = None,
    hard_neg_k: int = 30,
    hard_neg_weight: float = 1.5,
    patience: int = 3,
) -> dict:
    """Train retrieval model with hard negatives, temperature, early stopping."""
    model.to(device)

    # Weight decay on everything except the temperature parameter
    temp_params  = [model.log_temperature]
    other_params = [p for n, p in model.named_parameters() if n != "log_temperature"]

    optimizer = torch.optim.Adam([
        {"params": other_params, "weight_decay": weight_decay},
        {"params": temp_params,  "weight_decay": 0.0},
    ], lr=learning_rate)

    history = {
        "train_loss": [], "train_recall": [],
        "val_loss":   [], "val_recall":   [],
        "epoch_time": [], "temperature":  [],
    }

    best_val_recall = -1.0
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        start_time = time.time()

        train_loss, train_recall = train_one_epoch_retrieval(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            top_k=retrieval_topk,
            mf_model=mf_model,
            hard_neg_k=hard_neg_k,
            hard_neg_weight=hard_neg_weight,
        )

        val_loss, val_recall = evaluate_retrieval(
            model=model,
            loader=val_loader,
            device=device,
            top_k=retrieval_topk,
        )

        elapsed = time.time() - start_time
        current_temp = model.temperature.item()

        history["train_loss"].append(train_loss)
        history["train_recall"].append(train_recall)
        history["val_loss"].append(val_loss)
        history["val_recall"].append(val_recall)
        history["epoch_time"].append(elapsed)
        history["temperature"].append(current_temp)

        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Train Recall@{retrieval_topk}: {train_recall:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Recall@{retrieval_topk}: {val_recall:.4f} | "
            f"Temp: {current_temp:.4f}"
        )

        # Early stopping on val recall
        if val_recall > best_val_recall + 1e-4:
            best_val_recall = val_recall
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  New best val recall: {best_val_recall:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"Restored best model (val recall: {best_val_recall:.4f})")

    # Freeze parameters before candidate generation
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
        "model":      model,
        "history":    history,
        "candidates": candidates,
    }