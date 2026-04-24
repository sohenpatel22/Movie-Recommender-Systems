"""
Candidate-aware ranker training.

Instead of training the MF ranker on all (user, item) pairs uniformly,
we fine-tune it on only the (user, candidate) pairs produced by the
retrieval model. This aligns the ranker's training distribution with
the exact re-ranking task it will face at inference.

Usage in your run_two_tower script:

    from src.training.candidate_aware_ranker import fine_tune_ranker_on_candidates

    # After retrieval training produces `candidates`:
    ranker = fine_tune_ranker_on_candidates(
        ranker_model=mf_ranker,
        dataset=dataset,
        candidates=retrieval_results["candidates"],   # dict[user_idx -> [movie_idx]]
        relevant_ratings=train_relevant_ratings,
        device=device,
        epochs=5,
        learning_rate=1e-4,
        patience=5,
    )
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import compute_metrics_at_k
from src.training.rank_candidates import rerank_candidates_for_user
from src.evaluation.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CandidateRankingDataset(Dataset):
    """
    Builds (user_idx, movie_idx, label) triples from the candidate pool.

    For each user:
      - positive examples  : movies in candidates ∩ relevant_ratings  (label=1)
      - negative examples  : movies in candidates \\ relevant_ratings  (label=0)

    We balance positives and negatives at build time so the ranker doesn't
    just learn to predict 0 everywhere.
    """

    def __init__(
        self,
        candidates: dict[int, list[int]],
        relevant_ratings: dict,           # original user_id -> set of movie_ids
        dataset,
        neg_pos_ratio: int = 4,
    ):
        self.samples = []   # (user_internal_idx, movie_internal_idx, label)

        for user_internal_idx, candidate_movie_indices in candidates.items():
            original_user_id = dataset.idx2user[user_internal_idx]

            # Convert relevant movie ids to internal indices
            relevant_original = relevant_ratings.get(original_user_id, set())
            relevant_internal = {
                dataset.movie2idx[mid]
                for mid in relevant_original
                if mid in dataset.movie2idx
            }

            positives = [idx for idx in candidate_movie_indices if idx in relevant_internal]
            negatives = [idx for idx in candidate_movie_indices if idx not in relevant_internal]

            if not positives:
                continue

            # Sample negatives: up to neg_pos_ratio × number of positives
            n_neg = min(len(negatives), neg_pos_ratio * len(positives))
            sampled_negs = (
                np.random.choice(negatives, size=n_neg, replace=False).tolist()
                if n_neg > 0 else []
            )

            for movie_idx in positives:
                self.samples.append((user_internal_idx, movie_idx, 1.0))

            for movie_idx in sampled_negs:
                self.samples.append((user_internal_idx, movie_idx, 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_idx, movie_idx, label = self.samples[idx]
        return (
            torch.tensor(user_idx,  dtype=torch.long),
            torch.tensor(movie_idx, dtype=torch.long),
            torch.tensor(label,     dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def lambda_rank_loss(scores, labels, sigma=1.0):
    """
    Pairwise LambdaRank loss weighted by |delta NDCG|.
    scores: (B,) raw logits
    labels: (B,) binary relevance
    """
    scores = scores.unsqueeze(1) - scores.unsqueeze(0)   # (B, B) score differences
    labels = labels.unsqueeze(1) - labels.unsqueeze(0)   # (B, B) label differences

    # Only train on discordant pairs (label_i > label_j)
    mask = (labels > 0).float()

    loss = torch.log1p(torch.exp(-sigma * scores)) * mask
    return loss.sum() / (mask.sum() + 1e-8)


def fine_tune_ranker_on_candidates(
    ranker_model: torch.nn.Module,
    dataset,
    candidates: dict[int, list[int]],
    relevant_ratings: dict,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    batch_size: int = 512,
    neg_pos_ratio: int = 4,
    val_candidates: Optional[dict[int, list[int]]] = None,
    val_relevant_ratings: Optional[dict] = None,
    k: int = 10,
    patience: int = 5, #Updated
) -> torch.nn.Module:
    """
    Fine-tune `ranker_model` (a MatrixFactorisation instance) using only
    the candidate pairs produced by the retrieval stage.

    The loss is BCE on relevance labels, which is more appropriate than MSE
    for the binary ranking task the ranker is being used for.

    Args:
        ranker_model:         Pretrained MF model (already trained on all pairs).
        dataset:              The dataset object with idx2user, idx2movie, etc.
        candidates:           dict mapping user_internal_idx -> [movie_internal_idx]
        relevant_ratings:     dict mapping original_user_id -> set of relevant movie_ids
        device:               torch device
        epochs:               Fine-tuning epochs
        learning_rate:        Should be lower than initial LR (fine-tuning)
        batch_size:           Batch size
        neg_pos_ratio:        Negatives per positive in the dataset
        val_candidates:       Optional validation candidates for metric tracking
        val_relevant_ratings: Optional validation relevant ratings
        k:                    k for ranking metrics

    Returns:
        Fine-tuned ranker_model.
    """

    candidate_dataset = CandidateRankingDataset(
        candidates=candidates,
        relevant_ratings=relevant_ratings,
        dataset=dataset,
        neg_pos_ratio=neg_pos_ratio,
    )

    print(
        f"Candidate ranking dataset: {len(candidate_dataset)} samples "
        f"from {len(candidates)} users"
    )

    loader = DataLoader(
        candidate_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(ranker_model.parameters(), lr=learning_rate)
    loss_fn   = lambda_rank_loss

    ranker_model.to(device)

    # Early stopping state — only active when val set is provided
    best_ndcg         = -1.0
    epochs_no_improve = 0
    best_state        = None
    can_early_stop    = val_candidates is not None and val_relevant_ratings is not None

    for epoch in range(epochs):
        ranker_model.train()
        total_loss = 0.0

        for user_idx, movie_idx, labels in loader:
            user_idx  = user_idx.to(device)
            movie_idx = movie_idx.to(device)
            labels    = labels.to(device)

            scores = ranker_model(users=user_idx, items=movie_idx)
            loss   = loss_fn(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        log_line = f"  Fine-tune epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}"

        if val_candidates is not None and val_relevant_ratings is not None:
            val_p, val_r, val_hr, val_ndcg = _eval_ranking(
                ranker_model=ranker_model,
                dataset=dataset,
                candidates=val_candidates,
                relevant_ratings=val_relevant_ratings,
                device=device,
                k=k,
            )
            log_line += (
                f" | Val P@{k}: {val_p:.4f}"
                f" | Val R@{k}: {val_r:.4f}"
                f" | Val NDCG@{k}: {val_ndcg:.4f}"
            )
            print(log_line)

            # Check for improvement and save best checkpoint
            if val_ndcg > best_ndcg + 1e-4:
                best_ndcg         = val_ndcg
                epochs_no_improve = 0
                best_state        = {
                    k_: v.cpu().clone()
                    for k_, v in ranker_model.state_dict().items()
                }
                print(f" New best val NDCG@{k}: {best_ndcg:.4f}")
            else:
                epochs_no_improve += 1
                print(f"  No improvement ({epochs_no_improve}/{patience})")
                if epochs_no_improve >= patience:
                    print(f"  Early stopping fine-tune at epoch {epoch + 1}")
                    break
        else:
            # No val set provided — run all epochs, no early stopping
            print(log_line)

    # Restore best checkpoint if we were tracking one
    if best_state is not None:
        ranker_model.load_state_dict(
            {k_: v.to(device) for k_, v in best_state.items()}
        )
        print(f"Restored best fine-tuned model (val NDCG@{k}: {best_ndcg:.4f})")

    return ranker_model


# ---------------------------------------------------------------------------
# Internal eval helper
# ---------------------------------------------------------------------------

def _eval_ranking(
    ranker_model: torch.nn.Module,
    dataset,
    candidates: dict[int, list[int]],
    relevant_ratings: dict,
    device: torch.device,
    k: int = 10,
) -> tuple[float, float, float, float]:
    """Evaluate ranker on pre-generated candidates."""

    ranker_model.eval()

    precisions, recalls, hit_rates, ndcgs = [], [], [], []

    for user_internal_idx, candidate_item_indices in candidates.items():
        if not candidate_item_indices:
            continue

        original_user_id   = dataset.idx2user[user_internal_idx]
        relevant_original  = relevant_ratings.get(original_user_id, set())
        relevant_internal  = [
            dataset.movie2idx[mid]
            for mid in relevant_original
            if mid in dataset.movie2idx
        ]

        if not relevant_internal:
            continue

        reranked = rerank_candidates_for_user(
            ranker_model=ranker_model,
            user_internal_idx=user_internal_idx,
            candidate_item_indices=candidate_item_indices,
            device=device,
        )

        p, r, hr, ndcg = compute_metrics(
            pred_items=reranked[:k],
            relevant_items=relevant_internal,
        )
        precisions.append(p)
        recalls.append(r)
        hit_rates.append(hr)
        ndcgs.append(ndcg)

    if not precisions:
        return 0.0, 0.0, 0.0, 0.0

    return (
        sum(precisions) / len(precisions),
        sum(recalls)    / len(recalls),
        sum(hit_rates)  / len(hit_rates),
        sum(ndcgs)      / len(ndcgs),
    )
