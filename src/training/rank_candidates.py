import torch

from src.evaluation.metrics import compute_metrics


def rerank_candidates_for_user(
    ranker_model: torch.nn.Module,
    user_internal_idx: int,
    candidate_item_indices: list[int],
    device: torch.device,
) -> list[int]:
    """Rerank one user's candidate items using the ranker model."""
    if len(candidate_item_indices) == 0:
        return []

    ranker_model.eval()
    ranker_model.to(device)

    item_tensor = torch.tensor(candidate_item_indices, dtype=torch.long, device=device)
    user_tensor = torch.tensor(
        [user_internal_idx] * len(candidate_item_indices),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        scores = ranker_model(user_tensor, item_tensor).detach().cpu().numpy()

    sorted_positions = scores.argsort()[::-1]
    reranked_items = [candidate_item_indices[i] for i in sorted_positions]

    return reranked_items


def evaluate_reranked_candidates(
    ranker_model: torch.nn.Module,
    dataset,
    user_candidates: dict[int, list[int]],
    relevant_ratings: dict,
    device: torch.device,
    k: int = 10,
) -> dict:
    """Evaluate reranked candidates using top-k ranking metrics."""
    precisions = []
    recalls = []
    hit_rates = []
    ndcgs = []

    for user_internal_idx, candidate_item_indices in user_candidates.items():
        if len(candidate_item_indices) == 0:
            continue

        original_user_id = dataset.idx2user[user_internal_idx]

        true_relevant_original = relevant_ratings.get(original_user_id, set())
        true_relevant_internal = [
            dataset.movie2idx[movie_id]
            for movie_id in true_relevant_original
            if movie_id in dataset.movie2idx
        ]

        if len(true_relevant_internal) == 0:
            continue

        reranked_items = rerank_candidates_for_user(
            ranker_model=ranker_model,
            user_internal_idx=user_internal_idx,
            candidate_item_indices=candidate_item_indices,
            device=device,
        )

        topk_items = reranked_items[:k]

        precision, recall, hit_rate, ndcg = compute_metrics(
            pred_items=topk_items,
            relevant_items=true_relevant_internal,
        )

        precisions.append(precision)
        recalls.append(recall)
        hit_rates.append(hit_rate)
        ndcgs.append(ndcg)

    if len(precisions) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "hit_rate": 0.0,
            "ndcg": 0.0,
        }

    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "hit_rate": sum(hit_rates) / len(hit_rates),
        "ndcg": sum(ndcgs) / len(ndcgs),
    }


def run_ranking_stage(
    ranker_model: torch.nn.Module,
    dataset,
    user_candidates: dict[int, list[int]],
    relevant_ratings: dict,
    device: torch.device,
    k: int = 10,
) -> dict:
    """Run reranking and print final metrics."""
    results = evaluate_reranked_candidates(
        ranker_model=ranker_model,
        dataset=dataset,
        user_candidates=user_candidates,
        relevant_ratings=relevant_ratings,
        device=device,
        k=k,
    )

    print(f"Reranked Precision@{k}: {results['precision']:.4f}")
    print(f"Reranked Recall@{k}:    {results['recall']:.4f}")
    print(f"Reranked HitRate@{k}:   {results['hit_rate']:.4f}")
    print(f"Reranked NDCG@{k}:      {results['ndcg']:.4f}")

    return results