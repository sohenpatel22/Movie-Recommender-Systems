import math
import numpy as np
import torch
import torch.nn.functional as F


def evaluate_mse(model, loader, device):
    total_sq_error = 0.0
    total_count = 0

    model.eval()
    model.to(device)

    with torch.no_grad():
        for users, items, ratings, user_features, item_features in loader:
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)
            user_features = user_features.to(device)
            item_features = item_features.to(device)

            preds = model(
                users=users,
                items=items,
                user_features=user_features,
                item_features=item_features,
            ).float()

            total_sq_error += F.mse_loss(preds, ratings, reduction="sum").item()
            total_count += ratings.size(0)

    return total_sq_error / total_count


def compute_metrics(pred_items, relevant_items):
    k = len(pred_items)
    pred_set = set(pred_items)
    rel_set = set(relevant_items)

    hits = len(pred_set & rel_set)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0
    hit_rate = int(hits > 0)

    dcg = 0.0
    for i, item in enumerate(pred_items):
        if item in relevant_items:
            dcg += 1.0 / math.log2(i + 2)

    ideal_len = min(len(relevant_items), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
    ndcg = 0.0 if idcg == 0.0 else dcg / idcg

    return precision, recall, hit_rate, ndcg


def compute_metrics_at_k(user_ids, relevant_ratings, predict_fn, k=10):
    total_p, total_r, total_hr, total_ndcg = 0.0, 0.0, 0.0, 0.0

    for uid in user_ids:
        recs = predict_fn(uid, k)
        true_rel = relevant_ratings[uid]

        p, r, hr, ndcg = compute_metrics(recs, true_rel)
        total_p += p
        total_r += r
        total_hr += hr
        total_ndcg += ndcg

    num_users = len(user_ids)
    return (
        total_p / num_users,
        total_r / num_users,
        total_hr / num_users,
        total_ndcg / num_users,
    )


def aggregate_ranking_metric(runs, metric_name):
    values = [run[metric_name] for run in runs]
    return np.mean(values), np.std(values)