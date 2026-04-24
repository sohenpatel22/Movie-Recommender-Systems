import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import evaluate_mse, compute_metrics_at_k


class LossFunction:
    """
    Wrapper for prediction loss with optional L2 regularization.
    """

    def __init__(
        self,
        loss_method: nn.Module,
        reg_weights: Optional[np.ndarray] = None,
    ):
        self.loss_method = loss_method
        self.reg_weights = reg_weights

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        predictions = predictions.float()
        targets = targets.float()

        loss = self.loss_method(predictions, targets)

        if model is not None and self.reg_weights is not None:
            reg_loss = 0.0
            params = list(model.parameters())

            for i in range(min(len(params), len(self.reg_weights))):
                reg_loss += self.reg_weights[i] * torch.sum(params[i] ** 2)

            loss = loss + reg_loss / len(targets)

        return loss.float()


def train_one_epoch_mf(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: LossFunction,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train MF-based model for one epoch.
    Returns average training loss across batches.
    """
    model.train()
    model.to(device)

    total_loss = 0.0

    for users, items, ratings, user_features, item_features in loader:
        users = users.to(device)
        items = items.to(device)
        ratings = ratings.to(device)
        user_features = user_features.to(device)
        item_features = item_features.to(device)

        predictions = model(
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
        )

        loss = loss_fn(predictions, ratings, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


def run_mf_training(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    train_user_ids: list,
    val_user_ids: list,
    train_relevant_ratings: dict,
    val_relevant_ratings: dict,
    user_rate_map: dict,
    device: torch.device,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    loss_method: Optional[nn.Module] = None,
    reg_weights: Optional[np.ndarray] = None,
    k: int = 10,
) -> dict:
    """
    Full training loop for Matrix Factorization or MF+CBF models.

    Tracks:
    - training loss
    - validation MSE
    - ranking metrics on train and validation
    - epoch timing
    """
    if loss_method is None:
        loss_method = nn.MSELoss(reduction="sum")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = LossFunction(loss_method=loss_method, reg_weights=reg_weights)

    history = {
        "train_loss": [],
        "val_mse": [],
        "train_precision": [],
        "train_recall": [],
        "train_hit_rate": [],
        "train_ndcg": [],
        "val_precision": [],
        "val_recall": [],
        "val_hit_rate": [],
        "val_ndcg": [],
        "train_time": [],
        "eval_time": [],
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        start_train = time.perf_counter()
        train_loss = train_one_epoch_mf(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        train_elapsed = time.perf_counter() - start_train

        start_eval = time.perf_counter()

        val_mse = evaluate_mse(
            model=model,
            loader=val_loader,
            device=device,
        )

        train_p, train_r, train_hr, train_ndcg = compute_metrics_at_k(
            user_ids=train_user_ids,
            relevant_ratings=train_relevant_ratings,
            predict_fn=lambda uid, topk: model.recommend(
                user_id=uid,
                k=topk,
                user_rate_map=user_rate_map,
                device=device,
            ),
            k=k,
        )

        val_p, val_r, val_hr, val_ndcg = compute_metrics_at_k(
            user_ids=val_user_ids,
            relevant_ratings=val_relevant_ratings,
            predict_fn=lambda uid, topk: model.recommend(
                user_id=uid,
                k=topk,
                user_rate_map=user_rate_map,
                device=device,
            ),
            k=k,
        )

        eval_elapsed = time.perf_counter() - start_eval

        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)

        history["train_precision"].append(train_p)
        history["train_recall"].append(train_r)
        history["train_hit_rate"].append(train_hr)
        history["train_ndcg"].append(train_ndcg)

        history["val_precision"].append(val_p)
        history["val_recall"].append(val_r)
        history["val_hit_rate"].append(val_hr)
        history["val_ndcg"].append(val_ndcg)

        history["train_time"].append(train_elapsed)
        history["eval_time"].append(eval_elapsed)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"Val Precision@{k}: {val_p:.4f} | "
            f"Val Recall@{k}: {val_r:.4f} | "
            f"Val NDCG@{k}: {val_ndcg:.4f}"
        )

    return {
        "model": model,
        "history": history,
        "k": k,
    }


def evaluate_mf_ranking(
    model: torch.nn.Module,
    test_user_ids: list,
    test_relevant_ratings: dict,
    user_rate_map: dict,
    device: torch.device,
    k: int = 10,
) -> dict:
    """
    Final ranking evaluation on test users.
    """
    precision, recall, hit_rate, ndcg = compute_metrics_at_k(
        user_ids=test_user_ids,
        relevant_ratings=test_relevant_ratings,
        predict_fn=lambda uid, topk: model.recommend(
            user_id=uid,
            k=topk,
            user_rate_map=user_rate_map,
            device=device,
        ),
        k=k,
    )

    return {
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "ndcg": ndcg,
    }