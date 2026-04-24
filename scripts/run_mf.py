import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import RatingsDataset, get_unique_users_from_dataset
from src.data.download import download_movielens_100k
from src.data.preprocessing import (
    clean_ratings,
    load_movielens_data,
    preprocess_tables,
)
from src.data.split import temporal_split
from src.models.matrix_factorization import MatrixFactorisation
from src.training.train_mf import evaluate_mf_ranking, run_mf_training
from src.utils.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS_MF,
    DEFAULT_K,
    DEFAULT_LR,
    DEFAULT_NUM_FACTORS,
    DEVICE,
    METRICS_DIR,
    MODELS_DIR,
    SEEDS,
    create_directories,
)
from src.utils.seed import set_seed


def get_dataloaders(train_set, val_set, test_set, batch_size):
    """Create train, val, and test loaders."""
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def build_seen_maps(train_df, val_df=None):
    """Build seen-item maps for filtering recommendations."""
    train_seen = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    if val_df is None:
        return train_seen, train_seen

    val_seen = val_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    train_plus_val_seen = {}
    all_users = set(train_seen.keys()) | set(val_seen.keys())

    for user_id in all_users:
        train_items = train_seen.get(user_id, set())
        val_items = val_seen.get(user_id, set())
        train_plus_val_seen[user_id] = set(train_items) | set(val_items)

    return train_seen, train_plus_val_seen


def main():
    create_directories()

    print("Downloading/loading dataset")
    dataset_dir = download_movielens_100k()

    print("Reading raw tables")
    users_df, ratings_df, items_df, genres_df = load_movielens_data(dataset_dir)

    print("Cleaning ratings")
    ratings_df = clean_ratings(ratings_df, items_df)

    print("Preprocessing user and item features")
    users_processed, items_processed = preprocess_tables(users_df, items_df)

    print("Creating temporal split")
    train_df, val_df, test_df = temporal_split(ratings_df)

    print("Building datasets")
    train_set = RatingsDataset(train_df, users_processed, items_processed)
    val_set = RatingsDataset(val_df, users_processed, items_processed)
    test_set = RatingsDataset(test_df, users_processed, items_processed)

    print("Building data loaders")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    train_user_ids = get_unique_users_from_dataset(train_set)
    val_user_ids = get_unique_users_from_dataset(val_set)
    test_user_ids = get_unique_users_from_dataset(test_set)

    train_seen_map, train_val_seen_map = build_seen_maps(train_df, val_df)

    all_run_results = []

    for seed in SEEDS:
        print("=" * 60)
        print(f"Running MF with seed {seed}")
        set_seed(seed)

        global_mean = float(train_df["rating"].mean())

        model = MatrixFactorisation(
            dataset=train_set,
            num_factors=DEFAULT_NUM_FACTORS,
            use_bias=True,
            global_mean=global_mean,
        ).to(DEVICE)

        train_results = run_mf_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_user_ids=train_user_ids,
            val_user_ids=val_user_ids,
            train_relevant_ratings=train_set.relevant_ratings,
            val_relevant_ratings=val_set.relevant_ratings,
            user_rate_map=train_seen_map,
            device=DEVICE,
            epochs=DEFAULT_EPOCHS_MF,
            learning_rate=DEFAULT_LR,
            loss_method=nn.MSELoss(reduction="sum"),
            reg_weights=np.array([1e-1, 1e-1]),
            k=DEFAULT_K,
        )

        test_metrics = evaluate_mf_ranking(
            model=train_results["model"],
            test_user_ids=test_user_ids,
            test_relevant_ratings=test_set.relevant_ratings,
            user_rate_map=train_val_seen_map,
            device=DEVICE,
            k=DEFAULT_K,
        )

        print(
            f"Test Precision@{DEFAULT_K}: {test_metrics['precision']:.4f} | "
            f"Test Recall@{DEFAULT_K}: {test_metrics['recall']:.4f} | "
            f"Test HitRate@{DEFAULT_K}: {test_metrics['hit_rate']:.4f} | "
            f"Test NDCG@{DEFAULT_K}: {test_metrics['ndcg']:.4f}"
        )

        all_run_results.append(
            {
                "seed": seed,
                "history": train_results["history"],
                "test_metrics": test_metrics,
            }
        )

        model_path = MODELS_DIR / f"mf_seed_{seed}.pt"
        torch.save(train_results["model"].state_dict(), model_path)

    metrics_path = METRICS_DIR / "mf_results.pkl"
    with open(metrics_path, "wb") as f:
        pickle.dump(all_run_results, f)

    summary = {}
    metric_names = ["precision", "recall", "hit_rate", "ndcg"]

    for metric in metric_names:
        values = [run["test_metrics"][metric] for run in all_run_results]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    summary_path = METRICS_DIR / "mf_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("=" * 60)
    print("Final MF results across seeds:")
    for metric, stats in summary.items():
        print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print(f"Saved run results to: {metrics_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()