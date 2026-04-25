# Movie Recommendation System

A modular movie recommendation pipeline built on the MovieLens 100K dataset. The project started as a comparison between standard collaborative filtering approaches and grew into a full two-stage retrieval + ranking system, which ended up being the most interesting part of the work.

---

## What This Is

The core idea was to go beyond the typical "train a matrix factorization model and call it done" approach, and instead build something closer to how recommendation systems actually work in production, which is a retrieval stage that quickly narrows the full catalog down to a manageable candidate set, followed by a more expensive ranking stage that scores those candidates carefully.

The three models implemented are:

- **Matrix Factorization (MF)** — standard collaborative filtering baseline
- **MF + Content-Based Features (MF+CBF)** — adds user and item side features to MF
- **Two-Tower + MF Reranker** — two-stage pipeline with a retrieval model and a fine-tuned ranker

---

## Dataset

MovieLens 100K (~100,000 ratings from 943 users on 1,682 movies). Downloaded automatically when you run any script.

User features used: age, gender, occupation.
Movie features used: genres, release year.

---

## Data Processing

The main preprocessing steps are:

- Remove duplicate user-movie interactions
- Convert timestamps into monthly buckets
- Encode categorical user features
- Normalize numerical user and item features
- Build user and movie index mappings
- Construct PyTorch datasets for model training

---

## How the Data Is Split

A temporal split is used rather than a random split as this matters a lot for recommendation because random splits leak future interactions into training.

- **Train**: interactions up to February 1998
- **Validation**: March 1998
- **Test**: April 1998

Models are trained on past interactions and evaluated on future ones, which is a much more honest setup.

---

## Models

### Matrix Factorization

Standard MF with user embeddings, item embeddings, user/item bias terms, and a global mean. Trained with MSE loss and L2 regularization. Serves as the main collaborative filtering baseline.

### MF + Content-Based Features

Extends the base MF model by projecting user and item side features into the same latent space as the embeddings. The features are added directly to the embedding vectors before the dot product.

In practice this didn't improve over plain MF on ranking metrics. MF+CBF got better recall but worse NDCG, which suggests the features added noise to the ranking rather than helping.

### Two-Tower Retrieval + MF Reranker

This is the main model. It has two components:

**Retrieval (Two-Tower model)**

A dual-encoder with separate user and movie towers. Each tower has an embedding layer feeding into an MLP, with a residual connection adding the base embedding back to the MLP output. Outputs are L2-normalized so similarity is computed as cosine dot product.

Trained with InfoNCE (contrastive) loss with several improvements:
- Learnable temperature parameter (log-parameterized, clamped to [0.01, 1.0])
- Hard negatives mined from the pretrained MF model, for each positive (user, movie) pair, we query MF scores to find movies the MF model thinks are good for that user but aren't the ground truth, and add one of those to the loss denominator
- In-batch negatives combined with hard negatives in the same loss
- Gradient clipping (max norm 1.0) since temperature gradients can spike
- Early stopping on validation Recall@100

**Ranking (fine-tuned MF)**

After retrieval generates a candidate set of 200 movies per user, an MF model reranks them. The key insight here is that the ranker needs to be trained on the *same distribution it will see at inference* — i.e., the retrieved candidates — rather than on all (user, item) pairs uniformly.

So after retrieval training, we generate candidates for train and val users (without excluding seen items, since those are the positives we need), build a `CandidateRankingDataset` of (user, movie, relevant/not) triples, and fine-tune the ranker with LambdaRank loss. LambdaRank directly optimizes ranking order rather than treating it as binary classification.

Fine-tuning uses a much lower learning rate (5e-5 vs 1e-3 for initial MF training) and early stopping on val NDCG@10.

---

## Pipeline

```
Pretrain MF ranker (5 epochs, LR=1e-3)
        |
Train Two-Tower retrieval model
  - hard negatives from pretrained MF
  - temperature scaling
  - early stopping on val Recall@100
        |
Generate 200 candidates per user (train+val users, no seen-item exclusion)
        |
Fine-tune MF ranker on candidate pairs
  - LambdaRank loss
  - early stopping on val NDCG@10
  - LR=5e-5
        |
Generate 200 test candidates (with seen-item exclusion)
        |
Rerank test candidates with fine-tuned ranker → top-10
```

One non-obvious detail: when generating candidates for fine-tuning, we cannot exclude seen items. For train users, their relevant items (positives) ARE their seen items. Masking those out before retrieval means zero positives end up in the candidate pool and the dataset is empty. The seen-item exclusion only makes sense at test time, where you genuinely don't want to recommend things the user has already rated.

---

## Results

Evaluated on the April 1998 test set, Precision/Recall/HitRate/NDCG all at K=10.

| Model | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 |
|---|---|---|---|---|
| MF | 0.1389 | 0.0388 | 0.5309 | 0.1614 |
| MF + CBF | 0.1160 | 0.0580 | 0.4753 | 0.1364 |
| Two-Tower + Reranker | **0.2068** | **0.0833** | **0.6284** | **0.2352** |

The Two-Tower pipeline is the best model across every metric. Compared to the MF baseline, NDCG improved by ~46% and Hit Rate by ~18 percentage points.

A few things worth noting about these numbers:

- MF+CBF actually performed *worse* than plain MF on NDCG despite better recal, means adding features doesn't automatically help
- The original Two-Tower implementation (before the improvements) got NDCG 0.1023, worse than both MF baselines. Getting it above MF required fixing the retrieval stage, not just adding more model capacity
- Val Recall@100 for the retrieval model plateaus around 0.22, meaning ~78% of relevant items never enter the candidate pool. This is the main remaining bottleneck

---

## What Actually Made the Difference

In roughly descending order of impact:

1. **Candidate-aware ranker fine-tuning** — training the ranker on retrieved candidates with LambdaRank instead of on all pairs with MSE. This alone was the biggest jump.
2. **Hard negatives + temperature** — without these, the retrieval model's val Recall@100 stayed flat after epoch 1. Hard negatives from MF gave the model something meaningful to learn from.
3. **`seen_items_map=None` for fine-tune candidates** — a subtle but critical bug fix. Excluding seen items during candidate generation for train users produced zero positives and crashed the training.
4. **LayerNorm in tower MLPs** — more stable training than just ReLU stacks.
5. **Early stopping on val NDCG** for fine-tuning — the ranker was still improving at epoch 20 and converged around epoch 22-50 depending on LR, so running to a fixed epoch count would either underfit or waste time.
6. **Separate LRs for pre-training vs fine-tuning** — sharing a single `DEFAULT_LR` across MF training and fine-tuning caused the collapse once that LR was lowered to 5e-5. MF needs ~1e-3 to converge in 5 epochs; the fine-tuner benefits from 5e-5 for careful calibration.

---

## Project Structure

```
.
├── data/
│   ├── raw/                        # downloaded automatically
│   └── processed/
│
├── outputs/
│   ├── figures/
│   ├── metrics/                    # JSON summaries + pkl results per model
│   └── models/                     # saved model checkpoints
│
├── scripts/
│   ├── run_mf.py
│   ├── run_mf_cbf.py
│   ├── run_two_tower.py            # main pipeline
│   └── compare_models.py
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── download.py
│   │   ├── preprocessing.py
│   │   └── split.py
│   ├── models/
│   │   ├── matrix_factorization.py
│   │   └── two_tower.py
│   ├── training/
│   │   ├── train_mf.py
│   │   ├── train_retrieval.py
│   │   ├── candidate_aware_ranker.py
│   │   └── rank_candidates.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils/
│       ├── config.py
│       └── seed.py
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
pip install -r requirements.txt
```

```bash
# Run each model independently
python -m scripts.run_mf
python -m scripts.run_mf_cbf
python -m scripts.run_two_tower

# Compare results across models
python -m scripts.compare_models
```

The dataset downloads automatically on first run. Results are saved to `outputs/metrics/` as JSON summaries and pickle files. Model checkpoints go to `outputs/models/`.

---

## Configuration

All training hyperparameters live in `src/utils/config.py`. The ones most worth knowing about:

| Parameter | Value | Notes |
|---|---|---|
| `DEFAULT_EMBED_DIM` | 128 | Tower embedding dimension — 64 or 32 noticeably hurt retrieval |
| `DEFAULT_LR` | 1e-3 | Used for MF pretraining and retrieval |
| `DEFAULT_LR_FINETUNE` | 5e-5 | Fine-tuning only — needs to be separate from DEFAULT_LR |
| `DEFAULT_EPOCHS_RANKER` | 5 | MF pretraining epochs before retrieval |
| `DEFAULT_EPOCHS_RETRIEVAL` | 15 | Max retrieval epochs (early stopping usually triggers earlier) |
| `DEFAULT_EPOCHS_FINETUNE` | 75 | Max fine-tune epochs (early stopping triggers around 22-50) |
| `DEFAULT_RETRIEVAL_TOPK` | 100 | In-batch Recall@K during retrieval training |
| `candidate_topk` | 200 | Candidates passed to ranker per user |

---

## Possible Next Steps

Things I didn't get to but would try next:

- Replace the MF ranker with a deeper MLP ranker that concatenates user/item embeddings with side features — dot product MF is limited for distinguishing 200 candidates that are all plausible
- MoCo-style momentum queue for retrieval training — more negatives without scaling batch size, would likely push val Recall@100 above 0.22
- FAISS for approximate nearest neighbor search during candidate generation — currently using exact inner product which is fine at this scale but wouldn't be in production
- Multiple seeds for more reliable variance estimates — currently reporting single seed results

---

## Notes

- Raw data and model checkpoints are gitignored
- Results can vary slightly across runs due to hard negative sampling randomness in retrieval training
- Currently reporting single-seed results (`SEEDS = [0]`) — multi-seed averaging is supported by the pipeline but slow