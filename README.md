# Recommendation System Pipeline (MF vs MF+CBF vs Two-Tower)

## Overview

This project implements a **modular recommendation system pipeline** using the MovieLens 100K dataset.

The goal is to move beyond simple models and build a **structured end-to-end system** that includes:

* Data preprocessing and feature engineering
* Temporal train/validation/test splitting
* Multiple recommendation models
* Retrieval + ranking pipeline
* Evaluation using ranking metrics

The project compares three approaches:

1. Matrix Factorization (MF)
2. Matrix Factorization with Content Features (MF + CBF)
3. Two-Tower Retrieval + MF Reranking

---

## Motivation

Most beginner projects focus only on training a model.

In real-world systems:

* recommendations are **multi-stage**
* models are evaluated on **ranking metrics**, not just loss
* **data leakage must be avoided**

This project was built to understand:

* how recommendation systems are structured
* how retrieval and ranking work together
* how to evaluate recommendations properly

---

## Dataset

* **MovieLens 100K**
* ~100,000 user-movie interactions
* Includes:

  * user metadata (age, gender, occupation)
  * movie metadata (genres, release date)

---

## Data Processing

Key steps:

* Remove duplicate user-movie interactions
* Convert timestamps → monthly buckets
* Encode categorical features (gender, occupation)
* Normalize numerical features (age, release date)

---

## Train / Validation / Test Split

A **temporal split** is used:

* Train: data up to Feb 1998
* Validation: March 1998
* Test: April 1998

This prevents **data leakage** and ensures the model predicts future interactions using past data, which better reflects real-world recommendation systems.

---

## Models

### 1. Matrix Factorization (MF)

* Learns user and item embeddings
* Predicts ratings using dot product
* Includes bias terms and global mean

---

### 2. MF + Content-Based Features

* Extends MF with:

  * user features (age, occupation, etc.)
  * item features (genres, release date)
* Features are projected into embedding space

---

### 3. Two-Tower Retrieval Model

* Separate networks for:

  * users
  * items
* Learns embeddings for large-scale retrieval
* Uses in-batch negatives for training
* Uses dot-product similarity between user and item embeddings

---

## Two-Stage Pipeline

The final system follows a real-world structure:

### Stage 1: Retrieval

* Two-Tower model retrieves top-N candidate items

### Stage 2: Ranking

* MF model rescoring candidates
* Final top-K recommendations are selected

**Note:** The ranking stage is limited by the quality of candidates generated during retrieval.

---

## Evaluation Metrics

Models are evaluated using:

* Precision@K
* Recall@K
* Hit Rate@K
* NDCG@K

These metrics measure **ranking quality**, not just prediction accuracy.

---

## Results

Results are reported on the test set using ranking metrics.

| Model     | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 |
| --------- | -----------: | --------: | ----------: | ------: |
| MF        |       0.1389 |    0.0388 |      0.5309 |  0.1614 |
| MF + CBF  |       0.1160 |    0.0580 |      0.4753 |  0.1364 |
| Two-Tower |       0.1041 |    0.0535 |      0.4189 |  0.1296 |

**Observation:**

* Matrix Factorization (MF) provided the strongest overall ranking performance in this setup
* MF + CBF improved recall compared with MF, but reduced precision and overall ranking quality
* The Two-Tower retrieval + reranking pipeline did not outperform the MF baseline in these experiments

This highlights an important practical insight: **more complex architectures do not always outperform simpler baselines without careful tuning**, especially on smaller datasets like MovieLens 100K.

---

## Project Structure

```
src/
  data/
  models/
  training/
  evaluation/
  utils/

scripts/
  run_mf.py
  run_mf_cbf.py
  run_two_tower.py
  compare_models.py

data/
  raw/
  processed/

outputs/
  metrics/
  models/
  figures/
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run experiments

```bash
python scripts/run_mf.py
python scripts/run_mf_cbf.py
python scripts/run_two_tower.py
```

### 3. Compare results

```bash
python scripts/compare_models.py
```

---

## Key Learnings

* Recommendation systems are **pipeline-based**, not single models
* Temporal splitting is critical to avoid leakage
* Retrieval + ranking is a standard real-world system design
* Strong baselines like MF can outperform more complex models if not carefully tuned
* Evaluation should focus on **ranking metrics**, not just MSE

---

## Future Improvements

* Improve retrieval training with better negative sampling
* Use larger candidate pools or approximate nearest neighbor (ANN) search
* Tune Two-Tower architecture and loss functions
* Explore hybrid blending of MF and retrieval scores
* Deploy a simple recommendation demo

---

## Notes

* Models must be trained before evaluation scripts are run
* Data is downloaded automatically (not included in repo)
* Large artifacts such as model weights are excluded from version control

---
