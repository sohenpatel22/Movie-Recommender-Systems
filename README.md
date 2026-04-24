# Recommendation System Pipeline (MF vs MF+CBF vs Two-Tower)

## Overview

This project implements a **modular recommendation system pipeline** using the MovieLens 100K dataset.

The goal is to move beyond simple models and build a **structured end-to-end system** that includes:

- Data preprocessing and feature engineering  
- Temporal train/validation/test splitting  
- Multiple recommendation models  
- Retrieval + ranking pipeline  
- Evaluation using ranking metrics  

The project compares three approaches:

1. Matrix Factorization (MF)
2. Matrix Factorization with Content Features (MF + CBF)
3. Two-Tower Retrieval + MF Reranking

---

## Motivation

Most beginner projects focus only on training a model.

In real-world systems:
- recommendations are **multi-stage**
- models are evaluated on **ranking metrics**, not just loss
- **data leakage must be avoided**

This project was built to understand:
- how recommendation systems are structured
- how retrieval and ranking work together
- how to evaluate recommendations properly

---

## Dataset

- **MovieLens 100K**
- ~100,000 user-movie interactions
- Includes:
  - user metadata (age, gender, occupation)
  - movie metadata (genres, release date)

---

## Data Processing

Key steps:

- Remove duplicate user-movie interactions
- Convert timestamps → monthly buckets
- Encode categorical features (gender, occupation)
- Normalize numerical features (age, release date)

---

## Train / Validation / Test Split

A **temporal split** is used:

- Train: data up to Feb 1998  
- Validation: March 1998  
- Test: April 1998  

This prevents **data leakage** and simulates real-world recommendation scenarios.

---

## Models

### 1. Matrix Factorization (MF)

- Learns user and item embeddings
- Predicts ratings using dot product
- Includes bias terms and global mean

---

### 2. MF + Content-Based Features

- Extends MF with:
  - user features (age, occupation, etc.)
  - item features (genres, release date)
- Features are projected into embedding space

---

### 3. Two-Tower Retrieval Model

- Separate networks for:
  - users
  - items
- Learns embeddings for large-scale retrieval
- Uses in-batch negatives for training

---

## Two-Stage Pipeline

The final system follows a real-world structure:

### Stage 1: Retrieval
- Two-Tower model retrieves top-N candidate items

### Stage 2: Ranking
- MF model rescoring candidates
- Final top-K recommendations are selected

---

## Evaluation Metrics

Models are evaluated using:

- Precision@K  
- Recall@K  
- Hit Rate@K  
- NDCG@K  

These metrics measure **ranking quality**, not just prediction accuracy.

---

## Results

| Model        | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 |
|--------------|-------------|----------|------------|---------|
| MF           | ~0.29       | ~0.05    | ~0.84      | ~0.30   |
| MF + CBF     | ~0.20       | ~0.03    | ~0.75      | ~0.22   |
| Two-Tower    | **~0.34**   | **~0.09**| ~0.84      | **~0.36**|

**Observation:**
- Two-Tower pipeline improves recall and ranking quality
- MF remains a strong baseline
- Adding content features did not outperform pure MF in this setup

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

````

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
````

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
* Retrieval + ranking is more effective than standalone models
* Evaluation should focus on **ranking metrics**, not just MSE

---

## Future Improvements

* Add negative sampling strategies beyond in-batch negatives
* Improve feature engineering for MF + CBF
* Add approximate nearest neighbor (ANN) retrieval
* Deploy a simple recommendation demo

---

## Notes

* Models must be trained before evaluation scripts are run
* Data is downloaded automatically (not included in repo)

---