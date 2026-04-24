# Recommendation System Pipeline (MF vs MF+CBF vs Two-Tower)

## Overview

This project implements a **modular recommendation system pipeline** using the MovieLens 100K dataset.

The goal is to move beyond simple recommendation models and build a structured end-to-end system that includes:

- Data preprocessing and feature engineering
- Temporal train/validation/test splitting
- Multiple recommendation models
- Two-stage retrieval + ranking pipeline
- Evaluation using ranking metrics

The project compares three approaches:

1. Matrix Factorization (MF)
2. Matrix Factorization with Content Features (MF + CBF)
3. Two-Tower Retrieval + MF Reranking

---

## Motivation

Most beginner recommendation projects focus only on training a single model.

In real-world recommendation systems:

- recommendations are often **multi-stage**
- retrieval and ranking are handled separately
- models are evaluated using **ranking metrics**, not just loss
- temporal data leakage must be avoided

This project was built to understand how recommendation systems are structured, how retrieval and ranking work together, and how different modeling choices affect top-K recommendation quality.

---

## Dataset

This project uses the **MovieLens 100K** dataset.

The dataset contains approximately 100,000 user-movie interactions and includes:

- user metadata such as age, gender, and occupation
- movie metadata such as genres and release date
- explicit ratings from users

The dataset is downloaded automatically when the pipeline is run.

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

## Train / Validation / Test Split

A **temporal split** is used:

- Train: data up to February 1998
- Validation: March 1998
- Test: April 1998

This prevents data leakage and makes the setup closer to a real recommendation scenario, where the model is trained on past interactions and evaluated on future interactions.

---

## Models

### 1. Matrix Factorization (MF)

Matrix Factorization is used as the main collaborative filtering baseline.

It learns:

- user embeddings
- item embeddings
- user bias terms
- item bias terms
- global rating mean

The model predicts user-item preference using the dot product between user and item embeddings.

---

### 2. Matrix Factorization + Content-Based Features

This model extends the basic MF model by adding user and item side features.

The additional features include:

- user age, gender, and occupation
- movie genres
- movie release date

These features are projected into the same latent space as the user and item embeddings.

---

### 3. Two-Tower Retrieval + MF Reranking

The Two-Tower model is used as the retrieval stage.

It contains:

- a user tower
- a movie tower
- embedding layers
- MLP layers
- dot-product similarity between user and movie embeddings

The retrieval model is trained using:

- in-batch negatives
- hard negatives mined from the MF model
- temperature-scaled contrastive loss
- early stopping based on validation recall

After retrieval, an MF-based ranker is used to rerank the retrieved candidates.

The ranker is further fine-tuned on retrieved candidates so that its training distribution is closer to the final reranking task.

---

## Two-Stage Recommendation Pipeline

The final recommendation system follows a two-stage structure.

### Stage 1: Retrieval

The Two-Tower model retrieves a candidate set of movies for each user.

This stage focuses on narrowing the full item catalog into a smaller set of likely relevant candidates.

### Stage 2: Ranking

The MF ranker scores the retrieved candidate movies and sorts them.

The final top-K recommendations are selected after reranking.

This setup is useful because retrieval and ranking solve different parts of the recommendation problem.

---

## Evaluation Metrics

Models are evaluated using top-K ranking metrics:

- Precision@10
- Recall@10
- Hit Rate@10
- NDCG@10

These metrics are more suitable than MSE alone because recommendation quality depends on how well relevant items are ranked near the top.

---

## Results

Results are reported on the test set.

| Model      | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 |
|------------|-------------:|----------:|------------:|--------:|
| MF         | 0.1389       | 0.0388    | 0.5309      | 0.1614  |
| MF + CBF   | 0.1160       | 0.0580    | 0.4753      | 0.1364  |
| Two-Tower  | **0.2068**   | **0.0833**| **0.6284**  | **0.2352** |

---

## Observations

The Two-Tower + reranking pipeline achieved the best performance across all ranking metrics.

Key observations:

- The Two-Tower model improved Precision@10 from 0.1389 to 0.2068 compared with the MF baseline
- Recall@10 improved from 0.0388 to 0.0833
- Hit Rate@10 improved from 0.5309 to 0.6284
- NDCG@10 improved from 0.1614 to 0.2352
- MF remained a strong baseline despite its simpler structure
- MF + CBF improved recall compared with MF, but did not improve overall ranking quality

The main takeaway is that a two-stage retrieval + reranking setup can outperform a single-stage baseline when the retrieval and ranking stages are aligned properly.

---

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_results_summary.ipynb
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── scripts/
│   ├── run_mf.py
│   ├── run_mf_cbf.py
│   ├── run_two_tower.py
│   └── compare_models.py
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
│
├── requirements.txt
└── README.md
````

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Matrix Factorization

```bash
python -m scripts.run_mf
```

### 3. Train MF + CBF

```bash
python -m scripts.run_mf_cbf
```

### 4. Train Two-Tower retrieval + reranking pipeline

```bash
python -m scripts.run_two_tower
```

### 5. Compare all models

```bash
python -m scripts.compare_models
```

---

## Outputs

The project saves generated outputs under the `outputs/` directory.

Typical outputs include:

* model comparison CSV files
* summary JSON files
* comparison plots
* model checkpoints

Large artifacts such as model weights are excluded from version control.

---

## Key Learnings

This project helped me understand that:

* recommendation systems are usually pipeline-based, not just single models
* temporal splitting is important to avoid leakage
* ranking metrics are more meaningful than only reporting prediction loss
* simple MF baselines can be strong and should not be ignored
* retrieval quality strongly affects final recommendation quality
* hard negative mining can improve retrieval training
* candidate-aware ranker fine-tuning can improve final reranking performance

---

## Future Improvements

Possible future improvements include:

* Add approximate nearest neighbor search using FAISS
* Try deeper neural ranking models
* Improve hard negative sampling strategies
* Add item popularity and diversity-aware reranking
* Tune hyperparameters across multiple random seeds
* Build a simple Streamlit demo for user-level recommendations

---

## Notes

* The dataset is downloaded automatically
* Raw data and model checkpoints are not committed to GitHub
* Results may vary slightly depending on hardware and random seed
* The current reported results use one seed for faster experimentation

````
