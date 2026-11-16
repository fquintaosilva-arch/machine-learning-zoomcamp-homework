# Goodbooks-10k Book Rating Predictor & Recommender

## Problem Description
This project recommends books and predicts book ratings using the **Goodbooks-10k** dataset by combining weighted ratings (taking into consideration popularity), useful genre tags (removing tags as "favorites"), and author to predict how highly a user will rate a book (1-5 stars) and recommend similar titles.

## Exploratory Data Analysis (EDA) & Feature Engineering
* **Rating Skew:** Over 70% of ratings in Goodbooks-10k are 4 or 5 stars, requiring a weighted rating formula (`weighted_average_rating`) to prevent low-vote books from outranking classics.
* **Long-Tail Popularity:** A small number of bestsellers hold millions of ratings while thousands of books have far fewer.
* **Removing Tag Duplicates & Noise:** Merged `book_tags` and `tags`, removed noisy user shelf labels (`to-read`, `favorites`, `owned`, etc.) using regex, and aggregated duplicate tag counts per book.

## Models Evaluated & Validation Results
Models were evaluated using **Hit Rate@10** for unsupervised content similarity and **RMSE** for supervised rating prediction:

| Model | Tuned Hyperparameters | Metric | Score |
| :--- | :--- | :--- | :--- |
| **Mean Baseline** | Constant training mean | Validation RMSE | `0.9944` |
| **Content-Based Hybrid (TF-IDF + Rating)** | `similarity_weight = 0.2` | Hit Rate@10 | `72.00%` |
| **Decision Tree / Random Forest** | `max_depth = 7`, `n_estimators = 100`, `min_samples_leaf = 10` | Validation RMSE | `0.9606` |
| **Ridge Regression (Selected Winner)** | `alpha = 100.0` | Validation RMSE | **`0.9605`** |

* **Final Test Set Performance (`Ridge`, `alpha=100.0`):****`0.9660`**  Evaluated once on the unseen 20% test set (`df_test`).

## Project Structure
* `notebook.ipynb` — Full EDA, visualizations, tag cleaning, model comparisons, and hyperparameter tuning.
* `train.py` — Trains the winning `Ridge(alpha=100.0)` model on the full training set and exports `model.bin`.
* `predict.py` — FastAPI web service serving predictions on port `9696`.
* `test_predict.py` — Sample client script for testing the `/predict` endpoint.
* `Dockerfile` & `requirements.txt` — Container setup for reproducible deployment.

## How to Run Locally & with Docker

### 1. Train the Model
```bash
python train.py
```

### 2. Run with Docker
```bash
docker build -t goodbooks-predictor .
docker run -it --rm -p 9696:9696 goodbooks-predictor
```

### 3. Test the Prediction Endpoint
In a separate terminal:
```bash
python test_predict.py
```
Or send a `curl` request:
```bash
curl -X POST "http://localhost:9696/predict" \
  -H "Content-Type: application/json" \
  -d '{"weighted_average_rating": 4.46, "tag_name": "fantasy", "authors": "Patrick Rothfuss"}'
```
