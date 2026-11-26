"""Logistic-regression and LightGBM fits with grouped inner validation."""

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def logistic(c: float, seed: int):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=seed))


def choose_c(folds: list[tuple], values: list[float], seed: int) -> float:
    """Select one regularization value from task-disjoint inner predictions."""
    scores = []
    for c in values:
        aucs = [
            roc_auc_score(valid_y, logistic(c, seed).fit(train_x, train_y).predict_proba(valid_x)[:, 1])
            for train_x, train_y, valid_x, valid_y in folds
        ]
        scores.append(float(np.mean(aucs)))
    return values[int(np.argmax(scores))]


def lightgbm(params: tuple, seed: int, threads: int) -> LGBMClassifier:
    leaves, rate, trees = params
    return LGBMClassifier(
        n_estimators=trees, learning_rate=rate, num_leaves=leaves,
        objective="binary", metric="auc", random_state=seed, n_jobs=threads, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )


def choose_lightgbm(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray,
                    params: dict, seed: int) -> tuple:
    scores = []
    for cfg in params["lightgbm_candidates"]:
        model = lightgbm(tuple(cfg), seed, params["cpu_threads"])
        model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric="auc",
                  callbacks=[early_stopping(params["lightgbm_patience"], first_metric_only=True, verbose=False),
                             log_evaluation(0)])
        scores.append((roc_auc_score(valid_y, model.booster_.predict(valid_x)), tuple(cfg[:2]) + (model.best_iteration_,)))
    return max(scores, key=lambda item: item[0])[1]
