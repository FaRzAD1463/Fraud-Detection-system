import json
import numpy as np
from sklearn.metrics import f1_score
from src.config import THRESHOLD_PATH


def optimize_threshold(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:
        preds = (probs > t).astype(int)
        score = f1_score(y_test, preds)

        if score > best_f1:
            best_f1 = score
            best_threshold = t

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": best_threshold}, f)

    print(f"Best Threshold: {best_threshold} | F1: {best_f1}")