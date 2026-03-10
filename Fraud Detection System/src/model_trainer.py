import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from src.config import MODEL_PATH, RANDOM_STATE


def train_models(X_train, y_train):
    models = {}

    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_params = {
        "n_estimators": [200],
        "max_depth": [None, 15],
    }

    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring="f1", n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    models["RandomForest"] = rf_grid.best_estimator_

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb

    best_model = models["RandomForest"]
    joblib.dump(best_model, MODEL_PATH)

    return best_model