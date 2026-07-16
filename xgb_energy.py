import re

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)

CSV_PATH  = "ch1_energy.csv"
TARGET    = "Fuse at Mason Square CH-01 elec energy"
TIMESTAMP = "Timestamp"

THRESHOLD = 25.0
HORIZON   = 1
N_SPLITS  = 5
N_ITER    = 60
SEED      = 42

XGB_FIXED = dict(objective="reg:squarederror", tree_method="hist",
                 random_state=SEED)


def load(path):
    df = pd.read_csv(path)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], utc=True)
    return df.sort_values(TIMESTAMP).reset_index(drop=True)


def build_features(df):
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    X = df.drop(columns=[TIMESTAMP, TARGET] + constant)
    X.columns = [re.sub(r"[^0-9A-Za-z_]", "_", c) for c in X.columns]
    X["energy_now"] = df[TARGET].values
    return X


def build_target(df, X):
    y = df[TARGET].shift(-HORIZON)
    keep = X.notna().all(axis=1) & y.notna()
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


def tune(X, y):
    space = {
        "n_estimators":     randint(200, 900),
        "learning_rate":    uniform(0.01, 0.02),
        "max_depth":        randint(3, 9),
        "min_child_weight": randint(1, 10),
        "subsample":        uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "gamma":            uniform(0.0, 5.0),
        "reg_lambda":       uniform(0.0, 5.0),
        "reg_alpha":        uniform(0.0, 2.0),
    }
    search = RandomizedSearchCV(
        XGBRegressor(n_jobs=1, **XGB_FIXED),
        space,
        n_iter=N_ITER,
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        n_jobs=-1,
        random_state=SEED,
        verbose=1,
    )
    search.fit(X, y)
    return search.best_params_, -search.best_score_


def walk_forward(X, y, params):
    y_true, y_pred = [], []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=N_SPLITS).split(X):
        model = XGBRegressor(n_jobs=-1, **XGB_FIXED, **params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_true.extend(y.iloc[test_idx])
        y_pred.extend(model.predict(X.iloc[test_idx]))
    return np.array(y_true), np.array(y_pred)


def regression_report(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print("\nHow well do we predict energy?")
    print(f"  MAE  : {mae:.3f}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  R2   : {r2:.3f}")
    return mae, rmse, r2


def incident_report(y_true, y_pred, buffer=0.0):
    true_incident = (y_true > THRESHOLD).astype(int)
    pred_incident = (y_pred > THRESHOLD - buffer).astype(int)

    print(f"\nHow well do we predict spikes?  (buffer = {buffer:.2f})")
    print(classification_report(true_incident, pred_incident,
                                digits=3, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(true_incident, pred_incident))


def threshold_sweep(y_true, y_pred):
    true_incident = (y_true > THRESHOLD).astype(int)
    n_pos = int(true_incident.sum())

    print(f"\n{'buffer':>7} {'precision':>10} {'recall':>8} {'F1':>6} "
          f"{'caught':>8} {'missed':>8} {'false_alarms':>13}")
    for buf in np.arange(0.0, 8.5, 0.5):
        pred = (y_pred > THRESHOLD - buf).astype(int)
        tp = int(((pred == 1) & (true_incident == 1)).sum())
        fp = int(((pred == 1) & (true_incident == 0)).sum())
        fn = n_pos - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{buf:>7.2f} {prec:>10.3f} {rec:>8.3f} {f1:>6.3f} "
              f"{tp:>8} {fn:>8} {fp:>13}")


def feature_importance(X, y, params, top_n=20):
    model = XGBRegressor(n_jobs=-1, **XGB_FIXED, **params)
    model.fit(X, y)

    ranked = (pd.Series(model.feature_importances_, index=X.columns)
              .sort_values(ascending=False)
              .head(top_n))

    print(f"\nTop {top_n} most important features")
    print("-" * 40)
    width = max(len(name) for name in ranked.index)
    total = model.feature_importances_.sum() or 1
    for name, gain in ranked.items():
        print(f"{name:<{width}}  {gain / total:.3f}")


def main():
    df = load(CSV_PATH)
    X_full = build_features(df)
    X, y = build_target(df, X_full)

    incidents = int((y > THRESHOLD).sum())
    print(f"{len(X):,} rows, {X.shape[1]} features, "
          f"{incidents} spikes ({incidents / len(y):.1%})")

    best_params, cv_mae = tune(X, y)
    print(f"\nTuned CV MAE: {cv_mae:.3f}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    y_true, y_pred = walk_forward(X, y, best_params)
    regression_report(y_true, y_pred)
    incident_report(y_true, y_pred, buffer=0.0)
    threshold_sweep(y_true, y_pred)
    feature_importance(X, y, best_params)


if __name__ == "__main__":
    main()
