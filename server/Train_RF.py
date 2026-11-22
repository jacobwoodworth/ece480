"""
train_rf.py  -  trains and evaluates the random forest fault classifier

Gets its data from dataset.py (which reads the CSVs saved by motor_logger.py).

    python train_rf.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict

from Dataset import FEATURES, load_dataset

# Trained model is saved next to this script
MODEL_PATH = Path(__file__).resolve().parent / "motor_rf.joblib"


# Random forest settings
def make_model():
    return RandomForestClassifier(
        n_estimators=300,         # number of trees
        min_samples_leaf=2,       # stops trees memorizing single rows
        class_weight="balanced",  # treats labels fairly even if one has more data
        random_state=0,           # same results every time
        n_jobs=-1,                # use all CPU cores
    )


def main():
    # Load all logged runs
    X, y, groups = load_dataset()

    # Make sure there are enough runs to test on
    runs_per_class = groups.groupby(y).nunique()
    n_runs = groups.nunique()
    if n_runs < 2:
        print("Need at least 2 recorded runs to test the model. Log more data first.")
        return
    too_few = runs_per_class[runs_per_class < 2]
    if len(too_few):
        print("Warning: these classes have only 1 run, so they can't be tested on an unseen run:",
              ", ".join(too_few.index), "\n")

    # Split by run so the model is always tested on runs it never trained on
    k = int(min(5, max(2, runs_per_class.min())))
    if runs_per_class.min() >= 2:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
    else:
        cv = GroupKFold(n_splits=min(k, n_runs))

    # Train and test once per fold, collecting a prediction for every row
    pred = cross_val_predict(make_model(), X, y, groups=groups, cv=cv)

    # Accuracy for each label
    print(f"=== Cross-validated results ({cv.get_n_splits()} folds, split by run) ===")
    print(classification_report(y, pred, zero_division=0))

    # Table of which labels get confused with which
    labels = sorted(y.unique())
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels),
                      index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels])
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(cm.to_string(), "\n")

    # Train the final model on all the data
    model = make_model().fit(X, y)

    # Show which features the model relies on most
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Feature importance (how much the forest relies on each feature):")
    for name, val in importance.items():
        print(f"  {name:10s} {val:.3f}  {'#' * int(val * 50)}")

    # Save the model for later use
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()