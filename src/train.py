# train.py
"""
Train, tune and evaluate two models for tumour-type classification:
a Random Forest and a Neural Network (MLP).

Pipeline:
    zero-variance filter -> 200-gene panel (ANOVA F-test) -> scaler
    -> PCA (4 components) -> SMOTE -> model

Run from the project root:   python3 -m src.train
"""

import json
import logging
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.logger import setup_logging
from src.settings import (CLASS_NAMES, CONFIDENCE_THRESHOLD, CV_FOLDS, DATA_FILE, FIGURES_DIR,
                          LABELS_FILE, MODEL_DIR, N_COMPONENTS, N_GENES, RANDOM_STATE,
                          REPORTS_DIR, TEST_SIZE)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 1. LOAD AND SPLIT
# --------------------------------------------------
def load_data():
    """Load expression data and labels, check they line up, return X and y."""
    X = pd.read_csv(DATA_FILE, index_col=0)
    labels = pd.read_csv(LABELS_FILE, index_col=0)
    if not X.index.equals(labels.index):
        raise ValueError("Sample IDs in data.csv and labels.csv do not match")
    y = labels["Class"]
    logger.info("Loaded %d samples x %d genes", X.shape[0], X.shape[1])
    return X, y


# --------------------------------------------------
# 2. PIPELINE (as in the Day 6 labs: steps in an ImbPipeline, SMOTE before the model)
# --------------------------------------------------
def build_pipeline(model):
    """The same preprocessing for every model; only the last step changes."""
    return ImbPipeline([
        ("variance", VarianceThreshold(threshold=0)),          # drop genes identical in every sample
        ("select", SelectKBest(f_classif, k=N_GENES)),         # the gene panel (learned on training data)
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),           # balance classes, training only
        ("model", model),
    ])


# The two architectures and the settings to tune for each ('model__' = the step called 'model')
MODELS = {
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 3],
        },
    ),
    "Neural Network": (
        MLPClassifier(max_iter=2000, random_state=RANDOM_STATE),
        {
            "model__hidden_layer_sizes": [(16,), (32, 16)],
            "model__alpha": [1e-4, 1e-2],                      # regularisation strength
            "model__learning_rate_init": [1e-3, 1e-2],
        },
    ),
}


# --------------------------------------------------
# 3. TUNE (GridSearchCV + k-fold cross-validation on the TRAINING set only)
# --------------------------------------------------
def tune(name, model, param_grid, X_train, y_train):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(build_pipeline(model), param_grid, cv=cv,
                          scoring="f1_macro", n_jobs=-1, return_train_score=True)
    start = time.time()
    search.fit(X_train, y_train)
    train_time = time.time() - start

    best = search.best_index_
    cv_train = search.cv_results_["mean_train_score"][best]
    logger.info("%s: best settings %s", name, search.best_params_)
    logger.info("%s: CV macro F1 %.3f (+/- %.3f) | training-fold macro F1 %.3f | search took %.1f s",
                name, search.best_score_, search.cv_results_["std_test_score"][best], cv_train, train_time)
    return search, train_time


# --------------------------------------------------
# 4. EVALUATE ON THE UNTOUCHED TEST SET
# --------------------------------------------------
def evaluate(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)

    macro = f1_score(y_test, y_pred, average="macro")
    micro = f1_score(y_test, y_pred, average="micro")
    logger.info("%s: test macro F1 %.3f | micro F1 %.3f", name, macro, micro)
    logger.info("%s classification report:\n%s", name,
                classification_report(y_test, y_pred, labels=CLASS_NAMES, digits=3))

    # Presentation question 4: predictions whose top probability is low
    low_conf = (proba.max(axis=1) < CONFIDENCE_THRESHOLD).sum()
    logger.info("%s: %d of %d test predictions below the %.0f%% confidence threshold",
                name, low_conf, len(y_test), CONFIDENCE_THRESHOLD * 100)

    return {"test_macro_f1": macro, "test_micro_f1": micro, "low_confidence": int(low_conf),
            "misclassified": int((y_pred != y_test.to_numpy()).sum())}, y_pred


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    setup_logging()
    logger.info("=== Training started (scikit-learn %s) ===", sklearn.__version__)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    logger.info("Train: %d samples | Test: %d samples", len(X_train), len(X_test))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary, predictions = {}, {}
    for name, (model, param_grid) in MODELS.items():
        search, train_time = tune(name, model, param_grid, X_train, y_train)
        best_pipeline = search.best_estimator_            # already refitted on all training data
        results, y_pred = evaluate(name, best_pipeline, X_test, y_test)

        file_name = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(best_pipeline, MODEL_DIR / file_name)
        logger.info("%s saved to %s", name, MODEL_DIR / file_name)

        summary[name] = {"cv_macro_f1": search.best_score_, **results,
                         "grid_search_time_s": train_time,
                         "file_size_mb": (MODEL_DIR / file_name).stat().st_size / 1e6,
                         "best_params": {k.replace("model__", ""): str(v)
                                         for k, v in search.best_params_.items()}}
        predictions[name] = y_pred

    # Side-by-side confusion matrices
    fig, axes = plt.subplots(1, len(predictions), figsize=(7 * len(predictions), 6))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=CLASS_NAMES,
                                                cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(f"{name} (test set)")
    fig.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrices.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrices saved to %s", cm_path)

    # Comparison table (as in the Day 6.2 final report)
    table = pd.DataFrame({name: {k: v for k, v in s.items() if k != "best_params"}
                          for name, s in summary.items()}).round(3)
    logger.info("Model comparison:\n%s", table.to_string())
    table.to_csv(REPORTS_DIR / "model_comparison.csv")

    # Metadata: what was trained, how, and with which versions (traceability)
    metadata = {"n_genes": N_GENES, "n_components": N_COMPONENTS, "cv_folds": CV_FOLDS,
                "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
                "sklearn_version": sklearn.__version__, "models": summary}
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("=== Training finished ===")


if __name__ == "__main__":
    main()