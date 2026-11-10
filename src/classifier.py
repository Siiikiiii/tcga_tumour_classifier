# classifier.py
"""
TumourClassifier: everything needed to use a trained model on new patients,
in one object (encapsulation). The Flask app, the tests and any script use
this class, so the logic lives in one place.

    clf = TumourClassifier("Random Forest")
    results, problems = clf.run(profiles_df)
"""

import itertools
import logging
import math

import joblib
import numpy as np
import pandas as pd

from src.settings import CLASS_NAMES, CONFIDENCE_THRESHOLD, EXPRESSION_RANGE, MODEL_FILES, TOP_N_GENES

logger = logging.getLogger(__name__)


class TumourClassifier:

    def __init__(self, model_name):
        """Load a trained pipeline and work out everything else from it."""
        if model_name not in MODEL_FILES:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_FILES)}")
        self.model_name = model_name
        self.pipeline = joblib.load(MODEL_FILES[model_name])

        # Every gene the pipeline was trained on, in the right order
        self.all_genes = list(self.pipeline.feature_names_in_)
        # The panel genes: what remains after the variance filter and SelectKBest
        self.panel_genes = list(self.pipeline[:2].get_feature_names_out())
        self.classes = list(self.pipeline.classes_)

        logger.info("Loaded %s: %d input genes, %d-gene panel, classes %s",
                    model_name, len(self.all_genes), len(self.panel_genes), self.classes)

    # --------------------------------------------------
    # 1. VALIDATE
    # --------------------------------------------------
    def validate(self, df):
        """
        Check uploaded profiles. Returns (clean_df, problems):
          clean_df: valid patients, columns in exactly the model's order
          problems: {sample_id: reason} for every rejected patient
        Rules:
          - the panel genes are REQUIRED (the model uses only these)
          - other genes may be missing: filled with 0, as the model discards them anyway
          - values must be numbers within EXPRESSION_RANGE
        """
        missing_panel = [g for g in self.panel_genes if g not in df.columns]
        if missing_panel:
            raise ValueError(f"{len(missing_panel)} required panel genes are missing from the file, "
                             f"e.g. {missing_panel[:5]}")

        unknown = [c for c in df.columns if c not in self.all_genes]
        if unknown:
            logger.warning("Ignoring %d unknown columns, e.g. %s", len(unknown), unknown[:5])

        panel = df[self.panel_genes].apply(pd.to_numeric, errors="coerce")   # text -> NaN
        low, high = EXPRESSION_RANGE
        problems = {}
        for sample, row in panel.iterrows():
            n_missing = int(row.isna().sum())
            n_out = int(((row < low) | (row > high)).sum())
            if n_missing:
                problems[sample] = f"{n_missing} panel gene values missing or not numbers"
            elif n_out:
                problems[sample] = f"{n_out} panel gene values outside {low}-{high}"

        valid = df.loc[[s for s in df.index if s not in problems]]
        clean = valid.reindex(columns=self.all_genes, fill_value=0.0)   # add missing non-panel genes as 0
        clean[self.panel_genes] = panel.loc[valid.index]
        logger.info("Validated %d profiles: %d valid, %d rejected", len(df), len(clean), len(problems))
        return clean.astype(float), problems

    # --------------------------------------------------
    # 2. PREDICT
    # --------------------------------------------------
    def predict(self, X):
        """Predicted tumour type, the probability of each type, and a low-confidence flag."""
        proba = pd.DataFrame(self.pipeline.predict_proba(X), index=X.index, columns=self.classes)
        out = pd.DataFrame(index=X.index)
        out["predicted"] = proba.idxmax(axis=1)
        out["confidence"] = proba.max(axis=1)
        out["low_confidence"] = out["confidence"] < CONFIDENCE_THRESHOLD
        return out.join(proba[CLASS_NAMES])

    # --------------------------------------------------
    # 3. EMBED
    # --------------------------------------------------
    def embed(self, X):
        """The patient's gene expression embedding: their PCA component scores."""
        coords = self.pipeline[:4].transform(X)           # variance -> panel -> scaler -> PCA
        return pd.DataFrame(coords, index=X.index,
                            columns=[f"PC{i + 1}" for i in range(coords.shape[1])])

    # --------------------------------------------------
    # 4. EXPLAIN
    # --------------------------------------------------
    def explain(self, X, predicted, top_n=TOP_N_GENES):
        """
        Top driving genes for each patient's prediction, in two steps:

        Step 1, Shapley values (the method behind SHAP) for the PCA components.
          The model decides from only a few components, so this is calculated
          EXACTLY: the model is run on every combination of "this component at the
          patient's value" vs "at the average patient's value (0)", and each
          component's fair share of the change in probability is worked out.

        Step 2, from components back to genes.
          Each component score is a sum of (scaled gene value x loading), so each
          component's contribution is shared among the genes in proportion to how
          much each one makes up that patient's score.

        A gene's contribution is how much it raised (+) or lowered (-) the
        probability of the predicted tumour type, compared with an average patient.
        """
        model = self.pipeline.named_steps["model"]
        pca = self.pipeline.named_steps["pca"]
        k = pca.n_components_

        scaled = self.pipeline[:3].transform(X) - pca.mean_    # patient's panel genes, as PCA sees them
        scores = scaled @ pca.components_.T                    # = the patient's PC scores

        # Every combination of components: 1 = patient's value, 0 = average patient (0)
        masks = np.array(list(itertools.product([0, 1], repeat=k)))
        sizes = masks.sum(axis=1)
        weights = {s: math.factorial(s) * math.factorial(k - s - 1) / math.factorial(k) for s in range(k)}

        explanations = {}
        for row, sample in enumerate(X.index):
            cls = self.classes.index(predicted[sample])
            values = model.predict_proba(masks * scores[row])[:, cls]      # one prediction per combination
            value_of = {tuple(m): v for m, v in zip(masks, values)}

            shapley = np.zeros(k)                                          # Step 1
            for i in range(k):
                for m, size in zip(masks, sizes):
                    if m[i] == 0:
                        with_i = m.copy()
                        with_i[i] = 1
                        shapley[i] += weights[size] * (value_of[tuple(with_i)] - value_of[tuple(m)])

            # Step 2: each gene's share of each component score
            parts = scaled[row][:, None] * pca.components_.T               # genes x components
            safe = np.where(np.abs(scores[row]) < 1e-9, 1e-9, scores[row])
            gene_contrib = pd.Series((parts / safe) @ shapley, index=self.panel_genes)

            top = gene_contrib.sort_values(ascending=False).head(top_n)
            explanations[sample] = [(gene, round(float(value), 3)) for gene, value in top.items()]
        return explanations

    # --------------------------------------------------
    # ALL STEPS TOGETHER (what the app calls)
    # --------------------------------------------------
    def run(self, df):
        """Validate, then predict, embed and explain the valid patients."""
        X, problems = self.validate(df)
        if X.empty:
            return pd.DataFrame(), problems
        results = self.predict(X).join(self.embed(X))
        top_genes = self.explain(X, results["predicted"])
        results["top_genes"] = [top_genes[s] for s in results.index]
        logger.info("%s predicted %d profiles (%d low confidence)",
                    self.model_name, len(results), int(results["low_confidence"].sum()))
        return results, problems