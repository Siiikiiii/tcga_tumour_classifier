# app.py
"""
Flask web app for the TCGA tumour classifier.

Two endpoints, as in the variant classifier app:
  /predict        one patient (an example patient, or a one-row CSV upload)
  /predict_batch  many patients from a CSV upload

All the model logic lives in TumourClassifier (src/classifier.py);
the routes only read the input, call it, and pass the results to the page.

Run from the project root:   python3 app.py
"""

import logging

import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score, f1_score

from src.classifier import TumourClassifier
from src.logger import setup_logging
from src.settings import CLASS_NAMES, CONFIDENCE_THRESHOLD, EXAMPLES_FILE, MODEL_FILES

setup_logging()
logger = logging.getLogger("src.app")      # under "src", so it uses the project's log settings

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# 1. Load every model ONCE at start-up (as MODELS did in the variant app)
CLASSIFIERS = {name: TumourClassifier(name) for name in MODEL_FILES}
DEFAULT_MODEL = "Random Forest"

# Example (unseen test) patients for the single-patient demo, if train.py saved them
EXAMPLES = pd.read_csv(EXAMPLES_FILE, index_col=0) if EXAMPLES_FILE.exists() else pd.DataFrame()


# 2. Context processor: makes these available in index.html
@app.context_processor
def inject_global_vars():
    return dict(model_names=list(CLASSIFIERS), example_ids=list(EXAMPLES.index),
                class_names=CLASS_NAMES, threshold=int(CONFIDENCE_THRESHOLD * 100))


# 3. Helpers
def read_upload(file):
    """Read an uploaded CSV: first column = sample IDs (as in data.csv)."""
    df = pd.read_csv(file, encoding="utf-8-sig", index_col=0)    # utf-8-sig: handles Excel's hidden marker
    df.columns = df.columns.str.strip()
    df.index = df.index.astype(str)
    return df


def split_true_labels(df):
    """If the file has a 'Class' column (true tumour types), take it out to compare with later."""
    if "Class" in df.columns:
        return df.drop(columns="Class"), df["Class"]
    return df, None


def get_classifier():
    name = request.form.get("model_choice", DEFAULT_MODEL)
    return CLASSIFIERS.get(name, CLASSIFIERS[DEFAULT_MODEL])


@app.route("/")
def home():
    return render_template("index.html")


# 1. Single patient
@app.route("/predict", methods=["POST"])
def predict():
    try:
        clf = get_classifier()
        example_id = request.form.get("example_id", "")
        file = request.files.get("file")

        if file and file.filename:                               # an uploaded one-patient CSV ...
            df = read_upload(file)
            if len(df) != 1:
                return render_template("index.html", single_error=f"Expected 1 patient, found {len(df)}. "
                                                                  "Use the batch panel for several patients.")
        elif example_id:                                         # ... or an example patient
            df = EXAMPLES.loc[[example_id]]
        else:
            return render_template("index.html", single_error="Choose an example patient or upload a file")

        df, true_labels = split_true_labels(df)
        results, problems = clf.run(df)
        if problems:
            sample, reason = next(iter(problems.items()))
            return render_template("index.html", single_error=f"{sample} rejected: {reason}")

        row = results.iloc[0]
        single = {
            "sample": results.index[0],
            "model": clf.model_name,
            "predicted": row["predicted"],
            "confidence": round(row["confidence"] * 100, 1),
            "low_confidence": bool(row["low_confidence"]),
            "true_label": None if true_labels is None else true_labels.iloc[0],
            "probabilities": {c: round(row[c] * 100, 1) for c in CLASS_NAMES},
            "embedding": {pc: round(row[pc], 2) for pc in results.columns if pc.startswith("PC")},
            "top_genes": row["top_genes"],
        }
        logger.info("Single prediction (%s): %s -> %s (%.1f%%)", clf.model_name,
                    single["sample"], single["predicted"], single["confidence"])
        return render_template("index.html", single=single)

    except ValueError as e:                                       # expected input problems
        logger.warning("Single prediction rejected: %s", e)
        return render_template("index.html", single_error=f"Error: {e}")
    except Exception as e:                                        # unexpected: log the full traceback
        logger.exception("Single prediction failed")
        return render_template("index.html", single_error=f"Error: {e}")


# 2. Batch: many patients from a CSV
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    file = request.files.get("file")
    if not file or not file.filename:
        return render_template("index.html", batch_error="No file selected")

    try:
        clf = get_classifier()
        df, true_labels = split_true_labels(read_upload(file))
        results, problems = clf.run(df)

        # One row per patient, in the file's order; rejected patients keep their reason
        rows = []
        for sample in df.index:
            if sample in problems:
                rows.append({"sample": sample, "rejected": problems[sample]})
                continue
            r = results.loc[sample]
            rows.append({
                "sample": sample,
                "predicted": r["predicted"],
                "confidence": round(r["confidence"] * 100, 1),
                "low_confidence": bool(r["low_confidence"]),
                "true_label": None if true_labels is None else true_labels[sample],
                "embedding": ", ".join(f"{r[pc]:.2f}" for pc in results.columns if pc.startswith("PC")),
                "top_genes": ", ".join(g for g, _ in r["top_genes"][:3]),
            })

        summary = (f"Model: {clf.model_name}. {len(results)} of {len(df)} patients predicted; "
                   f"{len(problems)} rejected; "
                   f"{int(results['low_confidence'].sum()) if len(results) else 0} below "
                   f"{int(CONFIDENCE_THRESHOLD * 100)}% confidence.")

        # Evaluation report, if the file included true labels
        if true_labels is not None and len(results):
            truth = true_labels[results.index]
            summary += (f" Against the true labels: accuracy {accuracy_score(truth, results['predicted']):.3f}, "
                        f"macro F1 {f1_score(truth, results['predicted'], average='macro'):.3f}.")

        logger.info("Batch prediction (%s): %s", clf.model_name, summary)
        return render_template("index.html", batch_results=rows, batch_summary=summary,
                               has_truth=true_labels is not None)

    except ValueError as e:                                       # expected input problems
        logger.warning("Batch prediction rejected: %s", e)
        return render_template("index.html", batch_error=f"Error processing CSV: {e}")
    except Exception as e:                                        # unexpected: log the full traceback
        logger.exception("Batch prediction failed")
        return render_template("index.html", batch_error=f"Error processing CSV: {e}")


# Keep this block at the end of app.py
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)