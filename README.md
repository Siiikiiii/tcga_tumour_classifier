# TCGA Tumour Classifier

Predicts the primary tumour type of a cancer patient from an RNA-Seq gene expression profile,
and shows which genes drove each prediction.

Built for the precision oncology capstone project (PPB GTAC 2026): a multi-class classifier for five
tumour types from The Cancer Genome Atlas (TCGA), deployed as a Flask web app for clinical scientists.

| Code | Tumour type |
| :--- | :--- |
| BRCA | Breast invasive carcinoma |
| COAD | Colon adenocarcinoma |
| KIRC | Kidney renal clear cell carcinoma |
| LUAD | Lung adenocarcinoma |
| PRAD | Prostate adenocarcinoma |

---

## Key features

- **Dimensionality reduction:** 20,531 genes → a 200-gene panel (ANOVA F-test) → 4 principal components
- **Two architectures compared:** Random Forest and Neural Network (MLP), tuned with GridSearchCV
  and 5-fold stratified cross-validation, scored by macro F1
- **Class imbalance handled** with SMOTE (BRCA has 300 patients, COAD only 78)
- **Explainability:** the top driving genes for each patient's prediction (Shapley values on the
  PCA components, mapped back to the panel genes)
- **Low-confidence flag:** predictions below a confidence threshold are marked for expert review
- **Web app:** single-patient and batch (CSV) predictions, with an evaluation report when true labels are supplied
- **Structured logging** of every step to `logs/pipeline.log`, and **automated tests** with pytest

---

## Results (held-out test set, 161 patients)

| Model | Accuracy | Macro F1 | CV macro F1 |
| :--- | :---: | :---: | :---: |
| Random Forest | 0.994 | 0.992 | [from log] |
| Neural Network | 1.000 | 1.000 | [from log] |

The Random Forest misclassified one COAD patient as BRCA; the Neural Network classified every test patient
correctly. With 161 test patients, a one-patient difference is within chance, so the models perform
equivalently. See `reports/model_comparison.csv` for the full comparison.

---

## Pipeline

```
20,531 genes
  → remove zero-variance genes
  → select 200-gene panel (ANOVA F-test)
  → standardise (z-scores)
  → PCA: 4 components
  → SMOTE (balance tumour types; training only)
  → Random Forest or MLP
```

Every step is inside one scikit-learn / imbalanced-learn pipeline, so gene selection, scaling and PCA
are learned from the **training data only**, in every cross-validation fold. This prevents data leakage.

**How the main settings were chosen** (see `src/explore.py`):

- **Panel size (200 genes):** cross-validated macro F1 plateaued from about 20 genes; 200 gave the highest,
  most stable score and visibly better separation of BRCA, LUAD and COAD.
- **PCA components (4):** the panel's scree plot shows a sharp elbow at 4 components (~85% of the variance),
  consistent with five classes needing at most four dimensions. Each component separates mainly one
  tumour type (PC1 KIRC, PC2 PRAD, PC3 COAD, PC4 LUAD; BRCA by their combination).

---

## Project structure

```
tcga_tumour_classifier/
├── app.py                  Flask web app (single and batch endpoints)
├── templates/index.html    The web page
├── src/
│   ├── settings.py         All paths, settings and decisions in one place
│   ├── logger.py           Logging set-up
│   ├── explore.py          Data exploration: PCA plots, panel size and component tests
│   ├── train.py            Trains, tunes, evaluates and saves both models
│   └── classifier.py       TumourClassifier: validate, predict, embed, explain
├── tests/
│   └── test_classifier.py  Automated tests (pytest)
├── reports/                Results tables and figures
├── data/                   Input data (not in the repository; see below)
├── models/                 Trained models (created by train.py)
└── logs/                   Log files (created automatically)
```

---

## Getting started

### 1. Get the data

The data is not included in this repository. Download the **"gene expression cancer RNA-Seq"** dataset
from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

Extract it and place `data.csv` and `labels.csv` in the `data/` folder.

### 2. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

### 3. Run

Run everything **from the project root**:

```bash
python3 -m src.explore     # optional: exploration plots and tests -> reports/figures/
python3 -m src.train       # train, tune and evaluate both models -> models/, reports/
python3 -m pytest -v       # run the automated tests
python3 app.py             # start the web app
```

Then open http://127.0.0.1:5000 in your browser.

---

## Using the web app

**Single patient:** choose one of the example patients (unseen test-set patients saved by `train.py`),
or upload a CSV containing one patient.

**Batch screening:** upload a CSV with one patient per row, in the same format as `data.csv`:

- first column: sample ID
- then one column per gene (`gene_0`, `gene_1`, …)
- the **200 panel genes are required**; other genes may be missing
- optional `Class` column with the true tumour types: adds accuracy and macro F1 to the results

For each patient, the app shows the predicted tumour type, the probability of each type, the
gene expression embedding (PCA scores) and the top driving genes. Predictions below the confidence
threshold are flagged for expert review rather than reported as a diagnosis.

---

## Limitations

- **Anonymised genes:** the dataset labels genes as `gene_0` … `gene_20530`, so the panel and the
  driving genes can be reported only by index. Biological interpretation (e.g. pathway enrichment)
  would require mapping them to gene symbols.
- **One dataset:** all samples come from TCGA. Performance on data from other hospitals or
  sequencing platforms has not been tested.
- **Small test set:** 161 patients, with only 16 COAD. A single misclassification noticeably changes
  per-class scores.
- **Highly separable task:** these five tumour types differ strongly in expression (largely reflecting
  tissue of origin), so near-perfect scores are expected and may not carry over to harder problems,
  such as distinguishing subtypes.