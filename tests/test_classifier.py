# test_classifier.py
"""
Automated checks for TumourClassifier (src/classifier.py).

Run from the project root:   python3 -m pytest -v

Needs the trained models (python3 -m src.train) and the example patients file
that train.py writes. If they are missing, the tests are skipped, not failed.
"""

import numpy as np
import pandas as pd
import pytest

from src.classifier import TumourClassifier
from src.settings import (CLASS_NAMES, EXAMPLES_FILE, EXPRESSION_RANGE, MODEL_FILES,
                          N_COMPONENTS, N_GENES, TOP_N_GENES)

# Skip everything if training hasn't been run yet
if not all(path.exists() for path in MODEL_FILES.values()) or not EXAMPLES_FILE.exists():
    pytest.skip("Trained models or example patients not found: run python3 -m src.train first",
                allow_module_level=True)


# --------------------------------------------------
# FIXTURES: things several tests share, created once
# --------------------------------------------------
@pytest.fixture(scope="module", params=list(MODEL_FILES))
def clf(request):
    """Each test that uses 'clf' runs once per model (Random Forest AND Neural Network)."""
    return TumourClassifier(request.param)


@pytest.fixture(scope="module")
def examples():
    """Unseen test patients saved by train.py: (gene profiles, true tumour types)."""
    df = pd.read_csv(EXAMPLES_FILE, index_col=0)
    return df.drop(columns="Class"), df["Class"]


# --------------------------------------------------
# 1. LOADING
# --------------------------------------------------
def test_model_loads_with_expected_panel_and_classes(clf):
    assert len(clf.panel_genes) == N_GENES
    assert sorted(clf.classes) == sorted(CLASS_NAMES)
    assert set(clf.panel_genes) <= set(clf.all_genes)          # panel genes are model inputs


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError):
        TumourClassifier("Not A Model")


# --------------------------------------------------
# 2. PREDICTIONS
# --------------------------------------------------
def test_predictions_are_valid_tumour_types(clf, examples):
    profiles, _ = examples
    results, problems = clf.run(profiles)
    assert problems == {}                                       # clean examples: nothing rejected
    assert len(results) == len(profiles)
    assert results["predicted"].isin(CLASS_NAMES).all()


def test_probabilities_sum_to_one_and_match_confidence(clf, examples):
    profiles, _ = examples
    results, _ = clf.run(profiles)
    probabilities = results[CLASS_NAMES]
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(results["confidence"], probabilities.max(axis=1))
    # the predicted type is the one with the highest probability
    assert (results["predicted"] == probabilities.idxmax(axis=1)).all()


def test_low_confidence_flag_follows_threshold(clf, examples):
    from src.settings import CONFIDENCE_THRESHOLD
    profiles, _ = examples
    results, _ = clf.run(profiles)
    assert (results["low_confidence"] == (results["confidence"] < CONFIDENCE_THRESHOLD)).all()


def test_accuracy_on_unseen_examples_is_reasonable(clf, examples):
    """A sanity check, not the evaluation: the real evaluation is in train.py."""
    profiles, truth = examples
    results, _ = clf.run(profiles)
    accuracy = (results["predicted"] == truth[results.index]).mean()
    assert accuracy >= 0.8


# --------------------------------------------------
# 3. EMBEDDINGS AND EXPLANATIONS
# --------------------------------------------------
def test_embedding_has_one_score_per_component(clf, examples):
    profiles, _ = examples
    results, _ = clf.run(profiles)
    pcs = [c for c in results.columns if c.startswith("PC")]
    assert pcs == [f"PC{i}" for i in range(1, N_COMPONENTS + 1)]
    assert results[pcs].notna().all().all()


def test_top_genes_are_panel_genes(clf, examples):
    profiles, _ = examples
    results, _ = clf.run(profiles.head(3))
    for top_genes in results["top_genes"]:
        assert len(top_genes) == TOP_N_GENES
        genes = [gene for gene, _ in top_genes]
        assert set(genes) <= set(clf.panel_genes)
        contributions = [value for _, value in top_genes]
        assert contributions == sorted(contributions, reverse=True)   # strongest first


# --------------------------------------------------
# 4. VALIDATION: bad input must be caught, not silently predicted
# --------------------------------------------------
def test_missing_panel_gene_column_rejects_the_file(clf, examples):
    profiles, _ = examples
    with pytest.raises(ValueError, match="panel genes are missing"):
        clf.run(profiles.drop(columns=clf.panel_genes[0]))


def test_missing_or_text_value_rejects_only_that_patient(clf, examples):
    profiles, _ = examples
    bad = profiles.copy().astype(object)
    bad.iloc[0, bad.columns.get_loc(clf.panel_genes[0])] = np.nan
    bad.iloc[1, bad.columns.get_loc(clf.panel_genes[1])] = "abc"
    results, problems = clf.run(bad)
    assert set(problems) == {bad.index[0], bad.index[1]}
    assert len(results) == len(bad) - 2                        # everyone else still predicted


def test_out_of_range_value_is_rejected(clf, examples):
    profiles, _ = examples
    bad = profiles.copy()
    bad.iloc[0, bad.columns.get_loc(clf.panel_genes[0])] = EXPRESSION_RANGE[1] + 100
    _, problems = clf.run(bad)
    assert bad.index[0] in problems
    assert "outside" in problems[bad.index[0]]


def test_missing_non_panel_genes_do_not_change_predictions(clf, examples):
    """Non-panel genes are discarded by the model, so leaving them out must not matter."""
    profiles, _ = examples
    non_panel = [g for g in clf.all_genes if g not in clf.panel_genes]
    full, _ = clf.run(profiles)
    reduced, problems = clf.run(profiles.drop(columns=non_panel[:50]))
    assert problems == {}
    assert (full["predicted"] == reduced["predicted"]).all()
    assert np.allclose(full[CLASS_NAMES], reduced[CLASS_NAMES])


def test_column_order_and_extra_columns_do_not_matter(clf, examples):
    profiles, _ = examples
    full, _ = clf.run(profiles)
    shuffled = profiles[profiles.columns[::-1]].copy()
    shuffled["extra_column"] = 1
    other, _ = clf.run(shuffled)
    assert np.allclose(full[CLASS_NAMES], other[CLASS_NAMES])