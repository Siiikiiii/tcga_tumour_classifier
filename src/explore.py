#This is a Jupyter notebook that explores the data in the file specified by DATA_FILE in src/settings.py.
"""
First look at the TCGA data: shapes, alignment, class balance, missing
values, value range, zero-variance genes, and a PCA plot of the tumours.
 
Run from the project root:   python3 -m src.explore
Results are shown in the terminal and saved to logs/pipeline.log;
the PCA plot is saved to reports/figures/.
"""
 
# %% Setup
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger import setup_logging
from src.settings import CLASS_NAMES, DATA_FILE, FIGURES_DIR, LABELS_FILE, RANDOM_STATE, REPORTS_DIR

setup_logging()
logger = logging.getLogger(__name__)
# %% 1. Load both files
# index_col=0: the first (unnamed) column holds the sample IDs, so it becomes
# the row label instead of being read as a gene
X = pd.read_csv(DATA_FILE, index_col=0)
labels = pd.read_csv(LABELS_FILE, index_col=0)
logger.info("Expression data: %d samples x %d genes", X.shape[0], X.shape[1])
logger.info("Labels: %d samples x %d column(s): %s", labels.shape[0], labels.shape[1], list(labels.columns))
 
 
# %% 2. Do the two files line up?
# Same sample IDs in the same order -> row i of X belongs to row i of labels
if X.index.equals(labels.index):
    logger.info("Sample IDs match exactly in both files")
else:
    logger.error("Sample IDs do NOT match between data.csv and labels.csv")
    raise SystemExit("Stopping: fix the sample alignment before going further")
 
y = labels["Class"]
 
 
# %% 3. Class balance
counts = y.value_counts()
for tumour, n in counts.items():
    logger.info("Class %-4s: %3d samples (%.1f%%)", tumour, n, 100 * n / len(y))
 
unexpected = set(counts.index) - set(CLASS_NAMES)
if unexpected:
    logger.warning("Labels not in settings.CLASS_NAMES: %s", unexpected)
 
 
# %% 4. Missing values
n_missing = int(X.isna().sum().sum())
logger.info("Missing values in expression data: %d", n_missing)
 
 
# %% 5. Value range (is the data already log-transformed?)
logger.info("Expression values range from %.2f to %.2f", X.min().min(), X.max().max())
logger.info("Share of values that are exactly 0: %.1f%%", 100 * (X == 0).to_numpy().mean())
 
 
# %% 6. Zero-variance genes (same value in every sample: no information)
zero_var = X.columns[X.std() == 0]
logger.info("Zero-variance genes: %d of %d", len(zero_var), X.shape[1])
logger.debug("First zero-variance genes: %s", list(zero_var[:20]))
 
 
# %% 7. PCA plot: how well do the tumour types separate?
# For EXPLORATION only: fitted on all samples to look at the data.
# In training, scaling and PCA must be fitted on the training set only.
X_informative = X.drop(columns=zero_var)
X_scaled = StandardScaler().fit_transform(X_informative)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(X_scaled)
var1, var2 = pca.explained_variance_ratio_ * 100
logger.info("PC1 explains %.1f%% of the variance, PC2 %.1f%%", var1, var2)
 
fig, ax = plt.subplots(figsize=(8, 6))
for tumour in CLASS_NAMES:
    mask = (y == tumour).to_numpy()
    ax.scatter(coords[mask, 0], coords[mask, 1], label=f"{tumour} (n={mask.sum()})", s=15, alpha=0.7)
ax.set_xlabel(f"PC1 ({var1:.1f}% of variance)")
ax.set_ylabel(f"PC2 ({var2:.1f}% of variance)")
ax.set_title("TCGA tumour samples: first two principal components")
ax.legend(title="Tumour type")
 
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plot_path = FIGURES_DIR / "pca_2d.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
logger.info("PCA plot saved to %s", plot_path)
 
 
# %% 8. Scree (elbow) plot: how much variance does each component capture?
# Fit PCA with more components on the same scaled data as the 2D plot
N_SCREE = 50
pca_full = PCA(n_components=N_SCREE, random_state=RANDOM_STATE).fit(X_scaled)
explained = pca_full.explained_variance_ratio_ * 100
cumulative = explained.cumsum()
 
# How many components reach common cut-offs? (within the first N_SCREE)
for target in (80, 90, 95):
    reached = (cumulative >= target).nonzero()[0]
    if len(reached):
        logger.info("Components needed for %d%% of the variance: %d", target, reached[0] + 1)
    else:
        logger.info("Components needed for %d%% of the variance: more than %d", target, N_SCREE)
logger.info("First 3 components explain %.1f%% of the variance in total", cumulative[2])
 
fig, ax1 = plt.subplots(figsize=(9, 5))
components = range(1, N_SCREE + 1)
ax1.bar(components, explained, color="steelblue", label="Each component")
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Variance explained by each component (%)")
 
ax2 = ax1.twinx()                                   # second y-axis for the running total
ax2.plot(components, cumulative, color="darkorange", marker="o", markersize=3, label="Cumulative")
ax2.set_ylabel("Cumulative variance explained (%)")
ax2.set_ylim(0, 100)
 
ax1.set_title("Scree plot: variance explained by the first principal components")
fig.legend(loc="center right", bbox_to_anchor=(0.88, 0.5))
scree_path = FIGURES_DIR / "pca_scree.png"
fig.savefig(scree_path, dpi=150, bbox_inches="tight")
logger.info("Scree plot saved to %s", scree_path)
 
 
# %% 9. Interactive 3D plot: open the HTML file in a browser and drag to rotate
coords3 = pca_full.transform(X_scaled)[:, :3]
plot_df = pd.DataFrame(coords3, columns=["PC1", "PC2", "PC3"], index=X.index)
plot_df["Tumour type"] = y
plot_df["Sample"] = X.index
 
fig3d = px.scatter_3d(
    plot_df, x="PC1", y="PC2", z="PC3", color="Tumour type",
    category_orders={"Tumour type": CLASS_NAMES}, hover_name="Sample",
    labels={"PC1": f"PC1 ({explained[0]:.1f}%)", "PC2": f"PC2 ({explained[1]:.1f}%)",
            "PC3": f"PC3 ({explained[2]:.1f}%)"},
    title="TCGA tumour samples: first three principal components (drag to rotate)",
)
fig3d.update_traces(marker=dict(size=5))                    # CHANGED: points 3 -> 5
fig3d.update_layout(                                        # NEW: bigger legend
    legend=dict(
        font=dict(size=16),          # tumour type names
        title_font=dict(size=18),    # the "Tumour type" heading
        itemsizing="constant",       # legend dots a fixed, larger size
    )
)
html_path = FIGURES_DIR / "pca_3d.html"
fig3d.write_html(html_path)
logger.info("Interactive 3D plot saved to %s", html_path)
# %%

# %% 10. How many genes does the panel need? (no PCA)
X_all = X_informative          # all patients, zero-variance genes already removed
y_all = y
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

panel_results = []
for n_genes in (10, 20, 50, 100, 200, 500, 1000, 2000):
    if n_genes > X_all.shape[1]:                     # can't select more genes than exist
        continue
    model = Pipeline([
        ("select", SelectKBest(f_classif, k=n_genes)),   # the panel: re-chosen in every fold
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    scores = cross_val_score(model, X_all, y_all, cv=cv, scoring="f1_macro")
    logger.info("Panel of %4d genes: macro F1 %.3f (+/- %.3f)", n_genes, scores.mean(), scores.std())
    panel_results.append({"n_genes": n_genes, "mean_f1": scores.mean(), "std_f1": scores.std()})

logger.info("Panel results: %s", panel_results)

# Plot: panel size vs macro F1 (log scale, because sizes roughly double each step)
CHOSEN_N_GENES = 200                                 # update after looking at the results
panel_df = pd.DataFrame(panel_results)

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(panel_df["n_genes"], panel_df["mean_f1"], yerr=panel_df["std_f1"],
            marker="o", capsize=4, color="seagreen", label="Mean macro F1 (± 1 SD across folds)")
ax.axvline(CHOSEN_N_GENES, color="darkorange", linestyle="--", label=f"Chosen: {CHOSEN_N_GENES} genes")
ax.set_xscale("log")
ax.set_xticks(panel_df["n_genes"])
ax.set_xticklabels(panel_df["n_genes"])              # show 10, 20, 50 ... instead of 10^1, 10^2
ax.set_xlabel("Number of genes in the panel (log scale)")
ax.set_ylabel("Cross-validated macro F1")
ax.set_title("Classification performance vs gene panel size (5-fold CV, all tumour types)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

panel_plot_path = FIGURES_DIR / "panel_size_f1.png"
fig.savefig(panel_plot_path, dpi=150, bbox_inches="tight")
logger.info("Panel size vs macro F1 plot saved to %s", panel_plot_path)

# %% 11a. How does the panel size change the 2D PCA picture? (one figure per size)
# VISUALISATION ONLY: the panel is chosen with the labels of ALL patients here,
# so these plots look cleaner partly by design. Panel size is decided by the
# cross-validated numbers in section 11, not by these pictures.
TUMOUR_COLOURS = {"BRCA": "#636EFA", "COAD": "#EF553B", "KIRC": "#00CC96",
                  "LUAD": "#AB63FA", "PRAD": "#FFA15A"}      # same colours as the plotly figures

PANEL_DIR = FIGURES_DIR / "pca_by_panel_size"                # NEW: own folder, as there are 9 files
PANEL_DIR.mkdir(parents=True, exist_ok=True)

views = [("All genes", None)] + [(f"{n}-gene panel", n)       # CHANGED: every size from section 11
                                 for n in (10, 20, 50, 100, 200, 500, 1000, 2000)]

for title, n_genes in views:                                 # CHANGED: one figure per loop
    if n_genes is None:
        X_view = X_informative
        file_name = "pca_2d_all_genes.png"
    else:
        if n_genes > X_informative.shape[1]:                 # can't select more genes than exist
            continue
        selector = SelectKBest(f_classif, k=n_genes).fit(X_informative, y)
        X_view = X_informative.loc[:, selector.get_support()]
        file_name = f"pca_2d_panel_{n_genes:04d}.png"        # 0010, 0020 ... so files sort in order

    pca_view = PCA(n_components=2, random_state=RANDOM_STATE)
    coords_view = pca_view.fit_transform(StandardScaler().fit_transform(X_view))
    v1, v2 = pca_view.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6))                   # CHANGED: a new figure each time
    for tumour in CLASS_NAMES:
        mask = (y == tumour).to_numpy()
        ax.scatter(coords_view[mask, 0], coords_view[mask, 1], s=12, alpha=0.7,
                   color=TUMOUR_COLOURS[tumour], label=tumour)
    ax.set_title(f"2D PCA: {title}")
    ax.set_xlabel(f"PC1 ({v1:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({v2:.1f}% of variance)")
    ax.legend(title="Tumour type", markerscale=2)

    fig.savefig(PANEL_DIR / file_name, dpi=150, bbox_inches="tight")
    plt.close(fig)                                           # NEW: free memory before the next figure
    logger.info("%s: PC1 + PC2 explain %.1f%% of the variance -> %s", title, v1 + v2, file_name)

# %% 11b. The 200-gene panel: select once, save, and reuse on later runs
# EXPLORATION ONLY: chosen with the labels of all patients. In train.py the panel
# is re-selected inside the pipeline from training patients only.
N_GENES = 200
PANEL_FILE = REPORTS_DIR / f"panel_{N_GENES}_genes.csv"

if PANEL_FILE.exists():
    # Already calculated on an earlier run: just read the gene list
    panel_df = pd.read_csv(PANEL_FILE)
    logger.info("Loaded %d panel genes from %s", len(panel_df), PANEL_FILE)
else:
    # First run: rank all genes by F score, keep the top N_GENES, save them
    selector = SelectKBest(f_classif, k=N_GENES).fit(X_informative, y)
    panel_df = (pd.DataFrame({"gene": X_informative.columns, "f_score": selector.scores_})
                  .loc[selector.get_support()]
                  .sort_values("f_score", ascending=False)       # strongest gene first
                  .reset_index(drop=True))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    panel_df.to_csv(PANEL_FILE, index=False)
    logger.info("Selected %d panel genes and saved them to %s", len(panel_df), PANEL_FILE)

panel_genes = panel_df["gene"].tolist()
logger.info("Top 5 panel genes: %s", panel_genes[:5])
X_panel = X_informative[panel_genes]

# %% 11c. NEW: Scree plot for the 200-gene panel
N_SCREE_PANEL = min(50, N_GENES)                     # can't have more components than genes
X_panel_scaled = StandardScaler().fit_transform(X_panel)
pca_panel = PCA(n_components=N_SCREE_PANEL, random_state=RANDOM_STATE).fit(X_panel_scaled)
explained_p = pca_panel.explained_variance_ratio_ * 100
cumulative_p = explained_p.cumsum()

for target in (80, 90, 95):
    reached = (cumulative_p >= target).nonzero()[0]
    if len(reached):
        logger.info("Panel: components needed for %d%% of the variance: %d", target, reached[0] + 1)
    else:
        logger.info("Panel: components needed for %d%% of the variance: more than %d", target, N_SCREE_PANEL)

fig, ax1 = plt.subplots(figsize=(9, 5))
components = range(1, N_SCREE_PANEL + 1)
ax1.bar(components, explained_p, color="steelblue", label="Each component")
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Variance explained by each component (%)")
ax2 = ax1.twinx()
ax2.plot(components, cumulative_p, color="darkorange", marker="o", markersize=3, label="Cumulative")
ax2.set_ylabel("Cumulative variance explained (%)")
ax2.set_ylim(0, 100)
ax1.set_title(f"Scree plot: {N_GENES}-gene panel")
fig.legend(loc="center right", bbox_to_anchor=(0.88, 0.5))

scree_panel_path = FIGURES_DIR / f"pca_scree_panel_{N_GENES}.png"
fig.savefig(scree_panel_path, dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Panel scree plot saved to %s", scree_panel_path)

# %% 12. Which panel component separates which tumour type?
N_PCS = 4
coords_panel = pca_panel.transform(X_panel_scaled)[:, :N_PCS]
pc_names = [f"PC{i}" for i in range(1, N_PCS + 1)]
panel_pcs = pd.DataFrame(coords_panel, columns=pc_names, index=X.index)
panel_pcs["Tumour type"] = y

# --- Figure 1: every pair of PC1-PC4 (interactive)
fig_pairs4 = px.scatter_matrix(
    panel_pcs, dimensions=pc_names, color="Tumour type",
    category_orders={"Tumour type": CLASS_NAMES},
    labels={pc: f"{pc} ({explained_p[i]:.1f}%)" for i, pc in enumerate(pc_names)},
    title=f"{N_GENES}-gene panel: every pair of PC1-PC{N_PCS}",
)
fig_pairs4.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=4))
fig_pairs4.update_layout(width=900, height=900,
                         legend=dict(font=dict(size=16), title_font=dict(size=18), itemsizing="constant"))
pairs4_path = FIGURES_DIR / f"panel_{N_GENES}_pc1_to_pc{N_PCS}_pairs.html"
fig_pairs4.write_html(pairs4_path)
logger.info("Panel PC1-PC%d scatter matrix saved to %s", N_PCS, pairs4_path)

# --- Figure 2: one box plot per component, tumour types side by side
fig, axes = plt.subplots(1, N_PCS, figsize=(4.5 * N_PCS, 5), sharey=False)
for ax, pc in zip(axes, pc_names):
    data = [panel_pcs.loc[panel_pcs["Tumour type"] == t, pc] for t in CLASS_NAMES]
    box = ax.boxplot(data, patch_artist=True, widths=0.6)
    ax.set_xticks(range(1, len(CLASS_NAMES) + 1), CLASS_NAMES)
    for patch, tumour in zip(box["boxes"], CLASS_NAMES):
        patch.set_facecolor(TUMOUR_COLOURS[tumour])
        patch.set_alpha(0.7)
    ax.set_title(f"{pc} ({explained_p[pc_names.index(pc)]:.1f}% of variance)")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
axes[0].set_ylabel("Component score")
fig.suptitle(f"{N_GENES}-gene panel: which component separates which tumour type?", fontsize=14)
fig.tight_layout()
box_path = FIGURES_DIR / f"panel_{N_GENES}_pc_boxplots.png"
fig.savefig(box_path, dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Panel component box plots saved to %s", box_path)

# --- In numbers: average score of each tumour type on each component
logger.info("Average component score per tumour type:\n%s",
            panel_pcs.groupby("Tumour type")[pc_names].mean().round(1).to_string())