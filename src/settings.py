# settings.py
"""
Central configuration for the TCGA tumour classifier: project paths, how
logging is set up, and the constants shared by training, prediction and
the web app.
 
Every other module imports from here, so each path or setting is defined
in exactly one place.
 
The logging configuration follows the dictionary schema described in the
Python documentation:
https://docs.python.org/3/library/logging.config.html#logging-config-dictschema
"""
 
from pathlib import Path
 
# --------------------------------------------------
# Project paths
# --------------------------------------------------
# CHANGED: all paths are built from this file's location (src/settings.py),
# so the project finds its folders wherever it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
 
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "data.csv"          # gene expression: patients x genes
LABELS_FILE = DATA_DIR / "labels.csv"      # tumour type for each patient
 
MODEL_DIR = PROJECT_ROOT / "models"        # NEW
MODEL_FILE = MODEL_DIR / "tumour_pipeline.joblib"
 
REPORTS_DIR = PROJECT_ROOT / "reports"     # NEW
FIGURES_DIR = REPORTS_DIR / "figures"
 
 
# --------------------------------------------------
# Log file location
# --------------------------------------------------
# CHANGED: logs go to <project>/logs/ rather than a hidden file in the home
# directory. This project is run from its own folder (not installed into
# site-packages), and keeping the logs next to the code makes them easy to
# inspect and to show during the project walkthrough.
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "pipeline.log"
 
# CHANGED: the os.makedirs(...) call that created the log folder has moved to
# setup_logging() in logger.py, so importing settings has no side effects.
 
 
# --------------------------------------------------
# Logging configuration
# --------------------------------------------------
# The console shows INFO and above (progress, scores) so the terminal stays
# readable; the log file also keeps DEBUG messages for troubleshooting.
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
 
    # --------------------------------------------------
    # FORMATTERS
    # --------------------------------------------------
    "formatters": {
        "standard": {
            "format": (
                "%(asctime)s | %(levelname)s | %(name)s | "
                "%(filename)s:%(lineno)d | %(message)s"
            )
        }
    },
 
    # --------------------------------------------------
    # HANDLERS
    # --------------------------------------------------
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
        },
        "file": {
            # CHANGED: RotatingFileHandler starts a new file at ~5 MB and keeps
            # the 3 most recent, so the log can't grow without limit.
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": str(LOG_FILE),
            "maxBytes": 5_000_000,
            "backupCount": 3,
            "encoding": "utf-8",
        },
    },
 
    # The project's own loggers write directly to both handlers.
    # propagate=False stops each message also reaching the root logger,
    # which has the same handlers and would log it a second time.
    "loggers": {
        # CHANGED: "hgnc_app" -> "src". Modules in src/ get names like
        # "src.train" from logging.getLogger(__name__), so they all match here.
        "src": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        # NEW: a script run with "python3 -m src.explore" is named "__main__",
        # not "src.explore", so it needs the same settings to be logged.
        "__main__": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
 
    # --------------------------------------------------
    # ROOT LOGGER
    # --------------------------------------------------
    # Catches messages from third-party libraries (Flask, scikit-learn,
    # matplotlib, SHAP).
    # CHANGED: level DEBUG -> WARNING. matplotlib and SHAP (via numba) write
    # thousands of DEBUG lines per run; at WARNING, only their real problems
    # are recorded, so the project's own messages aren't buried.
    "root": {
        "level": "WARNING",
        "handlers": ["console", "file"],
    },
}


# --------------------------------------------------
# NEW: Modelling settings shared by training, prediction and the app
# --------------------------------------------------
RANDOM_STATE = 42          # same seed everywhere, so results are reproducible
TEST_SIZE = 0.2            # 80/20 train/test split
CV_FOLDS = 5               # k for k-fold cross-validation
N_COMPONENTS = 4                # PCA: number of panel components to keep
N_GENES = 200                     # number of top genes to select for the panel

# Model files
MODEL_FILES = {
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "Neural Network": MODEL_DIR / "neural_network.joblib",
}
EXPRESSION_RANGE = (0.0, 30.0)

# The five tumour types (labels.csv), in a fixed order for reports and plots
CLASS_NAMES = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]

# Number of top driving genes to show for each patient's prediction
TOP_N_GENES = 10

# Example patients
EXAMPLES_FILE = DATA_DIR / "sample" / "example_patients.csv"
N_EXAMPLES = 10

# Presentation question 4: if the highest class probability is below this,
# the prediction is flagged as low confidence instead of being reported as
# a firm diagnosis. Choose and justify the value yourself (placeholder).
CONFIDENCE_THRESHOLD = 0.60
