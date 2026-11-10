#This is a Jupyter notebook that explores the data in the file specified by DATA_FILE in src/settings.py.
#It prints the Python executable and numpy version, the current working directory,
#and the shape of the DataFrame loaded from the CSV file.

# %%
import sys, numpy
print(sys.executable, numpy.__version__)

# %%
import os
print(os.getcwd())

# %%
import pandas as pd
from src.settings import DATA_FILE
X = pd.read_csv(DATA_FILE, index_col=0)

# %%
print(X.shape)
# %%
