import numpy as np
import pandas as pd
import os

SRC = "data/airq/airq_m200.txt"
OUT_DIR = "data/airq"

print("Loading:", SRC)
df = pd.read_csv(SRC, sep=r"\s+", engine="python")

X = df.values.astype(float)

# No missing values originally → fill A with 1s
A = np.ones_like(X, dtype=int)

np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "A.npy"), A)

print("Saved:")
print(" - data/airq/X.npy  shape", X.shape)
print(" - data/airq/A.npy  shape", A.shape)
