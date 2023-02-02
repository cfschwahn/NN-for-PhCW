## script to run a speed test of the final NN on cluster for comparison with MPB 2D and 3D simulations
# Caspar Schwahn Oct 2022

import csv
import numpy as np
import pandas as pd
import pathlib

import tensorflow as tf
from tensorflow import keras

from datetime import datetime

# Import data

train_dir = pathlib.Path("/home/cfs4/PhCW2022/NN/datasets/branching-bands-101-3-3/")

X_train = pd.read_csv(train_dir / "X_train.csv", header=[0])
y_odd_train = pd.read_csv(train_dir/"y_odd_train.csv", header = [0,1])
y_even_train = pd.read_csv(train_dir/"y_even_train.csv", header = [0,1])

# Convert to numpy arrays
X_train = X_train.to_numpy()
y_odd_train = y_odd_train.to_numpy()
y_even_train = y_even_train.to_numpy()

# Import model
import_model_dir = pathlib.Path("/home/cfs4/PhCW2022/NN/tuning/t-7/model/")
model = keras.models.load_model(import_model_dir)

# start timing
tik = datetime.now()

# run inference
model.predict(X_train)

#stop timing
tok = datetime.now()

print(f"Runtime (s): {str(tok-tik)}")
print(f"time per design (s): {(tok-tik).total_seconds()/len(X_train)}")
