## Script to create train, val, test sets for band structure prediciton

import sys
import pandas as pd
import pathlib

out_dir = "/home/nanophotgrp/PhCW2022/NN-for-PhCW/training-sets/unsorted-bands-101-5/"
bands_path = "/home/nanophotgrp/PhCW2022/NN-for-PhCW/bands-gen/combined/combined_101_5_unsorted.csv"

out_dir = pathlib.Path(out_dir)
if not out_dir.exists():
    sys.exit("Output directory not found")

bands_csv = pathlib.Path(bands_path)
if not bands_csv.exists():
    sys.exit("Bands csv not found")

print("reading: ", bands_csv)

df = pd.read_csv(bands_csv, header=[0,1])

# names of design parameter columns
d_cols = ['r0', 'r1', 'r2', 'r3', 's1', 's2', 's3']
print("selecting parameters:", d_cols)
          
# Select bands to be fit
bands = ["band_21","band_22","band_23","band_24","band_25"]
print("selecting bands:", bands)

X = df["design_params"][d_cols].copy()
y = df[bands].copy()

# Widest possible range of desing parameters possible in 10D space.
cvec = {'r0': (0.2,0.45), 'r1': (0.2,0.45), 'r2': (0.2,0.45), 'r3':(0.2,0.45),
                          's1': (-2.165063509461097, 0.616025),
                          's2': (-1.73205080756, 1.4820508), 
                          's3': (-0.86602540378, 1.915063509),
                          'p1': (-0.5, 0.5), 'p2': (-0.5, 0.5), 'p3': (-0.5, 0.5)}

print("Normalising design parameters to the following ranges:")
print(cvec)

# normalised the 7D design vectors to 0-1 range using cvec
for feature in d_cols:
    min_f = cvec[feature][0]
    range_f = cvec[feature][1]-cvec[feature][0]
    X[feature]=(X[feature]-min_f)/range_f
print("Check scaling of design parameters:")
print(X.describe())

# train, validation, test split 
n_samples = len(df)
train_frac = 0.7
val_frac = 0.15
test_frac = 1-(val_frac+train_frac)

train_rest_split_i = int(n_samples*train_frac) # index to split train and validation/test
val_test_split_i = int(n_samples*(train_frac+val_frac)) # index to split validation and test

print("Number of samples in dataset {}".format(n_samples))
print("train:val:test ratio {:.2f}:{:.2f}:{:.2f}".format(train_frac,val_frac, test_frac))

X_train = X.iloc[:train_rest_split_i,:]
X_val = X.iloc[train_rest_split_i:val_test_split_i,:]
X_test = X.iloc[val_test_split_i:,:]

y_train = y.iloc[:train_rest_split_i,:]
y_val = y.iloc[train_rest_split_i:val_test_split_i,:]
y_test = y.iloc[val_test_split_i:,:]

print("Number of samples in X train:val:test {}:{}:{}".format(len(X_train),len(X_val),len(X_test)))
print("Number of samples in y train:val:test {}:{}:{}".format(len(y_train),len(y_val),len(y_test)))

# Save to file
to_save = [(y_train, "y_train"), (y_val, "y_val"), (y_test, "y_test"),
          (X_train, "X_train"), (X_val, "X_val"), (X_test, "X_test")]

for var, name in to_save:
    var.to_csv(out_dir/(name+".csv"), header=True, index=False)
    
print("Done")