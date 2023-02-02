## Script to create training, validation, test sets from bands.csv containing PhCW designs
# and MPB simulated band structures
# Caspar Schwahn, August 2022

import csv
import numpy as np
import pandas as pd
import pathlib


def prepare_training_sets(out_dir, bands_csv_path)
    ''' reads bands.csv and prepares training, validation, test sets which are saved in out_dir
    
    :param out_dir: directory to save datasets (must exist)
    :param bands_csv_path: path to bands.csv containing design parameters and simulated bands
    '''

    ## Define constants
    n_bands = 30
    n_k_points = 101

    # Bands to select as targets (counting begins at 1 as in MPB)
    even_band_nums = [10,11,12]
    odd_band_nums = [11,12,13]

    # design parameters to include as features 
    d_cols = ['r0', 'r1', 'r2', 'r3', 's1', 's2', 's3']

    # define train validation test split
    train_frac = 0.7
    val_frac = 0.15
    test_frac = 1-(val_frac+train_frac)

    ## Check I/O
    out_dir = pathlib.Path(out_dir)
    if not out_dir.exists():
        print(f"{out_dir} (output directory) not found.")
        return
    bands_csv_path = pathlib.Path(bands_csv_path)
    if not bands_csv_path.exists():
        print(f"{bands_csv_path} (bands.csv) not found.")

    # read csv containing parsing results (designs, bands and parities) from all log files
    all_df = pd.read_csv(bands_csv_path, header=[0,1], true_values = ["True"], false_values=["False"])
    n_samples = len(all_df)
    print(f"n samples: {n_samples}")

    # Split dataframe into relevant parts
    parity_df = all_df.iloc[:,10+n_bands*n_k_points:] # "is_even"
    bands_df = all_df.iloc[:,10:10+n_bands*n_k_points] # "band_{i}"


    ## Process features
    print("selecting parameters:", d_cols)
    X = all_df["design_params"][d_cols].copy()

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

    ## Process targets

    # convert band numbers to indices
    select_evens = np.array(even_band_nums) - 1
    select_odds = np.array(odd_band_nums) - 1

    # select correct odd even bands - slow 
    even_bands_list = []
    odd_bands_list = []
    for i in range(n_samples):
        parity_row = parity_df.iloc[i].to_numpy(dtype=bool)
        odd_is = np.where(~parity_row)[0]+1 # band numbers that are odd
        even_is = np.where(parity_row)[0]+1 # band numbers that are even
        even_bands = bands_df.iloc[[i]][[f"band_{j}" for j in even_is[select_evens]]].to_numpy()[0]
        odd_bands = bands_df.iloc[[i]][[f"band_{j}" for j in odd_is[select_odds]]].to_numpy()[0]
        even_bands_list.append(even_bands)
        odd_bands_list.append(odd_bands)

    # create multiindexed headers for new dataframes
    even_cols = [[],[]]
    for i in even_band_nums:
        for j in range(1, n_k_points+1):
            even_cols[1].append("k_{}".format(j))
            even_cols[0].append("even_band_{}".format(i))
    odd_cols = [[],[]]
    for i in odd_band_nums:
        for j in range(1, n_k_points+1):
            odd_cols[1].append("k_{}".format(j))
            odd_cols[0].append("odd_band_{}".format(i))

    # create new dataframes for selected bands of each parity
    y_odd = pd.DataFrame(np.stack(odd_bands_list), columns=odd_cols)
    y_even = pd.DataFrame(np.stack(even_bands_list), columns=even_cols)

    print("created parity dataframes with bands:")
    print(f"even: {even_band_nums}")
    print(f"odd: {odd_band_nums}")

    ## train, validation, test split 
    train_rest_split_i = int(n_samples*train_frac) # index to split train and validation/test
    val_test_split_i = int(n_samples*(train_frac+val_frac)) # index to split validation and test

    print("Number of samples in dataset {}".format(n_samples))
    print("train:val:test ratio {:.2f}:{:.2f}:{:.2f}".format(train_frac,val_frac, test_frac))

    X_train = X.iloc[:train_rest_split_i,:]
    X_val = X.iloc[train_rest_split_i:val_test_split_i,:]
    X_test = X.iloc[val_test_split_i:,:]

    y_odd_train = y_odd.iloc[:train_rest_split_i,:]
    y_odd_val = y_odd.iloc[train_rest_split_i:val_test_split_i,:]
    y_odd_test = y_odd.iloc[val_test_split_i:,:]

    y_even_train = y_even.iloc[:train_rest_split_i,:]
    y_even_val = y_even.iloc[train_rest_split_i:val_test_split_i,:]
    y_even_test = y_even.iloc[val_test_split_i:,:]

    print("Number of samples in X train:val:test {}:{}:{}".format(len(X_train),len(X_val),len(X_test)))
    print("Number of samples in y odd train:val:test {}:{}:{}".format(len(y_odd_train),len(y_odd_val),len(y_odd_test)))
    print("Number of samples in y even train:val:test {}:{}:{}".format(len(y_even_train),len(y_even_val),len(y_even_test)))

    ## Save files
    to_save = [(y_odd_train, "y_odd_train"), (y_odd_val, "y_odd_val"), (y_odd_test, "y_odd_test"),
                (y_even_train, "y_even_train"), (y_even_val, "y_even_val"), (y_even_test, "y_even_test"),
                (X_train, "X_train"), (X_val, "X_val"), (X_test, "X_test")]

    for var, name in to_save:
        var.to_csv(out_dir/(name+".csv"), header=True, index=False)
        
    print("Done")

if __name__ == "__main__":
    '''
    bands_csv_path = "/home/nanophotgrp/PhCW2022/NN-for-PhCW/bands-gen/101k-30b-parity/combined/combined_101_5_parity_sorted.csv"
    out_dir = "/home/nanophotgrp/PhCW2022/NN-for-PhCW/training-sets/branching-bands-101-3-3/"
    '''

    bands_csv_path = "/home/nanophotgrp/PhCW2022/NN-for-PhCW/bands-gen/gen-3d/combined/combined_101_30_parity_sorted_3d.csv"
    out_dir = "/home/nanophotgrp/PhCW2022/NN-for-PhCW/training-sets/3d-branching-bands-101-3-3/"

    prepare_training_sets(out_dir, bands_csv_path)