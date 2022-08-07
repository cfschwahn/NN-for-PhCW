"""
To be called with various fractions of the training data in each job
on HPC cluster for creation of learning curve.
Caspar Schwahn August 2022
"""

import argparse
import sys
import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorboard as tb
import json
from datetime import datetime

def build(lr=2e-5):
    """builds and compiles model
    
    All hyperparameters other than learning rate and the size of the bottleneck layer are fixed.
    Bottleneck layer: hidden layer 5. Layers 4 and 6 neurons are average of normal and bottleneck layer.
    
    :param lr: learning rate
    :param bottleneck: neurons in the bottleneck layer.
    """
    # fixed hyperparameters
    kernel_initializer = "he_normal" # recommended kernel initialisation for ReLU and variants
    leaky_alpha = 0.3 # the keras default value of alpha for Leaky ReLU
    units = 512
    hidden_layers = 9
    output_shape = 105
    input_shape = (7,)

    # Create and link layers
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(units, kernel_initializer=kernel_initializer)(inputs)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    for i in range(1,hidden_layers):
        x = layers.Dense(units, kernel_initializer=kernel_initializer)(x)
        x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    outputs = layers.Dense(output_shape)(x)

    # Create model   
    model = keras.Model(inputs=inputs, outputs=outputs, name="PhCW-regressor")
            
    # Compile model
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss = "mean_squared_error", metrics=['mean_squared_error']
    )
    
    return model

def MAPE(act, pred):
    '''
    Calculates the Mean Absolute Percentage Error from arrays of actual and predicted values.
    Arrays should be 1D and of the same length.
    
    :param act: 1D np.array of actual values
    :param pred: 1D np.array of predicted values
    :return: MAPE as percentage
    '''
    return 100/len(act)*np.sum(np.abs((act-pred)/act))

def MPE(act, pred):
    '''
    Calculates the Mean Percentage Error from arrays of actual and predicted values.
    Arrays should be 1D and of the same length.
    
    :param act: 1D np.array of actual values
    :param pred: 1D np.array of predicted values
    :return: MPE as percentage
    '''
    return 100/len(act)*np.sum((act-pred)/act)

def MSE(act, pred):
    '''
    Calculates the Mean Squared Error from arras of actual and predicted values.
    Arrays should be 1D and of the same length.
    
    :param act: 1D np.array of actual values
    :param pred: 1D np.array of predicted values
    :return: MSE
    '''
    return np.sum((act-pred)**2)/len(act)

def RMSE(act, pred):
    '''
    Calculates the Root Mean Squared Error from arras of actual and predicted values.
    Arrays should be 1D and of the same length.
    
    :param act: 1D np.array of actual values
    :param pred: 1D np.array of predicted values
    :return: RMSE
    '''
    return np.sqrt(np.sum((act-pred)**2)/len(act))

if __name__ == "__main__":

    tik = datetime.now()

    # TODO: 
    # adjust epochs and patience for dataset size
    # numpy random sampling with replacement
    #
    #
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=pathlib.Path,
        help='Path to directory with data sets for training')
    parser.add_argument('--out_dir', type=pathlib.Path,
        help='Directory in which outputs will be created')
    parser.add_argument('--out_name', type=str,
        help='Name of output folder to be created')
    parser.add_argument('--epochs', type=int,
        help='Maximum number of epochs to train for')
    parser.add_argument('--patience', type=int,
        help='Patience of early stopping callback')
    parser.add_argument('--learning_rate', type=float,
        help='Learning rate')
    parser.add_argument('--split', type=float,
        help='fraction of training set to be used')
    p = parser.parse_args()

    # Create output directory in the given directory
    out_dir = p.out_dir / p.out_name
    if not out_dir.exists():
        out_dir.mkdir()

    ## Data preparation
    # Import datasets
    X_train_df = pd.read_csv(p.train_dir / "X_train.csv", header=[0])
    X_val_df = pd.read_csv(p.train_dir / "X_val.csv", header=[0])
    X_test_df = pd.read_csv(p.train_dir / "X_test.csv", header=[0])
    y_train_df = pd.read_csv(p.train_dir / "y_train.csv", header=[0,1])
    y_val_df = pd.read_csv(p.train_dir / "y_val.csv", header=[0,1])
    y_test_df = pd.read_csv(p.train_dir / "y_test.csv", header=[0,1])


    # Convert to suitable format for TensorFlow
    X_train = X_train_df.to_numpy()
    X_val = X_val_df.to_numpy()
    X_test = X_test_df.to_numpy()
    y_train = y_train_df.to_numpy()
    y_val = y_val_df.to_numpy()
    y_test = y_test_df.to_numpy()

    # Sample from training set and validation set
    n_train = len(X_train)
    n_train_samples = int(p.split*n_train)
    n_val = len(X_val)
    n_val_samples = int(p.split*n_val)

    ran_train_mask = np.random.randint(n_train_samples, size=n_train_samples)
    X_train_subset = X_train[ran_train_mask,:]
    y_train_subset = y_train[ran_train_mask,:]

    ran_val_mask = np.random.randint(n_val_samples, size=n_val_samples)
    X_val_subset = X_val[ran_val_mask,:]
    y_val_subset = y_val[ran_val_mask,:]

    # ensure all runs have the same number of gradient descent steps
    adjusted_epochs = int(p.epochs/p.split)
    adjusted_patience = int(p.epochs/p.patience)

    model = build(lr = p.learning_rate)

    # Instantiate callbacks. Set histogram_freq to non-zero to see weight histograms in tensorboard
    # to diagnose exploding gradients
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=out_dir / (p.out_name+"_logs"),
        histogram_freq=0)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        patience=adjusted_patience, restore_best_weights=True)
    csv_logger = tf.keras.callbacks.CSVLogger(out_dir / 'training_log.csv', append=True)
    # Run the hyperparameter search
    history = model.fit(X_train_subset, y_train_subset, epochs=adjusted_epochs, validation_data=(X_val_subset, y_val_subset),
        callbacks=[tensorboard_callback, early_stopping_callback, csv_logger], verbose=2, batch_size=16)
    
    # evaluate model

    # make a header for dataframe/csv
    n_k_points=21
    top_cols = []
    bot_cols = []
    for i in range(21, 26):
        for j in range(1, n_k_points+1):
            top_cols.append("band_{}".format(i))
            bot_cols.append("kpoint_{}".format(j))
    col_names = [top_cols, bot_cols]
    
    y_test_predicts = pd.DataFrame(model.predict(X_test), columns=col_names)
    y_test_predicts.to_csv(out_dir/"preds.csv", index=False)

    hist = pd.DataFrame(history.history)
    hist.to_csv(out_dir/"hist.csv", index=True)

    model.save(out_dir / "model")

    eval_dict = {}
    eval_dict["split"] = p.split
    eval_dict["n_train_samples"] = n_train_samples
    eval_dict["n_val_samples"] = n_val_samples

    pred = y_test_predicts.to_numpy().flatten()
    act = y_test_df.to_numpy().flatten()

    for name, func in [("MSE", MSE), ("RMSE", RMSE), ("MPE", MPE), ("MAPE", MAPE)]:
        eval_dict[name]= func(act, pred)
    
    print(eval_dict)
    
    tok = datetime.now()
    
    eval_dict["runtime"] = str(tok-tik)
    with open(out_dir / "metrics.json","w") as f:
        json.dump(eval_dict,f)

    print("\nTotal Runtime: {}".format(str(tok-tik)))
