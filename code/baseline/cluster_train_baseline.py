"""
Trains and evaluates NNs on cluster
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

def build(learning_rate=2e-5, hidden_layers=9, units=512, leaky_alpha=0.3, clipnorm=True, clipnorm_value=1.0):
    """builds and compiles model
    """
    # fixed hyperparameters
    kernel_initializer = "he_normal" # recommended kernel initialisation for ReLU and variants
    output_shape = 505 # for 101 k points and 5 bands
    input_shape = (7,) # 7D design space

    # Create and link layers
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(units, kernel_initializer=kernel_initializer)(inputs)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    for i in range(hidden_layers):
        x = layers.Dense(units, kernel_initializer=kernel_initializer)(x)
        x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    outputs = layers.Dense(output_shape)(x)

    # Create model   
    model = keras.Model(inputs=inputs, outputs=outputs, name="PhCW-regressor")
            
    if clipnorm:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm_value)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    # Compile model
    model.compile(
        optimizer = optimizer,
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
    parser.add_argument('--units', type=int,
        help='Width of neural network layers')
    parser.add_argument('--hidden_layers', type=int,
        help='Depth of neural network - number of hidden layers')
    parser.add_argument('--leaky_alpha', type=float,
        help='LeakyRelU alpha value')
    parser.add_argument('--clipnorm', type=float,
        help='clipnorm value (optional)')
    p = parser.parse_args()

    if p.clipnorm is None:
        clipnorm = False
    else:
        clipnorm = True 
        clipnorm_value = p.clipnorm

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

    if clipnorm:

        model = build(learning_rate = p.learning_rate, units = p.units, hidden_layers = p.hidden_layers, leaky_alpha = p.leaky_alpha, clipnorm = clipnorm, clipnorm_value = clipnorm_value)
    else: 
        model = build(learning_rate = p.learning_rate, units = p.units, hidden_layers = p.hidden_layers, leaky_alpha = p.leaky_alpha)
    
    model.summary()
    
    # Instantiate callbacks. Set histogram_freq to non-zero to see weight histograms in tensorboard
    # to diagnose exploding gradients
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=out_dir / (p.out_name+"_logs"),
        histogram_freq=0)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',        patience=p.patience, restore_best_weights=True)
    csv_logger = tf.keras.callbacks.CSVLogger(out_dir / 'training_log.csv', append=True)
    # Run the hyperparameter search
    history = model.fit(X_train, y_train, epochs=p.epochs, validation_data=(X_val, y_val),
        callbacks=[tensorboard_callback, early_stopping_callback, csv_logger], verbose=2, batch_size=16)
    
    # evaluate model

    # make a header for dataframe/csv
    n_k_points=101
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

    eval_dict = {"epochs":p.epochs, "patience":p.patience, "learning_rate": p.learning_rate,
        "units":p.units, "hidden_layers":p.hidden_layers, "leaky_alpha":p.leaky_alpha,
        "clipnorm": clipnorm, "clipnorm_value": clipnorm_value}

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
