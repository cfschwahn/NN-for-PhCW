"""
Trains and evaluates branching NNs on cluster
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
from functools import partial


def build_branching_dropout(hidden_layers=5, units=500, branching_layers=5, branching_units=350, leaky_alpha=0.3,
        optimizer_type="Nadam", learning_rate=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-07, 
        dropout=True, dropout_rate=0.05, l2_reg=True, l2_value=1e-5):
    """builds and compiles branching model
    """

    # fixed hyperparameters
    output_shape = 303 # for 101 k points and 3 odd and 3 even bands. Each is flattened from shape (n_bands, n_kpoints)
    input_shape = (7,) # 7D design space 
    kernel_initializer = "he_normal" # recommended kernel initialisation for ReLU and variants ~ from Geron HOML

    # create a default dense layer using l2 regularisation if wanted
    if l2_reg:
        wrapped_dense = partial(keras.layers.Dense,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=keras.regularizers.l2(l2_value)
        )
    else:
        wrapped_dense = partial(keras.layers.Dense,
            kernel_initializer=kernel_initializer,
        )

    # Create and link layers
    inputs = keras.Input(shape=input_shape, name="design_params")
    x = wrapped_dense(units)(inputs)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    if dropout:
        x = layers.Dropout(dropout_rate)(x)
    for i in range(hidden_layers-1):
        x = wrapped_dense(units)(x)
        x = layers.LeakyReLU(alpha=leaky_alpha)(x)
        if dropout:
            x = layers.Dropout(dropout_rate)(x)

    even_x = wrapped_dense(branching_units)(x)
    even_x = layers.LeakyReLU(alpha=leaky_alpha)(even_x)

    odd_x = wrapped_dense(branching_units)(x)
    odd_x = layers.LeakyReLU(alpha=leaky_alpha)(odd_x)
    
    if dropout:
        even_x = layers.Dropout(dropout_rate)(even_x)
        odd_x = layers.Dropout(dropout_rate)(odd_x)
        
    for i in range(branching_layers-1):
        even_x = wrapped_dense(branching_units)(even_x)
        even_x = layers.LeakyReLU(alpha=leaky_alpha)(even_x)
        odd_x = wrapped_dense(branching_units)(odd_x)
        odd_x = layers.LeakyReLU(alpha=leaky_alpha)(odd_x)
        if dropout and i != branching_layers-2:
            even_x = layers.Dropout(dropout_rate)(even_x)
            odd_x = layers.Dropout(dropout_rate)(odd_x)
    even_outputs = wrapped_dense(output_shape, name="even")(even_x)
    odd_outputs = wrapped_dense(output_shape, name ="odd")(odd_x)
    
    # build model   
    model = keras.Model(inputs=inputs, outputs=[even_outputs,odd_outputs], name="odd-even-branching")
    
    # add optimizer
    if optimizer_type == "Nadam":
        optimizer = keras.optimizers.Nadam(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, name="Nadam",
        )
    else:
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, name="Adam",
        )
    
    # Compile model. Track mean absolute percentage error
    model.compile(
        optimizer = optimizer,
        loss = {"even":"mean_squared_error",
                "odd":"mean_squared_error"},
        metrics = {"even":'mean_absolute_percentage_error',
         "odd":'mean_absolute_percentage_error'},
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

def add_bool_arg(parser, name, default=False, help=''):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=name, action='store_false', help=help)
    parser.set_defaults(**{name:default})

def make_parser():
    """
    To parse calls from slurm jobs. Does not check for missing values, so all need to be supplied 
    (unless related to toggled off dropout, l2 regulisation or reduce_lr)
    """
    parser = argparse.ArgumentParser()
    
    ## I/O
    parser.add_argument('--train_dir', type=pathlib.Path,
        help='Path to directory containing datasets for training')
    parser.add_argument('--out_dir', type=pathlib.Path,
        help='Directory in which outputs will be created')
    parser.add_argument('--out_name', type=str,
        help='Name of output folder to be created')
    
    ## Architecture
    parser.add_argument('--units', type=int,
        help='Width of neural network layers')
    parser.add_argument('--hidden_layers', type=int,
        help='Depth of neural network - number of hidden layers')
    parser.add_argument('--branching_units', type=int,
        help='Width of neural network layers in odd/even branches')
    parser.add_argument('--branching_layers', type=int,
        help='Depth of neural network - number of hidden layers in branches')
    parser.add_argument('--leaky_alpha', type=float,
        help='LeakyRelU alpha value')

    ## Optimizer
    parser.add_argument('--optimizer_type', type=str,
        help="either Adam or Nadam")
    parser.add_argument('--learning_rate', type=float,
        help='Learning rate')
    parser.add_argument('--beta_1', type=float,
        help ="Adam/Nadam optimizer beta_1")
    parser.add_argument('--beta_2', type=float,
        help ="Adam/Nadam optimizer beta_2")
    parser.add_argument('--epsilon', type=float,
        help ="Adam/Nadam optimizer epsilon")

    ## Training
    parser.add_argument('--epochs', type=int,
        help='Maximum number of epochs to train for')
    parser.add_argument('--patience', type=int,
        help='Patience of early stopping callback')
    parser.add_argument('--batch_size', type=int,
        help='Batch size to use for training')

    ## Regularisation
    # Dropout
    add_bool_arg(parser, 'dropout', default="False",
        help="Add Dropout layers after all but the final hidden layer")
    parser.add_argument('--dropout_rate', type=float,
        help='Dropout rate between 0 and 1')
    # L2 
    add_bool_arg(parser, 'l2_reg', default="False",
        help="Toggle l2 regularisation")
    parser.add_argument('--l2_value', type=float,
        help="l2 regularisation value")

    ## Reduce learning rate on plateau callback
    add_bool_arg(parser, 'reduce_lr', default="True",
        help="Toggl use reduce learning rate on plateau callback")
    parser.add_argument('--reduce_factor', type=float, 
        help="Factor to reduce learning rate by on plateau")
    parser.add_argument('--reduce_patience', type=int, 
        help="Patience (in epochs) for reduce learning rate by on plateau")

    return parser


if __name__ == "__main__":

    tik = datetime.now()
    
    # Parse command line arguments
    parser = make_parser()
    p = parser.parse_args()

    # Create output directory in the given directory
    out_dir = p.out_dir / p.out_name
    if not out_dir.exists():
        out_dir.mkdir()

    ## Data preparation
    # Import datasets
    X_train = pd.read_csv(p.train_dir / "X_train.csv", header=[0])
    X_val = pd.read_csv(p.train_dir / "X_val.csv", header=[0])
    X_test = pd.read_csv(p.train_dir / "X_test.csv", header=[0])
    y_odd_train = pd.read_csv(p.train_dir / "y_odd_train.csv", header=[0,1])
    y_odd_val = pd.read_csv(p.train_dir / "y_odd_val.csv", header=[0,1])
    y_odd_test = pd.read_csv(p.train_dir / "y_odd_test.csv", header=[0,1])
    y_even_train = pd.read_csv(p.train_dir / "y_even_train.csv", header=[0,1])
    y_even_val = pd.read_csv(p.train_dir / "y_even_val.csv", header=[0,1])
    y_even_test = pd.read_csv(p.train_dir / "y_even_test.csv", header=[0,1])

    # Convert to numpy arrays
    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    X_test = X_test.to_numpy()
    y_odd_train = y_odd_train.to_numpy()
    y_odd_val = y_odd_val.to_numpy()
    y_odd_test = y_odd_test.to_numpy()
    y_even_train = y_even_train.to_numpy()
    y_even_val = y_even_val.to_numpy()
    y_even_test = y_even_test.to_numpy()
    
    # Instantiate model
    model = build_branching_dropout(
        hidden_layers=p.hidden_layers, units=p.units,
        branching_layers=p.branching_layers, branching_units=p.branching_units,
        leaky_alpha=p.leaky_alpha,
        optimizer_type=p.optimizer_type, learning_rate=p.learning_rate, beta_1=p.beta_1,
        beta_2=p.beta_2, epsilon=p.epsilon, 
        dropout=p.dropout, dropout_rate=p.dropout_rate, l2_reg=p.l2_reg, l2_value=p.l2_value,
    )
    
    model.summary()
    
    # Instantiate callbacks. Set histogram_freq to non-zero to see weight histograms in tensorboard
    # to diagnose exploding gradients
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=out_dir / (p.out_name+"_logs"),
        histogram_freq=0)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
        patience=p.patience,
        restore_best_weights=True,
        verbose=1,
    )
    csv_logger = tf.keras.callbacks.CSVLogger(out_dir / 'training_log.csv', append=True)
   
    callback_list = [tensorboard_callback, early_stopping_callback, csv_logger]
   
    if p.reduce_lr:
        default_min_delta=1e-8
        default_cooldown=1
        default_min_lr=1e-6
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=p.reduce_factor,
            patience=p.reduce_patience,
            verbose=1,
            min_delta=default_min_delta,
            cooldown=default_cooldown,
            min_lr=default_min_lr,
        )
        callback_list.append(reduce_lr)
 
    # Run the hyperparameter search
    history = model.fit({"design_params":X_train}, 
        {"even":y_even_train, "odd":y_odd_train}, 
        epochs=p.epochs, 
        validation_data=({"design_params":X_val},{"even":y_even_val, "odd":y_odd_val}),
        callbacks=callback_list,
        verbose=2,
        batch_size=p.batch_size,
    )
    
    ## Evaluate model

    # make a header for dataframe/csv
    n_k_points=101
    even_band_nums = [10,11,12]
    odd_band_nums = [11,12,13]
    
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
    
    default_pred_batch_size=1024
    y_test_pred_list = model.predict(X_test, batch_size=default_pred_batch_size)
    y_test_pred_dict = {name: pred for name, pred in zip(model.output_names, y_test_pred_list)}
    
    y_odd_test_pred = pd.DataFrame(y_test_pred_dict["odd"], columns=odd_cols)
    y_odd_test_pred.to_csv(out_dir/"odd_preds.csv", index=False)
    y_even_test_pred = pd.DataFrame(y_test_pred_dict["even"], columns=even_cols)
    y_even_test_pred.to_csv(out_dir/"even_preds.csv", index=False)
    
    y_odd_test_pred = y_odd_test_pred.to_numpy()    
    y_even_test_pred = y_even_test_pred.to_numpy()

    pd.DataFrame(history.history).to_csv(out_dir/"history.csv", index=True)

    model.save(out_dir / "model")

    run_dict = {
        "hidden_layers":p.hidden_layers, "units":p.units,
        "branching_layers":p.branching_layers, "branching_units":p.branching_units,
        "leaky_alpha":p.leaky_alpha,
        "optimizer_type":p.optimizer_type, "learning_rate":p.learning_rate, "beta_1":p.beta_1,
        "beta_2":p.beta_2, "epsilon":p.epsilon, 
        "dropout":p.dropout, "dropout_rate":p.dropout_rate, "l2_reg":p.l2_reg, "l2_value":p.l2_value,
        "epochs":p.epochs, "patience":p.patience, "batch_size":p.batch_size,
        "reduce_lr":p.reduce_lr, "reduce_factor":p.reduce_factor, "reduce_patience":p.reduce_patience,
    }

    print(run_dict)

    metrics_dict = {}
    for name, func in [("MSE", MSE), ("RMSE", RMSE), ("MPE", MPE), ("MAPE", MAPE)]:
        even = func(y_even_test.flatten(), y_even_test_pred.flatten())
        odd = func(y_odd_test.flatten(), y_odd_test_pred.flatten())
        combined = func(np.concatenate((y_even_test.flatten(),y_odd_test.flatten())),
                        np.concatenate((y_even_test_pred.flatten(),y_odd_test_pred.flatten())))
        metrics_dict[name]= {"even":even, "odd":odd, "combined":combined}
    
    print(metrics_dict)

    tok = datetime.now()
    
    run_dict["runtime"] = str(tok-tik)
    run_dict["metrics"] = metrics_dict
    with open(out_dir / "run_info.json","w") as f:
        json.dump(run_dict,f)

    print("\nTotal Runtime: {}".format(str(tok-tik)))
