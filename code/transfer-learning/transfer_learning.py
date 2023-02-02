# Collection of transfer learning scripts copied over from jupyter notebook cells into this file
# loads a NN pretrained using the 2D bands dataset and continues training on a 3d dataset
# Caspar Schwahn, September 2022

# Imports
import csv
import numpy as np
import pandas as pd
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorboard as tb
import tempfile

## Import 3D dataset
train_dir = pathlib.Path("../../training-sets/3d-branching-bands-101-3-3/")

X_train = pd.read_csv(train_dir / "X_train.csv", header=[0]).to_numpy()
X_val = pd.read_csv(train_dir / "X_val.csv", header=[0]).to_numpy()
X_test = pd.read_csv(train_dir / "X_test.csv", header=[0]).to_numpy()
y_odd_test = pd.read_csv(train_dir/"y_odd_test.csv", header = [0,1]).to_numpy()
y_even_test = pd.read_csv(train_dir/"y_even_test.csv", header = [0,1]).to_numpy()
y_odd_val = pd.read_csv(train_dir/"y_odd_val.csv", header = [0,1]).to_numpy()
y_even_val = pd.read_csv(train_dir/"y_even_val.csv", header = [0,1]).to_numpy()
y_odd_train = pd.read_csv(train_dir/"y_odd_train.csv", header = [0,1]).to_numpy()
y_even_train = pd.read_csv(train_dir/"y_even_train.csv", header = [0,1]).to_numpy()


# add_regularization function is a copy of:
# "How to Add Regularization to Keras Pre-trained Models the Right Way". Thalles Silva. Nov 26, 2019
# https://sthalles.github.io/keras-regularizer/
def add_regularization(model, regularizer=tf.keras.regularizers.l2(1e-7), train_layers=10):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
      print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
      return model

    for layer in model.layers[-train_layers:]:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = pathlib.Path(tempfile.gettempdir())/ 'tmp_weights.h5'
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)
    
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
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

def transfer_train(model, run_dict_list, run_name):
    """
    2D Pretrained model is trained according to run_dict_list on 3D dataset
    
    creates directory called run_name to save new model, plots and predictions
    
    :param model:
    :param run_dict_list: list of dicts. Each dict has keys "train_layers", "l2_reg_value",
                            "learning_rate","patience","epochs". The NN is trained with these 
                            hyperparameters for the given number of epochs and then moves on to the
                            next dict in the list
    :param run_name: name of new directory to store model, plots, predictions and metrics

    :return: model, fig, ax. (fig, ax of loss curve plot)
    """

    # Create output directory
    out_dir =  pathlib.Path(run_name)
    if not out_dir.exists():
        out_dir.mkdir()
    else:
        print("Run name already exits")
        return

    history_list = []
    for j, run_dict in enumerate(run_dict_list):
        print(f"Transfer learning part {j}")
        print(run_dict)
        model = add_regularization(model, regularizer=tf.keras.regularizers.l2(run_dict["l2_reg_value"]), train_layers=run_dict["train_layers"])
        
        model.trainable=True # unfreeze model
        for layer in model.layers[:-run_dict["train_layers"]]:
            layer.trainable=False # freeze lower layers

        # print model details
        for i,layer in enumerate(model.layers):
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    print(i,layer.name, "Trainable: ",layer.trainable ," kernel_regularizer: ", layer.kernel_regularizer.l2)
                else:
                    print(i,layer.name, "Trainable: ", layer.trainable ," kernel_regularizer: None")

        # check number of trianable params
        print(model.summary())
        

        # Set up training
        optimizer = keras.optimizers.Nadam(
                    learning_rate=run_dict["learning_rate"])

        model.compile(
                optimizer = optimizer,
                loss = {"even":"mean_squared_error",
                        "odd":"mean_squared_error"},
                metrics = {"even":'mean_absolute_percentage_error',
                 "odd":'mean_absolute_percentage_error'},
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=out_dir / f"{run_name}_part_{j}_logs",
                histogram_freq=0)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                   patience=run_dict["patience"], 
                                                                   restore_best_weights=True, 
                                                                   verbose=1)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=50,
            verbose=1,
            min_delta=1e-7,
            cooldown=1,
            min_lr=1e-7,
        )


        history = model.fit({"design_params":X_train},
                {"even":y_even_train, "odd":y_odd_train},
                epochs=run_dict["epochs"],
                callbacks=[tensorboard_callback,early_stopping_callback, reduce_lr],
                validation_data=({"design_params":X_val},{"even":y_even_val, "odd":y_odd_val}),
                verbose=2,
                batch_size=16,
        )
        history_list.append(history.history)

    model.save(out_dir / f"{run_name}_model")

    # Evaluate
    y_even_pred, y_odd_pred = model.predict(X_test)

    even = MAPE(y_even_test.flatten(), y_even_pred.flatten())
    odd = MAPE(y_odd_test.flatten(), y_odd_pred.flatten())
    combined =MAPE(np.concatenate((y_even_test.flatten(),y_odd_test.flatten())),
                            np.concatenate((y_even_pred.flatten(),y_odd_pred.flatten())))

    print("even", even)
    print("odd", odd)
    print("combined", combined)

    out_dict = {}
    out_dict["metrics"] = {"even":even, "odd":odd, "combined":combined}
    out_dict["config"] = run_dict_list
    
    # merge all histories into one
    out_hist = {}
    for k in history_list[0].keys():
        combined_list = []
        for history in history_list:
            combined_list+= [float(i) for i in history[k]]
        out_hist[str(k)] = combined_list
        
    out_dict["history"] = out_hist
    
    with open(out_dir/"run_info.json","w") as f:
        json.dump(out_dict,f)
    
    # Plot
    fig, ax = plt.subplots(1, figsize=(4,4), dpi=300)
    ax.plot(out_hist["val_even_mean_absolute_percentage_error"], label="val even")
    ax.plot(out_hist["even_mean_absolute_percentage_error"], label="train even")
    ax.plot(out_hist["val_odd_mean_absolute_percentage_error"], label="val odd")
    ax.plot(out_hist["odd_mean_absolute_percentage_error"], label="train odd")
    ax.set_ylim(0,1)
    ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel("Metric")
    fig.savefig(out_dir / f"{run_name}.png",facecolor='white', bbox_inches="tight")

    return model, fig, ax

#### Start a transfer learning run

run_name = "tr-5"

run_dict_list = [{"train_layers":6,
                  "l2_reg_value":2e-7,
                  "learning_rate":1e-6,
                  "patience":100,
                  "epochs":5000},
                 {"train_layers":14,
                  "l2_reg_value":1e-8,
                  "learning_rate":1e-6,
                  "patience":100,
                  "epochs":5000},
                 {"train_layers":24,
                  "l2_reg_value":1e-8,
                  "learning_rate":1e-6,
                  "patience":100,
                  "epochs":5000},
                  {"train_layers":33,
                  "l2_reg_value":1e-8,
                  "learning_rate":1e-7,
                  "patience":100,
                  "epochs":5000}]

# Import pretrained model
import_model_dir = pathlib.Path("../../data/tuning/t-7/model/")
model = keras.models.load_model(import_model_dir)

model, fig, ax = transfer_train(model, run_dict_list, run_name)


### Loads trained N for evaluation

import_model_dir = pathlib.Path("tr-5/tr-5_model/")
model = keras.models.load_model(import_model_dir)


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

y_test_pred_list = model.predict(X_test)
y_test_pred_dict = {name: pred for name, pred in zip(model.output_names, y_test_pred_list)}

out_dir=pathlib.Path("tr-5/")
y_odd_test_pred = pd.DataFrame(y_test_pred_dict["odd"], columns=odd_cols)
y_odd_test_pred.to_csv(out_dir/"odd_preds.csv", index=False)
y_even_test_pred = pd.DataFrame(y_test_pred_dict["even"], columns=even_cols)
y_even_test_pred.to_csv(out_dir/"even_preds.csv", index=False)