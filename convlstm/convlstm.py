import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Vars
var_list = ["ndvi", "t2m", "tp"]
var_colors = {"ndvi": "forestgreen", "t2m": "maroon", "tp": "mediumblue"}
var_titles = {"ndvi": "NDVI", "t2m": "Temperature", "tp": "Precipitation"}


def load_data(base_path="normalized_arrays", var_list=var_list):
    # data_arr.shape = (time, channel, x, y)

    print("Loading variables:", var_list)

    data_arrs = []

    for var in var_list:
        # Load array
        arr = np.load(os.path.join(base_path, f"{var}.npy"))
        ### TEMPORARY
        arr = np.transpose(arr, (2, 0, 1))
        # Get value stats
        print(var, "stats:")
        for metric, title in [
            (np.nanmean, "Mean:"),
            (np.nanstd, "Std:"),
            (np.nanmin, "Min:"),
            (np.nanmax, "Max:"),
        ]:
            print(" -", title, metric(arr))
        # Add to data arr list -- expand dims (axis=1) for concatenation
        data_arrs.append(np.expand_dims(arr, -1))

    # Concat on 1st axis (channels)
    data_arr = np.concatenate(
        data_arrs,
        axis=-1,
    )

    print("Data loaded with shape:", data_arr.shape)

    # ntime, nchannels, nx, ny = data_arr.shape
    ntime, nx, ny, nchannels = data_arr.shape

    fig, ax = plt.subplots(figsize=(15, 5))
    for i, var in enumerate(var_list):
        ax.plot(
            # np.nanmean(data_arr[:, i, :, :].reshape(ntime, nx * ny), axis=1),
            np.nanmean(data_arr[:, :, :, i].reshape(ntime, nx * ny), axis=1),
            color=var_colors[var],
            label=var_titles[var],
            alpha=0.75,
        )
    plt.legend()
    plt.title("Normalized Input Data")
    plt.show()

    return data_arr


def split_data(
    inputs_all, outputs_all, train_percent=0.5, val_percent=0.25
):  # , mask_all):

    train_end = int(inputs_all.shape[0] * train_percent)
    val_end = train_end + int(inputs_all.shape[0] * val_percent)

    inputs_train = inputs_all[:train_end]
    outputs_train = outputs_all[:train_end]

    inputs_val = inputs_all[train_end:val_end]
    outputs_val = outputs_all[train_end:val_end]

    inputs_test = inputs_all[val_end:]
    outputs_test = outputs_all[val_end:]

    print("Input train shape:", inputs_train.shape)
    print("Output train shape:", outputs_train.shape)
    print("Input val shape:", inputs_val.shape)
    print("Output val shape:", outputs_val.shape)
    print("Input test shape:", inputs_test.shape)
    print("Output test shape:", outputs_test.shape)

    return (
        inputs_train,
        outputs_train,
        inputs_val,
        outputs_val,
        inputs_test,
        outputs_test,
    )


def root_mean_squared_error(y_true, y_pred):
    nonzero = K.tf.count_nonzero(y_pred)
    return K.switch(
        K.equal(nonzero, 0),
        K.constant(value=0.0),
        K.sqrt(K.sum(K.square(y_pred - y_true)) / tf.cast(nonzero, tf.float32)),
    )


def mean_squared_error_loss(y_true, y_pred):
    # nonzero = K.tf.count_nonzero(y_pred)
    # return K.sum(K.square(y_pred - y_true))/tf.cast(tf.multiply(nonzero, tf.constant(2)), tf.float32)
    return K.sum(K.square(y_pred - y_true)) / tf.cast(tf.constant(2), tf.float32)
