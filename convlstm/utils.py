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


def load_data(base_path="arrays", subset=None, var_list=var_list, mask=False):
    """Load arrays. Arrays are assumed to be 3D with the shape (time, x, y) & are returned as 4D with the shape (time, x, y, channel).

    Args:
        base_path (str, optional): Path to the folder that stores the arrays. Defaults to "arrays".
        subset (list, optional): Min & max bounds to slice the array with, if a subset is desired. Defaults to None (entire array used).
        var_list (list, optional): List of variable names: the names of the array files. Defaults to global var_list.
        mask (bool, optional): True to use a mask (found at f"{base_path}/arrays.npy"). Defaults to False.

    Returns:
        data_arr (np.array): 4D array with arrays concatenated on the last axis (follows channels last format).
    """

    print("Loading variables:", var_list)

    data_arrs = []

    for var in var_list:
        # Load array
        arr = np.load(os.path.join(base_path, f"{var}.npy"))
        if var == "ndvi":
            # Convert NDVI scale from -0.3-1 to 0-1
            arr = (arr + 0.3) / 1.3
        if subset is not None:
            arr = arr[:, subset[0] : subset[1], subset[0] : subset[1]]
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

    if mask:
        # Load mask
        mask_arr = np.load(os.path.join(base_path, "mask.npy"))
        # Apply mask
        data_arr = np.where(mask_arr[None, :, :, None].astype(bool), np.nan, data_arr)

    return data_arr


def normalize_array(arr, mean=None, std=None):
    """Normalize array. Standard form: (arr - mean) / std.

    Args:
        arr (np.array): Array to be normalized.
        mean (float, optional): Mean to normalize array with. Defaults to None (using mean of the provdied array).
        std (float, optional): Std to normalize array with. Defaults to None (using mean of the provdied array).

    Returns:
        arr (np.array): Normalized numpy array.
    """
    if mean is None:
        mean = np.mean(arr)
    if std is None:
        std = np.mean(std)
    return (arr - mean) / std


def normalize_and_split_data(
    inputs_all, outputs_all, train_percent=0.6, val_percent=0.2
):
    """Normalize inputs & split data into training, validation, and testing sets.
    Done to organize input timesteps as tmin-tmax and outputs timesteps as tmin+1-tmax+1.

    Args:
        inputs_all (np.array): 4D array of inputs, timesteps first & channels last.
        outputs_all (np.array): 4D array of outputs, timesteps first & channels last.
        train_percent (float, optional): Percent of timesteps to used for training set. Defaults to 0.6.
        val_percent (float, optional): Percent of timesteps to used for validation set. Defaults to 0.2.

    Returns:
        Tuple of arrays: (inputs_train, outputs_train, inputs_val, outputs_val, inputs_test, outputs_test)
    """

    train_end = int(inputs_all.shape[0] * train_percent)
    val_end = train_end + int(inputs_all.shape[0] * val_percent)

    stats_arr = inputs_all[:train_end]
    stats_arr = np.reshape(
        stats_arr,
        (
            stats_arr.shape[0] * stats_arr.shape[1] * stats_arr.shape[2],
            stats_arr.shape[3],
        ),
    )
    mean = np.nanmean(stats_arr, axis=0)
    std = np.nanstd(stats_arr, axis=0)

    inputs_train = normalize_array(inputs_all[: train_end - 1], mean, std)
    outputs_train = np.expand_dims(outputs_all[1:train_end], [-1])

    inputs_val = normalize_array(inputs_all[train_end : val_end - 1], mean, std)
    outputs_val = np.expand_dims(outputs_all[train_end + 1 : val_end], [-1])

    inputs_test = normalize_array(inputs_all[val_end:-1], mean, std)
    outputs_test = np.expand_dims(outputs_all[val_end + 1 :], [-1])

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


def build_model(inputs_train, mask=None):

    input_shape = (None, *list(inputs_train.shape)[1:])

    input = keras.layers.Input(shape=input_shape, dtype="float32", name="input")

    x = keras.layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        # dropout=0.25,
        recurrent_dropout=0.25,
        data_format="channels_last",
        activation="tanh",
    )(input)

    x = keras.layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        # dropout=0.25,
        recurrent_dropout=0.25,
        data_format="channels_last",
        activation="tanh",
    )(x)

    x = keras.layers.Conv3D(
        filters=1,
        kernel_size=(3, 3, 3),
        activation="sigmoid",
        padding="same",
        kernel_regularizer=keras.regularizers.l2(0.01),
    )(x)

    if mask is not None:
        output = keras.layers.Multiply(name="apply_mask_output")([output, mask])

    model = keras.Model(inputs=input, outputs=x)

    return model
