import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import ConvLSTM2D, Conv2D

import matplotlib.pyplot as plt

import convlstm

if __name__ == "__main__":

    satellite_images = convlstm.loadData()
    (
        inputs_train,
        outputs_train,
        inputs_val,
        outputs_val,
        inputs_test,
        outputs_test,
    ) = convlstm.prepare_and_split_data(convlstm.TESTING_EXAMPLES, satellite_images)

    print(inputs_train.shape)
    print(outputs_train.shape)
    print(inputs_test.shape)
    print(outputs_test.shape)

    mc = keras.callbacks.ModelCheckpoint(
        "modelsPerEpoch/weights{epoch:06d}.hdf5", save_weights_only=False, period=1
    )

    decay_learner = convlstm.ValidationLearningRateScheduler()

    input = keras.layers.Input(
        shape=(None, *list(inputs_train.shape)[1:]), dtype="float32", name="input"
    )

    # mask = keras.layers.Input(shape=(1, 100, 100), dtype="float32", name="mask")

    hidden = ConvLSTM2D(
        filters=16,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=False,
        data_format="channels_first",
    )(input)

    output = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        padding="same",
        activation="sigmoid",
        kernel_initializer="glorot_uniform",
        data_format="channels_first",
        name="output",
    )(hidden)

    # output_with_mask = keras.layers.Multiply()([output, mask])

    sgd = keras.optimizers.SGD(lr=0.002, momentum=0.0, decay=0.0, nesterov=False)

    # model = keras.Model(inputs=[main_input, mask], outputs=output_with_mask)

    model = keras.Model(inputs=input, outputs=output)

    model.compile(
        optimizer=sgd,
        loss="mean_squared_error",
        metrics=[keras.metrics.mse, convlstm.root_mean_squared_error],
    )

    training_data = model.fit(
        inputs_train,
        outputs_train,
        epochs=20,
        batch_size=1,
        validation_data=(inputs_val, outputs_val),
        # verbose=1,
        callbacks=[mc, decay_learner],
        shuffle=True,
    )

    score = model.evaluate(inputs_test, outputs_test, batch_size=1)
    predictions = model.predict(inputs_test)

    np.save("test_arrays", predictions)
    np.save("training_loss", training_data.history["root_mean_squared_error"])
    np.save("validation_loss", training_data.history["val_root_mean_squared_error"])

    print(score)

    plt.plot(training_data.history["root_mean_squared_error"])
    plt.plot(training_data.history["val_root_mean_squared_error"])

    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validate"], loc="upper left")
    plt.show()
