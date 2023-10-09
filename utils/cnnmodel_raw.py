# -*- coding: utf-8 -*-

# TODO: rename file to more generic name and update external files to match
"""
Contains the class used to generate the CNN model
"""
import yaml
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import (
    Input,
    Dense,
    Activation,
    BatchNormalization,
    Flatten,
    Conv1D,
    LeakyReLU,
)
from tensorflow.keras.layers import MaxPooling1D, Dropout
from tensorflow.keras.models import Model

import utils.helper as h

with open("utils/config.yml", encoding="utf-8") as file:
    config = yaml.safe_load(file)


class PredictionCallback(tf.keras.callbacks.Callback):
    """
    Callback to output list of predictions of all training and dev data to a
        file after each epoch
    """

    def __init__(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        y_train: pd.Series,
        y_dev: pd.Series,
        train: str,
        val: str,
        fout_path: str,
    ):
        super().__init__()
        self.validation_data = validation_data
        self.train_data = train_data
        self.y_train = y_train
        self.y_dev = y_dev
        self.train = train
        self.val = val
        self.path = fout_path

    def on_epoch_end(self, epoch, logs={}):  # pylint: disable=dangerous-default-value
        """
        Callback method to generate logs on epoch end
        """

        train = self.path + self.train
        self.model: K.models.Model

        pd.DataFrame(
            data=self.model.predict(self.train_data), index=self.y_train.index
        ).to_csv(train, mode="a")

        val = self.path + self.val

        pd.DataFrame(
            data=self.model.predict(self.validation_data), index=self.y_dev.index
        ).to_csv(val, mode="a")


class CnnModel:
    """
    Manages CNN model, on initialization the appropriate configurations are
        selected depending on the mod_sel chosen

    Attributes:
        mod_sel: string identifying which model to build; determines default
            hyperparameters, filepaths, and structure. Options: 'cnn_raw'
        id_val: string prepended to output file names
        fast: bool skips custom callback and saving a history file if true.
            Default true
        lr: float learning rate
        batch_size: int batch size
        drop_rate: float dropout rate
        epochs: int number of epochs
        lrlu_alpha: float alpha rate for leaky relu
        metrics: list[string,...] of metrics to track during training
        threshold: float decision threshold for positive identification
        data_path: string
        fout_path: string
        r_state: int
        dev_size: float between 0.0 and 1.0, determines the proportion of
            the data set to be separated for the dev set. Default 0.2
        X_train: DataFrame or ndarray of training set input features
        y_train: DataFrame or Series of labels for the X_train samples, must be in
            the same order as X_train samples
        X_dev: DataFrame or ndarray of dev set input features
        y_dev: DataFrame or Series of labels for the X_dev samples, must be in the
            same order as the X_dev samples
        model: keras model object. Not instantiated until self.train_cnn_model
            is called
        test: testmodel class object. Not instantiated until self.cnn_model
            is called
    """

    def __init__(
        self,
        mod_sel: str,
        data_path,
        id_val="0_",
        fast=True,
        dev_size=None,
        hyperparameters=None,
    ):
        """
        Args:
            mod_sel: string identifying which model to build; determines default
                hyperparameters, filepaths, and structure. Options: 'cnn_raw'
            data_path: string source path for input data
            id_val: string prepended to output files
            fast: bool skips custom callback and saving a history file if true.
                Default true
            dev_size: float between 0.0 and 1.0, determines the proportion of
                the data set to be separated for the dev set. Default 0.2
            hyperparameters: dict containing hyperparameters. Valid keys
                include lr, batch_size, drop_rate, epochs
        """
        self.mod_sel = mod_sel

        self.id_val = id_val
        self.fast = fast

        hyperparameters = hyperparameters if hyperparameters else {}

        self.lr = hyperparameters.get("lr", config[self.mod_sel]["lr"])
        self.batch_size = hyperparameters.get(
            "batch_size", config[self.mod_sel]["batch_size"]
        )
        self.drop_rate = hyperparameters.get("drop_rate", config[self.mod_sel]["drop"])
        self.epochs = hyperparameters.get("epochs", config[self.mod_sel]["epochs"])

        self.lrlu_alpha = config[self.mod_sel]["lrlu_alpha"]
        self.metrics = [config[self.mod_sel]["metrics"]]
        self.threshold = config[self.mod_sel]["test_threshold"]
        self.dev_size = config[self.mod_sel]["dev_size"]

        self.data_path = data_path if data_path else config[self.mod_sel]["data_path"]
        self.fout_path = config[self.mod_sel]["fout_path"]
        self.r_state = config[self.mod_sel]["r_state"]
        self.dev_size = dev_size if dev_size else config[self.mod_sel]["dev_size"]

        # build dataframes for all data after splitting
        self.X_train, self.X_dev, self.y_train, self.y_dev = h.dfbuilder(
            fin_path=self.data_path,
            dev_size=self.dev_size,
            r_state=self.r_state,
            raw=config[mod_sel]["is_raw"],
        )

        self.model: Model
        self.test: TestModel

    def train_cnn_model(self, hyperparameters=None):
        """
        Trains a CNN model using the predetermined architecture and returns the model

        Args:
            hyperparameters: dict of hyperparameters. Valid keys include: 'lr'
                (learning rate) (default 0.0001), 'batch_size' (batch size)
                (default 100), 'drop_rate' (dropout rate) (default 0.55),
                'epochs' (number of epochs to train) (default 10)

        Returns:
            trained keras model object
        """

        # if no hyperparameters are set, set the defaults, otherwise extract them
        if hyperparameters:
            self.lr = hyperparameters[0]
            self.batch_size = hyperparameters[1]
            self.drop_rate = hyperparameters[2]
            self.epochs = hyperparameters[3]

        # initialize configured parameters for callbacks
        train_out = config[self.mod_sel]["train_log"]
        val_out = config[self.mod_sel]["val_log"]
        monitor = config[self.mod_sel]["monitor"]
        min_delta = config[self.mod_sel]["min_delta"]
        patience = config[self.mod_sel]["patience"]

        # determine the appropriate callbacks, depending on if fast is true or false
        assert isinstance(self.X_train, pd.DataFrame)
        assert isinstance(self.X_dev, pd.DataFrame)
        assert isinstance(self.y_train, pd.Series)
        assert isinstance(self.y_dev, pd.Series)

        if not self.fast:
            callbacks = [
                PredictionCallback(
                    self.X_train,
                    self.X_dev,
                    self.y_train,
                    self.y_dev,
                    self.id_val + train_out,
                    self.id_val + val_out,
                    self.fout_path,
                ),
                K.callbacks.EarlyStopping(
                    monitor=monitor, min_delta=min_delta, patience=patience
                ),
            ]
        else:
            callbacks = [
                K.callbacks.EarlyStopping(
                    monitor=monitor, min_delta=min_delta, patience=patience
                )
            ]

        # call build_cnn and train model, output trained model
        cnn_model = self.build_cnn(self.X_train.shape, self.y_train.max() + 1)

        cnn_model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_dev, self.y_dev),
            callbacks=callbacks,
        )

        return cnn_model

    def layer_cbnap(self, X_in, nfilters, size_c, s_c, size_p, lnum):
        """Defines one set of convolutional layers with a Convolution,
        BatchNormalization, LeakyReLU activation, and MaxPooling1D

        Args:
          X_in: input matrix for the layers
          nfilters: number of filters for the convolutional layer
          size_c: size of the convolutional kernel
          s_c: step size for the convolution layer
          size_p: size for MaxPooling layer
          lnum: layer number used for debugging

        Returns:
          Matrix of output values from this set of layers
        """

        # CONV -> BN -> RELU -> MaxPooling Block applied to X
        X_working = Conv1D(nfilters, size_c, s_c, name="conv" + lnum)(X_in)
        X_working = BatchNormalization(name="bn" + lnum)(X_working)
        X_working = LeakyReLU(
            alpha=config[self.mod_sel]["lrlu_alpha"], name="relu" + lnum
        )(X_working)
        X_working = MaxPooling1D(size_p, name="mpool" + lnum)(X_working)
        return X_working

    def build_cnn(self, X_shape, y_shape):
        """Defines and builds the CNN model with the given inputs

        Args:
          X_shape: the shape of the data for the model
          y_shape: the shape of the labels for the model

        Returns:
          a compiled model as defined by this method
        """

        # Define the input placeholder as a tensor with the shape of the features
        # this data has one-dimensional data with 1 channel
        X_input = Input((X_shape[1], 1))

        # first layer - conv, batch normalization, activation, pooling
        nfilters = config[self.mod_sel]["layer_1"]["nfilters"]
        size_c = config[self.mod_sel]["layer_1"]["conv_size"]
        s_c = config[self.mod_sel]["layer_1"]["conv_step"]
        size_p = config[self.mod_sel]["layer_1"]["pool_size"]
        X = self.layer_cbnap(X_input, nfilters, size_c, s_c, size_p, "1")

        # second layer - conv, batch normalization, activation, pooling
        nfilters = config[self.mod_sel]["layer_2"]["nfilters"]
        size_c = config[self.mod_sel]["layer_2"]["conv_size"]
        s_c = config[self.mod_sel]["layer_2"]["conv_step"]
        size_p = config[self.mod_sel]["layer_2"]["pool_size"]
        X = self.layer_cbnap(X, nfilters, size_c, s_c, size_p, "2")

        # third layer - conv, batch normalization, activation, pooling
        nfilters = config[self.mod_sel]["layer_3"]["nfilters"]
        size_c = config[self.mod_sel]["layer_3"]["conv_size"]
        s_c = config[self.mod_sel]["layer_3"]["conv_step"]
        size_p = config[self.mod_sel]["layer_3"]["pool_size"]
        X = self.layer_cbnap(X, nfilters, size_c, s_c, size_p, "3")

        # flatten for final layers
        X = Flatten()(X)

        # layer 4 - fully connected layer 1 dense,Batch normalization,activation,dropout
        d_units = config[self.mod_sel]["layer_4"]["units"]
        act_4 = config[self.mod_sel]["layer_4"]["activation"]
        X = Dense(d_units, use_bias=False, name="dense4")(X)
        X = BatchNormalization(name="bn4")(X)
        X = Activation(act_4, name=act_4 + "4")(X)
        X = Dropout(self.drop_rate, name="dropout4")(X)

        # layer 5 - fully connected layer 2 dense, batch normalization, softmax output
        X = Dense(y_shape, use_bias=False, name="dense5")(X)
        X = BatchNormalization(name="bn5")(X)
        outputs = Activation("softmax", name="softmax5")(X)

        model = Model(inputs=X_input, outputs=outputs)

        opt = K.optimizers.RMSprop(learning_rate=self.lr)

        model.summary()
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=opt,  # type: ignore
            metrics=config[self.mod_sel]["metrics"],
        )
        return model

    def cnn_model(self):
        """calls methods to build and train a model as well as testing against the
        validation sets

        Args:
          None

        Returns:
          CNN model built with raw data inputs
        """

        # train a cnn model
        self.model = self.train_cnn_model()

        # test cnn model with dev set
        self.test = TestModel(
            self.model,
            self.mod_sel,
            self.fout_path,
            self.X_dev,
            self.y_dev,
            self.id_val,
            threshold=self.threshold,
            fast=self.fast,
            test=False,
        )

        # save model
        if not self.fast:
            self.save_model()

    def save_model(self):
        """saves model data to the given output mout_path

        Args:
          None

        Returns:
          None. Creates a file at the given filepath
        """

        self.model.save(self.fout_path + "cnn.h5")
        print("model saved")
