# -*- coding: utf-8 -*-

# TODO: update all docstrings
"""
Contains methods that define the CNN model
"""
import yaml

from IPython.display import display
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
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

    def on_epoch_end(self, epoch, logs={}):
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
            optimizer=opt,
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
            self. model,
            self.mod_sel,
            self.fout_path,
            self.X_dev,
            self.y_dev,
            self.id_val,
            threshold=self.threshold,
            fast=self.fast,
            test=False
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


class TestModel:
    """
    Testing object for machine learning models. Can be used to build confusion 
        matrix and generate ROC plots

    Attributes:
        model: keras or scikit-learn model object
        mod_sel: string name of model, used for configuration. Current 
            options: "cnn_raw"
        X_test: ndarray-like object of test set features
        y_test: array-like object of test set labels, must be same lengh as X_test
        id_val: string prepended to output file names
        threshold: float decision threshold for positive identification
        fast: bool supresses output files for quick testing if true
        test: bool identifies output files as test files if true, validation if 
            false
        batch_size: batch size for model prediction
        fout_path: string filepath for output files
        y_pred: array-like object of model predictions for X_test
        y_dec_pred: array-like object of model predictions with decision 
            threshold applied
    """
    def __init__(
        self, 
        model, 
        mod_sel, 
        fout_path, 
        X_test, 
        y_test, 
        id_val, 
        threshold, 
        fast, 
        test
    ):
        self.model = model
        self.mod_sel = mod_sel
        self.X_test = X_test
        self.y_test = y_test
        self.id_val = id_val
        self.threshold = threshold
        self.fast = fast
        self.test = test
        self.batch_size = config[self.mod_sel]["test_batch"]
        self.fout_path = fout_path

        self.y_pred = self.run_pred()
        self.y_dec_pred = self.dec_pred()
        self.test_cnn_model()

    def run_pred(self):
        # todo: docstring
        """
        """

        # predict classes for provided test set
        self.y_pred = self.model.predict(self.X_test, batch_size=self.batch_size)
        return self.y_pred

    def test_cnn_model(self):
        # todo: docstring update
        """
        test a trained model with given parameters, creates a csv of confusion matrix
        at Model Data/CNN Model/ 'id_val'+comfmatout.csv

        Args:
          model: a trained keras model used to predict the categories of the test set
          X_test: a DataFrame or ndarray of sample features sets for testing
          y_test: a DataFrame or Series of sample labels the X_test feature set - must
            be in the same order as the X_test feature set
          id_val: a string used in the file name of the outputs for identification

        Returns:
          None; creates a file at the /Model Data/CNN Model/Raw/ folder
        """

        # report confusion matrix
        confmat = self.build_confmat()

        display(confmat)

        # if fast is not True, save the confusion matrix as either test or validation
        if not self.fast:
            if self.test:
                file_n = self.id_val + "_test"
            else:
                file_n = self.id_val + "_validation"
            # save confusion matrix as csv to drive
            confmatout_path = self.fout_path + file_n

            confmat.to_csv(confmatout_path + r"_confmat.csv")
            # save output weights
            pd.DataFrame(
                data=self.model.predict(self.X_test), index=self.y_test.index.values
            ).to_csv(confmatout_path + "_probs.csv")

    def dec_pred(self):
        # todo: docstring update
        """takes prediction weights and applies a decision threshold to deterime the
        predicted class for each sample

        Args:
          y_pred: an ndarray of prediction weights for a set of samples
          threshold: the determination threshold at which the model makes a prediction

        Returns:
          a 1-d array of class predictions, unknown classes are returned as class 6
        """

        probs_ls = np.amax(self.y_pred, axis=1)
        class_ls = np.argmax(self.y_pred, axis=1)
        pred_lab = np.zeros(len(self.y_pred))
        for i, ele in enumerate(probs_ls):
            if ele > self.threshold:
                pred_lab[i] = class_ls[i]
            else:
                pred_lab[i] = 15
        return pred_lab

    def build_confmat(self):
        # todo: docstring update
        """builds the confusion matrix with labeled axes

        Args:
          y_label: a list of true labels for each sample
          y_pred: a list of predicted labels for each samples
          threshhold: the decision threshhold for the mat_labels

        Returns:
          A DataFrame containing the confusion matrix, the column names are the
          predicted labels while the row indices are the true labels
        """
        print("y_pred=", self.y_pred)
        print("y_dec_pred=", self.y_dec_pred, "\n\n\ny_label=", self.y_test)

        mat_labels = range(max([max(self.y_test), int(max(self.y_dec_pred))]) + 1)

        return pd.DataFrame(
            confusion_matrix(self.y_test, self.y_dec_pred, labels=mat_labels),
            index=[f"true_{i}" for i in mat_labels],
            columns=[f"pred_{i}" for i in mat_labels],
        )
    
    def plot_roc(X_df,i):
      # todo method update to class method
      # todo docstring update
      """plots the receiver operating characteristic curve for the data in X_df with
      true binary labels in the final column

      Args:
        X_df: DataFrame with the target class probability in the column ['i']
        i: target integer label, eg use 0 to get an ROC curve for Quartz

      Returns:
        a tuple of arrays, the arrays of tpr, fpr, and threshold values

      Notes:
        The Reciever Operator Characteristic (ROC) curve is built by plotting x,y at
        various binary discrimination thresholds where x=true positive rate(tpr) and
        y=false positive rate(fpr)

        tpr=True positive/(True Positive + False Negative)
        fpr=False positive/(False Positive + True Negative)
      """
      #get tpr, fpr, and threshold lists
      try:
        from sklearn.metrics import roc_curve
        fpr_p,tpr_p,thresh=roc_curve(X_df['label'],X_df[i],i)
      except:
        return None

      #plot the roc curves
      from matplotlib import pyplot as plt
      plt.plot(fpr_p,tpr_p)
      plt.plot([0,1],[0,1],color='green')
      plt.title('ROC Curve for Class '+str(i))
      plt.show()

      return tpr_p,fpr_p,thresh

    def roc_all(outputs,labels):
        # todo: method update to class method
        # todo: docstring update
        """creates ROC curves for each class in the output of a classifier

        Args:
          outputs: DataFrame or ndarray of output probabilities for each class
          labels: an array or series of true labels for the samples in X

        Returns:
          None
        """
        master_df=pd.DataFrame(outputs)
        roc_d={}
        master_df['label']=labels.values
        for i in range(labels.max()+1):
          if len(master_df.loc[master_df['label']==i])==0:
            continue
          roc_d[i]=plot_roc(master_df,i)

        return pd.DataFrame(roc_d,index=['tpr','fpr','thresh'])
    
    def ouput_classification_report(self):
        

