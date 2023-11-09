# -*- coding: utf-8 -*-

"""
Contains the class used for testing ML models
"""
import yaml
from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

with open("utils/config.yml", encoding="utf-8") as file:
    config = yaml.safe_load(file)


class TestModel:
    """
    Testing object for machine learning models. Can be used to build confusion
        matrix and generate ROC plots

    Attributes:
        model: keras or scikit-learn model object
        mod_sel: string name of model, used for configuration. Current
            options: "cnn_raw"
        fout_path: string filepath for output files
        X_test: ndarray-like object of test set features
        y_test: array-like object of test set labels, must be same lengh as X_test
        id_val: string prepended to output file names
        threshold: float decision threshold for positive identification
        fast: bool supresses output files for quick testing if true
        test: bool identifies output files as test files if true, validation if
            false
        batch_size: batch size for model prediction

        y_pred: array-like object of model predictions for X_test
        y_dec_pred: array-like object of model predictions with decision
            threshold applied
    """
    # ignore linter for matrix naming conventions
    # pylint: disable=invalid-name

    # TODO: handle rarely changed options fast, test, and threshold as kwargs
    def __init__(
        self, model, mod_sel, fout_path, X_test, y_test, id_val, threshold, fast, test
    ):
        """
        Args:
            model: keras or scikit-learn model object
            mod_sel: string name of model, used for configuration. Current
                options: "cnn_raw"
            fout_path: string filepath for output files
            X_test: ndarray-like object of test set features
            y_test: array-like object of test set labels, must be same lengh as X_test
            id_val: string prepended to output file names
            threshold: float decision threshold for positive identification
            fast: bool supresses output files for quick testing if true
            test: bool identifies output files as test files if true, validation if
                false
        """
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
        """Produce predictions for all samples in X_test, this is both returned and
            set to self.y_pred

        Args:
            None

        Returns:
            y_pred: array of predictions for each sample
        
        """

        # predict classes for provided test set
        self.y_pred = self.model.predict(
            self.X_test, batch_size=self.batch_size)
        return self.y_pred

    def test_cnn_model(self):
        """
        test a trained model with given parameters, creates a csv of confusion matrix
        at Model Data/CNN Model/ 'id_val'+comfmatout.csv

        Args:
          None

        Returns:
          None. Creates a file at the /Model Data/CNN Model/Raw/ folder
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
            # save confusion matrix as csv to directory
            confmatout_path = self.fout_path + file_n
            confmat.to_csv(confmatout_path + r"_confmat.csv")

            # save output weights
            pd.DataFrame(
                data=self.model.predict(self.X_test), index=self.y_test.index.values
            ).to_csv(confmatout_path + "_probs.csv")

    def dec_pred(self):
        """takes prediction weights and applies a decision threshold to deterime the
        predicted class for each sample

        Args:
          None

        Returns:
          a 1-d array of class predictions, unknown classes are returned as class 15
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
        """builds the confusion matrix with labeled axes

        Args:
            None

        Returns:
            DataFrame containing the confusion matrix, the column names are the
            predicted labels while the row indices are the true labels
        """
        print("y_pred=", self.y_pred)
        print("y_dec_pred=", self.y_dec_pred, "\n\n\ny_label=", self.y_test)

        mat_labels = range(
            max([max(self.y_test), int(max(self.y_dec_pred))]) + 1)

        return pd.DataFrame(
            confusion_matrix(self.y_test, self.y_dec_pred, labels=mat_labels),
            index=[f"true_{i}" for i in mat_labels],
            columns=[f"pred_{i}" for i in mat_labels],
        )

    def plot_roc(self, X_df, i):
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
        # get tpr, fpr, and threshold lists
        try:
            fpr_p, tpr_p, thresh = roc_curve(
                X_df["label"], X_df[i], pos_label=i)
        except:
            return None

        # plot the roc curves

        plt.plot(fpr_p, tpr_p)
        plt.plot([0, 1], [0, 1], color="green")
        plt.title("ROC Curve for Class " + str(i))
        plt.show()

        return tpr_p, fpr_p, thresh

    def roc_all(self, outputs, labels):
        """creates ROC curves for each class in the output of a classifier

        Args:
          outputs: DataFrame or ndarray of output probabilities for each class
          labels: an array or series of true labels for the samples in X

        Returns:
          None
        """
        master_df = pd.DataFrame(outputs)
        roc_d = {}
        master_df["label"] = labels.values
        for i in range(labels.max() + 1):
            if len(master_df.loc[master_df["label"] == i]) == 0:
                continue
            roc_d[i] = self.plot_roc(master_df, i)

        return pd.DataFrame(roc_d, index=["tpr", "fpr", "thresh"])
