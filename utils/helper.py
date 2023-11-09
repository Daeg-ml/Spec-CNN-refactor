# -*- coding: utf-8 -*-
"""
Contains functions for managing, shaping, and modifying data or analysis
"""
from typing import List, Tuple, Union
import numpy as np
import pandas as pd


def fnamelsbuilder(fin_path):
    """Build a list of files in directory 'fin_path'.

    Args:
      fin_path: a string providing the path to the folder with the intended files

        In order to avoid unexpected behavior, ensure the fin_path folder only
        contains folders or data files

    Returns:
      A python list of file names in fin_path
    """

    from os import listdir
    from os.path import isfile, join

    # return files
    return [f for f in listdir(fin_path) if isfile(join(fin_path, f))]


def peakscleaning(df) -> pd.DataFrame:
    """Cleaning for peaks data - drop any rows containing NA

    Args:
      df: a dataframe with peaks data

    Returns:
      DataFrame of peaks data with no NA values
    """
    df.dropna(inplace=True)

    # drop relative intensities
    df.drop(columns=[i for i in df.columns.values if "val" in i], inplace=True)
    return df


def dfbuilder(
    fin_path, split_df=True, dev_size=0.2, r_state=1, raw=False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Imports data from all CSV files in 'fname_ls' found at location 'fin_path'
    and returns in one large dataframe or a split of data for training

    Args:
      fin_path: string, path to the folder with the intended files
      split_df: boolean, true causes 'df_builder' to return split data df;
        default True. If True, function will return split data; if False, function
        will return a single DataFrame
      dev_size: float on closed interval (0, 1.0), determines percentage of data
        used for the dev set in the train_test_split. Ignored if 'split_df' is
        False
      r_state: integer, provides random state for the train_test_split. Ignored if
        'split_df' is False
      raw: boolean, True if the input is raw data, false if it has been
        preprocessed (eg. with continuous wavelet transform)

    Returns:
      If split_data=True, Tuple of 4 DataFrames including all rows from files
      named in fname_ls split using the 'split_data()' function.
      If 'split_df' is False, will return one DataFrame of data in fin_path
    """

    # list of file names with data
    fname_ls = fnamelsbuilder(fin_path)

    # create list to hold dataframes
    df_ls = []
    # read in each file
    if raw:
        df_ls = raw_processing(df_ls, fname_ls, fin_path)

    else:
        for i in fname_ls:
            temp_df = pd.read_csv(
                fin_path + i, index_col="og-idx", delim_whitespace=False
            )
            df_ls.append(temp_df)

    # create one large df
    if len(df_ls) > 1:
        df = pd.concat(df_ls)
    else:
        df = df_ls[0]

    if not df[df.isna().any(axis=1)].empty:
        raise ValueError("the dataframe includes NaN values")

    # if peaks data, additional cleaning
    if "Peaks Only" in fin_path:
        df = peakscleaning(df)

    # split data for processing
    if split_df:
        return splitdata(df, dev_size, r_state)

    # split data for processing
    assert isinstance(df, pd.DataFrame)
    return df


def raw_processing(df_ls, fname_ls, fin_path) -> List[pd.DataFrame]:
    """imports and standardizes raw data to one intensity value per wave number
    between wave number 150 and 1100. Designed for use by the dfbuilder method

    Args:
      df_ls: an empty list for holding the imported DataFrames
      fname_ls: list of data file names
      fin_path: the path where the data files can be found

    Returns:
      A python list of DataFrames with standardized raw data files
    """
    for fil in fname_ls:
        temp_df = pd.read_csv(
            fin_path + fil, index_col="og-idx", delim_whitespace=False
        )

        # separate the labels
        temp_labels = temp_df["label"]

        # drop any column names that can't be converted into floats
        drop_cols = []
        for i in temp_df.columns.values:
            try:
                float(i)
            except:
                drop_cols.append(i)
        temp_df.drop(columns=drop_cols, inplace=True)

        # trim to cols to [150,1100]
        trim_range = (150.0, 1100.0)

        temp_df.drop(
            columns=[j for j in temp_df.columns.values if float(j) < trim_range[0]],
            inplace=True,
        )
        temp_df.drop(
            columns=[j for j in temp_df.columns.values if float(j) > trim_range[1]],
            inplace=True,
        )

        # standardize to 1 intensity value per wave number
        std_df = pd.DataFrame()

        for k in range(int(trim_range[0]), int(trim_range[1])):
            std_df[k] = temp_df[
                [j for j in temp_df.columns.values if k <= float(j) < (k + 1)]
            ].min(axis=1)

        # add labels back to DataFrame, append to the df_ls DataFrame list
        std_df["label"] = temp_labels.values
        df_ls.append(std_df)

    return df_ls


def splitdata(
    df: pd.DataFrame, dev_size=0.2, r_state=1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """splits X values from y values and returns tuple of DataFrames from sklearn
    train_test_split

    Args:
      X: DataFrame or ndarray with labels in the last column
      dev_size: float on closed interval (0, 1.0), determines percentage of data
        used for the dev set in the train_test_split
      r_state: integer, provides random state for the train_test_split

    Returns:
      Tuple of 4 DataFrames from 'X' split using sklearn train_test_split
    """
    X = df.copy()
    # separate y from X
    y = X[X.columns[-1]]
    X.drop(X.columns[-1], axis=1, inplace=True)
    # split into train and dev sets
    from sklearn.model_selection import train_test_split

    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, test_size=dev_size, random_state=r_state
    )
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_dev, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_dev, pd.Series)

    return X_train, X_dev, y_train, y_dev
