from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import cv2
import os


def load_house_attributes(inputPath):
    # initialize the list of column in the csv file and then load it using Pandas
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)

    #determine unique zip codes and number of data point with each zip code
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()

    # loop over each of theunique zip code and their corrosponding count
    for (zipcode, count) in zip(zipcodes, counts):
        # the zip code counts for our hoising dataset is extremely nbalanced
        # some only  having 1 or 2 houses per zip codes
        # remove houses with less than 25 houses in zip
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)

    #return the df
    return df


def process_house_attributes(df, train, test):
    # initialize the column names of continuous data
    continuous = ["bedrooms", "bathrooms", "area"]

    # permorm min max scaling
    # range [0,1]

    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContituous = cs.fit_transform(test[continuous])

    #one hot encoding the zip categorical data
    zipBinarizer = LabelBinarizer().fit(df["zipcode"])
    trainCategorical = zipBinarizer.transform(train["zipcode"])
    testCategorical = zipBinarizer.transform(test["zipcode"])

    #construct our training and testing data points by concatenating the categorical valuse with  our continuous values
    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical, testContituous])

    return (trainX, testX)
