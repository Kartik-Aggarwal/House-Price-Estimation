# House-Price-Estimation

## Abstract
Predicting house prices from combination of image, numerical and categorical data. Images of interiors of house are taken as image data. Number of Bedrooms, Number of bathrooms and House Area are taken as numerical data and Zip code.

## Contents
This repository contains four folders:
* **House-dataset**\
This folder contains the dataset used for training the model
    * Images of **n**th houses are named as n_bathroom, n_bedroom, n_frontal, n_kicthen
    * A text file contains numerical and categorical data of all houses
* **cnn_regression**\
This folder contains the code where only image data is used for training and cnn is used
* **mlp_regression**\
This folder contains code where training is done on mlp using numerical and categorical data only 
* **combine**\
This is the **main** folder where both types of input data are combined using keras multi input and mixed input regression.


## Explanation
* The number of houses per zipcode is counted. Since some zip codes contain very few houses, all the houses having zip codes that have less than 25 houses under them are removed.
* Then images of these selected houses are loaded. Each image is resized to 32x32x3 pixels.
* Images of one house are joined together in a 64x64x3 pixel image such that bathroom, bedroom, frontal and kitchen image is placed at top-left, top-right, bottom-left and bottom-right side respectively.
* Then MinMaxScaling is applied to continuous attributes (no. of bedrooms, no. of bathrooms, area) and categorical attribute, i.e. zip code is one-hot encoded.
* These processed data are separately fed into mlp and cnn. The number of output nodes of both the architecture is kept the same.
* Then both the output nodes are concatenated and passed through one more dense layer.
* output node has one neuron with a linear activation function
