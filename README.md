# House-Price-Estimation

## Abstract
Predicting house prices from combination of image, numerical and categorical data. Images of interiors of house are taken as image data. Number of Bedrooms, Number of bathrooms and House Area are taken as numerical data and Zip code.

## Contents
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
