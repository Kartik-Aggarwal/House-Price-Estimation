# python cnn_regression.py --dataset "Houses-dataset/Houses Dataset"

# import the necessary packages
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tools import datasets
from tools import models
import numpy as np
import argparse
import locale
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")
args = vars(ap.parse_args())

#construct the path to the input .txt file that contains information on each houses in the dataset and then load the dataset
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(inputPath)

#load the images and scale the pixel intensities
print("[INFO] loading house images...")
images = datasets.load_house_images(df, args["dataset"])
images = images/255.0

#train test split
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split


maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

#create and conpile CNN
# mean absolute errro as loss used

model = models.create_cnn(64, 64, 3, regress = True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

#train
print("[INFO] training model...")
model.fit(x = trainImagesX, y = trainY, validation_data = (testImagesX, testY), epochs=200, batch_size=8)

#make predictions on testing data
print("[INFO] predicting house prices...")
preds = model.predict(testImagesX)

# compute the diffecence between the predicted house prices and the actual house prices
# then compute the percentage difference and
# the absolute percentage difference

diff = preds.flatten() - testY
percentDiff = (diff/testY) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))