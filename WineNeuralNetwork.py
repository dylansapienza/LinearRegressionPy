#imports to pass in data from external file
import pandas as pd
from google.colab import files

#uploading the external file and processing
uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

uploaded

#cleaning the dataframe
import io

df = pd.read_csv(io.StringIO(uploaded['winequality-red.csv'].decode('utf-8')), sep=";")

df.head()

#import neural net modules
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.utils import shuffle

predict = "quality"

x = np.array(df.drop([predict], 1))
y = np.array(df[predict])
min_max_scaler = preprocessing.MinMaxScaler()

print(x)

x_scaled = min_max_scaler.fit_transform(x)

print(x_scaled)


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_scaled, y, test_size = 0.2)

model = keras.Sequential([
      keras.layers.Dense(64, activation='relu', input_shape=[11]),
      keras.layers.Dense(100, activation="relu"),
      keras.layers.Dense(11, activation="softmax")
])


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)

prediction = model.predict(x_test)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_scaled), y, test_size = 0.2)
