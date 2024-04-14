import tensorflow as tf
from keras.layers import Dense
from keras import Input
from keras.models import Sequential, load_model
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

#Importing data
data = pd.read_csv("./data/mnist_test.csv")
print("Csv read")

X_test = data.iloc[:1000, 1:].values.astype("float32")/255
Y_test = data.iloc[:1000, 0]
X_train = data.iloc[1000:, 1:].values.astype("float32")/255
Y_train = data.iloc[1000:, 0]

#Building model
model = Sequential()
def learn():
    model.add(Input(shape=(784,)))
    model.add(Dense(units=16, activation="relu"))
    model.add(Dense(units=10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")

    model.fit(X_train, Y_train, epochs=100, batch_size=64)

    results = np.argmax(model.predict(X_test),axis=1)
    print(f"Tarkkuus: {accuracy_score(Y_test, results)}")
    model.save_weights('./tfmodels/mnist.weights.h5')
def guess(imageData):
    return model.predict((imageData[:, np.newaxis].T)).flatten()