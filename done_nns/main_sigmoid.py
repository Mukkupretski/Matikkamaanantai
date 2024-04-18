import numpy as np
import pandas as pd
from done_nns.funcs2 import *
import random

data = np.array(pd.read_csv("./data/mnist_train.csv"))
print("Data loaded")

#Rows: a lot, columns: 785
np.random.shuffle(data)

#Rows: 785, columns: 1000 (each colums is a sample)
test_data = data[:1000].T
test_labels = test_data[0]
test_imgs = test_data[1:]/255

train_data = data[1000:].T
train_labels = train_data[0]
train_imgs = train_data[1:]/255

W1 = None
W2 = None
B1 = None
B2 = None

def learn():
    global W1
    global W2
    global B1
    global B2
    W1, B1, W2, B2 = start_learning(train_labels, train_imgs, 0.1, 500)

def start_learning(labels, imgdata, alpha, iterations):
    W1, B1, W2, B2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardpropagation_sigmoid(imgdata, W1,B1,W2,B2)
        dW1, dB1, dW2, dB2 = backpropagation_sigmoid(labels, imgdata, W2, Z1,A1,Z2,A2)
        W1, B1, W2, B2 = update_params(alpha, W1, B1, W2, B2, dW1, dB1, dW2, dB2)
        if i%10==0:
            print(f"Iterations: {i}")
            predictions = get_predictions(A2)
            print(f"Acc: {round(get_accuracy(predictions, labels)*100,2)}%")

    return W1, B1, W2, B2

def guess(imageData):
    _, _, _, A2 = forwardpropagation_sigmoid(imageData[:, np.newaxis], W1, B1, W2, B2)
    return A2.flatten()
def guessRandom():
    imageData = test_imgs[:, random.randint(0, len(test_imgs)-1)]
    return imageData, guess(imageData)