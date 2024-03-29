import pandas as pd
import numpy as np


#Loading data
test_data = pd.read_csv("./data/mnist_test.csv")
print("Test data loaded")
train_data = pd.read_csv("./data/mnist_train.csv")
print("Train data loaded")

#Initializing variables
structure = np.array([28*28, 16, 16, 10])

biases = np.array([np.zeros(i) for i in structure[1:]])
weights = np.array([np.random.randn(structure[i], structure[i+1]) for i in range(structure.size-1)])

A_l = np.array([np.zeros(i) for i in structure])
Z_l = np.array([np.zeros(i) for i in structure[1:]])

learning_rate = 0

def guess(imageData):
    pass

def learn():
    i = 1
    while learning_rate < 0.85:
        forwardpropagation()
        backpropagation()
        if i%10==0:
            calculateLearningRate()
            print(f"Kierros: {i}")
            print(f"Oppimistaso: {learning_rate*100}%")
        i += 1
    
    print("Oppiminen valmistui. Pidä hauskaa :)!")

def forwardpropagation():
    pass
def backpropagation():
    pass
def calculateLearningRate():
    pass

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

learn()