import numpy as np

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    B1 = np.random.rand(10, 1) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2
def ReLU(x):
    return np.maximum(x, 0)
def ReLU_deriv(x):
    return x>0
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))
def softmax(x):
    sum_all = sum(np.exp(x))
    return np.exp(x)/sum_all
def one_hot_y(labels):
    hot_y = np.zeros((labels.size, labels.max()+1))
    hot_y[np.arange(labels.size), labels] = 1
    return hot_y.T

def forwardpropagation_sigmoid(A0, W1, B1, W2, B2):
    Z1 = W1.dot(A0) + B1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def forwardpropagation_softmax(A0, W1, B1, W2, B2):
    Z1 = W1.dot(A0) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def backpropagation_sigmoid(labels, A0, W2, Z1, A1, Z2, A2):
    div = 1/labels.size
    dZ2 = (A2 - one_hot_y(labels))*sigmoid_deriv(Z2)
    dB2 = div*np.sum(dZ2,axis=1)
    dW2 = div*dZ2.dot(A1.T)
    dZ1 = W2.T.dot(dZ2)*sigmoid_deriv(Z1)
    dB1 = div*np.sum(dZ1,axis=1)
    dW1 = div*dZ1.dot(A0.T)
    return dW1, dB1, dW2, dB2
def backpropagation_softmax(labels, A0, W2, Z1, A1, Z2, A2):
    div = 1/labels.size
    dZ2 = A2 - one_hot_y(labels)
    dB2 = div*np.sum(dZ2,axis=1)
    dW2 = div*dZ2.dot(A1.T)
    dZ1 = W2.T.dot(dZ2)*ReLU_deriv(Z1)
    dB1 = div*np.sum(dZ1,axis=1)
    dW1 = div*dZ1.dot(A0.T)
    return dW1, dB1, dW2, dB2

def update_params(alpha, W1, B1, W2, B2, dW1, dB1, dW2, dB2):
    return W1 - alpha*dW1, B1 - alpha*dB1[:,np.newaxis], W2-alpha*dW2, B2-alpha*dB2[:,np.newaxis]

def get_predictions(data):
    return np.argmax(data, axis=0)
def get_accuracy(predictions, labels):
    return np.sum(predictions==labels)/labels.size

