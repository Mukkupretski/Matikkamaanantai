import pandas as pd
import numpy as np


#Loading data
test_data = pd.read_csv("./data/mnist_test.csv")
print("Test data loaded")
train_data = pd.read_csv("./data/mnist_train.csv")
print("Train data loaded")

#Initializing variables
structure = np.array([28*28, 16, 16, 10])
learning_rate = 1

biases = np.array([np.zeros(i) for i in structure[1:]])
weights = np.array([np.random.randn(structure[i]+1, structure[i]) for i in range(structure.size-1)])

correct_rate = 0

def guess(imageData):
    A_l, Z_l = forwardpropagation(imageData)
    return A_l[-1]

def learn():
    global correct_rate
    global weights
    global biases
    i = 1
    while correct_rate < 0.85:
        delta_W = np.zeros(weights.shape)
        delta_B = np.zeros(biases.shape)
        for j in range(((i-1)*1000)%60000,((i-1)*1000)%60000+1000):
            current_data = train_data.iloc[j]
            A_l, Z_l = forwardpropagation(current_data.iloc[1:])
            delta_W0, delta_B0 = backpropagation(current_data.iloc[0], A_l, Z_l)
            delta_W += delta_W0
            delta_B += delta_B0
        weights -= learning_rate*delta_W
        biases -= learning_rate*delta_B
        if i%10==0:
            calculateCorrectRate()
            print(f"Kierros: {i}")
            print(f"Oppimistaso: {correct_rate*100}%")
        i += 1
    
    print("Oppiminen valmistui. Pidä hauskaa :)!")

def forwardpropagation(imageData):
    A_l = np.array([imageData])
    Z_l = np.array([])
    for i in range(len(structure)-1):
        temp_Z = np.dot(weights[i]*A_l[i])+biases[i]
        Z_l = np.append(Z_l, temp_Z, 0)
        A_l = np.append(A_l, sigmoid(temp_Z))
    return A_l, Z_l

def backpropagation(label, A_l, Z_l):
    targets = np.array([1 if i==label else 0 for i in range(10)])
    delta_a = np.array([2*(A_l[-1][i]-targets[i]) for i in range(structure[-1])])
    delta_W0 = np.array([[[]]])
    delta_B0 = np.array([[]])
    for i in range(len(structure)-1):
        delta_B0_l = np.array([])
        delta_W0_l = np.array([[]])
        Z_layer = Z_l[-i-1]
        A_layer = A_l[-i-1]
        for j in range(len(delta_a)):
            delta_z_C = sigmoid_derivative(Z_layer[j])*delta_a[j]
            delta_B0_l = np.append(delta_B0_l, delta_z_C)
            delta_W0_j = np.array([])
            for k in range(structure[-i-2]):
                delta_W0_j = np.append(delta_W0_j, A_layer[j]*delta_z_C)
            delta_W0_l = np.append(delta_W0_l, delta_W0_j, axis=0)

        delta_W0 = np.insert(delta_W0,0,delta_W0_l,axis=0)
        delta_B0 = np.insert(delta_W0,0,delta_B0_l,axis=0)
        
        new_delta_a = np.array([])            

        for k in range(structure[-i-2]):
            sum = 0
            for j in range(len(delta_a)):
                sum += weights[-i-1][j][k]*sigmoid_derivative(Z_layer[j])*delta_a[j]
            new_delta_a = np.append(new_delta_a, sum)
        delta_a = new_delta_a
        
    return delta_W0, delta_B0
def calculateCorrectRate():
    global correct_rate
    corrects = 0
    for i in range(1000):
        current = test_data.iloc[i]
        label = current.iloc[0]
        imageData = current.iloc[1:]
        if label == np.argmax(guess(imageData)):
            corrects += 1
    correct_rate = corrects/1000

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

learn()