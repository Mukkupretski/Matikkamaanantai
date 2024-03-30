import pandas as pd
import numpy as np


#Loading data
test_data = pd.read_csv("./data/mnist_test.csv")
print("Test data loaded")
train_data = pd.read_csv("./data/mnist_train.csv")
print("Train data loaded")

#Initializing variables
structure = [28*28, 16, 16, 10]

learning_rate = 0.01

biases = [np.zeros(i) for i in structure[1:]]
weights = [np.random.randn(structure[i+1], structure[i]) for i in range(len(structure)-1)]

correct_rate = 0

def guess(imageData):
    A_l, Z_l = forwardpropagation(imageData)
    return A_l[-1]

def learn():
    global correct_rate
    global weights
    global biases
    i = 1
    while correct_rate < 0.15:
        delta_W = [np.zeros(w.shape) for w in weights]
        delta_B = [np.zeros(b.shape) for b in biases]
        for j in range(((i-1)*32)%60000,((i-1)*32)%60000+32):
            current_data = train_data.iloc[j]
            A_l, Z_l = forwardpropagation(np.array(current_data.iloc[1:])/255)
            delta_W0, delta_B0 = backpropagation(current_data.iloc[0], A_l, Z_l)
            delta_W = [delta_W[i]+delta_W0[i] for i in range(len(delta_W0))]
            delta_B = [delta_B[i]+delta_B0[i] for i in range(len(delta_B0))]
        weights =  [weights[i]-learning_rate*delta_W[i]/32 for i in range(len(delta_W))]
        biases =  [biases[i]-learning_rate*delta_B[i]/32 for i in range(len(delta_B))]
        if i%10==0 :
            correct_rate = calculateCorrectRate()
            print(f"Kierros: {i}")
            print(f"Oppimistaso: {correct_rate*100}%")
        i += 1
    
    print("Oppiminen valmistui. Pidä hauskaa :)!")

def forwardpropagation(imageData):
    A_l = [imageData]
    Z_l = []
    for i in range(len(structure)-1):
        temp_Z = np.dot(weights[i],A_l[i])+biases[i]
        Z_l.append(temp_Z)
        A_l.append(sigmoid(temp_Z))
    return A_l, Z_l

def backpropagation(label, A_l, Z_l ):
    targets = np.array([1 if i==label else 0 for i in range(structure[-1])])
    delta_a = np.array([2*(A_l[-1][i]-targets[i]) for i in range(structure[-1])])
    delta_W0 = [np.zeros(w.shape) for w in weights]
    delta_B0 = [np.zeros(b.shape) for b in biases]
    for i in range(len(structure)-1):
        current_pos = -1-i
        delta_W0_l = np.zeros(delta_W0[current_pos].shape)
        delta_B0_l = np.zeros(delta_B0[current_pos].shape)
        Z_layer = Z_l[current_pos]
        A_layer = A_l[current_pos]
        A_prevlayer = A_l[current_pos-1]
        for j in range(len(A_layer)):
            delta_z_C = sigmoid_derivative(Z_layer[j])*delta_a[j]
            delta_B0_l[j] = delta_z_C
            delta_W0_j = np.zeros(delta_W0_l[j].shape)
            for k in range(len(A_prevlayer)):
                delta_W0_j[k] = A_prevlayer[k]*delta_z_C
            delta_W0_l[j] = delta_W0_j

        delta_W0[current_pos] = delta_W0_l
        delta_B0[current_pos] = delta_B0_l
        
        new_delta_a = np.zeros(structure[current_pos-1])          

        for k in range(structure[current_pos-1]):
            sum = 0
            for j in range(len(delta_a)):
                sum += weights[-i-1][j][k]*sigmoid_derivative(Z_layer[j])*delta_a[j]
            new_delta_a[k] = sum
        delta_a = new_delta_a
        
    return delta_W0, delta_B0
def calculateCorrectRate():
    corrects = 0
    for i in range(1000):
        current = test_data.iloc[i]
        label = current.iloc[0]
        imageData = np.array(current.iloc[1:])/255
        if label == np.argmax(guess(imageData)):
            corrects += 1
    return corrects/1000

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

learn()