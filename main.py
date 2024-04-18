import time
import numpy as np

def learn():
    print("Learning...")
    time.sleep(10)
    print("Learning done!")
def guess(imageData):
    print(imageData)
    return np.random.rand(10)
def guessRandom():
    return np.random.rand(784), np.random.rand(10)