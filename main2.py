import pandas as pd
import numpy as np

structure = np.array([28*28, 16, 16, 10])

test_data = pd.read_csv("./data/mnist_test.csv")

def guess(imageData):
    pass

ith_row = test_data.iloc[100]
label = ith_row[0]
pixel_values = np.array(ith_row[1:])/255

print("Label: "+ str(label))

for pixel in pixel_values:
    print(pixel)