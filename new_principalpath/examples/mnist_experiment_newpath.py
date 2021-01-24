import numpy as np
import matplotlib.pyplot as plt
import os
from new_principalpath import utilities
from new_principalpath.pp2 import PrincipalPath
from mlxtend.data import loadlocal_mnist

#Load the dataset
x_test, y_test = loadlocal_mnist(
            images_path='../../datasets/t10k-images-idx3-ubyte',
            labels_path='../../datasets/t10k-labels-idx1-ubyte')

x_test = x_test.astype('float64') / 255.
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

X = x_test

N = X.shape[0]
d = X.shape[1]

#Number of waypoints
NC = 20

#Parameters
epochs = 20
s_span_2 = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]
s_span_1 = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]
learning_rates = [0.01]

#Boundaries
boundaries = [[568,270], [21,313], [75,19], [307,169], [457,422], [446,105]]

# Model selection
y_mode = 'length'
criterion = 'elbow'

i=0
for el in boundaries:
    boundary_ids = el
    print("New principal path from " + str(boundary_ids[0]) + " to " + str(boundary_ids[1]) + "\n")

    dir_res = "mnist" + str(i) + "/"
    if not os.path.exists(dir_res):
        os.mkdir(dir_res)
    i = i + 1

    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[0]].reshape(28, 28))
    plt.gray()
    plt.savefig(dir_res + "start_point.png")
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[1]].reshape(28, 28))
    plt.gray()
    plt.savefig(dir_res + "end_point.png")
    plt.close()

    pp = PrincipalPath(NC, boundary_ids, mode='self_tuning', batch_size=N)
    pp.init(X, k=100, filename=dir_res)
    pp.optimize(X, s_span_1, s_span_2, learning_rates, epochs, dir_res)