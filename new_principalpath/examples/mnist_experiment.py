import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from mlxtend.data import loadlocal_mnist
import new_principalpath.principalpath as pp
import os

#Number of waypoints
NC=20

#Load the dataset
x_test, y_test = loadlocal_mnist(
            images_path='../../datasets/t10k-images-idx3-ubyte',
            labels_path='../../datasets/t10k-labels-idx1-ubyte')

#Normalization and reshape
print("Data loaded \n\n")
x_test = x_test.astype('float64') / 255.
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#Some boundaries selected by visual inspection
boundaries = [[568,270], [21,313], [75,19], [307,169], [457,422], [446,105]]

for i in range(len(boundaries)):

    dir_res = "mnist_" + str(i)
    if not os.path.exists(dir_res):
        os.mkdir(dir_res)

    print("Principal path from " + str(boundaries[i][0]) + " to " + str(boundaries[i][1]))

    X = x_test
    d = X.shape[1]

    boundary_ids = boundaries[i]

    #Plot the boundaries
    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[0]].reshape(28, 28))
    plt.savefig(dir_res + "/mnist_start_" + str(i) + ".png")
    plt.gray()
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[1]].reshape(28, 28))
    plt.savefig(dir_res + "/mnist_end_" + str(i) + ".png")
    plt.gray()
    plt.close()

    #Prefilter the data for a local solution
    [dijkstra, init_path] =pp.rkm_prefilter(X, boundary_ids, k=10, NC=20)
    print("Data prefiltered")

    plt.figure(figsize=(15, 2))
    n = dijkstra.shape[0]
    for k in range(n):
        # Display original
        ax = plt.subplot(2, n, k + 1)
        plt.imshow(dijkstra[k].reshape(28, 28))
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(dir_res + "/pp_mnist_" + str(i) + "dijkstra.png")
    plt.close()

    plt.figure(figsize=(15, 2))
    n = init_path.shape[0]
    for k in range(n):
        # Display original
        ax = plt.subplot(2, n, k + 1)
        plt.imshow(init_path[k].reshape(28, 28))
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(dir_res + "/pp_mnist_" + str(i) + "initpath.png")
    plt.close()

    W_init = init_path

    #Optimization of the waypoints
    s_span = np.array([10000, 1000, 100, 10, 0])

    models=np.ndarray([s_span.size,NC+2,d])
    for j,s in enumerate(s_span):
        [W,u]=pp.rkm(init_path, W_init, s, plot_ax=None)
        #W_init = W
        models[j,:,:] = W
    print("Waypoints optimized")

    #Plot the models
    for j, s in enumerate(s_span):
        path = models[j, :, :]

        plt.figure(figsize=(15, 2))
        n = path.shape[0]
        for k in range(n):
            # Display original
            ax = plt.subplot(2, n, k + 1)
            plt.imshow(path[k].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(dir_res + "/pp_mnist_"+ str(i)+"_model_" + str(j) + ".png")
        plt.close()

    print("Plot of the models completed")

    for j, s in enumerate(s_span):
        path = models[j, :, :]
        dst_mat = distance.cdist(path, X, 'euclidean')
        idxs = np.argsort(dst_mat, axis=1)[:,0]
        plt.figure(figsize=(15, 2))
        n = path.shape[0]
        for k in range(n):
            # Display original
            ax = plt.subplot(2, n, k + 1)
            plt.imshow(X[idxs[k]].reshape(28, 28))
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.savefig(dir_res + "/pp_mnist_" + str(i) + "_model_" + str(j) + "_nearest_face_.png")
        plt.close()

    np.save(dir_res + "/pp_models_" + str(i), models)
    print("\n\n\n")