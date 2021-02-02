import numpy as np
from matplotlib import pyplot as plt
import new_principalpath.principalpath as pp
from sklearn.datasets import fetch_olivetti_faces
from scipy.spatial import distance
import os

#Number of waypoints
NC=20

#Load the dataset
data = fetch_olivetti_faces()
targets = data.target

#Normalization and reshape
print("Data loaded \n\n")
data = data.images.reshape((len(data.images), -1))
data = data.astype('float64') / 255.

#Some boundaries selected by visual inspection
boundaries = [[0,399], [10,39], [40,299]]

for i in range(len(boundaries)):
    #Create the folder to save the results
    dir_res = "face_" + str(i)
    if not os.path.exists(dir_res):
        os.mkdir(dir_res)

    print("Principal path from " + str(boundaries[i][0]) + " to " + str(boundaries[i][1]))

    X = data
    d = X.shape[1]

    boundary_ids = boundaries[i]

    #Plot the boundaries
    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[0]].reshape(64, 64))
    plt.gray()
    plt.savefig(dir_res + "/start_" + str(i) + ".png")
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[1]].reshape(64, 64))
    plt.gray()
    plt.savefig(dir_res + "/end_" + str(i) + ".png")
    plt.close()

    #Prefilter the data for a local solution
    [dijkstra, init_path] = pp.rkm_prefilter(X, boundary_ids, k=10, NC=20)

    #Plot Dijkstra path
    plt.figure(figsize=(15, 2))
    n = dijkstra.shape[0]
    for k in range(n):
        # Display original
        ax = plt.subplot(2, n, k + 1)
        plt.imshow(dijkstra[k].reshape(64, 64))
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(dir_res + "/pp_face_" + str(i) + "dijkstra.png")
    plt.close()

    #Plot initialized path
    plt.figure(figsize=(15, 2))
    n = init_path.shape[0]
    for k in range(n):
        # Display original
        ax = plt.subplot(2, n, k + 1)
        plt.imshow(init_path[k].reshape(64, 64))
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(dir_res + "/pp_face_" + str(i) + "initpath.png")
    plt.close()

    W_init = init_path
    print("Data prefiltered\n")

    #Optimization of the waypoints
    s_span = np.array([10000, 1000, 100, 10, 0])
    models=np.ndarray([s_span.size,NC+2,d])
    for j,s in enumerate(s_span):
        [W,u]=pp.rkm(init_path, W_init, s, plot_ax=None)
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
            plt.imshow(path[k].reshape(64, 64))
            #plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.savefig(dir_res + "/pp_face_"+ str(i)+"_model_" + str(j) + "_s=" + str(s) + ".png")
        plt.close()

    print("Plot of the models completed")

    #Plot the nearest picture corresponding to the waypoints of each model
    for j, s in enumerate(s_span):
        path = models[j, :, :]
        dst_mat = distance.cdist(path, X, 'euclidean')
        idxs = np.argsort(dst_mat, axis=1)[:,0]
        plt.figure(figsize=(15, 2))
        n = path.shape[0]
        for k in range(n):
            # Display original
            ax = plt.subplot(2, n, k + 1)
            plt.imshow(X[idxs[k]].reshape(64, 64))
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.savefig(dir_res + "/pp_face_" + str(i)+"_model_" + str(j)+ "_nearest_face_.png")
        plt.close()

    print("\n\n\n")