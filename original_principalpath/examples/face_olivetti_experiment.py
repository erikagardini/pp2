import numpy as np
from matplotlib import pyplot as plt
import original_principalpath.linear_utilities as lu
import original_principalpath.principalpath as pp
from sklearn.datasets import fetch_olivetti_faces
import os
from scipy.spatial import distance

#Seed
np.random.seed(7)

#Flag for local or global solution
prefiltering = True

#Number of waypoints
NC=20

#Load the dataset
data = fetch_olivetti_faces()
targets = data.target

#Normalization and reshape
print("Data loaded")
data = data.images.reshape((len(data.images), -1))
data = data.astype('float64') / 255.

#Some boundaries selected by visual inspection
boundaries = [[0,399], [10,39], [40,299]]

for i in range(len(boundaries)):

    dir_res = "face_" + str(i) + "_prefiltering_" + str(prefiltering)
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
    if prefiltering:
        [X, boundary_ids, X_g]=pp.rkm_prefilter(X, boundary_ids, Nf=50, plot_ax=None)
        print("Data prefiltered. Remaining samples = " + str(X.shape[0]))

    #Init waypoinys
    waypoint_ids = lu.initMedoids(X, NC, 'kpp',boundary_ids)
    waypoint_ids = np.hstack([boundary_ids[0],waypoint_ids,boundary_ids[1]])
    W_init = X[waypoint_ids,:]
    print("Waypoints initialized")

    #Optimization of the waypoints
    s_span = np.array([10000, 1000, 100, 10, 0])
    models=np.ndarray([s_span.size,NC+2,d])
    for j,s in enumerate(s_span):
        [W,u]=pp.rkm(X, W_init, s, plot_ax=None)
        W_init = W
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

    # Plot the nearest picture corresponding to the waypoints of each model
    for j, s in enumerate(s_span):
        path = models[j, :, :]
        dst_mat = distance.cdist(path, X, 'euclidean')

        idxs = np.argsort(dst_mat, axis=1)[:, 0]
        plt.figure(figsize=(15, 2))
        n = path.shape[0]
        for k in range(n):
            # Display original
            ax = plt.subplot(2, n, k + 1)
            plt.imshow(X[idxs[k]].reshape(64, 64))
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.savefig(dir_res + "/pp_face_" + str(i) + "_model_" + str(j) + "_nearest_face_.png")
        plt.close()

    print("Plot of the models completed")
    np.save(dir_res + "/pp_models_" + str(i), models)
    print("\n\n\n")