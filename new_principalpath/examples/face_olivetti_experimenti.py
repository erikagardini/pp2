import numpy as np
from matplotlib import pyplot as plt
from mlxtend.data import loadlocal_mnist
import v3_principalpath.linear_utilities as lu
import v3_principalpath.principalpath as pp
from sklearn.datasets import fetch_olivetti_faces
from scipy.spatial import distance

#Seed
np.random.seed(7)

#Flag for local or global solution
prefiltering = False

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

    print("Principal path from " + str(boundaries[i][0]) + " to " + str(boundaries[i][1]) + "\n")

    X = data
    d = X.shape[1]

    boundary_ids = boundaries[i]

    #Plot the boundaries
    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[0]].reshape(64, 64))
    plt.gray()
    plt.savefig("face_start_" + str(i) + ".png")
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[1]].reshape(64, 64))
    plt.gray()
    plt.savefig("face_end_" + str(i) + ".png")
    plt.close()

    #Prefilter the data for a local solution
    [dijkstra, init_path] = pp.rkm_prefilter(X, boundary_ids, k=10, NC=20)

    plt.figure(figsize=(15, 2))
    n = dijkstra.shape[0]
    for k in range(n):
        # Display original
        ax = plt.subplot(2, n, k + 1)
        plt.imshow(dijkstra[k].reshape(64, 64))
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig("pp_face_" + str(i) + "dijkstra.png")
    plt.close()

    plt.figure(figsize=(15, 2))
    n = init_path.shape[0]
    for k in range(n):
        # Display original
        ax = plt.subplot(2, n, k + 1)
        plt.imshow(init_path[k].reshape(64, 64))
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig("pp_face_" + str(i) + "initpath.png")
    plt.close()

    W_init = init_path
    print("Data prefiltered\n")

    #Optimization of the waypoints
    s_span = np.array([0, 10, 100, 1000, 10000])
    models=np.ndarray([s_span.size,NC+2,d])
    for j,s in enumerate(s_span):
        [W,u]=pp.rkm(init_path, W_init, s, plot_ax=None)
        #W_init = W
        models[j,:,:] = W
    print("Waypoints optimized\n")

    #Model selection
    W_dst_var = pp.rkm_MS_pathlen(models, s_span, X)
    s_elb_id = lu.find_elbow(np.stack([s_span, W_dst_var], -1))
    print("Model selected: " + str(s_elb_id) + "\n")

    #Plot the elbow
    plt.scatter(s_span, W_dst_var)
    x_a = [s_span[0], s_span[-1]]
    y_a = [W_dst_var[0], W_dst_var[-1]]
    plt.plot(x_a, y_a, '-r')
    plt.scatter(x_a, y_a)
    plt.axvline(s_span[s_elb_id], 0, 1)
    plt.savefig("elbowpp_way_face" + str(i) + ".png")
    plt.close()

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
        if j == s_elb_id:
            plt.savefig("pp_face_"+ str(i)+"_model_" + str(j) + "(best).png")
        else:
            plt.savefig("pp_face_"+ str(i)+"_model_" + str(j) + ".png")
        plt.close()

    print("Plot of the models and the elbow completed\n")

    best_path = models[s_elb_id, :, :]
    '''edit_distances = []
    for i in range(best_path.shape[0]-1):
        edit_distance = np.sum(np.abs(best_path[i + 1, :] - best_path[i, :]))
        edit_distances.append(edit_distance)

    edit_distances = np.array(edit_distances)
    print(np.mean(edit_distances))
    print(np.std(edit_distances))
    print(edit_distances)'''

    dst_mat = distance.cdist(best_path, X, 'euclidean')
    idxs = np.argsort(dst_mat, axis=1)[:,0]
    plt.figure(figsize=(15, 2))
    n = best_path.shape[0]
    for k in range(n):
        # Display original
        ax = plt.subplot(2, n, k + 1)
        plt.imshow(X[idxs[k]].reshape(64, 64))
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig("pp_face_" + str(i) + "_nearest_face_.png")
    plt.close()