import numpy as np
from matplotlib import pyplot as plt
from mlxtend.data import loadlocal_mnist
import v3_principalpath.linear_utilities as lu
import v3_principalpath.principalpath as pp

#Seed
np.random.seed(1234)

#Flag for local or global solution
prefiltering = True

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

    print("Principal path from " + str(boundaries[i][0]) + " to " + str(boundaries[i][1]) + "\n")

    X = x_test
    d = X.shape[1]

    boundary_ids = boundaries[i]

    #Plot the boundaries
    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[0]].reshape(28, 28))
    plt.savefig("mnist_start_" + str(i) + ".png")
    plt.gray()
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[1]].reshape(28, 28))
    plt.savefig("mnist_end_" + str(i) + ".png")
    plt.gray()
    plt.close()

    #Prefilter the data for a local solution
    init_path =pp.rkm_prefilter(X, boundary_ids, k=10, NC=20)
    print("Data prefiltered\n")

    plt.figure(figsize=(15, 2))
    n = init_path.shape[0]
    for k in range(n):
        # Display original
        ax = plt.subplot(2, n, k + 1)
        plt.imshow(init_path[k].reshape(28, 28))
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig("pp_mnist_" + str(i) + "initpath.png")
    plt.close()

    W_init = init_path

    #Optimization of the waypoints
    s_span = np.array([0, 10, 100, 1000, 10000])
    models=np.ndarray([s_span.size,NC+2,d])
    for j,s in enumerate(s_span):
        [W,u]=pp.rkm(init_path, W_init, s, plot_ax=None)
        #W_init = W
        models[j,:,:] = W
    print("Waypoints optimized\n")

    #Model selection
    W_dst_var = pp.rkm_MS_pathvar(models, s_span, X)
    s_elb_id = lu.find_elbow(np.stack([s_span, W_dst_var], -1))
    print("Model selected: " + str(s_elb_id) + "\n")

    #Plot the elbow
    plt.scatter(s_span, W_dst_var)
    x_a = [s_span[0], s_span[-1]]
    y_a = [W_dst_var[0], W_dst_var[-1]]
    plt.plot(x_a, y_a, '-r')
    plt.scatter(x_a, y_a)
    plt.axvline(s_span[s_elb_id], 0, 1)
    plt.savefig("elbowpp_way_mnist" + str(i) + ".png")
    plt.close()

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
        plt.savefig("pp_mnist_"+ str(i)+"_model_" + str(j) + ".png")
        plt.close()

    print("Plot of the models and the elbow completed\n")

    best_path = models[s_elb_id, :, :]
    edit_distances = []
    for i in range(best_path.shape[0]-1):
        edit_distance = np.sum(np.abs(best_path[i + 1, :] - best_path[i, :]))
        edit_distances.append(edit_distance)

    edit_distances = np.array(edit_distances)
    print(np.mean(edit_distances))
    print(np.std(edit_distances))
    print(edit_distances)