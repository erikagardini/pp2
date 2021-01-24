import numpy as np
from matplotlib import pyplot as plt
import original_principalpath.principalpath as pp
import original_principalpath.linear_utilities as lu

#Seed
np.random.seed(7)

#Number of waypoints
NC=50

#Load a 2d dataset
names = ["gss"]#["circle", "constellation", "dmp"]#, "gss"]
for name in names:
    X=np.genfromtxt('../../datasets/2D_' + name + '.csv',delimiter=',')
    d = X.shape[1]

    #Some boundaries by visual inspection
    boundaries = {'circle': [512, 506],
         'constellation': [227,899],
         'dmp': [999,5],
         'gss': [868,182]}
    boundary_ids = boundaries.get(name)
    print("Principal path from " + str(boundary_ids[0]) + " to " + str(boundary_ids[1]) + "\n")

    #Prefiltering
    prefiltering=True
    if prefiltering:
        X_old = X
        [X, boundary_ids, X_g]=pp.rkm_prefilter(X, boundary_ids, plot_ax=None)
        print("Data prefiltered\n")

    #Init waypoints
    waypoint_ids = lu.initMedoids(X, NC, 'kpp', boundary_ids)
    waypoint_ids = np.hstack([boundary_ids[0], waypoint_ids, boundary_ids[1]])
    W_init = X[waypoint_ids,:]
    print("Waypoints initialized\n")

    #Waypoints optimization
    s_span = np.logspace(5, -5)
    s_span = np.hstack([s_span, 0])
    models = np.ndarray([s_span.size, NC+2, d])
    for j, s in enumerate(s_span):
        [W, u] = pp.rkm(X, W_init, s, plot_ax=None)
        W_init = W
        models[j, :, :] = W
    print("Waypoints optimized\n")

    #Model selection
    W_dst_var = pp.rkm_MS_pathvar(models, s_span, X)
    s_elb_id = lu.find_elbow(np.stack([s_span, W_dst_var], -1))
    print("Model selected: " + str(s_elb_id) + "\n")

    # Plot the elbow
    plt.scatter(s_span, W_dst_var)
    x_a = [s_span[0], s_span[-1]]
    y_a = [W_dst_var[0], W_dst_var[-1]]
    plt.plot(x_a, y_a, '-r')
    plt.scatter(x_a, y_a)
    plt.axvline(s_span[s_elb_id], 0, 1)
    plt.savefig("elbowpp_" + name + ".png")
    plt.close()

    #Plot the models
    for i, s in enumerate(s_span):
        path = models[i, :, :]

        if prefiltering:
            plt.scatter(X_old[:, 0], X_old[:, 1])
        plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(path[:, 0], path[:, 1])
        plt.plot(path[:,0], path[:,1], '-r')

        plt.savefig("pp_" + name + "_" + str(i) + ".png")
        plt.close()

    print("Plot of the models and the elbow completed\n")