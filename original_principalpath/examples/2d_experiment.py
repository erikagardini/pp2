import numpy as np
from matplotlib import pyplot as plt
import original_principalpath.principalpath as pp
import original_principalpath.linear_utilities as lu
import os

#Seed
np.random.seed(1234)
prefiltering = True

#Number of waypoints
NC=50

#Load a 2d dataset
names = ["circle", "constellation", "dmp", "gss"]
for name in names:
    if prefiltering:
        dir_res = name + "_prefiltering_" + str(prefiltering)+"/"
    else:
        dir_res = name + "_no_prefiltering_" + str(prefiltering)+"/"
    if not os.path.exists(dir_res):
        os.mkdir(dir_res)

    X=np.genfromtxt('../../datasets/2D_' + name + '.csv',delimiter=',')
    d = X.shape[1]

    #Some boundaries by visual inspection
    boundaries = {'circle': [512, 506],
         'constellation': [227,899],
         'dmp': [999,5],
         'gss': [833,199]}
    boundary_ids = boundaries.get(name)
    print("Principal path from " + str(boundary_ids[0]) + " to " + str(boundary_ids[1]))

    plt.scatter(X[:, 0], X[:, 1], c='C0', alpha=0.5)
    plt.scatter(X[boundary_ids[0], 0], X[boundary_ids[0], 1], marker=(5, 1), s=300.0, alpha=1.0, c='g')
    plt.scatter(X[boundary_ids[1], 0], X[boundary_ids[1], 1], marker=(5, 1), s=300.0, alpha=1.0, c='m')
    plt.savefig(dir_res + "/boundaries.png")
    plt.close()

    #Prefiltering
    if prefiltering:
        X_old = X
        [X, boundary_ids, X_g]=pp.rkm_prefilter(X, boundary_ids, plot_ax=None)
        print("Data prefiltered\n")
        plt.scatter(X_old[:, 0], X_old[:, 1], c='grey', alpha=0.5)
        plt.scatter(X[:, 0], X[:, 1], c='C0', alpha=0.5)
        plt.scatter(X[boundary_ids[0], 0], X[boundary_ids[0], 1], marker=(5, 1), s=300.0, alpha=1.0, c='g')
        plt.scatter(X[boundary_ids[1], 0], X[boundary_ids[1], 1], marker=(5, 1), s=300.0, alpha=1.0, c='m')

        plt.savefig(dir_res + "/pp_filter" + name + "_" + str(200) + "_" + str(0.1) + "_" + str(1234) + ".png")
        plt.close()

    #Init waypoints
    waypoint_ids = lu.initMedoids(X, NC, 'kpp', boundary_ids)
    waypoint_ids = np.hstack([boundary_ids[0], waypoint_ids, boundary_ids[1]])
    W_init = X[waypoint_ids,:]
    print("Waypoints initialized\n")

    if prefiltering:
        plt.scatter(X_old[:, 0], X_old[:, 1], c='grey', alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c='C0', alpha=0.5)
    plt.scatter(W_init[:, 0], W_init[:, 1], c='C1')

    plt.scatter(X[boundary_ids[0], 0], X[boundary_ids[0], 1], marker=(5, 1), s=300.0, alpha=1.0, c='g')
    plt.scatter(X[boundary_ids[1], 0], X[boundary_ids[1], 1], marker=(5, 1), s=300.0, alpha=1.0, c='m')
    plt.savefig(dir_res + "/pp_init" + name + "_" + str(1234) + ".png")
    plt.close()

    #Waypoints optimization
    s_span = np.array([1000000, 100000, 10000, 1000, 100, 10, 0])
    models = np.ndarray([s_span.size, NC+2, d])
    for j, s in enumerate(s_span):
        [W, u] = pp.rkm(X, W_init, s, plot_ax=None)
        W_init = W
        models[j, :, :] = W
    print("Waypoints optimized\n")

    #Model selection
    evidence = pp.rkm_MS_evidence(models, s_span, X)
    max_evidence = np.argmax(evidence)
    print("Model selected: " + str(max_evidence) + "\n")

    plt.plot(evidence)
    plt.savefig(dir_res + "max_evidence.png")
    plt.close()

    #Plot the models
    for i, s in enumerate(s_span):
        path = models[i, :, :]

        if prefiltering:
            plt.scatter(X_old[:, 0], X_old[:, 1], c='grey', alpha=0.5)
        plt.scatter(X[:, 0], X[:, 1], c='C0', alpha=0.5)
        plt.scatter(path[:, 0], path[:, 1], c='C1', zorder=1)
        plt.plot(path[:,0], path[:,1], '-r', zorder=2)
        plt.scatter(X[boundary_ids[0], 0], X[boundary_ids[0], 1], marker=(5, 1), s=300.0, alpha=1.0, c='g', zorder=10)
        plt.scatter(X[boundary_ids[1], 0], X[boundary_ids[1], 1], marker=(5, 1), s=300.0, alpha=1.0, c='m', zorder=10)

        if i == max_evidence:
            plt.savefig(dir_res + "pp_" + name + "_" + str(i) + "(best).png")
        else:
            plt.savefig(dir_res + "pp_" + name + "_" + str(i) + ".png")

        plt.close()

    print("Plot of the models and the elbow completed\n")