import numpy as np
from matplotlib import pyplot as plt
import new_principalpath.principalpath as pp
import os

#Number of waypoints
NC=50

names = ["circle", "constellation", "dmp", "gss"]
for name in names:

    #Create the folder to save the results
    dir_res = name
    if not os.path.exists(dir_res):
        os.mkdir(dir_res)

    # Load a 2d dataset
    X=np.genfromtxt('../../datasets/2D_' + name + '.csv',delimiter=',')
    d = X.shape[1]

    #Some boundaries by visual inspection
    boundaries = {'circle': [512, 506],
         'constellation': [227,899],
         'dmp': [999,5],
         'gss': [833,199]}
    boundary_ids = boundaries.get(name)
    print("Principal path from " + str(boundary_ids[0]) + " to " + str(boundary_ids[1]))

    #Prefiltering
    [dijkstra, init_path] = pp.rkm_prefilter(X, boundary_ids, k=5, NC=NC, name=name)

    #Plot the Dijkstra path
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(dijkstra[:, 0], dijkstra[:, 1])
    plt.plot(dijkstra[:, 0], dijkstra[:, 1], '-r')
    plt.scatter(dijkstra[0, 0], dijkstra[0, 1])
    plt.scatter(dijkstra[-1, 0], dijkstra[-1, 1])

    plt.savefig(dir_res + "/dijkstra_path.png")
    plt.close()

    #Plot the initialized path
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(init_path[:, 0], init_path[:, 1])
    plt.plot(init_path[:, 0], init_path[:, 1], '-r')
    plt.scatter(init_path[0, 0], init_path[0, 1])
    plt.scatter(init_path[-1, 0], init_path[-1, 1])

    plt.savefig(dir_res + "/updated_path.png")
    plt.close()

    W_init = init_path
    print("Path initialized")

    #Waypoints optimization
    s_span = np.array([1000000, 100000, 10000, 1000, 100, 10, 0])
    models = np.ndarray([s_span.size, NC+2, d])
    for j, s in enumerate(s_span):
        [W, u] = pp.rkm(init_path, W_init, s, plot_ax=None)
        #W_init = W
        models[j, :, :] = W
    print("Waypoints optimized")

    #Model selection
    evidence = pp.rkm_MS_evidence(models, s_span, X)
    max_evidence = np.argmax(evidence)
    print("Model selected evidence: " + str(max_evidence))

    #Plot evidence graph
    plt.plot(evidence)
    plt.savefig(dir_res + "/max_evidence.png")
    plt.close()

    #Plot the models
    for i, s in enumerate(s_span):
        path = models[i, :, :]
        plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(path[:, 0], path[:, 1])
        plt.plot(path[:,0], path[:,1], '-r')
        if i == max_evidence:
            plt.savefig(dir_res + "/pp_" + name + "_" + str(i) + "(best).png")
        else:
            plt.savefig(dir_res + "/pp_" + name + "_" + str(i) + ".png")
        plt.close()

    print("Plot of the models completed")

    print("\n\n\n")
