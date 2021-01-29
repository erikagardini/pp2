import numpy as np
from matplotlib import pyplot as plt
import v3_principalpath.principalpath as pp
import v3_principalpath.linear_utilities as lu

#Seed
np.random.seed(7)

#Number of waypoints
NC=20

#Load a 2d dataset
names = ["circle", "constellation", "dmp", "gss"]
for name in names:
    X=np.genfromtxt('../../datasets/2D_' + name + '.csv',delimiter=',')
    d = X.shape[1]

    #Some boundaries by visual inspection
    boundaries = {'circle': [512, 506],
         'constellation': [227,899],
         'dmp': [999,5],
         'gss': [833,199]}
    boundary_ids = boundaries.get(name)
    print("Principal path from " + str(boundary_ids[0]) + " to " + str(boundary_ids[1]) + "\n")

    #Prefiltering
    [dijkstra, init_path] = pp.rkm_prefilter(X, boundary_ids, k=10, NC=20, name=name)

    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(dijkstra[:, 0], dijkstra[:, 1])
    plt.plot(dijkstra[:, 0], dijkstra[:, 1], '-r')
    plt.scatter(dijkstra[0, 0], dijkstra[0, 1])
    plt.scatter(dijkstra[-1, 0], dijkstra[-1, 1])

    plt.savefig(name + "dijkstra_path.png")
    plt.close()

    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(init_path[:, 0], init_path[:, 1])
    plt.plot(init_path[:, 0], init_path[:, 1], '-r')
    plt.scatter(init_path[0, 0], init_path[0, 1])
    plt.scatter(init_path[-1, 0], init_path[-1, 1])

    plt.savefig(name + "updated_path.png")
    plt.close()

    W_init = init_path
    print("Path initialized\n")

    #Waypoints optimization
    #s_span = np.logspace(1, -1)
    #s_span = np.hstack([s_span, 0])
    s_span = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])#, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 35, 30, 20, 10, 9, 8, 7, 6, 5, 0])
    models = np.ndarray([s_span.size, NC+2, d])
    for j, s in enumerate(s_span):
        [W, u] = pp.rkm(init_path, W_init, s, plot_ax=None)
        #W_init = W
        models[j, :, :] = W
    print("Waypoints optimized\n")

    #Model selection
    W_dst_var = pp.rkm_MS_pathlen(models, s_span, X)
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
        plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(path[:, 0], path[:, 1])
        plt.plot(path[:,0], path[:,1], '-r')
        if i == s_elb_id:
            plt.savefig("pp_" + name + "_" + str(i) + "(best).png")
        else:
            plt.savefig("pp_" + name + "_" + str(i) + ".png")
        plt.close()

    '''best_path = models[s_elb_id,:,:]
    gss_centers = np.array([[1, 1], [2.2, 2], [3.4, 1], [4.6, 2], [5.8, 1]])
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(best_path[:, 0], best_path[:, 1])
    plt.plot(best_path[:, 0], best_path[:, 1], '-r')
    plt.scatter(X[boundary_ids[0], 0], X[boundary_ids[0], 1])
    plt.scatter(X[boundary_ids[0], 0], X[boundary_ids[1], 1])
    plt.scatter(gss_centers[:, 0], gss_centers[:, 1], c='yellow')'''

    plt.savefig("gss_centers" + ".png")
    plt.close()

    print("Plot of the models and the elbow completed\n")