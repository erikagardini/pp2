import numpy as np
import os
import new_principalpath.utilities as utilities
from new_principalpath.pp2 import PrincipalPath
from new_principalpath.examples import linear_utilities as lu
import matplotlib.pyplot as plt
#Number of waypoints
NC = 50

names = ["constellation"]#["constellation", "circle", "dmp", "gss"]
for name in names:
    #Load dataset
    X = np.genfromtxt('../../datasets/2D_' + name + '.csv',delimiter=',')

    N = X.shape[0]
    d = X.shape[1]

    # Some boundaries by visual inspection
    boundaries = {'circle': [512, 506],
                  'constellation': [227, 899],
                  'dmp': [999, 5],
                  'gss': [833, 199]}
    boundary_ids = boundaries.get(name) #lu.getMouseSamples2D(X, 2)#
    print("New principal path from " + str(boundary_ids[0]) + " to " + str(boundary_ids[1]) + "\n")

    #Plot boundaries
    '''plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    plt.scatter(X[boundary_ids[1], 0], X[boundary_ids[1], 1])
    plt.scatter(X[boundary_ids[0], 0], X[boundary_ids[0], 1])

    plt.savefig("boundaries.png")
    plt.close()'''

    #Parameters for the optimization
    epochs = 20
    s_span_1 = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    s_span_2 = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    lr = [0.01]

    dir_res = name + "_v2/"
    if not os.path.exists(dir_res):
        os.mkdir(dir_res)

    #Model selection
    y_mode = 'length'
    criterion = 'elbow'

    pp = PrincipalPath(NC, boundary_ids, mode='self_tuning', batch_size=N)
    pp.init(X, k = X.shape[1], filename=dir_res)
    pp.optimize(X, s_span_1, s_span_2, lr, epochs, dir_res)