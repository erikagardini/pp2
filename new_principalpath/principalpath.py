import numpy as np
from scipy.spatial import distance
from scipy.sparse import csgraph
from matplotlib import pyplot
import original_principalpath.linear_utilities as lu
import tensorflow as tf

def rkm(X, init_W, s, plot_ax=None):
    """
    Regularized K-means for principal path, MINIMIZER.

    Args:
        [ndarray float] X: data matrix

        [ndarray float] init_W: initial waypoints matrix

        [float] s: regularization parameter 

        [matplotlib.axis.Axes] plot_ax: Axes for the 2D plot (first 2 dim of X), None to avoid plotting

    Returns:
        [ndarray float] W: final waypoints matrix

        [ndarray int] labels: final

    References:
        [1] 'Finding Prinicpal Paths in Data Space', M.J.Ferrarotti, W.Rocchia, S.Decherchi, [submitted]
        [2] 'Design and HPC Implementation of Unsupervised Kernel Methods in the Context of Molecular Dynamics', M.J.Ferrarotti, PhD Thesis.
    """

    #extract useful info from args
    N = X.shape[0]
    d = X.shape[1]
    NC = init_W.shape[0]-2

    #construct boundary matrix
    boundary = init_W[[0,NC+1],:]
    B=np.zeros([NC,d],float)
    B[[0,NC-1],:]=boundary

    #construct regularizer hessian
    AW = np.diag(np.ones(NC))+np.diag(-0.5*np.ones(NC-1),1)+np.diag(-0.5*np.ones(NC-1),-1)

    #compute initial labels
    XW_dst = distance.cdist(X,init_W,'sqeuclidean')
    u = XW_dst.argmin(1)

    #iterate the minimizer
    converged = False
    it = 0
    while(not converged):
        it = it+1
        #print('iteration '+repr(it))

        #compute cardinality
        W_card=np.zeros(NC+2,int)
        for i in range(NC+2):
            W_card[i] = np.sum(u==i)

        #compute centroid matrix
        C = np.ndarray([NC,d],float)
        for i in range(NC):
            C[i,:] = np.sum(X[u==i+1,:],0)

        #construct k-means hessian 
        AX = np.diag(W_card[1:NC+1])

        #update waypoints
        W = np.matmul(np.linalg.pinv(AX+s*AW),C+0.5*s*B)
        W = np.vstack([boundary[0,:],W,boundary[1,:]])

        #compute new labels
        XW_dst = distance.cdist(X,W,'sqeuclidean')
        u_new = XW_dst.argmin(1)

        #check for convergence
        converged = not np.sum(u_new!=u)
        u=u_new

        #plot
        if(plot_ax is not None):
            pyplot.sca(plot_ax)
            pyplot.ion()
            pyplot.cla()
            pyplot.title('Annealing, s='+repr(s))
            pyplot.plot(X[:,0],X[:,1],'bo')
            pyplot.plot(W[:,0],W[:,1],'-ro')
            pyplot.axis('equal')

            pyplot.pause(1.0/60)
    
    return W, u


def rkm_prefilter(X, boundary_ids, k = 10, NC = 20, name=""):
    # Put the boundaries at the beginning and the end of the input matrix X
    X_ = np.delete(X, (boundary_ids[0], boundary_ids[1]), axis=0)
    initial = X[boundary_ids[0], :].reshape(1, X.shape[1])
    final = X[boundary_ids[1], :].reshape(1, X.shape[1])
    X = np.concatenate((initial, X_, final))

    # Compute Dijkstra
    dst_mat = distance.cdist(X, X, 'sqeuclidean')

    '''dst = distance.cdist(X, X, 'euclidean')
    ind = np.argsort(dst, axis=1)
    sigma_x = np.zeros(dst.shape[0], dtype='float32')
    for i in range(dst.shape[0]):
        sigma_x[i] = dst[i, ind[i, 7]]

    a = np.transpose(np.tile(sigma_x, (X.shape[0], 1)))
    b = np.tile(sigma_x, (X.shape[0], 1))
    s_matrix = a * b

    kernel_matrix = np.exp(-1 * (np.divide(dst_mat, s_matrix)))

    kernel_distance = (2-(2*kernel_matrix))**2'''

    if k != X.shape[1]:
        idxs = np.argsort(dst_mat, axis=1)[:, 1:k + 1]
        for i in range(dst_mat.shape[0]):
            for j in range(dst_mat.shape[1]):
                if j not in idxs[i, :]:
                    dst_mat[i, j] = dst_mat[i, j] * 100000

    [path_dst, path_pre] = csgraph.dijkstra(dst_mat, False, 0, True)
    path = np.ndarray(0, int)

    i = X.shape[0] - 1
    while i != 0:
        path = np.hstack([i, path])
        i = path_pre[i]
    path = np.hstack([i, path])

    # Plot Dijkstra path path
    dijkstra = X[path, :]

    # Waypoints adjustment
    if NC != dijkstra.shape[0]:
        init_path = movePath(dijkstra, NC)

    return [dijkstra, init_path]

def movePath(init_path, NC):
    new_path = []
    new_path.append(init_path[0,:])

    #Compute path length
    p1 = init_path[:-1, ]
    p2 = init_path[1:, ]
    dist = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
    p_len = np.sum(dist)
    if NC == 0:
        we = p_len / p1.shape[0]
    else:
        we = p_len / (NC + 1)

    way_to_do = we

    #Start updating
    i = 0
    end_p = init_path[i+1,:]
    start_p = init_path[i,:]

    while len(new_path) != (NC+1):
        d = np.sqrt(np.sum((end_p - start_p) ** 2))
        if d > way_to_do:
            #Insert a new point after We
            lam = way_to_do / d
            new_point = (lam * end_p) + ((1 - lam) * start_p)
            new_path.append(new_point)
            #Computing the remaining segment
            remaining = d - way_to_do
            if round(remaining, 4) < round(we,4):
                i = i + 1
                way_to_do = we - remaining
                end_p = init_path[i + 1, :]
                start_p = init_path[i, :]
            else:
                way_to_do = we
                end_p = init_path[i + 1, :]
                start_p = new_point
        elif d == way_to_do:
            new_path.append(init_path[i+1,:])
            i = i + 1
            end_p = init_path[i + 1, :]
            start_p = init_path[i, :]
        elif d < way_to_do:
            way_to_do = way_to_do - d
            i = i + 1
            end_p = init_path[i + 1, :]
            start_p = init_path[i, :]

    new_path.append(init_path[-1, :])

    return np.array(new_path)

def rkm_MS_pathvar(models, s_span, X):
    """
    Regularized K-means for principal path, MODEL SELECTION, variance on waypoints interdistance.

    Args:
        [ndarray float] models: matrix with path models, shape N_models x N x (NC+2)

        [ndarray float] s_span: array with values of the reg parameter for each model (sorted in decreasing order, with 0 as last value)

        [ndarray float] X: data matrix

    Returns:
        [ndarray float] W_dst_var: array with values of variance for each model
    """
    W_dst_var=np.ndarray(models.shape[0],float)
    for i in range(models.shape[0]):
        W = models[i,:,:]
        W_dst=np.linalg.norm(W[1:,:]-W[0:-1,:],axis=1)
        W_dst_var[i] = np.var(W_dst)

    return W_dst_var

def rkm_MS_pathlen(models, s_span, X):
    """
    Regularized K-means for principal path, MODEL SELECTION, Path length.
    Args:
        [ndarray float] models: matrix with path models, shape N_models x N x (NC+2)
        [ndarray float] s_span: array with values of the reg parameter for each model (sorted in decreasing order, with 0 as last value)
        [ndarray float] X: data matrix
    Returns:
        [ndarray float] len_s: array with values of path length for each model
    """
    len_s=np.zeros(s_span.size,float)
    for i,s in enumerate(s_span):
        W = models[i,:,:]
        NC = W.shape[0]-2
        for j,w in enumerate(W[0:-1,:]):
            w_nxt = W[j+1,:]
            len_s[i] = len_s[i] + np.sqrt(np.dot(w,w)+np.dot(w_nxt,w_nxt)-2*np.dot(w,w_nxt))

    return len_s
