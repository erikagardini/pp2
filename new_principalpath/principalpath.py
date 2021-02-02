import numpy as np
from scipy.spatial import distance
from scipy.sparse import csgraph
from matplotlib import pyplot

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

    #Build penalized graph
    dst_mat = distance.cdist(X, X, 'sqeuclidean')

    if k != X.shape[1]:
        idxs = np.argsort(dst_mat, axis=1)[:, 1:k + 1]
        for i in range(dst_mat.shape[0]):
            for j in range(dst_mat.shape[1]):
                if j not in idxs[i, :]:
                    dst_mat[i, j] = dst_mat[i, j] * 100000

    #Compute Dijkstra
    [path_dst, path_pre] = csgraph.dijkstra(dst_mat, False, 0, True)
    path = np.ndarray(0, int)

    i = X.shape[0] - 1
    while i != 0:
        path = np.hstack([i, path])
        i = path_pre[i]
    path = np.hstack([i, path])

    dijkstra = X[path, :]

    # Linear interpolation
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


def rkm_MS_evidence(models, s_span, X):
    """
    Regularized K-means for principal path, MODEL SELECTION, Bayesian Evidence.
    Args:
        [ndarray float] models: matrix with path models, shape N_models x N x (NC+2)
        [ndarray float] s_span: array with values of the reg parameter for each model (sorted in decreasing order, with 0 as last value)
        [ndarray float] X: data matrix
    Returns:
        [ndarray float] logE_s: array with values of log evidence for each model
    """

    if(s_span[-1]>0.0):
        raise ValueError('In order to evaluate the evidence a model with s=0 has to be provided')

    #Evaluate unregularized cost
    cost_ureg=np.sum(rkm_cost(X, models[-1,:,:],s_span[-1]))

    logE_s = np.ndarray(s_span.size,float)
    for i,s in enumerate(s_span):
        N = X.shape[0]
        W = models[i,:,:]
        NC = W.shape[0]-2
        d = W.shape[1]

        #Set gamma (empirical rational) and compute lambda
        gamma = np.sqrt(N)*0.125/np.mean(distance.cdist(X,X,'euclidean'))
        lambd = s*gamma

        #Maximum Posterior cost
        cost_MP=np.sum(rkm_cost(X, W, s))

        #Find labels
        XW_dst = distance.cdist(X,W,'sqeuclidean')
        u = XW_dst.argmin(1)
        #Compute cardinality
        W_card=np.zeros(NC+2,int)
        for j in range(NC+2):
            W_card[j] = np.sum(u==j)

        #Construct boundary matrix
        boundary = W[[0,NC+1],:]
        B=np.zeros([NC,d],float)
        B[[0,NC-1],:]=boundary

        #Construct regularizer hessian
        AW = np.diag(np.ones(NC))+np.diag(-0.5*np.ones(NC-1),1)+np.diag(-0.5*np.ones(NC-1),-1)

        #Construct k-means hessian
        AX = np.diag(W_card[1:NC+1])

        #Compute global hessian
        A = AX+s*AW

        #Evaluate log-evidence
        logE = -0.5*d*np.log(np.sum(np.linalg.eigvals(A)))
        logE = logE + gamma*(cost_ureg-cost_MP)
        if(lambd>0):
            logE = logE + 0.5*d*NC*np.log(lambd)
        else:
            logE = logE + 0.5*d*NC*np.log(lambd+np.finfo(np.float).eps)

        logE = logE - 0.125*lambd*np.trace(np.matmul(B.T,np.matmul(np.linalg.pinv(AW),B)))
        logE = logE + 0.25*lambd*np.trace(np.matmul(B.T,B))

        logE_s[i] = logE

    return logE_s


def rkm_cost(X, W, s):
    """
    Regularized K-means for principal path, COST EVALUATION.
    (most stupid implementation)
    Args:
        [ndarray float] X: data matrix
        [ndarray float] W: waypoints matrix
        [float] s: regularization parameter
    Returns:
        [float] cost_km: K-means part of the cost
        [float] cost_reg: regularizer part of the cost
    """

    XW_dst = distance.cdist(X, W, 'sqeuclidean')
    u = XW_dst.argmin(1)

    cost_km = 0.0
    for i, x in enumerate(X):
        w = W[u[i], :]
        cost_km = cost_km + np.dot(x, x) + np.dot(w, w) - 2 * np.dot(x, w)

    cost_reg = 0.0
    for i, w in enumerate(W[0:-1, :]):
        w_nxt = W[i + 1, :]
        cost_reg = cost_reg + np.dot(w, w) + np.dot(w_nxt, w_nxt) - 2 * np.dot(w, w_nxt)
    cost_reg = s * cost_reg

    return cost_km, cost_reg
