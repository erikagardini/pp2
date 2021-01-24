import numpy as np
from scipy.spatial import distance
from scipy.sparse import csgraph
from matplotlib import pyplot
import original_principalpath.linear_utilities as lu

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


def rkm_prefilter(X, boundary_ids, Nf=200, k=5, p=1000, T=0.1, plot_ax=None):
    """
    Regularized K-means for principal path, PREFILTER.

    Args:
        [ndarray float] X: data matrix

        [ndarray int] boundary_ids: start/end waypoints as sample indices

        [int] Nf: number of filter centroids

        [int] k: number of nearest neighbor for the penalized graph

        [float] p: penalty factor for the penalized graph

        [float] T: filter threshold

        [matplotlib.axis.Axes] plot_ax: Axes for the 2D plot (first 2 dim of X), None to avoid plotting

    Returns:
        [ndarray float] X_filtered

        [ndarray int] boundary_ids_filtered

        [ndarray float] X_garbage
    """

    #pick Nf medoids with k-means++ and compute pairwise distance matrix
    med_ids = lu.initMedoids(X, Nf-2, 'kpp', boundary_ids)
    med_ids = np.hstack([boundary_ids[0],med_ids,boundary_ids[1]])
    medmed_dst = distance.cdist(X[med_ids,:],X[med_ids,:],'sqeuclidean')

    #build k-nearest-neighbor penalized matrix
    knn_ids = np.argsort(medmed_dst,1)
    medmed_dst_p = medmed_dst.copy()*p
    for i in range(Nf):
        for j in range(k):
            k=knn_ids[i,j]
            medmed_dst_p[i,k] = medmed_dst[i,k]
            medmed_dst_p[k,i] = medmed_dst[k,i]
    medmed_dst_p[0,Nf-1]=0
    medmed_dst_p[Nf-1,0]=0

    #find shortest path using dijkstra
    [path_dst, path_pre] = csgraph.dijkstra(medmed_dst_p, False, 0,True)
    path=np.ndarray(0,int)
    i=Nf-1
    while(i != 0):
        path=np.hstack([i,path])
        i = path_pre[i]
    path=np.hstack([i,path])

    #filter out medoids too close to the shortest path
    T=T*np.mean(medmed_dst)

    to_filter_ids=np.ndarray(0,int)
    for i in path:
        to_filter_ids = np.hstack([np.where(medmed_dst[i,:]<T)[0], to_filter_ids])
    to_filter_ids = np.setdiff1d(to_filter_ids,path)
    to_filter_ids = np.unique(to_filter_ids)

    to_keep_ids = np.setdiff1d(np.asarray(range(Nf)),to_filter_ids)

    Xmed_dst = distance.cdist(X,X[med_ids[to_keep_ids],:],'sqeuclidean')
    u = med_ids[to_keep_ids][Xmed_dst.argmin(1)]

    N=X.shape[0]
    filter_mask = np.zeros(N,bool)
    for i in range(N):
        if u[i] in med_ids[path]:
            filter_mask[i]=True
    
    #convert boundary indices
    boundary_ids_filtered = boundary_ids.copy()
    boundary_ids_filtered[0] = boundary_ids[0] - boundary_ids[0] + np.sum(filter_mask[0:boundary_ids[0]])
    boundary_ids_filtered[1] = boundary_ids[1] - boundary_ids[1] + np.sum(filter_mask[0:boundary_ids[1]])

    #plot filter figure 
    if(plot_ax is not None):
        pyplot.sca(plot_ax)
        pyplot.ion()
        pyplot.plot(X[np.logical_not(filter_mask),0],X[np.logical_not(filter_mask),1],'yo',label='data filtered out')
        pyplot.plot(X[filter_mask,0],X[filter_mask,1],'bo',label='data kept')
        pyplot.plot(X[med_ids,0],X[med_ids,1],'ro',label='filter medoids')
        pyplot.plot(X[med_ids[to_filter_ids],0],X[med_ids[to_filter_ids],1],'kx',label='filter medoids dropped')
        pyplot.plot(X[med_ids[path],0],X[med_ids[path],1],'-go',label='filter shortest path')
        pyplot.plot(X[filter_mask,:][boundary_ids_filtered,0],X[filter_mask,:][boundary_ids_filtered,1],'mo',label='boundary samples')
        pyplot.legend()
        pyplot.axis('equal')

    return X[filter_mask,:], boundary_ids_filtered, X[np.logical_not(filter_mask),:]


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
