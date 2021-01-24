import numpy as np
import matplotlib.pyplot as plt

def model_selection(models, x, entries, filename="", y_mode='length', criterion='elbow', r2=None, r3=None, plot=False):
    if y_mode == 'length':
        y = rkm_MS_pathlen(models, x, entries)
    elif y_mode == 'var':
        y = rkm_MS_pathvar(models, x, entries)

    if criterion == 'elbow':
        f = np.stack([x, y], -1)
        elb_id = find_elbow(f)
    elif criterion == 'min':
        elb_id = np.argmin(y)

    if plot:
        plt.scatter(x, y)
        plt.plot(x, y, '-g')
        x_a = [x[0], x[-1]]
        y_a = [y[0], y[-1]]
        plt.plot(x_a, y_a, '-r')
        plt.scatter(x_a, y_a)
        plt.axvline(x[elb_id])
        plt.savefig(filename + "_" + str(y_mode) + "_" + str(criterion) + "_pp.png")
        plt.close()

    return [models[elb_id], x[elb_id]]

def plotPath(X, path, filename="", mode="scatter"):

    if mode == "scatter":
        plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(path[:, 0], path[:, 1])
        plt.plot(path[:, 0], path[:, 1], '-r')
        plt.scatter(path[0, 0], path[0, 1])
        plt.scatter(path[-1, 0], path[-1, 1])

        plt.savefig(filename + ".png")
        plt.close()
    elif mode == "figure":
        n = path.shape[0]
        plt.figure(figsize=(15, 2))
        for i in range(n):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(path[i].reshape(28, 28))
            #ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)
        plt.savefig(filename + ".png")
        plt.close()

def find_elbow(f):
    """
    Find the elbow in a function f, as the point on f with max distance from the line connecting f[0,:] and f[-1,:]

    Args:
        [ndarray float] f: function (Nx2 array in the form [x,f(x)])

    Returns:
        [int]  elb_id: index of the elbow
    """
    ps = np.asarray([f[0,0],f[0,1]])
    pe = np.asarray([f[-1,0],f[-1,1]])
    p_line_dst = np.ndarray(f.shape[0]-2,float)
    for i in range(1,f.shape[0]-1):
        p = np.asarray([f[i,0],f[i,1]])
        p_line_dst[i-1] = np.linalg.norm(np.cross(pe-ps,ps-p))/np.linalg.norm(pe-ps)
    elb_id = np.argmax(p_line_dst)+1

    return elb_id

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
    len_s=np.zeros(models.shape[0],float)

    for i,s in enumerate(s_span):
        W = models[i,:,:]
        NC = W.shape[0]-2
        for j,w in enumerate(W[0:-1,:]):
            w_nxt = W[j+1,:]
            len_s[i] = len_s[i] + np.sqrt(np.dot(w,w)+np.dot(w_nxt,w_nxt)-2*np.dot(w,w_nxt))

    return len_s


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