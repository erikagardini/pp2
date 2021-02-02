import numpy as np
from scipy.spatial import distance

def initMedoids(X, n, init_type, exclude_ids=[]):
    """
    Initialize NC medoids with init_type rational.

    Args:
        [ndarray float] X: data matrix

        [int] n: number of medoids to be selected

        [string] init_type: rational to be used
            'uniform': randomly selected with uniform distribution
            'kpp': k-means++ algorithm

        [ndarray int] exclude_ids: blacklisted ids that shouldn't be selected

    Returns:
        [ndarray int] med_ids: indices of the medoids selected
    """

    N=X.shape[0]
    D=X.shape[1]
    med_ids=-1*np.ones(n,int)

    if(init_type=='uniform'):
        while(n>0):
            med_id = np.random.randint(0,N)
            if(np.count_nonzero(med_ids==med_id)==0 and np.count_nonzero(exclude_ids==med_id)==0):
                med_ids[n-1]=med_id
                n = n-1

    elif(init_type=='kpp'):
        accepted = False
        while(not accepted):
            med_id = np.random.randint(0,N)
            if(np.count_nonzero(exclude_ids==med_id)==0):
                accepted = True
        med_ids[0]=med_id

        for i in range(1,n):
            Xmed_dst = distance.cdist(X,np.vstack([X[med_ids[0:i],:],X[exclude_ids,:]]),'sqeuclidean')
            D2 = Xmed_dst.min(1)
            D2_n = 1.0/np.sum(D2)
            accepted = False
            while(not accepted):
                med_id = np.random.randint(0,N)
                if(np.random.rand()<D2[med_id]*D2_n):
                    accepted = True
            med_ids[i]=med_id
    else:
        raise ValueError('init_type not recognized.')

    return(med_ids)