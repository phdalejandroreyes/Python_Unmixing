import numpy as np
from scipy.sparse.linalg import svds
import scipy.linalg as splin
from scipy.linalg import  pinv, eigh

from scipy.linalg import svd,  sqrtm
from joblib import Parallel, delayed

def pca(X,d):
    L, N = X.shape
    xMean = X.mean(axis=1).reshape((L,1),order='F')
    xzm = xMean - np.tile(xMean, (1, N))
    U, _ , _  = svds( (xzm @ xzm.T)/N , k=d)
    return U

def NFINDR(Y, N):
    """
    N-FINDR endmembers estimation in multi/hyperspectral dataset
    """
    L, K = Y.shape
    # dimention redution by PCA
    U = pca(Y,N)
    Yr = U.T @ Y
    # Initialization
    Po = np.zeros((L,N))
    IDX = np.zeros((1,K))
    TestM = np.zeros((N,N))
    TestM[0,:]=1
    for i in range (N):
        idx = np.floor(float(np.random.rand(1))*K) + 1
        TestM[1:N,i]= Yr[:N-1,i].copy()
        IDX[0,i]=idx
    actualVolume = np.abs(np.linalg.det(TestM))
    it=1
    v1=-1
    v2=actualVolume
    #  Algorithm
    maxit = 3 * N
    while (it<maxit and v2>v1):
        for k in range (N):
            for i in range (K):
                actualSample = TestM[1:N,k].copy()
                TestM[1:N,k] = Yr[:N-1,i].copy()
                volume = np.abs(np.linalg.det(TestM))
                if volume > actualVolume:
                    actualVolume = volume.copy()
                    IDX[0,k] = i
                else:
                    TestM[1:N,k]=actualSample.copy()
        it = it + 1
        v1 = v2
        v2 = actualVolume.copy()
    
    for i in range (N):
        Po[:,i] = Y[:,int(IDX[0,i])].copy()
    return Po

def vca(Y,R):
    #############################################
    # Initializations
    #############################################
    [L, N]=Y.shape   # L number of bands (channels), N number of pixels     
    R = int(R)
    #############################################
    # SNR Estimates
    #############################################  
    y_m = np.mean(Y,axis=1,keepdims=True)
    Y_o = Y - y_m           # data with zero-mean
    Ud  = splin.svd(np.dot(Y_o,Y_o.T)/float(N))[0][:,:R]  # computes the R-projection matrix 
    x_p = np.dot(Ud.T, Y_o)                 # project the zero-mean data onto p-subspace
    P_y     = np.sum(Y**2)/float(N)
    P_x     = np.sum(x_p**2)/float(N) + np.sum(y_m**2)
    SNR = 10*np.log10( (P_x - R/L*P_y)/(P_y - P_x) ) 
    SNR_th = 15 + 10*np.log10(R)+8
    
    #############################################
    # Choosing Projective Projection or 
    #          projection to p-1 subspace
    #############################################
    if SNR < SNR_th:
        d = R-1
        Ud = Ud[:,:d].copy()
        Yp =  np.dot(Ud,x_p[:d,:]) + y_m      # again in dimension L
        x = x_p[:d,:].copy() #  x_p =  Ud.T * Y_o is on a R-dim subspace
        c = np.argmax(np.sum(x**2,axis=0))**0.5
        y = np.vstack(( x, c*np.ones((1,N))))
    else:
        d=R
        Ud  = splin.svd(np.dot(Y,Y.T)/float(N))[0][:,:d] # computes the p-projection matrix 
        x_p = np.dot(Ud.T,Y)
        Yp =  np.dot(Ud,x_p[:d,:])      # again in dimension L (note that x_p has no null mean)
        x =  np.dot(Ud.T,Y)
        u =  np.mean(x,axis=1,keepdims=True)        #equivalent to  u = Ud.T * r_m
        y =  x / np.dot(u.T,x)
    #############################################
    # VCA algorithm
    #############################################
            
    indice = np.zeros((R),dtype=int)
    A = np.zeros((R,R))
    A[-1,0] = 1

    for i in range(R):
        w = np.random.rand(R,1);   
        f = w - np.dot(A,np.dot(splin.pinv(A),w))
        f = f / splin.norm(f)
        v = np.dot(f.T,y)
        indice[i] = np.argmax(np.abs(v))
        A[:,i] = y[:,indice[i]].copy()        # same as x(:,indice(i))
    Ae = Yp[:,indice].copy()
    Ae[Ae < 0] = 0
    return Ae

def SVMAX(X, N):
    """
    An implementation of SVMAX algorithm for endmember estimation.
    
    Parameters:
    X : numpy.ndarray
        Data matrix where each column represents a pixel and each row represents a spectral band.
    N : int
        Number of endmembers to estimate.
        
    Returns:
    A_est : numpy.ndarray
        Estimated endmember signatures (or mixing matrix).
    """
    M, L = X.shape
    d = np.mean(X, axis=1, keepdims=True)
    U = X - np.outer(d, np.ones((1, L)))
    C = eigh(U @ U.T, lower=False, subset_by_index=(M-N+1, M-1))[1]
    Xd_t = C.T @ U;
    
    A_set = np.zeros((N,N))
    
    index = np.zeros(3); 
    P = np.eye(N);    
    
                         
    Xd = np.vstack((Xd_t, np.ones((1, L))))
    
    
    for i in range(N):
        ind = np.argmax((np.sum((abs(P@Xd))**2,axis=0,keepdims=True))**(1/2));
        A_set[:,i] = Xd[:, ind]
        P = np.eye(N) - A_set @ pinv(A_set);
        index[i] = ind;
    Po = C @ Xd_t[:,index.astype(int)]+d @np.ones((1,N));
    
    
    return Po