import math
import numpy as np
from sklearn import preprocessing
from scipy.sparse import csc_matrix, csr_matrix
from scipy.linalg import qr
from scipy.sparse import diags
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import as_float_array

import warnings
warnings.filterwarnings("ignore")

np.get_include()

def orthNMF(X, F, G, T=10):
    print("orthogonal non-negative matrix factorization")
    for t in range(T):
        XTF = X.T.dot(F)
        GFTF = G.dot( F.T.dot(F) )
        GFTF[GFTF.nonzero()] = 1.0 / GFTF[GFTF.nonzero()]
        XTF = XTF.multiply(GFTF)
        XTF[XTF<0]=0
        G = G.multiply(XTF)

        XG = X.dot(G)
        FFTXG = F.dot( F.T.dot(XG) )
        FFTXG[FFTXG.nonzero()] = 1.0/FFTXG[FFTXG.nonzero()]
        XG = XG.multiply(FFTXG)
        XG[XG<0]=0
        XG.data = np.sqrt(XG.data)
        F = F.multiply(XG)

    return F

def approxGK(X):
    print("approximate Gaussian kernel")
    d = X.shape[1]
    W = np.random.standard_normal(size=(d, d))
    W, _ = qr(W, mode='economic')
    X = X.dot(W.T)
    X_cos = np.cos(X)
    X_sin = np.sin(X)

    X = csr_matrix(np.hstack([X_sin,X_cos])*math.sqrt(math.exp(1)/1.0*d))

    xsum = X.sum(axis=0)
    deg = 1.0/np.sqrt(X.dot(xsum.T))
    X = csr_matrix(X.multiply(deg))

    return X

def getL(W):
    c = np.array(np.sqrt(W.sum(axis=0)))
    c[c==0]=1
    c = 1.0/c
    c = c.flatten().tolist()
    c = diags(c)

    r = np.array(np.sqrt(W.sum(axis=1)))
    r[r==0]=1
    r_wo_inv = diags(r.flatten().tolist())
    r = 1.0/r
    r = diags(r.flatten().tolist())

    return csr_matrix(r.dot(W.dot(c))), r_wo_inv

def trunc_propagate(B, X, alpha, T=5):
    print("feature propagation")
    Xt = X.copy()
    for i in range(T):
        Xt = X + alpha*B.dot(B.T.dot(Xt))

    X = preprocessing.normalize(Xt, norm='l2', axis=1)

    return X

def SVD_init(X, k):
    print("SVD initialization")
    U, s, V = randomized_svd(X, n_components=k)
    U = csr_matrix(U)

    V = csr_matrix(V).T.dot(diags(np.array(s)))

    return U, V

def NCI_gen(vectors, T=100):
    vectors = np.asarray(vectors, dtype=np.float64)
    n_samples = vectors.shape[0]
    n_feats = vectors.shape[1]

    labels = vectors.argmax(axis=1)
    print(type(labels), labels.shape)
    vectors_discrete = csc_matrix(
            (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
            shape=(n_samples, n_feats))

    vectors_sum = np.sqrt(vectors_discrete.sum(axis=0))
    vectors_sum[vectors_sum==0]=1
    vectors_discrete = vectors_discrete*1.0/vectors_sum

    for _ in range(T):
        Q = vectors.T @ vectors_discrete

        vectors_discrete = vectors @ Q 
        vectors_discrete = as_float_array(vectors_discrete)

        labels = vectors_discrete.argmax(axis=1)
        vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_feats))

        vectors_sum = np.sqrt(vectors_discrete.sum(axis=0))
        vectors_sum[vectors_sum==0]=1
        vectors_discrete = vectors_discrete*1.0/vectors_sum

    return labels

def TPC(B, X, alpha, dim, k, gamma, tf, tg):
    B, r = getL(B)

    if dim>0 and dim<X.shape[1]:
        print("dimensionality reduction")
        X, s, _ = randomized_svd(X, n_components=dim)
        X = csr_matrix(X).dot(diags(np.array(s)))

    X = trunc_propagate(B, X, alpha, gamma)

    X = approxGK(X)

    U, V = SVD_init(X, k)

    U = orthNMF(X, U, V, tf)

    U = U.todense()
    labels = NCI_gen(U, tg)

    return labels