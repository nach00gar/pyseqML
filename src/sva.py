import pandas as pd
import numpy as np
from scipy.linalg import solve, eigh, eig, svd
from scipy.stats import mode
from scipy.stats import f
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.kde import KDEUnivariate


def batchEffectRemoval(expressionMatrix, labels):
    if len(np.unique(labels)) == 2:
        mod = np.vstack((np.ones(labels.shape), pd.get_dummies(labels, dtype="int").iloc[:, 1])).T
    else:
        mod = pd.get_dummies(labels, dtype="int")
        mod.columns = np.unique(labels)

    mod0 = np.ones(labels.shape)

    n_sv = num_sv(expressionMatrix, mod)
    res = irwsva_build(expressionMatrix, mod, mod0, n_sv, B=5)

    ndb = expressionMatrix.shape[1]
    nmod = mod.shape[1]
    mod1 = np.hstack([mod, res["sv"]])
    gammahat = (expressionMatrix @ mod1 @ np.linalg.inv(mod1.T @ mod1)).iloc[:, nmod:(nmod + n_sv)]
    modeled = gammahat @ res["sv"].T
    modeled.columns = expressionMatrix.columns
    expressionMatrixBatchCorrected = expressionMatrix - modeled

    return expressionMatrixBatchCorrected



def num_sv(pre, mod):
    def row_vars(matrix):
        return np.var(matrix, axis=1, ddof=1)

    dat = pre.to_numpy()
    dims = dat.shape
    a = np.linspace(0, 2, 100)
    n = dims[0] // 10
    rhat = np.zeros((100, 10))
    P = np.eye(dims[1]) - mod @ solve(mod.T @ mod, mod.T)

    for j in range(1, 11):
        dats = dat[:j*n, :]
        ee = eigh(dats.T @ dats, eigvals_only=True)[::-1]
        sigbar = ee[dims[1] - 1] / (j * n)
        R = dats @ P
        wm = (1 / (j * n)) * (R.T @ R) - P * sigbar
        ee = eigh(wm, eigvals_only=True)[::-1]
        v = np.concatenate((np.ones(100, dtype=bool), np.zeros(dims[1], dtype=bool)))
        v = v[np.argsort(np.concatenate((a * (j * n) ** (-1 / 3) * dims[1], ee)))[::-1]]
        u = np.arange(1, len(v) + 1)
        w = np.arange(1, 101)
        rhat[:, j - 1] = np.flip(u[v] - w)

    ss = row_vars(rhat)

    bumpstart = np.argmax(ss > (2 * ss[0]))
    start = np.argmax(np.concatenate((np.full(bumpstart, 1e5), ss[bumpstart + 1:100])) < 0.5 * ss[0])
    finish = np.argmax(ss * np.concatenate((np.zeros(start), np.ones(100 - start))) > ss[0])
    if finish == 0:
        finish = 100
        
    n_sv = mode(rhat[start:finish+1, 9])
    print(n_sv[0])
    return(int(n_sv[0]))


def f_pvalue(dat, mod, mod0):
    n = dat.shape[1]
    m = dat.shape[0]
    df1 = mod.shape[1]
    df0 = mod0.shape[1]
    Id = np.eye(n)
    
    resid = dat @ (Id - mod @ solve(mod.T @ mod, mod.T))
    rss1 = np.sum(resid**2, axis=1)
    del resid
    
    resid0 = dat @ (Id - mod0 @ solve(mod0.T @ mod0, mod0.T))
    rss0 = np.sum(resid0**2, axis=1)
    del resid0
    
    fstats = ((rss0 - rss1) / (df1 - df0)) / (rss1 / (n - df1))
    
    p = 1 - f.cdf(fstats, dfn=(df1 - df0), dfd=(n - df1))
    
    return p

def mono(lfdr):
    for i in range(1, len(lfdr)):
        if lfdr[i] < lfdr[i - 1]:
            lfdr[i] = lfdr[i - 1]
    return lfdr

def edge_lfdr(p, trunc=True, monotone=True, adj=1.5, eps=1e-8, lambda_val=0.8):
    pi0 = np.mean(p >= lambda_val) / (1 - lambda_val)
    pi0 = min(pi0, 1)

    n = len(p)
    
    p = np.clip(p, eps, 1 - eps)
    x = norm.ppf(p)
    bw = bw_nrd0(x)
    
    myd = KDEUnivariate(x)
    myd.fit(kernel="gau", bw=bw, adjust=adj, gridsize=512)
    mys = UnivariateSpline(myd.support, myd.density, s=0)
    y = mys(x)
    lfdr = pi0 * norm.pdf(x) / y

    if trunc:
        lfdr = np.minimum(lfdr, 1)
        
    if monotone:
        sorted_indices = np.argsort(p)
        sorted_lfdr = np.sort(lfdr)
        lfdr[sorted_indices] = mono(sorted_lfdr)
        
    return lfdr


def bw_nrd0(x):

    if len(x) < 2:
        raise(Exception("need at least 2 data points"))

    hi = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    lo = min(hi, iqr/1.34)

    if not ((lo == hi) or (lo == abs(x[0])) or (lo == 1)):
        lo = 1

    return 0.9 * lo *len(x)**-0.2



def irwsva_build(dat, mod, mod0=None, n_sv=None, B=10):
    n = dat.shape[1]
    m = dat.shape[0]
    n_sv = int(n_sv)
    
    if mod0 is None:
        mod0 = mod[:, [0]]
    
    Id = np.eye(n)
    resid = np.dot(dat, (Id - mod @ solve(mod.T @ mod, mod.T)))
    eigenvalues, eigenvectors = eig(resid.T @ resid)


    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    uu = sorted_eigenvalues
    vv = sorted_eigenvectors
    ndf = n - mod.shape[1]
    
    pprob = np.ones(m)
    one = np.ones(n)
    Id = np.eye(n)
    df1 = mod.shape[1] + n_sv
    df0 = 1 + n_sv #mod0.shape[1] + n_sv
    
    print(f"Iteration (out of {B}):", end=" ")
    for i in range(B):
        mod_b = np.hstack((mod, vv[:, :n_sv]))
        mod0_b = np.hstack((mod0.reshape(-1, 1), vv[:, :n_sv]))
        ptmp = f_pvalue(dat, mod_b.astype(np.float64), mod0_b.astype(np.float64))
        pprob_b = 1 - edge_lfdr(ptmp)
        
        mod_gam = np.hstack((mod0.reshape(-1, 1), vv[:, :n_sv]))
        mod0_gam = mod0.reshape(-1, 1)
        ptmp = f_pvalue(dat, mod_gam.astype(np.float64), mod0_gam.astype(np.float64))
        pprob_gam = 1 - edge_lfdr(ptmp)
        
        pprob = pprob_gam * (1 - pprob_b)
        dats = dat * pprob[:, None]
        dats -= np.mean(np.array(dats), axis=1)[:, None]
        eigenvalues, eigenvectors = eig(dats.T @ dats)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        uu = sorted_eigenvalues
        vv = sorted_eigenvectors
        print(f"{i + 1} ", end="")
    
    sv = svd(dats, full_matrices=False)[2][:, :n_sv]
    retval = {
        'sv': sv,
        'pprob_gam': pprob_gam,
        'pprob_b': pprob_b,
        'n_sv': n_sv
    }
    return retval
