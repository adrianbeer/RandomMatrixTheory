
import numpy as np
from numpy.linalg import eigh, inv, eigvals, eigh, eig, norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from fitter import Fitter


def generate_GOE(N=100, sigma=1):
    # Generates Gaussian Orthogonal Ensemble. rotationally invariant with diagonal elements have twice the variance as off-diagonal elements
    H = np.random.normal(loc=0, scale=np.sqrt(sigma**2/(2*N)), size=N*N).reshape((N, N))
    goe = (H + H.T)
    return goe


# Wigner Matrix NxN, i.e. real symmetric matrix with IID Gaussian random numbers with 
# zero mean and variance sigma^2/N 
def generate_Wigner(N, sigma):
    H = np.random.normal(loc=0, scale=1, size=N*N).reshape((N, N))
    X = sigma*(H + H.T) / np.sqrt(2*N)
    return X