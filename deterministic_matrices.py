import numpy as np
from numpy.linalg import svd, eig, inv, eigvals
import matplotlib.pyplot as plt
from tqdm import tqdm

N = 100
N_samples = 100
sigmas = (0.1, 0.2, 0.3)


sigma = sigmas[0]

# M is a real symmetric orthogonal matrix with M = M.T = M^-1
def generate_M():
    H = np.random.normal(loc=0, scale=1, size=N*N).reshape((N, N))
    M = sigma*(H + H.T) / np.sqrt(2*N)
    D, P = eig(M)
    D = np.sign(D)
    M = P * D * inv(P)
    return M

# Wigner Matrix NxN, i.e. real symmetric matrix with IID Gaussian random numbers with 
# zero mean and variance sigma^2/N 
def generate_X(sigma):
    H = np.random.normal(loc=0, scale=1, size=N*N).reshape((N, N))
    X = sigma*(H + H.T) / np.sqrt(2*N)
    return X

fig, ax = plt.subplots(nrows=len(sigmas), ncols=2)
for sigma_idx, sigma in enumerate(sigmas):
    M = generate_M()
    eigs_list = []
    for i in tqdm(range(N_samples)):
        X = generate_X(sigma)
        E = M + X
        eigs = eigvals(M)
        if i == 0:
            ax[0, sigma_idx].hist(eigs, density=True, bins=30)
        eigs_list.append(eigs)

        # Plot Eigenvalues of E 
        ax[1, sigma_idx].hist(eigs_list, density=True, bins=30)
fig.savefig("plot1.pdf")