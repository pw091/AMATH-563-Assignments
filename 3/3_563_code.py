import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from sklearn.datasets import make_blobs

#constants
m = 2048
C = 1
epsilon = C * np.log(m) ** (3/4) / np.sqrt(m)

#random points
np.random.seed(0)
X = np.random.rand(m, 2)

#sparse
def create_weight_matrix(X, epsilon):
    distances = cdist(X, X)
    W = np.exp(-distances ** 2 / epsilon ** 2)
    np.fill_diagonal(W, 0)
    return W

def create_laplacian(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L

W = create_weight_matrix(X, epsilon)
L = create_laplacian(W)
evals, evecs = eigsh(L, k=4, which='SM')

def Q1(plot_3d: bool):
    if plot_3d:
        fig = plt.figure(figsize=(10, 10))
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            sc = ax.scatter(X[:, 0], X[:, 1], evecs[:, i], c=evecs[:, i], cmap='coolwarm')
            ax.set_title(f'Eigenvector {i + 1}')
            ax.set_xlabel('x-axis')
            ax.set_ylabel('y-axis')
            ax.set_zlabel('Eigenvector value')
            fig.colorbar(sc, ax=ax)
        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.ravel()
        for i in range(4):
            ax = axes[i]
            cont = ax.tricontourf(X[:, 0], X[:, 1], evecs[:, i], cmap='coolwarm', levels=20)
            ax.set_title(f'Eigenvector {i + 1}')
            ax.set_xlabel('x-axis')
            ax.set_ylabel('y-axis')
            fig.colorbar(cont, ax=ax)
        plt.tight_layout()
        plt.show()
    return

#Q1(plot_3d=True)
#Q1(plot_3d=False)

def Q2(plot_3d: bool):
    #Neumann eval solutions for L
    psi = lambda x,n,k: np.cos(n * np.pi * x[:, 0]) * np.cos(k * np.pi * x[:, 1])

    #first 4 integers of vectorized psi (wrt X)
    psi_tilde = np.zeros((m, 4))
    psi_tilde[:, 0] = psi(X, 0, 0)
    psi_tilde[:, 1] = psi(X, 1, 0)
    psi_tilde[:, 2] = psi(X, 0, 1)
    psi_tilde[:, 3] = psi(X, 1, 1)

    #normalize
    psi_normal = psi_tilde / np.linalg.norm(psi_tilde, axis=0)

    if plot_3d:
        fig = plt.figure(figsize=(10, 10))
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            sc = ax.scatter(X[:, 0], X[:, 1], psi_normal[:, i], c=psi_normal[:, i], cmap='coolwarm')
            ax.set_title(f'Normalized Eigenfunction {i+1}')
            ax.set_xlabel('x-axis')
            ax.set_ylabel('y-axis')
            ax.set_zlabel('Eigenfunction value')
            fig.colorbar(sc, ax=ax)
        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.ravel()
        for i in range(4):
            ax = axes[i]
            cont = ax.tricontourf(X[:, 0], X[:, 1], psi_normal[:, i], cmap='coolwarm', levels=20)
            ax.set_title(f'Normalized Eigenfunction {i + 1}')
            ax.set_xlabel('x-axis')
            ax.set_ylabel('y-axis')
            fig.colorbar(cont, ax=ax)
        plt.tight_layout()
        plt.show()
    return

#Q2(plot_3d=True)
#Q2(plot_3d=False)

def Q3():
    psi = lambda x,n,k: np.cos(n * np.pi * x[:, 0]) * np.cos(k * np.pi * x[:, 1])
    m_vals = [2**i for i in range(4,11)]
    mean_err_vals = []
    for m in m_vals:
        m_errs = []
        for trial in range(30):
            #random points
            np.random.seed(0)
            X = np.random.rand(m, 2)
            #eigendecomp
            epsilon = C * np.log(m) ** (3/4) / np.sqrt(m)
            distances = cdist(X, X, 'euclidean')
            W = np.where(distances <= epsilon, (np.pi * epsilon ** 2) ** -1, 0)
            # Sparse
            W_sparse = csr_matrix(W)
            D_sparse = csr_matrix(np.diag(np.sum(W, axis=1)))
            L_sparse = D_sparse - W_sparse
            _, evecs = eigsh(L_sparse, k=4, which='SM')
            #vectorized eigenfuncs
            psi_tilde = np.zeros((m, 4))
            psi_tilde[:, 0] = psi(X, 0, 0)
            psi_tilde[:, 1] = psi(X, 1, 0)
            psi_tilde[:, 2] = psi(X, 0, 1)
            psi_tilde[:, 3] = psi(X, 1, 1)
            psi_normalized = psi_tilde / np.linalg.norm(psi_tilde, axis=0)
            #projectors
            proj_Q = evecs @ evecs.T
            proj_Psi = psi_normalized @ psi_normalized.T
            #error
            trial_err = np.linalg.norm(proj_Q@proj_Psi - proj_Psi@proj_Q, ord='fro')
            m_errs.append(trial_err)
        mean_err_vals.append(np.mean(m_errs))
    plt.loglog(m_vals, mean_err_vals, marker='o')
    plt.xlabel('m')
    plt.ylabel('Average Error')
    plt.title('Average Error vs. m')
    plt.show()
    return
#Q3()

#L-shaped domain
def L_domain(m):
    points = []
    while len(points) < m:
        point = 2 * np.random.rand(2)
        if not (1 <= point[0] <= 2 and 1 <= point[1] <= 2):
            points.append(point)
    return np.array(points)

def Q4(plot_3d: bool):
    m = 2**13
    X = L_domain(m)

    epsilon = np.log(m) ** (3 / 4) / np.sqrt(m)
    distances = cdist(X, X, 'euclidean')
    W = np.where(distances <= epsilon, (np.pi * epsilon ** 2) ** -1, 0)
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    L_sparse = csr_matrix(L)
    _, evecs = eigsh(L_sparse, k=10, which='SM')

    if plot_3d:
        fig = plt.figure(figsize=(10, 10))
        for i in range(4):
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')
            sc = ax.scatter(X[:, 0], X[:, 1], evecs[:, i + 6], c=evecs[:, i + 6], cmap='coolwarm')
            ax.set_title(f'Eigenvector {i + 7}')
            ax.set_xlabel('x-axis')
            ax.set_ylabel('y-axis')
            ax.set_zlabel('Eigenvector value')
            fig.colorbar(sc, ax=ax)
        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.ravel()
        for i in range(4):
            ax = axes[i]
            cont = ax.tricontourf(X[:, 0], X[:, 1], evecs[:, i + 6], cmap='coolwarm', levels=20)
            ax.set_title(f'Eigenvector {i + 7}')
            ax.set_xlabel('x-axis')
            ax.set_ylabel('y-axis')
            fig.colorbar(cont, ax=ax)
        plt.tight_layout()
        plt.show()
    return

#Q4(plot_3d=True)
#Q4(plot_3d=False)
