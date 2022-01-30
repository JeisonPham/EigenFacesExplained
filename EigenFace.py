import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class EigenFace:
    def __init__(self, data_file, variance):
        self.data = np.load(data_file)
        self.eigenvalues, self.eigenvectors, self.mu = self.pca(variance)
        self.eigenfaces = normalize(self.data @ self.eigenvectors, axis=0)

    def preserve_variance(self, eigenvalues, variance=0.95):
        for i, cumsum in enumerate(np.cumsum(eigenvalues) / np.sum(eigenvalues)):
            if cumsum > variance:
                return i

    def pca(self, variance):
        X = self.data
        [n, d] = X.shape

        mu = X.mean(axis=1).reshape(-1, 1)
        X = X - mu

        cov = np.matmul(X.T, X)
        [eigenvalues, eigenvectors] = np.linalg.eigh(cov)

        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        num_components = self.preserve_variance(eigenvalues, variance)

        eigenvalues = eigenvalues[:num_components].copy()
        eigenvectors = eigenvectors[:, :num_components].copy()

        return eigenvalues, eigenvectors, mu

    def projection(self, image):
        image = normalize(image.reshape(-1, 1))

        return np.matmul(image.T, self.data - self.mu)

    def reconstruction(self, projection):
        return np.matmul(self.eigenfaces, projection) + self.mu

    def subplot(self, title, image_idx, rows, cols, filename=None, figsize=(10, 10)):
        fig = plt.figure(figsize=figsize)
        fig.text(0.5, 0.95, title, horizontalalignment="center")

        for i in range(rows * cols):
            ax0 = fig.add_subplot(rows, cols, (i + 1))
            plt.setp(ax0.get_xticklabels(), visible=False)
            plt.setp(ax0.get_yticklabels(), visible=False)

            plt.imshow(np.asarray(self.eigenfaces[:, image_idx[i]].reshape(500, 500)), cmap="gray")

        if filename is not None:
            fig.savefig(filename)
        plt.show()
