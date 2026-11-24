import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (rows = n_samples, columns = n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        # X = np.asarray(X, dtype = np.float64)
        # y = np.array(y,dtype = np.float64 )
        n_samples, n_features = X.shape 
        w = np.zeros(n_features, dtype = np.float64)
        b = 0.0
        for _ in range(epochs):
            y_hat = X @ w  + b 
            error = y_hat -y 
            # Gradient of MSE: (2/N) * X^T (y_hat - y) and (2/N) * sum(y_hat - y)
            grad_w = (2.0/n_samples)*(X.T @ error)
            grad_b = (2.0/ n_samples)*(np.sum(error))

            w = w - lr *grad_w
            b = b - lr* grad_b 
        return (np.round(w, 5), np.round(b, 5))