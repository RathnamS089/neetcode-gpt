import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        N: int,
        X: NDArray[np.float64],
        desired_weight: int,
    ) -> float:

        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(
        self, X: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        num_features = X.shape[1]
        while num_iterations > 0:
            out = self.get_model_prediction(X, initial_weights)
            dell = np.zeros(num_features)
            for i in range(num_features):
                dell[i] = self.get_derivative(out, Y, X.shape[0], X, i)
            initial_weights = initial_weights - self.learning_rate * dell
            num_iterations -= 1
        return np.round(initial_weights, 5)
        pass
