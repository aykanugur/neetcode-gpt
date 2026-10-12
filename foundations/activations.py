import numpy as np
from numpy.typing import NDArray


class Solution:

    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Formula: 1 / (1 + e^(-z))
        answer = 1 / (1 + np.exp(-z))
        return np.round(answer, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Formula: max(0, z) element-wise
        answer = np.maximum(0, z)
        return np.round(answer, 5)