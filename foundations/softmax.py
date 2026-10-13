import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Numerical stability için en büyük değeri çıkarıyoruz
        shifted_z = z - np.max(z)

        exp_z = np.exp(shifted_z)

        answer = exp_z / np.sum(exp_z)

        return np.round(answer, 4)