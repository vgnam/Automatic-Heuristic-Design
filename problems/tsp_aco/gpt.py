import numpy as np
import random
import math
import scipy
import torch
def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the inverse of the distance matrix
    inv_distance = 1 / distance_matrix

    # Calculate the row-wise and column-wise averages
    row_avg = inv_distance.mean(axis=1)
    col_avg = inv_distance.mean(axis=0)

    # Calculate the ratio of the inverse distance to the row and column averages
    row_ratio = inv_distance / row_avg[:, np.newaxis]
    col_ratio = inv_distance / col_avg[np.newaxis, :]

    # Combine the factors
    combined = inv_distance * row_ratio * col_ratio

    # Calculate the adaptive threshold based on the 75th quantile
    threshold = np.quantile(combined, 0.75)

    # Sparsify the matrix by setting elements below the threshold to zero
    sparsified = np.where(combined > threshold, combined, 0)

    # Efficiently normalize the matrix by dividing by the sum of non-zero elements in each row
    row_sums = sparsified.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized = sparsified / row_sums

    return normalized
