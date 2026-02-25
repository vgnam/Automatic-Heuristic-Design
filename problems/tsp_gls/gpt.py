import numpy as np
import random
import math
import scipy
import torch
def heuristics_v2(
    distance_matrix: np.ndarray,
    epsilon: float = 9.367943022608458e-06,
    percentile_threshold: float = 37.29708199134508,
    centrality_weight: float = 0.44001348839913285,
    diversification_weight: float = 0.35138673575704493,
    connectivity_weight: float = 0.1544234258898896) -> np.ndarray:
    """
    Combines distance, centrality, diversification, and connectivity with balanced weights and sophisticated normalization.
    Key idea: Weighted combination of factors with min-max scaling and balanced weights for effective guidance.
    """
    # Initialize heuristic matrix
    heuristic_matrix = np.zeros_like(distance_matrix)

    # Calculate average and standard deviation of distances
    avg_distances = np.mean(distance_matrix, axis=1)
    std_distances = np.std(distance_matrix, axis=1)

    # Calculate centrality, diversification, and connectivity factors
    centrality = 1 / (avg_distances + epsilon)
    diversification = 1 / (std_distances + epsilon)
    connectivity = np.sum(distance_matrix < np.percentile(distance_matrix, percentile_threshold), axis=1)

    # Normalize factors using min-max scaling
    centrality = (centrality - np.min(centrality)) / (np.max(centrality) - np.min(centrality) + epsilon)
    diversification = (diversification - np.min(diversification)) / (np.max(diversification) - np.min(diversification) + epsilon)
    connectivity = (connectivity - np.min(connectivity)) / (np.max(connectivity) - np.min(connectivity) + epsilon)
    normalized_distance_matrix = (distance_matrix - np.min(distance_matrix)) / (np.max(distance_matrix) - np.min(distance_matrix) + epsilon)

    # Calculate heuristic values with balanced weights
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                heuristic_matrix[i, j] = normalized_distance_matrix[i, j] * (1 + (centrality[i] + centrality[j]) / 2 * centrality_weight) * (1 + (diversification[i] + diversification[j]) / 2 * diversification_weight) * (1 + (connectivity[i] + connectivity[j]) / 2 * connectivity_weight)

    # Normalize heuristic matrix to ensure valid probability distribution
    heuristic_matrix = heuristic_matrix / (np.sum(heuristic_matrix) + epsilon)

    return heuristic_matrix