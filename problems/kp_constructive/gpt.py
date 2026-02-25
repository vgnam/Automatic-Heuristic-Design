import numpy as np

def select_next_item_v2(remaining_capacity, weights, values):
    ratios = values / weights
    feasible_indices = np.where(weights <= remaining_capacity)[0]
    if len(feasible_indices) == 0:
        return -1

    normalized_ratios = (ratios[feasible_indices] - np.min(ratios[feasible_indices])) / (np.max(ratios[feasible_indices]) - np.min(ratios[feasible_indices]) + 1e-10)
    capacity_utilization = weights[feasible_indices] / remaining_capacity
    knapsack_efficiency = (values[feasible_indices] * np.log(weights[feasible_indices] + 1)) / weights[feasible_indices]
    long_term_utilization = np.sum(weights[feasible_indices] <= (remaining_capacity - weights[feasible_indices])) / len(feasible_indices)
    adaptive_pressure = 1 / (1 + np.exp(-5 * (remaining_capacity / np.sum(weights) - 0.5)))

    hybrid_scores = (
        normalized_ratios * (1 - adaptive_pressure) * 0.5 +
        capacity_utilization * adaptive_pressure * 0.3 +
        knapsack_efficiency * adaptive_pressure * 0.4 +
        long_term_utilization * adaptive_pressure * 0.2 +
        (values[feasible_indices] / np.sum(values)) * 0.4
    )

    best_index = feasible_indices[np.argmax(hybrid_scores)]
    updated_edge_distance = best_index if (remaining_capacity - weights[best_index]) >= 0 else -1
    return updated_edge_distance
