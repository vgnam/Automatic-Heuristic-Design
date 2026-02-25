import numpy as np

def update_edge_distance_v2(edge_distance: np.ndarray, local_opt_tour: np.ndarray, edge_n_used: np.ndarray) -> np.ndarray:
    """
    Args:
        edge_distance (np.ndarray): Original edge distance matrix.
        local_opt_tour (np.ndarray): Local optimal solution path.
        edge_n_used (np.ndarray): Matrix representing the number of times each edge is used.
    Return:
        updated_edge_distance: updated score of each edge distance matrix.
    """
    num_nodes = edge_distance.shape[0]
    updated_edge_distance = np.copy(edge_distance)

    # Precompute factors
    max_usage = np.max(edge_n_used)
    if max_usage > 0:
        edge_n_used_normalized = edge_n_used / max_usage
    else:
        edge_n_used_normalized = edge_n_used

    # Identify edges in the local optimal tour
    tour_edges = np.zeros_like(edge_n_used)
    for i in range(num_nodes):
        current_node = local_opt_tour[i]
        next_node = local_opt_tour[(i + 1) % num_nodes]
        tour_edges[current_node, next_node] = 1

    # Calculate the average usage of edges in the local optimal tour
    avg_tour_usage = np.sum(edge_n_used_normalized * tour_edges) / num_nodes

    # Apply targeted adjustments to tour edges
    for i in range(num_nodes):
        current_node = local_opt_tour[i]
        next_node = local_opt_tour[(i + 1) % num_nodes]

        # Update the edge distance based on the number of times it has been used, the average usage, and the edge position
        usage_factor = 1 + 2 * edge_n_used_normalized[current_node, next_node]
        position_factor = (i + 1) / num_nodes
        tour_usage_factor = 1 + 2 * (avg_tour_usage - edge_n_used_normalized[current_node, next_node])
        updated_edge_distance[current_node, next_node] *= usage_factor * (1 + 4 * position_factor) * tour_usage_factor

    # Apply non-linear penalties to non-tour edges with a stronger emphasis
    non_tour_edges = 1 - tour_edges
    penalty_factor = 1 + 2 * edge_n_used_normalized * non_tour_edges
    updated_edge_distance *= penalty_factor

    # Introduce a moderate random perturbation to encourage exploration
    perturbation = np.random.normal(0, 0.01, size=edge_distance.shape)
    updated_edge_distance = np.maximum(updated_edge_distance + perturbation, 0)

    # Normalize the edge distances to maintain the relative importance of edges
    max_distance = np.max(updated_edge_distance)
    if max_distance > 0:
        updated_edge_distance = updated_edge_distance / max_distance

    # Apply a local adjustment based on the average number of edge usages in the tour
    local_factor = 1 + 2 * avg_tour_usage
    updated_edge_distance *= local_factor

    # Apply a distance-based adjustment to prioritize shorter edges
    distance_factor = 1 / (1 + 2 * edge_distance)
    updated_edge_distance *= distance_factor

    return updated_edge_distance