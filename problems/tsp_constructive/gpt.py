def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if not unvisited_nodes:
        return destination_node

    def calculate_subpath_cost(start, end, remaining_nodes, dist_matrix):
        if not remaining_nodes:
            return dist_matrix[start][end]
        remaining = set(remaining_nodes)
        current = start
        cost = 0.0
        while remaining:
            next_node = min(remaining, key=lambda node: dist_matrix[current][node])
            cost += dist_matrix[current][next_node]
            remaining.remove(next_node)
            current = next_node
        cost += dist_matrix[current][end]
        return cost

    def evaluate_node(node):
        immediate_cost = distance_matrix[current_node][node]
        remaining = unvisited_nodes - {node}

        if not remaining:
            return immediate_cost

        subpath_cost = calculate_subpath_cost(node, destination_node, remaining, distance_matrix)
        density_factor = 1.0 / (len(unvisited_nodes) + 1)
        total_cost = immediate_cost * 0.7 + subpath_cost * (0.3 + 0.7 * density_factor)

        if hasattr(evaluate_node, 'q_values'):
            evaluate_node.q_values[node] = evaluate_node.q_values.get(node, total_cost) * 0.8 + total_cost * 0.2
            total_cost = evaluate_node.q_values[node]
        else:
            evaluate_node.q_values = {}

        exploration_factor = 0.4 * density_factor
        return total_cost * (1 - exploration_factor)

    def local_search_improvement(node, remaining, dist_matrix):
        if len(remaining) < 2:
            return node
        best_node = node
        best_cost = evaluate_node(node)
        candidates = sorted(remaining, key=lambda x: dist_matrix[current_node][x])[:min(10, len(remaining))]
        for candidate in candidates:
            cost = evaluate_node(candidate)
            if cost < best_cost:
                best_cost = cost
                best_node = candidate
        return best_node

    next_node = min(unvisited_nodes, key=evaluate_node)
    next_node = local_search_improvement(next_node, unvisited_nodes - {next_node}, distance_matrix)
    return next_node