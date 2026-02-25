from complexipy import code_complexity
import math
import numpy as np
from population_encode import remove_comments_and_docstrings, format_python_code

code = '''
import scipy
import torch

def select_next_node_v2(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    """Select the next node to visit from the unvisited nodes."""
    threshold = 0.7
    c1, c2, c3, c4 = 0.4, 0.3, 0.2, 0.1
    scores = {}
    for node in unvisited_nodes:
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_distance_to_unvisited = np.mean(all_distances)
        std_dev_distance_to_unvisited = np.std(all_distances)
        score = c1 * distance_matrix[current_node][node] - c2 * average_distance_to_unvisited + c3 * std_dev_distance_to_unvisited - c4 * distance_matrix[destination_node][node]
        scores[node] = score
    next_node = min(scores, key=scores.get)
    return next_node
'''

# Phân tích độ phức tạp
result = code_complexity(format_python_code(remove_comments_and_docstrings(code)))
print(f"Time Complexity:", result.complexity)