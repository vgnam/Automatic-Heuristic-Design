import multiprocessing
import sys
from typing import Any, Callable

sys.path.append('../')
from baselines.eoh.gls_tsp_adapt import _evaluator_accelerate
from baselines.eoh.gls_tsp_adapt.tsp_eval_helper import ael_evaluation

import numpy as np
import random
import math

def evaluate(function: Callable):
    evaluator = ael_evaluation.Evaluation()
    return evaluator.evaluate(function)


class Sandbox:
    def __init__(self, verbose=False, numba_accelerate=True, timeout=30):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate
        self._timeout = timeout

    def run(self,
            function_to_evolve: str,  # RZ: accelerate the code by decorating @numba.jit() on function_to_evolve.
            ) -> tuple[Any, bool]:
        try:
            func_name = _evaluator_accelerate._extract_function_name(function_to_evolve)
            function_to_evolve = _evaluator_accelerate.add_import_package_statement(function_to_evolve, 'numpy', 'np')
            if self._numba_accelerate:
                function_to_evolve = _evaluator_accelerate.add_numba_decorator(
                    program=function_to_evolve,
                    function_name=[func_name]
                )
            # print(function_to_evolve)
            # compile the program, and maps the global func/var/class name to its address
            all_globals_namespace = {}
            # execute the program, map func/var/class to global namespace
            exec(function_to_evolve, all_globals_namespace)
            # get the pointer of 'function_to_run'
            func_pointer = all_globals_namespace[func_name]
            # return the execution results
            results = evaluate(func_pointer)
            if results is not None:
                if not isinstance(results, (int, float)):
                    results = (None, False)
                else:
                    # negation because our optimization objective is bigger, the better
                    results = (results, True)  # convert to FunSearch result format
            else:
                results = None, False
        except Exception as e:
            results = None, False
        return results


test_code = '''
def update_edge_distance_v2(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = edge_distance.copy()
    n = len(local_opt_tour)

    # Enhanced fractal dimension calculation with self-similarity preservation
    if np.max(edge_n_used) > 0:
        edge_pattern = edge_n_used > 0
        fractal_dim = np.sum(edge_pattern) / (n * n)
        fractal_dim = max(0.1, min(0.9, fractal_dim))
        # Self-similarity adjustment
        self_similarity = np.sum(np.diag(edge_n_used)) / np.sum(edge_n_used)
        fractal_dim = fractal_dim * (1 + 0.1 * self_similarity)
    else:
        fractal_dim = 0.5

    # Dynamic multi-scale geometric transformation with scale weighting
    scales = [1, 2, 3]
    scale_weights = [0.7, 0.2, 0.1]  # Higher weight for smaller scales
    for scale, weight in zip(scales, scale_weights):
        for i in range(n):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + scale) % n]

            usage = edge_n_used[u, v]
            if usage > 0:
                transform_factor = 1 + 0.2 * (scale ** fractal_dim) * (1 - (usage / np.max(edge_n_used))) * weight
                updated_edge_distance[u, v] *= transform_factor

    # Enhanced hierarchical edge connectivity analysis with dynamic level weighting
    for i in range(n):
        u = local_opt_tour[i]
        for level in range(1, 4):
            v = local_opt_tour[(i + level) % n]
            connectivity = np.sum(edge_n_used[u, :] > 0) / n
            # Dynamic level weighting (higher weight for closer levels)
            level_weight = 1 - (level - 1) / 3
            updated_edge_distance[u, v] *= (1 + 0.15 * connectivity * level_weight)

    # Fractal-based edge prioritization with dynamic exploration factor
    for u in range(n):
        for v in range(n):
            if edge_n_used[u, v] == 0:
                # Dynamic exploration factor based on local fractal dimension
                local_fractal = np.sum(edge_n_used[u, :] > 0) / n
                exploration = 1 + 0.3 * (1 - local_fractal) * (1 + 0.1 * np.random.randn())
                updated_edge_distance[u, v] *= exploration

    # Adaptive fractal dimension modulation with entropy-based adjustment
    usage_entropy = -np.sum((edge_n_used / np.sum(edge_n_used)) * np.log(edge_n_used + 1e-6))
    # Add dynamic entropy weight based on current fractal dimension
    entropy_weight = 1 + 0.05 * (1 - fractal_dim)
    fractal_modulation = 1 + 0.1 * usage_entropy * entropy_weight * (1 - fractal_dim)
    updated_edge_distance *= fractal_modulation

    # Enhanced position-aware fractal transformation with dynamic position weighting
    for i in range(n):
        u = local_opt_tour[i]
        for j in range(n):
            v = local_opt_tour[j]
            if u != v:
                # Dynamic position weighting (higher weight for closer positions)
                position_factor = 1 - abs(i - j) / n
                position_weight = 1 + 0.2 * position_factor * (1 - fractal_dim)
                # Add scale-dependent adjustment
                for scale in scales:
                    if abs(i - j) <= scale:
                        position_weight *= (1 + 0.1 * scale_weights[scales.index(scale)])
                updated_edge_distance[u, v] *= position_weight

    return updated_edge_distance
'''

code2 = '''
def update_edge_distance_v2(edge_distance: np.ndarray, local_opt_tour: np.ndarray, edge_n_used: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """
    Args:
        edge_distance (np.ndarray): Original edge distance matrix.
        local_opt_tour (np.ndarray): Local optimal solution path.
        edge_n_used (np.ndarray): Matrix representing the number of times each edge is used.
        temperature (float): Temperature parameter to balance exploration and exploitation.
    Return:
        updated_edge_distance: updated score of each edge distance matrix.
    """
    num_nodes = edge_distance.shape[0]
    updated_edge_distance = np.copy(edge_distance)

    # Calculate the edge usage factor for all edges in the local optimal tour
    edge_usage_factor = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes - 1):
        current_node = local_opt_tour[i]
        next_node = local_opt_tour[i + 1]
        edge_usage_factor[current_node, next_node] = 1 / (1 + edge_n_used[current_node, next_node])

    edge_usage_factor[local_opt_tour[-1], local_opt_tour[0]] = 1 / (1 + edge_n_used[local_opt_tour[-1], local_opt_tour[0]])

    # Apply the edge usage factor to the edge distances
    updated_edge_distance *= edge_usage_factor

    # Introduce a small random factor to escape local optima
    random_factor = np.random.uniform(0.95, 1.05, size=updated_edge_distance.shape)
    updated_edge_distance *= random_factor

    # Add a small penalty for edges that are used more frequently
    penalty_factor = 1 + edge_n_used / (edge_n_used.max() + 1)
    updated_edge_distance *= penalty_factor

    # Normalize the updated edge distances to maintain the original scale
    updated_edge_distance = updated_edge_distance / np.mean(updated_edge_distance) * np.mean(edge_distance)

    # Ensure symmetry in the distance matrix
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2

    # Add a small constant to avoid zero distances
    updated_edge_distance += 1e-6

    # Introduce a temperature-based factor to balance exploration and exploitation
    updated_edge_distance = (1 - temperature) * updated_edge_distance + temperature * edge_distance

    return updated_edge_distance
'''
if __name__ == '__main__':
    sandbox = Sandbox()
    res = sandbox.run(test_code)
    print(res)
    print(sandbox.run(code2))
