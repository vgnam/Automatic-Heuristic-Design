import numpy as np
from scipy.spatial import distance_matrix
from pathlib import Path
import logging
from tqdm import tqdm

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Danh sách instance và optimal values
names = ["ts225", "rat99", "rl1889", "u1817", "d1655", "bier127", "lin318", "eil51", "d493", "kroB100", "kroC100", "ch130", "pr299", "fl417", "d657", "kroA150", "fl1577", "u724", "pr264", "pr226", "pr439"]

opt = {
    'ts225': 126643,
    'rat99': 1211,
    'rl1889': 316536,
    'u1817': 57201,
    'd1655': 62128,
    'bier127': 118282,
    'lin318': 42029,
    'eil51': 426,
    'd493': 35002,
    'kroB100': 22141,
    'kroC100': 20749,
    'ch130': 6110,
    'pr299': 48191,
    'fl417': 11861,
    'd657': 48912,
    'kroA150': 26524,
    'fl1577': 22249,
    'u724': 41910,
    'pr264': 49135,
    'pr226': 80369,
    'pr439': 107217
}

# Import gpt và ACO
import gpt
from aco import ACO

# Tìm hàm heuristic trong gpt.py
possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]
heuristic_name = None
for name in possible_func_names:
    if hasattr(gpt, name):
        heuristic_name = name
        break
if heuristic_name is None:
    raise AttributeError(f"Không tìm thấy hàm heuristic nào trong {possible_func_names}")

heuristics = getattr(gpt, heuristic_name)

# Thông số ACO
N_ITERATIONS = 100
N_ANTS = 30


def parse_and_scale_tsp(filepath):
    """
    Đọc file .tsp, parse tọa độ và scale về [0, 1]^2
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse the data
    data = lines[6:-1]
    data = [x.strip().split() for x in data]
    data = [[float(x) for x in row[1:]] for row in data]

    # Scale the data to [0, 1]^2 to align with the training data
    data = np.array(data)
    scale = max(np.max(data, axis=0) - np.min(data, axis=0))
    data = (data - np.min(data, axis=0)) / scale

    return data


def solve(node_pos):
    """
    Giải TSP bằng ACO + heuristic do LLM sinh
    """
    dist_mat = distance_matrix(node_pos, node_pos)
    np.fill_diagonal(dist_mat, 1e9)

    # Tính heuristic matrix
    heu = heuristics(dist_mat.copy())
    heu = np.where(np.isfinite(heu), heu, 0)
    heu += 1e-9  # tránh 0

    aco = ACO(dist_mat, heu, n_ants=N_ANTS)
    obj = aco.run(N_ITERATIONS)
    return obj


def calculate_gap(instance_name, predicted_obj, optimal_dict):
    """
    Tính gap (%) so với optimal
    """
    if instance_name not in optimal_dict:
        raise ValueError(f"Instance {instance_name} not in optimal dictionary")
    optimal = optimal_dict[instance_name]
    gap = (predicted_obj - optimal) / optimal * 100
    return round(gap, 2)


def main():
    results = []

    print(f"{'Instance':12} | {'Predicted':10} | {'Optimal':10} | {'Gap (%)':8}")
    print("-" * 50)

    for name in tqdm(names, desc="Solving instances"):
        try:
            filepath = f'D:/Automatic-Heuristic-Design/problems/tsp_constructive/test/tsplib/{name}.tsp'
            if not Path(filepath).exists():
                logging.warning(f"File not found: {filepath}")
                results.append({
                    'instance': name,
                    'predicted': None,
                    'optimal': opt.get(name, None),
                    'gap (%)': None
                })
                continue

            # Parse và scale dữ liệu
            node_pos = parse_and_scale_tsp(filepath)

            # Giải
            predicted = solve(node_pos)
            gap = calculate_gap(name, predicted, opt)

            results.append({
                'instance': name,
                'predicted': predicted,
                'optimal': opt[name],
                'gap (%)': gap
            })

            print(f"{name:12} | {predicted:10.0f} | {opt[name]:10} | {gap:8.2f}")

        except Exception as e:
            logging.error(f"Error solving {name}: {e}")
            results.append({
                'instance': name,
                'predicted': None,
                'optimal': opt.get(name, None),
                'gap (%)': None
            })

    # Tổng kết
    valid_results = [r for r in results if r['gap (%)'] is not None]
    if valid_results:
        mean_gap = np.mean([r['gap (%)'] for r in valid_results])
        print("-" * 50)
        print(f"{'MEAN GAP':12} | {'':10} | {'':10} | {mean_gap:8.2f}%")
        print(f"Solved {len(valid_results)}/{len(results)} instances.")
    else:
        print("No valid results to compute mean gap.")


if __name__ == "__main__":
    main()