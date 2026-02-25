import numpy as np
import logging
from gen_inst import TSPInstance, load_dataset, dataset_conf
from gls import guided_local_search
from tqdm import tqdm

import gpt

heuristics = None
for name in [
    "heuristics_v3",
    "heuristics_v2",
    "heuristics_v1",
    "heuristics",   # fallback cuối
]:
    if hasattr(gpt, name):
        heuristics = getattr(gpt, name)
        break

if heuristics is None:
    raise ImportError(
        "No valid heuristic found in gpt.py "
        "(expected update_edge_distance_v1/v2/v3 or heuristics)"
    )

perturbation_moves = 30
iter_limit = 1200

def calculate_cost(inst: TSPInstance, path: np.ndarray) -> float:
    return inst.distmat[path, np.roll(path, 1)].sum().item()

def solve(inst: TSPInstance) -> float:
    heu = heuristics(inst.distmat.copy())
    assert tuple(heu.shape) == (inst.n, inst.n)
    result = guided_local_search(inst.distmat, heu, perturbation_moves, iter_limit)
    # print(result)
    return calculate_cost(inst, result)

if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    mood = sys.argv[3]
    assert mood in ['train', 'val', "test"]

    basepath = os.path.dirname(__file__)
    # automacially generate dataset if nonexists
    if not os.path.isfile(os.path.join(basepath, f"dataset/train{dataset_conf['train'][0]}_dataset.npy")):
        from gen_inst import generate_datasets
        generate_datasets()
    
    if mood == 'train':
        dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npy")
        dataset = load_dataset(dataset_path)
        n_instances = dataset[0].n

        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
        
        objs = []
        for i, instance in enumerate(dataset):
            obj = solve(instance)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj)
        
        print("[*] Average:")
        print(np.mean(objs))

    else: # mood == 'val'
        for problem_size in dataset_conf['val']:
            dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npy")
            dataset = load_dataset(dataset_path)
            n_instances = dataset[0].n
            logging.info(f"[*] Evaluating {dataset_path}")

            objs = []
            for i, instance in enumerate(tqdm(dataset)):
                obj = solve(instance)
                objs.append(obj)
            
            print(f"[*] Average for {problem_size}: {np.mean(objs)}")