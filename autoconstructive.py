# autoconstructive_ea.py
import random
import math
import copy
from typing import List, Dict

random.seed(42)

# ---------- Problem (minimize) ----------
def fitness_sphere(genome: List[float]) -> float:
    return sum(x*x for x in genome)

# ---------- Individual representation ----------
# Each individual is a dict with:
# - genome: List[float]
# - maker_params: Dict[str, float]  # encodes reproduction strategy
# - fitness: float

def make_random_individual(dim=5):
    genome = [random.uniform(-5,5) for _ in range(dim)]
    # maker_params encode how this individual makes children
    maker_params = {
        "mut_scale": random.uniform(0.01, 1.0),       # gaussian sigma for genome mutation
        "mut_rate": random.uniform(0.01, 0.5),        # per-gene mutation probability
        "mut_param_scale": random.uniform(0.01, 0.3)  # how fast maker_params mutate
    }
    return {"genome": genome, "maker_params": maker_params, "fitness": None}

# ---------- Autoconstructive reproduction ----------
def make_child(parent: Dict, dim=5):
    """
    Child genome is produced by a mutation that uses parent's maker_params.
    Child's maker_params are mutated from parent's maker_params -> autoconstructive part.
    """
    p = parent
    mp = p["maker_params"]
    mut_scale = mp["mut_scale"]
    mut_rate = mp["mut_rate"]
    param_scale = mp["mut_param_scale"]

    # mutate genome
    new_genome = []
    for g in p["genome"]:
        if random.random() < mut_rate:
            new_genome.append(g + random.gauss(0, mut_scale))
        else:
            new_genome.append(g)

    # mutate maker_params (so reproduction strategy evolves)
    new_maker_params = {}
    # keep ranges reasonable
    new_maker_params["mut_scale"] = max(1e-6, mp["mut_scale"] + random.gauss(0, param_scale))
    new_maker_params["mut_rate"] = min(0.99, max(0.0, mp["mut_rate"] + random.gauss(0, param_scale/5)))
    new_maker_params["mut_param_scale"] = max(1e-6, mp["mut_param_scale"] + random.gauss(0, param_scale/5))

    return {"genome": new_genome, "maker_params": new_maker_params, "fitness": None}

# ---------- Simple selection and replacement ----------
def evaluate(ind):
    ind["fitness"] = fitness_sphere(ind["genome"])

def tournament_select(pop, k=3):
    aspirants = random.sample(pop, k)
    aspirants.sort(key=lambda x: x["fitness"])
    return aspirants[0]

# ---------- Main autoconstructive EA ----------
def autoconstructive_ea(
    pop_size=50, dim=5, generations=200, offspring_per_parent=2,
    elite=2, tournament_k=3, verbose=True
):
    # init
    pop = [make_random_individual(dim) for _ in range(pop_size)]
    for ind in pop:
        evaluate(ind)

    history = []
    for gen in range(1, generations+1):
        # produce offspring: each selected parent makes children using its own maker_params
        offspring = []
        for _ in range(pop_size * offspring_per_parent):
            parent = tournament_select(pop, k=tournament_k)
            child = make_child(parent, dim=dim)
            evaluate(child)
            offspring.append(child)

        # combine and select next generation
        combined = pop + offspring
        combined.sort(key=lambda x: x["fitness"])
        pop = combined[:pop_size]

        # logging
        best = pop[0]
        avg = sum(ind["fitness"] for ind in pop)/len(pop)
        history.append((best["fitness"], avg))
        if verbose and (gen % max(1, generations//10) == 0 or gen==1):
            avg_mut_scale = sum(ind["maker_params"]["mut_scale"] for ind in pop)/len(pop)
            avg_mut_rate  = sum(ind["maker_params"]["mut_rate"]  for ind in pop)/len(pop)
            print(f"Gen {gen:3d} | best {best['fitness']:.6f} | avg {avg:.6f} | "
                  f"avg_mut_scale {avg_mut_scale:.4f} avg_mut_rate {avg_mut_rate:.4f}")

    return pop, history

if __name__ == "__main__":
    final_pop, hist = autoconstructive_ea(
        pop_size=60, dim=10, generations=200, offspring_per_parent=2, elite=2, tournament_k=3
    )
    best = final_pop[0]
    print("\nBest individual fitness:", best["fitness"])
    print("Best genome (first 10):", [round(x,4) for x in best["genome"]])
    print("Best maker_params:", best["maker_params"])
