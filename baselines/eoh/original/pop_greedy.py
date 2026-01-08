import heapq
import numpy as np
import math
def population_management(pop,size):
    pop = [individual for individual in pop if individual['objective'] is not None]
    if size > len(pop):
        size = len(pop)
    unique_pop = []
    unique_objectives = []
    for individual in pop:
        if individual['objective'] not in unique_objectives:
            unique_pop.append(individual)
            unique_objectives.append(individual['objective'])
    # Delete the worst individual
    #pop_new = heapq.nsmallest(size, pop, key=lambda x: x['objective'])
    pop_new = heapq.nsmallest(size, unique_pop, key=lambda x: x['objective'])
    return pop_new


# def population_management(pop, size):
#
#     cleaned_pop = []
#
#     for ind in pop:
#         # ---- Clean objective ----
#         obj = ind.get('objective')
#         try:
#             obj_val = float(obj)
#             if math.isnan(obj_val) or math.isinf(obj_val):
#                 ind['objective'] = 10000.0
#             else:
#                 ind['objective'] = obj_val
#         except (TypeError, ValueError):
#             ind['objective'] = 10000.0
#
#         # ---- Clean complexity ----
#         comp = ind.get('complexity')
#         try:
#             comp_val = float(comp)
#             if math.isnan(comp_val) or math.isinf(comp_val):
#                 ind['complexity'] = 10000
#             else:
#                 ind['complexity'] = comp_val
#         except (TypeError, ValueError):
#             ind['complexity'] = 0.0
#
#         # ---- Clean novelty ----
#         nov = ind.get('novelty', 0)
#         try:
#             nov_val = float(nov)
#             if math.isnan(nov_val) or math.isinf(nov_val):
#                 ind['novelty'] = 0.0
#             else:
#                 ind['novelty'] = nov_val
#         except (TypeError, ValueError):
#             ind['novelty'] = 0.0
#
#         cleaned_pop.append(ind)
#
#     unique_pop = []
#     seen = set()
#     for ind in cleaned_pop:
#         obj = ind['objective']
#         if obj not in seen:
#             seen.add(obj)
#             unique_pop.append(ind)
#
#     if len(unique_pop) <= size:
#         return unique_pop
#
#     F = np.array([
#         [float(ind.get('objective', 10000)),
#          float(ind.get('complexity', 10000)),
#          float(-ind.get('novelty', 0))]
#         for ind in unique_pop
#     ])
#
#     from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
#     nds = NonDominatedSorting()
#     fronts = nds.do(F)
#
#     selected = []
#     for front in fronts:
#         if len(selected) + len(front) <= size:
#             selected.extend([unique_pop[i] for i in front])
#         else:
#             remaining = size - len(selected)
#             selected.extend([unique_pop[i] for i in front[:remaining]])
#             break
#
#     return selected
