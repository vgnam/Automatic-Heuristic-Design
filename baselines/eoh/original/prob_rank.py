import random

def parent_selection(pop,m):
    def to_float_safe(obj_value):
        try:
            if obj_value is None:
                return float('inf')
            return float(obj_value)
        except (ValueError, TypeError):
            return float('inf')

    pop = sorted(pop, key=lambda x: to_float_safe(x.get('objective')))
    ranks = [i for i in range(len(pop))]
    probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    parents = random.choices(pop, weights=probs, k=m)
    return parents