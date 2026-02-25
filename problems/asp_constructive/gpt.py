import numpy as np
import math

def priority_v2(el, n, w):
    # Calculate value diversity
    values = [val for val in el if val != 0]
    if not values:
        return 0.0
    value_counts = np.bincount(values, minlength=3)
    value_diversity = len(value_counts[value_counts > 0]) / 3.0

    # Calculate position spread
    positions = [i for i, val in enumerate(el) if val != 0]
    if not positions:
        return 0.0
    position_spread = len(set(positions)) / n

    # Penalize consecutive zeros
    zero_penalty = 0.0
    consecutive_zeros = 0
    for i in range(n - 1):
        if el[i] == 0 and el[i + 1] == 0:
            consecutive_zeros += 1
    zero_penalty = consecutive_zeros / (n - 1)

    # Combine metrics with different weights
    priority = (value_diversity * 0.6) + (position_spread * 0.3) - (zero_penalty * 0.1)
    return priority