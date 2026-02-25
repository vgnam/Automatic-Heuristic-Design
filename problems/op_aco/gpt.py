import numpy as np
import random
import math
import scipy
import torch
def heuristics_v2(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    return prize[np.newaxis, :] / distance
