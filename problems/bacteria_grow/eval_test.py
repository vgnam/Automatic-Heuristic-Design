# Module Name: BGEvaluation
# Last Revision: 2025/2/10
# Description: Implements a mathematical function skeleton that models the growth rate of
# E. Coli bacteria. The function takes as input observations of population density, substrate
# concentration, temperature, and pH level, along with a set of parameters to be optimized.
# The output is the calculated bacterial growth rate. This module is part of the LLM4AD project
# (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
# - b: np.ndarray - population density of the bacterial species (default: None).
# - s: np.ndarray - substrate concentration (default: None).
# - temp: np.ndarray - temperature observations (default: None).
# - pH: np.ndarray - pH level observations (default: None).
# - params: np.ndarray - array of numeric constants or parameters to be optimized (default: None).
# - timeout_seconds: int - Maximum allowed time (in seconds) for the evaluation process (default: 20).
#
# References:
# - Shojaee, Parshin, et al. "Llm-sr: Scientific equation discovery via programming
#   with large language models." arXiv preprint arXiv:2404.18400 (2024).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
import logging
import sys

sys.path.insert(0, "../../../")

import gpt

import numpy as np

import numpy as np

import numpy as np

import numpy as np

import numpy as np

import numpy as np

def equation(b, s, temp, pH, params):
    k1 = params[0]
    k2 = params[1]
    k3 = params[2]
    k4 = params[3]
    k5 = params[4]
    k6 = params[5]
    k7 = params[6]
    k8 = params[7]
    k9 = params[8]
    k10 = params[9]

    density_term = k1 * np.tanh(k2 * b)
    substrate_term = k3 * s**k4 / (k5 + s**k4)
    temp_term = k6 / (1 + np.exp(-k7 * (temp - k8)))
    pH_term = k9 * np.exp(-k10 * (pH - 7)**2)
    interaction_term = k1 * b * s / (1 + np.exp(-k2 * (b * s - k3)))

    result = (density_term * substrate_term * temp_term * pH_term) + interaction_term
    return result



MAX_NPARAMS = 10


def solve(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, db: np.ndarray):
    from scipy.optimize import minimize

    def loss(params):
        y_pred = equation(b, s, temp, pH, params)
        return np.mean((y_pred - db) ** 2)

    result = minimize(loss, [1.0] * MAX_NPARAMS, method='BFGS')

    loss_value = result.fun
    if np.isnan(loss_value) or np.isinf(loss_value):
        return None
    else:
        return loss_value



if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")

    # problem_size = int(sys.argv[1])
    # root_dir = sys.argv[2]
    # mood = sys.argv[3]
    # assert mood in ['train', 'val']
    mood = 'train'

    basepath = os.path.dirname(__file__)

    if mood == 'train':
        import test1
        import test2

        data = np.concatenate([test1.data, test2.data], axis=0)


        b = np.array([d['b'] for d in data])
        s = np.array([d['s'] for d in data])
        temp = np.array([d['temp'] for d in data])
        pH = np.array([d['pH'] for d in data])
        db = np.array([d['db'] for d in data])
        n_instances = 1

        print(f"[*] Dataset loaded with {n_instances} instances.")

        objs = []
        obj = solve(b, s, temp, pH, db)
        print(f"[*] Instance 0: {obj}")
        objs.append(obj)

        print("[*] Average:")
        print(np.mean(objs))

    else:  # mood == 'val'
        import train

        data = train.data

        b = np.array([d['b'] for d in data])
        s = np.array([d['s'] for d in data])
        temp = np.array([d['temp'] for d in data])
        pH = np.array([d['pH'] for d in data])
        db = np.array([d['db'] for d in data])

        logging.info(f"[*] Evaluating validation dataset")

        objs = []
        obj = solve(b, s, temp, pH, db)
        objs.append(obj)

        print(f"[*] Average: {np.mean(objs)}")


# GENESIS:

# 2-13_19-09-18: 0.04265387463460248
# 2-12_19-50-38: 0.02950748255457558
# 00-29-20: 0.042162654327434025

# MCTS:

# 2026-02-13_10-12-49: 0.03798263890628418
# 2026-02-13_07-18-21: 0.04545445693040738
# 2026-02-12_22-54-37: 0.029690478133522796



