import numpy as np

def equation_v2(b, s, temp, pH, params):
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

    density_term = k1 * b**k2 / (k3 + b**k2)
    substrate_term = k4 * s**k5 / (k6 + s**k5)
    temp_term = k7 * np.exp(-(temp - 37)**2 / (2 * k8**2))
    pH_term = k9 * (1 + np.tanh(k10 * (pH - 7))) / 2
    inhibition_term = k2 * b**k3 / (b**k3 + k4**k5)

    normalization_term = (1 + np.exp(-k6 * (temp - 37))) / (1 + np.exp(-k7 * (pH - 7)))
    updated_edge_distance = density_term * substrate_term * temp_term * pH_term * inhibition_term * normalization_term
    return updated_edge_distance
