import numpy as np

def predict_parameter(features, theta_0, theta_1, theta_2, p_min, p_max):
    # Equation (2) from the paper
    linear_term = np.dot(theta_1, features)
    quadratic_term = np.dot(features.T, np.dot(theta_2, features))
    h = theta_0 + linear_term + quadratic_term
    
    # Equation (1) from the paper
    return p_min + (p_max - p_min) / (1 + np.exp(-h))