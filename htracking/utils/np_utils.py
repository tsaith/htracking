import numpy as np

def to_magnitude(vec):
    """
    Estimate the magnitude of a vector.
    """
    return np.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
