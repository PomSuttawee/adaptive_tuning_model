import numpy as np
from scipy.ndimage import generic_filter, sobel
from skimage.measure import shannon_entropy

# Function for local variance
def local_variance(image, size=3):
    return generic_filter(image, np.var, size=size)

# Function for local entropy
def local_entropy(image, size=3):
    return generic_filter(image, shannon_entropy, size=size)

# Function for gradient magnitude
def gradient_magnitude(image):
    dx = sobel(image, axis=0)
    dy = sobel(image, axis=1)
    return np.hypot(dx, dy)