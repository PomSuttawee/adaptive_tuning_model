import numpy as np
import cv2
from scipy.optimize import minimize
from skimage.metrics import structural_similarity as ssim
import model
from features import local_variance, local_entropy, gradient_magnitude

def extract_features(image):
    variance = local_variance(image)
    gradient = gradient_magnitude(image)
    return np.stack([variance, gradient], axis=1)

def apply_processing(image, parameters):
    gaussian_blur_kernel_size, clahe_clip_limit, clahe_grid_size, adaptive_thresholding_block_size, adaptive_thresholding_constant, morph_kernel_size = parameters
    image = cv2.GaussianBlur(image, gaussian_blur_kernel_size)
    clahe_object = cv2.createCLAHE(clahe_clip_limit, clahe_grid_size)
    image = clahe_object.apply(image)
    mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, adaptive_thresholding_block_size, adaptive_thresholding_constant)
    mask = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size))
    return cv2.bitwise_and(image, mask)

def calculate_psnr(image1, image2):
    return cv2.PSNR(image1, image2)

def calculate_ssim(image1, image2):
    return ssim(image1, image2, data_range=image2.max() - image2.min()) 

def cost_function(list_theta, image, reference_image, features, p_min, p_max):
    theta_0 = list_theta[0]
    theta_1 = list_theta[1:features.shape[1] + 1]
    theta_2 = list_theta[features.shape[1] + 1:].reshape(features.shape[1], features.shape[1])
    
    predicted_params = model.predict_parameter(features, theta_0, theta_1, theta_2, p_min, p_max)
    processed_image = apply_processing(image, predicted_params)

    psnr_value = calculate_psnr(processed_image, reference_image)
    ssim_value = calculate_ssim(processed_image, reference_image)

    return -psnr_value

def train_model(images, reference_images, p_min, p_max):
    """
    1. Extract feature from images
    2. Initialize theta_0 (scalar), theta_1 (vector), and theta_2 (matrix)
    3. Minimize cost function (PSNR, SSIM) by optimize theta value
    4. Return optimal theta
    """
    features = np.vstack([extract_features(img) for img in images])
    
    theta_0_init = np.random.randn(1)
    theta_1_init = np.random.randn(features.shape[1])
    theta_2_init = np.random.randn(features.shape[1], features.shape[1])
    initial_theta = np.concatenate([theta_0_init, theta_1_init, theta_2_init.flatten()], )

    result = minimize(cost_function, initial_theta, args=(images[0], reference_images[0], features, p_min, p_max), method='Nelder-Mead')
    theta_0 = result.x[0]
    theta_1 = result.x[1:features.shape[1] + 1]
    theta_2 = result.x[features.shape[1] + 1:].reshape(features.shape[1], features.shape[1])
    
    return theta_0, theta_1, theta_2