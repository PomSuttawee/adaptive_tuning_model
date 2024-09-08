from utils import load_image, resize_with_padding
from train import train_model
import numpy as np
import cv2

if __name__ == "__main__":
    file_path = 'C:\\Users\\Suttawee\\Desktop\\adaptive_tuning_model\\images'
    input_image_names = ['5CY.jpg', 'LPY.jpg', 'NGG.jpg']
    target_image_names = ['5CY.jpg', 'LPY.jpg', 'NGG.jpg']
    input_images = []
    target_images = []
    
    for image_name in input_image_names:
        image = load_image(file_path + '\\input\\' + image_name)
        input_images.append(resize_with_padding(image, target_size=(500, 750)))
    
    for image_name in target_image_names:
        image = load_image(file_path + '\\target\\' + image_name)
        target_images.append(resize_with_padding(image, target_size=(500, 750)))
    
    """
    PARAMETER LISTS
        - GAUSSIAN_BLUR_KERNEL_SIZE
        - CLAHE_CLIP_LIMIT
        - CLAHE_GRID_SIZE
        - ADAPTIVE_THRESHOLDING_BLOCK_SIZE
        - ADAPTIVE_THRESHOLDING_CONSTANT
        - MORPH_KERNEL_SIZE
    """
    p_min = np.array([3, 1.0, 3, 3, 0, 3])
    p_max = np.array([29, 5.0, 29, 29, 100, 29])
    
    theta_0, theta_1, theta_2 = train_model(input_images, target_images, p_min, p_max)
    print(theta_0, theta_1, theta_2)