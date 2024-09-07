from utils import load_image, resize_with_padding
from train import train_model
import numpy as np
import cv2

if __name__ == "__main__":
    file_path = 'C:\\Users\\Suttawee\\Desktop\\adaptive_tuning_model\\test_images'
    input_image_names = ['5CY.jpg', 'LPY.jpg', 'NGG.jpg']
    ground_truth_image_names = []
    input_images = []
    ground_truth_images = []
    
    for image_name in input_image_names:
        image = load_image(file_path + '\\' + image_name)
        image = cv2.resize(image, (750, 500))
        image = cv2.bitwise_not(image)
        image = cv2.GaussianBlur(image, (151, 151), 0)
        mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 5)
        
        cv2.imshow('image', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # input_images.append(resize_with_padding(image, target_size=(500, 750)))
    
    # for image_name in ground_truth_image_names:
    #     image = load_image(file_path + '\\' + image_name)
    #     ground_truth_images.append(resize_with_padding(image, target_size=(500, 750)))
    
    # """
    # PARAMETER LISTS
    #     - GAUSSIAN_BLUR_KERNEL_SIZE
    #     - CLAHE_CLIP_LIMIT
    #     - CLAHE_GRID_SIZE
    #     - ADAPTIVE_THRESHOLDING_BLOCK_SIZE
    #     - ADAPTIVE_THRESHOLDING_CONSTANT
    #     - MORPH_KERNEL_SIZE
    # """
    # p_min = np.array([3, 1.0, 3, 3, 0, 3])
    # p_max = np.array([29, 5.0, 29, 29, 100, 29])
    
    # train_model(input_images, ground_truth_images, p_min, p_max)