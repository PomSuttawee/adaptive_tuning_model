import cv2
import numpy as np

def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def resize_with_padding(image, target_size=(256, 256)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize while keeping aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad the resized image to fit the target size
    pad_h = (target_size[0] - new_h) // 2
    pad_w = (target_size[1] - new_w) // 2
    padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)
    
    return padded