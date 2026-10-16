import cv2
import numpy as np
import os, csv

def explain_image(img):
    """
    Print out the main properties of the input image.

    Args:
        img: The input image as a NumPy array.

    Returns:
        A dictionary containing basic information about the image.
    """
    if img is None:
        print("Image is None or could not be loaded.")
        return None

    info = {}
    info['shape'] = img.shape
    info['dtype'] = img.dtype
    info['size'] = img.size
    info['min'] = np.min(img)
    info['max'] = np.max(img)
    unique_vals = None
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        unique_vals = np.unique(img)
    elif img.ndim == 3:
        try:
            flat = img.reshape(-1, img.shape[2])
            unique_vals = np.unique(flat, axis=0)
        except Exception:
            pass
    if unique_vals is not None:
        info['num_unique_colors'] = len(unique_vals)
        # If number is small, print a few
        if len(unique_vals) < 20:
            info['unique_colors'] = unique_vals.tolist()
    print("Image Information:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info