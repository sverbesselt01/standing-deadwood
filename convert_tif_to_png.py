import cv2
import numpy as np


def tif_to_png(tif_path: str, png_path: str):
    # Read the TIFF image preserving its original bit depth and channels
    img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Could not load image at {tif_path}")



    # Save as PNG
    cv2.imwrite(png_path, img)


# Example usage
tif_to_png("C:/Users/sebastiaan_verbessel/Downloads/clip.tif", "C:/Users/sebastiaan_verbessel/Downloads/output.png")