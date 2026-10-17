import cv2

def is_blurry(image_crop, threshold=100.0):
    # Convert to grayscale
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    # Compute Laplacian variance
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # If variance is less than threshold, it is blurry
    return variance < threshold
