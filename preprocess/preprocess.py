# Necessary imports
import cv2

# TODO: Enhance preprocessing

def preprocess(image_path):
    """
    Function to perform grayscaling and thresholding 
    as a basic preprocessing step.
    Parameter
    ---------
        image_path: str
            Input image path
    """
    image = cv2.imread(image_path, 0)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return image
