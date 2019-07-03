# Necessary imports
import cv2

# More preprocessing steps to be added in future


def preprocess(image_path):  # Driver function 
	image = cv2.imread(image_path, 0)
	thresh_val, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
	return image
