import cv2
import os
import extraction.extract_micr as extract
import preprocess.preprocess as p
import numpy as np
import imutils
import skimage
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--image', required=True, help='Absolute path of image')
args = vars(ap.parse_args())

# directory = os.getcwd()
# data_directory = directory + '/cheques/'
# print(f'Images available: {os.listdir(data_directory)}')

input_ = args['image']

image = input_
print(f'Extracting from: {image}')

preprocessed_img = p.preprocess(image_path=image)
extracted_micr, contour_img = extract.extract_micr(image=preprocessed_img)

print(f'MICR Code: {extracted_micr}')
cv2.imwrite('ocr_cheque.jpg', contour_img)
cv2.imshow('Detected MICR code', cv2.resize(contour_img, (1000, 400)))
cv2.waitKey(0)
cv2.destroyAllWindows()
print('Exiting...')
