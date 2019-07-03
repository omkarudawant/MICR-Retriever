import argparse
import cv2
import os
import extraction.extract_micr as extract
import preprocess.preprocess as p
import numpy as np
import imutils
import skimage

directory = os.getcwd()
data_directory = directory + '/cheques/'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Name of image")
args = vars(ap.parse_args())
image = data_directory + args['image']

print('Extracting from: ' + args['image'].split('/')[-1])
preprocessed_img = p.preprocess(image_path=image)
extracted_micr, contour_img = extract.extract_micr(image=preprocessed_img)
print('MICR Code: {0}'.format(extracted_micr))
cv2.imwrite('ocr_cheque.jpg', contour_img)

print('opencv: ', cv2.__version__)
print('numpy: ', np.__version__)
print('skimage: ', skimage.__version__)
print('imutils: ', imutils.__version__)
