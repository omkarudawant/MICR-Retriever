# coding: utf-8

from cv2 import imread, imwrite
import os

files = [file for file in os.listdir('300/') if file.endswith('.tif')]

out_path = 'jpg_images/'
if not os.path.exists(out_path):
    os.mkdir(out_path)

for f in files:
    img = cv2.imread('300/'+f)
    cv2.imwrite(out_path + f.split('.')[0] + '.jpg', img)

