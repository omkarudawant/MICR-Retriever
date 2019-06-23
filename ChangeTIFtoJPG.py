# coding: utf-8
# author: Omkar 
# Date: 22/06/2019 09:32:05

from cv2 import imread, imwrite
import os

path_to_tig = 'TIF_dir/'
files = [file for file in os.listdir(path_to_tif) if file.endswith('.tif')]

out_path = 'jpg_images/'
if not os.path.exists(out_path):
    os.mkdir(out_path)

for f in files:
    img = cv2.imread(path_to_tif + f)
    cv2.imwrite(out_path + f.split('.')[0] + '.jpg', img)
