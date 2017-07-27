import cv2
from pathlib import Path
from tqdm import tqdm, trange
from PIL import Image
import numpy as np 
from openpyxl import load_workbook
import os
from pprint import PrettyPrinter
import argparse
import sys
 
PICTURE_SIZE = 128
root_path = "C:\\Users\\admin\\Desktop\\DATASETS\\10k FACES"
write_path = "C:\\Users\\admin\\Desktop\\resized pics"
imageDIR = os.path.join(root_path, "Face Annotations", "Images and Annotations")
for base in trange(1, 2222 + 1):
    fname = "{}.jpg".format(base)
    lname = "{}_landmarks.txt".format(base)
    
    original_image = cv2.imread(os.path.join(imageDIR, fname),0)
    #landmarksFILE = open(os.path.join(imageDIR, lname),newline='')
    x = y =[]
    #print(landmarksFILE.read())  
    landmarks = np.loadtxt(os.path.join(imageDIR, lname))
    offset_image = np.zeros((PICTURE_SIZE,PICTURE_SIZE),np.uint8)
    width, height = original_image.shape[:2]

    if height>width:
        resized_height = PICTURE_SIZE
        resized_width = round(PICTURE_SIZE*width/height)
        x_offset=round((PICTURE_SIZE-resized_width)/2)
        y_offset=0
        landmarks[:, 0] = landmarks[:, 0] * (PICTURE_SIZE/width) + x_offset
        landmarks[:, 1] = landmarks[:, 1] * (PICTURE_SIZE/height)
    elif height<width:
        resized_height = round(PICTURE_SIZE*height/width)
        resized_width = PICTURE_SIZE
        x_offset=0
        y_offset=round((PICTURE_SIZE-resized_height)/2)
        landmarks[:, 0] = landmarks[:, 0] * (PICTURE_SIZE/width)
        landmarks[:, 1] = landmarks[:, 1] * (PICTURE_SIZE/height) + y_offset
    else:
        resized_height = PICTURE_SIZE
        resized_width = PICTURE_SIZE
        x_offset=y_offset=0
        landmarks[:, 0] = landmarks[:, 0] * (PICTURE_SIZE/width)
        landmarks[:, 1] = landmarks[:, 1] * (PICTURE_SIZE/height)

    resized_image = cv2.resize(original_image, (resized_height,resized_width)) 
    # print(original_image.shape)
    # print(resized_image.shape)
    # print(resized_height,resized_width)
    # print(x_offset," ",y_offset)
    #x_offset=y_offset=0
    #y_offset=10
    #offset_image=resized_image
    #print (x_offset, " ", y_offset)
    offset_image[x_offset:x_offset+resized_image.shape[0], y_offset:y_offset+resized_image.shape[1]] = resized_image
    cv2.imwrite(os.path.join(write_path, fname), offset_image)
    #print(landmarks)
    np.savetxt(os.path.join(write_path, lname),landmarks)
    # cv2.imshow("original", original_image)
    # cv2.imshow("resized", offset_image)
#cv2.waitKey(0)
