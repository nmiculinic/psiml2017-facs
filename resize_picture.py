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
 
image_path = "C:\\Users\\admin\\Desktop\\DATASETS\\10k FACES"
landmarks_path = "C:\\Users\\admin\\Desktop\\resized pics"

####################################################################
def resize_datapoint(datapoint, PICTURE_SIZE=132,SMALLER_SIZE=4):
    img = datapoint['image'].convert("RGB")
    landmarks = datapoint['landmarks']
    width, height = img.size[:2]

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
    img = img.resize((resized_width,resized_height),Image.ANTIALIAS)
    
    crop_x = int(random()*SMALLER_SIZE)
    crop_y = int(random()*4)
    mirror = int(random())
    img = img.crop(0+crop_x,0+crop_y,PICTURE_SIZE-SMALLER_SIZE+crop_x,PICTURE_SIZE-SMALLER_SIZE+crop_y)
    if mirror == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        landmarks[:, 0] = PICTURE_SIZE-SMALLER_SIZE - landmarks[:,0]
        landmarks[:, 1] = PICTURE_SIZE-SMALLER_SIZE - landmarks[:,1]

    offset_image = np.zeros(PICTURE_SIZE-SMALLER_SIZE,PICTURE_SIZE-SMALLER_SIZE,3),np.uint8)
    offset_image= Image.fromarray(offset_image)
    offset_image.paste(img,(x_offset,y_offset))

    datapoint['image'] = offset_image.convert("L")
    datapoint['landmarks'] = landmarks
    return datapoint