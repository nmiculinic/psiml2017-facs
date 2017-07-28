import numpy as np
from PIL import Image, ImageDraw
import cv2
import time

# def crop_image(img, picture_size):
#     """Crops datapoint

#     :param datapoint: Datapoint to transform
#     :param picture_size: Output picture square dimension
#     """
#     w, h = img.size
#     crop_x = #random.randint(0, w - picture_size)
#     crop_y = #random.randint(0, h - picture_size)
#     img = img.crop((crop_x, crop_y, picture_size + crop_x, picture_size + crop_y))    
#     return img

def resize_image(img, picture_size):
    img = datapoint['image']
    width, height = img.size[:2]

    if height>width:
        resized_height = picture_size
        resized_width = round(picture_size*width/height)
    else:  # height<= width:
        resized_height = round(picture_size*height/width)
        resized_width = picture_size

    img = img.resize((resized_width,resized_height), Image.ANTIALIAS)
    #img = img.crop((0, 0, picture_size, picture_size))

    return img

def resize_landmarks(landmarks, camera_image_size, resized_image_size):
    width, height = camera_image_size[:2]

    if height>width:
        landmarks[:, 0] = landmarks[:, 0] * (height/picture_size)
        landmarks[:, 1] = landmarks[:, 1] * (height/picture_size)
    else:  
        landmarks[:, 0] = landmarks[:, 0] * (width/picture_size)
        landmarks[:, 1] = landmarks[:, 1] * (width/picture_size)

    return landmarks

def draw_landmarks(image, landmarks, r=1, fill_color=(255,0,0,100)):
    draw = ImageDraw.Draw(image)
    for row in landmarks:
        x, y = row
        draw.ellipse((x-r, y-r, x+r, y+r), fill=fill_color)

def landmarks_from_camera(img_size = 160):
    while(True):   
        cap = cv2.VideoCapture()
        _, img = cap.read()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)
        #cap.set(3, img_size)
        #cap.set(4, img_size)
        landmarks_img = img.resize_image(img, img_size)
        landmarks = f(landmarks_img)
        landmarks = resize_landmarks(landmarks = landmarks,camera_image_size = img.size, resized_image_size = img_size)
        draw_landmarks(img,landmarks)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

