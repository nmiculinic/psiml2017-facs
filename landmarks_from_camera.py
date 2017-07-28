import argparse
import numpy as np
import cv2
import time
import dataset
from PIL import Image, ImageDraw
# from keras.models import load_model

args = argparse.ArgumentParser()
args.add_argument("model_path")
args = args.parse_args()

# model = load_model(args.model_path)
# print(model.inputs)
# dim = int(model.input.shape[1])
dim = 160

def draw_landmarks(image, landmarks, r=1, fill_color=(255,0,0,100)):
    draw = ImageDraw.Draw(image)
    for row in landmarks:
        x, y = row
        draw.ellipse((x-r, y-r, x+r, y+r), fill=fill_color)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while(True):   
    _, img = cap.read()
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    dp = {
        'image': img,
        'landmarks': np.zeros((66,2))
    }
    dp = dataset.resize_mirror_datapoint(dp, dim, False)
    img = dp['image']

#         landmark = model.predict(np.array(img.convert("L"))[None, :, :, None] / 255.0)
    landmark = np.array([[
        [0,0],
        [50,50],
        [100,50]
    ]])
    landmark = np.squeeze(landmark, axis=0)
    
    h, w = img.size
    img = img.resize((2*h, 2*w))
    landmark *= 2
    
    draw_landmarks(img, landmark, r=1)        
    
    # Convert RGB to BGR 
    img = np.array(img) 
    img = img[:, :, ::-1].copy() 
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
