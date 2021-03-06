import argparse
import numpy as np
import cv2
import time
import dataset
from PIL import Image, ImageDraw
from keras.models import load_model

args = argparse.ArgumentParser()
args.add_argument("model_path")
args = args.parse_args()

model = load_model(args.model_path)
print(model.inputs)
dim = int(model.input.shape[1])
# dim = 160

def draw_landmarks(image, landmarks, r=1, fill_color=(255,0,0,100)):
    draw = ImageDraw.Draw(image)
    for row in landmarks:
        x, y = row
        draw.ellipse((x-r, y-r, x+r, y+r), fill=fill_color)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
while(True):   
    #_, img = cap.read()
    img = cv2.imread(r"C:\Users\admin\Desktop\random\fn059t2afunaff001.png",cv2.IMREAD_COLOR)
    print(img.shape)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = Image.fromarray(img[y:y+h, x:x+w])

        dp = {
            'image': img,
            'landmarks': np.zeros((66,2))
        }
        dp = dataset.resize_mirror_datapoint(dp, dim, False)
        img = dp['image']

        landmark = np.array([[
            [0,0],
            [50,50],
            [100,50]
        ]])
        landmark = model.predict(np.array(img.convert("L"))[None, :, :, None] / 255.0)
        landmark = np.squeeze(landmark, axis=0)
        
        h, w = img.size
        img = img.resize((2*h, 2*w)).convert("L")
        landmark *= 2
        draw_landmarks(img, landmark, r=2)        
        
        # Convert RGB to BGR 
        img = np.array(img.convert("RGB")) 
        img = img[:, :, ::-1].copy() 
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
