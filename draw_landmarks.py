import cv2
from pathlib import Path
import numpy as np
import csv
import os
from tqdm import tqdm, trange



root_path = "C:\\Users\\admin\\Desktop\\DATASETS\\10k FACES"
write_path = "C:\\Users\\admin\\Desktop\\resized pics"
imageDIR = os.path.join(root_path, "Face Annotations", "Images and Annotations")
for base in trange(1, 2222 + 1):
    pic_name = "{}.jpg".format(base)
    original_image = cv2.imread(os.path.join(imageDIR, pic_name),0)
    landmarks_name = "{}_landmarks.txt".format(base)
    with open(os.path.join(imageDIR, landmarks_name),newline='') as landmarksFILE:
        #landmarksReader = csv.reader(landmarksFILE,delimeter=' ', quotechar='|')
        x =[]
        y =[]
        x,y = zip(*[l.split() for l in landmarksFILE])
        # for row in landmarksReader:
        #     x.append(row[0])
        #     y = np.append(row[1])

        leftEyebrow = []
        faceShape = []
        rightEyebrow = []
        nose = []
        leftEye = []
        rightEye = []
        upperLip = []
        bottomLip = []
        ####################################
        # img = np.zeros((768, 1024, 3), dtype='uint8')

        # points = np.array([[10, 41], [6, 32], [96, 88], [58, 85]])
        # cv2.polylines(img, [points], 1, (255,255,255))

        # winname = 'example'
        # cv2.namedWindow(winname)
        # cv2.imshow(winname, img)
        # cv2.waitKey()
        # cv2.destroyWindow(winname)
        #####################################
        for i in range (8):
            leftEyebrow.append(( int(float(x[i])),( int(float(y[i])))))
        #print(leftEyebrow)
        for i in range (9,24):
            faceShape.append(( int(float(x[i])),( int(float(y[i])))))
        for i in range (25,32):
            rightEyebrow.append(( int(float(x[i])),( int(float(y[i])))))
        for i in range (33,45):
            nose.append(( int(float(x[i])),( int(float(y[i])))))
        for i in range (46,53):
            leftEye.append(( int(float(x[i])),( int(float(y[i])))))
        for i in range (54,61):
            rightEye.append(( int(float(x[i])),( int(float(y[i])))))
        for i in range (62,70):
            upperLip.append(( int(float(x[i])),( int(float(y[i])))))
        for i in range (71,77):
            bottomLip.append(( int(float(x[i])),( int(float(y[i])))))

        leftEyebrow = np.array(leftEyebrow,dtype='int32')
        faceShape = np.array(faceShape,dtype='int32')
        rightEyebrow = np.array(rightEyebrow,dtype='int32')
        nose = np.array(nose,dtype='int32')
        leftEye = np.array(leftEye,dtype='int32')
        rightEye = np.array(rightEye,dtype='int32')
        upperLip = np.array(upperLip,dtype='int32')
        bottomLip = np.array(bottomLip,dtype='int32')

        cv2.polylines(original_image, [leftEyebrow], 1, (255,255,255))
        cv2.polylines(original_image, [faceShape], 1, (255,255,255))
        cv2.polylines(original_image, [rightEyebrow], 1, (255,255,255))
        cv2.polylines(original_image, [nose], 1, (255,255,255))
        cv2.polylines(original_image, [leftEye], 1, (255,255,255))
        cv2.polylines(original_image, [rightEye], 1, (255,255,255))
        cv2.polylines(original_image, [upperLip], 1, (255,255,255))
        cv2.polylines(original_image, [bottomLip], 1, (255,255,255))

        # cv2.fillPoly(original_image, [points], 1)#, (255,255,255))
        # cv2.fillPoly(original_image, [points], 1)#, (255,255,255))
        # cv2.fillPoly(original_image, [points], 1)#, (255,255,255))
        # cv2.fillPoly(original_image, [points], 1)#, (255,255,255))
        # cv2.fillPoly(original_image, [points], 1)#, (255,255,255))
        # cv2.fillPoly(original_image, [points], 1)#, (255,255,255))
        # cv2.fillPoly(original_image, [points], 1)#, (255,255,255))
        # cv2.fillPoly(original_image, [points], 1)#, (255,255,255))
        cv2.imshow("original", original_image)
        cv2.waitKey(0)

