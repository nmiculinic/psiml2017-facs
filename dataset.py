from tqdm import tqdm, trange
from PIL import Image
import numpy as np 
from openpyxl import load_workbook
import os
from pprint import PrettyPrinter
import argparse
import sys
from glob import glob
import logging


pp = PrettyPrinter()

def faces_10k_dataset(root_path):
    wb = load_workbook(
            filename=os.path.join(root_path, "Full Attribute Scores", "psychology attributes","psychology-attributes.xlsx"),
            read_only=True
    )
    ws = wb['Final Values']
    gen_rows = ws.rows
    index = [x.value for x in next(gen_rows)]
    
    img_annot = os.path.join(root_path, "Face Annotations", "Images and Annotations")
    data = []
    for base in trange(1, 2222 + 1):
        fname = "{}.jpg".format(base)
        attrs = {
            k: v.value
            for k, v in zip(index, next(gen_rows))
        }
        attrs['image'] = Image.open(os.path.join(img_annot, fname)).copy()
        attrs['landmarks'] = np.loadtxt(os.path.join(img_annot, "{}_landmarks.txt".format(base)))
        data.append(attrs)
    return data 


def cohn_kanade_dataset(root_path, max_num=1000000):
    rootdir_image = os.path.join(root_path, "cohn-kanade-images")
    rootdir_facs = os.path.join(root_path, "FACS")
    rootdir_emotions = os.path.join(root_path, "Emotion")
    rootdir_landmarks = os.path.join(root_path, "Landmarks")

    file_list_image = [y for x in os.walk(rootdir_image) for y in glob(os.path.join(x[0], '*.png'))]

    sol = []
    for image_fname in tqdm(file_list_image[:max_num]):
        try:
            basename, ext = os.path.splitext(os.path.basename(image_fname))
            sub, seq, sequence_num = basename.split('_')

            attrs = {
                'image': Image.open(image_fname).copy()
            }

            facs_fname = os.path.join(rootdir_facs, sub, seq, "%s_%s_%s_facs.txt" % (sub, seq, sequence_num))
            if os.path.exists(facs_fname):
                attrs['facs'] = np.loadtxt(facs_fname)
            else:
                pass
                # print(facs_fname, "doesn't exist; skipping!")

            landmarks_fname = os.path.join(rootdir_landmarks, sub, seq, "%s_%s_%s_landmarks.txt" % (sub, seq, sequence_num))
            if os.path.exists(landmarks_fname):
                attrs['landmarks'] = np.loadtxt(landmarks_fname)
            else:
                print(landmarks_fname, "doesn't exist; skipping!")
                raise ValueError("Landmark must exist!")

            sol.append(attrs)
        except Exception:
            print("ERROR", image_fname)
            raise
    logging.info(
            "CK has %d images, %d has FACS info", 
            len(sol),
            len(list(filter(lambda x: 'facs' in x.keys(), sol)))
    )
    return sol

def resize_picture(self, fname, lname):
    img = Image.open(fname)
    landmarks = np.loadtxt(lname)
    width, height = img.size[:2]
    PICTURE_SIZE = 128

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
    offset_image = np.zeros((PICTURE_SIZE,PICTURE_SIZE,3),np.uint8)
    offset_image= Image.fromarray(offset_image)
    offset_image.paste(img,(x_offset,y_offset))
    #offset_image.save(imOut)
    #np.savetxt(landOut,landmarks)
    return offset_image, landmarks

if __name__ == "__main__":
    cohn_kanade_dataset(sys.argv[1])
    # faces_10k_dataset(os.path.join('.', 'data', '10k FACES'))


