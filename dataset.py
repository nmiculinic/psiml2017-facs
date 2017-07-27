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




if __name__ == "__main__":
    cohn_kanade_dataset(sys.argv[1])
    # faces_10k_dataset(os.path.join('.', 'data', '10k FACES'))


