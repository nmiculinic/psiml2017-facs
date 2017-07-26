from tqdm import tqdm, trange
from PIL import Image
import numpy as np 
from openpyxl import load_workbook
import os
from pprint import PrettyPrinter
import argparse
import sys
from pathlib import Path


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
        attrs['image'] = np.array(Image.open(os.path.join(img_annot, fname)))
        attrs['landmarks'] = np.loadtxt(os.path.join(img_annot, "{}_landmarks.txt".format(base)))
        data.append(attrs)
    return data 


def cohn_kanade_dataset(root_path):
    rootdir_image = Path(os.path.join(root_path, "cohn-kanade-images"))
    rootdir_facs = Path(os.path.join(root_path, "FACS"))
    rootdir_emotions = Path(os.path.join(root_path, "Emotion"))
    rootdir_landmarks = Path(os.path.join(root_path, "Landmarks"))

    # For absolute paths instead of relative the current dir
    file_list_image = [f for f in rootdir_image.resolve().glob('**/*') if f.is_file()]
    file_list_facs = [f for f in rootdir_facs.resolve().glob('**/*') if f.is_file()]
    file_list_emotions = [f for f in rootdir_emotions.resolve().glob('**/*') if f.is_file()]
    file_list_landmarks = [f for f in rootdir_landmarks.resolve().glob('**/*') if f.is_file()]

    sol = []
    for facs_fname in tqdm(file_list_facs):
        try:
            if str(facs_fname).endswith("_facs.txt"):
                sub, seq, sequence_num, _ = os.path.basename(facs_fname).split('_')
                image_fname = os.path.join(rootdir_image, sub, seq, "%s_%s_%s.png" % (sub, seq, sequence_num))
                landmarks_fname = os.path.join(rootdir_landmarks, sub, seq, "%s_%s_%s_landmarks.txt" % (sub, seq, sequence_num))
                if not os.path.exists(image_fname):
                    print(image_fname, "doesn't exist; skipping!")
                    continue

                #with open(str(emotionsDIR), 'rb') as emotionsFile:
                    #emotions = emotionsFile.read()

                sol.append({
                    'image': np.array(Image.open(image_fname)), 
                    'facs': np.loadtxt(facs_fname),
                    #'emotion': emotions,
                    'landmarks': np.loadtxt(landmarks_fname)
                })
        except Exception:
            print("ERROR", facs_fname)
            raise
    return sol

if __name__ == "__main__":
    cohn_kanade_dataset(sys.argv[1])
    # faces_10k_dataset(os.path.join('.', 'data', '10k FACES'))


