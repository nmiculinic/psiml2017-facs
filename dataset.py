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
        fname = f"{base}.jpg"
        attrs = {
            k: v.value
            for k, v in zip(index, next(gen_rows))
        }
        attrs['image'] = np.array(Image.open(os.path.join(img_annot, fname)))
        attrs['landmarks'] = np.loadtxt(os.path.join(img_annot, f"{base}_landmarks.txt"))
        data.append(attrs)


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
    print(file_list_facs, rootdir_facs)
    for facsDIR in tqdm(file_list_facs):
        if str(facsDIR).endswith("_facs.txt"):
            imageDIR=facsDIR 
            imageDIR=str(imageDIR).replace("FACS","cohn-kanade-images")
            imageDIR=str(imageDIR).replace("_facs.txt",".png")

            emotionsDIR=facsDIR 
            emotionsDIR=str(emotionsDIR).replace("FACS","Emotion")
            emotionsDIR=str(emotionsDIR).replace("_facs.txt","_emotion.txt")

            landmarksDIR=facsDIR 
            landmarksDIR=str(landmarksDIR).replace("FACS","Landmarks")
            landmarksDIR=str(landmarksDIR).replace("_facs.txt","_landmarks.txt")
            
            while(not os.path.exists(imageDIR)):
                lhs1, rhs1 = os.path.basename(imageDIR).split(".", 1)
                lhs2, rhs2 = lhs1.split("000000", 1)
                imageDIR=str(imageDIR).replace(str(rhs2),str(int(rhs2)-1))

                #lhs1, rhs1 = os.path.basename(landmarksDIR).split(".", 1)
                #lhs2, rhs2 = lhs1.split("000000", 1)
                landmarksDIR=str(landmarksDIR).replace(str(rhs2),str(int(rhs2)-1))
            
            #with open(str(emotionsDIR), 'rb') as emotionsFile:
                #emotions = emotionsFile.read()

            with open(str(landmarksDIR), 'rb') as landmarksFile:
                landmarks = landmarksFile.read()
            
            sol.append({
                'image': np.array(Image.open(imageDIR)), 
                'facs': np.loadtxt(facsDIR),
                #'emotion': emotions,
                'landmarks': landmarks
            })
        pp.pprint(sol[:4])

if __name__ == "__main__":
    cohn_kanade_dataset(sys.argv[1])
    # faces_10k_dataset(os.path.join('.', 'data', '10k FACES'))


