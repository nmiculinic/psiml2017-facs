from tqdm import tqdm, trange
from PIL import Image
import numpy as np 
from openpyxl import load_workbook
import os
from pprint import PrettyPrinter

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

if __name__ == "__main__":
    faces_10k_dataset(os.path.join('.', 'data', '10k FACES'))

