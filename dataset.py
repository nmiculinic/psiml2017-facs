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
import random


pp = PrettyPrinter()

class Faces10K:
    def __init__(self, root_path, train_test_split=0.3):
        self.root_path = root_path
        self.train_test_split = train_test_split

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


def resize_datapoint(datapoint, PICTURE_SIZE):
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
    offset_image = np.zeros((PICTURE_SIZE,PICTURE_SIZE,3),np.uint8)
    offset_image= Image.fromarray(offset_image)
    offset_image.paste(img,(x_offset,y_offset))

    datapoint['image'] = offset_image.convert("L")
    datapoint['landmarks'] = landmarks
    return datapoint
    
class CohnKanade:
    def __init__(self, root_path, picture_size=128, train_test_split=0.7):
        self.root_path = root_path
        self.rootdir_image = os.path.join(root_path, "cohn-kanade-images")
        self.rootdir_facs = os.path.join(root_path, "FACS")
        self.rootdir_emotions = os.path.join(root_path, "Emotion")
        self.rootdir_landmarks = os.path.join(root_path, "Landmarks")
        file_list_image = [y for x in os.walk(self.rootdir_image) for y in glob(os.path.join(x[0], '*.png'))]

        split_point = int(len(file_list_image) * train_test_split)
        self.train_file_list = file_list_image[:split_point]
        self.test_file_list = file_list_image[split_point:]
        self.picture_size = picture_size

        self.logger = logging.getLogger(__name__)
        self.logger.info("Train set size %d", len(self.train_file_list))
        self.logger.info("Test set size %d",  len(self.test_file_list))

    def datapoint_for_file(self, image_fname):
        basename, ext = os.path.splitext(os.path.basename(image_fname))
        sub, seq, sequence_num = basename.split('_')

        attrs = {
            'image': Image.open(image_fname).copy()
        }

        facs_fname = os.path.join(self.rootdir_facs, sub, seq, "%s_%s_%s_facs.txt" % (sub, seq, sequence_num))
        if os.path.exists(facs_fname):
            attrs['facs'] = np.loadtxt(facs_fname)
        else:
            pass
            # print(facs_fname, "doesn't exist; skipping!")

        landmarks_fname = os.path.join(self.rootdir_landmarks, sub, seq, "%s_%s_%s_landmarks.txt" % (sub, seq, sequence_num))
        if os.path.exists(landmarks_fname):
            attrs['landmarks'] = np.loadtxt(landmarks_fname)
        else:
            print(landmarks_fname, "doesn't exist; skipping!")
            raise ValueError("Landmark must exist!")
        return attrs

    def datapoint_generator(self, flist, batch_size):
        curr_batch_x = []
        curr_batch_y = []
        flist = flist[:]
        while True:
            random.shuffle(flist)
            for image_fname in flist:
                try:
                    dp = self.datapoint_for_file(image_fname)
                    dp = resize_datapoint(dp, self.picture_size)
                    curr_batch_x.append(np.array(dp['image'])[:,:,None])
                    curr_batch_y.append(dp['landmarks'])
                    if len(curr_batch_x) == batch_size:
                        yield (
                            np.array(curr_batch_x),
                            np.array(curr_batch_y)
                        )
                        curr_batch_x = []
                        curr_batch_y = [] 
                except Exception as ex:
                    self.logger.error("In %s %s happend", image_fname, ex)

    def train_generator(self, batch_size):
        for x in self.datapoint_generator(self.train_file_list, batch_size):
            yield x

    def test_generator(self, batch_size):
        for x in self.datapoint_generator(self.test_file_list, batch_size):
            yield x

if __name__ == "__main__":
    cohn_kanade_dataset(sys.argv[1])
    # faces_10k_dataset(os.path.join('.', 'data', '10k FACES'))


