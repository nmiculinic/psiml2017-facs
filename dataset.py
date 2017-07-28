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
import warnings

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


def crop_datapoint(datapoint, picture_size):
    """Crops datapoint

    :param datapoint: Datapoint to transform
    :param picture_size: Output picture square dimension
    """
    img = datapoint['image']
    w, h = img.size
    crop_x = random.randint(0, w - picture_size)
    crop_y = random.randint(0, h - picture_size)
    img = img.crop((crop_x, crop_y, picture_size + crop_x, picture_size + crop_y))    
    datapoint['image'] = img
    datapoint['landmarks'] =np.array([crop_x, crop_y]).reshape((1,2))
    return datapoint


def resize_mirror_datapoint(datapoint, picture_size):
    """Resize and randomly horizontally mirror
    """
    img = datapoint['image']
    landmarks = datapoint['landmarks']
    width, height = img.size[:2]

    if height>width:
        resized_height = picture_size
        resized_width = round(picture_size*width/height)
        x_offset=round((picture_size-resized_width)/2)
        y_offset=0
        assert x_offset >= 0
        landmarks[:, 0] = landmarks[:, 0] * (picture_size/height) + x_offset
        landmarks[:, 1] = landmarks[:, 1] * (picture_size/height)
    else:  # height<= width:
        resized_height = round(picture_size*height/width)
        resized_width = picture_size
        x_offset=0
        y_offset=round((picture_size-resized_height)/2)
        assert y_offset >= 0
        landmarks[:, 0] = landmarks[:, 0] * (picture_size/width)
        landmarks[:, 1] = landmarks[:, 1] * (picture_size/width) + y_offset

    img = img.resize((resized_width,resized_height), Image.ANTIALIAS)
    img = img.crop((0, 0, picture_size, picture_size))

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        landmarks[:, 0] = picture_size - landmarks[:,0]
    
    offset_image = Image.new("RGB", (picture_size, picture_size), color='black')
    offset_image.paste(img,(x_offset,y_offset))
    datapoint['image'] = offset_image
    datapoint['landmarks'] = landmarks
    return datapoint


def rotate_datapoint(datapoint, angle):
    theta = np.radians(angle)
    offset_image = datapoint['image'].rotate(angle, resample=Image.BICUBIC)
    landmarks = datapoint['landmarks']
    rotMatrix = np.array([
        [np.cos(theta),-np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    landmarks_offset = np.array(datapoint['image'].size).reshape((1, 2)) / 2.0
    landmarks -= landmarks_offset
    landmarks = np.dot(landmarks, rotMatrix)
    landmarks += landmarks_offset

    datapoint['landmarks'] = landmarks
    datapoint['image'] = offset_image
    return datapoint

def preprocess_image(datapoint, picture_size, crop_window, max_angle):
    """Preprocesses the image, tranformation only
    """
    datapoint = resize_mirror_datapoint(datapoint, picture_size + crop_window)
    datapoint = rotate(datapoint, -max_angle + 2*random.random()*max_angle)
    datapoint = resize_mirror_datapoint(datapoint, picture_size)
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
                    curr_batch_x.append(np.array(dp['image'])[:,:,None] / 255.0)
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

class Pain:
    def __init__(self, root_path, picture_size=128, crop_window=10, max_angle=15.0, train_test_split=0.7):
        self.root_path = root_path
        self.rootdir_image = os.path.join(root_path, "Images")
        self.rootdir_facs = os.path.join(root_path, "Frame_Labels", "FACS")
        #self.rootdir_emotions = os.path.join(root_path, "Emotion")
        self.rootdir_landmarks = os.path.join(root_path, "AAM_landmarks")
        file_list_image = [y for x in os.walk(self.rootdir_image) for y in glob(os.path.join(x[0], '*.png'))]

        split_point = int(len(file_list_image) * train_test_split)
        self.train_file_list = file_list_image[:split_point]
        self.test_file_list = file_list_image[split_point:]
        self.picture_size = picture_size
        self.crop_window = crop_window 
        self.max_angle = max_angle

        self.logger = logging.getLogger(__name__)
        self.logger.info("Train set size %d", len(self.train_file_list))
        self.logger.info("Test set size %d",  len(self.test_file_list))

    def datapoint_for_file(self, image_fname):
        #"Z:\data\pain\Images\042-ll042\ll042t1aaaff\ll042t1aaaff001.png"
        # self.logger.info("basename %s, sep%s", basename, os.sep)
        basename = os.path.splitext(image_fname)[0]
        *args, b1, b2, b3 = basename.split(os.sep)
        
        attrs = {
            'image': Image.open(image_fname).copy().convert("L")
        }

        #"Z:\data\pain\Frame_Labels\FACS\042-ll042\ll042t1aaaff\ll042t1aaaff001_facs.txt"
        facs_fname = os.path.join(self.rootdir_facs, b1, b2, b3 + "_facs.txt")
        if os.path.exists(facs_fname):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                attrs['facs'] = np.loadtxt(facs_fname)
                if attrs['facs'].shape == (0,):
                    del attrs['facs']
        else:
            pass

        #"Z:\data\pain\AAM_landmarks\042-ll042\ll042t1aaaff\ll042t1aaaff001_aam.txt"
        landmarks_fname = os.path.join(self.rootdir_landmarks, b1, b2, b3 +  "_aam.txt")
        if os.path.exists(landmarks_fname):
            attrs['landmarks'] = np.loadtxt(landmarks_fname)
            if attrs['landmarks'].shape != (66, 2):
                raise ValueError("Empty landmarks!")
        else:
            self.logger.error("Missing landmark %s", landmarks_fname)
            raise ValueError("Landmark must exist!")
        return attrs

    def datapoint_generator(self, flist, batch_size):
        curr_batch_x = []
        curr_batch_y = []
        flist = flist[:]
        total = 0
        facs = 0
        while True:
            random.shuffle(flist)
            for image_fname in flist:
                try:
                    dp = self.datapoint_for_file(image_fname)
                    dp = preprocess_image(dp, 
                        picture_size=self.picture_size,
                        crop_window=self.crop_window,
                        max_angle=self.max_angle
                    )
                    if 'facs' in dp:
                        facs += 1
                    total += 1
                    curr_batch_x.append(np.array(dp['image'].convert("L"))[:,:,None] / 255.0)
                    curr_batch_y.append(dp['landmarks'])
                    if len(curr_batch_x) == batch_size:
                        yield (
                            np.array(curr_batch_x),
                            np.array(curr_batch_y)
                        )
                        curr_batch_x = []
                        curr_batch_y = [] 

                    if total in [10, 20, 100] or total % 20000 == 0:
                        self.logger.info("Read %5d points Facs has %f%% datapoints", total, 100 * facs / total)
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


