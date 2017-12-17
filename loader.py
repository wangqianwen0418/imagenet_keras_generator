import tarfile
import os
import numpy as np
from PIL import Image
import random
import scipy.io as sio

import keras
# from keras.preprocessing import image as keras_img
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input

class imagenetData(object):
    def __init__(self, img_set, batch_size, target_size=(224, 224)):
        self.dataDir = '/data/imagenet'
        self.img_set = img_set
        self.batch_size = batch_size
        self.target_size = target_size
        
        if self.img_set == "train":
            self.imgs, self.labels = self.get_train_data()
        else:
            self.imgs, self.labels = self.get_val_data()
        self.num_images = len(self.imgs)
        self.cur = 0
        self.perm = np.random.permutation(np.arange(self.num_images))
        self.num_classes = 1000
    def get_val_labels(self):
        filepath = os.path.join(self.dataDir, "assets/val.txt")
        with open(filepath, 'r') as content:
            labels = [line for line in content]
            labels = labels[:-1] # the last one is ''
            labels = [str(line).split(' ')[1] for line in labels]
            labels = [int(i) for i in labels]
        return labels

    def get_train_data(self):
        '''
        # return:
        training images path and labels as list
        '''
        imgs = []
        labels = []
        filepath = os.path.join(self.dataDir, "train")
        # label_dict = self.get_train_labels()
        for dirpath, dirnames, filenames in os.walk(filepath):
            for i, dirname in enumerate(sorted(dirnames)):
                for sub_dirpath, sub_dirname, sub_filenames in os.walk(os.path.join(dirpath, dirname)):
                    for img in sub_filenames:
                        imgs.append(os.path.join(sub_dirpath, img))
                        wnid = img.split("_")[0]
                        # id = label_dict[wnid]
                        labels.append(i)
        return imgs, np.array(labels)

    def get_val_data(self):
        '''
        # return:
        validation images path and labels as list
        '''
        filepath = os.path.join(self.dataDir, "val")
        imgs = []
        labels = self.get_val_labels()
        for dirpath, dirnames, filenames in os.walk(filepath):
            for filename in sorted(filenames):
                imgs.append(os.path.join(dirpath, filename))
        return imgs, labels
    
    def _shuffle_roidb_idx(self):
        """Randomly permute the training roidb."""
        self.perm = np.random.permutation(np.arange(self.num_images))
        self.cur = 0

    def _get_next_minibatch_idx(self):
        """Return the roidb indices for the next minibatch."""
        if self.cur + self.batch_size >= self.num_images:
            self._shuffle_roidb_idx()
        db_idx = self.perm[self.cur:self.cur + self.batch_size]
        self.cur += self.batch_size
        return db_idx

    def _get_next_minibatch(self):
        minibatch_imgs = np.zeros(shape=((self.batch_size,) + self.target_size + (3,)))
        minibatch_labels = np.zeros(shape=(self.batch_size))
        db_idx = self._get_next_minibatch_idx()
        for i,idx in enumerate(db_idx):
            img = Image.open(self.imgs[idx])
            img = self.img_process(img)
            minibatch_imgs[i] = img
            minibatch_labels[i] = int(self.labels[idx])
        return minibatch_imgs, self.to_onehot(minibatch_labels)

    def generate(self):
        self.cur = 0
        while 1:
            x, y = self._get_next_minibatch()
            print(self.cur)
            yield (x, y)

    def img_process(self, img, s1= 256, s2= 480, interpolation="bilinear"):
        '''
        modify the images
        # arguments:
        img: a PIL Image object obtained by Image.open(path). 
        w, h: the size of width and height.
        s1, s2: training scale.
        # return: 
        arr: a modified image as a nparray.
        '''
        w, h = self.target_size
        _PIL_INTERPOLATION_METHODS = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'hamming': Image.HAMMING,
            'box': Image.BOX,
            "lanczos": Image.LANCZOS
        }
        
        if (img.mode!='RGB'):
            img = img.convert("RGB")
        # resize and corp
        if img.size != (w, h):
            short_edg = min(img.size)
            scale = random.randint(s1, s2)/short_edg
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            scaled_w = int(img.size[0]*scale)
            scaled_h = int(img.size[1]*scale)
            resized = img.resize((scaled_w, scaled_h), resample)
            xx = random.randint(0, scaled_w - w)
            yy = random.randint(0, scaled_h - h)
            crop = resized.crop((xx, yy, w+xx, h+yy))
        else: 
            crop = img

        arr = np.asarray(crop)
        arr = arr.astype(dtype=K.floatx())
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        return arr

    def to_onehot(self, y):
        """
        # Arguments
            y: class vector to be converted into a matrix
                (integers from 0 to num_classes).
            num_classes: total number of classes.
        # Returns
            A binary matrix representation of the input.
        """
        n = y.shape[0]
        categorical = np.zeros((n, self.num_classes))
        for i in range(n):
            categorical[int(i), int(y[i])] = 1
        return categorical
    
