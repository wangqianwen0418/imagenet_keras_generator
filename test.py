import keras
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions
from keras.metrics import top_k_categorical_accuracy

from read_tar import imagenetData

import os

data_dir = "/data/imagenet"
batch_size = 16

vgg = VGG19(weights='imagenet')
resnet = ResNet50(weights='imagenet')
vgg.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[top_k_categorical_accuracy])
resnet.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[top_k_categorical_accuracy])


data = imagenetData(img_set='train', batch_size=batch_size)

i = 0
# while i < data.num_images//batch_size:
while i < 3:
    x, y = data._get_next_minibatch()
    preds = resnet.predict_on_batch(x)
    score = resnet.test_on_batch(x,y)
    # print("preds", decode_predictions(preds, top=3))
    # print("truth", decode_predictions(y, top=1))
    print("score", score)
    i+=1
