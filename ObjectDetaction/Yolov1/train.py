import os
import os.path
import sys 

from pathlib import Path
from glob import glob
from random import choice

import matplotlib.pyplot as plt
#import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


from DataLoader import DataGenerator
from YoloV1Model import GetYoloV1Model
from YoloV1Loss import yolo_loss

def decay(epoch, steps=100):
    initial_lrate = 0.0001
    drop = 0.98
    epochs_drop = 10
    lrate = initial_lrate * np.math.pow(drop, np.math.floor((1+epoch)/epochs_drop))
    return lrate

if __name__ == "__main__":


    batch_size = 16
    csv_file = "D:\\programing\\DataSets\\ObjectDetection\\pascalVoc\\train.csv"
    img_dir = "D:\\programing\\DataSets\\ObjectDetection\\pascalVoc\\images"
    labels_dir = "D:\\programing\\DataSets\\ObjectDetection\\pascalVoc\\labels"
    val_csv_file = "D:\\programing\\DataSets\\ObjectDetection\\pascalVoc\\test.csv"

    train_dg = DataGenerator(csv_file, img_dir, labels_dir, (448, 448), batch_size, 7, 2, 20)
    train_val = DataGenerator(val_csv_file, img_dir, labels_dir, (448, 448), batch_size, 7, 2, 20, shuffle=False)

    x_train, y_train = train_dg.__getitem__(0)
    x_val, y_val = train_val.__getitem__(0)
    print(x_train.shape)
    print(y_train.shape)

    print(x_val.shape)
    print(y_val.shape)

    mcp = ModelCheckpoint('best_weight.hdf5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    lr_sc = LearningRateScheduler(decay, verbose=1)
    rl = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=5,verbose=1,mode="max",min_lr=0.000001)

    initial_lrate = 0.001

    model = GetYoloV1Model(7, 2, 20, input_shape = (448, 448, 3))
    model.compile(loss=yolo_loss ,optimizer='adam')

    d = 1
    
    history = model.fit(x=train_dg,
            epochs = 100,
            validation_data = train_val,
            callbacks=[mcp, lr_sc, rl]
            )
    
    model.save("yolov1_model")

    x = 0