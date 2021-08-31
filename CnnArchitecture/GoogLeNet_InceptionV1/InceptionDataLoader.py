

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical ,Sequence
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from pathlib import Path


class DataGenerator(Sequence):
    
    def __init__(self, folder, batch_size=16, dim=(224,224,3), shuffle=True, augmentation=True):
        self.dim = dim
        self.folder = folder
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.augmentation = augmentation
        
        self.augmentor = None
        if augmentation:
            self.augmentor = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.2,
                zoom_range=0.2,
                brightness_range=(0.9,0.1),
                horizontal_flip=True,
                fill_mode='nearest')

        self.get_files_pairs()

        self.on_epoch_end()

    def get_files_pairs(self):

        dataset_path = Path(self.folder)
        list(dataset_path.iterdir())
        dirs = dataset_path.iterdir()

        self.pairs = []

        for i, dir in enumerate(dirs):
            files = []
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                files.extend(list((dir).glob(ext)))

            for f in files:
                self.pairs.append([f, i])    

        self.num_classes = i+1

        return i


    def __len__(self):
        return int(np.floor(len(self.pairs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [k for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.pairs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        
        batch_imgs = list()
        batch_labels = list()

        for i in list_IDs_temp:

            color_mode = "rgb" if self.dim[2] == 3 else "grayscale"

            img = load_img(self.pairs[i][0] ,target_size=self.dim, color_mode=color_mode)
            img = img_to_array(img)/255.
            batch_imgs.append(img)

            label = self.pairs[i][1]
            batch_labels.append(label)

        if self.augmentation:
            X_gen = self.augmentor.flow(np.array(batch_imgs), batch_size=self.batch_size, shuffle=False)
            batch_imgs = next(X_gen)/255.
            
        batch_labels = np.array(batch_labels)

        return np.array(batch_imgs) ,[batch_labels, batch_labels, batch_labels]



if __name__ == "__main__":

    #dg = DataGenerator("D:\\programing\\DataSets\\Classification\\Covid19-dataset\\train")

    dg = DataGenerator("D:\\programing\\DataSets\\Classification\\IntelImageClassification/seg_train/seg_train")

    data = dg.__getitem__(1)

    x = 1 