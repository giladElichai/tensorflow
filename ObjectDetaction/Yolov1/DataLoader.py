import os 

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence) :
  
    def __init__(self, csv_file, images_dir, labels_dir, image_size, batch_size, grid_size, B, num_classes, shuffle=True) :

        self.data =  np.array(pd.read_csv(csv_file))
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.S = grid_size # S
        self.B = B
        self.C = num_classes # C
        self.shuffle = shuffle

        self.data_size = self.data.shape[0]
        for i in range(self.data.shape[0]):
            self.data[i][0] = os.path.join(images_dir, self.data[i][0])
            self.data[i][1] = os.path.join(labels_dir, self.data[i][1])

        self.indexes = np.arange(self.data_size)

        self.on_epoch_end()


    def __len__(self) :
        return int(np.floor(self.data_size / self.batch_size))

    def __getitem__(self, index):
        
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_indexes = [k for k in batch_indexes]

        X, y = self.data_generation(batch_indexes)

        return X, y

    def data_generation(self, batch_indexes):

        batch_data = self.data[batch_indexes]

        train_image = []
        train_label = []

        for i in range(self.batch_size):

            img_path = batch_data[i][0]
            label = batch_data[i][1]
            image, label_matrix = self.read_data(img_path, label)
            train_image.append(image)
            train_label.append(label_matrix)

        return np.array(train_image), np.array(train_label)


    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)    
    

    def read_data(self, image_path, label):

        img = load_img(image_path ,target_size=self.image_size)
        img = img_to_array(img, dtype="float32")/255.

        label_matrix = np.zeros((self.S, self.S, self.C + 5 * self.B), dtype="float32")
        with open(label) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                i = int(self.S * y)
                j = int(self.S * x)

                x_cell =  self.S * x - j
                y_cell = self.S * y - i

                width_cell = width * self.S
                height_cell = height * self.S
                
                if label_matrix[i, j, 20] == 0:
                    label_matrix[i, j, 20] = 1 # Set that there exists an object
                    label_matrix[i, j, 21:25] = [x_cell, y_cell, width_cell, height_cell] # Box coordinates
                    label_matrix[i, j, class_label] = 1 # Set one hot encoding for class_label

                # if label_matrix[i, j, 24] == 0:
                #     label_matrix[i, j, 24] = 1 # Set that there exists an object
                #     label_matrix[i, j, 20:24] = [x_cell, y_cell, width_cell, height_cell] # Box coordinates
                #     label_matrix[i, j, class_label] = 1 # Set one hot encoding for class_label

        return img, label_matrix
        




def test():

    csv_file = "D:\\programing\\DataSets\\ObjectDetection\\pascalVoc\\train.csv"
    img_dir = "D:\\programing\\DataSets\\ObjectDetection\\pascalVoc\\images"
    labels_dir = "D:\\programing\\DataSets\\ObjectDetection\\pascalVoc\\labels"
    val_csv_file = "D:\\programing\\DataSets\\ObjectDetection\\pascalVoc\\test.csv"

    train_dg = DataGenerator(csv_file, img_dir, labels_dir, (448, 448), 4, 7, 2, 20)
    train_val = DataGenerator(val_csv_file, img_dir, labels_dir, (448, 448), 16, 7, 2, 20, shuffle=False)

    x_train, y_train = train_dg.__getitem__(0)
    x_val, y_val = train_val.__getitem__(0)

    print(x_train.shape)
    print(y_train.shape)

    print(x_val.shape)
    print(y_val.shape)

    x = 1 


if __name__ == "__main__":
    test()