print("Initializing...")

import Train as train
from preprocess_main import data_train_path
from preprocess_main import IMAGE_SQUARE_SIZE
import cv2
import numpy as np
import os

Xtest = np.load(data_train_path + "Xt.npy")
print("Shp", Xtest.shape)
model = train.def_model((IMAGE_SQUARE_SIZE, IMAGE_SQUARE_SIZE, 1))
model.load_weights(train.MODEL_SAVE_PATH + "my_model.h5")

file_name = str(input("Enter the file name from the input_files folder of the project: "))

image = cv2.imread(train.MAIN_PATH + 'input_files\\' + file_name, flags=cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (150, 200))
# resize and reduce the size using cv2.resize() to increase the speed
string, face = train.testModel(model=model, imshow=True, img=image)

