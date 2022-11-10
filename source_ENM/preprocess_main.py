import CropFace as crop_face
import os
import cv2
import numpy as np
import pandas as pandas
import math
import traceback
from PIL import Image
import random
# from scipy.misc import toimage


MAIN_PATH = str(os.path.dirname(os.path.abspath(__file__)).split('source')[0])

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
######################################################################
data_train_path = MAIN_PATH + "data_ENM\\"
######################################################################

df = pandas.read_excel(data_train_path + 'fer2013.xlsx')


def pixelToImage(pixels, name, len=48):
    pixels = list(str(pixels).split(" "))
    img_ar = []
    row = []
    for i in range(pixels.__len__()):
        row.append(int(pixels[i]))
        if (((i + 1) % len) == 0):
            img_ar.append(row)
            row = []
    img = np.array(img_ar)
    print(img.shape)
    cv2.imwrite(name + ".png", img)
    return img_ar


# X as face and Y as the landmarks(cords)
def create_data():
    try:
        # a = 1/0
        landmarks_X = (np.load(data_train_path + "emotion_data_X.npy")).tolist()
        labels_Y = (np.load(data_train_path + "emotion_data_Y.npy")).tolist()
        print(np.array(landmarks_X).shape)
        print(np.array(labels_Y).shape)
        return landmarks_X, labels_Y
    except:
        traceback.print_exc()
        landmarks_X = []
        labels_Y = []
        i_ = 0
        for i in range(35800):
            print(i, i_)
            pixelToImage(all_pixels[i], MAIN_PATH + "faces/" + str(i))
            landM = []

            face_cords, face_landmarks, foundface = crop_face.getFacialLandmarks(image_dir=MAIN_PATH + "faces/" + str(i) + '.png',
                                                                                 showImg=False)
            # print("f", foundface)
            if foundface:
                for lan in face_landmarks[0][0]:
                    landM.append(lan[0])
                    landM.append(lan[1])
                landmarks_X.append(landM)
                if labels[i] == 3:
                    labels_Y.append(1)
                else:
                    labels_Y.append(0)
                print(labels[i])
                i_ += 1
            if (i % 100 == 0) and i > 0:
                np.save(data_train_path + "emotion_data_X", landmarks_X)
                np.save(data_train_path + "emotion_data_Y", labels_Y)
        np.save(data_train_path + "emotion_data_X", landmarks_X)
        np.save(data_train_path + "emotion_data_Y", labels_Y)

        return (landmarks_X, labels_Y)


def getAsSoftmax(labels, count):
    full_soft = []
    labelCnt = []
    for i in range(7):
        labelCnt.append(0)
    for i in range(len(labels)):
        soft = []
        labelCnt[labels[i]] += 1
        for j in range(count):
            if labels[i] == j:
                soft.append(1.0)
            else:
                soft.append(0.0)
        full_soft.append(soft)
    # print(labelCnt)
    return full_soft


def compDistance(landmarks):
    dists = []
    X = []
    Y = []
    for i in range(0, len(landmarks) - 1, 2):
        X.append(landmarks[i])
        Y.append(landmarks[i + 1])
    for i in range(len(X)):
        for j in range(len(X)):
            x1 = X[i]
            x2 = X[j]
            y1 = Y[i]
            y2 = Y[j]
            dist = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            dists.append(dist / 60)
    return dists


def convInpts(X):
    x = []
    X = np.array(X)
    for i in range(len(X[0])):
        xx = []
        for j in range(len(X)):
            xx.append(X[j][i])
        x.append(xx)
    return x


def shuffel(x, y, seed=4):
    random.Random(seed).shuffle(x)
    random.Random(seed).shuffle(y)


def getData():
    print("Loading data, please wait..")
    global all_pixels, labels
    all_pixels = list(df["pixels"].values)
    labels = list(df["label"].values)
    landm_X, label_Y = create_data()
    X = []
    Y = label_Y

    for i in range(len(landm_X)):
        x = landm_X[i]
        X.append(compDistance(x))

    shuffel(X, Y)

    return np.array(X), np.array(Y)

