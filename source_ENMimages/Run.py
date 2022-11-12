print("Initializing...")

import Train as train
from preprocess_main import data_train_path
from preprocess_main import IMAGE_SQUARE_SIZE
import cv2
import numpy as np
import traceback


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])



# resize and reduce the size using cv2.resize() to increase the speed
# cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

while True:
    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
        cv2.imshow("preview", frame)
        frame = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
        cv2.waitKey(1)
        break

# else:
#     rval = False

frame_ = ""
Xtest = np.load(data_train_path + "Xt.npy")
print("Shp", Xtest.shape)
model = train.def_model((IMAGE_SQUARE_SIZE, IMAGE_SQUARE_SIZE, 1))
model.load_weights(train.MODEL_SAVE_PATH + "my_model.h5")

while True:
    rval, frame = vc.read()
    # cv2.imshow("preview", frame)
    # time.sleep(10)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    string, face = train.testModel(model=model, imshow=False, img=frame)
    if string == '..':
        print("No face found")
        cv2.imshow("preview", frame)
        cv2.waitKey(1)
        continue

    try:
        face = cv2.resize(face, (500, 500))
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = cv2.rectangle(face, (50, 50), (450, 450), (0, 0, 255),)
        cv2.putText(face, string, (200, 50), cv2.QT_FONT_NORMAL, 1, (0, 0, 255), 2)
        cv2.imshow("preview", face)
        cv2.waitKey(1)
        print(str)
    except:
        traceback.print_exc()
        print()


cv2.destroyWindow("preview")
vc.release()
