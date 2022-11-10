import tensorflow as tf
import preprocess_main as data
import CropFace as face

import cv2

from keras.models import *
from keras.layers import *
from keras.callbacks import *

from tensorflow import keras
from sklearn.utils import class_weight

neuralNetworkArch = [13, 256, 256, 256, 256, 256, 128, 128, 128, 128, 64, 64, 32, 1]
# neuralNetworkArch = [13, 1024, 1024, 512, 256, 128, 64, 32, 16, 1]
batch_size = 64
initial_epoch = 50000
dropout = 0.05

MAIN_PATH = str(os.path.dirname(os.path.abspath(__file__)).split('source')[0])

MODEL_SAVE_PATH = MAIN_PATH + "models\\"



def load_data():
    try:
        # a = 1/0
        X = np.load(data.data_train_path + "X.npy")
        Y = np.load(data.data_train_path + "Y.npy")
        Xtest = np.load(data.data_train_path + "Xt.npy")
        Ytest = np.load(data.data_train_path + "Yt.npy")
        neuralNetworkArch[0] = X.shape[1]
        neuralNetworkArch[len(neuralNetworkArch) - 1] = 1
        print(neuralNetworkArch)

        return np.array(X), np.array(Y), np.array(Xtest), np.array(Ytest)
    except:
        print("Loading data")

    X, Y = data.getData()
    print(X.shape, "X")
    print(Y.shape, "Y")
    neuralNetworkArch[0] = X.shape[1]
    neuralNetworkArch[len(neuralNetworkArch) - 1] = 1
    print(neuralNetworkArch)

    print('X', X.shape)
    print('Y', Y.shape)
    trainNum = int(np.array(X).shape[0] * 0.9)


    Xtest = X[trainNum:]
    Ytest = Y[trainNum:]
    X = X[0:trainNum]
    Y = Y[0:trainNum]

    print(X.shape)
    print(Y.shape)
    print(Xtest.shape)
    print(Ytest.shape)

    for i in range(50):
        print(X[i])
    print('y0', Ytest[0])
    print('y1', Ytest[1])
    print('y2', Ytest[2])
    print('y3', Ytest[3])

    np.save(data.data_train_path + "X", X)
    np.save(data.data_train_path + "Y", Y)
    np.save(data.data_train_path + "Xt", Xtest)
    np.save(data.data_train_path + "Yt", Ytest)

    return np.array(X), np.array(Y), np.array(Xtest), np.array(Ytest)


def def_model(input_shape):
    model = Sequential()
    model.add(Input(input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout))
    for neu in neuralNetworkArch[1:len(neuralNetworkArch) - 1]:
        model.add(Dense(neu, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=dropout))

    model.add(Dense(1, activation='sigmoid'))

    return model


def train(X_train, Y_train, X_test, Y_test, learning_rate=0.001, epochs=1200, batch_size=64):
    print("X_train", X_train.shape)
    print("Y_train", Y_train.shape)
    print("X_test", X_test.shape)
    print("Y_test", Y_test.shape)

    model = def_model((neuralNetworkArch[0], ))

    model.summary()

    model.compile(keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision()])

    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
    class_weights = dict(zip(np.unique(Y_train), class_weights))
    print(class_weights)

    Reducing_LR = ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.5,
                                    patience=8,
                                    verbose=1, )

    checkpoint_path = MODEL_SAVE_PATH + 'model{epoch:08d}.h5'

    Checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_loss",
                                 save_best_only=False, period=1, )

    # Create Early Stopping Callback to monitor the accuracy
    Early_Stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)

    callbacks = [Early_Stopping, Reducing_LR, Checkpoint]

    preds = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, sample_weight=None)
    predicts = model.predict(X_test, batch_size=batch_size)
    pos_pred_cnt = 0
    pos_correct_cnt = 0
    label_pos = 0

    for i, pred in enumerate(predicts):
        if i < 20:
            print(round(pred[0], 3), end=", ")
        if Y_test[i] == 1:
            label_pos += 1
        if pred[0] > 0.5:
            pos_pred_cnt += 1
            if Y_test[i] == 1:
                pos_correct_cnt += 1

    if pos_pred_cnt == 0:
        pos_pred_cnt = 1

    print("True positive", (pos_correct_cnt, pos_pred_cnt))
    print("True positive %", (pos_correct_cnt / pos_pred_cnt))
    print("Label pos", label_pos)

    print()
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]) + "\n\n\n\n\n")

    history = model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=epochs,class_weight=class_weights,callbacks=callbacks, validation_data=(X_test, Y_test))
    history.model.save(MODEL_SAVE_PATH + 'my_model.h5')

    return model


def testModel(img=None, model=None, imshow=True):
    string = '..'

    faces, face_landmarks, foundface = face.getFacialLandmarks(img=img, showImg=False)
    f = faces.copy()
    for i in range(len(faces)):
        print("\n\nFace " + str(i + 1) + ":-")

        print("faceland", np.array(face_landmarks[i]).shape)
        landM = []
        print("Found faces") if foundface else print("No face found")
        for lan in face_landmarks[i][0]:
            landM.append(lan[0])
            landM.append(lan[1])
        dists = data.compDistance(landM)

        x = np.array([dists])
        print(np.array(x).shape)

        pred = model.predict(x)
        print("NN output:\n", str(pred))

        if pred[0] > 0.5:
            string = 'Happy ' + str(round(((pred[0][0] * 2) * 100) - 100)) + '%'
        else:
            string = ""

        if imshow:
            if pred[0] > 0.5:
                print("\n\nHappy face found!")
                string = 'Happy ' + str(round(((pred[0][0] * 2) * 100) - 100)) + '%'
            else:
                print("\n\nNo happy face found")
                string = ""

            f[i] = cv2.resize(f[i], (500, 500))
            f[i] = cv2.cvtColor(f[i], cv2.COLOR_GRAY2RGB)
            f[i] = cv2.rectangle(f[i], (50, 50), (450, 450), (0, 0, 255), )
            cv2.putText(f[i], string, (200, 50), cv2.QT_FONT_NORMAL, 1, (0, 0, 255), 2)
            cv2.imshow("Face " + str(i + 1), cv2.resize(f[i], (500, 500)))

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
    if string != '..':
        return str(string), f[0]

    return string, face


def softmaxToProbs(soft):
    z_exp = [np.math.exp(i[0]) for i in soft]
    sum_z_exp = sum(z_exp)
    return [(i / sum_z_exp) * 100 for i in z_exp]


if __name__ == '__main__':
    X, Y, Xtest, Ytest = load_data()
    train(X, Y, Xtest, Ytest, epochs=initial_epoch, batch_size=batch_size)
