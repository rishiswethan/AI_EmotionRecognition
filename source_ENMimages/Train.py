import numpy as np
import tensorflow as tf
import preprocess_main as data
import CropFace as face

import cv2

from keras.models import *
from keras.layers import *
from keras.callbacks import *

from tensorflow import keras
from sklearn.utils import class_weight
import traceback
from PIL import Image
import matplotlib.pyplot as plt

import Train_2 as train_2_prg

import keras.backend as K
import math

# neuralNetworkArch = [13, 256, 256, 256, 256, 256, 128, 128, 128, 128, 64, 64, 32, 1]
neuralNetworkArch = [13, 1024, 512, 256, 256, 256, 128, 128, 128, 128, 64, 64, 32, 1]
# neuralNetworkArch = [13, 1024, 1024, 512, 256, 128, 64, 32, 16, 1]
batch_size = 64
initial_epoch = 50000
drop_out = 0.0
l1_l2_reg = 0.0

MAIN_PATH = str(os.path.dirname(os.path.abspath(__file__)).split('source')[0])

MODEL_SAVE_PATH = MAIN_PATH + "models_ENMimages\\"



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
    X = X / 255.0

    print(X.shape, "X")
    print(Y.shape, "Y")
    neuralNetworkArch[0] = X.shape[1]
    neuralNetworkArch[len(neuralNetworkArch) - 1] = 1
    print(neuralNetworkArch)

    print('X', X.shape)
    print('Y', Y.shape)
    trainNum = int(np.array(X).shape[0] * 0.85)

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


def def_model(input_shape, input_shape_2):
    # model = Sequential()

    X_input_1 = Input(input_shape)
    X_input_2 = Input(input_shape_2)
    X = BatchNormalization()(X_input_1)

    # CNN1
    X = Conv2D(32, (3, 3), activation='relu',
               # kernel_regularizer=keras.regularizers.l1_l2(l1_l2_reg),
               )(X)

    # X = BatchNormalization()(X)

    X = Conv2D(64, (3, 3), activation='relu',
               # kernel_regularizer=keras.regularizers.l1_l2(l1_l2_reg),
               )(X)

    X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)

    # X = BatchNormalization()(X)

    X = Conv2D(64, (3, 3), activation='relu',
               # kernel_regularizer=keras.regularizers.l1_l2(l1_l2_reg),
               )(X)

    # X = BatchNormalization()(X)

    X = Conv2D(128, (3, 3), activation='relu', padding='same',
               # kernel_regularizer=keras.regularizers.l1_l2(l1_l2_reg),
               )(X)

    X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
    #
    # X = BatchNormalization()(X)

    X = Conv2D(128, (3, 3), activation='relu',
               # kernel_regularizer=keras.regularizers.l1_l2(l1_l2_reg),
               )(X)

    # X = BatchNormalization()(X)

    X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)

    X = BatchNormalization()(X)

    X = Flatten()(X)

    X_2 = BatchNormalization()(X_input_2)

    X = Concatenate()([X, X_2])

    for neu in neuralNetworkArch[1:len(neuralNetworkArch) - 1]:
        X = Dense(neu, activation='relu',
                  # kernel_regularizer=keras.regularizers.l1_l2(l1_l2_reg),
                  )(X)

        X = Dropout(rate=drop_out)(X)

        X = BatchNormalization()(X)

    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=[X_input_1, X_input_2], outputs=X, name='Model')

    return model


def train(X_train, Y_train, X_test, Y_test, train_2, learning_rate=0.0001, epochs=1200, batch_size=64, display_model_graphs=True, display_model_stats_only=False):
    X_train_2, Y_train_2, X_test_2, Y_test_2 = train_2

    print("X_train", X_train.shape)
    print("Y_train", Y_train.shape)
    print("X_test", X_test.shape)
    print("Y_test", Y_test.shape)

    model = def_model((data.IMAGE_SQUARE_SIZE, data.IMAGE_SQUARE_SIZE, 1), (Xtest_2.shape[1], ))

    model.summary()

    model.compile(keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(), 'accuracy'])

    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
    class_weights = dict(zip(np.unique(Y_train), class_weights))
    print(class_weights)

    Reducing_LR = ReduceLROnPlateau(monitor='loss',
                                    factor=0.2,
                                    patience=5,
                                    verbose=1, )

    checkpoint_path = MODEL_SAVE_PATH + 'model{epoch:08d}.h5'

    Checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_precision",
                                 save_best_only=False, period=1, )

    # Create Early Stopping Callback to monitor the accuracy
    Early_Stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=1)

    callbacks = [Early_Stopping, Reducing_LR, Checkpoint]

    # X_test_ = X_test.copy()

    X_train = X_train.reshape(((X_train.shape[0], data.IMAGE_SQUARE_SIZE, data.IMAGE_SQUARE_SIZE, 1)))
    X_test = X_test.reshape(((X_test.shape[0], data.IMAGE_SQUARE_SIZE, data.IMAGE_SQUARE_SIZE, 1)))

    # datagen = keras.preprocessing.image.ImageDataGenerator(
    #     zoom_range=0.2,
    #     rotation_range=180,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True,
    #     vertical_flip=True
    # )

    for i in range(1): # Setting it to one for now
        print("Inr", i)

        print("Validation set:")
        preds = model.evaluate([X_test, X_test_2], Y_test, batch_size=batch_size, verbose=1, sample_weight=None)
        if display_model_stats_only:

            # final convolution layer
            print(model.layers[1].name)

            # global average pooling layer
            print(model.layers[-2].name)

            # output of the classifier
            print(model.layers[-1].name)

            print("Train set:")
            preds = model.evaluate([X_train, X_train_2], Y_train, batch_size=batch_size, verbose=1, sample_weight=None)
            return
        else:
            history = model.fit(x=(X_train, X_train_2), y=Y_train, batch_size=batch_size, epochs=epochs, class_weight=class_weights, callbacks=callbacks, validation_data=((X_test, X_test_2), Y_test))

            history.model.save(MODEL_SAVE_PATH + 'my_model.h5')

        if display_model_graphs:
            # list all data in history
            # print(history.history.keys())
            history = model.history
            # summarize history for accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history of precision
            plt.plot(history.history['precision'])
            plt.plot(history.history['val_precision'])
            plt.title('model precision')
            plt.ylabel('precision')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history of lr
            plt.plot(history.history['lr'])
            plt.title('Learning rate')
            plt.ylabel('learning rate')
            plt.xlabel('epoch')
            plt.legend(['lr'], loc='upper left')
            plt.show()

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
        face_input = faces[i]

        x = samp_image = np.array(face_input)
        x = x / 255.
        x = x.reshape(((1, data.IMAGE_SQUARE_SIZE, data.IMAGE_SQUARE_SIZE, 1)))
        print(np.array([x]).shape)

        for lan in face_landmarks[i][0]:
            landM.append(lan[0])
            landM.append(lan[1])
        dists = data.compDistance(landM)
        dists = np.array([dists])
        print(dists.shape)

        pred = model.predict((x, dists))
        print("NN output:\n", str(pred))

        # outputs = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4']
        outputs = [layer.output for layer in [model.layers[2], model.layers[3], model.layers[5], model.layers[6], model.layers[8]]]

        vis_model = Model(model.input, outputs)

        layer_names = []
        for layer in outputs:
            layer_names.append(layer.name.split("/")[0])

        print("Layers that will be used for visualization: ")
        print(layer_names)

        # with tf.GradientTape() as gtape:
        #     conv_output, predictions = model_grad((x, dists))
        #     loss = predictions[:, np.argmax(predictions[0])]
        #     grads = gtape.gradient(loss, conv_output)
        #     pooled_grads = K.mean(grads, axis=(0, 1, 2))
        #
        # heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        # heatmap = np.maximum(heatmap, 0)
        # max_heat = np.max(heatmap)
        # if max_heat == 0:
        #     max_heat = 1e-10
        # heatmap /= max_heat
        #
        # Image.fromarray(heatmap).show()
        #
        # return

        def get_CAM(processed_image, actual_label, layer_name='conv2d_4', heat_map_threshold=0.8):
            model_grad = Model([model.inputs],
                               [model.get_layer(layer_name).output, model.output])

            with tf.GradientTape() as tape:
                conv_output_values, predictions = model_grad((processed_image, dists))

                tape.watch(conv_output_values)

                pred_prob = predictions[:,0]
                print(predictions)

                actual_label = tf.cast([actual_label], dtype=tf.float32)

                # add a tiny value to avoid log of 0
                smoothing = 0.00001

                loss = keras.losses.binary_crossentropy(actual_label, predictions,label_smoothing=smoothing)
                print(f"binary loss: {loss}")
            grads_values = tape.gradient(loss, conv_output_values)
            grads_values = K.mean(grads_values, axis=(0, 1, 2))

            conv_output_values = np.squeeze(conv_output_values.numpy())
            grads_values = grads_values.numpy()

            for i in range(128):
                conv_output_values[:, :, i] *= grads_values[i]

            heatmap = np.mean(conv_output_values, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= heatmap.max()

            def sigmoid(x):
                return 1/(1 + np.exp(x))

            heatmap = sigmoid((-1 * ((heatmap - heat_map_threshold))) * 15)


            del model_grad, conv_output_values, grads_values, loss

            return heatmap

        def show_sample(image, idx=None, super_impose_threshold=0.35):


            sample_image = image
            sample_label = [1]

            sample_image_processed = np.expand_dims(sample_image, axis=0)

            activations = vis_model.predict((x, dists))

            pred_label = model.predict((x, dists))[0][0]

            sample_activation = activations[0][0, :, :, 16]

            sample_activation -= sample_activation.mean()
            sample_activation /= sample_activation.std()

            sample_activation *= 255
            sample_activation = np.clip(sample_activation, 0, 255).astype(np.uint8)

            heatmap = get_CAM(sample_image_processed, sample_label)
            heatmap = np.pad(heatmap, [(2, 0), (2, 0)], mode='constant', constant_values=0)
            print(heatmap.shape)
            heatmap = cv2.resize(heatmap, (sample_image.shape[0], sample_image.shape[1]))
            heatmap = heatmap * 255
            heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
            # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_GRAY2RGB)
            sample_image = np.array(sample_image, dtype='int64')
            print(heatmap.shape, sample_image.shape)
            super_imposed_image = cv2.addWeighted(sample_image, 0.6, heatmap.astype('int64'), super_impose_threshold, 0.0)

            f, ax = plt.subplots(2, 2, figsize=(15, 8))

            ax[0, 0].imshow(sample_image)
            ax[0, 0].set_title(f"True label: {sample_label} \n Predicted label: {pred_label}")
            ax[0, 0].axis('off')

            ax[0, 1].imshow(sample_activation)
            ax[0, 1].set_title("Random feature map")
            ax[0, 1].axis('off')

            ax[1, 0].imshow(heatmap)
            ax[1, 0].set_title("heat map")
            ax[1, 0].axis('off')

            ax[1, 1].imshow(super_imposed_image)
            ax[1, 1].set_title("heat map superimposed")
            ax[1, 1].axis('off')
            plt.tight_layout()
            plt.show()

            plt.tight_layout()
            plt.show()

            return activations

        def visualize_intermediate_activations(layer_names, activations):
            assert len(layer_names) == len(activations), "Make sure layers and activation values match"
            images_per_row = 16

            for layer_name, layer_activation in zip(layer_names, activations):
                nb_features = layer_activation.shape[-1]
                size = layer_activation.shape[1]

                nb_cols = nb_features // images_per_row
                grid = np.zeros((size * nb_cols, size * images_per_row))

                for col in range(nb_cols):
                    for row in range(images_per_row):
                        feature_map = layer_activation[0, :, :, col * images_per_row + row]
                        feature_map -= feature_map.mean()
                        feature_map /= feature_map.std()
                        feature_map *= 255
                        feature_map = np.clip(feature_map, 0, 255).astype(np.uint8)

                        grid[col * size:(col + 1) * size, row * size:(row + 1) * size] = feature_map

                scale = 1. / size
                plt.figure(figsize=(scale * grid.shape[1], scale * grid.shape[0]))
                plt.title(layer_name)
                plt.grid(False)
                plt.axis('off')
                plt.imshow(grid, aspect='auto', cmap='viridis')
            plt.show()

        if pred[0] > 0.5:
            string = 'Happy ' + str(round(((pred[0][0] * 2) * 100) - 100)) + '%'
        else:
            string = ""

        if imshow:
            # model.summary()

            activations = show_sample(face_input, idx=None)

            visualize_intermediate_activations(activations=activations,
                                               layer_names=layer_names)

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
    print("Loading data...")
    X, Y, Xtest, Ytest = load_data()
    X_2, Y_2, Xtest_2, Ytest_2 = train_2_prg.load_data()

    ch = int(input("1) Train the network\n"
                   "2) Display the stats of the model\n"
                   "Enter your choice: "))
    if ch == 1:
        train(X, Y, Xtest, Ytest, (X_2, Y_2, Xtest_2, Ytest_2), epochs=initial_epoch, batch_size=batch_size, display_model_stats_only=False)
    elif ch == 2:
        train(X, Y, Xtest, Ytest, (X_2, Y_2, Xtest_2, Ytest_2), epochs=initial_epoch, batch_size=batch_size, display_model_stats_only=True)
