import os, matplotlib

matplotlib.use("Pdf")
import matplotlib.pyplot

import numpy as np
from skimage import color, exposure, transform
from skimage import io, img_as_float32
import os
import glob
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from sklearn.metrics import accuracy_score
from keras import backend as K
import tensorflow as tf

K.set_image_data_format('channels_first')

NUM_CLASSES = 2
IMG_SIZE=40

root_dir = './'
imgs = []
labels = []

test_imgs = []
test_labels = []

f1 = open('full_plus_shufd', 'r')

all_img_paths = f1.readlines()
print("preprocessing images... ")
np.random.shuffle(all_img_paths)


def preprocess_img(img):
    # Histogram
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2, centre[1] - min_side // 2:centre[1] + min_side // 2,:]

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img



for line in all_img_paths:
    line = line.strip()
    img_path = line.split()[0]
    img = preprocess_img(io.imread(img_path))
    label = line.split()[1]
    imgs.append(img)
    labels.append(label)

X = np.array(imgs)
# Make one hot targets
Y = np.zeros((X.shape[0], NUM_CLASSES), dtype='uint8')
labels = np.array(labels, dtype='int')
labels = labels - 1
# print labels
for i, xi in enumerate(labels):
    Y[i, xi] = 1


def cnn_model():
    print("setting up the network... ")
    model = Sequential()

    model.add(Conv2D(5, (5, 5), padding='same', input_shape=(3, 40, 40), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    #	print "flattenting... "
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


model = cnn_model()

print(model.summary())
print("Training the model using SGD + momentum")
lr = 0.01
sgd = SGD(lr=lr, decay=0.00004, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# print "fitting the model... "

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 30))


batch_size = 32
epochs = 30

model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.0,
          callbacks=[LearningRateScheduler(lr_schedule), ModelCheckpoint('model_scratch.h5', save_best_only=True)])

# serialize whole model to H5
model.save("model_scratch.h5")
keras_file = "keras_model.h5"
tf.keras.models.save_model(model, keras_file)


# Convert to TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

f2 = open('full_test_plus', 'r')
all_test_paths = f2.readlines()

np.random.shuffle(all_test_paths)
for test_line in all_test_paths:
    test_line = test_line.strip()
    test_path = test_line.split()[0]
    test = preprocess_img(io.imread(test_path))
    test_label = test_line.split()[1]
    test_imgs.append(test)
    test_labels.append(test_label)

X_test = np.array(test_imgs)
test_labels = np.array(test_labels, dtype='int')
Y_test = test_labels - 1
print("predicting and evaluating... ")
# predict and evaluate
y_pred = model.predict_classes(X_test)
print(y_pred.shape)
print(Y_test.shape)
print(accuracy_score(Y_test, y_pred))

