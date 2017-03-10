import matplotlib.image as mpimg
import numpy as np
import os
import glob
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Dropout, Flatten, Activation
import json

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf


carfld = './../data/vehicles/'
nonfld = './../data/non-vehicles/'

cars = []
for folder in os.walk(carfld):
    folder_path = folder[0]
    for image in folder[2]:
        if '.DS_Store' in image:
            continue
        cars.append(folder_path + '/' + image)

notcars = []
for folder in os.walk(nonfld):
    folder_path = folder[0]
    for image in folder[2]:
        if '.DS_Store' in image:
            continue
        notcars.append(folder_path + '/' + image)


allcars = cars + notcars
allcars = np.asarray(allcars)
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))
print(len(cars), len(notcars), len(allcars))

test_images = glob.glob('./CarND-Vehicle-Detection/test_images/*.jpg')


y_binary = to_categorical(y)
# y_binary = y

rand_state = np.random.randint(0, 100)
sss = StratifiedShuffleSplit(n_splits=1,
                             test_size=0.2,
                             random_state=rand_state)


X_train = None
y_train = None
X_valid = None
y_valid = None
n_trn = None
n_val = None
for trn_idx, tst_idx in sss.split(allcars, y):
    X_train = allcars[trn_idx]
    y_train = y_binary[trn_idx]
    X_valid = allcars[tst_idx]
    y_valid = y_binary[tst_idx]
    n_trn = len(X_train)
    n_val = len(X_valid)
    print('trn:', len(trn_idx), round(float(sum(trn_idx)) / len(trn_idx), 3),
          'tst:', len(tst_idx), round(float(sum(tst_idx)) / len(tst_idx), 3))


def generator(samples, y, batch_size=32, normalize=True):
    ns = len(y)
    while True:
        samples, y = shuffle(samples, y)
        for ofs in range(0, ns, batch_size):
            batch_samples = samples[ofs:ofs + batch_size]
            by = y[ofs:ofs + batch_size]
            images = []
            for sample in batch_samples:
                image = mpimg.imread(sample)
                images.append(image)
            X_train = np.array(images, dtype=np.float32)
            if normalize:
                X_train -= 0.5
            y_train = np.array(by, dtype=np.float32)
            yield shuffle(X_train, y_train)


trn_gen = generator(X_train, y_train)
val_gen = generator(X_valid, y_valid)


def get_model25(dropout_rate=0.5):
    row, col, ch = 64, 64, 3

    model = Sequential()
    model.add(Convolution2D(6, 8, 8, subsample=(4, 4),
                            border_mode="same", input_shape=(row, col, ch)))
    model.add(Dropout(dropout_rate))
    model.add(Activation('relu'))
    model.add(Convolution2D(12, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

    return model


models = []
modelnames = []


models.append(get_model25())
modelnames.append('25a')


flog = open('log.csv', 'w')
flog.write('model,epoch,val_acc\n')
flog.close()
fhistory = open('history.txt', 'w')
fhistory.close()

for model, modelname in zip(models, modelnames):
    epochs = 25
    if 'b' in modelname:
        epochs = 100
    elif 'a' in modelname:
        epochs = 50
    for e in range(epochs):
        history = model.fit_generator(trn_gen,
                                      samples_per_epoch=n_trn,
                                      validation_data=val_gen,
                                      nb_val_samples=n_val,
                                      nb_epoch=e + 1,
                                      initial_epoch=e)
        val_acc = history.history['val_acc'][-1]

        outfold = "./models/%s" % (modelname)
        if not os.path.exists(outfold):
            os.makedirs(outfold)
        outfold = "./models/%s/%s" % (modelname, str(e))
        if not os.path.exists(outfold):
            os.makedirs(outfold)
        model.save_weights(outfold + "/model.h5", True)
        with open(outfold + '/model.json', 'w') as outfile:
            json.dump(model.to_json(), outfile)

        flog = open('log.csv', 'a')
        flog.write('%s,%d,%f\n' % (modelname, e, val_acc))
        flog.close()
        fhistory = open('history.txt', 'a')
        fhistory.write('%s\n' % str(history.history))
        fhistory.close()
