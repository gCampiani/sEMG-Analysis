import os
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

normalized_data = []

def getDatabase():
    data_dir = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'Database 1')
    onlyfiles = [os.path.join(data_dir, f) for f in os.listdir(
        data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    return onlyfiles


def buildData(ch1, ch2, label):
    global normalized_data
    if len(ch1) == len(ch2):
        for i in range(len(ch1)):
            normalized_data.append({"label": label,
                    "channel_1": ch1[i] / np.linalg.norm(ch1[i]),
                    "channel_2": ch2[i] / np.linalg.norm(ch2[i])})
    else:
        raise Exception("Data not formatted")
    return 0


def consolidateData():
    files = getDatabase()
    
    for file in files:
        actual_file = sio.loadmat(file)
        buildData(actual_file["spher_ch1"], actual_file["spher_ch2"], 0) #spher
        buildData(actual_file["tip_ch1"], actual_file["tip_ch2"], 1) #tip        
        buildData(actual_file["palm_ch1"], actual_file["palm_ch2"], 2) #palm
        buildData(actual_file["lat_ch1"], actual_file["lat_ch2"], 3) #lat
        buildData(actual_file["cyl_ch1"], actual_file["cyl_ch2"], 4) #cyl
        buildData(actual_file["hook_ch1"], actual_file["hook_ch2"], 5) #hook
    return 0


# https://missinglink.ai/guides/keras/keras-conv1d-working-1d-convolutional-neural-networks-keras/
def AnnArchitecture1D(raw):
    x_train, valid_label, y_train, valid_y, classes = getTrainData1D(raw)
    batch_size = 32
    epochs = 10
    num_classes = len(classes)

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(3000,2)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_train = model.fit(y_train, x_train, batch_size=batch_size,epochs=epochs,verbose=0,validation_data=(valid_y, valid_label))
    test_eval = model.evaluate(valid_y, valid_label, verbose=0)
    return model, test_eval


def getTrainData1D(raw):
    x = np.array([d['label'] for d in raw])
    y = np.array([d['channel_1'] for d in raw])
    z = np.array([d['channel_2'] for d in raw])
    y = np.column_stack((y,z))
    y = y.reshape(-1, 3000, 2)
    x = x.reshape(-1, 1)
    x_one_hot = to_categorical(x)    

    x_train, valid_label, y_train, valid_y = train_test_split(x_one_hot, y, test_size=0.3, random_state=13)
    classes = np.unique(x)
    
    return x_train, valid_label, y_train, valid_y, classes


# https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
def AnnArchitecture2D(raw):
    x_train, valid_label, y_train, valid_y, classes = getTrainData2D(raw)
    batch_size = 64
    epochs = 15
    num_classes = len(classes)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2),activation='linear',input_shape=(3000,2,1),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (2, 2), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (2, 2), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    #model.summary()
    
    model_train = model.fit(y_train, x_train, batch_size=batch_size,epochs=epochs,verbose=0,validation_data=(valid_y, valid_label))
    test_eval = model.evaluate(valid_y, valid_label, verbose=0)
    return model, test_eval


def getTrainData2D(raw):
    x = np.array([d['label'] for d in raw])
    y = np.array([d['channel_1'] for d in raw])
    z = np.array([d['channel_2'] for d in raw])
    y = np.column_stack((y,z))
    y = y.reshape(-1, 3000, 2, 1)
    x = x.reshape(-1, 1)
    x_one_hot = to_categorical(x)    

    x_train, valid_label, y_train, valid_y = train_test_split(x_one_hot, y, test_size=0.3, random_state=13)
    classes = np.unique(x)
    
    return x_train, valid_label, y_train, valid_y, classes
    
    
def main():
    consolidateData()
    true_data = normalized_data.copy()
    model, test_eval = AnnArchitecture1D(true_data)
    print('Which?', 'all', 'Test loss:', test_eval[0], 'Test accuracy:', test_eval[1])
    
    for i in list(set([d['label'] for d in true_data])):
        model, test_eval = AnnArchitecture1D([{"label": 1 if d['label'] == i else 0, "channel_1": d['channel_1'],"channel_2": d['channel_2']} for d in true_data])
        print('Which?', i, 'Test loss:', test_eval[0], 'Test accuracy:', test_eval[1])
    
main()