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
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

normalized_data = []

def getDatabase():
    data_dir = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'Database 1')
    onlyfiles = [os.path.join(data_dir, f) for f in os.listdir(
        data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    return onlyfiles


def buildData_deprecated(ch1, ch2, label):
    ch1 = normalize(ch1, axis=1, norm='l1')
    ch2 = normalize(ch2, axis=1, norm='l1') 

    consolidate_data = {"label": label,
                        "channel": []}
    if len(ch1) == len(ch2):
        for i in range(len(ch1)):
            normalized_data.append({"label": label,
                    "channel_1": ch1[i] / np.linalg.norm(ch1[i]),
                    "channel_2": ch2[i] / np.linalg.norm(ch2[i])})
            consolidate_data['channel'].append(np.vstack((ch1[i], ch2[i])).T)    
    else:
        raise Exception("Data not formatted")
    return consolidate_data, normalized_data


def consolidateData_deprecated():
    files = getDatabase()
    raw = {"label": [],
           "channel": []}
    normalized_array = []
    for file in files:
        actual_file = sio.loadmat(file)
        
        lines, normalized = buildData(actual_file["spher_ch1"], actual_file["spher_ch2"], 0) #spher
        normalized_array.append(normalized)
        for i in range(len(lines['channel'])):
            raw['label'].append(lines['label'])
            raw['channel'].append(lines['channel'][i])

        
        lines, normalized = buildData(actual_file["tip_ch1"], actual_file["tip_ch2"], 1) #tip
        normalized_array.append(normalized)
        for i in range(len(lines['channel'])):
            raw['label'].append(lines['label'])
            raw['channel'].append(lines['channel'][i])

                   
        lines, normalized = buildData(actual_file["palm_ch1"], actual_file["palm_ch2"], 2) #palm
        normalized_array.append(normalized)
        for i in range(len(lines['channel'])):
            raw['label'].append(lines['label'])
            raw['channel'].append(lines['channel'][i])

                     
        lines, normalized = buildData(actual_file["lat_ch1"], actual_file["lat_ch2"], 3) #lat
        normalized_array.append(normalized)
        for i in range(len(lines['channel'])):
            raw['label'].append(lines['label'])
            raw['channel'].append(lines['channel'][i])

            
        lines, normalized = buildData(actual_file["cyl_ch1"], actual_file["cyl_ch2"], 4) #cyl
        normalized_array.append(normalized)
        for i in range(len(lines['channel'])):
            raw['label'].append(lines['label'])
            raw['channel'].append(lines['channel'][i])

            
        lines, normalized = buildData(actual_file["hook_ch1"], actual_file["hook_ch2"], 5) #hook
        normalized_array.append(normalized)
        for i in range(len(lines['channel'])):
            raw['label'].append(lines['label'])
            raw['channel'].append(lines['channel'][i])

    return raw, normalized_array


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


def AnnArchitecture(raw):
    x_train, valid_label, y_train, valid_y, classes = getTrainData(raw)
    batch_size = 64
    epochs = 20
    num_classes = len(classes)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(3000,2,1),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
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


def getTrainData(raw):
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
    
    
# https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
def main():
    consolidateData()
    true_data = normalized_data.copy()
    model, test_eval = AnnArchitecture(true_data)
    print('Which?', 'all', 'Test loss:', test_eval[0], 'Test accuracy:', test_eval[1])
    
    for i in list(set([d['label'] for d in true_data])):
        model, test_eval = AnnArchitecture([{"label": 1 if d['label'] == i else 0, "channel_1": d['channel_1'],"channel_2": d['channel_2']} for d in true_data])
        print('Which?', i, 'Test loss:', test_eval[0], 'Test accuracy:', test_eval[1])
    
main()