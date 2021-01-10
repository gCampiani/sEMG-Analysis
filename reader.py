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

import seaborn as sns

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
            normalized_data.append({"label": label,
                    "channel_1": ch1[i],
                    "channel_2": ch2[i]})
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
def AnnArchitecture1D(raw, epochs):
    x_train, valid_label, y_train, valid_y, classes = getTrainData1D(raw)
    batch_size = 256
    num_classes = len(classes)

    # model = Sequential()
    # model.add(Conv1D(filters=128, kernel_size=3, activation='linear', input_shape=(3000,2)))
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.4))
    # model.add(Conv1D(filters=64, kernel_size=3, activation='linear'))
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dropout(0.2))
    # model.add(Dense(128, activation='linear'))
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(Dropout(0.3))
    # model.add(Dense(num_classes if num_classes > 2 else 1, activation='softmax' if num_classes > 2 else 'sigmoid'))

    # model.compile(loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
    #             optimizer='adam', metrics=[keras.metrics.CategoricalAccuracy() if num_classes > 2 else keras.metrics.BinaryAccuracy()])

    # model_train = model.fit(y_train, x_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_y, valid_label))
    # test_eval = model.evaluate(valid_y, valid_label, verbose=0)
    
    # model.save("model_{}_epochs.h5py".format(epochs))
    model = keras.models.load_model("model_{}_epochs.h5py".format(epochs))
    predicted = np.argmax(model.predict(valid_y), axis=1)
    true_values = np.argmax(valid_label, axis=1)
    correct = np.where(predicted==true_values)[0]
    incorrect = np.where(predicted!=true_values)[0]
    print("correct: ", len(correct), "incorrect", len(incorrect))
    
    dict_model = {0:0,
                  1:0,
                  2:0,
                  3:0,
                  4:0,
                  5:0}
    
    import copy
    result = {0:copy.deepcopy(dict_model),
              1:copy.deepcopy(dict_model),
              2:copy.deepcopy(dict_model),
              3:copy.deepcopy(dict_model),
              4:copy.deepcopy(dict_model),
              5:copy.deepcopy(dict_model)}
        
    for i in range(len(predicted)):
        result[true_values[i]][predicted[i]] += 1 
        
    cf_matrix = []
    for label in result.keys():
            cf_matrix.append([result[label][internal_label] for internal_label in result[label].keys()])

    return model, cf_matrix


def getTrainData1D(raw):
    x = np.array([d['label'] for d in raw])
    y = np.array([d['channel_1'] for d in raw])
    z = np.array([d['channel_2'] for d in raw])
    y = np.column_stack((y,z))
    y = y.reshape(-1, 3000, 2)
    x = x.reshape(-1, 1)
    classes = np.unique(x)

    x_one_hot = to_categorical(x)

    x_train, valid_label, y_train, valid_y = train_test_split(x_one_hot if len(classes) > 2 else x, y, test_size=0.3, random_state=13)
    
    return x_train, valid_label, y_train, valid_y, classes


# https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
def AnnArchitecture2D(raw):
    x_train, valid_label, y_train, valid_y, classes = getTrainData2D(raw)
    batch_size = 128
    epochs = 10
    num_classes = len(classes)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 2),activation='linear',input_shape=(3000,2,1),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.5))
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
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.CategoricalAccuracy(),
                                                              keras.metrics.Recall(),
                                                              keras.metrics.Precision(),
                                                              keras.metrics.TruePositives(),
                                                              keras.metrics.TrueNegatives(),
                                                              keras.metrics.FalsePositives(),
                                                              keras.metrics.FalseNegatives()])
    #model.summary()
    
    model_train = model.fit(y_train, x_train, batch_size=batch_size,epochs=epochs,verbose=0,validation_data=(valid_y, valid_label))
    test_eval = model.evaluate(valid_y, valid_label, verbose=1)
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
    
    epochs = [20, 50, 100]
    
    labels = ["spher",
              "tip",
              "palm",
              "lat",
              "cyl",
              "hook",]
    
    for epoch in epochs:
        fig = plt.figure(epoch)
        model, cf_matrix = AnnArchitecture1D(true_data, epoch)
        sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.savefig("confusion_matrix_{}.png".format(epoch))
        plt.show()
    
main()