# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

plt.switch_backend('agg')

def getdata():
    # read data and shuffle
    index=[i for i in range(64)]
    random.shuffle(index)

    f=open("./id_to_data","rb+")
    data=pickle.load(f)
    data_train=data
    f=open("./id_to_box","rb+")
    box=pickle.load(f)
    box_train=box

    f=open("./id_to_data_test","rb+")
    data=pickle.load(f)
    data_test=data
    f=open("./id_to_box_test","rb+")
    box=pickle.load(f)
    box_test=box
    
    return data_train,box_train,data_test,box_test


def plot_model(model_details):
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(range(1,len(model_details.history['my_metric'])+1),model_details.history['my_metric'])
    axs[0].plot(range(1,len(model_details.history['val_my_metric'])+1),[1.7*x for x in model_details.history['val_my_metric']])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_details.history['my_metric'])+1),len(model_details.history['my_metric'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
    axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1),len(model_details.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')

    plt.savefig("model.png")
