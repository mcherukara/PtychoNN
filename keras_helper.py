#Keras modules
from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, Flatten, Reshape, Lambda, Dropout
from keras.models import Model, Sequential, load_model
from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model, np_utils
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import keras
import numpy as np

#File I/O
import os, glob
import glob
import tempfile, os
from tqdm import tqdm_notebook as tqdm

#Image transforms
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn-white')
matplotlib.rc('font',family='Times New Roman')
matplotlib.rcParams['font.size'] = 20
plt.rcParams['image.cmap'] = 'viridis'


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def roc_plotter(fpr_keras, tpr_keras, auc_keras):
    fig, ax = plt.subplots(1,2, figsize=(20,6))
    ax[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax[0].plot(fpr_keras, tpr_keras, linewidth=3, label='Area = {:.3f}'.format(auc_keras))
    #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    ax[0].set_xlabel('False positive rate')
    ax[0].set_ylabel('True positive rate')
    ax[0].set_title('ROC curve')
    ax[0].legend(loc='best')
    #plt.show()
    # Zoom in view of the upper left corner.
    ax[1].set_xlim(0, 0.2)
    ax[1].set_ylim(0.8, 1)
    ax[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax[1].plot(fpr_keras, tpr_keras, linewidth=3, label='Area = {:.3f}'.format(auc_keras))
    #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    ax[1].set_xlabel('False positive rate')
    ax[1].set_ylabel('True positive rate')
    ax[1].set_title('ROC curve (zoomed in at top left)')
    ax[1].legend(loc='best')
    return ax

def repeat_channels(test_data):
	dims=test_data.shape
	dims=dims+(3,)
	print (dims)
	test_dataset3=np.zeros(dims,np.float16)
	for i in tqdm(range(dims[0])):
    		test_dataset3[i,:,:,0],test_dataset3[i,:,:,1],test_dataset3[i,:,:,2]=\
    		test_data[i,:,:],test_data[i,:,:],test_data[i,:,:]
	return test_dataset3

def plot_training_hist(history, init_epoch=0, n_last=5):
	
	loss=history['loss']
	val_loss=history['val_loss']
	acc=history['acc']
	val_acc=history['val_acc']

	epochs = np.linspace(init_epoch, len(loss)+init_epoch, len(loss))

	f, axarr = plt.subplots(2,2, sharex=False, figsize=(15, 6))

	axarr[0,0].set(ylabel='Loss')
	axarr[0,0].plot(epochs,loss, 'C3o', label='Training')
	axarr[0,0].plot(epochs,val_loss, 'C3-', label='Validation')
	axarr[0,0].grid()
	axarr[0,0].legend(loc='center right', bbox_to_anchor=(1.0, 0.5))

	axarr[1,0].plot(epochs,acc, 'C0o', label='Training')
	axarr[1,0].plot(epochs,val_acc, 'C0-', label='Validation')
	axarr[1,0].legend(loc='center right', bbox_to_anchor=(1.0, 0.5))
	axarr[1,0].set_xlabel('Epochs')
	axarr[1,0].set_ylabel('Accuracy')
	#axarr[1,0].tight_layout()
	axarr[1,0].grid()

	#axarr[0,1].set(ylabel='Loss')
	axarr[0,1].plot(epochs[-n_last:],loss[-n_last:], 'C3o', label='Training')
	axarr[0,1].plot(epochs[-n_last:],val_loss[-n_last:], 'C3-', label='Validation')
	axarr[0,1].grid()
	axarr[0,1].legend(loc='center right', bbox_to_anchor=(1.0, 0.5))

	axarr[1,1].plot(epochs[-n_last:],acc[-n_last:], 'C0o', label='Training')
	axarr[1,1].plot(epochs[-n_last:],val_acc[-n_last:], 'C0-', label='Validation')
	axarr[1,1].legend(loc='center right', bbox_to_anchor=(1.0, 0.5))
	axarr[1,1].set_xlabel('Epochs')
	#axarr[1,1].set_ylabel('Accuracy')
	#plt.tight_layout()
	axarr[1,1].grid()

class ModelMGPU(Model):
	def __init__(self, ser_model, gpus):
		if gpus>1:
			pmodel = multi_gpu_model(ser_model, gpus)
		else:
			pmodel = ser_model
		self.__dict__.update(pmodel.__dict__)
		self._smodel = ser_model

	def __getattribute__(self, attrname):
		'''Override load and save methods to be used from the serial-model. The
		serial-model holds references to the weights in the multi-gpu model.
		'''
        # return Model.__getattribute__(self, attrname)
		if 'load' in attrname or 'save' in attrname:
			return getattr(self._smodel, attrname)

		return super(ModelMGPU, self).__getattribute__(attrname)

def Conv_Pool_block(x0,nfilters,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last'):
	x0 = Conv2D(nfilters, (w1, w2), activation='relu', padding=padding, data_format=data_format)(x0)
	x0 = Conv2D(nfilters, (w1, w2), activation='relu', padding=padding, data_format=data_format)(x0)
	x0 = MaxPool2D((p1, p2), padding=padding, data_format=data_format)(x0)
	return x0

def Conv_Up_block(x0,nfilters,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last'):
	x0 = Conv2D(nfilters, (w1, w2), activation='relu', padding=padding, data_format=data_format)(x0)
	x0 = Conv2D(nfilters, (w1, w2), activation='relu', padding=padding, data_format=data_format)(x0)
	x0 = UpSampling2D((p1, p2), data_format=data_format)(x0)
	return x0
