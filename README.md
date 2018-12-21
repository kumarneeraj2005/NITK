

Minimum Requirement

**** Code and Data file we will be sharing at the time of workshop, preferred way to share code and data will be GitHub.


Windows or Mac laptops (RAM – minimum 4 GB)

These are the steps, packages and apps to be installed:
1.	Preferably install anaconda environment with python 3.6 (https://repo.anaconda.com/archive/ )
Search for version : Anaconda 4.3.0 (https://repo.anaconda.com/archive/Anaconda2-4.3.0-Windows-x86.exe ) if you will click on above link it will automatically download Anaconda version for your windows system.

Please explore anaconda python before workshop and get basic understanding for conda environment.

2.	Install TensorFlow from conda repository ( pip install tensorflow) (https://www.tensorflow.org )

Explore Tensorflow Libs and try to understand basic uses.
3.	Install Keras from conda repository ( pip install keras) (https://keras.io 
4.	Install NLTK (pip install nltk)    
Afterwards open Pycharm python IDE and do following

Import nltk
Then nltk.download()

 

Then run this .py file to download nltk related files and dictionaries 
 
you will see screen like below, it will take an hour depends on your internet speed.

 



5.	Install OpenCV for your system (pip install pip install opencv-python)
Python binding for Opencv (for image processing required)
6.	Install utility packages ( pip install imutils)


For info only: If any thing missing while workshop please use PIP command to install it.
Most of the libs will be by default come with Anaconda Python.
We will be using following imports in our workshop

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from keras import backend as K
K.set_image_dim_ordering('th') 
import h5py
import cv2
import os

import nltk
#nltk.download()
import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
#from keras.layers import Dense, Dropout, Conv1D
#from keras.layers import Flatten
#from keras.layers import Embedding
from keras.layers import Dense, Embedding, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.utils import to_categorical


#%% Load cleaner function 
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords 
import string
