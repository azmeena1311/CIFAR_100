import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import sys
from imgaug import augmenters as iaa
import random
from datetime import datetime
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


meta = unpickle(r'E:\University of Windsor\Machine Learning and Pattern Recognition\Project\cifar-100-python\meta')
train = unpickle(r'E:\University of Windsor\Machine Learning and Pattern Recognition\Project\cifar-100-python\train')
test = unpickle(r'E:\University of Windsor\Machine Learning and Pattern Recognition\Project\cifar-100-python\test')


Classes = pd.DataFrame(meta[b'fine_label_names'],columns = ['Classes'])


X = train[b"data"]


X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")


img_num = np.random.randint(0,1000)
plt.figure(figsize=(10,5))
plt.xticks([])
plt.yticks([])
plt.imshow(X[img_num])
Classes.iloc[train[b'fine_labels'][img_num]]
plt.show()


plt.figure(figsize=(10,5))
num_images_row = 4
num_images_column = 4
img_nums = np.random.randint(0,len(X),num_images_row*num_images_column)

f, axarr = plt.subplots(num_images_row,num_images_column)

for i in range(0,num_images_row):
    for j in range(0,num_images_column):
        axarr[i,j].imshow(X[img_nums[(i*num_images_column)+j]])
        #axarr[i,j].set_title(str(Classes.iloc[train[b'fine_labels'][img_nums[(i+1)*(j+1)-1]]]).split()[1])
        axarr[i,j].axis('off')

plt.show()
