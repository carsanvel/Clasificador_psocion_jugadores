# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:22:50 2021

@author: carvsk
"""

#from matplotlib.pyplot import imshow
import numpy as np
#from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
#from keras import backend as K
#import keras
#from time import time
#from matplotlib import pyplot as plt 
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt 

model = keras.models.load_model("mimodelo_alternativo2.h5")

validation_data_dir = './Datos_FSI/validation'
batch_size = 5

validation_datagen = ImageDataGenerator(
        rescale=1./255
)

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

print(model.summary())

Y_pred = model.predict_generator(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['Por', 'Ld', 'Dfc', 'Li', 'Mcd', 'Mc', 'Mco', 'Ed', 'Ei', 'Dc']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

