import cv2
import numpy as np
from keras.models import load_model
from keras.utils.image_utils import img_to_array
number_label = ['0','1','1','1','0','1','Confirm','**','%','Clear','2','3','4','5','6','7','8','9']
alphabet_label = ['-','a','c','a','-','h','-','k','-','m','r','e','o','t',' ','s','t','u']

## Loading model using the loacation where the model is saved
model = load_model('saved_model.h5')

def get_prediction(img):
    for_pred = cv2.resize(img,(64,64))
    x = img_to_array(for_pred)
    x = x/255.0
    x = x.reshape((1,) + x.shape)
    predictions = model.predict(x)[0]
    gesture = np.argmax(predictions)
    probability = predictions[gesture]
    gesture = number_label[gesture]
    return gesture, probability

def get_alphabet(img):
    for_pred = cv2.resize(img,(64,64))
    x = img_to_array(for_pred)
    x = x/255.0
    x = x.reshape((1,) + x.shape)
    predictions = model.predict(x)[0]
    gesture = np.argmax(predictions)
    probability = predictions[gesture]
    gesture = alphabet_label[gesture]
    return gesture, probability
    
