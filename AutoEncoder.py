# Passe l'image dans un autoencoder simple entrainé 
# sur des images normales (à simuler ici), calcule
# une erreur de contruction, et envoie une alérte si 
# l'erreur dépasse un seuil
# 1. Définir autoencoder simple
import cv2
#import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
#import smtplib
#from email.mime.text import MIMEText

# autoencodeur simple CNN (pour exemple)
def build_autoencoder(input_shape):
    input_img = layers.Input(shape=input_shape)

    # encoder
    x = layers.Conv2D(16, (3,3), activation= 'relu', padding= 'same')(input_img)
    x = layers.MaxPool2D((2,2), padding = 'same')(x)
    x = layers.Conv2D(9, (3,3), activation='relu', padding='same')(x)
    encoded = layers.MaxPool2D((2,2), padding='same')(x)

    # decoder
    x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy')
    return autoencoder

#build_autoencoder((512,1))