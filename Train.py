# train 
import os
import cv2
import numpy as np
from AutoEncoder import *

def load_images_from_folder(folder, size=(64,64)):
    images=[]
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, size)
                img = img.astype('float32') / 255.0
                images.append(img)
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)
    return images

# 
def train(imagesFolder, epochs=50,batch_size=32): 
    # charger dataset d'images normales
    train_images = load_images_from_folder(imagesFolder) 
    print(f"Images chargée pour entrainement : {train_images.shape}")

    # Train model
    input_shape = (64, 64, 1)
    autoencoder = build_autoencoder(input_shape)
    autoencoder.fit(train_images, train_images,
                    epochs,
                    batch_size,
                    shuffle= True,
                    validation_split=0.1)
    # Sauvgarde du modèle
    
    autoencoder.save_weights('autoencoder.weights.h5')
