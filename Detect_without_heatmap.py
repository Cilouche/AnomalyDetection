import cv2
import numpy as np
from CapPretraitementDetect import *
from AutoEncoder import *

def detect_without_heatmap(autoencoder, image, threshold=0.1):
        alert_send = False
        input_img = preprocess_frame(image)
        # ajouter un nouvel axe position 0: debut (1,64, 64, 3), 
        # position 1 : entre largeur et heuteur (64, 1, 64, 3)
        # position -1 : à la fin de la forme de tableau (64, 64, 3, 1)
        # pour n images faut utiliser np.stack
        input_batch = np.expand_dims(input_img, axis=0) # batch size 1
        reconstruction = autoencoder.predict(input_batch)

        # Erreur de reconstruction (MSE)
        error = np.mean(input_img - reconstruction[0]**2)
        print(f"ERROR WITHOUT HEATMAP IS : " , error)

         # Affichage
        disp_frame = cv2.resize(image,(320,240))
        cv2.putText(disp_frame, f"Erreur: {error:.3f}", (10, 70),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,0,255), 2)
        if error > threshold:
            print("Anomalie ......................")
            cv2.putText(disp_frame, " ANOMALIE DETECTE", (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1, (0,0,255), 3)
            if not alert_send:
             send_alert(f"Anomalie détectée avec erreur {error: .3f}")
             alert_send = True
        
        return disp_frame
