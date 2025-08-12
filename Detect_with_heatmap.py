""" Detection acec heatmap
    Heatmap : Pour la visualisation de l'anomalies, le principes  : 
        1. Calculer la difference pixel à pixel entre l'image d'entrée et sa reconstruction par l'autoencoder
        2. Cette difference (erreur) est normalisée, colorée en "fet" (couleur chaudes = anomalie forte)
        3. On superpose cette heatmap à l'image d'origine pour mieux localiser les défauts
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np
from CapPretraitementDetect import *

def detect_anomaly_with_heatmap(autoencoder, frame, threshold=0.1):
    input_img = preprocess_frame(frame)
    input_batch = np.expand_dims(input_img, axis=0)
    reconstruction = autoencoder.predict(input_batch)

    # calcul erreur pixel à pixel
    error_map = np.abs(input_img - reconstruction[0]).squeeze() # squeeze suprime que les dimension de taille 1

    # Erreur moyenne globale
    error = np.mean(error_map)
    print(f"ERROR WITH HEATMAP IS : " , error)
    # Normalisation de la heatmap [0,255]
    heatmap = (error_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # redimension pour correspondre à l'image originale (320X240 ici)
    heatmap_color = cv2.resize(heatmap_color, (frame.shape[1], frame.shape[0]))

    # superposition (alpha blending)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
   
    # Test
    disp_frame = cv2.resize(overlay, (640,480))
    cv2.putText(disp_frame, f"Erreur: {error : .3f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    if error > threshold:
        cv2.putText(disp_frame, "ANOMALIE DETECTEE", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 3)
        if not alert_send:
            send_alert(f"Anomalie détectée avec erreur {error: .3f}")
            alert_send = True
        
    return disp_frame


