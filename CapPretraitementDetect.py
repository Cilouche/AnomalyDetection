# 2. Cpature vidéo prétraitement + detection
import cv2
import numpy as  np
from email.mime.text import MIMEText
import smtplib
from AutoEncoder import *

def preprocess_frame(frame):
    # passage en gris + redimensionnement + normalisation [0-1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64,64))
    norm = resized.astype('float32') / 255.0
    norm = np.expand_dims(norm, axis = -1) # canal unique
    return norm

def send_alert(message):
    # exemple basique d'envoi mail ( = configurer avec ton SMTP)
    sender = "appcilouche@gmail.com"  # ton adresse mail
    receiver = "xxxxxxx@gmail.com"  # L'adresse qui recevra l'alerte 
    password =  "xxxxxxxxx"         # Le mot de passe d'application Gmail, pas ton vrai MDP

    msg = MIMEText(message)
    msg['Subject'] = "Alerte Anomalie détectée"
    msg["From"] = sender
    msg["To"] = receiver
    # Envoir l'émail via les serveurs sécurisés SMTP de Gmail (port 465)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password=password)
            server.sendmail(sender, receiver, msg.as_string())
            print("Alerte envoyée. ")

    except Exception as e : 
        print("Erreur envoi alerte mail : ", e)
   
