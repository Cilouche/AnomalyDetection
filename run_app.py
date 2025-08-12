import cv2
import numpy as  np
from email.mime.text import MIMEText
import smtplib
from AutoEncoder import *
from CapPretraitementDetect import *
from Detect_with_heatmap import *
from Detect_without_heatmap import *
import argparse
from Train import *
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#           TODO: 
#           1. Ajouter les argumlent type --run_heatmap, --train, --run , --threshold, --epoch  DONE
#           2. Revoir la fonction send alert                                                    DONE
#           3. Déploiement sur des docker                                                       DONE
#           4. Régler le problèmes des chemin absolue vers image, modèle                        DONE
#           5. Utiliser MQTT au lieu de SMTP car ce dernier est asynchrone
#               (pas de transmission en temps reel),lourd, 
#            6. Un probleme avec requirement.txt : comment ecrire ce fichier et l'installer

def run_app(threshold=0.2):
    AutoEncoder = build_autoencoder((64,64,1))
    # ici, on doit charger un modèle pré-entrainé sur images normales
    AutoEncoder.load_weights("autoencoder.weights.h5", skip_mismatch=True) 
    # load image from local folder
    image = cv2.imread("images/21.jpg")  
    # detect without heatmap 
    disp_frame =detect_without_heatmap(autoencoder=AutoEncoder,image=image, threshold=threshold)
    
   
    # Pour le docker
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir,"image.jpg"), disp_frame)
    # detect with heatmap
    #disp_frame = detect_anomaly_with_heatmap(autoencoder=AutoEncoder,frame=image, threshold=threshold)
    #cv2.imshow(" Détection anomalie avec Heatmap", disp_frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    os.chdir(output_dir)
    server = HTTPServer(("0.0.0.0", 8080), SimpleHTTPRequestHandler)
    print("Serving on http://localhost:8080/image.jpg")
    server.serve_forever()

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Pipeline vsion industrielle - Detection anomalie")
    parser.add_argument('--train', action='store_true', help="Entrainer le modèle sur les images normale")
    parser.add_argument('--run', action='store_true', help= "Lancer la détéction en temps réel")
    parser.add_argument('--data_folder', type=str, default='data', help="Chemin vers dossier image normale")
    parser.add_argument('--threshold', type=float, default=0.1, help="Seuil d'anomalie")
    parser.add_argument('--epochs', type=int, default=50, help="Nombre d'époques pour entrainnement")
    parser.add_argument('--batch_size', type=int, default=32, help="Nombre d'époques pour entrainnement")

    args = parser.parse_args()

    if args.train:
       train(data_folder=args.data_folder, epochs=args.epochs, batch_size=args.batch_size)
    if args.run:
       run_app(threshold=args.threshold)
    