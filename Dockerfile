# Image de base avec Python3.11 slim (léger)
FROM python:3.11-slim

# Varaibles d'environnement pour Tensorflow (optionnel)
ENV TF_CPP_MIN_LOG_LEVEL=2

# Installer les dépendances système nécessaires pour OPENCV et matplotlib
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt et installer les dépendances python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source dans le conteneur
COPY . /app
WORKDIR /app

# (Optionnel) Exposer un port si ton app l'utilise, comme 8501 pour Streamlit par exemple
EXPOSE 8080

# Commande par défaut pour éxécuter ton script
CMD ["python", "run_app.py", "--run"]
