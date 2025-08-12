# Crée un dossier d'env venv
python3.11 -m venv myenv

# Active l'environnement virtuel
source myenv/bin/activate

# Mets pip à jour
pip install --upgrade pip

# Installe TensorFlow, OpenCV et NumPy
pip install tensorflow opencv-python numpy