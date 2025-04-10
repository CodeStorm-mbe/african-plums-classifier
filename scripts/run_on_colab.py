"""
Script d'adaptation pour exécuter le classificateur de prunes sur Google Colab.
Ce script configure l'environnement Colab, télécharge le dataset et exécute l'entraînement.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from google.colab import drive
import argparse
import json

def setup_colab_environment():
    """
    Configure l'environnement Google Colab.
    
    Returns:
        bool: True si l'environnement est correctement configuré
    """
    # Vérifier si nous sommes dans Colab
    try:
        import google.colab
        IN_COLAB = True
        print("Environnement Google Colab détecté.")
    except:
        IN_COLAB = False
        print("Environnement Google Colab non détecté.")
        return False
    
    # Monter Google Drive
    drive.mount('/content/drive')
    print("Google Drive monté avec succès.")
    
    # Créer les répertoires nécessaires
    os.makedirs('/content/plum_classifier', exist_ok=True)
    os.makedirs('/content/plum_classifier/models', exist_ok=True)
    os.makedirs('/content/plum_classifier/data', exist_ok=True)
    
    # Vérifier si GPU est disponible
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU non disponible, utilisation du CPU.")
    
    return True

def download_dataset(kaggle_username=None, kaggle_key=None):
    """
    Télécharge le dataset African Plums depuis Kaggle.
    
    Args:
        kaggle_username (str): Nom d'utilisateur Kaggle
        kaggle_key (str): Clé API Kaggle
        
    Returns:
        str: Chemin vers le répertoire du dataset
    """
    # Installer kaggle si nécessaire
    try:
        import kaggle
    except:
        print("Installation de la bibliothèque kaggle...")
        os.system('pip install kaggle')
    
    # Configurer les identifiants Kaggle
    os.makedirs('/root/.kaggle', exist_ok=True)
    
    if kaggle_username and kaggle_key:
        # Utiliser les identifiants fournis
        kaggle_config = {
            "username": kaggle_username,
            "key": kaggle_key
        }
        with open('/root/.kaggle/kaggle.json', 'w') as f:
            json.dump(kaggle_config, f)
    else:
        # Vérifier si le fichier kaggle.json existe dans Drive
        kaggle_json_path = '/content/drive/MyDrive/kaggle.json'
        if os.path.exists(kaggle_json_path):
            print("Utilisation du fichier kaggle.json trouvé dans Drive.")
            os.system(f'cp "{kaggle_json_path}" /root/.kaggle/')
        else:
            print("Aucun identifiant Kaggle trouvé. Veuillez fournir votre nom d'utilisateur et votre clé API.")
            return None
    
    # Définir les permissions
    os.system('chmod 600 /root/.kaggle/kaggle.json')
    
    # Télécharger le dataset
    print("Téléchargement du dataset African Plums...")
    os.system('kaggle datasets download -d arnaudfadja/african-plums-quality-and-defect-assessment-data')
    
    # Extraire le dataset
    print("Extraction du dataset...")
    os.system('unzip -q african-plums-quality-and-defect-assessment-data.zip -d /content/plum_classifier/data')
    
    return '/content/plum_classifier/data/african_plums_dataset'

def clone_repository(repo_url='https://github.com/CodeStorm-mbe/plum-sorter.git'):
    """
    Clone le dépôt GitHub contenant le code du classificateur.
    
    Args:
        repo_url (str): URL du dépôt GitHub
        
    Returns:
        bool: True si le clonage a réussi
    """
    try:
        print(f"Clonage du dépôt {repo_url}...")
        os.system(f'git clone {repo_url} /content/plum_classifier_repo')
        
        # Copier les fichiers nécessaires
        os.system('cp -r /content/plum_classifier_repo/utils /content/plum_classifier/')
        os.system('cp -r /content/plum_classifier_repo/models /content/plum_classifier/')
        os.system('cp /content/plum_classifier_repo/train.py /content/plum_classifier/')
        
        print("Dépôt cloné avec succès.")
        return True
    except:
        print("Erreur lors du clonage du dépôt.")
        return False

def upload_code_manually():
    """
    Instructions pour télécharger manuellement le code dans Colab.
    """
    from google.colab import files
    
    print("Veuillez télécharger les fichiers suivants:")
    print("1. utils/data_preprocessing.py")
    print("2. models/model_architecture.py")
    print("3. train.py")
    
    # Créer les répertoires
    os.makedirs('/content/plum_classifier/utils', exist_ok=True)
    os.makedirs('/content/plum_classifier/models', exist_ok=True)
    
    # Télécharger data_preprocessing.py
    print("\nTéléchargez data_preprocessing.py:")
    uploaded = files.upload()
    for filename in uploaded.keys():
        os.system(f'mv "{filename}" /content/plum_classifier/utils/data_preprocessing.py')
    
    # Télécharger model_architecture.py
    print("\nTéléchargez model_architecture.py:")
    uploaded = files.upload()
    for filename in uploaded.keys():
        os.system(f'mv "{filename}" /content/plum_classifier/models/model_architecture.py')
    
    # Télécharger train.py
    print("\nTéléchargez train.py:")
    uploaded = files.upload()
    for filename in uploaded.keys():
        os.system(f'mv "{filename}" /content/plum_classifier/train.py')
    
    return True

def train_model_on_colab(data_dir, model_type='standard', base_model='resnet18', num_epochs=20):
    """
    Entraîne le modèle sur Google Colab.
    
    Args:
        data_dir (str): Chemin vers le répertoire de données
        model_type (str): Type de modèle ('standard' ou 'lightweight')
        base_model (str): Modèle de base à utiliser
        num_epochs (int): Nombre d'époques
        
    Returns:
        str: Chemin vers le répertoire contenant les modèles entraînés
    """
    # Ajouter le répertoire courant au path
    sys.path.append('/content/plum_classifier')
    
    # Vérifier si les modules nécessaires sont disponibles
    try:
        from utils.data_preprocessing import load_and_prepare_data
        from models.model_architecture import get_model
    except ImportError:
        print("Erreur: Modules nécessaires non trouvés.")
        return None
    
    # Définir le répertoire de sauvegarde
    save_dir = '/content/plum_classifier/models'
    os.makedirs(save_dir, exist_ok=True)
    
    # Exécuter l'entraînement
    cmd = f'cd /content/plum_classifier && python train.py --data_dir "{data_dir}" --save_dir "{save_dir}" --model_type {model_type} --base_model {base_model} --num_epochs {num_epochs} --batch_size 32'
    print(f"Exécution de la commande: {cmd}")
    os.system(cmd)
    
    # Copier les modèles entraînés vers Google Drive
    drive_save_dir = '/content/drive/MyDrive/plum_classifier_models'
    os.makedirs(drive_save_dir, exist_ok=True)
    os.system(f'cp -r {save_dir}/* {drive_save_dir}/')
    print(f"Modèles entraînés copiés vers Google Drive: {drive_save_dir}")
    
    return save_dir

def main():
    """
    Fonction principale pour l'exécution sur Google Colab.
    """
    # Parser les arguments
    parser = argparse.ArgumentParser(description='Exécution du classificateur de prunes sur Google Colab')
    parser.add_argument('--kaggle_username', type=str, help='Nom d\'utilisateur Kaggle')
    parser.add_argument('--kaggle_key', type=str, help='Clé API Kaggle')
    parser.add_argument('--repo_url', type=str, help='URL du dépôt GitHub contenant le code')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'lightweight'], 
                        help='Type de modèle à utiliser')
    parser.add_argument('--base_model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50', 'mobilenet_v2', 'efficientnet_b0'], 
                        help='Modèle de base à utiliser')
    parser.add_argument('--num_epochs', type=int, default=20, help='Nombre d\'époques')
    parser.add_argument('--manual_upload', action='store_true', help='Télécharger manuellement le code')
    args = parser.parse_args()
    
    # Configurer l'environnement Colab
    if not setup_colab_environment():
        print("Erreur: Impossible de configurer l'environnement Colab.")
        return
    
    # Obtenir le code
    if args.manual_upload:
        upload_code_manually()
    elif args.repo_url:
        clone_repository(args.repo_url)
    else:
        print("Aucune méthode spécifiée pour obtenir le code. Utilisation du téléchargement manuel.")
        upload_code_manually()
    
    # Télécharger le dataset
    data_dir = download_dataset(args.kaggle_username, args.kaggle_key)
    if not data_dir:
        print("Erreur: Impossible de télécharger le dataset.")
        return
    
    # Entraîner le modèle
    save_dir = train_model_on_colab(data_dir, args.model_type, args.base_model, args.num_epochs)
    if not save_dir:
        print("Erreur: Échec de l'entraînement du modèle.")
        return
    
    print(f"Entraînement terminé avec succès. Modèles sauvegardés dans {save_dir} et Google Drive.")

# Code pour générer un notebook Colab
def generate_colab_notebook():
    """
    Génère un notebook Colab pour faciliter l'utilisation.
    """
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Classificateur de Prunes Africaines - Google Colab\n",
                          "\n",
                          "Ce notebook permet d'entraîner un modèle de classification des prunes africaines sur Google Colab.\n",
                          "\n",
                          "## Étapes:\n",
                          "1. Configuration de l'environnement\n",
                          "2. Téléchargement du dataset\n",
                          "3. Préparation du code\n",
                          "4. Entraînement du modèle\n",
                          "5. Évaluation et sauvegarde"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Installation des dépendances\n",
                          "!pip install torch torchvision tqdm matplotlib seaborn scikit-learn"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Configuration de Kaggle\n",
                          "from google.colab import files\n",
                          "import os\n",
                          "\n",
                          "# Télécharger le fichier kaggle.json\n",
                          "print(\"Veuillez télécharger votre fichier kaggle.json\")\n",
                          "uploaded = files.upload()\n",
                          "\n",
                          "# Créer le répertoire .kaggle si nécessaire\n",
                          "!mkdir -p ~/.kaggle\n",
                          "\n",
                          "# Copier le fichier kaggle.json\n",
                          "for fn in uploaded.keys():\n",
                          "    !cp {fn} ~/.kaggle/kaggle.json\n",
                          "\n",
                          "# Définir les permissions\n",
                          "!chmod 600 ~/.kaggle/kaggle.json\n",
                          "\n",
                          "print(\"Configuration de Kaggle terminée.\")"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Téléchargement du dataset\n",
                          "!kaggle datasets download -d arnaudfadja/african-plums-quality-and-defect-assessment-data\n",
                          "!mkdir -p /content/plum_classifier/data\n",
                          "!unzip -q african-plums-quality-and-defect-assessment-data.zip -d /content/plum_classifier/data\n",
                          "\n",
                          "print(\"Dataset téléchargé et extrait.\")"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Téléchargement du code\n",
                          "!mkdir -p /content/plum_classifier/utils\n",
                          "!mkdir -p /content/plum_classifier/models\n",
                          "\n",
                          "# Vous pouvez soit télécharger manuellement les fichiers, soit les récupérer depuis GitHub\n",
                          "# Option 1: Depuis GitHub\n",
                          "# !git clone https://github.com/username/plum-classifier.git /content/plum_classifier_repo\n",
                          "# !cp -r /content/plum_classifier_repo/utils /content/plum_classifier/\n",
                          "# !cp -r /content/plum_classifier_repo/models /content/plum_classifier/\n",
                          "# !cp /content/plum_classifier_repo/train.py /content/plum_classifier/\n",
                          "\n",
                          "# Option 2: Téléchargement manuel (décommentez et exécutez chaque section séparément)\n",
                          "print(\"Veuillez télécharger les fichiers suivants:\")\n",
                          "print(\"1. data_preprocessing.py\")\n",
                          "print(\"2. model_architecture.py\")\n",
                          "print(\"3. train.py\")"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Téléchargement de data_preprocessing.py\n",
                          "from google.colab import files\n",
                          "uploaded = files.upload()\n",
                          "for fn in uploaded.keys():\n",
                          "    !mv \"{fn}\" /content/plum_classifier/utils/data_preprocessing.py"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Téléchargement de model_architecture.py\n",
                          "from google.colab import files\n",
                          "uploaded = files.upload()\n",
                          "for fn in uploaded.keys():\n",
                          "    !mv \"{fn}\" /content/plum_classifier/models/model_architecture.py"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Téléchargement de train.py\n",
                          "from google.colab import files\n",
                          "uploaded = files.upload()\n",
                          "for fn in uploaded.keys():\n",
                          "    !mv \"{fn}\" /content/plum_classifier/train.py"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Entraînement du modèle\n",
                          "import sys\n",
                          "sys.path.append('/content/plum_classifier')\n",
                          "\n",
                          "# Définir les paramètres\n",
                          "data_dir = '/content/plum_classifier/data/african_plums_dataset'\n",
                          "save_dir = '/content/plum_classifier/models'\n",
                          "model_type = 'standard'  # ou 'lightweight'\n",
                          "base_model = 'resnet18'  # ou 'resnet50', 'mobilenet_v2', 'efficientnet_b0'\n",
                          "num_epochs = 20\n",
                          "batch_size = 32\n",
                          "\n",
                          "# Exécuter l'entraînement\n",
                          "!cd /content/plum_classifier && python train.py --data_dir \"{data_dir}\" --save_dir \"{save_dir}\" --model_type {model_type} --base_model {base_model} --num_epochs {num_epochs} --batch_size {batch_size}"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Sauvegarder les modèles dans Google Drive\n",
                          "from google.colab import drive\n",
                          "drive.mount('/content/drive')\n",
                          "\n",
                          "drive_save_dir = '/content/drive/MyDrive/plum_classifier_models'\n",
                          "!mkdir -p {drive_save_dir}\n",
                          "!cp -r /content/plum_classifier/models/* {drive_save_dir}/\n",
                          "\n",
                          "print(f\"Modèles entraînés copiés vers Google Drive: {drive_save_dir}\")"],
                "execution_count": None,
                "outputs": []
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    import json
    with open('plum_classifier_colab.ipynb', 'w') as f:
        json.dump(notebook, f)
    
    print("Notebook Colab généré: plum_classifier_colab.ipynb")
    return 'plum_classifier_colab.ipynb'

if __name__ == '__main__':
    # Si exécuté directement, générer le notebook Colab
    if not 'google.colab' in sys.modules:
        notebook_path = generate_colab_notebook()
        print(f"Notebook Colab généré: {notebook_path}")
        print("Veuillez télécharger ce notebook et l'ouvrir dans Google Colab.")
    else:
        main()
