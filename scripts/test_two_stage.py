"""
Script de test pour le modèle en deux étapes (détection puis classification).
Ce script permet de tester le modèle avec des images individuelles.
"""

import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Importer nos modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_preprocessing import preprocess_single_image, get_val_transforms
from models.model_architecture import get_two_stage_model

def load_two_stage_model(detection_model_path, classification_model_path, model_info_path, device):
    """
    Charge le modèle en deux étapes à partir des fichiers sauvegardés.
    
    Args:
        detection_model_path (str): Chemin vers le fichier de poids du modèle de détection
        classification_model_path (str): Chemin vers le fichier de poids du modèle de classification
        model_info_path (str): Chemin vers le fichier d'informations du modèle
        device: Device sur lequel charger les modèles
        
    Returns:
        tuple: (modèle en deux étapes, informations du modèle)
    """
    # Charger les informations du modèle
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    # Extraire les informations nécessaires
    detection_class_names = model_info['detection_class_names']
    classification_class_names = model_info['classification_class_names']
    detection_threshold = model_info['detection_threshold']
    
    # Créer le modèle en deux étapes
    two_stage_model = get_two_stage_model(
        detection_model_name='lightweight',
        classification_model_name='standard',
        detection_base_model='mobilenet_v2',
        classification_base_model='resnet18',
        num_detection_classes=len(detection_class_names),
        num_classification_classes=len(classification_class_names),
        pretrained=False,  # Nous utilisons nos propres poids entraînés
        detection_threshold=detection_threshold
    )
    
    # Charger les poids entraînés
    two_stage_model.detection_model.load_state_dict(torch.load(detection_model_path, map_location=device))
    two_stage_model.classification_model.load_state_dict(torch.load(classification_model_path, map_location=device))
    
    # Mettre les modèles en mode évaluation
    two_stage_model.detection_model.eval()
    two_stage_model.classification_model.eval()
    
    return two_stage_model, model_info

def predict_image(model, image_path, model_info, device, transform=None):
    """
    Prédit si l'image contient une prune et, si oui, son état.
    
    Args:
        model: Modèle en deux étapes
        image_path (str): Chemin vers l'image
        model_info (dict): Informations du modèle
        device: Device sur lequel exécuter les modèles
        transform: Transformations à appliquer (si None, utilise get_val_transforms)
        
    Returns:
        tuple: (is_plum, predicted_class, detection_probs, classification_probs)
    """
    # Prétraiter l'image
    if transform is None:
        img_size = model_info.get('img_size', 224)
        transform = get_val_transforms(img_size)
    
    image_tensor = preprocess_single_image(image_path, transform)
    
    # Prédiction
    is_plum, predicted_idx, probs = model.predict(image_tensor, device)
    
    if is_plum:
        # C'est une prune, retourner la classe prédite
        predicted_class = model_info['classification_class_names'][predicted_idx]
        return True, predicted_class, None, probs
    else:
        # Ce n'est pas une prune
        return False, "non_plum", probs, None

def visualize_prediction(image_path, is_plum, predicted_class, detection_probs=None, classification_probs=None, model_info=None):
    """
    Visualise l'image avec sa prédiction.
    
    Args:
        image_path (str): Chemin vers l'image
        is_plum (bool): Si l'image contient une prune
        predicted_class (str): Classe prédite
        detection_probs (numpy.ndarray): Probabilités de détection
        classification_probs (numpy.ndarray): Probabilités de classification
        model_info (dict): Informations du modèle
    """
    # Charger l'image
    image = Image.open(image_path).convert('RGB')
    
    # Créer la figure
    plt.figure(figsize=(12, 6))
    
    # Afficher l'image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    if is_plum:
        plt.title(f"Prédiction : Prune - {predicted_class}")
    else:
        plt.title("Prédiction : Pas une prune")
    plt.axis('off')
    
    # Afficher les probabilités
    plt.subplot(1, 2, 2)
    
    if is_plum and classification_probs is not None and model_info is not None:
        # Afficher les probabilités de classification
        class_names = model_info['classification_class_names']
        y_pos = np.arange(len(class_names))
        plt.barh(y_pos, classification_probs, align='center')
        plt.yticks(y_pos, class_names)
        plt.xlabel('Probabilité')
        plt.title('Probabilités par classe')
    elif not is_plum and detection_probs is not None and model_info is not None:
        # Afficher les probabilités de détection
        class_names = model_info['detection_class_names']
        y_pos = np.arange(len(class_names))
        plt.barh(y_pos, detection_probs, align='center')
        plt.yticks(y_pos, class_names)
        plt.xlabel('Probabilité')
        plt.title('Probabilités de détection')
    
    plt.tight_layout()
    plt.show()
    
    # Afficher les probabilités en pourcentage
    print("\nProbabilités détaillées :")
    if is_plum and classification_probs is not None and model_info is not None:
        for i, (cls, prob) in enumerate(zip(model_info['classification_class_names'], classification_probs)):
            print(f"{cls}: {prob*100:.2f}%")
    elif not is_plum and detection_probs is not None and model_info is not None:
        for i, (cls, prob) in enumerate(zip(model_info['detection_class_names'], detection_probs)):
            print(f"{cls}: {prob*100:.2f}%")

def main():
    """
    Fonction principale pour tester le modèle en deux étapes.
    """
    # Parser les arguments
    parser = argparse.ArgumentParser(description='Test du modèle en deux étapes pour les prunes')
    parser.add_argument('--detection_model_path', type=str, required=True, help='Chemin vers le fichier de poids du modèle de détection')
    parser.add_argument('--classification_model_path', type=str, required=True, help='Chemin vers le fichier de poids du modèle de classification')
    parser.add_argument('--model_info_path', type=str, required=True, help='Chemin vers le fichier d\'informations du modèle')
    parser.add_argument('--image_path', type=str, required=True, help='Chemin vers l\'image à tester')
    args = parser.parse_args()
    
    # Déterminer le device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de: {device}")
    
    # Charger le modèle en deux étapes
    print("Chargement du modèle en deux étapes...")
    model, model_info = load_two_stage_model(
        args.detection_model_path,
        args.classification_model_path,
        args.model_info_path,
        device
    )
    
    # Tester l'image
    print(f"Test de l'image: {args.image_path}")
    is_plum, predicted_class, detection_probs, classification_probs = predict_image(
        model,
        args.image_path,
        model_info,
        device
    )
    
    # Afficher les résultats
    if is_plum:
        print(f"L'image contient une prune de type: {predicted_class}")
    else:
        print("L'image ne contient pas de prune.")
    
    # Visualiser la prédiction
    visualize_prediction(
        args.image_path,
        is_plum,
        predicted_class,
        detection_probs,
        classification_probs,
        model_info
    )

if __name__ == '__main__':
    main()
