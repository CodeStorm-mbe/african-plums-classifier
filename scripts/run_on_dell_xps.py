"""
Script d'adaptation pour exécuter le classificateur de prunes sur un Dell XPS 15 avec GPU Intel Arc 370.
Ce script configure l'environnement pour utiliser efficacement le GPU Intel Arc.
"""

import os
import sys
import argparse
import torch
import intel_extension_for_pytorch as ipex  # Nécessite d'être installé

# Importer nos modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_architecture import get_model
from utils.data_preprocessing import preprocess_single_image, get_val_transforms

def setup_intel_gpu():
    """
    Configure l'environnement pour utiliser le GPU Intel Arc.
    
    Returns:
        torch.device: Device à utiliser
    """
    # Vérifier si le GPU Intel est disponible
    if torch.xpu.is_available():
        print("GPU Intel Arc détecté et disponible.")
        device = torch.device("xpu")
        
        # Optimisations pour Intel GPU
        torch.xpu.set_device(0)
        print(f"Utilisation du GPU Intel: {torch.xpu.get_device_name()}")
    else:
        print("GPU Intel Arc non détecté, utilisation du CPU.")
        device = torch.device("cpu")
    
    return device

def optimize_model_for_intel(model):
    """
    Optimise le modèle pour les GPU Intel.
    
    Args:
        model: Modèle PyTorch
        
    Returns:
        model: Modèle optimisé
    """
    if torch.xpu.is_available():
        # Déplacer le modèle sur le GPU Intel
        model = model.to("xpu")
        
        # Optimiser avec IPEX
        model = ipex.optimize(model)
        
        # Utiliser FP16 pour de meilleures performances si supporté
        try:
            model = model.to(memory_format=torch.channels_last)
            model = ipex.optimize(model, dtype=torch.float16)
            print("Modèle optimisé avec FP16.")
        except:
            print("FP16 non supporté, utilisation de FP32.")
    
    return model

def load_model(model_path, num_classes=6, model_type='lightweight'):
    """
    Charge un modèle pré-entraîné et l'optimise pour Intel GPU.
    
    Args:
        model_path (str): Chemin vers le fichier de poids du modèle
        num_classes (int): Nombre de classes
        model_type (str): Type de modèle ('standard' ou 'lightweight')
        
    Returns:
        model: Modèle chargé et optimisé
    """
    # Créer le modèle (utiliser le modèle léger par défaut pour de meilleures performances)
    model = get_model(model_name=model_type, num_classes=num_classes)
    
    # Charger les poids
    if torch.xpu.is_available():
        model.load_state_dict(torch.load(model_path, map_location="xpu"))
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    # Mettre en mode évaluation
    model.eval()
    
    # Optimiser pour Intel
    model = optimize_model_for_intel(model)
    
    return model

def predict_image(model, image_path, class_names, device):
    """
    Prédit la classe d'une image.
    
    Args:
        model: Modèle
        image_path (str): Chemin vers l'image
        class_names (list): Liste des noms de classes
        device: Device à utiliser
        
    Returns:
        tuple: (classe prédite, probabilités)
    """
    # Prétraiter l'image
    image_tensor = preprocess_single_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted_idx = torch.max(outputs, 1)
    
    # Convertir en numpy pour faciliter la manipulation
    probabilities = probabilities.cpu().numpy()
    predicted_idx = predicted_idx.cpu().numpy()[0]
    
    return class_names[predicted_idx], probabilities

def main():
    """
    Fonction principale pour l'exécution sur Dell XPS avec Intel Arc.
    """
    # Parser les arguments
    parser = argparse.ArgumentParser(description='Exécution du classificateur de prunes sur Dell XPS avec Intel Arc')
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers le fichier de poids du modèle')
    parser.add_argument('--image_path', type=str, required=True, help='Chemin vers l\'image à classifier')
    parser.add_argument('--model_info', type=str, required=True, help='Chemin vers le fichier d\'informations du modèle')
    parser.add_argument('--model_type', type=str, default='lightweight', choices=['standard', 'lightweight'], 
                        help='Type de modèle à utiliser')
    args = parser.parse_args()
    
    # Configurer pour Intel GPU
    device = setup_intel_gpu()
    
    # Charger les informations du modèle
    import json
    with open(args.model_info, 'r') as f:
        model_info = json.load(f)
    
    class_names = model_info['class_names']
    num_classes = len(class_names)
    
    # Charger et optimiser le modèle
    model = load_model(args.model_path, num_classes, args.model_type)
    
    # Prédire la classe de l'image
    predicted_class, probabilities = predict_image(model, args.image_path, class_names, device)
    
    # Afficher les résultats
    print(f"Classe prédite: {predicted_class}")
    print("Probabilités par classe:")
    for i, (cls, prob) in enumerate(zip(class_names, probabilities)):
        print(f"  {cls}: {prob:.4f}")

if __name__ == '__main__':
    # Installer les dépendances nécessaires si elles ne sont pas déjà installées
    try:
        import intel_extension_for_pytorch
    except ImportError:
        print("Installation de intel_extension_for_pytorch...")
        os.system('pip install intel-extension-for-pytorch')
        import intel_extension_for_pytorch as ipex
    
    main()
