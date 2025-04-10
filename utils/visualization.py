"""
Module d'utilitaires pour le classificateur de prunes.
Ce module contient des fonctions utilitaires diverses pour le projet.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

def visualize_results(image_path, is_plum, predicted_class, detection_probs=None, classification_probs=None, 
                     detection_class_names=None, classification_class_names=None, save_path=None):
    """
    Visualise les résultats de prédiction.
    
    Args:
        image_path (str): Chemin vers l'image
        is_plum (bool): Si l'image contient une prune
        predicted_class (str): Classe prédite
        detection_probs (numpy.ndarray): Probabilités de détection
        classification_probs (numpy.ndarray): Probabilités de classification
        detection_class_names (list): Noms des classes de détection
        classification_class_names (list): Noms des classes de classification
        save_path (str): Chemin où sauvegarder la visualisation (si None, affiche seulement)
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
    
    if is_plum and classification_probs is not None and classification_class_names is not None:
        # Afficher les probabilités de classification
        y_pos = np.arange(len(classification_class_names))
        plt.barh(y_pos, classification_probs, align='center')
        plt.yticks(y_pos, classification_class_names)
        plt.xlabel('Probabilité')
        plt.title('Probabilités par classe')
    elif not is_plum and detection_probs is not None and detection_class_names is not None:
        # Afficher les probabilités de détection
        y_pos = np.arange(len(detection_class_names))
        plt.barh(y_pos, detection_probs, align='center')
        plt.yticks(y_pos, detection_class_names)
        plt.xlabel('Probabilité')
        plt.title('Probabilités de détection')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_confusion_matrix_plot(cm, class_names, save_path=None):
    """
    Crée une visualisation de la matrice de confusion.
    
    Args:
        cm (numpy.ndarray): Matrice de confusion
        class_names (list): Noms des classes
        save_path (str): Chemin où sauvegarder la visualisation (si None, affiche seulement)
    """
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité')
    plt.title('Matrice de confusion')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_metrics_plot(report, save_path=None):
    """
    Crée une visualisation des métriques par classe.
    
    Args:
        report (dict): Rapport de classification
        save_path (str): Chemin où sauvegarder la visualisation (si None, affiche seulement)
    """
    plt.figure(figsize=(12, 6))
    
    # Extraire les métriques par classe
    classes = list(report.keys())[:-3]  # Exclure 'accuracy', 'macro avg', 'weighted avg'
    precision = [report[cls]['precision'] for cls in classes]
    recall = [report[cls]['recall'] for cls in classes]
    f1 = [report[cls]['f1-score'] for cls in classes]
    
    # Créer le graphique
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-score')
    
    plt.xlabel('Classe')
    plt.ylabel('Score')
    plt.title('Métriques par classe')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_training_curves_plot(history, save_path=None):
    """
    Crée une visualisation des courbes d'entraînement.
    
    Args:
        history (dict): Historique d'entraînement
        save_path (str): Chemin où sauvegarder la visualisation (si None, affiche seulement)
    """
    plt.figure(figsize=(12, 4))
    
    # Graphique des pertes
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.title('Évolution des pertes')
    
    # Graphique de l'accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Évolution de l\'accuracy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def get_device_info():
    """
    Retourne des informations sur le device disponible.
    
    Returns:
        tuple: (device, device_name)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        device_name = torch.xpu.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    return device, device_name

def count_parameters(model):
    """
    Compte le nombre de paramètres entraînables dans un modèle.
    
    Args:
        model: Modèle PyTorch
        
    Returns:
        int: Nombre de paramètres
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model):
    """
    Génère un résumé du modèle.
    
    Args:
        model: Modèle PyTorch
        
    Returns:
        str: Résumé du modèle
    """
    from io import StringIO
    import sys
    
    # Rediriger stdout vers un StringIO
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    # Afficher le modèle
    print(model)
    
    # Compter les paramètres
    total_params = count_parameters(model)
    print(f"\nTotal des paramètres entraînables: {total_params:,}")
    
    # Restaurer stdout
    sys.stdout = old_stdout
    
    return mystdout.getvalue()
