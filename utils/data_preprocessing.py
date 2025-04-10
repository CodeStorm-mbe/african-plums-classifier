"""
Module de prétraitement des données pour le classificateur de prunes.
Ce module contient des fonctions pour charger, transformer et préparer les images de prunes
pour l'entraînement et l'évaluation du modèle.
"""

import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Définition des transformations pour les images
def get_train_transforms(img_size=224):
    """
    Retourne les transformations à appliquer aux images d'entraînement.
    Inclut l'augmentation de données pour améliorer la généralisation.
    
    Args:
        img_size (int): Taille des images (carré)
        
    Returns:
        transforms.Compose: Composition de transformations
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_val_transforms(img_size=224):
    """
    Retourne les transformations à appliquer aux images de validation/test.
    Pas d'augmentation de données pour ces ensembles.
    
    Args:
        img_size (int): Taille des images (carré)
        
    Returns:
        transforms.Compose: Composition de transformations
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_and_prepare_data(data_dir, batch_size=32, img_size=224, val_split=0.2, test_split=0.1, num_workers=4):
    """
    Charge et prépare les données pour l'entraînement, la validation et le test.
    
    Args:
        data_dir (str): Chemin vers le répertoire contenant les images organisées en sous-dossiers par classe
        batch_size (int): Taille des lots pour le DataLoader
        img_size (int): Taille des images (carré)
        val_split (float): Proportion des données à utiliser pour la validation
        test_split (float): Proportion des données à utiliser pour le test
        num_workers (int): Nombre de workers pour le chargement parallèle des données
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    # Vérifier si le répertoire existe
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Le répertoire {data_dir} n'existe pas")
    
    # Charger le dataset complet avec les transformations d'entraînement
    full_dataset = datasets.ImageFolder(root=data_dir, transform=get_train_transforms(img_size))
    
    # Récupérer les noms des classes
    class_names = full_dataset.classes
    print(f"Classes trouvées: {class_names}")
    
    # Calculer les tailles des ensembles
    dataset_size = len(full_dataset)
    test_size = int(dataset_size * test_split)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size - test_size
    
    # Diviser le dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Pour la reproductibilité
    )
    
    # Appliquer les transformations appropriées à chaque ensemble
    # Note: Nous devons modifier la transformation du dataset sous-jacent pour chaque sous-ensemble
    train_dataset.dataset.transform = get_train_transforms(img_size)
    val_dataset.dataset.transform = get_val_transforms(img_size)
    test_dataset.dataset.transform = get_val_transforms(img_size)
    
    print(f"Répartition des données: Train={train_size}, Validation={val_size}, Test={test_size}")
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names

def visualize_batch(dataloader, class_names, num_images=8):
    """
    Visualise un lot d'images avec leurs étiquettes.
    
    Args:
        dataloader: DataLoader contenant les images
        class_names (list): Liste des noms de classes
        num_images (int): Nombre d'images à afficher
    """
    # Obtenir un lot d'images
    images, labels = next(iter(dataloader))
    
    # Limiter au nombre d'images demandé
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Dénormaliser les images pour l'affichage
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    images = images * std + mean
    
    # Convertir en numpy et transposer pour l'affichage
    images = images.numpy().transpose((0, 2, 3, 1))
    
    # Afficher les images
    fig, axes = plt.subplots(2, num_images//2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, (img, label) in enumerate(zip(images, labels)):
        # Clip pour s'assurer que les valeurs sont entre 0 et 1
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"Classe: {class_names[label]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('batch_visualization.png')
    plt.close()
    
    return 'batch_visualization.png'

def analyze_dataset_distribution(data_dir):
    """
    Analyse la distribution des classes dans le dataset.
    
    Args:
        data_dir (str): Chemin vers le répertoire contenant les images organisées en sous-dossiers par classe
        
    Returns:
        dict: Dictionnaire contenant le nombre d'images par classe
    """
    class_counts = {}
    
    # Parcourir les sous-dossiers (classes)
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            # Compter les fichiers d'image
            image_count = len([f for f in os.listdir(class_path) 
                              if os.path.isfile(os.path.join(class_path, f)) and 
                              f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
            class_counts[class_name] = image_count
    
    # Afficher la distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Distribution des classes dans le dataset')
    plt.xlabel('Classe')
    plt.ylabel('Nombre d\'images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()
    
    return class_counts, 'class_distribution.png'

def preprocess_single_image(image_path, transform=None):
    """
    Prétraite une seule image pour la prédiction.
    
    Args:
        image_path (str): Chemin vers l'image
        transform: Transformations à appliquer (si None, utilise get_val_transforms)
        
    Returns:
        torch.Tensor: Tenseur de l'image prétraitée
    """
    if transform is None:
        transform = get_val_transforms()
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Ajouter une dimension de batch
