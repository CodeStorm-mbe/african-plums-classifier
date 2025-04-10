"""
Script principal d'entraînement pour le classificateur de prunes.
Ce script gère l'entraînement, la validation et l'évaluation du modèle.
"""

import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Importer nos modules
from data.data_preprocessing import load_and_prepare_data, visualize_batch, analyze_dataset_distribution
from models.model_architecture import get_model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, num_epochs=25, early_stopping_patience=7, save_dir='models'):
    """
    Entraîne le modèle et sauvegarde le meilleur modèle.
    
    Args:
        model: Modèle à entraîner
        train_loader: DataLoader pour les données d'entraînement
        val_loader: DataLoader pour les données de validation
        criterion: Fonction de perte
        optimizer: Optimiseur
        scheduler: Scheduler pour ajuster le learning rate
        device: Device sur lequel entraîner (cuda ou cpu)
        num_epochs (int): Nombre d'époques
        early_stopping_patience (int): Nombre d'époques sans amélioration avant d'arrêter
        save_dir (str): Répertoire où sauvegarder les modèles
        
    Returns:
        dict: Historique d'entraînement
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialiser les variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stopping_counter = 0
    
    # Historique pour tracer les courbes
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    # Heure de début
    start_time = time.time()
    
    # Boucle d'entraînement
    for epoch in range(num_epochs):
        print(f"Époque {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        # Mode entraînement
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Boucle sur les batches d'entraînement
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Réinitialiser les gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass et optimisation
                loss.backward()
                optimizer.step()
            
            # Statistiques
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # Calculer les métriques d'entraînement
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Mode évaluation
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        # Boucle sur les batches de validation
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Statistiques
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # Calculer les métriques de validation
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.double() / len(val_loader.dataset)
        
        # Ajuster le learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)
        
        # Afficher les métriques
        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Mettre à jour l'historique
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_acc'].append(epoch_val_acc.item())
        history['lr'].append(current_lr)
        
        # Sauvegarder le meilleur modèle selon la perte de validation
        if epoch_val_loss < best_val_loss:
            print(f"Amélioration de la perte de validation de {best_val_loss:.4f} à {epoch_val_loss:.4f}. Sauvegarde du modèle...")
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_loss.pth'))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Sauvegarder le meilleur modèle selon l'accuracy de validation
        if epoch_val_acc > best_val_acc:
            print(f"Amélioration de l'accuracy de validation de {best_val_acc:.4f} à {epoch_val_acc:.4f}. Sauvegarde du modèle...")
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_acc.pth'))
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping après {early_stopping_counter} époques sans amélioration")
            break
        
        print()
    
    # Temps total d'entraînement
    time_elapsed = time.time() - start_time
    print(f"Entraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Meilleure perte de validation: {best_val_loss:.4f}")
    print(f"Meilleure accuracy de validation: {best_val_acc:.4f}")
    
    # Sauvegarder le dernier modèle
    torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))
    
    # Sauvegarder l'historique
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    return history

def evaluate_model(model, test_loader, criterion, device, class_names, save_dir='models'):
    """
    Évalue le modèle sur l'ensemble de test et génère des visualisations.
    
    Args:
        model: Modèle à évaluer
        test_loader: DataLoader pour les données de test
        criterion: Fonction de perte
        device: Device sur lequel évaluer (cuda ou cpu)
        class_names (list): Liste des noms de classes
        save_dir (str): Répertoire où sauvegarder les résultats
        
    Returns:
        dict: Métriques d'évaluation
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)
    
    # Mode évaluation
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    # Pour la matrice de confusion
    all_preds = []
    all_labels = []
    
    # Boucle sur les batches de test
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        
        # Statistiques
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # Collecter les prédictions et les labels pour la matrice de confusion
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculer les métriques
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    
    print(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
    
    # Créer la matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    
    # Visualiser la matrice de confusion
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité')
    plt.title('Matrice de confusion')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Générer le rapport de classification
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Sauvegarder le rapport
    with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Visualiser les métriques par classe
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
    plt.savefig(os.path.join(save_dir, 'metrics_by_class.png'))
    plt.close()
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc.item(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

def plot_training_history(history, save_dir='models'):
    """
    Trace les courbes d'entraînement.
    
    Args:
        history (dict): Historique d'entraînement
        save_dir (str): Répertoire où sauvegarder les graphiques
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)
    
    # Tracer les courbes de perte
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
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
    # Tracer l'évolution du learning rate
    plt.figure(figsize=(10, 4))
    plt.plot(history['lr'])
    plt.xlabel('Époque')
    plt.ylabel('Learning Rate')
    plt.title('Évolution du Learning Rate')
    plt.yscale('log')
    plt.savefig(os.path.join(save_dir, 'learning_rate.png'))
    plt.close()

def main():
    """
    Fonction principale pour l'entraînement du modèle.
    """
    # Parser les arguments
    parser = argparse.ArgumentParser(description='Entraînement du classificateur de prunes')
    parser.add_argument('--data_dir', type=str, required=True, help='Chemin vers le répertoire de données')
    parser.add_argument('--save_dir', type=str, default='models', help='Chemin pour sauvegarder les modèles')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'lightweight'], 
                        help='Type de modèle à utiliser')
    parser.add_argument('--base_model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50', 'mobilenet_v2', 'efficientnet_b0'], 
                        help='Modèle de base à utiliser (pour le type standard)')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille des batches')
    parser.add_argument('--num_epochs', type=int, default=25, help='Nombre d\'époques')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate initial')
    parser.add_argument('--img_size', type=int, default=224, help='Taille des images')
    parser.add_argument('--num_workers', type=int, default=4, help='Nombre de workers pour le chargement des données')
    parser.add_argument('--seed', type=int, default=42, help='Seed pour la reproductibilité')
    args = parser.parse_args()
    
    # Fixer les seeds pour la reproductibilité
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Déterminer le device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de: {device}")
    
    # Analyser la distribution des classes
    print("Analyse de la distribution des classes...")
    class_counts, dist_plot = analyze_dataset_distribution(args.data_dir)
    print(f"Distribution des classes: {class_counts}")
    
    # Charger et préparer les données
    print("Chargement des données...")
    train_loader, val_loader, test_loader, class_names = load_and_prepare_data(
        args.data_dir, 
        batch_size=args.batch_size, 
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # Visualiser un batch
    print("Visualisation d'un batch d'images...")
    batch_viz = visualize_batch(train_loader, class_names)
    
    # Créer le modèle
    print(f"Création du modèle {args.model_type} avec base {args.base_model if args.model_type == 'standard' else 'mobilenet_v2'}...")
    if args.model_type == 'standard':
        model = get_model(
            model_name=args.model_type, 
            num_classes=len(class_names), 
            base_model=args.base_model, 
            pretrained=True
        )
    else:
        model = get_model(
            model_name=args.model_type, 
            num_classes=len(class_names), 
            pretrained=True
        )
    
    # Déplacer le modèle sur le device
    model = model.to(device)
    
    # Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Scheduler pour ajuster le learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # Entraîner le modèle
    print("Début de l'entraînement...")
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        device, 
        num_epochs=args.num_epochs, 
        save_dir=args.save_dir
    )
    
    # Tracer les courbes d'entraînement
    print("Génération des courbes d'entraînement...")
    plot_training_history(history, save_dir=args.save_dir)
    
    # Charger le meilleur modèle (selon l'accuracy)
    print("Chargement du meilleur modèle...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model_acc.pth')))
    
    # Évaluer le modèle
    print("Évaluation du modèle sur l'ensemble de test...")
    metrics = evaluate_model(model, test_loader, criterion, device, class_names, save_dir=args.save_dir)
    
    # Afficher les métriques finales
    print(f"Métriques finales:")
    print(f"  - Accuracy: {metrics['test_acc']:.4f}")
    print(f"  - Loss: {metrics['test_loss']:.4f}")
    
    # Sauvegarder les informations du modèle
    model_info = model.get_model_info()
    model_info.update({
        'img_size': args.img_size,
        'class_names': class_names,
        'metrics': metrics
    })
    
    with open(os.path.join(args.save_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f"Entraînement et évaluation terminés. Résultats sauvegardés dans {args.save_dir}")

if __name__ == '__main__':
    main()
