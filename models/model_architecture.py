"""
Module définissant l'architecture du modèle pour le classificateur de prunes.
Ce module contient différentes architectures de modèles pour la classification des prunes.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url

class PlumClassifier(nn.Module):
    """
    Modèle de classification des prunes basé sur un réseau pré-entraîné avec
    des couches personnalisées pour la classification.
    """
    def __init__(self, num_classes=6, base_model='resnet18', pretrained=True, dropout_rate=0.5):
        """
        Initialise le modèle de classification.
        
        Args:
            num_classes (int): Nombre de classes à prédire
            base_model (str): Modèle de base à utiliser ('resnet18', 'resnet50', 'mobilenet_v2', 'efficientnet_b0')
            pretrained (bool): Si True, utilise les poids pré-entraînés sur ImageNet
            dropout_rate (float): Taux de dropout pour la régularisation
        """
        super(PlumClassifier, self).__init__()
        
        self.base_model_name = base_model
        
        # Sélectionner le modèle de base
        if base_model == 'resnet18':
            self.base_model = models.resnet18(pretrained=pretrained)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Retirer la dernière couche
            
        elif base_model == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
            
        elif base_model == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(pretrained=pretrained)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
            
        elif base_model == 'efficientnet_b0':
            try:
                self.base_model = models.efficientnet_b0(pretrained=pretrained)
                num_features = self.base_model.classifier[1].in_features
                self.base_model.classifier = nn.Identity()
            except:
                # Fallback si EfficientNet n'est pas disponible dans la version de torchvision
                print("EfficientNet non disponible, utilisation de ResNet18 à la place")
                self.base_model = models.resnet18(pretrained=pretrained)
                num_features = self.base_model.fc.in_features
                self.base_model.fc = nn.Identity()
                self.base_model_name = 'resnet18'
        else:
            raise ValueError(f"Modèle de base '{base_model}' non supporté")
        
        # Classifier personnalisé avec dropout pour la régularisation
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),  # Moins de dropout dans les couches intermédiaires
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass du modèle.
        
        Args:
            x (torch.Tensor): Batch d'images d'entrée
            
        Returns:
            torch.Tensor: Logits de sortie
        """
        # Extraire les features avec le modèle de base
        features = self.base_model(x)
        # Classifier
        return self.classifier(features)
    
    def get_model_info(self):
        """
        Retourne les informations sur le modèle.
        
        Returns:
            dict: Informations sur le modèle
        """
        return {
            "base_model": self.base_model_name,
            "num_classes": self.classifier[-1].out_features,
            "dropout_rate": self.classifier[0].p
        }

class LightweightPlumClassifier(nn.Module):
    """
    Version légère du classificateur de prunes, optimisée pour les appareils
    avec des ressources limitées comme le Dell XPS avec Intel Arc.
    """
    def __init__(self, num_classes=6, pretrained=True):
        """
        Initialise le modèle léger.
        
        Args:
            num_classes (int): Nombre de classes à prédire
            pretrained (bool): Si True, utilise les poids pré-entraînés sur ImageNet
        """
        super(LightweightPlumClassifier, self).__init__()
        
        # Utiliser MobileNetV2 qui est plus léger
        self.base_model = models.mobilenet_v2(pretrained=pretrained)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass du modèle.
        
        Args:
            x (torch.Tensor): Batch d'images d'entrée
            
        Returns:
            torch.Tensor: Logits de sortie
        """
        return self.base_model(x)
    
    def get_model_info(self):
        """
        Retourne les informations sur le modèle.
        
        Returns:
            dict: Informations sur le modèle
        """
        return {
            "base_model": "mobilenet_v2_lightweight",
            "num_classes": self.base_model.classifier[1].out_features,
            "dropout_rate": self.base_model.classifier[0].p
        }

def get_model(model_name='standard', num_classes=6, **kwargs):
    """
    Factory function pour créer un modèle.
    
    Args:
        model_name (str): Nom du modèle ('standard' ou 'lightweight')
        num_classes (int): Nombre de classes
        **kwargs: Arguments supplémentaires pour le modèle
        
    Returns:
        nn.Module: Instance du modèle
    """
    if model_name == 'standard':
        return PlumClassifier(num_classes=num_classes, **kwargs)
    elif model_name == 'lightweight':
        return LightweightPlumClassifier(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Modèle '{model_name}' non supporté")
