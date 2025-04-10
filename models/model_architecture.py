"""
Module définissant l'architecture du modèle pour le classificateur de prunes.
Ce module contient les classes et fonctions pour créer les modèles de détection et de classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class PlumClassifier(nn.Module):
    """Modèle de classification des prunes."""
    def __init__(self, num_classes=6, base_model='resnet18', pretrained=True, dropout_rate=0.5):
        super(PlumClassifier, self).__init__()
        
        self.base_model_name = base_model
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
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
            self.base_model = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        
        # Classifier personnalisé avec dropout pour la régularisation
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)
    
    def get_model_info(self):
        return {
            "base_model": self.base_model_name,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate
        }

class LightweightPlumClassifier(nn.Module):
    """Version légère du classificateur de prunes."""
    def __init__(self, num_classes=6, pretrained=True):
        super(LightweightPlumClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Utiliser MobileNetV2 qui est plus léger
        self.base_model = models.mobilenet_v2(pretrained=pretrained)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)
    
    def get_model_info(self):
        return {
            "base_model": "mobilenet_v2_lightweight",
            "num_classes": self.num_classes,
            "dropout_rate": 0.2
        }

class TwoStageModel:
    """
    Modèle en deux étapes pour la détection et la classification des prunes.
    Première étape : détection de prune (prune ou non-prune)
    Deuxième étape : classification de l'état de la prune (si une prune est détectée)
    """
    def __init__(self, detection_model, classification_model, detection_threshold=0.7):
        self.detection_model = detection_model
        self.classification_model = classification_model
        self.detection_threshold = detection_threshold
        
    def predict(self, image_tensor, device):
        """
        Prédit si l'image contient une prune et, si oui, son état.
        
        Args:
            image_tensor (torch.Tensor): Tenseur de l'image prétraitée
            device: Device sur lequel exécuter les modèles
            
        Returns:
            tuple: (is_plum, predicted_class, probabilities)
            où is_plum est un booléen indiquant si l'image contient une prune,
            predicted_class est la classe prédite (si is_plum est True),
            et probabilities sont les probabilités pour chaque classe.
        """
        # Déplacer l'image sur le device
        image_tensor = image_tensor.to(device)
        
        # Étape 1 : Détection de prune
        self.detection_model.eval()
        with torch.no_grad():
            detection_outputs = self.detection_model(image_tensor)
            detection_probs = torch.nn.functional.softmax(detection_outputs, dim=1)[0]
            
            # Classe 0 = prune, Classe 1 = non-prune
            is_plum = detection_probs[0] > self.detection_threshold
            
        # Si ce n'est pas une prune, retourner le résultat
        if not is_plum:
            return False, "non_plum", detection_probs.cpu().numpy()
        
        # Étape 2 : Classification de l'état de la prune
        self.classification_model.eval()
        with torch.no_grad():
            classification_outputs = self.classification_model(image_tensor)
            classification_probs = torch.nn.functional.softmax(classification_outputs, dim=1)[0]
            _, predicted_idx = torch.max(classification_outputs, 1)
            
        return True, predicted_idx.item(), classification_probs.cpu().numpy()
    
    def get_model_info(self):
        detection_info = self.detection_model.get_model_info()
        classification_info = self.classification_model.get_model_info()
        
        return {
            "detection_model": detection_info,
            "classification_model": classification_info,
            "detection_threshold": self.detection_threshold
        }

def get_model(model_name='standard', num_classes=6, base_model='resnet18', pretrained=True):
    """
    Factory function pour créer un modèle.
    
    Args:
        model_name (str): Type de modèle ('standard' ou 'lightweight')
        num_classes (int): Nombre de classes
        base_model (str): Modèle de base à utiliser (pour le type standard)
        pretrained (bool): Utiliser des poids pré-entraînés
        
    Returns:
        nn.Module: Modèle créé
    """
    if model_name == 'standard':
        return PlumClassifier(num_classes=num_classes, base_model=base_model, pretrained=pretrained)
    elif model_name == 'lightweight':
        return LightweightPlumClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Modèle '{model_name}' non supporté")

def get_two_stage_model(detection_model_name='lightweight', classification_model_name='standard',
                       detection_base_model='mobilenet_v2', classification_base_model='resnet18',
                       num_detection_classes=2, num_classification_classes=6,
                       pretrained=True, detection_threshold=0.7):
    """
    Factory function pour créer un modèle en deux étapes.
    
    Args:
        detection_model_name (str): Type de modèle pour la détection
        classification_model_name (str): Type de modèle pour la classification
        detection_base_model (str): Modèle de base pour la détection
        classification_base_model (str): Modèle de base pour la classification
        num_detection_classes (int): Nombre de classes pour la détection (généralement 2)
        num_classification_classes (int): Nombre de classes pour la classification
        pretrained (bool): Utiliser des poids pré-entraînés
        detection_threshold (float): Seuil de confiance pour la détection
        
    Returns:
        TwoStageModel: Modèle en deux étapes
    """
    detection_model = get_model(
        model_name=detection_model_name,
        num_classes=num_detection_classes,
        base_model=detection_base_model,
        pretrained=pretrained
    )
    
    classification_model = get_model(
        model_name=classification_model_name,
        num_classes=num_classification_classes,
        base_model=classification_base_model,
        pretrained=pretrained
    )
    
    return TwoStageModel(detection_model, classification_model, detection_threshold)
