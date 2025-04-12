import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
import timm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block pour l'attention de canal.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EnhancedPlumClassifier(nn.Module):
    """
    Modèle amélioré de classification des prunes basé sur EfficientNet.
    Utilise des intervalles de confiance pour déterminer si un échantillon est une prune ou non.
    Inclut des mécanismes d'attention et des connexions résiduelles.
    """
    def __init__(self, num_classes=6, model_name='efficientnet_b4', pretrained=True, dropout_rate=0.4, confidence_threshold=0.7):
        """
        Initialise le modèle de classification amélioré.
        
        Args:
            num_classes (int): Nombre de classes (6 catégories de prunes)
            model_name (str): Nom du modèle de base à utiliser
            pretrained (bool): Si True, utilise des poids pré-entraînés
            dropout_rate (float): Taux de dropout
            confidence_threshold (float): Seuil de confiance pour les prédictions
        """
        super(EnhancedPlumClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        
        # Chargement du modèle de base
        self.base_model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        # Récupération des dimensions des features de sortie
        dummy_input = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            features = self.base_model(dummy_input)
        
        # Utilisation des features de la dernière couche
        last_channel = features[-1].shape[1]
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Squeeze-and-Excitation block
        self.se_block = SEBlock(last_channel)
        
        # Couches de classification avec connexions résiduelles
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(last_channel, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        # Connexion résiduelle
        self.shortcut = nn.Linear(last_channel, 512)
        
        # Couche finale de classification
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Couche de confiance (pour estimer la fiabilité de la prédiction)
        self.confidence_fc = nn.Linear(512, 1)
        
        # Mapping des indices aux noms de classes
        self.idx_to_class = {
            0: 'bonne_qualite',
            1: 'non_mure',
            2: 'tachetee',
            3: 'fissuree',
            4: 'meurtrie',
            5: 'pourrie'
        }
    
    def forward(self, x):
        """
        Passe avant du modèle.
        
        Args:
            x (torch.Tensor): Batch d'images
            
        Returns:
            tuple: (logits, confidence)
        """
        # Extraction des features
        features = self.base_model(x)
        x = features[-1]  # Utiliser les features de la dernière couche
        
        # Appliquer l'attention
        x = self.se_block(x)
        
        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Sauvegarde pour la connexion résiduelle
        residual = self.shortcut(x)
        
        # Première couche fully connected
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Deuxième couche fully connected
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        
        # Ajouter la connexion résiduelle
        x = x + residual
        x = F.relu(x)
        
        # Couche finale pour la classification
        features_for_confidence = x  # Sauvegarder les features pour la confiance
        
        x = self.dropout3(x)
        logits = self.fc3(x)
        
        # Score de confiance
        confidence = torch.sigmoid(self.confidence_fc(features_for_confidence))
        
        return logits, confidence
    
    def predict_with_confidence(self, x):
        """
        Prédit la classe avec un score de confiance.
        Utilise les intervalles de confiance pour déterminer si un échantillon est une prune ou non.
        
        Args:
            x (torch.Tensor): Batch d'images
            
        Returns:
            dict: Dictionnaire contenant les résultats de la prédiction
        """
        self.eval()
        with torch.no_grad():
            logits, confidence = self(x)
            
            # Probabilités de classe
            probs = F.softmax(logits, dim=1)
            
            # Classe prédite et probabilité maximale
            max_prob, predicted = torch.max(probs, 1)
            
            # Ajustement de la confiance en fonction de la probabilité maximale
            adjusted_confidence = confidence.squeeze() * max_prob
            
            # Déterminer si l'échantillon est une prune en utilisant l'intervalle de confiance
            est_prune = adjusted_confidence >= self.confidence_threshold
            
            # Récupérer le nom de la classe
            class_idx = predicted.item()
            class_name = self.idx_to_class[class_idx]
            
            # Retourner les résultats sous forme de dictionnaire
            results = {
                'class_idx': class_idx,
                'confidence': adjusted_confidence.item(),
                'class_name': class_name,
                'est_prune': est_prune.item(),
                'probabilities': probs.squeeze().tolist()
            }
            
            return results

class EnsemblePlumClassifier(nn.Module):
    """
    Modèle d'ensemble combinant plusieurs modèles EnhancedPlumClassifier.
    """
    def __init__(self, models):
        super(EnsemblePlumClassifier, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = models[0].num_classes
        self.confidence_threshold = models[0].confidence_threshold
        self.idx_to_class = models[0].idx_to_class

    def forward(self, x):
        """
        Passe avant combinant les prédictions des modèles.
        """
        logits_list = []
        confidence_list = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits, confidence = model(x)
                logits_list.append(logits)
                confidence_list.append(confidence)
        
        # Moyenne des logits et confidences
        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        avg_confidence = torch.mean(torch.stack(confidence_list), dim=0)
        return avg_logits, avg_confidence

    def predict_with_confidence(self, x):
        """
        Prédit la classe avec un score de confiance pour l'ensemble.
        """
        self.eval()
        with torch.no_grad():
            logits, confidence = self(x)
            probs = F.softmax(logits, dim=1)
            max_prob, predicted = torch.max(probs, 1)
            adjusted_confidence = confidence.squeeze() * max_prob
            est_prune = adjusted_confidence >= self.confidence_threshold
            class_idx = predicted.item()
            class_name = self.idx_to_class[class_idx]

            results = {
                'class_idx': class_idx,
                'confidence': adjusted_confidence.item(),
                'class_name': class_name,
                'est_prune': est_prune.item(),
                'probabilities': probs.squeeze().tolist()
            }
            return results

class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer le déséquilibre des classes.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # S'assurer que alpha est sur le même dispositif que les entrées
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
        else:
            alpha = None
            
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss pour améliorer la généralisation.
    """
    def __init__(self, classes, smoothing=0.1, dim=-1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.weight = weight
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        if self.weight is not None:
            # S'assurer que weight est sur le même dispositif que pred
            weight = self.weight.to(pred.device)
            loss = -torch.sum(true_dist * pred * weight.unsqueeze(0), dim=self.dim)
        else:
            loss = -torch.sum(true_dist * pred, dim=self.dim)
            
        return loss.mean()

class MixupCutmixTransform:
    """
    Implémentation combinée des techniques Mixup et CutMix pour l'augmentation de données.
    """
    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, switch_prob=0.5, num_classes=6):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        
    def __call__(self, batch):
        images, labels, paths = batch
        
        # Décider si on utilise Mixup ou CutMix
        use_cutmix = np.random.rand() < self.switch_prob
        
        if use_cutmix:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            batch_size = images.size(0)
            index = torch.randperm(batch_size)
            
            # Générer les coordonnées du rectangle à couper
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            
            # Appliquer CutMix
            images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
            
            # Ajuster les labels en fonction de la proportion de l'image originale
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        else:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch_size = images.size(0)
            index = torch.randperm(batch_size)
            
            # Mixer les images
            mixed_images = lam * images + (1 - lam) * images[index, :]
            images = mixed_images
        
        # Créer des labels mixés en utilisant one-hot encoding
        one_hot_labels = torch.zeros(batch_size, self.num_classes, device=images.device)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
        
        one_hot_labels_mixed = lam * one_hot_labels + (1 - lam) * one_hot_labels[index]
        
        return images, one_hot_labels_mixed, paths
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        # Coordonnées uniformes
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

class EnhancedPlumClassifierModule(pl.LightningModule):
    """
    Module PyTorch Lightning amélioré pour l'entraînement du modèle de classification des prunes.
    """
    def __init__(self, model_params, class_weights=None, use_mixup_cutmix=True, mixup_alpha=1.0, cutmix_alpha=1.0, 
                 learning_rate=3e-4, weight_decay=1e-4, label_smoothing=0.1, use_focal_loss=True, gamma=2.0,
                 use_one_cycle=True, max_epochs=30, use_amp=True):
        """
        Initialise le module.
        """
        super(EnhancedPlumClassifierModule, self).__init__()
        
        # Sauvegarde des hyperparamètres
        self.save_hyperparameters()
        
        # Initialisation du modèle
        self.model = EnhancedPlumClassifier(**model_params)
        
        # Poids des classes - Assurez-vous qu'ils sont sur le bon périphérique
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)  # Ceci déplacera automatiquement les poids sur le bon périphérique
        else:
            self.class_weights = None
        
        # Mixup et CutMix
        self.use_mixup_cutmix = use_mixup_cutmix
        self.mixup_cutmix_transform = MixupCutmixTransform(
            mixup_alpha=mixup_alpha, 
            cutmix_alpha=cutmix_alpha,
            num_classes=model_params['num_classes']
        ) if use_mixup_cutmix else None
        
        # Paramètres d'entraînement
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_one_cycle = use_one_cycle
        self.max_epochs = max_epochs
        self.use_amp = use_amp
        
        # Pertes
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma
        
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights, gamma=gamma)
        else:
            self.criterion = LabelSmoothingLoss(
                classes=model_params['num_classes'],
                smoothing=label_smoothing,
                weight=class_weights
            )
        
        # Métriques
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=model_params['num_classes'])
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=model_params['num_classes'])
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=model_params['num_classes'])
        
        # Température pour le calibrage des probabilités
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Compteur d'époques pour le gel/dégel progressif
        self.current_epoch_num = 0
    
    def forward(self, x):
        """
        Passe avant du modèle.
        
        Args:
            x (torch.Tensor): Batch d'images
            
        Returns:
            tuple: (logits, confidence)
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Étape d'entraînement.
        
        Args:
            batch (tuple): (images, labels, _)
            batch_idx (int): Indice du batch
            
        Returns:
            torch.Tensor: Perte
        """
        images, labels, _ = batch
        
        # Appliquer Mixup/CutMix si activé
        if self.use_mixup_cutmix and self.current_epoch_num < self.max_epochs // 2:
            images, mixed_labels, _ = self.mixup_cutmix_transform((images, labels, _))
            
            # Passe avant avec AMP si activé
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    logits, confidence = self(images)
                    
                    # Perte avec labels mixés
                    if isinstance(mixed_labels, torch.Tensor) and len(mixed_labels.shape) > 1:
                        # Si les labels sont one-hot encoded
                        loss = -torch.sum(F.log_softmax(logits, dim=1) * mixed_labels, dim=1).mean()
                    else:
                        # Sinon, utiliser la perte standard
                        loss = self.criterion(logits, mixed_labels)
            else:
                # Passe avant sans AMP
                logits, confidence = self(images)
                
                # Perte avec labels mixés
                if isinstance(mixed_labels, torch.Tensor) and len(mixed_labels.shape) > 1:
                    # Si les labels sont one-hot encoded
                    loss = -torch.sum(F.log_softmax(logits, dim=1) * mixed_labels, dim=1).mean()
                else:
                    # Sinon, utiliser la perte standard
                    loss = self.criterion(logits, mixed_labels)
        else:
            # Passe avant avec AMP si activé
            if self.use_amp:
                with autocast():
                    logits, confidence = self(images)
                    loss = self.criterion(logits, labels)
            else:
                # Passe avant sans AMP
                logits, confidence = self(images)
                loss = self.criterion(logits, labels)
        
        # Calcul de l'exactitude
        if isinstance(labels, torch.Tensor) and len(labels.shape) > 1:
            # Si les labels sont one-hot encoded
            _, labels_idx = torch.max(labels, dim=1)
            preds = torch.argmax(logits, dim=1)
            acc = self.train_acc(preds, labels_idx)
        else:
            # Sinon, utiliser les labels directement
            preds = torch.argmax(logits, dim=1)
            acc = self.train_acc(preds, labels)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Étape de validation.
        
        Args:
            batch (tuple): (images, labels, _)
            batch_idx (int): Indice du batch
            
        Returns:
            dict: Résultats de validation
        """
        images, labels, _ = batch
        
        # Passe avant avec AMP si activé
        if self.use_amp:
            with autocast():
                logits, confidence = self(images)
                loss = self.criterion(logits, labels)
        else:
            # Passe avant sans AMP
            logits, confidence = self(images)
            loss = self.criterion(logits, labels)
        
        # Calcul de l'exactitude
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)
        
        # Logging
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        # Retourner les prédictions et les labels pour l'analyse
        return {'val_loss': loss, 'preds': preds, 'labels': labels, 'confidence': confidence.squeeze()}
    
    def test_step(self, batch, batch_idx):
        """
        Étape de test.
        
        Args:
            batch (tuple): (images, labels, _)
            batch_idx (int): Indice du batch
            
        Returns:
            dict: Résultats de test
        """
        images, labels, _ = batch
        
        # Passe avant avec AMP si activé
        if self.use_amp:
            with autocast():
                logits, confidence = self(images)
                
                # Appliquer la température pour le calibrage
                scaled_logits = logits / self.temperature
                
                # Perte
                loss = self.criterion(scaled_logits, labels)
        else:
            # Passe avant sans AMP
            logits, confidence = self(images)
            
            # Appliquer la température pour le calibrage
            scaled_logits = logits / self.temperature
            
            # Perte
            loss = self.criterion(scaled_logits, labels)
        
        # Calcul de l'exactitude
        preds = torch.argmax(scaled_logits, dim=1)
        acc = self.test_acc(preds, labels)
        
        # Logging
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)
        
        # Retourner les prédictions et les labels pour l'analyse
        return {'test_loss': loss, 'preds': preds, 'labels': labels, 'logits': scaled_logits, 'confidence': confidence.squeeze()}
    
    def configure_optimizers(self):
        """
        Configure l'optimiseur et le scheduler.
        
        Returns:
            dict: Configuration de l'optimiseur et du scheduler
        """
        # Optimiseur
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        if self.use_one_cycle:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1000.0
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
                eta_min=self.learning_rate / 100
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
    
    def on_train_epoch_start(self):
        """
        Actions au début d'une époque d'entraînement.
        """
        # Mettre à jour le compteur d'époques
        self.current_epoch_num = self.current_epoch
        
        # Gel/dégel progressif des couches du modèle de base
        if self.current_epoch_num < self.max_epochs // 4:
            # Geler le modèle de base pendant les premières époques
            for param in self.model.base_model.parameters():
                param.requires_grad = False
        elif self.current_epoch_num < self.max_epochs // 2:
            # Dégeler progressivement les dernières couches
            layers = list(self.model.base_model.children())
            for layer in layers[:-3]:
                for param in layer.parameters():
                    param.requires_grad = False
            for layer in layers[-3:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            # Dégeler complètement le modèle de base
            for param in self.model.base_model.parameters():
                param.requires_grad = True
    
    def on_validation_epoch_end(self):
        """
        Actions à la fin d'une époque de validation.
        """
        # Calibrage de la température
        if self.current_epoch_num % 5 == 0:
            self.temperature.data = torch.clamp(self.temperature.data, min=0.5, max=5.0)
            self.log('temperature', self.temperature.item())

class ModelTrainer:
    """
    Classe pour l'entraînement et l'évaluation du modèle.
    """
    def __init__(self, model_params, trainer_params, models_dir, logs_dir, trainer_config=None):
        """
        Initialise le trainer.
        
        Args:
            model_params (dict): Paramètres du modèle
            trainer_params (dict): Paramètres du trainer PyTorch Lightning
            models_dir (str): Répertoire où sauvegarder les modèles
            logs_dir (str): Répertoire où sauvegarder les logs
            trainer_config (dict, optional): Configuration supplémentaire pour l'entraînement
        """
        self.model_params = model_params
        self.trainer_params = trainer_params
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        self.trainer_config = trainer_config or {}
        
        # Création des répertoires
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Initialisation du modèle
        self.model = None
        self.trainer = None
    
    def train(self, train_dataloader, val_dataloader, class_weights=None, experiment_name=None):
        """
        Entraîne le modèle.
        
        Args:
            train_dataloader (DataLoader): DataLoader d'entraînement
            val_dataloader (DataLoader): DataLoader de validation
            class_weights (torch.Tensor, optional): Poids des classes
            experiment_name (str, optional): Nom de l'expérience
            
        Returns:
            dict: Résultats de l'entraînement
        """
        # Nom de l'expérience
        if experiment_name is None:
            experiment_name = f"plum_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialisation du module
        self.model = EnhancedPlumClassifierModule(
            model_params=self.model_params,
            class_weights=class_weights,
            use_mixup_cutmix=self.trainer_config.get('use_mixup_cutmix', True),
            mixup_alpha=self.trainer_config.get('mixup_alpha', 1.0),
            cutmix_alpha=self.trainer_config.get('cutmix_alpha', 1.0),
            learning_rate=self.trainer_config.get('learning_rate', 3e-4),
            weight_decay=self.trainer_config.get('weight_decay', 1e-4),
            label_smoothing=self.trainer_config.get('label_smoothing', 0.1),
            use_focal_loss=self.trainer_config.get('use_focal_loss', True),
            gamma=self.trainer_config.get('gamma', 2.0),
            use_one_cycle=self.trainer_config.get('use_one_cycle', True),
            max_epochs=self.trainer_params.get('max_epochs', 30),
            use_amp=self.trainer_config.get('use_amp', True)
        )
        
        # Callbacks
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=self.models_dir,
                filename=f"{experiment_name}_{{epoch:02d}}_{{val_acc:.4f}}",
                monitor='val_acc',
                mode='max',
                save_top_k=3,
                verbose=True
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min',
                verbose=True
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ]
        
        # Logger
        logger = pl.loggers.TensorBoardLogger(
            save_dir=self.logs_dir,
            name=experiment_name
        )
        
        # Initialisation du trainer
        self.trainer = pl.Trainer(
            **self.trainer_params,
            callbacks=callbacks,
            logger=logger
        )
        
        # Entraînement
        self.trainer.fit(self.model, train_dataloader, val_dataloader)
        
        # Chargement du meilleur modèle
        best_model_path = callbacks[0].best_model_path
        if best_model_path:
            self.model = EnhancedPlumClassifierModule.load_from_checkpoint(best_model_path)
        
        # Sauvegarde du modèle final
        model_path, metadata_path = self.save_model(experiment_name=experiment_name)
        
        # Résultats
        results = {
            'model_path': model_path,
            'metadata_path': metadata_path,
            'best_model_path': best_model_path,
            'experiment_name': experiment_name
        }
        
        return results

    def train_with_cross_validation(self, folds, experiment_name=None):
        """
        Entraîne le modèle avec validation croisée.
        
        Args:
            folds (list): Liste de dictionnaires contenant les données pour chaque pli
            experiment_name (str, optional): Nom de l'expérience
            
        Returns:
            dict: Résultats de l'entraînement
        """
        # Nom de l'expérience
        if experiment_name is None:
            experiment_name = f"plum_classifier_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Résultats pour chaque pli
        fold_results = []
        
        # Entraînement sur chaque pli
        for fold_idx, fold_data in enumerate(folds):
            fold_num = fold_idx + 1
            fold_experiment_name = f"{experiment_name}_fold_{fold_num}"
            
            print(f"Entraînement sur le pli {fold_num}...")
            
            # Entraînement sur ce pli
            result = self.train(
                train_dataloader=fold_data['train_dataloader'],
                val_dataloader=fold_data['val_dataloader'],
                class_weights=fold_data.get('class_weights', None),
                experiment_name=fold_experiment_name
            )
            
            # Sauvegarde explicite avec fold_num
            model_path, metadata_path = self.save_model(experiment_name=experiment_name, fold_num=fold_num)
            
            fold_results.append({
                'fold': fold_num,
                'model_path': model_path,
                'metadata_path': metadata_path,
                'best_model_path': result['best_model_path']
            })
        
        # Créer le modèle d'ensemble
        ensemble_path, ensemble_metadata_path = self.create_ensemble_model(fold_results, experiment_name)
        
        # Résultats globaux
        results = {
            'fold_results': fold_results,
            'experiment_name': experiment_name,
            'ensemble_path': ensemble_path,
            'ensemble_metadata_path': ensemble_metadata_path
        }
        
        return results
    
    def evaluate(self, test_dataloader):
        """
        Évalue le modèle sur l'ensemble de test.
        
        Args:
            test_dataloader (DataLoader): DataLoader de test
            
        Returns:
            dict: Résultats de l'évaluation
        """
        if self.model is None or self.trainer is None:
            raise ValueError("Le modèle doit être entraîné avant l'évaluation")
        
        # Évaluation
        test_results = self.trainer.test(self.model, test_dataloader)[0]
        
        # Récupération des prédictions et des labels
        all_preds = []
        all_labels = []
        all_confidences = []
        all_logits = []
        
        for batch in test_dataloader:
            images, labels, _ = batch
            
            # Passe avant
            with torch.no_grad():
                logits, confidence = self.model(images)
                
                # Appliquer la température pour le calibrage
                scaled_logits = logits / self.model.temperature
                
                # Prédictions
                probs = F.softmax(scaled_logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Stockage
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_confidences.append(confidence.squeeze().cpu().numpy())
                all_logits.append(scaled_logits.cpu().numpy())
        
        # Concaténation
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_confidences = np.concatenate(all_confidences)
        all_logits = np.concatenate(all_logits)
        
        # Calcul des métriques
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        # Matrice de confusion
        cm = confusion_matrix(all_labels, all_preds)
        
        # Rapport de classification
        class_names = [self.model.model.idx_to_class[i] for i in range(self.model.model.num_classes)]
        report = classification_report(all_labels, all_preds, target_names=class_names)
        
        # Visualisation de la matrice de confusion
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Prédiction')
        plt.ylabel('Vérité terrain')
        plt.title('Matrice de confusion')
        plt.tight_layout()
        
        # Sauvegarde de la matrice de confusion
        confusion_matrix_path = os.path.join(self.logs_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.close()
        
        # Analyse des intervalles de confiance
        confidence_thresholds = np.linspace(0.1, 0.9, 9)
        confidence_metrics = []
        
        for threshold in confidence_thresholds:
            # Filtrer les prédictions avec une confiance supérieure au seuil
            high_confidence_mask = all_confidences >= threshold
            
            if np.sum(high_confidence_mask) > 0:
                # Calculer l'exactitude pour ces prédictions
                high_confidence_acc = accuracy_score(
                    all_labels[high_confidence_mask], 
                    all_preds[high_confidence_mask]
                )
                
                # Calculer le pourcentage d'échantillons conservés
                retention_rate = np.mean(high_confidence_mask)
                
                confidence_metrics.append({
                    'threshold': threshold,
                    'accuracy': high_confidence_acc,
                    'retention_rate': retention_rate
                })
        
        # Visualisation de l'analyse des intervalles de confiance
        if confidence_metrics:
            thresholds = [m['threshold'] for m in confidence_metrics]
            accuracies = [m['accuracy'] for m in confidence_metrics]
            retention_rates = [m['retention_rate'] for m in confidence_metrics]
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            color = 'tab:blue'
            ax1.set_xlabel('Seuil de confiance')
            ax1.set_ylabel('Exactitude', color=color)
            ax1.plot(thresholds, accuracies, marker='o', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Taux de rétention', color=color)
            ax2.plot(thresholds, retention_rates, marker='s', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Analyse des intervalles de confiance')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Sauvegarde de l'analyse des intervalles de confiance
            confidence_analysis_path = os.path.join(self.logs_dir, 'confidence_analysis.png')
            plt.savefig(confidence_analysis_path)
            plt.close()
        
        # Résultats
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'confusion_matrix_path': confusion_matrix_path,
            'confidence_metrics': confidence_metrics
        }
        
        if confidence_metrics:
            results['confidence_analysis_path'] = confidence_analysis_path
        
        return results
    
    def evaluate_with_tta(self, test_dataloader, tta_transforms):
        """
        Évalue le modèle sur l'ensemble de test avec Test-Time Augmentation (TTA).
        
        Args:
            test_dataloader (DataLoader): DataLoader de test
            tta_transforms (list): Liste de transformations pour TTA
            
        Returns:
            dict: Résultats de l'évaluation
        """
        if self.model is None or self.trainer is None:
            raise ValueError("Le modèle doit être entraîné avant l'évaluation")
        
        # Récupération des prédictions et des labels
        all_preds = []
        all_labels = []
        
        # Mettre le modèle en mode évaluation
        self.model.eval()
        
        # Parcourir le dataset de test
        for images, labels, paths in test_dataloader:
            batch_size = images.size(0)
            
            # Prédictions pour chaque transformation
            tta_probs = []
            
            # Prédiction avec l'image originale
            with torch.no_grad():
                logits, _ = self.model(images.to(self.model.device))
                probs = F.softmax(logits, dim=1)
                tta_probs.append(probs.cpu().numpy())
            
            # Prédictions avec les transformations TTA
            for transform in tta_transforms:
                # Appliquer la transformation à chaque image du batch
                transformed_images = []
                for i in range(batch_size):
                    img = images[i].cpu().numpy().transpose(1, 2, 0)
                    
                    # Dénormaliser l'image
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = std * img + mean
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    
                    # Appliquer la transformation
                    augmented = transform(image=img)
                    transformed_img = augmented['image']
                    
                    transformed_images.append(transformed_img)
                
                # Convertir en tensor et prédire
                transformed_tensor = torch.stack(transformed_images).to(self.model.device)
                
                with torch.no_grad():
                    logits, _ = self.model(transformed_tensor)
                    probs = F.softmax(logits, dim=1)
                    tta_probs.append(probs.cpu().numpy())
            
            # Moyenne des probabilités
            avg_probs = np.mean(tta_probs, axis=0)
            preds = np.argmax(avg_probs, axis=1)
            
            # Stockage
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
        
        # Concaténation
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Calcul des métriques
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        # Matrice de confusion
        cm = confusion_matrix(all_labels, all_preds)
        
        # Rapport de classification
        class_names = [self.model.model.idx_to_class[i] for i in range(self.model.model.num_classes)]
        report = classification_report(all_labels, all_preds, target_names=class_names)
        
        # Visualisation de la matrice de confusion
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Prédiction')
        plt.ylabel('Vérité terrain')
        plt.title('Matrice de confusion (TTA)')
        plt.tight_layout()
        
        # Sauvegarde de la matrice de confusion
        confusion_matrix_path = os.path.join(self.logs_dir, 'confusion_matrix_tta.png')
        plt.savefig(confusion_matrix_path)
        plt.close()
        
        # Résultats
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'confusion_matrix_path': confusion_matrix_path
        }
        
        return results
    
    def save_model(self, path=None, experiment_name=None, fold_num=None):
        """
        Sauvegarde le modèle.
        
        Args:
            path (str, optional): Chemin où sauvegarder le modèle
            experiment_name (str, optional): Nom de l'expérience
            fold_num (int, optional): Numéro du pli pour la validation croisée
        
        Returns:
            tuple: (model_path, metadata_path)
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant la sauvegarde")
        
        if path is None:
            # Générer un nom unique basé sur l'expérience et le pli
            base_name = "plum_classifier"
            if experiment_name:
                base_name += f"_{experiment_name}"
            if fold_num is not None:
                base_name += f"_fold_{fold_num}"
            path = os.path.join(self.models_dir, f"{base_name}.pt")
        
        # Sauvegarde du modèle PyTorch
        torch.save(self.model.model.state_dict(), path)
        
        # Sauvegarde des métadonnées
        metadata = {
            'num_classes': self.model.model.num_classes,
            'confidence_threshold': self.model.model.confidence_threshold,
            'model_name': self.model_params.get('model_name', 'efficientnet_b4'),
            'idx_to_class': self.model.model.idx_to_class,
            'temperature': self.model.temperature.item()
        }
        
        metadata_path = os.path.splitext(path)[0] + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Modèle sauvegardé dans {path}")
        print(f"Métadonnées sauvegardées dans {metadata_path}")
        
        return path, metadata_path
    
    def create_ensemble_model(self, fold_results, experiment_name=None):
        """
        Crée un modèle d'ensemble à partir des modèles de chaque pli.
        
        Args:
            fold_results (list): Liste des résultats de chaque pli contenant les chemins des modèles
            experiment_name (str, optional): Nom de l'expérience pour nommer le modèle final
            
        Returns:
            tuple: (ensemble_model_path, ensemble_metadata_path)
        """
        if not fold_results:
            raise ValueError("Aucun résultat de pli fourni pour créer l'ensemble")

        # Charger les modèles de chaque pli
        models = []
        for result in fold_results:
            model_path = result['model_path']
            metadata_path = result['metadata_path']
            
            # Charger le modèle
            model = self.load_model(model_path, metadata_path)
            model.eval()
            models.append(model)

        # Créer le modèle d'ensemble
        ensemble_model = EnsemblePlumClassifier(models)
        
        # Définir le chemin pour sauvegarder le modèle d'ensemble
        if experiment_name is None:
            experiment_name = f"plum_classifier_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        ensemble_path = os.path.join(self.models_dir, f"{experiment_name}_ensemble.pt")
        
        # Sauvegarder l'état de l'ensemble
        torch.save(ensemble_model.state_dict(), ensemble_path)
        
        # Sauvegarder les métadonnées
        metadata = {
            'num_classes': ensemble_model.num_classes,
            'confidence_threshold': ensemble_model.confidence_threshold,
            'model_name': 'ensemble_efficientnet_b4',
            'idx_to_class': ensemble_model.idx_to_class,
            'num_models': len(models)
        }
        
        metadata_path = os.path.splitext(ensemble_path)[0] + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Modèle d'ensemble sauvegardé dans {ensemble_path}")
        print(f"Métadonnées de l'ensemble sauvegardées dans {metadata_path}")
        
        return ensemble_path, metadata_path

    def load_model(self, path, metadata_path=None):
        """
        Charge un modèle sauvegardé.
        
        Args:
            path (str): Chemin vers le modèle sauvegardé
            metadata_path (str, optional): Chemin vers les métadonnées du modèle
            
        Returns:
            EnhancedPlumClassifier: Modèle chargé
        """
        # Chargement des métadonnées
        if metadata_path is None:
            metadata_path = os.path.splitext(path)[0] + '_metadata.json'
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Initialisation du modèle
        model = EnhancedPlumClassifier(
            num_classes=metadata['num_classes'],
            model_name=metadata['model_name'],
            confidence_threshold=metadata['confidence_threshold']
        )
        
        # Chargement des poids
        model.load_state_dict(torch.load(path))
        
        # Mise à jour du mapping des indices aux classes
        model.idx_to_class = metadata['idx_to_class']
        
        return model
    
    def load_ensemble_model(self, ensemble_path, metadata_path):
        """
        Charge un modèle d'ensemble sauvegardé.
        
        Args:
            ensemble_path (str): Chemin vers le modèle d'ensemble sauvegardé
            metadata_path (str): Chemin vers les métadonnées du modèle
            
        Returns:
            EnsemblePlumClassifier: Modèle d'ensemble chargé
        """
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Charger les modèles individuels (suppose que les fichiers .pt des plis sont disponibles)
        # Note : Vous devez conserver les chemins des modèles individuels dans les métadonnées
        models = []
        for i in range(metadata.get('num_models', 1)):
            # Supposons que les modèles individuels sont dans le même répertoire
            model_path = ensemble_path.replace('_ensemble.pt', f'_fold_{i+1}.pt')
            if os.path.exists(model_path):
                model = self.load_model(model_path, model_path.replace('.pt', '_metadata.json'))
                models.append(model)
            else:
                raise FileNotFoundError(f"Modèle individuel {model_path} introuvable")
        
        # Créer le modèle d'ensemble
        ensemble_model = EnsemblePlumClassifier(models)
        
        # Charger les poids de l'ensemble
        ensemble_model.load_state_dict(torch.load(ensemble_path))
        
        return ensemble_model

    def predict(self, image_path, transform=None, tta=False, tta_transforms=None):
        """
        Prédit la classe d'une image.
        
        Args:
            image_path (str): Chemin vers l'image
            transform (callable, optional): Transformations à appliquer à l'image
            tta (bool): Utiliser Test-Time Augmentation
            tta_transforms (list, optional): Liste de transformations pour TTA
            
        Returns:
            dict: Résultats de la prédiction
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné ou chargé avant la prédiction")
        
        # Chargement de l'image
        from PIL import Image
        import numpy as np
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Application des transformations
        if transform is None:
            # Utiliser les transformations de validation par défaut
            transform = A.Compose([
                A.Resize(height=320, width=320),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        # Si TTA est activé
        if tta and tta_transforms:
            # Prédictions pour chaque transformation
            tta_probs = []
            
            # Transformation et prédiction avec l'image originale
            transformed_image = transform(image=image)['image']
            transformed_image = transformed_image.unsqueeze(0).to(self.model.device)
            
            with torch.no_grad():
                logits, confidence = self.model.model(transformed_image)
                probs = F.softmax(logits, dim=1)
                tta_probs.append(probs.cpu().numpy())
            
            # Prédictions avec les transformations TTA
            for tta_transform in tta_transforms:
                # Appliquer la transformation
                transformed_image = tta_transform(image=image)['image']
                transformed_image = transformed_image.unsqueeze(0).to(self.model.device)
                
                with torch.no_grad():
                    logits, _ = self.model.model(transformed_image)
                    probs = F.softmax(logits, dim=1)
                    tta_probs.append(probs.cpu().numpy())
            
            # Moyenne des probabilités
            avg_probs = np.mean(tta_probs, axis=0)[0]
            class_idx = np.argmax(avg_probs)
            confidence_value = avg_probs[class_idx] * confidence.item()
            
            # Déterminer si l'échantillon est une prune
            est_prune = confidence_value >= self.model.model.confidence_threshold
            
            # Récupérer le nom de la classe
            class_name = self.model.model.idx_to_class[class_idx]
            
            # Résultats
            results = {
                'class_idx': int(class_idx),
                'confidence': float(confidence_value),
                'class_name': class_name,
                'est_prune': bool(est_prune),
                'probabilities': avg_probs.tolist()
            }
            
            return results
        else:
            # Transformation standard
            transformed_image = transform(image=image)['image']
            transformed_image = transformed_image.unsqueeze(0).to(self.model.device)
            
            # Prédiction
            results = self.model.model.predict_with_confidence(transformed_image)
            
            return results
    
    def analyze_confidence_distribution(self, dataloader):
        """
        Analyse la distribution des scores de confiance.
        
        Args:
            dataloader (DataLoader): DataLoader contenant les images
            
        Returns:
            dict: Résultats de l'analyse
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné ou chargé avant l'analyse")
        
        # Récupération des prédictions et des confidences
        all_preds = []
        all_labels = []
        all_confidences = []
        
        for batch in dataloader:
            images, labels, _ = batch
            
            # Passe avant
            with torch.no_grad():
                logits, confidence = self.model(images)
                
                # Prédictions
                probs = F.softmax(logits, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                
                # Ajustement de la confiance
                adjusted_confidence = confidence.squeeze() * max_probs
                
                # Stockage
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_confidences.append(adjusted_confidence.cpu().numpy())
        
        # Concaténation
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_confidences = np.concatenate(all_confidences)
        
        # Calcul des métriques
        correct = all_preds == all_labels
        
        # Visualisation de la distribution des confidences
        plt.figure(figsize=(12, 6))
        
        # Distribution pour les prédictions correctes et incorrectes
        plt.hist(all_confidences[correct], bins=20, alpha=0.5, label='Prédictions correctes')
        plt.hist(all_confidences[~correct], bins=20, alpha=0.5, label='Prédictions incorrectes')
        
        plt.xlabel('Score de confiance')
        plt.ylabel('Nombre d\'échantillons')
        plt.title('Distribution des scores de confiance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Sauvegarde de la distribution des confidences
        confidence_dist_path = os.path.join(self.logs_dir, 'confidence_distribution.png')
        plt.savefig(confidence_dist_path)
        plt.close()
        
        # Analyse de la relation entre confiance et exactitude
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (all_confidences >= confidence_bins[i]) & (all_confidences < confidence_bins[i+1])
            if np.sum(bin_mask) > 0:
                bin_acc = np.mean(correct[bin_mask])
                bin_accuracies.append(bin_acc)
                bin_counts.append(np.sum(bin_mask))
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        # Visualisation de la relation entre confiance et exactitude
        plt.figure(figsize=(12, 6))
        
        # Barplot des exactitudes par bin de confiance
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        plt.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.6)
        
        # Ligne d'exactitude idéale (confiance = exactitude)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Calibrage idéal')
        
        plt.xlabel('Score de confiance')
        plt.ylabel('Exactitude')
        plt.title('Relation entre confiance et exactitude')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Sauvegarde de la relation entre confiance et exactitude
        confidence_acc_path = os.path.join(self.logs_dir, 'confidence_accuracy.png')
        plt.savefig(confidence_acc_path)
        plt.close()
        
        # Résultats
        results = {
            'confidence_distribution_path': confidence_dist_path,
            'confidence_accuracy_path': confidence_acc_path,
            'bin_centers': bin_centers.tolist(),
            'bin_accuracies': bin_accuracies,
            'bin_counts': bin_counts,
            'mean_confidence': np.mean(all_confidences),
            'mean_accuracy': np.mean(correct)
        }
        
        return results
    
    def export_to_onnx(self, output_path=None, input_shape=(1, 3, 320, 320)):
        """
        Exporte le modèle au format ONNX.
        
        Args:
            output_path (str, optional): Chemin de sortie pour le modèle ONNX
            input_shape (tuple): Forme de l'entrée du modèle
            
        Returns:
            str: Chemin vers le modèle ONNX exporté
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné ou chargé avant l'exportation")
        
        if output_path is None:
            output_path = os.path.join(self.models_dir, "plum_classifier_final.onnx")
        
        # Déterminer si c'est un modèle d'ensemble
        model_to_export = self.model.model if isinstance(self.model, EnhancedPlumClassifierModule) else self.model
        
        # Création d'un exemple d'entrée
        dummy_input = torch.randn(*input_shape, device=self.model.device)
        
        # Exportation du modèle
        torch.onnx.export(
            model_to_export,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['logits', 'confidence'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'confidence': {0: 'batch_size'}
            }
        )
        
        print(f"Modèle exporté au format ONNX: {output_path}")
        
        return output_path