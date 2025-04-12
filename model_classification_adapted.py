import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics  # Ajoutez cette ligne
import pytorch_lightning as pl
import timm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

class PlumClassifier(nn.Module):
    """
    Modèle de classification des prunes basé sur EfficientNet.
    Inclut une catégorie supplémentaire "autre" et un mécanisme de confiance.
    """
    def __init__(self, num_classes=7, model_name='efficientnet_b3', pretrained=True, dropout_rate=0.3, confidence_threshold=0.7):
        """
        Initialise le modèle de classification.
        
        Args:
            num_classes (int): Nombre de classes (6 catégories de prunes + 1 catégorie "autre")
            model_name (str): Nom du modèle de base à utiliser
            pretrained (bool): Si True, utilise des poids pré-entraînés
            dropout_rate (float): Taux de dropout
            confidence_threshold (float): Seuil de confiance pour les prédictions
        """
        super(PlumClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        
        # Chargement du modèle de base
        self.base_model = timm.create_model(model_name, pretrained=pretrained)
        
        # Récupération du nombre de features de la dernière couche
        if hasattr(self.base_model, 'classifier'):
            in_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        elif hasattr(self.base_model, 'fc'):
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError(f"Architecture non supportée: {model_name}")
        
        # Couches de classification
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Couche de confiance (pour estimer la fiabilité de la prédiction)
        self.confidence_fc = nn.Linear(512, 1)
        
        # Mapping des indices aux noms de classes
        self.idx_to_class = {
            0: 'bonne_qualite',
            1: 'non_mure',
            2: 'tachetee',
            3: 'fissuree',
            4: 'meurtrie',
            5: 'pourrie',
            6: 'autre'  # Nouvelle catégorie "autre"
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
        
        # Couches de classification
        x = self.dropout(features)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Logits pour la classification
        logits = self.fc2(x)
        
        # Score de confiance
        confidence = torch.sigmoid(self.confidence_fc(x))
        
        return logits, confidence
    
    def predict_with_confidence(self, x):
        """
        Prédit la classe avec un score de confiance.
        
        Args:
            x (torch.Tensor): Batch d'images
            
        Returns:
            tuple: (classe prédite, score de confiance, nom de la classe)
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
            
            # Si la confiance est inférieure au seuil, prédire "autre"
            low_confidence_mask = adjusted_confidence < self.confidence_threshold
            predicted[low_confidence_mask] = self.num_classes - 1  # Indice de la classe "autre"
            
            # Récupérer le nom de la classe
            class_idx = predicted.item()
            class_name = self.idx_to_class[class_idx]
            
            return class_idx, adjusted_confidence.item(), class_name

class MixupTransform:
    """
    Implémentation de la technique Mixup pour l'augmentation de données.
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        """
        Applique Mixup à un batch.
        
        Args:
            batch (tuple): (images, labels)
            
        Returns:
            tuple: (images mixées, labels mixés)
        """
        images, labels = batch
        
        # Générer un lambda à partir d'une distribution Beta
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mélanger les indices
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        # Mixer les images et les labels
        mixed_images = lam * images + (1 - lam) * images[index, :]
        mixed_labels = (lam * F.one_hot(labels, num_classes=7) + 
                       (1 - lam) * F.one_hot(labels[index], num_classes=7))
        
        return mixed_images, mixed_labels

class PlumClassifierModule(pl.LightningModule):
    """
    Module PyTorch Lightning pour l'entraînement du modèle de classification des prunes.
    """
    def __init__(self, model_params, class_weights=None, use_mixup=True, mixup_alpha=0.2, learning_rate=1e-4, weight_decay=1e-5):
        """
        Initialise le module.
        """
        super(PlumClassifierModule, self).__init__()
        
        # Sauvegarde des hyperparamètres
        self.save_hyperparameters()
        
        # Initialisation du modèle
        self.model = PlumClassifier(**model_params)
        
        # Poids des classes - Assurez-vous qu'ils sont sur le bon périphérique
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)  # Ceci déplacera automatiquement les poids sur le bon périphérique
        else:
            self.class_weights = None
        
        # Mixup
        self.use_mixup = use_mixup
        self.mixup_transform = MixupTransform(alpha=mixup_alpha) if use_mixup else None
        
        # Taux d'apprentissage et régularisation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Métriques
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=model_params['num_classes'])
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=model_params['num_classes'])
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=model_params['num_classes'])
        
        # Température pour le calibrage des probabilités
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    
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
        
        # Appliquer Mixup si activé
        if self.use_mixup and self.current_epoch < self.trainer.max_epochs // 2:
            images, mixed_labels = self.mixup_transform((images, labels))
            
            # Passe avant
            logits, confidence = self(images)
            
            # Perte avec labels mixés
            loss = F.cross_entropy(logits, mixed_labels, weight=self.class_weights)
        else:
            # Passe avant
            logits, confidence = self(images)
            
            # Perte
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        
        # Calcul de l'exactitude
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
        
        # Passe avant
        logits, confidence = self(images)
        
        # Perte
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        
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
        
        # Passe avant
        logits, confidence = self(images)
        
        # Appliquer la température pour le calibrage
        scaled_logits = logits / self.temperature
        
        # Perte
        loss = F.cross_entropy(scaled_logits, labels, weight=self.class_weights)
        
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
            lr=self.learning_rate,  # Au lieu de self.hparams.model_params['learning_rate']
            weight_decay=self.weight_decay  # Au lieu de self.hparams.model_params['weight_decay']
        )

        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_validation_epoch_end(self):
        """
        Actions à la fin d'une époque de validation.
        """
        # Calibrage de la température
        if self.current_epoch % 5 == 0:
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
            PlumClassifier: Modèle entraîné
        """
        # Nom de l'expérience
        if experiment_name is None:
            experiment_name = f"plum_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialisation du module
        self.model = PlumClassifierModule(
            model_params=self.model_params,
            class_weights=class_weights,
            use_mixup=self.trainer_config.get('use_mixup', True),
            mixup_alpha=self.trainer_config.get('mixup_alpha', 0.2),
            learning_rate=self.trainer_config.get('learning_rate', 1e-4),
            weight_decay=self.trainer_config.get('weight_decay', 1e-5)
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
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
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
            self.model = PlumClassifierModule.load_from_checkpoint(best_model_path)
        
        # Sauvegarde du modèle final
        self.save_model()
        
        return self.model.model
    
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
        all_logits = []
        all_confidences = []
        
        for batch in test_dataloader:
            images, labels, _ = batch
            
            # Passe avant
            with torch.no_grad():
                logits, confidence = self.model(images)
                
                # Appliquer la température
                scaled_logits = logits / self.model.temperature
                
                # Prédictions
                preds = torch.argmax(scaled_logits, dim=1)
                
                # Stocker les résultats
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_logits.append(scaled_logits.cpu().numpy())
                all_confidences.append(confidence.squeeze().cpu().numpy())
        
        # Concaténation des résultats
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_logits = np.concatenate(all_logits)
        all_confidences = np.concatenate(all_confidences)
        
        # Calcul des métriques
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        # Matrice de confusion
        cm = confusion_matrix(all_labels, all_preds)
        
        # Rapport de classification
        class_report = classification_report(all_labels, all_preds, target_names=list(self.model.model.idx_to_class.values()), output_dict=True)
        
        # Visualisation de la matrice de confusion
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.model.model.idx_to_class.values()),
                   yticklabels=list(self.model.model.idx_to_class.values()))
        plt.xlabel('Prédiction')
        plt.ylabel('Vérité terrain')
        plt.title('Matrice de confusion')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(self.logs_dir), 'results', 'confusion_matrix.png'))
        plt.close()
        
        # Visualisation des métriques par classe
        metrics_df = pd.DataFrame(class_report).transpose()
        metrics_df = metrics_df.drop('accuracy', errors='ignore')
        
        plt.figure(figsize=(12, 6))
        metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
        plt.title('Métriques par classe')
        plt.ylabel('Score')
        plt.xlabel('Classe')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(self.logs_dir), 'results', 'metrics_by_class.png'))
        plt.close()
        
        # Résultats
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'class_report': class_report,
            'temperature': self.model.temperature.item()
        }
        
        # Sauvegarde des résultats
        with open(os.path.join(os.path.dirname(self.logs_dir), 'results', 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        return results
    
    def save_model(self):
        """
        Sauvegarde le modèle.
        
        Returns:
            str: Chemin du modèle sauvegardé
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant la sauvegarde")
        
        # Chemin du modèle
        model_path = os.path.join(self.models_dir, 'plum_classifier_final.pt')
        
        # Sauvegarde du modèle PyTorch
        torch.save(self.model.model.state_dict(), model_path)
        
        # Sauvegarde des métadonnées
        metadata = {
            'num_classes': self.model.model.num_classes,
            'confidence_threshold': self.model.model.confidence_threshold,
            'idx_to_class': self.model.model.idx_to_class,
            'temperature': self.model.temperature.item(),
            'model_name': self.model_params.get('model_name', 'efficientnet_b3'),
            'image_size': 224,
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(self.models_dir, 'plum_classifier_final_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Modèle sauvegardé à {model_path}")
        print(f"Métadonnées sauvegardées à {metadata_path}")
        
        return model_path
    
    def export_to_onnx(self):
        """
        Exporte le modèle au format ONNX.
        
        Returns:
            str: Chemin du modèle ONNX
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant l'export")
        
        # Chemin du modèle ONNX
        onnx_path = os.path.join(self.models_dir, 'plum_classifier.onnx')
        
        # Exemple d'entrée
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Export au format ONNX
        torch.onnx.export(
            self.model.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['logits', 'confidence'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'confidence': {0: 'batch_size'}
            }
        )
        
        print(f"Modèle exporté au format ONNX à {onnx_path}")
        
        return onnx_path
