"""
Pipeline d'entraînement complet pour le projet de tri automatique des prunes
JCIA Hackathon 2025 - Version optimisée avec techniques avancées

Ce script intègre tous les modules du projet pour créer un pipeline d'entraînement complet:
1. Configuration de l'environnement avec GPU
2. Téléchargement et prétraitement des données avec augmentation avancée
3. Entraînement et évaluation du modèle avec 6 classes de prunes
4. Utilisation de techniques avancées (TTA, validation croisée, ensembling)
5. Visualisation des résultats et export du modèle
6. Utilisation des intervalles de confiance pour déterminer si un échantillon est une prune ou non
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import subprocess
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Vérification de l'environnement Colab
try:
    import google.colab
    IN_COLAB = True
    logger.info("Exécution dans l'environnement Google Colab")
except:
    IN_COLAB = False
    logger.info("Exécution dans un environnement local")

# Importation des modules du projet
# Note: Ces imports seront ajustés en fonction de l'environnement d'exécution
try:
    from data_preprocessing_enhanced import KaggleDatasetDownloader, DataPreprocessor, CutMix
    from model_classification_enhanced import EnhancedPlumClassifier, ModelTrainer
except ImportError:
    logger.warning("Impossible d'importer les modules du projet directement")
    logger.info("Tentative d'importation via l'ajout du répertoire courant au path")
    # Ajout du répertoire courant au path pour permettre l'importation des modules
    sys.path.append(os.getcwd())
    try:
        from data_preprocessing_enhanced import KaggleDatasetDownloader, DataPreprocessor, CutMix
        from model_classification_enhanced import EnhancedPlumClassifier, ModelTrainer
        logger.info("Modules importés avec succès après ajout du répertoire courant au path")
    except ImportError:
        logger.error("Échec de l'importation des modules du projet")
        raise

# Définition des constantes
PLUM_CATEGORIES = ['bonne_qualite', 'non_mure', 'tachetee', 'fissuree', 'meurtrie', 'pourrie']
NUM_CLASSES = len(PLUM_CATEGORIES)
IMAGE_SIZE = 320  # Augmenté pour capturer plus de détails
BATCH_SIZE = 16   # Réduit pour éviter les problèmes de mémoire avec des images plus grandes
NUM_WORKERS = 2   # Réduit pour éviter les avertissements
CONFIDENCE_THRESHOLD = 0.7

class EnhancedPlumClassificationPipeline:
    """
    Pipeline optimisé pour le projet de tri automatique des prunes.
    Utilise des techniques avancées pour améliorer les performances.
    """
    def __init__(
        self,
        base_dir='/content/plum_classifier',
        kaggle_dataset='arnaudfadja/african-plums-quality-and-defect-assessment-data',
        use_wandb=False,
        wandb_project='plum-classifier',
        wandb_entity=None,
        use_cross_validation=True,
        n_folds=5,
        use_tta=True,
        use_ensemble=True,
        n_models=3
    ):
        """
        Initialise le pipeline.
        
        Args:
            base_dir (str): Répertoire de base pour le projet
            kaggle_dataset (str): ID du dataset Kaggle des prunes (format: 'username/dataset-name')
            use_wandb (bool): Si True, utilise Weights & Biases pour le suivi des expériences
            wandb_project (str): Nom du projet Weights & Biases
            wandb_entity (str, optional): Entité Weights & Biases
            use_cross_validation (bool): Si True, utilise la validation croisée
            n_folds (int): Nombre de plis pour la validation croisée
            use_tta (bool): Si True, utilise Test-Time Augmentation
            use_ensemble (bool): Si True, utilise un ensemble de modèles
            n_models (int): Nombre de modèles dans l'ensemble
        """
        self.base_dir = base_dir
        self.kaggle_dataset = kaggle_dataset
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        # Options avancées
        self.use_cross_validation = use_cross_validation
        self.n_folds = n_folds
        self.use_tta = use_tta
        self.use_ensemble = use_ensemble
        self.n_models = n_models
        
        # Création des répertoires
        self.data_dir = os.path.join(base_dir, 'data')
        self.models_dir = os.path.join(base_dir, 'models')
        self.logs_dir = os.path.join(base_dir, 'logs')
        self.results_dir = os.path.join(base_dir, 'results')
        
        for directory in [self.data_dir, self.models_dir, self.logs_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialisation des composants
        self.downloader = None
        self.preprocessor = None
        self.trainer = None
        self.ensemble_models = []
        
        # Vérification de la disponibilité du GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Utilisation du dispositif: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU disponible: {torch.cuda.get_device_name(0)}")
            logger.info(f"Mémoire GPU totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Configuration de Weights & Biases si activé
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                logger.info("Weights & Biases importé avec succès")
            except ImportError:
                logger.warning("Impossible d'importer Weights & Biases, désactivation du suivi")
                self.use_wandb = False
    
    def setup_environment(self):
        """
        Configure l'environnement d'exécution.
        """
        logger.info("Configuration de l'environnement...")
        
        # Vérification et installation des dépendances
        try:
            import torch
            import pytorch_lightning
            import albumentations
            import timm
            logger.info("Toutes les dépendances sont déjà installées")
        except ImportError as e:
            logger.warning(f"Dépendance manquante: {e}")
            logger.info("Installation des dépendances manquantes...")
            
            # Installation des dépendances
            if IN_COLAB:
                # Installation des packages dans Colab en utilisant subprocess au lieu de !pip
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                                      "pytorch-lightning", "albumentations", "timm", "wandb", "kaggle", "onnx", "onnxruntime"])
                
                # Redémarrage du runtime si nécessaire (dans un notebook Colab)
                import IPython
                IPython.display.display(IPython.display.HTML(
                    "<script>Jupyter.notebook.kernel.restart()</script>"
                ))
            else:
                # Installation locale
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "pytorch-lightning", "albumentations", "timm", "wandb", "kaggle", "onnx", "onnxruntime"
                ])
        
        # Configuration de Kaggle
        self._setup_kaggle()
        
        # Initialisation de Weights & Biases si activé
        if self.use_wandb:
            self._setup_wandb()
        
        logger.info("Environnement configuré avec succès")
    
    def _setup_kaggle(self):
        """
        Configure l'accès à Kaggle.
        """
        # Vérification de la présence du fichier kaggle.json
        kaggle_dir = os.path.expanduser('~/.kaggle')
        kaggle_config = os.path.join(kaggle_dir, 'kaggle.json')
        
        if not os.path.exists(kaggle_config):
            # Si le fichier n'existe pas, demander à l'utilisateur de le télécharger
            if IN_COLAB:
                from google.colab import files
                import shutil
                
                logger.info("Le fichier kaggle.json n'a pas été trouvé.")
                logger.info("Veuillez télécharger votre fichier kaggle.json depuis votre compte Kaggle")
                logger.info("et le téléverser ici.")
                
                uploaded = files.upload()
                
                if 'kaggle.json' in uploaded:
                    # Création du répertoire .kaggle s'il n'existe pas
                    os.makedirs(kaggle_dir, exist_ok=True)
                    
                    # Copie du fichier téléversé
                    shutil.copy('kaggle.json', kaggle_config)
                    
                    # Définition des permissions
                    os.chmod(kaggle_config, 0o600)
                    
                    logger.info("Fichier kaggle.json configuré avec succès")
                else:
                    logger.error("Le fichier kaggle.json n'a pas été téléversé")
                    raise FileNotFoundError("Fichier kaggle.json requis pour accéder au dataset Kaggle")
            else:
                logger.error("Le fichier kaggle.json n'a pas été trouvé")
                logger.info("Veuillez télécharger votre fichier kaggle.json depuis votre compte Kaggle")
                logger.info(f"et le placer dans {kaggle_dir}")
                raise FileNotFoundError("Fichier kaggle.json requis pour accéder au dataset Kaggle")
        else:
            # Vérification des permissions
            os.chmod(kaggle_config, 0o600)
            logger.info("Fichier kaggle.json trouvé et configuré")
    
    def _setup_wandb(self):
        """
        Configure Weights & Biases pour le suivi des expériences.
        """
        if not self.use_wandb:
            return
        
        try:
            # Initialisation de wandb
            self.wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                config={
                    "model": "EfficientNetB4",
                    "dataset": self.kaggle_dataset,
                    "batch_size": BATCH_SIZE,
                    "image_size": IMAGE_SIZE,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "num_classes": NUM_CLASSES,
                    "use_cross_validation": self.use_cross_validation,
                    "n_folds": self.n_folds,
                    "use_tta": self.use_tta,
                    "use_ensemble": self.use_ensemble,
                    "n_models": self.n_models
                }
            )
            logger.info(f"Weights & Biases initialisé: {self.wandb.run.name}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de Weights & Biases: {e}")
            self.use_wandb = False
    
    def download_datasets(self):
        """
        Télécharge le dataset des prunes depuis Kaggle.
        
        Returns:
            dict: Chemins vers les répertoires contenant les données
        """
        logger.info("Téléchargement des datasets...")
        
        datasets = {}
        
        # Téléchargement du dataset principal des prunes
        logger.info(f"Téléchargement du dataset des prunes {self.kaggle_dataset} depuis Kaggle...")
        self.downloader = KaggleDatasetDownloader(self.kaggle_dataset, self.data_dir)
        datasets['plums'] = self.downloader.download_and_extract()
        logger.info(f"Dataset des prunes téléchargé et extrait dans {datasets['plums']}")
        
        return datasets
    
    def preprocess_data(self, datasets):
        """
        Prétraite les données et crée les DataLoaders.
        
        Args:
            datasets (dict): Chemins vers les répertoires contenant les données
            
        Returns:
            dict: Dictionnaire contenant les DataLoaders et les datasets
        """
        logger.info("Prétraitement des données...")
        
        # Initialisation du préprocesseur
        self.preprocessor = DataPreprocessor(
            data_dir=self.data_dir,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
        
        # Analyse du dataset
        stats = self.preprocessor.analyze_dataset(save_dir=self.results_dir)
        
        # Préparation des données
        if self.use_cross_validation:
            # Création des plis pour la validation croisée
            folds = self.preprocessor.create_cross_validation_folds(n_splits=self.n_folds)
            data = {
                'folds': folds,
                'class_counts': stats['class_counts']
            }
        else:
            # Préparation standard des données
            data = self.preprocessor.prepare_data(use_weighted_sampler=True)
        
        # Calcul des poids des classes
        class_weights = self.preprocessor.get_class_weights(stats['class_counts'])
        
        # Visualisation d'un batch
        if not self.use_cross_validation:
            self.preprocessor.visualize_batch(
                data['train_dataloader'],
                save_path=os.path.join(self.results_dir, 'batch_visualization.png')
            )
        
        # Logging des statistiques avec wandb si activé
        if self.use_wandb:
            self.wandb.log({
                "num_images": stats['num_images'],
                "num_classes": stats['num_classes'],
                "avg_width": stats['avg_width'],
                "avg_height": stats['avg_height'],
                "avg_aspect_ratio": stats['avg_aspect_ratio'],
                "class_distribution": self.wandb.Image(
                    os.path.join(self.results_dir, 'class_distribution.png')
                ),
                "dimension_distribution": self.wandb.Image(
                    os.path.join(self.results_dir, 'dimension_distribution.png')
                )
            })
            
            if not self.use_cross_validation:
                self.wandb.log({
                    "batch_visualization": self.wandb.Image(
                        os.path.join(self.results_dir, 'batch_visualization.png')
                    )
                })
        
        logger.info("Prétraitement des données terminé")
        
        # Ajout des poids des classes aux données
        data['class_weights'] = class_weights
        
        return data
    
    def train_model(self, data, experiment_name=None):
        """
        Entraîne le modèle de classification.
        
        Args:
            data (dict): Dictionnaire contenant les DataLoaders et les datasets
            experiment_name (str, optional): Nom de l'expérience
            
        Returns:
            dict: Résultats de l'entraînement
        """
        if experiment_name is None:
            # Génération d'un nom d'expérience basé sur la date et l'heure
            experiment_name = f"plum_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Entraînement du modèle: {experiment_name}")
        
        # Paramètres du modèle
        model_params = {
            'num_classes': NUM_CLASSES,
            'dropout_rate': 0.4,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'model_name': 'efficientnet_b4',
            'pretrained': True
        }

        # Paramètres d'entraînement
        trainer_config = {
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'use_mixup_cutmix': True,
            'mixup_alpha': 1.0,
            'cutmix_alpha': 1.0,
            'label_smoothing': 0.1,
            'use_focal_loss': True,
            'gamma': 2.0,
            'use_one_cycle': True,
            'use_amp': True
        }
        
        # Paramètres du trainer
        trainer_params = {
            'max_epochs': 30,
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'precision': '16-mixed' if torch.cuda.is_available() else 32,
            'log_every_n_steps': 10,
            'deterministic': False  # Désactivé pour permettre les optimisations CUDA
        }
        
        # Initialisation du trainer
        self.trainer = ModelTrainer(
            model_params=model_params,
            trainer_params=trainer_params,
            models_dir=self.models_dir,
            logs_dir=self.logs_dir,
            trainer_config=trainer_config
        )
        
        # Entraînement du modèle
        if self.use_cross_validation:
            # Entraînement avec validation croisée
            results = self.trainer.train_with_cross_validation(
                folds=data['folds'],
                experiment_name=experiment_name
            )
        elif self.use_ensemble:
            # Entraînement d'un ensemble de modèles
            ensemble_results = []
            
            for i in range(self.n_models):
                model_experiment_name = f"{experiment_name}_model{i+1}"
                logger.info(f"Entraînement du modèle {i+1}/{self.n_models}...")
                
                # Entraînement du modèle
                result = self.trainer.train(
                    train_dataloader=data['train_dataloader'],
                    val_dataloader=data['val_dataloader'],
                    class_weights=data['class_weights'],
                    experiment_name=model_experiment_name
                )
                
                ensemble_results.append(result)
                
                # Sauvegarder le modèle dans l'ensemble
                self.ensemble_models.append(self.trainer.model.model)
            
            results = {
                'ensemble_results': ensemble_results,
                'experiment_name': experiment_name
            }
        else:
            # Entraînement standard
            results = self.trainer.train(
                train_dataloader=data['train_dataloader'],
                val_dataloader=data['val_dataloader'],
                class_weights=data['class_weights'],
                experiment_name=experiment_name
            )
        
        # Évaluation du modèle
        logger.info("Évaluation du modèle sur l'ensemble de test...")
        
        if self.use_cross_validation:
            # Évaluation avec le meilleur modèle de validation croisée
            # Utiliser le premier pli pour l'évaluation
            test_dataloader = data['folds'][0]['val_dataloader']
        else:
            test_dataloader = data['test_dataloader']
        
        if self.use_tta:
            # Évaluation avec Test-Time Augmentation
            eval_results = self.trainer.evaluate_with_tta(
                test_dataloader=test_dataloader,
                tta_transforms=self.preprocessor.tta_transforms
            )
            eval_type = "TTA"
        else:
            # Évaluation standard
            eval_results = self.trainer.evaluate(test_dataloader)
            eval_type = "standard"
        
        # Affichage des résultats
        logger.info(f"Résultats de l'évaluation ({eval_type}):")
        logger.info(f"- Exactitude: {eval_results['accuracy']:.4f}")
        logger.info(f"- Précision: {eval_results['precision']:.4f}")
        logger.info(f"- Rappel: {eval_results['recall']:.4f}")
        logger.info(f"- F1-score: {eval_results['f1']:.4f}")
        logger.info(f"- Matrice de confusion sauvegardée dans {eval_results['confusion_matrix_path']}")
        
        # Analyse des intervalles de confiance
        logger.info("Analyse des intervalles de confiance...")
        confidence_results = self.trainer.analyze_confidence_distribution(test_dataloader)
        
        logger.info(f"Résultats de l'analyse des intervalles de confiance:")
        logger.info(f"- Confiance moyenne: {confidence_results['mean_confidence']:.4f}")
        logger.info(f"- Exactitude moyenne: {confidence_results['mean_accuracy']:.4f}")
        logger.info(f"- Distribution des confidences sauvegardée dans {confidence_results['confidence_distribution_path']}")
        logger.info(f"- Relation confiance-exactitude sauvegardée dans {confidence_results['confidence_accuracy_path']}")
        
        # Logging des résultats avec wandb si activé
        if self.use_wandb:
            self.wandb.log({
                "test_accuracy": eval_results['accuracy'],
                "test_precision": eval_results['precision'],
                "test_recall": eval_results['recall'],
                "test_f1": eval_results['f1'],
                "confusion_matrix": self.wandb.Image(eval_results['confusion_matrix_path']),
                "confidence_distribution": self.wandb.Image(confidence_results['confidence_distribution_path']),
                "confidence_accuracy": self.wandb.Image(confidence_results['confidence_accuracy_path']),
                "mean_confidence": confidence_results['mean_confidence'],
                "mean_accuracy": confidence_results['mean_accuracy']
            })
            
            # Logging des métriques de confiance
            for i, (center, acc, count) in enumerate(zip(
                confidence_results['bin_centers'],
                confidence_results['bin_accuracies'],
                confidence_results['bin_counts']
            )):
                self.wandb.log({
                    f"confidence_bin_{i}_center": center,
                    f"confidence_bin_{i}_accuracy": acc,
                    f"confidence_bin_{i}_count": count
                })
        
        # Combiner les résultats
        combined_results = {
            'training_results': results,
            'evaluation_results': eval_results,
            'confidence_results': confidence_results
        }
        
        # SAUVEGARDE DES MODELS CKPT DANS GOOGLE DRIVE
        if IN_COLAB:
            from google.colab import drive
            import shutil

            drive.mount('/content/drive', force_remount=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"/content/drive/MyDrive/colab_backups/plum_ckpts/{experiment_name}_{timestamp}"
            os.makedirs(backup_dir, exist_ok=True)

            # Ne copier que les 3 meilleurs modèles (ckpt)
            ckpt_files = sorted(
                [f for f in os.listdir(self.models_dir) if f.endswith(".ckpt")],
                key=lambda x: os.path.getmtime(os.path.join(self.models_dir, x)),
                reverse=True
            )[:3]

            for f in ckpt_files:
                src = os.path.join(self.models_dir, f)
                dst = os.path.join(backup_dir, f)
                shutil.copy(src, dst)
                logger.info(f"✅ Modèle sauvegardé dans Drive : {dst}")

        return combined_results
    
    def export_model(self, model_path=None, format='onnx'):
        """
        Exporte le modèle dans un format spécifique.
        
        Args:
            model_path (str, optional): Chemin où sauvegarder le modèle exporté
            format (str): Format d'exportation ('onnx', 'torchscript')
            
        Returns:
            str: Chemin vers le modèle exporté
        """
        if self.trainer is None or self.trainer.model is None:
            raise ValueError("Le modèle doit être entraîné avant l'exportation")
        
        logger.info(f"Exportation du modèle au format {format}...")
        
        # Chemin par défaut
        if model_path is None:
            model_path = os.path.join(self.models_dir, f"plum_classifier_final.{format}")
        
        # Exportation au format ONNX
        if format.lower() == 'onnx':
            try:
                import onnx
                import onnxruntime
            except ImportError:
                logger.warning("Les packages onnx et onnxruntime ne sont pas installés")
                logger.info("Installation des packages...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime"])
                import onnx
                import onnxruntime
            
            # Exportation du modèle
            onnx_path = self.trainer.export_to_onnx(
                output_path=model_path,
                input_shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            )
            
            # Vérification du modèle exporté
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"Modèle exporté avec succès au format ONNX: {onnx_path}")
            
            # Sauvegarde des métadonnées
            metadata_path = os.path.splitext(onnx_path)[0] + '_metadata.json'
            metadata = {
                'num_classes': NUM_CLASSES,
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'idx_to_class': self.trainer.model.model.idx_to_class,
                'image_size': IMAGE_SIZE,
                'format': 'onnx'
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Métadonnées sauvegardées dans {metadata_path}")
            
            return onnx_path
            
        # Exportation au format TorchScript
        elif format.lower() == 'torchscript':
            # Création d'un exemple d'entrée
            dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=self.trainer.model.device)
            
            # Traçage du modèle
            traced_model = torch.jit.trace(self.trainer.model.model, dummy_input)
            
            # Sauvegarde du modèle
            traced_model.save(model_path)
            
            logger.info(f"Modèle exporté avec succès au format TorchScript: {model_path}")
            
            # Sauvegarde des métadonnées
            metadata_path = os.path.splitext(model_path)[0] + '_metadata.json'
            metadata = {
                'num_classes': NUM_CLASSES,
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'idx_to_class': self.trainer.model.model.idx_to_class,
                'image_size': IMAGE_SIZE,
                'format': 'torchscript'
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Métadonnées sauvegardées dans {metadata_path}")
            
            return model_path
            
        else:
            raise ValueError(f"Format d'exportation non supporté: {format}")
    
    def run_pipeline(self):
        """
        Exécute le pipeline complet.
        
        Returns:
            dict: Résultats du pipeline
        """
        # Configuration de l'environnement
        self.setup_environment()
        
        # Téléchargement des datasets
        datasets = self.download_datasets()
        
        # Prétraitement des données
        data = self.preprocess_data(datasets)
        
        # Entraînement du modèle
        training_results = self.train_model(data)
        
        # Exportation du modèle
        onnx_path = self.export_model(format='onnx')
        
        # Fermeture de wandb si activé
        if self.use_wandb:
            self.wandb.finish()
        
        # Résultats finaux
        results = {
            'training_results': training_results,
            'onnx_path': onnx_path,
            'model_path': training_results['training_results'].get('model_path', None)
        }
        
        return results
    
    def demo_prediction(self, image_path):
        """
        Démontre la prédiction sur une image.
        
        Args:
            image_path (str): Chemin vers l'image
            
        Returns:
            dict: Résultats de la prédiction
        """
        if self.trainer is None or self.trainer.model is None:
            raise ValueError("Le modèle doit être entraîné avant la prédiction")
        
        # Prédiction
        if self.use_tta:
            # Prédiction avec Test-Time Augmentation
            results = self.trainer.predict(
                image_path=image_path,
                tta=True,
                tta_transforms=self.preprocessor.tta_transforms
            )
        else:
            # Prédiction standard
            results = self.trainer.predict(image_path=image_path)
        
        # Affichage des résultats
        logger.info(f"Résultats de la prédiction pour {image_path}:")
        logger.info(f"- Classe prédite: {results['class_name']} (indice: {results['class_idx']})")
        logger.info(f"- Score de confiance: {results['confidence']:.4f}")
        logger.info(f"- Est une prune: {'Oui' if results['est_prune'] else 'Non'}")
        
        # Affichage de l'image avec la prédiction
        from PIL import Image
        import matplotlib.pyplot as plt
        
        # Chargement de l'image
        img = Image.open(image_path).convert('RGB')
        
        # Affichage
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        
        # Titre avec la prédiction
        if results['est_prune']:
            plt.title(f"Prédiction: {results['class_name']} (confiance: {results['confidence']:.2f})")
        else:
            plt.title(f"Prédiction: Non prune (confiance trop faible: {results['confidence']:.2f})")
        
        plt.axis('off')
        
        # Sauvegarde de l'image avec la prédiction
        save_path = os.path.join(self.results_dir, 'prediction_demo.png')
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Visualisation de la prédiction sauvegardée dans {save_path}")
        
        return results

def main():
    """
    Fonction principale.
    """
    # Parsing des arguments
    parser = argparse.ArgumentParser(description='Pipeline optimisé de classification des prunes')
    parser.add_argument('--base_dir', type=str, default='/content/plum_classifier',
                        help='Répertoire de base pour le projet')
    parser.add_argument('--kaggle_dataset', type=str, 
                        default='arnaudfadja/african-plums-quality-and-defect-assessment-data',
                        help='ID du dataset Kaggle des prunes')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Utiliser Weights & Biases pour le suivi des expériences')
    parser.add_argument('--wandb_project', type=str, default='plum-classifier',
                        help='Nom du projet Weights & Biases')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Entité Weights & Biases')
    parser.add_argument('--use_cross_validation', action='store_true',
                        help='Utiliser la validation croisée')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Nombre de plis pour la validation croisée')
    parser.add_argument('--use_tta', action='store_true',
                        help='Utiliser Test-Time Augmentation')
    parser.add_argument('--use_ensemble', action='store_true',
                        help='Utiliser un ensemble de modèles')
    parser.add_argument('--n_models', type=int, default=3,
                        help='Nombre de modèles dans l\'ensemble')
    
    args = parser.parse_args()
    
    # Initialisation du pipeline
    pipeline = EnhancedPlumClassificationPipeline(
        base_dir=args.base_dir,
        kaggle_dataset=args.kaggle_dataset,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        use_cross_validation=args.use_cross_validation,
        n_folds=args.n_folds,
        use_tta=args.use_tta,
        use_ensemble=args.use_ensemble,
        n_models=args.n_models
    )
    
    # Exécution du pipeline
    results = pipeline.run_pipeline()
    
    return results

if __name__ == '__main__':
    results = main()
    
    # Afficher les résultats
    print(f"Modèle sauvegardé à: {results['model_path']}")
    print(f"Modèle ONNX exporté à: {results['onnx_path']}")
