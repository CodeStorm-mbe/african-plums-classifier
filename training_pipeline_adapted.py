"""
Pipeline d'entraînement complet pour le projet de tri automatique des prunes
JCIA Hackathon 2025 - Version adaptée avec catégorie "autre"

Ce script intègre tous les modules du projet pour créer un pipeline d'entraînement complet:
1. Configuration de l'environnement Google Colab avec GPU T4
2. Téléchargement et prétraitement des données (prunes + autres catégories)
3. Entraînement et évaluation du modèle avec 7 classes (6 prunes + autre)
4. Visualisation des résultats et export du modèle
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
import idx2numpy

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
    from data_preprocessing_adapted import KaggleDatasetDownloader, DataPreprocessor
    from model_classification_adapted import PlumClassifier, ModelTrainer
except ImportError:
    logger.warning("Impossible d'importer les modules du projet directement")
    logger.info("Tentative d'importation via l'ajout du répertoire courant au path")
    # Ajout du répertoire courant au path pour permettre l'importation des modules
    sys.path.append(os.getcwd())
    try:
        from data_preprocessing_adapted import KaggleDatasetDownloader, DataPreprocessor
        from model_classification_adapted import PlumClassifier, ModelTrainer
        logger.info("Modules importés avec succès après ajout du répertoire courant au path")
    except ImportError:
        logger.error("Échec de l'importation des modules du projet")
        raise

# Définition des constantes
PLUM_CATEGORIES = ['bonne_qualite', 'non_mure', 'tachetee', 'fissuree', 'meurtrie', 'pourrie', 'autre']
NUM_CLASSES = len(PLUM_CATEGORIES)
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
CONFIDENCE_THRESHOLD = 0.7

class PlumClassificationPipeline:
    """
    Pipeline complet pour le projet de tri automatique des prunes avec catégorie "autre".
    """
    def __init__(
        self,
        base_dir='/content/plum_classifier',
        kaggle_dataset='arnaudfadja/african-plums-quality-and-defect-assessment-data',
        use_wandb=False,
        wandb_project='plum-classifier',
        wandb_entity=None
    ):
        """
        Initialise le pipeline.
        
        Args:
            base_dir (str): Répertoire de base pour le projet
            kaggle_dataset (str): ID du dataset Kaggle des prunes (format: 'username/dataset-name')
            use_wandb (bool): Si True, utilise Weights & Biases pour le suivi des expériences
            wandb_project (str): Nom du projet Weights & Biases
            wandb_entity (str, optional): Entité Weights & Biases
        """
        self.base_dir = base_dir
        self.kaggle_dataset = kaggle_dataset
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
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
            import idx2numpy
            logger.info("Toutes les dépendances sont déjà installées")
        except ImportError as e:
            logger.warning(f"Dépendance manquante: {e}")
            logger.info("Installation des dépendances manquantes...")
            
            # Installation des dépendances
            if IN_COLAB:
                # Installation des packages dans Colab en utilisant subprocess au lieu de !pip
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                                      "pytorch-lightning", "albumentations", "timm", "wandb", "kaggle", "idx2numpy"])
                
                # Redémarrage du runtime si nécessaire (dans un notebook Colab)
                import IPython
                IPython.display.display(IPython.display.HTML(
                    "<script>Jupyter.notebook.kernel.restart()</script>"
                ))
            else:
                # Installation locale
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "pytorch-lightning", "albumentations", "timm", "wandb", "kaggle", "idx2numpy"
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
                    "model": "EfficientNetB3",
                    "dataset": self.kaggle_dataset,
                    "batch_size": BATCH_SIZE,
                    "image_size": IMAGE_SIZE,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "num_classes": NUM_CLASSES
                }
            )
            logger.info(f"Weights & Biases initialisé: {self.wandb.run.name}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de Weights & Biases: {e}")
            self.use_wandb = False
    
    def download_datasets(self):
        """
        Télécharge les datasets depuis Kaggle (prunes et autres catégories).
        
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
        
        # Téléchargement du dataset Fashion MNIST pour la catégorie "autre"
        try:
            logger.info("Téléchargement du dataset Fashion MNIST pour la catégorie 'autre'...")
            fashion_downloader = KaggleDatasetDownloader('zalando-research/fashionmnist', self.data_dir)
            datasets['fashion'] = fashion_downloader.download_and_extract()
            logger.info(f"Dataset Fashion MNIST téléchargé et extrait dans {datasets['fashion']}")
        except Exception as e:
            logger.warning(f"Erreur lors du téléchargement du dataset Fashion MNIST: {e}")
            datasets['fashion'] = None
        
        # Téléchargement du dataset Intel Image Classification pour la catégorie "autre"
        try:
            logger.info("Téléchargement du dataset Intel Image Classification pour la catégorie 'autre'...")
            intel_downloader = KaggleDatasetDownloader('puneet6060/intel-image-classification', self.data_dir)
            datasets['intel'] = intel_downloader.download_and_extract()
            logger.info(f"Dataset Intel Image Classification téléchargé et extrait dans {datasets['intel']}")
        except Exception as e:
            logger.warning(f"Erreur lors du téléchargement du dataset Intel Image Classification: {e}")
            datasets['intel'] = None
        
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
        data = self.preprocessor.prepare_data()
        
        # Calcul des poids des classes
        class_weights = self.preprocessor.get_class_weights(data['class_counts'])
        
        # Visualisation d'un batch
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
                ),
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
            PlumClassifier: Modèle entraîné
        """
        if experiment_name is None:
            # Génération d'un nom d'expérience basé sur la date et l'heure
            experiment_name = f"plum_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Entraînement du modèle: {experiment_name}")
        
        # Paramètres du modèle (uniquement ceux acceptés par PlumClassifier)
        model_params = {
            'num_classes': NUM_CLASSES,
            'dropout_rate': 0.3,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'model_name': 'efficientnet_b3',
            'pretrained': True
        }

        # Paramètres d'entraînement (pour le ModelTrainer)
        trainer_config = {
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'freeze_backbone': True,
            'mixup_alpha': 0.2,
            'use_mixup': True
        }

        
        # Paramètres du trainer
        trainer_params = {
            'max_epochs': 30,
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'precision': '16-mixed' if torch.cuda.is_available() else 32,
            'log_every_n_steps': 10,
            'deterministic': True
        }
        
        # Initialisation du trainer
        self.trainer = ModelTrainer(
            model_params=model_params,
            trainer_params=trainer_params,
            models_dir=self.models_dir,
            logs_dir=self.logs_dir,
            trainer_config=trainer_config  # Ajoutez ce paramètre
        )

        
        # Entraînement du modèle
        model = self.trainer.train(
            train_dataloader=data['train_dataloader'],
            val_dataloader=data['val_dataloader'],
            class_weights=data['class_weights'],
            experiment_name=experiment_name
        )
        
        logger.info("Entraînement du modèle terminé")
        
        return model
    
    def evaluate_model(self, data):
        """
        Évalue le modèle sur l'ensemble de test.
        
        Args:
            data (dict): Dictionnaire contenant les DataLoaders et les datasets
            
        Returns:
            dict: Résultats de l'évaluation
        """
        logger.info("Évaluation du modèle sur l'ensemble de test...")
        
        # Évaluation du modèle
        results = self.trainer.evaluate(data['test_dataloader'])
        
        # Affichage des résultats
        logger.info(f"Résultats de l'évaluation: {results}")
        
        # Logging des résultats avec wandb si activé
        if self.use_wandb:
            self.wandb.log(results)
        
        return results
    
    def save_and_export_model(self):
        """
        Sauvegarde et exporte le modèle.
        
        Returns:
            tuple: (chemin du modèle PyTorch, chemin du modèle ONNX)
        """
        logger.info("Sauvegarde et export du modèle...")
        
        # Sauvegarde du modèle PyTorch
        model_path = self.trainer.save_model()
        
        # Export au format ONNX
        onnx_path = self.trainer.export_to_onnx()
        
        logger.info(f"Modèle sauvegardé à {model_path}")
        logger.info(f"Modèle exporté au format ONNX à {onnx_path}")
        
        # Logging des artefacts avec wandb si activé
        if self.use_wandb:
            self.wandb.save(model_path)
            self.wandb.save(onnx_path)
            
            # Sauvegarde des métadonnées
            metadata_path = model_path.replace('.pt', '_metadata.json')
            if os.path.exists(metadata_path):
                self.wandb.save(metadata_path)
        
        return model_path, onnx_path
    
    def visualize_predictions(self, model, data, num_samples=10):
        """
        Visualise les prédictions du modèle sur quelques exemples.
        
        Args:
            model (PlumClassifier): Modèle entraîné
            data (dict): Dictionnaire contenant les DataLoaders et les datasets
            num_samples (int): Nombre d'exemples à visualiser
        """
        logger.info(f"Visualisation des prédictions sur {num_samples} exemples...")
        
        # Passage du modèle en mode évaluation
        model.eval()
        
        # Récupération de quelques exemples du dataset de test
        test_dataset = data['test_dataset']
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        # Configuration de la figure
        fig, axes = plt.subplots(2, num_samples // 2, figsize=(20, 8))
        axes = axes.flatten()
        
        # Dénormalisation pour l'affichage
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Prédictions et visualisation
        with torch.no_grad():
            for i, idx in enumerate(indices):
                # Récupération de l'image et de l'étiquette
                if len(test_dataset[idx]) == 2:
                    image, label = test_dataset[idx]
                else:
                    # Si le dataset inclut les chemins d'images
                    image, label, _ = test_dataset[idx]
                
                # Prédiction
                image_tensor = image.unsqueeze(0).to(self.device)
                class_idx, confidence, class_name = model.predict_with_confidence(image_tensor)
                
                # Dénormalisation de l'image pour l'affichage
                img = image.cpu()
                img = img * std + mean  # Dénormalisation
                img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
                img = np.clip(img, 0, 1)
                
                # Récupération du nom de la vraie classe
                true_class_name = PLUM_CATEGORIES[label] if label < len(PLUM_CATEGORIES) else "autre"
                
                # Affichage
                axes[i].imshow(img)
                color = 'green' if class_idx == label else 'red'
                axes[i].set_title(
                    f"Vraie: {true_class_name}\nPrédiction: {class_name}\nConfiance: {confidence:.2f}",
                    color=color
                )
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'predictions_visualization.png'))
        
        # Logging avec wandb si activé
        if self.use_wandb:
            self.wandb.log({
                "predictions_visualization": self.wandb.Image(
                    os.path.join(self.results_dir, 'predictions_visualization.png')
                )
            })
        
        logger.info(f"Visualisation des prédictions sauvegardée dans {self.results_dir}")
    
    def run_pipeline(self, experiment_name=None):
        """
        Exécute le pipeline complet.
        
        Args:
            experiment_name (str, optional): Nom de l'expérience
            
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
        model = self.train_model(data, experiment_name=experiment_name)
        
        # Évaluation du modèle
        results = self.evaluate_model(data)
        
        # Visualisation des prédictions
        self.visualize_predictions(model, data)
        
        # Sauvegarde et export du modèle
        model_path, onnx_path = self.save_and_export_model()
        
        # Finalisation de wandb si activé
        if self.use_wandb:
            self.wandb.finish()
        
        # Résultats du pipeline
        pipeline_results = {
            'model_path': model_path,
            'onnx_path': onnx_path,
            'evaluation_results': results,
            'experiment_name': experiment_name or self.trainer.model.logger.name
        }
        
        logger.info("Pipeline exécuté avec succès")
        
        return pipeline_results

def main():
    """
    Fonction principale pour l'exécution du pipeline depuis la ligne de commande.
    """
    parser = argparse.ArgumentParser(description='Pipeline de classification des prunes avec catégorie "autre"')
    parser.add_argument('--base_dir', type=str, default='/content/plum_classifier',
                        help='Répertoire de base pour le projet')
    parser.add_argument('--kaggle_dataset', type=str, default='arnaudfadja/african-plums-quality-and-defect-assessment-data',
                        help='ID du dataset Kaggle des prunes (format: username/dataset-name)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Nom de l\'expérience')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Utiliser Weights & Biases pour le suivi des expériences')
    parser.add_argument('--wandb_project', type=str, default='plum-classifier',
                        help='Nom du projet Weights & Biases')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Entité Weights & Biases')
    
    args = parser.parse_args()
    
    # Initialisation et exécution du pipeline
    pipeline = PlumClassificationPipeline(
        base_dir=args.base_dir,
        kaggle_dataset=args.kaggle_dataset,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )
    
    results = pipeline.run_pipeline(
        experiment_name=args.experiment_name
    )
    
    # Affichage des résultats
    logger.info(f"Résultats du pipeline: {results}")

# Exemple d'utilisation dans un notebook Colab
if __name__ == "__main__" and IN_COLAB:
    # Configuration pour l'exécution dans Colab
    base_dir = '/content/plum_classifier'
    kaggle_dataset = 'arnaudfadja/african-plums-quality-and-defect-assessment-data'
    
    # Initialisation du pipeline
    pipeline = PlumClassificationPipeline(
        base_dir=base_dir,
        kaggle_dataset=kaggle_dataset,
        use_wandb=False  # Désactivé par défaut dans Colab
    )
    
    # Exécution du pipeline
    results = pipeline.run_pipeline()
    
    # Affichage des résultats
    print(f"Modèle sauvegardé à: {results['model_path']}")
    print(f"Modèle ONNX exporté à: {results['onnx_path']}")
    print(f"Résultats de l'évaluation: {results['evaluation_results']}")

# Exécution depuis la ligne de commande
if __name__ == "__main__" and not IN_COLAB:
    main()
