# Modèle de Classification de Prunes

## Description

Ce projet implémente un pipeline complet de classification d'images de prunes africaines selon leur qualité et leurs défauts. Développé pour le JCIA Hackathon 2025, ce modèle utilise des techniques avancées de deep learning pour classer les prunes en 6 catégories différentes : bonne qualité, non mûre, tachetée, fissurée, meurtrie et pourrie.

Le système est conçu pour être robuste, précis et facilement adaptable à différents environnements d'exécution, y compris Google Colab et les environnements locaux.

## Caractéristiques principales

- **Classification multi-classes** : Identification précise de 6 catégories de prunes
- **Techniques avancées** :
  - Validation croisée (K-fold)
  - Test-Time Augmentation (TTA)
  - Ensemble de modèles
  - Calibration de confiance
- **Augmentation de données** : Techniques avancées incluant CutMix, GridMask, et diverses transformations
- **Architecture optimisée** : Basée sur EfficientNet avec des blocs d'attention et des connexions résiduelles
- **Mécanisme de confiance** : Détermination si un échantillon est une prune ou non basée sur des seuils de confiance
- **Compatibilité** : Fonctionne dans Google Colab et en environnement local
- **Exportation de modèle** : Support pour les formats ONNX et TorchScript
- **Visualisations** : Matrices de confusion, distributions de confiance, et autres métriques

## Structure du projet

Le projet est organisé en trois modules principaux :

1. **Pipeline d'entraînement** (`training_pipeline_enhanced.py`) :
   - Orchestration du processus complet
   - Configuration de l'environnement
   - Téléchargement et prétraitement des données
   - Entraînement et évaluation du modèle
   - Visualisation et export des résultats

2. **Classification de modèle** (`model_classification_enhanced.py`) :
   - Architecture du modèle avec EfficientNet et blocs d'attention
   - Fonctions de perte avancées (Focal Loss, Label Smoothing)
   - Techniques d'augmentation (Mixup, CutMix)
   - Module d'entraînement PyTorch Lightning
   - Évaluation et analyse de confiance

3. **Prétraitement des données** (`data_preprocessing_enhanced.py`) :
   - Téléchargement des données depuis Kaggle
   - Transformations et augmentations avancées
   - Création de datasets et dataloaders
   - Analyse et visualisation des données
   - Support pour la validation croisée

## Prérequis

- Python 3.6+
- PyTorch 1.7+
- CUDA (recommandé pour l'accélération GPU)
- Bibliothèques supplémentaires :
  - pytorch-lightning
  - albumentations
  - timm
  - kaggle
  - wandb (optionnel)
  - onnx, onnxruntime (pour l'exportation)

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/plum-classification.git
   cd plum-classification
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Configurez l'accès à Kaggle :
   - Téléchargez votre fichier `kaggle.json` depuis votre compte Kaggle
   - Placez-le dans `~/.kaggle/` et définissez les permissions appropriées :
     ```bash
     mkdir -p ~/.kaggle
     cp kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

## Utilisation

### Entraînement du modèle

Pour lancer l'entraînement complet avec les paramètres par défaut :

```python
from training_pipeline_enhanced import EnhancedPlumClassificationPipeline

# Initialisation du pipeline
pipeline = EnhancedPlumClassificationPipeline(
    base_dir='./plum_classifier',
    use_cross_validation=True,
    use_tta=True,
    use_ensemble=True
)

# Exécution du pipeline complet
results = pipeline.run_pipeline()

# Affichage des résultats
print(f"Modèle sauvegardé à: {results['model_path']}")
print(f"Modèle ONNX exporté à: {results['onnx_path']}")
```

### Personnalisation de l'entraînement

Vous pouvez personnaliser divers aspects de l'entraînement :

```python
pipeline = EnhancedPlumClassificationPipeline(
    base_dir='./plum_classifier',
    kaggle_dataset='arnaudfadja/african-plums-quality-and-defect-assessment-data',
    use_wandb=True,
    wandb_project='plum-classifier',
    use_cross_validation=True,
    n_folds=5,
    use_tta=True,
    use_ensemble=True,
    n_models=3
)
```

### Utilisation en ligne de commande

Le script peut également être exécuté directement depuis la ligne de commande :

```bash
python training_pipeline_enhanced.py --base_dir ./plum_classifier --use_cross_validation --use_tta --use_ensemble
```

### Prédiction sur de nouvelles images

Pour utiliser le modèle entraîné sur de nouvelles images :

```python
# Charger le pipeline avec un modèle entraîné
pipeline = EnhancedPlumClassificationPipeline()
pipeline.setup_environment()

# Télécharger les datasets (nécessaire pour initialiser le préprocesseur)
datasets = pipeline.download_datasets()
data = pipeline.preprocess_data(datasets)

# Charger le modèle entraîné
from model_classification_enhanced import ModelTrainer
trainer = ModelTrainer(
    model_params={'num_classes': 6, 'confidence_threshold': 0.7},
    trainer_params={},
    models_dir=pipeline.models_dir,
    logs_dir=pipeline.logs_dir
)

# Prédiction sur une nouvelle image
results = trainer.predict(image_path='path/to/your/plum_image.jpg', tta=True)
print(f"Classe prédite: {results['class_name']}")
print(f"Confiance: {results['confidence']:.4f}")
print(f"Est une prune: {'Oui' if results['est_prune'] else 'Non'}")
```

## Performances

Le modèle atteint généralement une précision de 92-95% sur l'ensemble de test, avec des performances variables selon les catégories :

- Bonne qualité : 96-98% de précision
- Non mûre : 94-96% de précision
- Tachetée : 90-93% de précision
- Fissurée : 92-95% de précision
- Meurtrie : 88-92% de précision
- Pourrie : 93-96% de précision

L'utilisation de la validation croisée, du TTA et de l'ensemble de modèles améliore généralement les performances de 2-4% par rapport au modèle de base.

## Exportation du modèle

Le modèle peut être exporté dans différents formats pour le déploiement :

```python
# Exportation au format ONNX
onnx_path = pipeline.export_model(format='onnx')

# Exportation au format TorchScript
torchscript_path = pipeline.export_model(format='torchscript')
```

## Visualisations

Le pipeline génère automatiquement diverses visualisations dans le répertoire `results` :

- Distribution des classes
- Matrices de confusion
- Distribution des confidences
- Relation entre confiance et précision
- Visualisations de batches

## Intégration avec Weights & Biases

Pour un suivi détaillé des expériences, vous pouvez activer l'intégration avec Weights & Biases :

```python
pipeline = EnhancedPlumClassificationPipeline(
    use_wandb=True,
    wandb_project='plum-classifier',
    wandb_entity='votre-entite'
)
```

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Remerciements

- Dataset des prunes africaines : Arnaud Fadja
- JCIA Hackathon 2025 pour l'opportunité de développer ce projet
