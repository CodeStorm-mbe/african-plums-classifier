# Documentation Technique du Modèle de Classification de Prunes

## Table des matières

1. [Architecture du système](#architecture-du-système)
2. [Module de prétraitement des données](#module-de-prétraitement-des-données)
3. [Module de classification](#module-de-classification)
4. [Pipeline d'entraînement](#pipeline-dentraînement)
5. [Flux de travail](#flux-de-travail)
6. [Techniques avancées](#techniques-avancées)
7. [Métriques et évaluation](#métriques-et-évaluation)
8. [Exportation et déploiement](#exportation-et-déploiement)
9. [Intégration avec d'autres outils](#intégration-avec-dautres-outils)
10. [Dépannage et bonnes pratiques](#dépannage-et-bonnes-pratiques)

## Architecture du système

Le système de classification de prunes est organisé en trois modules principaux qui interagissent entre eux pour former un pipeline complet d'apprentissage automatique :

```
┌─────────────────────────┐
│ Pipeline d'entraînement │
│ (Orchestration)         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ Prétraitement des       │◄────►  Classification         │
│ données                 │     │  (Architecture modèle)   │
└─────────────────────────┘     └─────────────────────────┘
```

### Composants principaux

1. **EnhancedPlumClassificationPipeline** : Classe principale qui orchestre l'ensemble du processus
2. **DataPreprocessor** : Gère le téléchargement, le prétraitement et l'augmentation des données
3. **EnhancedPlumClassifier** : Architecture du modèle de classification
4. **ModelTrainer** : Gère l'entraînement, l'évaluation et l'exportation du modèle

## Module de prétraitement des données

### Classes principales

#### `KaggleDatasetDownloader`

Responsable du téléchargement et de l'extraction des datasets depuis Kaggle.

```python
downloader = KaggleDatasetDownloader(
    dataset_id='arnaudfadja/african-plums-quality-and-defect-assessment-data',
    data_dir='/path/to/data'
)
extracted_dir = downloader.download_and_extract()
```

#### `DataPreprocessor`

Gère le prétraitement des données, la création des datasets et des dataloaders.

**Méthodes principales :**

- `analyze_dataset()` : Analyse le dataset et génère des visualisations
- `prepare_data()` : Prépare les données pour l'entraînement, la validation et le test
- `get_class_weights()` : Calcule les poids des classes pour gérer le déséquilibre
- `create_cross_validation_folds()` : Crée des plis pour la validation croisée

**Transformations :**

Le préprocesseur utilise des transformations avancées d'Albumentations pour l'augmentation de données :

```python
train_transform = A.Compose([
    A.Resize(height=320, width=320),
    A.RandomResizedCrop(...),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.MotionBlur(p=0.2),
    A.ColorJitter(...),
    A.ShiftScaleRotate(...),
    A.Normalize(...),
    ToTensorV2(),
])
```

#### `PlumDataset`

Dataset personnalisé pour les images de prunes.

```python
dataset = PlumDataset(
    image_paths=image_paths,
    labels=labels,
    transform=transform
)
```

#### `CutMix`

Implémentation de la technique CutMix pour l'augmentation de données.

```python
cutmix = CutMix(alpha=1.0)
mixed_images, mixed_labels, _ = cutmix(batch)
```

## Module de classification

### Classes principales

#### `EnhancedPlumClassifier`

Architecture du modèle de classification basée sur EfficientNet avec des blocs d'attention et des connexions résiduelles.

**Caractéristiques :**
- Utilisation de EfficientNet comme backbone
- Blocs d'attention Squeeze-and-Excitation
- Connexions résiduelles
- Mécanisme de confiance pour déterminer si un échantillon est une prune

```python
model = EnhancedPlumClassifier(
    num_classes=6,
    model_name='efficientnet_b4',
    pretrained=True,
    dropout_rate=0.4,
    confidence_threshold=0.7
)
```

**Méthodes principales :**
- `forward()` : Passe avant du modèle
- `predict_with_confidence()` : Prédit la classe avec un score de confiance

#### `EnsemblePlumClassifier`

Modèle d'ensemble combinant plusieurs modèles EnhancedPlumClassifier.

```python
ensemble = EnsemblePlumClassifier(models=[model1, model2, model3])
```

#### `EnhancedPlumClassifierModule`

Module PyTorch Lightning pour l'entraînement du modèle.

**Caractéristiques :**
- Support pour Mixup et CutMix
- Focal Loss et Label Smoothing
- Calibration de température
- Gel/dégel progressif des couches

```python
module = EnhancedPlumClassifierModule(
    model_params={...},
    class_weights=class_weights,
    use_mixup_cutmix=True,
    learning_rate=3e-4
)
```

**Méthodes principales :**
- `training_step()` : Étape d'entraînement
- `validation_step()` : Étape de validation
- `test_step()` : Étape de test
- `configure_optimizers()` : Configuration de l'optimiseur et du scheduler

#### `ModelTrainer`

Classe pour l'entraînement et l'évaluation du modèle.

```python
trainer = ModelTrainer(
    model_params={...},
    trainer_params={...},
    models_dir='./models',
    logs_dir='./logs'
)
```

**Méthodes principales :**
- `train()` : Entraîne le modèle
- `train_with_cross_validation()` : Entraîne avec validation croisée
- `evaluate()` : Évalue le modèle sur l'ensemble de test
- `evaluate_with_tta()` : Évalue avec Test-Time Augmentation
- `predict()` : Prédit la classe d'une image
- `analyze_confidence_distribution()` : Analyse la distribution des confidences
- `export_to_onnx()` : Exporte le modèle au format ONNX

### Fonctions de perte

#### `FocalLoss`

Focal Loss pour gérer le déséquilibre des classes.

```python
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
```

#### `LabelSmoothingLoss`

Label Smoothing Loss pour améliorer la généralisation.

```python
criterion = LabelSmoothingLoss(
    classes=6,
    smoothing=0.1,
    weight=class_weights
)
```

## Pipeline d'entraînement

### `EnhancedPlumClassificationPipeline`

Classe principale qui orchestre l'ensemble du processus.

**Méthodes principales :**

- `setup_environment()` : Configure l'environnement d'exécution
- `download_datasets()` : Télécharge les datasets
- `preprocess_data()` : Prétraite les données
- `train_model()` : Entraîne le modèle
- `evaluate_model()` : Évalue le modèle
- `analyze_confidence()` : Analyse la distribution des confidences
- `export_model()` : Exporte le modèle
- `run_pipeline()` : Exécute le pipeline complet

**Options avancées :**

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

## Flux de travail

Le flux de travail complet du système suit ces étapes :

1. **Configuration de l'environnement**
   - Vérification et installation des dépendances
   - Configuration de Kaggle et Weights & Biases

2. **Téléchargement et prétraitement des données**
   - Téléchargement du dataset depuis Kaggle
   - Analyse du dataset (distribution des classes, dimensions)
   - Préparation des données (train/val/test split)
   - Augmentation des données

3. **Entraînement du modèle**
   - Entraînement standard ou avec validation croisée
   - Utilisation de techniques avancées (Mixup, CutMix)
   - Suivi des métriques et early stopping

4. **Évaluation du modèle**
   - Évaluation sur l'ensemble de test
   - Évaluation avec Test-Time Augmentation
   - Analyse des confidences et calibration

5. **Exportation et visualisation**
   - Exportation du modèle (ONNX, TorchScript)
   - Génération de visualisations (matrices de confusion, etc.)
   - Sauvegarde des résultats

## Techniques avancées

### Validation croisée (K-fold)

La validation croisée est implémentée via `create_cross_validation_folds()` et `train_with_cross_validation()`.

```python
folds = preprocessor.create_cross_validation_folds(n_splits=5)
results = trainer.train_with_cross_validation(folds)
```

### Test-Time Augmentation (TTA)

Le TTA est implémenté via `evaluate_with_tta()` et `predict()` avec l'option `tta=True`.

```python
results = trainer.evaluate_with_tta(test_dataloader, tta_transforms)
```

### Ensemble de modèles

L'ensemble de modèles est créé via `create_ensemble_model()`.

```python
ensemble_path, _ = trainer.create_ensemble_model(fold_results)
```

### Calibration de confiance

La calibration de confiance est réalisée via le paramètre de température dans `EnhancedPlumClassifierModule`.

```python
# La température est ajustée automatiquement pendant l'entraînement
self.temperature = nn.Parameter(torch.ones(1) * 1.5)
```

## Métriques et évaluation

Le système utilise diverses métriques pour évaluer les performances :

- **Exactitude (Accuracy)** : Proportion de prédictions correctes
- **Précision (Precision)** : Proportion de prédictions positives correctes
- **Rappel (Recall)** : Proportion de positifs réels correctement identifiés
- **Score F1** : Moyenne harmonique de la précision et du rappel
- **Matrice de confusion** : Visualisation des prédictions par classe
- **Distribution des confidences** : Analyse de la fiabilité des prédictions

## Exportation et déploiement

### Exportation ONNX

```python
onnx_path = trainer.export_to_onnx(
    output_path='./models/plum_classifier.onnx',
    input_shape=(1, 3, 320, 320)
)
```

### Exportation TorchScript

```python
torchscript_path = pipeline.export_model(format='torchscript')
```

### Métadonnées

Les métadonnées du modèle sont sauvegardées avec le modèle exporté :

```json
{
    "num_classes": 6,
    "confidence_threshold": 0.7,
    "idx_to_class": {
        "0": "bonne_qualite",
        "1": "non_mure",
        "2": "tachetee",
        "3": "fissuree",
        "4": "meurtrie",
        "5": "pourrie"
    },
    "image_size": 320,
    "format": "onnx"
}
```

## Intégration avec d'autres outils

### Weights & Biases

L'intégration avec Weights & Biases permet de suivre les expériences :

```python
if self.use_wandb:
    self.wandb.init(
        project=self.wandb_project,
        entity=self.wandb_entity,
        config={...}
    )
```

### Google Colab

Le système détecte automatiquement l'environnement Colab et s'adapte en conséquence :

```python
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False
```

### Google Drive

Sauvegarde automatique des modèles dans Google Drive si exécuté dans Colab :

```python
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    # Sauvegarde des modèles
```

## Dépannage et bonnes pratiques

### Gestion des erreurs de chargement d'images

```python
try:
    image = Image.open(image_path).convert('RGB')
    # Traitement de l'image
except Exception as e:
    print(f"Erreur lors du chargement de l'image {image_path}: {e}")
    # Retourner une image noire de la bonne taille
```

### Optimisation de la mémoire GPU

- Utilisation de la précision mixte (AMP)
- Réduction de la taille des batchs si nécessaire
- Gel progressif des couches du modèle

### Reproductibilité

Pour assurer la reproductibilité des résultats :

```python
# Définir les graines aléatoires
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Gestion du déséquilibre des classes

- Utilisation de WeightedRandomSampler
- Focal Loss
- Calcul des poids des classes inversement proportionnels à leur fréquence
