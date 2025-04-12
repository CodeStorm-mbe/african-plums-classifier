# Guide d'utilisation du modèle de classification de prunes dans Google Colab

Ce guide explique comment utiliser le modèle de classification de prunes spécifiquement dans l'environnement Google Colab. Il complète la documentation principale et le README.md en fournissant des instructions détaillées pour l'exécution dans cet environnement cloud.

## Configuration initiale

### 1. Téléchargement des fichiers sources

Commencez par télécharger les trois fichiers Python principaux dans votre environnement Colab :

```python
# Téléchargement des fichiers sources
!wget -q https://raw.githubusercontent.com/votre-repo/plum-classification/main/training_pipeline_enhanced.py
!wget -q https://raw.githubusercontent.com/votre-repo/plum-classification/main/model_classification_enhanced.py
!wget -q https://raw.githubusercontent.com/votre-repo/plum-classification/main/data_preprocessing_enhanced.py
```

Alternativement, vous pouvez télécharger ces fichiers depuis votre ordinateur local :

```python
from google.colab import files
uploaded = files.upload()  # Sélectionnez les trois fichiers Python
```

### 2. Installation des dépendances

Le pipeline installera automatiquement les dépendances nécessaires, mais vous pouvez également les installer manuellement :

```python
!pip install -q pytorch-lightning albumentations timm wandb kaggle onnx onnxruntime
```

### 3. Configuration de Kaggle

Pour accéder au dataset des prunes sur Kaggle, vous devez configurer vos identifiants Kaggle :

```python
from google.colab import files
files.upload()  # Téléchargez votre fichier kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### 4. Vérification du GPU

Vérifiez que vous avez bien accès à un GPU :

```python
import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Modèle GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

## Exécution du pipeline complet

### Option 1 : Exécution avec les paramètres par défaut

```python
from training_pipeline_enhanced import EnhancedPlumClassificationPipeline

# Initialisation du pipeline
pipeline = EnhancedPlumClassificationPipeline(
    base_dir='/content/plum_classifier',
    use_cross_validation=True,
    use_tta=True,
    use_ensemble=True
)

# Exécution du pipeline complet
results = pipeline.run_pipeline()
```

### Option 2 : Exécution avec personnalisation

```python
from training_pipeline_enhanced import EnhancedPlumClassificationPipeline

# Initialisation du pipeline avec des options personnalisées
pipeline = EnhancedPlumClassificationPipeline(
    base_dir='/content/plum_classifier',
    kaggle_dataset='arnaudfadja/african-plums-quality-and-defect-assessment-data',
    use_wandb=True,  # Activer le suivi avec Weights & Biases
    wandb_project='plum-classifier',
    use_cross_validation=True,
    n_folds=5,
    use_tta=True,
    use_ensemble=True,
    n_models=3
)

# Exécution du pipeline
results = pipeline.run_pipeline()
```

## Sauvegarde des modèles dans Google Drive

Le pipeline sauvegarde automatiquement les meilleurs modèles dans Google Drive si vous exécutez dans Colab. Vous pouvez également le faire manuellement :

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
import os
from datetime import datetime

# Créer un répertoire de sauvegarde dans Drive
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = f"/content/drive/MyDrive/colab_backups/plum_models/{timestamp}"
os.makedirs(backup_dir, exist_ok=True)

# Copier les modèles
models_dir = '/content/plum_classifier/models'
for f in os.listdir(models_dir):
    if f.endswith('.pt') or f.endswith('.onnx') or f.endswith('.json'):
        src = os.path.join(models_dir, f)
        dst = os.path.join(backup_dir, f)
        shutil.copy(src, dst)
        print(f"Modèle sauvegardé dans Drive : {dst}")
```

## Visualisation des résultats

### Affichage des matrices de confusion et autres visualisations

```python
import matplotlib.pyplot as plt
from IPython.display import display, Image

# Afficher la matrice de confusion
confusion_matrix_path = '/content/plum_classifier/results/confusion_matrix.png'
display(Image(filename=confusion_matrix_path))

# Afficher la distribution des confidences
confidence_distribution_path = '/content/plum_classifier/results/confidence_distribution.png'
display(Image(filename=confidence_distribution_path))
```

### Visualisation avec Weights & Biases

Si vous avez activé Weights & Biases, vous pouvez accéder à votre tableau de bord pour visualiser les métriques d'entraînement en temps réel :

```python
import wandb
wandb.init(project="plum-classifier")
print(f"Tableau de bord W&B : {wandb.run.get_url()}")
```

## Test du modèle sur de nouvelles images

### Téléchargement d'images de test

```python
from google.colab import files
uploaded = files.upload()  # Téléchargez une image de prune pour le test

# Obtenir le nom du fichier téléchargé
image_path = next(iter(uploaded.keys()))
```

### Prédiction avec le modèle

```python
# Charger le pipeline avec un modèle entraîné
from model_classification_enhanced import ModelTrainer

trainer = ModelTrainer(
    model_params={'num_classes': 6, 'confidence_threshold': 0.7},
    trainer_params={},
    models_dir='/content/plum_classifier/models',
    logs_dir='/content/plum_classifier/logs'
)

# Charger le meilleur modèle
import os
models_dir = '/content/plum_classifier/models'
best_model = [f for f in os.listdir(models_dir) if f.endswith('.pt')][0]
best_model_path = os.path.join(models_dir, best_model)
metadata_path = best_model_path.replace('.pt', '_metadata.json')

# Prédiction sur l'image téléchargée
model = trainer.load_model(best_model_path, metadata_path)
results = trainer.predict(image_path=image_path, tta=True)

# Afficher les résultats
print(f"Classe prédite: {results['class_name']}")
print(f"Confiance: {results['confidence']:.4f}")
print(f"Est une prune: {'Oui' if results['est_prune'] else 'Non'}")

# Visualiser la prédiction
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open(image_path).convert('RGB')
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.title(f"Prédiction: {results['class_name']} (confiance: {results['confidence']:.2f})")
plt.axis('off')
plt.show()
```

## Exportation du modèle

### Exportation au format ONNX

```python
# Exporter le modèle au format ONNX
onnx_path = trainer.export_to_onnx(
    output_path='/content/plum_classifier/models/plum_classifier_final.onnx',
    input_shape=(1, 3, 320, 320)
)
print(f"Modèle exporté au format ONNX: {onnx_path}")

# Télécharger le modèle ONNX
files.download(onnx_path)
```

## Astuces pour Colab

### Éviter les déconnexions

Pour éviter que votre session Colab ne se déconnecte pendant un long entraînement, vous pouvez utiliser cette astuce JavaScript :

```javascript
function ClickConnect(){
  console.log("Clicking connect button"); 
  document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect, 60000)
```

### Surveillance de l'utilisation du GPU

```python
# Installer et utiliser gputil pour surveiller l'utilisation du GPU
!pip install -q gputil
import GPUtil
GPUtil.showUtilization()
```

### Notification à la fin de l'entraînement

```python
# Jouer un son à la fin de l'entraînement
from IPython.display import Audio
import numpy as np

def notify_completion():
    rate = 22050  # échantillons par seconde
    duration = 2  # secondes
    frequency = 440  # Hz (La4)
    t = np.linspace(0, duration, int(rate * duration))
    signal = np.sin(2 * np.pi * frequency * t)
    return Audio(signal, rate=rate, autoplay=True)

# À la fin de l'entraînement
notify_completion()
print("Entraînement terminé!")
```

## Résolution des problèmes courants

### Problème de mémoire GPU

Si vous rencontrez des problèmes de mémoire GPU, essayez ces solutions :

```python
# Réduire la taille du batch
pipeline = EnhancedPlumClassificationPipeline(
    base_dir='/content/plum_classifier',
    batch_size=8  # Valeur par défaut: 16
)

# Utiliser un modèle plus petit
pipeline = EnhancedPlumClassificationPipeline(
    base_dir='/content/plum_classifier',
    model_params={'model_name': 'efficientnet_b0'}  # Au lieu de b4
)

# Libérer la mémoire cache CUDA
import torch
torch.cuda.empty_cache()
```

### Problème d'accès à Kaggle

Si vous rencontrez des problèmes avec l'API Kaggle :

```python
# Vérifier que le fichier kaggle.json est correctement configuré
!cat ~/.kaggle/kaggle.json

# Vérifier les permissions
!ls -la ~/.kaggle/kaggle.json

# Réinstaller l'API Kaggle
!pip install --upgrade kaggle
```

### Problème avec PyTorch Lightning

Si vous rencontrez des problèmes avec PyTorch Lightning :

```python
# Désactiver la détection automatique de la précision
pipeline = EnhancedPlumClassificationPipeline(
    base_dir='/content/plum_classifier',
    trainer_params={'auto_select_gpus': False, 'precision': 32}
)
```
