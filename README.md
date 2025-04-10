# Projet de Classification des Prunes Africaines

Ce projet implémente un système de classification automatique des prunes africaines en six catégories (bonne qualité, non mûre, tachetée, fissurée, meurtrie et pourrie) en utilisant des techniques de vision par ordinateur et d'apprentissage profond.

## Structure du Projet

```
plum_classifier/
├── data/                         # Dossier pour les données
├── models/                       # Dossier pour les modèles
│   └── model_architecture.py     # Définition de l'architecture du modèle
├── utils/                        # Utilitaires
│   └── data_preprocessing.py     # Prétraitement des données
├── django_app/                   # Application Django
│   └── README.md                 # Guide d'intégration avec Django
├── train.py                      # Script principal d'entraînement
├── run_on_dell_xps.py            # Script pour exécution sur Dell XPS avec Intel Arc
└── run_on_colab.py               # Script pour exécution sur Google Colab
```

## Fonctionnalités

- **Prétraitement des données** : Chargement, transformation et préparation des images de prunes
- **Modèles de classification** : Implémentation de différentes architectures (ResNet, MobileNet, EfficientNet)
- **Entraînement et évaluation** : Scripts pour entraîner et évaluer les modèles
- **Adaptation pour différents environnements** : Support pour Dell XPS avec GPU Intel Arc et Google Colab
- **Intégration avec Django** : Guide pour intégrer le modèle dans une application web

## Prérequis

- Python 3.8+
- PyTorch 1.8+
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- Pillow

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/username/plum-classifier.git
cd plum-classifier

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Entraînement du modèle

```bash
python train.py --data_dir /chemin/vers/dataset --save_dir models --model_type standard --base_model resnet18 --num_epochs 20
```

Options disponibles :
- `--data_dir` : Chemin vers le répertoire de données
- `--save_dir` : Chemin pour sauvegarder les modèles
- `--model_type` : Type de modèle (`standard` ou `lightweight`)
- `--base_model` : Modèle de base (`resnet18`, `resnet50`, `mobilenet_v2`, `efficientnet_b0`)
- `--batch_size` : Taille des batches
- `--num_epochs` : Nombre d'époques
- `--learning_rate` : Learning rate initial
- `--img_size` : Taille des images
- `--num_workers` : Nombre de workers pour le chargement des données

### Exécution sur Dell XPS avec GPU Intel Arc

```bash
python run_on_dell_xps.py --model_path models/best_model_acc.pth --image_path /chemin/vers/image.jpg --model_info models/model_info.json
```

### Exécution sur Google Colab

Pour exécuter sur Google Colab, vous pouvez soit :

1. Générer un notebook Colab :
```bash
python run_on_colab.py
```

2. Ou exécuter directement dans Colab :
```python
!git clone https://github.com/username/plum-classifier.git
%cd plum-classifier
!python run_on_colab.py --kaggle_username votre_username --kaggle_key votre_key
```

### Intégration avec Django

Voir le guide détaillé dans `django_app/README.md`.

## Dataset

Ce projet utilise le dataset "African Plums" disponible sur Kaggle :
[African Plums Dataset](https://www.kaggle.com/datasets/arnaudfadja/african-plums-quality-and-defect-assessment-data)

Le dataset contient 4,507 images annotées de prunes africaines collectées au Cameroun, classées en six catégories :
- Bonne qualité (unaffected)
- Non mûre (unripe)
- Tachetée (spotted)
- Fissurée (cracked)
- Meurtrie (bruised)
- Pourrie (rotten)

## Licence

Ce projet est sous licence MIT.
