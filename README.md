# Classificateur de Prunes Africaines en Deux Étapes

Ce projet implémente un système de classification en deux étapes pour les prunes africaines, développé pour le hackathon JCIA 2025.

## Fonctionnalités

1. **Approche en deux étapes** :
   - **Détection** : Détermine d'abord si l'image contient une prune ou non
   - **Classification** : Si une prune est détectée, classifie son état parmi six catégories

2. **Catégories de classification** :
   - Bonne qualité (unaffected)
   - Non mûre (unripe)
   - Tachetée (spotted)
   - Fissurée (cracked)
   - Meurtrie (bruised)
   - Pourrie (rotten)

3. **Environnements supportés** :
   - Dell XPS 15 avec GPU Intel Arc 370
   - Google Colab (avec GPU)

4. **Intégration avec Django** :
   - Guide complet pour intégrer le modèle dans une application web

## Structure du projet

```
african-plums-classifier/
├── data/                         # Prétraitement des données
│   └── data_preprocessing.py     # Fonctions de chargement et prétraitement
├── models/                       # Définitions des modèles
│   └── model_architecture.py     # Architecture du modèle en deux étapes
├── scripts/                      # Scripts d'entraînement et de test
│   ├── train_two_stage.py        # Entraînement du modèle en deux étapes
│   └── test_two_stage.py         # Test du modèle avec des images
├── utils/                        # Utilitaires
│   └── visualization.py          # Fonctions de visualisation
├── plum_classifier_test_two_stage.ipynb  # Notebook Colab pour tester le modèle
└── README.md                     # Documentation
```

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/african-plums-classifier.git
cd african-plums-classifier

# Installer les dépendances
pip install torch torchvision numpy matplotlib scikit-learn pillow
```

## Utilisation

### Entraînement du modèle en deux étapes

```bash
python scripts/train_two_stage.py \
  --plum_data_dir /chemin/vers/african_plums_dataset \
  --non_plum_data_dir /chemin/vers/non_plum_images \
  --save_dir models \
  --batch_size 32 \
  --num_epochs 25
```

### Test du modèle

```bash
python scripts/test_two_stage.py \
  --detection_model_path models/detection_best_acc.pth \
  --classification_model_path models/classification_best_acc.pth \
  --model_info_path models/two_stage_model_info.json \
  --image_path /chemin/vers/image.jpg
```

### Utilisation sur Google Colab

Le notebook `plum_classifier_test_two_stage.ipynb` fournit une interface complète pour tester le modèle sur Google Colab. Il vous permet de :

1. Charger les modèles entraînés depuis Google Drive
2. Télécharger des images individuelles pour les tester
3. Tester des lots d'images depuis un dossier
4. Visualiser les résultats avec des graphiques de probabilités

## Préparation des données

Pour entraîner le modèle en deux étapes, vous aurez besoin de deux ensembles de données :

1. **Images de prunes** : Le dataset "African Plums Dataset" de Kaggle, organisé en six catégories
2. **Images qui ne sont pas des prunes** : Un ensemble d'images diverses qui ne contiennent pas de prunes

Le script `train_two_stage.py` s'attend à ce que les données soient organisées comme suit :

```
african_plums_dataset/
├── unaffected/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── unripe/
│   ├── image1.jpg
│   └── ...
└── ...

non_plum_images/
├── image1.jpg
├── image2.jpg
└── ...
```

## Intégration avec Django

Pour intégrer ce modèle dans une application Django, suivez ces étapes :

1. Créez une nouvelle application Django
2. Copiez les fichiers du modèle entraîné dans votre projet Django
3. Créez une vue qui utilise le modèle pour faire des prédictions
4. Créez un formulaire pour permettre aux utilisateurs de télécharger des images

Un guide détaillé est disponible dans le dépôt séparé : [african-plums-web](https://github.com/votre-username/african-plums-web)

## Performances

Le modèle en deux étapes offre plusieurs avantages :

1. **Robustesse** : Rejette les images qui ne contiennent pas de prunes
2. **Précision** : Se concentre sur la classification des prunes uniquement
3. **Flexibilité** : Permet d'ajuster le seuil de détection selon les besoins

## Auteurs

Développé pour le hackathon JCIA 2025 par [Votre Nom].

## Licence

Ce projet est sous licence MIT.
