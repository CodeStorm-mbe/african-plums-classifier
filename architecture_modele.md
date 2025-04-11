# Architecture du modèle de deep learning pour la classification des prunes africaines

## Analyse des besoins
- Classification en 6 catégories: meurtries, fissurées, pourries, tachetées, non affectées, non mûres
- Distribution déséquilibrée des classes (162 à 1721 images par classe)
- Besoin d'un modèle robuste et précis pour le hackathon JCIA

## Choix de l'architecture de base
Après analyse des performances sur des tâches similaires de classification d'images, j'ai sélectionné plusieurs architectures pré-entraînées à comparer:

1. **EfficientNetB3**
   - Excellent équilibre entre précision et efficacité computationnelle
   - Scaling uniforme des dimensions du réseau (profondeur, largeur, résolution)
   - Performances supérieures avec moins de paramètres que d'autres architectures

2. **ResNet50V2**
   - Architecture éprouvée avec connexions résiduelles
   - Permet d'entraîner des réseaux très profonds sans problème de dégradation
   - Bonne généralisation sur diverses tâches de vision par ordinateur

3. **Xception**
   - Utilise des convolutions séparables en profondeur
   - Efficace pour capturer des motifs complexes
   - Bon compromis entre taille du modèle et performances

4. **DenseNet201**
   - Connexions denses entre les couches
   - Réutilisation efficace des caractéristiques
   - Moins susceptible au surapprentissage

## Structure du modèle final
Le modèle final sera un ensemble de ces architectures, avec la structure suivante pour chaque modèle individuel:

```
Input (224x224x3)
    ↓
Modèle pré-entraîné (sans couches de classification)
    ↓
Global Average Pooling 2D
    ↓
BatchNormalization
    ↓
Dense (512 unités, activation ReLU, régularisation L2)
    ↓
BatchNormalization
    ↓
Dropout (0.3)
    ↓
Dense (256 unités, activation ReLU, régularisation L2)
    ↓
BatchNormalization
    ↓
Dropout (0.3)
    ↓
Dense (6 unités, activation Softmax)
```

## Techniques d'augmentation de données
Pour gérer le déséquilibre des classes et améliorer la généralisation:

1. **Augmentations classiques**:
   - Rotation aléatoire (±15°)
   - Retournement horizontal et vertical
   - Décalage, mise à l'échelle et rotation
   - Modifications de luminosité et de contraste
   - Ajout de bruit gaussien
   - Flou gaussien et flou de mouvement

2. **Techniques avancées**:
   - **Mixup**: Combinaison linéaire de paires d'images et de leurs étiquettes
   - **CutMix**: Découpage et mélange de régions d'images
   - **CoarseDropout**: Suppression aléatoire de zones rectangulaires

## Stratégies d'entraînement
Pour maximiser les performances et la robustesse:

1. **Validation croisée à 5 plis**:
   - Entraînement de 5 modèles sur différentes partitions des données
   - Amélioration de la généralisation et réduction de la variance

2. **Optimiseur AdamW**:
   - Combinaison d'Adam avec décroissance de poids
   - Meilleure généralisation que l'optimiseur Adam standard

3. **Gestion du déséquilibre des classes**:
   - Pondération des classes inversement proportionnelle à leur fréquence
   - Surpondération des classes minoritaires (fissurées, meurtries)

4. **Techniques de régularisation**:
   - Dropout (taux de 0.3)
   - Régularisation L2 (1e-4)
   - BatchNormalization
   - Early stopping
   - Réduction du taux d'apprentissage sur plateau

5. **Optimisation des hyperparamètres**:
   - Recherche des meilleurs hyperparamètres avec Optuna
   - Optimisation du taux d'apprentissage, de la taille des batchs, du taux de dropout

## Modèle d'ensemble
Pour maximiser la précision finale:

1. **Moyenne des prédictions**:
   - Combinaison des prédictions des modèles individuels
   - Réduction de la variance et amélioration de la robustesse

2. **Diversité des modèles**:
   - Utilisation de différentes architectures de base
   - Entraînement sur différentes partitions des données

## Métriques d'évaluation
Pour mesurer les performances du modèle:

1. **Précision (Accuracy)**:
   - Métrique principale demandée par le hackathon
   - Proportion d'images correctement classifiées

2. **Métriques supplémentaires**:
   - Précision (Precision): Mesure la proportion de vrais positifs parmi les prédictions positives
   - Rappel (Recall): Mesure la proportion de vrais positifs détectés
   - Score F1: Moyenne harmonique de la précision et du rappel
   - Matrice de confusion: Visualisation détaillée des erreurs de classification

## Gestion des erreurs et robustesse
Pour assurer la fiabilité du modèle:

1. **Détection et gestion des cas limites**:
   - Vérification de la qualité des images en entrée
   - Seuil de confiance pour les prédictions

2. **Techniques de test**:
   - Test-time augmentation (TTA)
   - Ensemble de modèles pour réduire les erreurs

Cette architecture a été conçue pour maximiser la précision de classification tout en assurant la robustesse face aux variations dans les données d'entrée et au déséquilibre des classes.
