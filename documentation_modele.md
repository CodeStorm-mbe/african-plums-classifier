# Documentation du Modèle de Classification des Prunes Africaines

## Présentation du projet

Ce projet a été développé dans le cadre du Hackathon JCIA 2025, qui porte sur le tri automatique des prunes africaines en six catégories différentes à l'aide de techniques de vision par ordinateur et d'apprentissage profond.

Le modèle permet de classifier les prunes africaines dans les catégories suivantes :
- Bonne qualité (unaffected)
- Non mûre (unripe)
- Tachetée (spotted)
- Fissurée (cracked)
- Meurtrie (bruised)
- Pourrie (rotten)

Cette documentation détaille l'architecture du modèle, les techniques de robustesse implémentées, et les instructions d'utilisation.

## Structure du dataset

Le dataset utilisé pour ce projet est le "African Plums Quality and Defect Assessment Data" disponible sur Kaggle. Il contient 4507 images annotées de prunes africaines collectées au Cameroun, réparties comme suit :

| Catégorie | Nombre d'images |
|-----------|-----------------|
| Meurtries (bruised) | 319 |
| Fissurées (cracked) | 162 |
| Pourries (rotten) | 720 |
| Tachetées (spotted) | 759 |
| Bonne qualité (unaffected) | 1721 |
| Non mûres (unripe) | 826 |

Cette distribution est déséquilibrée, avec beaucoup plus d'images de prunes de bonne qualité que de prunes fissurées. Le modèle intègre des techniques pour gérer ce déséquilibre.

## Architecture du modèle

### Vue d'ensemble

Le modèle de classification des prunes africaines utilise une architecture avancée basée sur des réseaux pré-entraînés. L'approche adoptée est celle d'un ensemble de modèles pour maximiser les performances et la robustesse.

### Modèles de base

Quatre architectures pré-entraînées sont utilisées comme extracteurs de caractéristiques :
1. **EfficientNetB3** : Excellent rapport performance/coût computationnel
2. **ResNet50V2** : Architecture résiduelle profonde avec connexions skip
3. **Xception** : Architecture basée sur des convolutions séparables en profondeur
4. **DenseNet201** : Architecture avec connexions denses entre les couches

### Structure du modèle

Chaque modèle individuel comprend :
- Un modèle pré-entraîné comme extracteur de caractéristiques
- Des couches de pooling global et de normalisation par lots
- Des couches denses avec régularisation L2 et dropout
- Une couche de sortie softmax à 6 unités (une pour chaque catégorie)

### Techniques d'augmentation de données

Pour améliorer la généralisation du modèle, plusieurs techniques d'augmentation de données sont utilisées :

#### Augmentations classiques
- Rotation aléatoire
- Retournement horizontal et vertical
- Modifications de luminosité et de contraste
- Zoom et décalage

#### Techniques avancées
- **Mixup** : Mélange linéaire de paires d'images et de leurs étiquettes
- **CutMix** : Découpage et collage de régions d'images avec ajustement proportionnel des étiquettes

### Stratégies d'entraînement

- **Validation croisée à 5 plis** : Améliore la généralisation et réduit la variance
- **Optimiseur AdamW** : Adam avec décroissance de poids pour une meilleure régularisation
- **Pondération des classes** : Compense le déséquilibre du dataset
- **Techniques de régularisation** : Dropout, L2, batch normalization

## Techniques de robustesse

Le modèle intègre de nombreuses techniques avancées pour améliorer sa robustesse :

### 1. Test-Time Augmentation (TTA)

Cette technique applique plusieurs transformations à l'image lors de la prédiction et moyenne les résultats. Les transformations incluent :
- Retournement horizontal et vertical
- Rotation légère
- Modifications de luminosité et de contraste
- Flou gaussien
- Bruit gaussien
- Modifications gamma
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Modifications HSV (Hue, Saturation, Value)

Avantages :
- Réduit l'impact des variations dans l'image d'entrée
- Améliore la stabilité des prédictions
- Augmente la précision globale

### 2. Stochastic Weight Averaging (SWA)

SWA améliore la généralisation en moyennant les poids du modèle sur plusieurs époques d'entraînement. Cette technique :
- Trouve des minima plus plats dans la fonction de perte
- Améliore la généralisation sur des données non vues
- Réduit la sensibilité aux conditions initiales

### 3. Calibration de confiance

Le temperature scaling est utilisé pour calibrer les scores de confiance du modèle. Cette technique :
- Ajuste les logits avant l'application de softmax
- Produit des probabilités mieux calibrées
- Permet une meilleure interprétation des scores de confiance

### 4. Détection d'anomalies et d'exemples hors distribution

Le modèle peut détecter si une image est une anomalie ou un exemple hors distribution en :
- Calculant l'entropie de la distribution de probabilité prédite
- Comparant la confiance maximale à un seuil prédéfini
- Combinant ces métriques pour identifier les cas anormaux

### 5. Snapshot Ensemble

Cette technique crée un ensemble de modèles en sauvegardant les poids à différents moments de l'entraînement. Elle utilise :
- Un planificateur de taux d'apprentissage avec décroissance en cosinus
- Plusieurs cycles d'entraînement
- Une combinaison des prédictions de tous les modèles snapshot

### 6. Optimisation bayésienne des hyperparamètres

Les hyperparamètres du modèle sont optimisés avec des méthodes bayésiennes via Optuna. Les hyperparamètres optimisés incluent :
- Architecture du modèle de base
- Taux de dropout
- Coefficient de régularisation L2
- Taux d'apprentissage
- Taille du batch
- Choix de l'optimiseur

### 7. Gestion avancée des erreurs et des cas limites

Le pipeline de prédiction robuste gère de nombreux cas d'erreur potentiels :
- Fichiers inexistants
- Images illisibles
- Formats d'image non supportés
- Images trop petites
- Détection d'anomalies
- Faible confiance dans les prédictions

## Instructions d'utilisation

### Prérequis

- Google Colab avec accès GPU
- Compte Kaggle pour accéder au dataset

### Installation et configuration

1. Importez le notebook `african_plums_demo.ipynb` dans Google Colab
2. Exécutez la section "Configuration de l'environnement" pour installer les bibliothèques nécessaires
3. Configurez l'accès à Kaggle en suivant les instructions dans la section correspondante
4. Exécutez la section "Téléchargement du dataset" pour obtenir les données

### Utilisation de l'interface de démonstration

1. Exécutez toutes les cellules jusqu'à la section "Interface de démonstration interactive"
2. Utilisez le widget de téléchargement pour charger une image de prune africaine
3. Ajustez le seuil de confiance selon vos besoins
4. Activez ou désactivez les techniques de robustesse (TTA, détection d'anomalies)
5. Cliquez sur le bouton "Prédire" pour obtenir la classification de l'image

### Interprétation des résultats

- **Classe prédite** : Catégorie de la prune (bonne qualité, non mûre, etc.)
- **Score de confiance** : Probabilité associée à la prédiction (entre 0 et 1)
- **Statut** : Indication de la fiabilité de la prédiction
  - "OK" : Prédiction fiable
  - "Avertissement: Confiance faible" : Prédiction incertaine
  - "Avertissement: Possible anomalie" : Image potentiellement hors distribution

## Performances et limitations

### Performances

Le modèle atteint d'excellentes performances sur le dataset des prunes africaines, avec :
- Une précision globale élevée
- Une bonne gestion des classes minoritaires
- Une robustesse face aux variations dans les images d'entrée

### Limitations

- Le modèle peut avoir des difficultés avec des images très différentes de celles du dataset d'entraînement
- Les prédictions peuvent être moins fiables pour les classes sous-représentées (comme "fissurées")
- Le temps d'inférence est plus long lorsque toutes les techniques de robustesse sont activées

## Conclusion

Ce modèle de deep learning robuste pour la classification des prunes africaines intègre de nombreuses techniques avancées pour maximiser ses performances et sa fiabilité. Il est prêt à être utilisé dans le cadre du Hackathon JCIA 2025 pour le tri automatique des prunes africaines.

Les techniques de robustesse implémentées permettent au modèle de gérer efficacement les variations dans les données d'entrée et de fournir des prédictions fiables même dans des conditions difficiles.
