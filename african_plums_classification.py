#!/usr/bin/env python
# coding: utf-8

# # Hackathon JCIA 2025 - Tri Automatique des Prunes Africaines
# 
# Ce notebook présente un modèle de deep learning robuste pour la classification des prunes africaines en six catégories:
# - Bonne qualité
# - Non mûre
# - Tachetée
# - Fissurée
# - Meurtrie
# - Pourrie
# 
# ## Configuration de l'environnement

# ### Installation des bibliothèques nécessaires

# In[1]:


# Installation des bibliothèques essentielles
!pip install -q tensorflow==2.15.0 tensorflow-addons
!pip install -q keras==2.15.0
!pip install -q scikit-learn==1.3.2
!pip install -q matplotlib seaborn pandas
!pip install -q opencv-python
!pip install -q albumentations
!pip install -q kaggle
!pip install -q efficientnet
!pip install -q tensorflow_probability
!pip install -q wandb
!pip install -q optuna


# ### Configuration de l'accès à Kaggle

# In[2]:


# Configuration de l'accès à Kaggle
import os
import json

# Créer le dossier kaggle s'il n'existe pas
!mkdir -p ~/.kaggle

# Créer le fichier kaggle.json avec vos identifiants
# Remplacez 'your_username' et 'your_key' par vos identifiants Kaggle
kaggle_json = {
    "username": "your_username",
    "key": "your_key"
}

# Écrire les identifiants dans le fichier kaggle.json
with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump(kaggle_json, f)

# Définir les permissions appropriées
!chmod 600 ~/.kaggle/kaggle.json


# ### Téléchargement du dataset African Plums

# In[3]:


# Télécharger le dataset depuis Kaggle
# Remplacez 'dataset_owner/african-plums-dataset' par le chemin réel du dataset sur Kaggle
!kaggle datasets download -d dataset_owner/african-plums-dataset
!unzip -q african-plums-dataset.zip -d african_plums


# ## Importation des bibliothèques

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications, optimizers, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2, Xception, DenseNet201
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import albumentations as A
from albumentations.tensorflow import ToTensorV2
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# Définir les graines aléatoires pour la reproductibilité
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.keras.utils.set_random_seed(SEED)


# ## Configuration des paramètres du modèle

# In[5]:


# Paramètres généraux
IMG_SIZE = 224  # Taille des images d'entrée
BATCH_SIZE = 32  # Taille des batchs
EPOCHS = 50  # Nombre d'époques maximum
LEARNING_RATE = 1e-4  # Taux d'apprentissage initial
NUM_CLASSES = 6  # Nombre de classes (catégories de prunes)
VALIDATION_SPLIT = 0.2  # Proportion des données pour la validation
TEST_SPLIT = 0.1  # Proportion des données pour le test
N_FOLDS = 5  # Nombre de folds pour la validation croisée
PATIENCE = 10  # Patience pour l'early stopping

# Chemins des données
DATA_DIR = 'african_plums'
MODELS_DIR = 'models'
LOGS_DIR = 'logs'

# Créer les répertoires nécessaires s'ils n'existent pas
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Noms des classes
CLASS_NAMES = ['bonne_qualite', 'non_mure', 'tachetee', 'fissuree', 'meurtrie', 'pourrie']


# ## Exploration et préparation des données

# In[6]:


def load_and_preprocess_data(data_dir, img_size):
    """
    Charge et prétraite les images du dataset.
    
    Args:
        data_dir: Chemin vers le répertoire des données
        img_size: Taille cible des images
        
    Returns:
        X: Images prétraitées
        y: Étiquettes correspondantes
        class_weights: Poids des classes pour gérer le déséquilibre
    """
    images = []
    labels = []
    
    # Parcourir chaque classe
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Le répertoire {class_dir} n'existe pas.")
            continue
            
        print(f"Chargement des images de la classe {class_name}...")
        class_files = os.listdir(class_dir)
        
        for file_name in class_files:
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, file_name)
                
                # Lire et prétraiter l'image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Impossible de lire l'image: {img_path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                img = img / 255.0  # Normalisation
                
                images.append(img)
                labels.append(idx)
    
    # Convertir en tableaux numpy
    X = np.array(images, dtype=np.float32)
    y = np.array(labels)
    
    # Calculer les poids des classes pour gérer le déséquilibre
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights = dict(enumerate(class_weights))
    
    print(f"Données chargées: {X.shape[0]} images, {len(np.unique(y))} classes")
    return X, y, class_weights


# In[7]:


def explore_dataset(X, y, class_names):
    """
    Explore le dataset et affiche des statistiques et visualisations.
    
    Args:
        X: Images prétraitées
        y: Étiquettes correspondantes
        class_names: Noms des classes
    """
    print(f"Forme des données: {X.shape}")
    print(f"Nombre total d'images: {X.shape[0]}")
    
    # Distribution des classes
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(class_names, counts))
    
    print("\nDistribution des classes:")
    for class_name, count in class_distribution.items():
        print(f"{class_name}: {count} images ({count/len(y)*100:.2f}%)")
    
    # Visualiser la distribution des classes
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_distribution.keys()), y=list(class_distribution.values()))
    plt.title('Distribution des classes')
    plt.xlabel('Classe')
    plt.ylabel('Nombre d\'images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Afficher quelques exemples d'images de chaque classe
    plt.figure(figsize=(15, 10))
    for i, class_idx in enumerate(unique):
        # Trouver les indices des images de cette classe
        indices = np.where(y == class_idx)[0]
        # Sélectionner aléatoirement 5 images (ou moins si moins disponibles)
        selected_indices = np.random.choice(indices, min(5, len(indices)), replace=False)
        
        for j, idx in enumerate(selected_indices):
            plt.subplot(len(unique), 5, i*5 + j + 1)
            plt.imshow(X[idx])
            plt.title(class_names[class_idx])
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# ## Augmentation de données et prétraitement

# In[8]:


def get_train_augmentations(img_size):
    """
    Définit les augmentations de données pour l'entraînement.
    
    Args:
        img_size: Taille cible des images
        
    Returns:
        Transformations d'augmentation pour l'entraînement
    """
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=3),
            A.GaussNoise(var_limit=(10, 50)),
            A.MotionBlur(blur_limit=3),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=img_size//8, max_width=img_size//8, min_holes=1, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_valid_augmentations(img_size):
    """
    Définit les transformations pour la validation et le test.
    
    Args:
        img_size: Taille cible des images
        
    Returns:
        Transformations pour la validation et le test
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# In[9]:


class PlumDataGenerator(tf.keras.utils.Sequence):
    """
    Générateur de données personnalisé pour les images de prunes avec augmentation.
    """
    def __init__(self, images, labels, batch_size, augmentations, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = [self.images[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        
        # Appliquer les augmentations
        X = np.array([self.augmentations(image=img)["image"] for img in batch_images])
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=NUM_CLASSES)
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# ## Architecture du modèle

# In[10]:


def create_model(model_name='efficientnet', img_size=224, num_classes=6, dropout_rate=0.3):
    """
    Crée un modèle de deep learning basé sur une architecture pré-entraînée.
    
    Args:
        model_name: Nom de l'architecture de base ('efficientnet', 'resnet', 'xception', 'densenet')
        img_size: Taille des images d'entrée
        num_classes: Nombre de classes
        dropout_rate: Taux de dropout pour la régularisation
        
    Returns:
        Modèle compilé
    """
    # Définir l'entrée
    inputs = Input(shape=(img_size, img_size, 3))
    
    # Sélectionner le modèle de base
    if model_name == 'efficientnet':
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_tensor=inputs)
    elif model_name == 'resnet':
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_tensor=inputs)
    elif model_name == 'xception':
        base_model = Xception(weights='imagenet', include_top=False, input_tensor=inputs)
    elif model_name == 'densenet':
        base_model = DenseNet201(weights='imagenet', include_top=False, input_tensor=inputs)
    else:
        raise ValueError(f"Modèle {model_name} non supporté")
    
    # Geler les couches du modèle de base
    for layer in base_model.layers:
        layer.trainable = False
    
    # Ajouter des couches personnalisées
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # Couches denses avec dropout et régularisation
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Couche de sortie avec activation softmax pour la classification multi-classes
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(1e-4))(x)
    
    # Créer le modèle
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compiler le modèle
    optimizer = tfa.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=1e-5
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tfa.metrics.F1Score(num_classes=num_classes, average='macro')
        ]
    )
    
    return model


# In[11]:


def create_ensemble_model(models, input_shape, num_classes):
    """
    Crée un modèle d'ensemble à partir de plusieurs modèles.
    
    Args:
        models: Liste des modèles à combiner
        input_shape: Forme des données d'entrée
        num_classes: Nombre de classes
        
    Returns:
        Modèle d'ensemble
    """
    # Créer une entrée commune
    input_layer = Input(shape=input_shape)
    
    # Obtenir les prédictions de chaque modèle
    outputs = [model(input_layer) for model in models]
    
    # Moyenne des prédictions
    ensemble_output = layers.Average()(outputs)
    
    # Créer le modèle d'ensemble
    ensemble_model = Model(inputs=input_layer, outputs=ensemble_output)
    
    # Compiler le modèle
    ensemble_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tfa.metrics.F1Score(num_classes=num_classes, average='macro')
        ]
    )
    
    return ensemble_model


# ## Entraînement avec validation croisée

# In[12]:


def train_with_cross_validation(X, y, n_folds=5, model_name='efficientnet', img_size=224, 
                               batch_size=32, epochs=50, class_weights=None):
    """
    Entraîne le modèle avec validation croisée.
    
    Args:
        X: Images prétraitées
        y: Étiquettes correspondantes
        n_folds: Nombre de folds pour la validation croisée
        model_name: Nom de l'architecture de base
        img_size: Taille des images d'entrée
        batch_size: Taille des batchs
        epochs: Nombre d'époques maximum
        class_weights: Poids des classes pour gérer le déséquilibre
        
    Returns:
        Liste des modèles entraînés, scores de validation
    """
    # Préparer la validation croisée stratifiée
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Initialiser les listes pour stocker les résultats
    fold_models = []
    fold_scores = []
    
    # Transformations d'augmentation
    train_aug = get_train_augmentations(img_size)
    valid_aug = get_valid_augmentations(img_size)
    
    # Boucle sur chaque fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n{'='*20} Fold {fold+1}/{n_folds} {'='*20}")
        
        # Diviser les données
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Créer les générateurs de données
        train_gen = PlumDataGenerator(X_train, y_train, batch_size, train_aug)
        val_gen = PlumDataGenerator(X_val, y_val, batch_size, valid_aug, shuffle=False)
        
        # Créer et compiler le modèle
        model = create_model(model_name=model_name, img_size=img_size, num_classes=NUM_CLASSES)
        
        # Définir les callbacks
        callbacks_list = [
            ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, f'model_{model_name}_fold{fold+1}.h5'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            CSVLogger(os.path.join(LOGS_DIR, f'training_log_fold{fold+1}.csv'))
        ]
        
        # Entraîner le modèle
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
        # Évaluer le modèle sur le fold de validation
        val_loss, val_acc, val_precision, val_recall, val_f1 = model.evaluate(val_gen, verbose=1)
        print(f"Fold {fold+1} - Validation Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")
        
        # Stocker le modèle et les scores
        fold_models.append(model)
        fold_scores.append({
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        })
        
        # Libérer la mémoire
        tf.keras.backend.clear_session()
        gc.collect()
    
    # Calculer et afficher les scores moyens
    avg_val_acc = np.mean([score['val_accuracy'] for score in fold_scores])
    avg_val_f1 = np.mean([score['val_f1'] for score in fold_scores])
    print(f"\nValidation croisée terminée - Accuracy moyenne: {avg_val_acc:.4f}, F1 Score moyen: {avg_val_f1:.4f}")
    
    return fold_models, fold_scores


# ## Techniques de robustesse avancées

# In[13]:


def apply_mixup(x, y, alpha=0.2):
    """
    Applique la technique de mixup aux données.
    
    Args:
        x: Images
        y: Étiquettes (one-hot encoded)
        alpha: Paramètre de la distribution beta
        
    Returns:
        Images et étiquettes mixées
    """
    batch_size = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Générer un lambda à partir d'une distribution beta
    lam = tfp.distributions.Beta(alpha, alpha).sample(1)[0]
    
    # Mélanger les images et les étiquettes
    mixed_x = lam * x + (1 - lam) * tf.gather(x, indices)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, indices)
    
    return mixed_x, mixed_y


# In[14]:


def apply_cutmix(x, y, alpha=1.0):
    """
    Applique la technique de cutmix aux données.
    
    Args:
        x: Images
        y: Étiquettes (one-hot encoded)
        alpha: Paramètre de la distribution beta
        
    Returns:
        Images et étiquettes mixées
    """
    batch_size = tf.shape(x)[0]
    image_height, image_width = x.shape[1], x.shape[2]
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Générer un lambda à partir d'une distribution beta
    lam = tfp.distributions.Beta(alpha, alpha).sample(1)[0]
    
    # Calculer les dimensions du rectangle à couper
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(image_height * cut_ratio, tf.int32)
    cut_w = tf.cast(image_width * cut_ratio, tf.int32)
    
    # Calculer les coordonnées du rectangle
    cx = tf.random.uniform([], 0, image_width, dtype=tf.int32)
    cy = tf.random.uniform([], 0, image_height, dtype=tf.int32)
    
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, image_width)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, image_height)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, image_width)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, image_height)
    
    # Créer un masque pour le rectangle
    mask = tf.ones((batch_size, image_height, image_width, 1))
    mask_y1 = tf.repeat(tf.expand_dims(tf.range(0, image_height), 0), batch_size, axis=0)
    mask_x1 = tf.repeat(tf.expand_dims(tf.range(0, image_width), 0), batch_size, axis=0)
    mask_y1 = tf.expand_dims(mask_y1, -1)
    mask_x1 = tf.expand_dims(mask_x1, 2)
    
    mask = tf.where((mask_y1 >= y1) & (mask_y1 < y2) & (mask_x1 >= x1) & (mask_x1 < x2),
                   tf.zeros_like(mask), mask)
    
    # Appliquer le masque aux images
    x1 = x * mask
    x2 = tf.gather(x, indices) * (1 - mask)
    mixed_x = x1 + x2
    
    # Calculer le ratio réel de mixage
    mixed_area = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
    total_area = tf.cast(image_height * image_width, tf.float32)
    mix_ratio = 1.0 - (mixed_area / total_area)
    
    # Mélanger les étiquettes
    mixed_y = mix_ratio * y + (1.0 - mix_ratio) * tf.gather(y, indices)
    
    return mixed_x, mixed_y


# In[15]:


class RobustPlumDataGenerator(tf.keras.utils.Sequence):
    """
    Générateur de données robuste avec techniques avancées d'augmentation.
    """
    def __init__(self, images, labels, batch_size, augmentations, 
                 use_mixup=True, use_cutmix=True, mixup_alpha=0.2, cutmix_alpha=1.0,
                 shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = [self.images[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        
        # Appliquer les augmentations
        X = np.array([self.augmentations(image=img)["image"] for img in batch_images])
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=NUM_CLASSES)
        
        # Appliquer mixup ou cutmix aléatoirement
        if self.use_mixup and self.use_cutmix:
            if np.random.random() < 0.5:
                X, y = apply_mixup(X, y, self.mixup_alpha)
            else:
                X, y = apply_cutmix(X, y, self.cutmix_alpha)
        elif self.use_mixup:
            X, y = apply_mixup(X, y, self.mixup_alpha)
        elif self.use_cutmix:
            X, y = apply_cutmix(X, y, self.cutmix_alpha)
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# ## Optimisation des hyperparamètres

# In[16]:


def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=20):
    """
    Optimise les hyperparamètres du modèle avec Optuna.
    
    Args:
        X_train: Images d'entraînement
        y_train: Étiquettes d'entraînement
        X_val: Images de validation
        y_val: Étiquettes de validation
        n_trials: Nombre d'essais d'optimisation
        
    Returns:
        Meilleurs hyperparamètres
    """
    import optuna
    from optuna.integration import TFKerasPruningCallback
    
    def objective(trial):
        # Hyperparamètres à optimiser
        model_name = trial.suggest_categorical('model_name', ['efficientnet', 'resnet', 'xception', 'densenet'])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        use_mixup = trial.suggest_categorical('use_mixup', [True, False])
        use_cutmix = trial.suggest_categorical('use_cutmix', [True, False])
        
        # Transformations d'augmentation
        train_aug = get_train_augmentations(IMG_SIZE)
        valid_aug = get_valid_augmentations(IMG_SIZE)
        
        # Créer les générateurs de données
        train_gen = RobustPlumDataGenerator(
            X_train, y_train, batch_size, train_aug,
            use_mixup=use_mixup, use_cutmix=use_cutmix
        )
        val_gen = PlumDataGenerator(X_val, y_val, batch_size, valid_aug, shuffle=False)
        
        # Créer et compiler le modèle
        model = create_model(
            model_name=model_name,
            img_size=IMG_SIZE,
            num_classes=NUM_CLASSES,
            dropout_rate=dropout_rate
        )
        
        # Mettre à jour l'optimiseur avec le taux d'apprentissage
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-5
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Définir les callbacks
        callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            TFKerasPruningCallback(trial, 'val_accuracy')
        ]
        
        # Entraîner le modèle
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,  # Réduire pour l'optimisation
            callbacks=callbacks_list,
            verbose=0
        )
        
        # Retourner la meilleure précision de validation
        return history.history['val_accuracy'][-1]
    
    # Créer l'étude Optuna
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Meilleurs hyperparamètres: {study.best_params}")
    print(f"Meilleure précision de validation: {study.best_value:.4f}")
    
    return study.best_params


# ## Évaluation du modèle

# In[17]:


def evaluate_model(model, X_test, y_test, class_names, batch_size=32):
    """
    Évalue le modèle sur l'ensemble de test.
    
    Args:
        model: Modèle entraîné
        X_test: Images de test
        y_test: Étiquettes de test
        class_names: Noms des classes
        batch_size: Taille des batchs
        
    Returns:
        Métriques d'évaluation
    """
    # Préparer les données de test
    test_aug = get_valid_augmentations(IMG_SIZE)
    test_gen = PlumDataGenerator(X_test, y_test, batch_size, test_aug, shuffle=False)
    
    # Évaluer le modèle
    test_loss, test_acc, test_precision, test_recall, test_f1 = model.evaluate(test_gen, verbose=1)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # Prédire les classes
    y_pred_proba = model.predict(test_gen)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_test
    
    # Rapport de classification
    print("\nRapport de classification:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes')
    plt.title('Matrice de confusion')
    plt.tight_layout()
    plt.show()
    
    # Visualiser quelques prédictions
    plt.figure(figsize=(15, 10))
    for i in range(min(15, len(X_test))):
        plt.subplot(3, 5, i+1)
        plt.imshow(X_test[i])
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"Vraie: {true_label}\nPréd: {pred_label}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    }


# ## Gestion des erreurs et robustesse

# In[18]:


def handle_errors_and_edge_cases(model, img_path, class_names, img_size=224):
    """
    Gère les erreurs et les cas limites lors de la prédiction.
    
    Args:
        model: Modèle entraîné
        img_path: Chemin vers l'image à prédire
        class_names: Noms des classes
        img_size: Taille des images d'entrée
        
    Returns:
        Prédiction et score de confiance
    """
    try:
        # Vérifier si le fichier existe
        if not os.path.exists(img_path):
            return "Erreur: Fichier non trouvé", 0.0
        
        # Lire l'image
        img = cv2.imread(img_path)
        if img is None:
            return "Erreur: Impossible de lire l'image", 0.0
        
        # Vérifier les dimensions de l'image
        if img.shape[0] < 10 or img.shape[1] < 10:
            return "Erreur: Image trop petite", 0.0
        
        # Prétraiter l'image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        
        # Normaliser l'image
        img = img / 255.0
        
        # Appliquer les transformations de validation
        valid_aug = get_valid_augmentations(img_size)
        img_processed = valid_aug(image=img)["image"]
        
        # Prédire avec le modèle
        img_batch = np.expand_dims(img_processed, axis=0)
        predictions = model.predict(img_batch)[0]
        
        # Obtenir la classe prédite et le score de confiance
        predicted_class_idx = np.argmax(predictions)
        confidence_score = predictions[predicted_class_idx]
        
        # Vérifier le seuil de confiance
        if confidence_score < 0.5:
            return "Incertain: Confiance faible", confidence_score
        
        return class_names[predicted_class_idx], confidence_score
        
    except Exception as e:
        return f"Erreur: {str(e)}", 0.0


# ## Fonction principale pour l'entraînement et l'évaluation

# In[19]:


def train_and_evaluate_robust_model():
    """
    Fonction principale pour entraîner et évaluer le modèle robuste.
    """
    try:
        print("Chargement et prétraitement des données...")
        X, y, class_weights = load_and_preprocess_data(DATA_DIR, IMG_SIZE)
        
        print("\nExploration du dataset...")
        explore_dataset(X, y, CLASS_NAMES)
        
        # Diviser les données en ensembles d'entraînement, de validation et de test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT, stratify=y, random_state=SEED
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=VALIDATION_SPLIT, stratify=y_train_val, random_state=SEED
        )
        
        print(f"Forme des données d'entraînement: {X_train.shape}")
        print(f"Forme des données de validation: {X_val.shape}")
        print(f"Forme des données de test: {X_test.shape}")
        
        # Optimiser les hyperparamètres (optionnel, peut être commenté pour gagner du temps)
        print("\nOptimisation des hyperparamètres...")
        best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=5)
        
        # Entraîner avec validation croisée
        print("\nEntraînement avec validation croisée...")
        fold_models, fold_scores = train_with_cross_validation(
            X_train_val, y_train_val, n_folds=N_FOLDS,
            model_name=best_params.get('model_name', 'efficientnet'),
            batch_size=best_params.get('batch_size', BATCH_SIZE),
            epochs=EPOCHS, class_weights=class_weights
        )
        
        # Créer un modèle d'ensemble
        print("\nCréation d'un modèle d'ensemble...")
        ensemble_model = create_ensemble_model(
            fold_models, (IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES
        )
        
        # Évaluer le modèle d'ensemble
        print("\nÉvaluation du modèle d'ensemble...")
        evaluation_metrics = evaluate_model(ensemble_model, X_test, y_test, CLASS_NAMES)
        
        # Sauvegarder le modèle final
        ensemble_model.save(os.path.join(MODELS_DIR, 'final_ensemble_model.h5'))
        print(f"Modèle final sauvegardé dans {os.path.join(MODELS_DIR, 'final_ensemble_model.h5')}")
        
        return ensemble_model, evaluation_metrics
        
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


# ## Démonstration et utilisation du modèle

# In[20]:


def predict_image(model, img_path, class_names, img_size=224):
    """
    Prédit la classe d'une image.
    
    Args:
        model: Modèle entraîné
        img_path: Chemin vers l'image à prédire
        class_names: Noms des classes
        img_size: Taille des images d'entrée
        
    Returns:
        Classe prédite et score de confiance
    """
    # Gérer les erreurs et les cas limites
    prediction, confidence = handle_errors_and_edge_cases(model, img_path, class_names, img_size)
    
    # Si une erreur s'est produite
    if isinstance(prediction, str) and prediction.startswith("Erreur"):
        return prediction, confidence
    
    return prediction, confidence


# In[21]:


def visualize_predictions(model, data_dir, class_names, num_samples=5, img_size=224):
    """
    Visualise les prédictions du modèle sur des échantillons aléatoires.
    
    Args:
        model: Modèle entraîné
        data_dir: Répertoire des données
        class_names: Noms des classes
        num_samples: Nombre d'échantillons à visualiser par classe
        img_size: Taille des images d'entrée
    """
    plt.figure(figsize=(15, 3 * len(class_names)))
    
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Le répertoire {class_dir} n'existe pas.")
            continue
            
        # Obtenir les fichiers d'images
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sélectionner des échantillons aléatoires
        selected_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        for j, file_name in enumerate(selected_files):
            img_path = os.path.join(class_dir, file_name)
            
            # Prédire la classe
            predicted_class, confidence = predict_image(model, img_path, class_names, img_size)
            
            # Afficher l'image et la prédiction
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(len(class_names), num_samples, i * num_samples + j + 1)
            plt.imshow(img)
            
            # Déterminer la couleur en fonction de la précision de la prédiction
            color = 'green' if predicted_class == class_name else 'red'
            
            plt.title(f"Vraie: {class_name}\nPréd: {predicted_class}\nConf: {confidence:.2f}", color=color)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# ## Interface de démonstration

# In[22]:


def create_demo_interface():
    """
    Crée une interface de démonstration simple pour le modèle.
    """
    from IPython.display import display, HTML, clear_output
    import ipywidgets as widgets
    
    # Charger le modèle
    try:
        model = load_model(os.path.join(MODELS_DIR, 'final_ensemble_model.h5'), compile=False)
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    except:
        print("Modèle non trouvé. Veuillez d'abord entraîner le modèle.")
        return
    
    # Créer les widgets
    file_upload = widgets.FileUpload(
        accept='.jpg, .jpeg, .png',
        multiple=False,
        description='Choisir une image'
    )
    
    output = widgets.Output()
    
    predict_button = widgets.Button(
        description='Prédire',
        button_style='primary',
        disabled=False
    )
    
    # Fonction de prédiction
    def on_predict_button_clicked(b):
        with output:
            clear_output()
            
            if not file_upload.value:
                print("Veuillez d'abord télécharger une image.")
                return
            
            # Récupérer l'image téléchargée
            uploaded_file = next(iter(file_upload.value.values()))
            content = uploaded_file['content']
            
            # Sauvegarder l'image temporairement
            temp_path = 'temp_image.jpg'
            with open(temp_path, 'wb') as f:
                f.write(content)
            
            # Prédire la classe
            predicted_class, confidence = predict_image(model, temp_path, CLASS_NAMES, IMG_SIZE)
            
            # Afficher l'image et la prédiction
            img = cv2.imread(temp_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(f"Prédiction: {predicted_class}\nConfiance: {confidence:.2f}")
            plt.axis('off')
            plt.show()
            
            # Supprimer le fichier temporaire
            os.remove(temp_path)
    
    # Associer la fonction au bouton
    predict_button.on_click(on_predict_button_clicked)
    
    # Afficher l'interface
    display(HTML("<h2>Démonstration du modèle de classification des prunes africaines</h2>"))
    display(HTML("<p>Téléchargez une image de prune pour obtenir une prédiction.</p>"))
    display(file_upload)
    display(predict_button)
    display(output)


# ## Exécution principale

# In[23]:


if __name__ == "__main__":
    # Exécuter l'entraînement et l'évaluation du modèle
    model, metrics = train_and_evaluate_robust_model()
    
    if model is not None:
        # Visualiser quelques prédictions
        visualize_predictions(model, DATA_DIR, CLASS_NAMES)
        
        # Créer l'interface de démonstration
        create_demo_interface()
