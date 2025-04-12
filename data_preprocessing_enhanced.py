import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from albumentations.augmentations.blur import GaussianBlur, MotionBlur
from albumentations.augmentations.transforms import GaussNoise, ColorJitter
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, Affine, ElasticTransform
from albumentations.augmentations.crops.transforms import RandomResizedCrop

import random

class KaggleDatasetDownloader:
    """
    Classe pour télécharger et extraire des datasets depuis Kaggle.
    """
    def __init__(self, dataset_id, data_dir):
        """
        Initialise le téléchargeur de dataset Kaggle.
        
        Args:
            dataset_id (str): ID du dataset Kaggle (format: 'username/dataset-name')
            data_dir (str): Répertoire où stocker les données
        """
        self.dataset_id = dataset_id
        self.data_dir = data_dir
        self.extracted_dir = os.path.join(data_dir, 'extracted')
        
        # Création des répertoires nécessaires
        os.makedirs(self.extracted_dir, exist_ok=True)
    
    def download_and_extract(self):
        """
        Télécharge et extrait le dataset depuis Kaggle.
        
        Returns:
            str: Chemin vers le répertoire contenant les données extraites
        """
        import kaggle
        import zipfile
        
        # Téléchargement du dataset
        print(f"Téléchargement du dataset {self.dataset_id}...")
        kaggle.api.dataset_download_files(self.dataset_id, path=self.data_dir)
        
        # Extraction du dataset
        zip_file = os.path.join(self.data_dir, f"{self.dataset_id.split('/')[-1]}.zip")
        extract_dir = os.path.join(self.extracted_dir, self.dataset_id.split('/')[-1])
        
        print(f"Extraction du dataset dans {extract_dir}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        return extract_dir

class CutMix:
    """
    Implémentation de la technique CutMix pour l'augmentation de données.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch):
        images, labels, paths = batch
        
        # Générer un lambda à partir d'une distribution Beta
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mélanger les indices
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        # Générer les coordonnées du rectangle à couper
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        # Appliquer CutMix
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        
        # Ajuster les labels en fonction de la proportion de l'image originale
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        
        # Créer des labels mixés en utilisant one-hot encoding
        one_hot_labels = torch.zeros(batch_size, 6, device=images.device)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
        
        one_hot_labels_mixed = lam * one_hot_labels + (1 - lam) * one_hot_labels[index]
        
        return images, one_hot_labels_mixed, paths
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        # Coordonnées uniformes
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

# Implémentation alternative de GridMask sans hériter de A.DualTransform
class CustomGridMask:
    """
    Implémentation personnalisée de GridMask pour l'augmentation de données.
    """
    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, p=0.5):
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.p = p
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid**2
            for i in range(n_masks):
                mask = np.ones((height, width), np.float32)
                w = int(width / self.num_grid)
                h = int(height / self.num_grid)
                y = i // self.num_grid
                x = i % self.num_grid
                mask[y*h:(y+1)*h, x*w:(x+1)*w] = self.fill_value
                self.masks.append(mask)
            if self.rotate:
                self.masks = [self._rotate_mask(mask, angle) for mask in self.masks for angle in range(0, 360, self.rotate)]
            self.rand_h_max = [mask.shape[0] - height for mask in self.masks]
            self.rand_w_max = [mask.shape[1] - width for mask in self.masks]

    def _rotate_mask(self, mask, angle):
        center = (mask.shape[0] // 2, mask.shape[1] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(mask, rot_mat, (mask.shape[1], mask.shape[0]))

    def __call__(self, image):
        if random.random() > self.p:
            return image
            
        h, w = image.shape[:2]
        self.init_masks(h, w)
        
        mask_idx = random.randint(0, len(self.masks) - 1)
        mask = self.masks[mask_idx]
        rand_h = 0
        rand_w = 0
        if self.rand_h_max[mask_idx] > 0:
            rand_h = np.random.randint(0, self.rand_h_max[mask_idx])
        if self.rand_w_max[mask_idx] > 0:
            rand_w = np.random.randint(0, self.rand_w_max[mask_idx])
        mask = mask[rand_h:rand_h + h, rand_w:rand_w + w]
        
        if self.mode == 0:
            return image * mask[:, :, np.newaxis]
        else:
            return image * (1 - mask[:, :, np.newaxis]) + self.fill_value * mask[:, :, np.newaxis]

class PlumDataset(Dataset):
    """
    Dataset personnalisé pour les images de prunes.
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialise le dataset.
        
        Args:
            image_paths (list): Liste des chemins d'images
            labels (list): Liste des étiquettes correspondantes
            transform (callable, optional): Transformations à appliquer aux images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
            # Vérification optionnelle pour le débogage
            # if image.shape[1] != 320 or image.shape[2] != 320:
            #     print(f"Image de taille incorrecte: {image.shape} pour {image_path}")
            
            return image, self.labels[idx], image_path
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {image_path}: {e}")
            # Retourner une image noire de la BONNE taille en cas d'erreur
            if self.transform:
                dummy_image = np.zeros((320, 320, 3), dtype=np.uint8)
                augmented = self.transform(image=dummy_image)
                return augmented['image'], self.labels[idx], image_path
            return torch.zeros((3, 320, 320)), self.labels[idx], image_path


class DataPreprocessor:
    """
    Classe pour prétraiter les données d'images de prunes.
    """
    def __init__(self, data_dir, image_size=320, batch_size=16, num_workers=2):
        """
        Initialise le préprocesseur de données.
        
        Args:
            data_dir (str): Répertoire contenant les données
            image_size (int): Taille des images après redimensionnement
            batch_size (int): Taille des batchs pour les DataLoaders
            num_workers (int): Nombre de workers pour les DataLoaders
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Définition des catégories (sans la catégorie "autre")
        self.categories = {
            'unaffected': 0,  # bonne_qualite
            'unripe': 1,      # non_mure
            'spotted': 2,     # tachetee
            'cracked': 3,     # fissuree
            'bruised': 4,     # meurtrie
            'rotten': 5       # pourrie
        }
        
        # Transformations avancées pour l'entraînement
        self.train_transform = A.Compose([
            A.Resize(height=320, width=320),  # Taille fixe pour toutes les images

            A.OneOf([
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), ratio=(0.75, 1.33), p=1.0),
                A.Sequential([
                    A.Resize(height=int(image_size*1.1), width=int(image_size*1.1)),
                    A.CenterCrop(height=image_size, width=image_size)  # Assurez-vous que la sortie a toujours la même taille
                ], p=1.0),
            ], p=1.0),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ], p=0.7),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.ElasticTransform(p=0.2),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Emboss(p=0.3),
                A.Sharpen(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
                A.ChannelShuffle(p=0.1),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ], p=0.8),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ], p=0.5),
            A.OneOf([
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.3
                ),
                # Vous pouvez ajouter d'autres transformations ici
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, border_mode=0, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Transformations pour la validation et le test
        self.val_transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Transformations pour le test-time augmentation (TTA)
        self.tta_transforms = [
            A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.RandomRotate90(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.Transpose(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
        ]
    
    def collect_plum_images(self, plum_dir):
        """
        Collecte les images de prunes et leurs étiquettes.
        
        Args:
            plum_dir (str): Répertoire contenant les images de prunes
            
        Returns:
            tuple: (image_paths, labels) pour les prunes
        """
        image_paths = []
        labels = []
        
        # Parcourir les catégories de prunes
        for category in self.categories.keys():
            category_dir = os.path.join(plum_dir, category)
            if not os.path.exists(category_dir):
                print(f"Répertoire {category_dir} non trouvé, vérification des alternatives...")
                
                # Vérifier les alternatives (structure différente)
                alt_category_dir = os.path.join(plum_dir, 'african_plums_dataset', 'african_plums', category)
                if os.path.exists(alt_category_dir):
                    category_dir = alt_category_dir
                    print(f"Utilisation du répertoire alternatif: {category_dir}")
                else:
                    print(f"Catégorie {category} non trouvée, ignorée.")
                    continue
            
            # Collecter les images de cette catégorie
            for img_name in os.listdir(category_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(category_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(self.categories[category])
        
        print(f"Collecté {len(image_paths)} images de prunes dans {len(set(labels))} catégories")
        return image_paths, labels
    
    def analyze_dataset(self, save_dir=None):
        """
        Analyse le dataset et génère des visualisations.
        
        Args:
            save_dir (str, optional): Répertoire où sauvegarder les visualisations
            
        Returns:
            dict: Statistiques du dataset
        """
        # Recherche du répertoire des prunes
        plum_dir = os.path.join(self.data_dir, 'extracted', 'african-plums-quality-and-defect-assessment-data')
        if os.path.exists(os.path.join(plum_dir, 'african_plums_dataset', 'african_plums')):
            plum_dir = os.path.join(plum_dir, 'african_plums_dataset', 'african_plums')
        
        # Collecter les images de prunes
        image_paths, labels = self.collect_plum_images(plum_dir)
        
        # Calculer les statistiques
        num_images = len(image_paths)
        num_classes = len(set(labels))
        
        # Calculer les dimensions moyennes
        widths = []
        heights = []
        aspect_ratios = []
        
        for img_path in image_paths[:100]:  # Limiter à 100 images pour la performance
            try:
                img = Image.open(img_path)
                width, height = img.size
                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)
            except Exception as e:
                print(f"Erreur lors de l'ouverture de l'image {img_path}: {e}")
        
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)
        avg_aspect_ratio = np.mean(aspect_ratios)
        
        # Calculer la distribution des classes
        class_counts = {}
        for label in labels:
            class_name = list(self.categories.keys())[list(self.categories.values()).index(label)]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        # Générer des visualisations
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Distribution des classes
            plt.figure(figsize=(12, 6))
            plt.bar(class_counts.keys(), class_counts.values())
            plt.title('Distribution des classes')
            plt.xlabel('Classe')
            plt.ylabel('Nombre d\'images')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
            plt.close()
            
            # Distribution des dimensions
            plt.figure(figsize=(12, 6))
            plt.scatter(widths, heights, alpha=0.5)
            plt.title('Distribution des dimensions')
            plt.xlabel('Largeur (pixels)')
            plt.ylabel('Hauteur (pixels)')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'dimension_distribution.png'))
            plt.close()
        
        # Retourner les statistiques
        stats = {
            'num_images': num_images,
            'num_classes': num_classes,
            'avg_width': avg_width,
            'avg_height': avg_height,
            'avg_aspect_ratio': avg_aspect_ratio,
            'class_counts': class_counts
        }
        
        print(f"Statistiques du dataset:")
        print(f"- Nombre d'images: {num_images}")
        print(f"- Nombre de classes: {num_classes}")
        print(f"- Dimensions moyennes: {avg_width:.1f} x {avg_height:.1f} pixels")
        print(f"- Ratio d'aspect moyen: {avg_aspect_ratio:.2f}")
        print(f"- Distribution des classes: {class_counts}")
        
        return stats
    
    def prepare_data(self, test_size=0.2, val_size=0.1, random_state=42, use_stratified=True, use_weighted_sampler=True):
        """
        Prépare les données pour l'entraînement, la validation et le test.
        
        Args:
            test_size (float): Proportion des données pour le test
            val_size (float): Proportion des données pour la validation
            random_state (int): Graine aléatoire pour la reproductibilité
            use_stratified (bool): Utiliser une stratification pour la division des données
            use_weighted_sampler (bool): Utiliser un échantillonneur pondéré pour gérer le déséquilibre des classes
            
        Returns:
            dict: Dictionnaire contenant les DataLoaders et les datasets
        """
        # Recherche du répertoire des prunes
        plum_dir = os.path.join(self.data_dir, 'extracted', 'african-plums-quality-and-defect-assessment-data')
        if os.path.exists(os.path.join(plum_dir, 'african_plums_dataset', 'african_plums')):
            plum_dir = os.path.join(plum_dir, 'african_plums_dataset', 'african_plums')
        
        # Collecter les images de prunes
        plum_image_paths, plum_labels = self.collect_plum_images(plum_dir)
        
        # Diviser les données en ensembles d'entraînement, de validation et de test
        if use_stratified:
            # Première division: train+val vs test
            train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
                plum_image_paths, plum_labels, test_size=test_size, random_state=random_state, stratify=plum_labels
            )
            
            # Deuxième division: train vs val
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_val_paths, train_val_labels, 
                test_size=val_size/(1-test_size),  # Ajustement pour obtenir la bonne proportion
                random_state=random_state, 
                stratify=train_val_labels
            )
        else:
            # Division standard sans stratification
            train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
                plum_image_paths, plum_labels, test_size=test_size, random_state=random_state
            )
            
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_val_paths, train_val_labels, 
                test_size=val_size/(1-test_size),
                random_state=random_state
            )
        
        # Créer les datasets
        train_dataset = PlumDataset(train_paths, train_labels, transform=self.train_transform)
        val_dataset = PlumDataset(val_paths, val_labels, transform=self.val_transform)
        test_dataset = PlumDataset(test_paths, test_labels, transform=self.val_transform)
        
        # Calculer les poids pour l'échantillonneur pondéré
        if use_weighted_sampler:
            class_sample_counts = np.bincount(train_labels)
            class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
            sample_weights = class_weights[train_labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_labels),
                replacement=True
            )
            shuffle = False  # Ne pas mélanger si on utilise un sampler
        else:
            sampler = None
            shuffle = True
        
        # Créer les DataLoaders
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Calculer la distribution des classes
        class_counts = {}
        for label in plum_labels:
            class_name = list(self.categories.keys())[list(self.categories.values()).index(label)]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        # Retourner les données
        data = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'train_dataloader': train_dataloader,
            'val_dataloader': val_dataloader,
            'test_dataloader': test_dataloader,
            'class_counts': class_counts,
            'train_paths': train_paths,
            'val_paths': val_paths,
            'test_paths': test_paths,
            'train_labels': train_labels,
            'val_labels': val_labels,
            'test_labels': test_labels
        }
        
        print(f"Données préparées:")
        print(f"- Ensemble d'entraînement: {len(train_dataset)} images")
        print(f"- Ensemble de validation: {len(val_dataset)} images")
        print(f"- Ensemble de test: {len(test_dataset)} images")
        
        return data
    
    def get_class_weights(self, class_counts):
        """
        Calcule les poids des classes pour gérer le déséquilibre.
        
        Args:
            class_counts (dict): Nombre d'images par classe
            
        Returns:
            torch.Tensor: Poids des classes
        """
        # Convertir les noms de classes en indices
        class_indices = {self.categories[name]: count for name, count in class_counts.items()}
        
        # Calculer les poids inversement proportionnels à la fréquence
        total_samples = sum(class_counts.values())
        weights = []
        
        for i in range(len(self.categories)):
            if i in class_indices:
                weight = total_samples / (len(self.categories) * class_indices[i])
            else:
                weight = 1.0  # Valeur par défaut pour les classes sans échantillons
            weights.append(weight)
        
        # Normaliser les poids
        weights = np.array(weights)
        weights = weights / np.sum(weights) * len(weights)
        
        return torch.FloatTensor(weights)
    
    def visualize_batch(self, dataloader, save_path=None):
        """
        Visualise un batch d'images.
        
        Args:
            dataloader (DataLoader): DataLoader contenant les images
            save_path (str, optional): Chemin où sauvegarder la visualisation
        """
        # Récupérer un batch
        images, labels, _ = next(iter(dataloader))
        
        # Convertir les tenseurs en images
        images = images.numpy().transpose(0, 2, 3, 1)
        
        # Dénormaliser les images
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        images = std * images + mean
        images = np.clip(images, 0, 1)
        
        # Afficher les images
        plt.figure(figsize=(20, 10))
        for i in range(min(16, len(images))):
            plt.subplot(4, 4, i+1)
            plt.imshow(images[i])
            class_idx = labels[i].item()
            class_name = list(self.categories.keys())[list(self.categories.values()).index(class_idx)]
            plt.title(f"{class_name} (idx: {class_idx})")
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualisation du batch sauvegardée dans {save_path}")
        
        plt.close()
    
    def create_cross_validation_folds(self, n_splits=5, random_state=42):
        """
        Crée des plis pour la validation croisée.
        
        Args:
            n_splits (int): Nombre de plis
            random_state (int): Graine aléatoire pour la reproductibilité
            
        Returns:
            list: Liste de dictionnaires contenant les indices d'entraînement et de validation pour chaque pli
        """
        # Recherche du répertoire des prunes
        plum_dir = os.path.join(self.data_dir, 'extracted', 'african-plums-quality-and-defect-assessment-data')
        if os.path.exists(os.path.join(plum_dir, 'african_plums_dataset', 'african_plums')):
            plum_dir = os.path.join(plum_dir, 'african_plums_dataset', 'african_plums')
        
        # Collecter les images de prunes
        plum_image_paths, plum_labels = self.collect_plum_images(plum_dir)
        
        # Initialiser la validation croisée stratifiée
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Créer les plis
        folds = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(plum_image_paths, plum_labels)):
            # Extraire les chemins et les labels pour ce pli
            train_paths = [plum_image_paths[i] for i in train_idx]
            train_labels = [plum_labels[i] for i in train_idx]
            val_paths = [plum_image_paths[i] for i in val_idx]
            val_labels = [plum_labels[i] for i in val_idx]
            
            # Créer les datasets
            train_dataset = PlumDataset(train_paths, train_labels, transform=self.train_transform)
            val_dataset = PlumDataset(val_paths, val_labels, transform=self.val_transform)
            
            # Calculer les poids des classes pour ce pli
            class_counts = {}
            for label in train_labels:
                class_name = list(self.categories.keys())[list(self.categories.values()).index(label)]
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
            
            class_weights = self.get_class_weights(class_counts)
            
            # Calculer les poids pour l'échantillonneur pondéré
            class_sample_counts = np.bincount(train_labels)
            class_weights_sampler = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
            sample_weights = class_weights_sampler[train_labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_labels),
                replacement=True
            )
            
            # Créer les DataLoaders
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )
            
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            # Ajouter les données de ce pli à la liste
            fold_data = {
                'fold': fold,
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'train_dataloader': train_dataloader,
                'val_dataloader': val_dataloader,
                'class_weights': class_weights,
                'train_paths': train_paths,
                'val_paths': val_paths,
                'train_labels': train_labels,
                'val_labels': val_labels
            }
            
            folds.append(fold_data)
            
            print(f"Pli {fold}:")
            print(f"- Ensemble d'entraînement: {len(train_dataset)} images")
            print(f"- Ensemble de validation: {len(val_dataset)} images")
        
        return folds
