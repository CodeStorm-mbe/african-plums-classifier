import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import struct
import idx2numpy

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
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, self.labels[idx], image_path

class OtherCategoryDataset(Dataset):
    """
    Dataset personnalisé pour les images de la catégorie "autre".
    Combine des images de différentes sources (Intel, Fashion MNIST, etc.)
    """
    def __init__(self, image_paths, transform=None, fashion_mnist_data=None):
        """
        Initialise le dataset pour la catégorie "autre".
        
        Args:
            image_paths (list): Liste des chemins d'images
            transform (callable, optional): Transformations à appliquer aux images
            fashion_mnist_data (tuple, optional): Données Fashion MNIST (images, labels)
        """
        self.image_paths = image_paths
        self.transform = transform
        self.fashion_mnist_data = fashion_mnist_data
        self.use_fashion_mnist = fashion_mnist_data is not None
        
        # Si Fashion MNIST est inclus, calculer la taille totale
        if self.use_fashion_mnist:
            self.fashion_images, _ = fashion_mnist_data
            self.total_size = len(image_paths) + len(self.fashion_images)
        else:
            self.total_size = len(image_paths)
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # Déterminer si l'index correspond à une image de chemin ou à Fashion MNIST
        if self.use_fashion_mnist and idx >= len(self.image_paths):
            # Utiliser une image de Fashion MNIST
            fashion_idx = idx - len(self.image_paths)
            image = self.fashion_images[fashion_idx]
            
            # Convertir l'image Fashion MNIST en RGB (elle est en niveaux de gris)
            image = np.stack([image] * 3, axis=-1)
            
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
            return image, 6, "fashion_mnist"  # 6 est l'index de la catégorie "autre"
        else:
            # Utiliser une image de chemin
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
            return image, 6, image_path  # 6 est l'index de la catégorie "autre"

class DataPreprocessor:
    """
    Classe pour prétraiter les données d'images de prunes et d'autres catégories.
    """
    def __init__(self, data_dir, image_size=224, batch_size=32, num_workers=4):
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
        
        # Définition des catégories
        self.categories = {
            'unaffected': 0,  # bonne_qualite
            'unripe': 1,      # non_mure
            'spotted': 2,     # tachetee
            'cracked': 3,     # fissuree
            'bruised': 4,     # meurtrie
            'rotten': 5,      # pourrie
            'other': 6        # autre (nouvelle catégorie)
        }
        
        # Transformations pour l'entraînement
        self.train_transform = A.Compose([
            A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33), interpolation=0, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Transformations pour la validation et le test
        self.val_transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def load_fashion_mnist(self, fashion_dir):
        """
        Charge les données Fashion MNIST.
        
        Args:
            fashion_dir (str): Répertoire contenant les données Fashion MNIST
            
        Returns:
            tuple: (images, labels) pour Fashion MNIST
        """
        # Chemins des fichiers
        train_images_path = os.path.join(fashion_dir, 'train-images-idx3-ubyte')
        train_labels_path = os.path.join(fashion_dir, 'train-labels-idx1-ubyte')
        test_images_path = os.path.join(fashion_dir, 't10k-images-idx3-ubyte')
        test_labels_path = os.path.join(fashion_dir, 't10k-labels-idx1-ubyte')
        
        # Charger les images et les labels
        try:
            # Utiliser idx2numpy pour charger les fichiers IDX
            train_images = idx2numpy.convert_from_file(train_images_path)
            train_labels = idx2numpy.convert_from_file(train_labels_path)
            test_images = idx2numpy.convert_from_file(test_images_path)
            test_labels = idx2numpy.convert_from_file(test_labels_path)
            
            # Combiner les ensembles d'entraînement et de test
            all_images = np.vstack([train_images, test_images])
            all_labels = np.hstack([train_labels, test_labels])
            
            print(f"Données Fashion MNIST chargées: {all_images.shape} images")
            return all_images, all_labels
            
        except Exception as e:
            print(f"Erreur lors du chargement des données Fashion MNIST: {e}")
            
            # Alternative: charger à partir des fichiers CSV
            try:
                train_csv_path = os.path.join(fashion_dir, 'fashion-mnist_train.csv')
                test_csv_path = os.path.join(fashion_dir, 'fashion-mnist_test.csv')
                
                train_df = pd.read_csv(train_csv_path)
                test_df = pd.read_csv(test_csv_path)
                
                # Extraire les labels et les images
                train_labels = train_df.iloc[:, 0].values
                train_images = train_df.iloc[:, 1:].values.reshape(-1, 28, 28)
                
                test_labels = test_df.iloc[:, 0].values
                test_images = test_df.iloc[:, 1:].values.reshape(-1, 28, 28)
                
                # Combiner les ensembles d'entraînement et de test
                all_images = np.vstack([train_images, test_images])
                all_labels = np.hstack([train_labels, test_labels])
                
                print(f"Données Fashion MNIST chargées depuis CSV: {all_images.shape} images")
                return all_images, all_labels
                
            except Exception as e2:
                print(f"Erreur lors du chargement des données Fashion MNIST depuis CSV: {e2}")
                return None, None
    
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
            if category == 'other':
                continue  # Ignorer la catégorie "autre" pour les prunes
                
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
    
    def collect_other_images(self, intel_dir, max_per_category=500):
        """
        Collecte les images pour la catégorie "autre" à partir du dataset Intel.
        
        Args:
            intel_dir (str): Répertoire contenant les images Intel
            max_per_category (int): Nombre maximum d'images à collecter par catégorie
            
        Returns:
            list: Chemins des images pour la catégorie "autre"
        """
        other_image_paths = []
        
        # Vérifier si le répertoire seg_train existe
        seg_train_dir = os.path.join(intel_dir, 'seg_train', 'seg_train')
        if not os.path.exists(seg_train_dir):
            print(f"Répertoire {seg_train_dir} non trouvé, vérification des alternatives...")
            
            # Vérifier les alternatives
            alt_seg_train_dir = os.path.join(intel_dir, 'seg_train')
            if os.path.exists(alt_seg_train_dir) and os.path.isdir(alt_seg_train_dir):
                # Vérifier s'il contient des sous-dossiers de catégories
                subdirs = [d for d in os.listdir(alt_seg_train_dir) 
                          if os.path.isdir(os.path.join(alt_seg_train_dir, d))]
                
                if any(d in ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'] for d in subdirs):
                    seg_train_dir = alt_seg_train_dir
                    print(f"Utilisation du répertoire alternatif: {seg_train_dir}")
                else:
                    # Chercher plus profondément
                    for subdir in subdirs:
                        potential_dir = os.path.join(alt_seg_train_dir, subdir)
                        sub_subdirs = [d for d in os.listdir(potential_dir) 
                                      if os.path.isdir(os.path.join(potential_dir, d))]
                        
                        if any(d in ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'] for d in sub_subdirs):
                            seg_train_dir = potential_dir
                            print(f"Utilisation du répertoire alternatif: {seg_train_dir}")
                            break
        
        # Parcourir les catégories Intel
        intel_categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        for category in intel_categories:
            category_dir = os.path.join(seg_train_dir, category)
            if not os.path.exists(category_dir):
                print(f"Catégorie Intel {category} non trouvée, ignorée.")
                continue
            
            # Collecter les images de cette catégorie
            img_paths = [os.path.join(category_dir, img_name) 
                        for img_name in os.listdir(category_dir)
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limiter le nombre d'images par catégorie
            if len(img_paths) > max_per_category:
                img_paths = np.random.choice(img_paths, max_per_category, replace=False).tolist()
            
            other_image_paths.extend(img_paths)
            print(f"Collecté {len(img_paths)} images de la catégorie Intel {category}")
        
        print(f"Total de {len(other_image_paths)} images Intel pour la catégorie 'autre'")
        return other_image_paths
    
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
        
        # Calculer la distribution des classes
        class_counts = {}
        for label in labels:
            category = list(self.categories.keys())[list(self.categories.values()).index(label)]
            if category not in class_counts:
                class_counts[category] = 0
            class_counts[category] += 1
        
        # Calculer les dimensions moyennes
        widths = []
        heights = []
        aspect_ratios = []
        
        for img_path in image_paths[:min(1000, len(image_paths))]:  # Limiter pour des raisons de performance
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    widths.append(width)
                    heights.append(height)
                    aspect_ratios.append(width / height)
            except Exception as e:
                print(f"Erreur lors de l'ouverture de l'image {img_path}: {e}")
        
        avg_width = np.mean(widths) if widths else 0
        avg_height = np.mean(heights) if heights else 0
        avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 0
        
        # Générer des visualisations
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Distribution des classes
            plt.figure(figsize=(12, 6))
            categories = list(class_counts.keys())
            counts = list(class_counts.values())
            plt.bar(categories, counts)
            plt.title('Distribution des classes')
            plt.xlabel('Catégorie')
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
            'class_counts': class_counts,
            'avg_width': avg_width,
            'avg_height': avg_height,
            'avg_aspect_ratio': avg_aspect_ratio
        }
        
        return stats
    
    def prepare_data(self, test_size=0.2, val_size=0.1, negative_dir=None):
        """
        Prépare les données pour l'entraînement, la validation et le test.
        
        Args:
            test_size (float): Proportion des données pour l'ensemble de test
            val_size (float): Proportion des données pour l'ensemble de validation
            negative_dir (str, optional): Répertoire contenant des images négatives
            
        Returns:
            dict: Dictionnaire contenant les DataLoaders et les datasets
        """
        # Recherche du répertoire des prunes
        plum_dir = os.path.join(self.data_dir, 'extracted', 'african-plums-quality-and-defect-assessment-data')
        if os.path.exists(os.path.join(plum_dir, 'african_plums_dataset', 'african_plums')):
            plum_dir = os.path.join(plum_dir, 'african_plums_dataset', 'african_plums')
        
        # Collecter les images de prunes
        plum_image_paths, plum_labels = self.collect_plum_images(plum_dir)
        
        # Collecter les images pour la catégorie "autre"
        other_image_paths = []
        
        # Ajouter les images Intel
        intel_dir = os.path.join(self.data_dir, 'extracted', 'intel-image-classification')
        if os.path.exists(intel_dir):
            other_image_paths.extend(self.collect_other_images(intel_dir))
        
        # Charger les données Fashion MNIST
        fashion_dir = os.path.join(self.data_dir, 'extracted', 'fashionmnist')
        fashion_data = None
        if os.path.exists(fashion_dir):
            fashion_images, fashion_labels = self.load_fashion_mnist(fashion_dir)
            if fashion_images is not None:
                fashion_data = (fashion_images, fashion_labels)
        
        # Diviser les données de prunes en ensembles d'entraînement, de validation et de test
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            plum_image_paths, plum_labels, test_size=test_size, random_state=42, stratify=plum_labels
        )
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=val_size/(1-test_size), random_state=42, stratify=train_labels
        )
        
        # Diviser les images "autre" en ensembles d'entraînement, de validation et de test
        if other_image_paths:
            other_train, other_test = train_test_split(
                other_image_paths, test_size=test_size, random_state=42
            )
            
            other_train, other_val = train_test_split(
                other_train, test_size=val_size/(1-test_size), random_state=42
            )
        else:
            other_train, other_val, other_test = [], [], []
        
        # Diviser les données Fashion MNIST si disponibles
        fashion_train_data, fashion_val_data, fashion_test_data = None, None, None
        if fashion_data is not None:
            fashion_images, fashion_labels = fashion_data
            
            # Diviser les indices
            indices = np.arange(len(fashion_images))
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=42
            )
            
            train_idx, val_idx = train_test_split(
                train_idx, test_size=val_size/(1-test_size), random_state=42
            )
            
            # Créer les sous-ensembles
            fashion_train_data = (fashion_images[train_idx], fashion_labels[train_idx])
            fashion_val_data = (fashion_images[val_idx], fashion_labels[val_idx])
            fashion_test_data = (fashion_images[test_idx], fashion_labels[test_idx])
        
        # Créer les datasets
        train_dataset = PlumDataset(train_paths, train_labels, transform=self.train_transform)
        val_dataset = PlumDataset(val_paths, val_labels, transform=self.val_transform)
        test_dataset = PlumDataset(test_paths, test_labels, transform=self.val_transform)
        
        # Créer les datasets pour la catégorie "autre"
        other_train_dataset = OtherCategoryDataset(other_train, transform=self.train_transform, fashion_mnist_data=fashion_train_data)
        other_val_dataset = OtherCategoryDataset(other_val, transform=self.val_transform, fashion_mnist_data=fashion_val_data)
        other_test_dataset = OtherCategoryDataset(other_test, transform=self.val_transform, fashion_mnist_data=fashion_test_data)
        
        # Créer les DataLoaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        # Créer les DataLoaders pour la catégorie "autre"
        other_train_dataloader = DataLoader(
            other_train_dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        other_val_dataloader = DataLoader(
            other_val_dataset, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        other_test_dataloader = DataLoader(
            other_test_dataset, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        # Calculer la distribution des classes
        class_counts = {}
        for label in plum_labels:
            category = list(self.categories.keys())[list(self.categories.values()).index(label)]
            if category not in class_counts:
                class_counts[category] = 0
            class_counts[category] += 1
        
        # Ajouter la catégorie "autre"
        class_counts['other'] = len(other_image_paths)
        if fashion_data is not None:
            class_counts['other'] += len(fashion_data[0])
        
        # Retourner les données préparées
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'train_dataloader': train_dataloader,
            'val_dataloader': val_dataloader,
            'test_dataloader': test_dataloader,
            'other_train_dataset': other_train_dataset,
            'other_val_dataset': other_val_dataset,
            'other_test_dataset': other_test_dataset,
            'other_train_dataloader': other_train_dataloader,
            'other_val_dataloader': other_val_dataloader,
            'other_test_dataloader': other_test_dataloader,
            'class_counts': class_counts
        }
    
    def get_class_weights(self, class_counts):
        """
        Calcule les poids des classes pour gérer le déséquilibre.
        
        Args:
            class_counts (dict): Nombre d'exemples par classe
            
        Returns:
            torch.Tensor: Poids des classes
        """
        # Convertir le dictionnaire en liste ordonnée selon les indices des catégories
        counts = []
        for category, idx in self.categories.items():
            if category in class_counts:
                counts.append(class_counts[category])
            else:
                counts.append(0)
        
        # Calculer les poids inversement proportionnels aux fréquences
        weights = 1.0 / torch.tensor(counts, dtype=torch.float)
        weights = weights / weights.sum() * len(counts)
        
        return weights
    
    def visualize_batch(self, dataloader, save_path=None):
        """
        Visualise un batch d'images.
        
        Args:
            dataloader (DataLoader): DataLoader contenant les images
            save_path (str, optional): Chemin où sauvegarder la visualisation
        """
        # Obtenir un batch
        images, labels, _ = next(iter(dataloader))
        
        # Convertir les tensors en numpy arrays
        images = images.numpy()
        labels = labels.numpy()
        
        # Dénormaliser les images
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        images = images * std + mean
        images = np.clip(images, 0, 1)
        
        # Créer la grille d'images
        n = min(16, len(images))
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(n):
            img = images[i].transpose(1, 2, 0)
            label = labels[i]
            
            # Obtenir le nom de la catégorie
            category = list(self.categories.keys())[list(self.categories.values()).index(label)]
            
            axes[i].imshow(img)
            axes[i].set_title(category)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def combine_dataloaders(self, plum_dataloader, other_dataloader, ratio=0.5):
        """
        Combine les DataLoaders de prunes et de la catégorie "autre".
        
        Args:
            plum_dataloader (DataLoader): DataLoader des prunes
            other_dataloader (DataLoader): DataLoader de la catégorie "autre"
            ratio (float): Ratio d'images "autre" par rapport aux prunes
            
        Returns:
            DataLoader: DataLoader combiné
        """
        # Créer un dataset combiné
        combined_dataset = CombinedDataset(plum_dataloader.dataset, other_dataloader.dataset, ratio)
        
        # Créer un DataLoader pour le dataset combiné
        combined_dataloader = DataLoader(
            combined_dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        return combined_dataloader

class CombinedDataset(Dataset):
    """
    Dataset combinant les prunes et la catégorie "autre".
    """
    def __init__(self, plum_dataset, other_dataset, ratio=0.5):
        """
        Initialise le dataset combiné.
        
        Args:
            plum_dataset (Dataset): Dataset des prunes
            other_dataset (Dataset): Dataset de la catégorie "autre"
            ratio (float): Ratio d'images "autre" par rapport aux prunes
        """
        self.plum_dataset = plum_dataset
        self.other_dataset = other_dataset
        self.ratio = ratio
        
        # Calculer le nombre d'images "autre" à utiliser
        self.num_plums = len(plum_dataset)
        self.num_other = min(len(other_dataset), int(self.num_plums * ratio))
        
        # Indices aléatoires pour les images "autre"
        self.other_indices = np.random.choice(len(other_dataset), self.num_other, replace=False)
    
    def __len__(self):
        return self.num_plums + self.num_other
    
    def __getitem__(self, idx):
        if idx < self.num_plums:
            # Retourner une image de prune
            return self.plum_dataset[idx]
        else:
            # Retourner une image "autre"
            other_idx = self.other_indices[idx - self.num_plums]
            return self.other_dataset[other_idx]
