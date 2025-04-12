import os
import sys
import unittest
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil

# Ajout du répertoire courant au path pour permettre l'importation des modules
sys.path.append(os.getcwd())

# Import des modules adaptés
try:
    from data_preprocessing_adapted import DataPreprocessor, PlumDataset, OtherCategoryDataset
    from model_classification_adapted import PlumClassifier
except ImportError:
    sys.path.append('/home/ubuntu/workspace')
    from data_preprocessing_adapted import DataPreprocessor, PlumDataset, OtherCategoryDataset
    from model_classification_adapted import PlumClassifier

class TestAdaptedCode(unittest.TestCase):
    """
    Tests unitaires pour vérifier le code adapté avec la catégorie "autre".
    """
    
    def setUp(self):
        """
        Configuration initiale pour les tests.
        """
        # Création d'un répertoire temporaire pour les tests
        self.test_dir = tempfile.mkdtemp()
        
        # Création de sous-répertoires pour simuler la structure des datasets
        self.plum_dir = os.path.join(self.test_dir, 'plums')
        self.other_dir = os.path.join(self.test_dir, 'other')
        
        os.makedirs(self.plum_dir, exist_ok=True)
        os.makedirs(self.other_dir, exist_ok=True)
        
        # Création de sous-répertoires pour les catégories de prunes
        self.categories = ['unaffected', 'unripe', 'spotted', 'cracked', 'bruised', 'rotten']
        for category in self.categories:
            os.makedirs(os.path.join(self.plum_dir, category), exist_ok=True)
        
        # Création d'images factices pour les tests
        self.create_dummy_images()
        
        # Initialisation du préprocesseur
        self.preprocessor = DataPreprocessor(
            data_dir=self.test_dir,
            image_size=64,  # Taille réduite pour les tests
            batch_size=4,
            num_workers=0
        )
        
        # Initialisation du modèle
        self.model = PlumClassifier(
            num_classes=7,  # 6 catégories de prunes + 1 catégorie "autre"
            model_name='efficientnet_b0',  # Modèle plus petit pour les tests
            pretrained=False
        )
    
    def tearDown(self):
        """
        Nettoyage après les tests.
        """
        # Suppression du répertoire temporaire
        shutil.rmtree(self.test_dir)
    
    def create_dummy_images(self):
        """
        Crée des images factices pour les tests.
        """
        # Création d'images pour chaque catégorie de prunes
        for category in self.categories:
            category_dir = os.path.join(self.plum_dir, category)
            for i in range(3):  # 3 images par catégorie
                img = Image.new('RGB', (64, 64), color=(i*50, 100, 150))
                img.save(os.path.join(category_dir, f'{category}_plum_{i}.png'))
        
        # Création d'images pour la catégorie "autre"
        for i in range(5):  # 5 images pour "autre"
            img = Image.new('RGB', (64, 64), color=(200, i*50, 100))
            img.save(os.path.join(self.other_dir, f'other_image_{i}.jpg'))
    
    def test_plum_dataset(self):
        """
        Teste la classe PlumDataset.
        """
        # Collecte des images de prunes
        image_paths, labels = self.preprocessor.collect_plum_images(self.plum_dir)
        
        # Vérification du nombre d'images collectées
        self.assertEqual(len(image_paths), 18)  # 6 catégories * 3 images
        self.assertEqual(len(labels), 18)
        
        # Création du dataset
        dataset = PlumDataset(image_paths, labels)
        
        # Vérification de la taille du dataset
        self.assertEqual(len(dataset), 18)
        
        # Vérification de l'accès aux éléments
        image, label, path = dataset[0]
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape[2], 3)  # RGB
        self.assertIsInstance(label, int)
        self.assertIsInstance(path, str)
    
    def test_other_category_dataset(self):
        """
        Teste la classe OtherCategoryDataset.
        """
        # Collecte des images pour la catégorie "autre"
        other_paths = [os.path.join(self.other_dir, f) for f in os.listdir(self.other_dir)
                      if f.endswith('.jpg')]
        
        # Création du dataset
        dataset = OtherCategoryDataset(other_paths)
        
        # Vérification de la taille du dataset
        self.assertEqual(len(dataset), 5)  # 5 images "autre"
        
        # Vérification de l'accès aux éléments
        image, label, path = dataset[0]
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape[2], 3)  # RGB
        self.assertEqual(label, 6)  # Indice de la catégorie "autre"
        self.assertIsInstance(path, str)
    
    def test_plum_classifier(self):
        """
        Teste la classe PlumClassifier.
        """
        # Création d'un tenseur d'entrée factice
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # Passe avant
        logits, confidence = self.model(input_tensor)
        
        # Vérification des dimensions de sortie
        self.assertEqual(logits.shape, (1, 7))  # 7 classes
        self.assertEqual(confidence.shape, (1, 1))  # Score de confiance
        
        # Test de la prédiction avec confiance
        class_idx, conf_score, class_name = self.model.predict_with_confidence(input_tensor)
        
        # Vérification des types de retour
        self.assertIsInstance(class_idx, int)
        self.assertIsInstance(conf_score, float)
        self.assertIsInstance(class_name, str)
        
        # Vérification que l'indice de classe est valide
        self.assertIn(class_idx, range(7))
        
        # Vérification que le score de confiance est entre 0 et 1
        self.assertGreaterEqual(conf_score, 0.0)
        self.assertLessEqual(conf_score, 1.0)
    
    def test_data_preprocessor(self):
        """
        Teste la classe DataPreprocessor.
        """
        # Analyse du dataset
        stats = self.preprocessor.analyze_dataset()
        
        # Vérification des statistiques
        self.assertIsInstance(stats, dict)
        self.assertIn('num_images', stats)
        self.assertIn('num_classes', stats)
        self.assertIn('class_counts', stats)
        
        # Préparation des données
        data = self.preprocessor.prepare_data()
        
        # Vérification des DataLoaders
        self.assertIn('train_dataloader', data)
        self.assertIn('val_dataloader', data)
        self.assertIn('test_dataloader', data)
        
        # Vérification des poids des classes
        class_weights = self.preprocessor.get_class_weights(data['class_counts'])
        self.assertIsInstance(class_weights, torch.Tensor)
        self.assertEqual(len(class_weights), 7)  # 7 classes

if __name__ == '__main__':
    unittest.main()
