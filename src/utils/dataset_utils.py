"""
Dataset utilities for face recognition system.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import urllib.request
import zipfile
import tarfile
from loguru import logger
from tqdm import tqdm

from ..core.config import settings


class DatasetDownloader:
    """Download and prepare face recognition datasets."""
    
    def __init__(self, dataset_path: Optional[Path] = None):
        """
        Initialize dataset downloader.
        
        Args:
            dataset_path: Path to store datasets
        """
        self.dataset_path = dataset_path or settings.dataset_path
        self.dataset_path.mkdir(parents=True, exist_ok=True)
    
    def download_lfw_dataset(self) -> Path:
        """
        Download and extract LFW dataset.
        
        Returns:
            Path to extracted dataset
        """
        lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
        lfw_path = self.dataset_path / "lfw"
        
        if lfw_path.exists():
            logger.info("LFW dataset already exists")
            return lfw_path
        
        logger.info("Downloading LFW dataset...")
        
        # Download
        tgz_path = self.dataset_path / "lfw.tgz"
        self._download_file(lfw_url, tgz_path)
        
        # Extract
        logger.info("Extracting LFW dataset...")
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(self.dataset_path)
        
        # Clean up
        tgz_path.unlink()
        
        logger.info(f"LFW dataset downloaded to {lfw_path}")
        return lfw_path
    
    def prepare_vggface2_info(self) -> Dict[str, str]:
        """
        Provide information about VGGFace2 dataset.
        
        Returns:
            Information about VGGFace2 dataset
        """
        info = {
            "name": "VGGFace2",
            "description": "Large-scale face recognition dataset",
            "url": "https://github.com/ox-vgg/vgg_face2",
            "note": "VGGFace2 requires manual download due to licensing",
            "instructions": [
                "1. Visit https://github.com/ox-vgg/vgg_face2",
                "2. Follow the download instructions",
                "3. Extract to " + str(settings.vggface2_dataset_path),
                "4. Ensure the directory structure is: identity_name/image_name.jpg"
            ]
        }
        
        logger.info("VGGFace2 dataset requires manual download")
        return info
    
    def _download_file(self, url: str, filepath: Path):
        """Download file with progress bar."""
        def progress_hook(block_num, block_size, total_size):
            if hasattr(progress_hook, 'pbar'):
                progress_hook.pbar.update(block_size)
            else:
                progress_hook.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        if hasattr(progress_hook, 'pbar'):
            progress_hook.pbar.close()


class DatasetLoader:
    """Load and preprocess face recognition datasets."""
    
    def __init__(self, dataset_path: Optional[Path] = None):
        """
        Initialize dataset loader.
        
        Args:
            dataset_path: Path to datasets
        """
        self.dataset_path = dataset_path or settings.dataset_path
    
    def load_lfw_pairs(self, pairs_file: Optional[str] = None) -> List[Tuple[str, str, bool]]:
        """
        Load LFW pairs for evaluation.
        
        Args:
            pairs_file: Path to pairs file
            
        Returns:
            List of (image1_path, image2_path, is_same_person)
        """
        if pairs_file is None:
            pairs_file = self.dataset_path / "lfw" / "pairs.txt"
        
        pairs = []
        
        if not Path(pairs_file).exists():
            logger.warning(f"Pairs file not found: {pairs_file}")
            return pairs
        
        with open(pairs_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            
            for line in lines:
                parts = line.strip().split()
                
                if len(parts) == 3:  # Same person
                    name, img1_num, img2_num = parts
                    img1_path = self.dataset_path / "lfw" / name / f"{name}_{img1_num.zfill(4)}.jpg"
                    img2_path = self.dataset_path / "lfw" / name / f"{name}_{img2_num.zfill(4)}.jpg"
                    pairs.append((str(img1_path), str(img2_path), True))
                
                elif len(parts) == 4:  # Different persons
                    name1, img1_num, name2, img2_num = parts
                    img1_path = self.dataset_path / "lfw" / name1 / f"{name1}_{img1_num.zfill(4)}.jpg"
                    img2_path = self.dataset_path / "lfw" / name2 / f"{name2}_{img2_num.zfill(4)}.jpg"
                    pairs.append((str(img1_path), str(img2_path), False))
        
        logger.info(f"Loaded {len(pairs)} pairs from LFW dataset")
        return pairs
    
    def load_identity_images(self, dataset_name: str = "lfw") -> Dict[str, List[str]]:
        """
        Load images grouped by identity.
        
        Args:
            dataset_name: Name of dataset ('lfw' or 'vggface2')
            
        Returns:
            Dictionary mapping identity names to image paths
        """
        if dataset_name == "lfw":
            dataset_path = self.dataset_path / "lfw"
        elif dataset_name == "vggface2":
            dataset_path = settings.vggface2_dataset_path
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        identities = {}
        
        if not dataset_path.exists():
            logger.warning(f"Dataset path not found: {dataset_path}")
            return identities
        
        for identity_dir in dataset_path.iterdir():
            if identity_dir.is_dir():
                identity_name = identity_dir.name
                image_paths = []
                
                for img_file in identity_dir.glob("*.jpg"):
                    if img_file.is_file():
                        image_paths.append(str(img_file))
                
                if image_paths:
                    identities[identity_name] = image_paths
        
        logger.info(f"Loaded {len(identities)} identities from {dataset_name} dataset")
        return identities


class ImagePreprocessor:
    """Preprocess images for face recognition."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array or None if error
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return None
            
            # Resize
            image_resized = cv2.resize(image, self.target_size)
            
            # Normalize
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            return image_normalized
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def preprocess_batch(self, image_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Tuple of (batch_array, valid_paths)
        """
        batch_images = []
        valid_paths = []
        
        for path in tqdm(image_paths, desc="Preprocessing images"):
            image = self.preprocess_image(path)
            if image is not None:
                batch_images.append(image)
                valid_paths.append(path)
        
        if batch_images:
            batch_array = np.stack(batch_images)
            return batch_array, valid_paths
        else:
            return np.array([]), []


class EvaluationMetrics:
    """Evaluation metrics for face recognition."""
    
    def __init__(self):
        """Initialize evaluation metrics."""
        pass
    
    def calculate_accuracy(self, predictions: List[bool], ground_truth: List[bool]) -> float:
        """
        Calculate accuracy.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth labels
            
        Returns:
            Accuracy score
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        correct = sum(p == gt for p, gt in zip(predictions, ground_truth))
        return correct / len(predictions)
    
    def calculate_precision_recall_f1(self, predictions: List[bool], 
                                    ground_truth: List[bool]) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth labels
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        tp = sum(p and gt for p, gt in zip(predictions, ground_truth))
        fp = sum(p and not gt for p, gt in zip(predictions, ground_truth))
        fn = sum(not p and gt for p, gt in zip(predictions, ground_truth))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1_score
    
    def calculate_roc_auc(self, scores: List[float], ground_truth: List[bool]) -> float:
        """
        Calculate ROC AUC score.
        
        Args:
            scores: List of similarity scores
            ground_truth: List of ground truth labels
            
        Returns:
            ROC AUC score
        """
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(ground_truth, scores)
        except ImportError:
            logger.warning("scikit-learn not available, cannot calculate ROC AUC")
            return 0.0