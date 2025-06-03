#!/usr/bin/env python3
"""
Main entry point for the Advanced Face Recognition System.
"""
import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config import settings
from src.core.logging import setup_logging
from src.utils.dataset_utils import DatasetDownloader, DatasetLoader
from loguru import logger


def setup_datasets():
    """Setup and download required datasets."""
    logger.info("Setting up datasets...")
    
    downloader = DatasetDownloader()
    
    # Download LFW dataset
    try:
        lfw_path = downloader.download_lfw_dataset()
        logger.info(f"LFW dataset ready at: {lfw_path}")
    except Exception as e:
        logger.error(f"Failed to download LFW dataset: {e}")
    
    # Provide VGGFace2 information
    vgg_info = downloader.prepare_vggface2_info()
    logger.info("VGGFace2 dataset information:")
    for key, value in vgg_info.items():
        if key == "instructions":
            logger.info(f"{key}:")
            for instruction in value:
                logger.info(f"  {instruction}")
        else:
            logger.info(f"{key}: {value}")


def run_api_server():
    """Run the FastAPI server."""
    import uvicorn
    
    logger.info("Starting Face Recognition API server...")
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
        log_level="info"
    )


def run_evaluation():
    """Run model evaluation on LFW dataset."""
    logger.info("Running model evaluation...")
    
    # Import here to avoid startup delays
    from src.core.face_detection import FaceDetector
    from src.models.facenet import FaceNet
    from src.utils.dataset_utils import EvaluationMetrics
    
    # Initialize models
    face_detector = FaceDetector(method="mtcnn")
    facenet_model = FaceNet(pretrained=True)
    metrics = EvaluationMetrics()
    
    # Load LFW pairs
    loader = DatasetLoader()
    pairs = loader.load_lfw_pairs()
    
    if not pairs:
        logger.error("No LFW pairs found. Please ensure LFW dataset is downloaded.")
        return
    
    logger.info(f"Evaluating on {len(pairs)} pairs...")
    
    # TODO: Implement full evaluation loop
    # This would involve:
    # 1. Loading image pairs
    # 2. Detecting faces
    # 3. Extracting embeddings
    # 4. Computing similarities
    # 5. Calculating metrics
    
    logger.info("Evaluation completed (placeholder)")


def test_installation():
    """Test if all components are working correctly."""
    logger.info("Testing installation...")
    
    try:
        # Test imports
        logger.info("Testing imports...")
        from src.core.face_detection import FaceDetector
        from src.models.facenet import FaceNet
        from src.models.arcface import ArcFaceModel
        from src.anti_spoofing.detector import AntiSpoofingDetector
        logger.info("✓ All imports successful")
        
        # Test model initialization
        logger.info("Testing model initialization...")
        face_detector = FaceDetector(method="mtcnn")
        logger.info("✓ Face detector initialized")
        
        # Test with dummy data
        import numpy as np
        import torch
        
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = face_detector.detect_faces(dummy_image)
        logger.info(f"✓ Face detection test completed (found {len(faces)} faces)")
        
        logger.info("Installation test completed successfully!")
        
    except Exception as e:
        logger.error(f"Installation test failed: {e}")
        return False
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Advanced Face Recognition System")
    parser.add_argument(
        "command",
        choices=["setup", "server", "evaluate", "test"],
        help="Command to run"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        settings.debug = True
        settings.log_level = "DEBUG"
    
    setup_logging()
    
    logger.info("Advanced Face Recognition System")
    logger.info(f"Command: {args.command}")
    
    if args.command == "setup":
        setup_datasets()
    elif args.command == "server":
        run_api_server()
    elif args.command == "evaluate":
        run_evaluation()
    elif args.command == "test":
        if test_installation():
            logger.info("System is ready to use!")
        else:
            logger.error("System test failed")
            sys.exit(1)


if __name__ == "__main__":
    main()