"""
FastAPI application for face recognition system.
"""
import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from typing import List, Dict, Any
import redis.asyncio as redis
import json
import hashlib
from loguru import logger

from ..core.config import settings
from ..core.logging import setup_logging
from ..core.face_detection import FaceDetector
from ..models.facenet import FaceNet
from ..models.arcface import ArcFaceModel
from ..models.sphereface import SphereFaceModel
from ..anti_spoofing.detector import AntiSpoofingDetector


# Setup logging
setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Face Recognition System",
    description="A robust face recognition system with anti-spoofing capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
face_detector = None
facenet_model = None
arcface_model = None
sphereface_model = None
antispoofing_detector = None
redis_client = None


async def get_redis_client():
    """Get Redis client."""
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=True
        )
    return redis_client


@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup."""
    global face_detector, facenet_model, arcface_model, sphereface_model, antispoofing_detector
    
    logger.info("Initializing face recognition system...")
    
    try:
        # Initialize face detector
        face_detector = FaceDetector(method="mtcnn")
        logger.info("Face detector initialized")
        
        # Initialize FaceNet model
        facenet_model = FaceNet(pretrained=True)
        logger.info("FaceNet model initialized")
        
        # Initialize ArcFace model
        arcface_model = ArcFaceModel(pretrained=False)
        logger.info("ArcFace model initialized")
        
        # Initialize SphereFace model
        sphereface_model = SphereFaceModel(pretrained=False)
        logger.info("SphereFace model initialized")
        
        # Initialize anti-spoofing detector
        antispoofing_detector = AntiSpoofingDetector()
        logger.info("Anti-spoofing detector initialized")
        
        # Test Redis connection
        redis_client = await get_redis_client()
        await redis_client.ping()
        logger.info("Redis connection established")
        
        logger.info("Face recognition system startup completed")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global redis_client
    if redis_client:
        await redis_client.close()
    logger.info("Face recognition system shutdown completed")


def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Load image from uploaded file."""
    try:
        # Read image data
        image_data = file.file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")


def generate_cache_key(data: str) -> str:
    """Generate cache key from data."""
    return hashlib.md5(data.encode()).hexdigest()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Advanced Face Recognition System API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Redis connection
        redis_client = await get_redis_client()
        await redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy",
        "models": {
            "face_detector": "loaded" if face_detector else "not loaded",
            "facenet": "loaded" if facenet_model else "not loaded",
            "arcface": "loaded" if arcface_model else "not loaded",
            "sphereface": "loaded" if sphereface_model else "not loaded",
            "antispoofing": "loaded" if antispoofing_detector else "not loaded"
        },
        "redis": redis_status
    }


@app.post("/detect_faces")
async def detect_faces(file: UploadFile = File(...)):
    """Detect faces in an uploaded image."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Load image
        image = load_image_from_upload(file)
        
        # Detect faces
        faces = face_detector.detect_faces(image)
        
        # Prepare response
        response = {
            "num_faces": len(faces),
            "faces": []
        }
        
        for i, face in enumerate(faces):
            response["faces"].append({
                "face_id": i,
                "bbox": face["bbox"],
                "confidence": face["confidence"],
                "has_landmarks": bool(face["landmarks"])
            })
        
        logger.info(f"Detected {len(faces)} faces in uploaded image")
        return response
        
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_embeddings")
async def extract_embeddings(
    file: UploadFile = File(...),
    model: str = "facenet",
    use_cache: bool = True
):
    """Extract face embeddings from an uploaded image."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if model not in ["facenet", "arcface", "sphereface"]:
        raise HTTPException(status_code=400, detail="Model must be 'facenet', 'arcface', or 'sphereface'")
    
    try:
        # Generate cache key
        file_content = await file.read()
        await file.seek(0)  # Reset file pointer
        cache_key = f"embeddings_{model}_{generate_cache_key(file_content.hex())}"
        
        # Check cache
        redis_client = await get_redis_client()
        if use_cache:
            cached_result = await redis_client.get(cache_key)
            if cached_result:
                logger.info("Returning cached embeddings")
                return json.loads(cached_result)
        
        # Load image
        image = load_image_from_upload(file)
        
        # Detect faces
        faces = face_detector.detect_faces(image)
        
        if not faces:
            return {"message": "No faces detected", "embeddings": []}
        
        # Extract face images
        face_images = []
        for face in faces:
            face_img = face_detector.extract_face(image, face["bbox"])
            face_images.append(face_img)
        
        # Convert to tensor
        face_tensors = []
        for face_img in face_images:
            # Convert to tensor and normalize
            face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).float()
            face_tensors.append(face_tensor)
        
        face_batch = torch.stack(face_tensors)
        
        # Extract embeddings
        if model == "facenet":
            embeddings = facenet_model.extract_embeddings(face_batch)
        elif model == "arcface":
            embeddings = arcface_model.extract_embeddings(face_batch)
        else:  # sphereface
            embeddings = sphereface_model.extract_embeddings(face_batch)
        
        # Prepare response
        response = {
            "num_faces": len(faces),
            "model_used": model,
            "embeddings": []
        }
        
        for i, (face, embedding) in enumerate(zip(faces, embeddings)):
            response["embeddings"].append({
                "face_id": i,
                "bbox": face["bbox"],
                "embedding": embedding.tolist(),
                "embedding_dim": len(embedding)
            })
        
        # Cache result
        if use_cache:
            await redis_client.setex(cache_key, 3600, json.dumps(response))  # Cache for 1 hour
        
        logger.info(f"Extracted embeddings for {len(faces)} faces using {model}")
        return response
        
    except Exception as e:
        logger.error(f"Error in embedding extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/anti_spoofing_check")
async def anti_spoofing_check(file: UploadFile = File(...)):
    """Check if faces in an image are real or spoofed."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Load image
        image = load_image_from_upload(file)
        
        # Detect faces
        faces = face_detector.detect_faces(image)
        
        if not faces:
            return {"message": "No faces detected", "results": []}
        
        # Check each face for spoofing
        results = []
        for i, face in enumerate(faces):
            face_img = face_detector.extract_face(image, face["bbox"], size=(224, 224))
            
            # Convert back to uint8 for anti-spoofing
            face_img_uint8 = (face_img * 255).astype(np.uint8)
            
            is_real, confidence, features = antispoofing_detector.detect_spoofing(face_img_uint8)
            
            results.append({
                "face_id": i,
                "bbox": face["bbox"],
                "is_real": is_real,
                "confidence": confidence,
                "spoofing_probability": 1.0 - confidence,
                "analysis_features": {
                    "cnn_score": features.get("cnn_score", 0.0),
                    "lbp_uniformity": features.get("lbp_uniformity", 0.0)
                }
            })
        
        response = {
            "num_faces": len(faces),
            "results": results,
            "threshold_used": settings.anti_spoofing_threshold
        }
        
        logger.info(f"Performed anti-spoofing check on {len(faces)} faces")
        return response
        
    except Exception as e:
        logger.error(f"Error in anti-spoofing check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare_faces")
async def compare_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    model: str = "facenet"
):
    """Compare faces in two images."""
    if not file1.content_type.startswith('image/') or not file2.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Both files must be images")
    
    if model not in ["facenet", "arcface", "sphereface"]:
        raise HTTPException(status_code=400, detail="Model must be 'facenet', 'arcface', or 'sphereface'")
    
    try:
        # Load images
        image1 = load_image_from_upload(file1)
        image2 = load_image_from_upload(file2)
        
        # Detect faces in both images
        faces1 = face_detector.detect_faces(image1)
        faces2 = face_detector.detect_faces(image2)
        
        if not faces1 or not faces2:
            return {
                "message": "Need at least one face in each image",
                "faces_in_image1": len(faces1),
                "faces_in_image2": len(faces2),
                "comparisons": []
            }
        
        # Extract embeddings for all faces
        def extract_embeddings_for_faces(image, faces):
            face_images = []
            for face in faces:
                face_img = face_detector.extract_face(image, face["bbox"])
                face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).float()
                face_images.append(face_tensor)
            
            if face_images:
                face_batch = torch.stack(face_images)
                if model == "facenet":
                    return facenet_model.extract_embeddings(face_batch)
                elif model == "arcface":
                    return arcface_model.extract_embeddings(face_batch)
                else:  # sphereface
                    return sphereface_model.extract_embeddings(face_batch)
            return []
        
        embeddings1 = extract_embeddings_for_faces(image1, faces1)
        embeddings2 = extract_embeddings_for_faces(image2, faces2)
        
        # Compare all face pairs
        comparisons = []
        for i, emb1 in enumerate(embeddings1):
            for j, emb2 in enumerate(embeddings2):
                if model == "facenet":
                    similarity = facenet_model.compute_similarity(emb1, emb2)
                elif model == "arcface":
                    similarity = arcface_model.compute_similarity(emb1, emb2)
                else:  # sphereface
                    similarity = sphereface_model.compute_similarity(emb1, emb2)
                
                is_same_person = similarity > settings.face_recognition_threshold
                
                comparisons.append({
                    "face1_id": i,
                    "face1_bbox": faces1[i]["bbox"],
                    "face2_id": j,
                    "face2_bbox": faces2[j]["bbox"],
                    "similarity": float(similarity),
                    "is_same_person": is_same_person,
                    "threshold_used": settings.face_recognition_threshold
                })
        
        response = {
            "faces_in_image1": len(faces1),
            "faces_in_image2": len(faces2),
            "model_used": model,
            "comparisons": comparisons,
            "best_match": max(comparisons, key=lambda x: x["similarity"]) if comparisons else None
        }
        
        logger.info(f"Compared {len(faces1)} faces with {len(faces2)} faces using {model}")
        return response
        
    except Exception as e:
        logger.error(f"Error in face comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clear_cache")
async def clear_cache():
    """Clear Redis cache."""
    try:
        redis_client = await get_redis_client()
        await redis_client.flushdb()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug
    )