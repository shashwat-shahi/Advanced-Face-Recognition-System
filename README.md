# Advanced Face Recognition System

A robust face recognition system using modern deep learning techniques with anti-spoofing capabilities, built with PyTorch, OpenCV, and FastAPI.

## Features

- **Multiple Face Recognition Models**: FaceNet, ArcFace, and SphereFace implementations
- **Anti-Spoofing Detection**: Advanced spoofing detection using CNN, LBP, color space analysis, and texture features
- **Multiple Face Detection Methods**: MTCNN, dlib, and OpenCV Haar cascades
- **RESTful API**: FastAPI-based web service with comprehensive endpoints
- **Redis Caching**: High-performance caching for embeddings and results
- **Dataset Support**: LFW and VGGFace2 dataset integration
- **Evaluation Tools**: Built-in evaluation metrics and testing utilities

## Tech Stack

### Frameworks & Libraries
- **PyTorch**: Deep learning framework for neural networks
- **OpenCV**: Computer vision and image processing
- **dlib**: Face detection and facial landmark detection
- **face_recognition**: Simple face recognition library
- **MTCNN**: Multi-task CNN for face detection
- **FastAPI**: Modern web framework for APIs
- **Redis**: In-memory caching and data store
- **Pydantic**: Data validation and settings management

### Models
- **FaceNet**: Face recognition using deep neural networks
- **ArcFace**: Additive angular margin loss for face recognition
- **SphereFace**: Deep hypersphere embedding for face recognition

## Installation

### Prerequisites
- Python 3.8 or higher
- Redis server (for caching)
- CUDA-compatible GPU (optional, for faster inference)

### Quick Setup

1. **Clone the repository**:
```bash
git clone https://github.com/shashwat-shahi/Advanced-Face-Recognition-System.git
cd Advanced-Face-Recognition-System
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env file with your configurations
```

4. **Test the installation**:
```bash
python main.py test
```

5. **Setup datasets** (optional):
```bash
python main.py setup
```

## Usage

### Command Line Interface

The system provides a simple CLI for common operations:

```bash
# Test installation
python main.py test

# Setup datasets
python main.py setup

# Start API server
python main.py server

# Run evaluation
python main.py evaluate
```

### API Server

Start the FastAPI server:

```bash
python main.py server
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### API Endpoints

#### Face Detection
```http
POST /detect_faces
Content-Type: multipart/form-data

{
  "file": "image_file.jpg"
}
```

#### Extract Embeddings
```http
POST /extract_embeddings?model=facenet
Content-Type: multipart/form-data

{
  "file": "image_file.jpg"
}
```

#### Anti-Spoofing Check
```http
POST /anti_spoofing_check
Content-Type: multipart/form-data

{
  "file": "image_file.jpg"
}
```

#### Compare Faces
```http
POST /compare_faces?model=facenet
Content-Type: multipart/form-data

{
  "file1": "image1.jpg",
  "file2": "image2.jpg"
}
```

### Python API

```python
from src.core.face_detection import FaceDetector
from src.models.facenet import FaceNet
from src.anti_spoofing.detector import AntiSpoofingDetector

# Initialize components
detector = FaceDetector(method="mtcnn")
facenet = FaceNet(pretrained=True)
antispoofing = AntiSpoofingDetector()

# Detect faces
faces = detector.detect_faces(image)

# Extract embeddings
embeddings = facenet.extract_embeddings(face_tensors)

# Check for spoofing
is_real, confidence, features = antispoofing.detect_spoofing(face_image)
```

## Project Structure

```
Advanced-Face-Recognition-System/
├── src/
│   ├── api/                 # FastAPI application
│   │   └── main.py         # API endpoints
│   ├── core/               # Core functionality
│   │   ├── config.py       # Configuration management
│   │   ├── logging.py      # Logging setup
│   │   └── face_detection.py  # Face detection
│   ├── models/             # Recognition models
│   │   ├── facenet.py      # FaceNet implementation
│   │   ├── arcface.py      # ArcFace implementation
│   │   └── sphereface.py   # SphereFace implementation
│   ├── anti_spoofing/      # Anti-spoofing detection
│   │   └── detector.py     # Spoofing detection
│   └── utils/              # Utility functions
│       └── dataset_utils.py  # Dataset handling
├── tests/                  # Test suite
├── config/                 # Configuration files
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── main.py               # Main CLI application
└── README.md             # This file
```

## Datasets

### LFW Dataset
The Labeled Faces in the Wild (LFW) dataset is automatically downloaded when running:
```bash
python main.py setup
```

### VGGFace2 Dataset
Due to licensing requirements, VGGFace2 must be downloaded manually:
1. Visit: https://github.com/ox-vgg/vgg_face2
2. Follow the download instructions
3. Extract to `./datasets/vggface2`

## Configuration

Configuration is managed through environment variables and the `.env` file:

```env
# Environment
DEBUG=True
LOG_LEVEL=INFO

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
FACE_DETECTION_CONFIDENCE=0.5
FACE_RECOGNITION_THRESHOLD=0.6
ANTI_SPOOFING_THRESHOLD=0.5
```

## Anti-Spoofing Features

The system includes multiple anti-spoofing techniques:

1. **CNN-based Detection**: Deep learning model trained to distinguish real vs. fake faces
2. **Local Binary Patterns (LBP)**: Texture analysis for liveness detection
3. **Color Space Analysis**: Analysis across multiple color spaces (HSV, LAB, YUV)
4. **GLCM Texture Features**: Gray-Level Co-occurrence Matrix features

## Performance

The system is optimized for both accuracy and speed:
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Redis Caching**: Embeddings and results caching for improved response times
- **Batch Processing**: Efficient batch processing for multiple faces
- **Model Optimization**: Optimized model architectures for inference

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest

# Run specific test file
pytest tests/test_face_detection.py

# Run with coverage
pytest --cov=src tests/
```

## Development

### Adding New Models

1. Create a new file in `src/models/`
2. Implement the model class with required methods:
   - `extract_embeddings(faces)`
   - `compute_similarity(emb1, emb2)`
3. Add the model to the API endpoints

### Adding New Detection Methods

1. Extend `FaceDetector` class in `src/core/face_detection.py`
2. Implement the detection method
3. Update the method selection logic

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **FaceNet**: Schroff, F., Kalenichenko, D., & Philbin, J. (2015)
- **ArcFace**: Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019)
- **MTCNN**: Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016)
- **LFW Dataset**: Huang, G. B., Ramesh, M., Berg, T., & Learned-Miller, E. (2007)

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation at `http://localhost:8000/docs` when running the API
- Review the test files for usage examples

## Roadmap

- [ ] SphereFace model implementation
- [ ] Real-time video processing
- [ ] Mobile deployment support
- [ ] Advanced anti-spoofing techniques
- [ ] Federated learning capabilities
- [ ] Performance benchmarking tools
