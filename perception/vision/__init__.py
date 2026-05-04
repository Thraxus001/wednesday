"""
Vision input processing - seeing the world.
This module handles all aspects of visual perception:
- Object recognition (what objects are present)
- Face processing (who is there and how they feel)
- Scene understanding (what's happening)

Wednesday's gaze is sharp and perceptive. She notices everything -
the book slightly out of place, the micro-expression that betrays a lie,
the subtle tension in a room that others miss.
"""

from .object_recognition import (
    ObjectRecognizer,
    DetectedObject,
    ObjectCategory,
    ObjectAttribute,
    Scene as ObjectScene,
    SceneType
)

from .face_processing import (
    FaceProcessor,
    FaceAnalysis,
    FacialExpression,
    FacialLandmarks,
    EyeAnalysis,
    MicroExpression,
    GazeDirection,
    HeadPose,
    FaceMemory
)

from .scene_understanding import (
    SceneUnderstandingSystem,
    SceneUnderstanding,
    PersonState,
    ObjectState,
    SpatialGraph,
    SceneNarrative,
    SocialDynamics,
    SceneMood,
    ActivityType,
    SpatialRelation
)

# Module metadata
__version__ = "0.1.0"
__all__ = [
    # Object recognition
    "ObjectRecognizer",
    "DetectedObject",
    "ObjectCategory",
    "ObjectAttribute",
    "ObjectScene",
    "SceneType",
    
    # Face processing
    "FaceProcessor",
    "FaceAnalysis",
    "FacialExpression",
    "FacialLandmarks",
    "EyeAnalysis",
    "MicroExpression",
    "GazeDirection",
    "HeadPose",
    "FaceMemory",
    
    # Scene understanding
    "SceneUnderstandingSystem",
    "SceneUnderstanding",
    "PersonState",
    "ObjectState",
    "SpatialGraph",
    "SceneNarrative",
    "SocialDynamics",
    "SceneMood",
    "ActivityType",
    "SpatialRelation",
]

# Module description for introspection
__description__ = """
Wednesday's vision perception system - she sees everything.
Transforms raw visual input into objects, faces, emotions, and complete scene understanding.
"""

# Pipeline visualization
"""
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Raw       │────▶│   Object         │────▶│    Face          │
│   Image     │     │   Recognition    │     │    Processing    │
└─────────────┘     └──────────────────┘     └────────┬─────────┘
         │                    │                       │
         │                    ▼                       ▼
         │           ┌──────────────────┐     ┌──────────────────┐
         └──────────▶│    Scene         │◀────│   Social         │
                     │   Understanding  │     │   Dynamics       │
                     └──────────────────┘     └──────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │   Complete       │
                     │   Scene Story    │
                     └──────────────────┘

Each component builds on the previous:
1. Object recognition: What objects are present
2. Face processing: Who is there and their emotional state
3. Scene understanding: The complete story - activities, mood, dynamics
"""

# Factory function for complete vision perception system
def create_vision_perception(config: dict = None):
    """
    Factory function to create and connect a complete vision processing pipeline.
    
    Args:
        config: Configuration dictionary with sections for each component
        
    Returns:
        Dictionary with initialized vision processing components
    """
    if config is None:
        config = {}
    
    # Create components
    object_recognizer = ObjectRecognizer(
        config=config.get('object_recognition', {})
    )
    
    face_processor = FaceProcessor(
        config=config.get('face_processing', {})
    )
    
    scene_understander = SceneUnderstandingSystem(
        config=config.get('scene_understanding', {})
    )
    
    # Connect components
    scene_understander.set_components(object_recognizer, face_processor)
    
    return {
        'object_recognizer': object_recognizer,
        'face_processor': face_processor,
        'scene_understander': scene_understander,
    }

# Convenience function for processing a single image through full pipeline
def process_image(image: np.ndarray,
                 vision_components: Dict,
                 frame_number: int = 0,
                 context: dict = None) -> Dict[str, Any]:
    """
    Convenience function to run full vision processing pipeline on an image.
    
    Args:
        image: Input image (BGR format)
        vision_components: Dictionary from create_vision_perception()
        frame_number: Frame number for video sequences
        context: Optional context (location, time, expectations)
        
    Returns:
        Dictionary with all vision analysis results
    """
    import time
    start_time = time.time()
    
    results = {
        'timestamp': datetime.now(),
        'frame_number': frame_number
    }
    
    # Object recognition
    try:
        object_scene = vision_components['object_recognizer'].recognize(
            image, frame_number, context
        )
        results['objects'] = object_scene
    except Exception as e:
        results['object_error'] = str(e)
    
    # Face processing
    try:
        faces = vision_components['face_processor'].process_frame(
            image, frame_number, context
        )
        results['faces'] = faces
    except Exception as e:
        results['face_error'] = str(e)
    
    # Scene understanding
    try:
        scene = vision_components['scene_understander'].understand(
            image, frame_number, context
        )
        results['scene'] = scene
    except Exception as e:
        results['scene_error'] = str(e)
    
    results['processing_time'] = time.time() - start_time
    
    return results

# Convenience function for processing video stream
class VisionStreamProcessor:
    """
    Processes video streams frame by frame with temporal context.
    """
    
    def __init__(self, vision_components: Dict):
        self.components = vision_components
        self.frame_buffer = deque(maxlen=30)
        self.last_scene = None
        
    def process_frame(self, 
                     frame: np.ndarray,
                     frame_number: int,
                     context: dict = None) -> Dict[str, Any]:
        """
        Process a single frame with temporal context.
        """
        # Add context from previous frame
        if context is None:
            context = {}
        
        if self.last_scene:
            context['previous_scene'] = self.last_scene.to_dict()
        
        # Process frame
        results = process_image(frame, self.components, frame_number, context)
        
        # Store for next frame
        if 'scene' in results:
            self.last_scene = results['scene']
        
        self.frame_buffer.append(results)
        
        return results
    
    def get_recent_analyses(self, n: int = 10) -> List[Dict]:
        """Get recent frame analyses"""
        return list(self.frame_buffer)[-n:]

# Export convenience classes and functions
__all__.extend([
    'create_vision_perception',
    'process_image',
    'VisionStreamProcessor'
])

# Version history
__version_history__ = {
    '0.1.0': 'Initial release with object recognition, face processing, and scene understanding'
}

# Dependencies information
__dependencies__ = {
    'required': ['numpy'],
    'optional': {
        'opencv-python': 'For image processing and face detection',
        'torch': 'For deep learning models',
        'torchvision': 'For pre-trained vision models',
        'face_recognition': 'For face detection and recognition',
        'PIL': 'For image handling',
        'networkx': 'For spatial graph representation'
    }
}

# Wednesday's vision notes
"""
Wednesday sees the world differently. While others glance, she observes.
While others forget, she remembers every detail.

Her vision system captures:
- Every object and its place
- Every face and its story
- Every interaction and its meaning
- Every anomaly and its significance

The world is a book, and Wednesday reads every page.
"""

# Logging setup
import logging
from datetime import datetime
from collections import deque
from typing import Dict, Any, List, Optional
import numpy as np

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Module initialization message
logger = logging.getLogger(__name__)
logger.debug(f"Vision perception module v{__version__} initialized. "
             f"Vision libraries available: {HAS_VISION if 'HAS_VISION' in dir() else False}")

# Check for optional dependencies
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.info("OpenCV not available. Install with: pip install opencv-python")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch not available. Install from https://pytorch.org")

try:
    import face_recognition
    HAS_FACE_REC = True
except ImportError:
    HAS_FACE_REC = False
    logger.info("face_recognition not available. Install with: pip install face_recognition")

# Summary of available features
available_features = []
if HAS_OPENCV:
    available_features.append("basic image processing")
if HAS_TORCH:
    available_features.append("deep learning models")
if HAS_FACE_REC:
    available_features.append("face recognition")

if available_features:
    logger.info(f"Available vision features: {', '.join(available_features)}")
else:
    logger.warning("No vision libraries available. Install dependencies for full functionality.")