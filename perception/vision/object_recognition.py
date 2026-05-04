"""
Identifies objects in images/video; scene understanding.
Wednesday sees everything - the subtle details others miss.
A book out of place, a unfamiliar face, an object that shouldn't be there.
"""
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
from collections import defaultdict, deque

# Computer vision libraries
try:
    import cv2
    from PIL import Image
    import torch
    import torchvision
    HAS_VISION = True
except ImportError as e:
    HAS_VISION = False
    logging.warning(f"Computer vision libraries not available: {e}. Install opencv-python, torch, torchvision")

logger = logging.getLogger(__name__)

class ObjectCategory(Enum):
    """Categories of objects that can be recognized"""
    # People
    PERSON = "person"
    FACE = "face"
    GROUP = "group"  # Multiple people
    
    # Animals
    ANIMAL = "animal"
    PET = "pet"  # Domestic animals
    WILDLIFE = "wildlife"
    
    # Furniture
    CHAIR = "chair"
    TABLE = "table"
    SOFA = "sofa"
    BED = "bed"
    DESK = "desk"
    SHELF = "shelf"
    CABINET = "cabinet"
    
    # Electronics
    TV = "tv"
    MONITOR = "monitor"
    LAPTOP = "laptop"
    PHONE = "phone"
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    SPEAKER = "speaker"
    CAMERA = "camera"
    
    # Kitchen items
    CUP = "cup"
    BOTTLE = "bottle"
    PLATE = "plate"
    BOWL = "bowl"
    UTENSIL = "utensil"
    KNIFE = "knife"
    APPLIANCE = "appliance"
    
    # Books/papers
    BOOK = "book"
    NOTEBOOK = "notebook"
    PAPER = "paper"
    MAGAZINE = "magazine"
    
    # Clothing
    CLOTHING = "clothing"
    HAT = "hat"
    BAG = "bag"
    SHOES = "shoes"
    
    # Personal items
    WALLET = "wallet"
    KEYS = "keys"
    JEWELRY = "jewelry"
    WATCH = "watch"
    
    # Environmental
    WINDOW = "window"
    DOOR = "door"
    WALL = "wall"
    FLOOR = "floor"
    CEILING = "ceiling"
    LIGHT = "light"
    PLANT = "plant"
    TREE = "tree"
    BENCH = "bench"
    
    # Vehicles
    CAR = "car"
    BICYCLE = "bicycle"
    MOTORCYCLE = "motorcycle"
    TRUCK = "truck"
    BUS = "bus"
    
    # Food
    FOOD = "food"
    DRINK = "drink"
    
    # Toys
    TOY = "toy"
    
    # Other
    REMOTE = "remote"
    VASE = "vase"
    CLOCK = "clock"
    SCISSORS = "scissors"
    
    # Unknown
    UNKNOWN = "unknown"

class SceneType(Enum):
    """Types of scenes"""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    
    # Specific indoor
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    OFFICE = "office"
    CLASSROOM = "classroom"
    HALLWAY = "hallway"
    BASEMENT = "basement"
    ATTIC = "attic"
    GARAGE = "garage"
    
    # Specific outdoor
    STREET = "street"
    PARK = "park"
    FOREST = "forest"
    BEACH = "beach"
    MOUNTAIN = "mountain"
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"
    
    # Special
    VEHICLE_INTERIOR = "vehicle_interior"
    CROWDED = "crowded"
    EMPTY = "empty"
    DARK = "dark"
    
    UNKNOWN = "unknown"

class ObjectAttribute(Enum):
    """Attributes of detected objects"""
    # Size
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    HUGE = "huge"
    
    # Position
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    
    # Motion
    STATIONARY = "stationary"
    MOVING = "moving"
    MOVING_FAST = "moving_fast"
    MOVING_SLOW = "moving_slow"
    
    # State
    OPEN = "open"
    CLOSED = "closed"
    FULL = "full"
    EMPTY = "empty"
    BROKEN = "broken"
    NEW = "new"
    OLD = "old"
    DIRTY = "dirty"
    CLEAN = "clean"
    
    # Interaction
    BEING_USED = "being_used"
    WITHIN_REACH = "within_reach"
    OUT_OF_REACH = "out_of_reach"

@dataclass
class DetectedObject:
    """Represents an object detected in an image"""
    category: ObjectCategory
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    size: Tuple[int, int]  # width, height
    area: float  # pixels²
    
    # Attributes
    attributes: List[Tuple[ObjectAttribute, float]] = field(default_factory=list)
    color: Optional[Tuple[int, int, int]] = None  # Average color (BGR)
    text_detected: Optional[str] = None  # If object contains text
    
    # Tracking
    object_id: Optional[str] = None  # For tracking across frames
    velocity: Optional[Tuple[float, float]] = None  # pixels per second
    
    # Relationships
    part_of: Optional[str] = None  # If object is part of a larger object
    contains: List[str] = field(default_factory=list)  # Objects inside this one
    
    # Multiple possibilities
    alternatives: List[Tuple[ObjectCategory, float]] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    frame_number: int = 0
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'category': self.category.value,
            'confidence': float(self.confidence),
            'bbox': [int(x) for x in self.bounding_box],
            'center': [int(x) for x in self.center],
            'size': [int(x) for x in self.size],
            'area': float(self.area),
            'attributes': [(a.value, float(s)) for a, s in self.attributes],
            'object_id': self.object_id,
            'text': self.text_detected,
            'frame_number': self.frame_number
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DetectedObject':
        """Create DetectedObject from dictionary"""
        bbox = tuple(data.get('bbox', [0, 0, 0, 0]))
        center = tuple(data.get('center', [0, 0]))
        size = tuple(data.get('size', [0, 0]))
        
        return cls(
            category=ObjectCategory(data.get('category', 'unknown')),
            confidence=data.get('confidence', 0.0),
            bounding_box=bbox,
            center=center,
            size=size,
            area=data.get('area', 0.0),
            attributes=[(ObjectAttribute(a), s) for a, s in data.get('attributes', [])],
            object_id=data.get('object_id'),
            text_detected=data.get('text'),
            frame_number=data.get('frame_number', 0)
        )

@dataclass
class Scene:
    """Complete scene understanding"""
    scene_type: SceneType
    confidence: float
    
    # Objects
    objects: List[DetectedObject] = field(default_factory=list)
    object_count: int = 0
    unique_categories: Set[ObjectCategory] = field(default_factory=set)
    
    # Spatial layout
    layout: Dict[str, Any] = field(default_factory=dict)
    # e.g., {'walls': [...], 'floor_area': [...], 'furniture_arrangement': ...}
    
    # Lighting
    brightness: float = 0.0  # 0-1
    lighting_quality: str = "normal"  # dark, dim, normal, bright, harsh
    light_sources: List[Tuple[int, int, int, int]] = field(default_factory=list)
    
    # People
    people_count: int = 0
    faces_detected: List[Dict] = field(default_factory=list)
    gaze_direction: Optional[str] = None  # where people are looking
    
    # Motion
    motion_areas: List[Tuple[int, int, int, int]] = field(default_factory=list)
    motion_intensity: float = 0.0
    
    # Scene properties
    is_indoor: bool = True
    is_crowded: bool = False
    is_cluttered: bool = False
    is_orderly: bool = True
    
    # Wednesday's observations
    anomalies: List[DetectedObject] = field(default_factory=list)  # Out of place
    points_of_interest: List[DetectedObject] = field(default_factory=list)
    threats: List[DetectedObject] = field(default_factory=list)
    changes: List[Dict] = field(default_factory=list)  # Changes from previous frames
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    frame_number: int = 0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'scene_type': self.scene_type.value,
            'confidence': float(self.confidence),
            'object_count': self.object_count,
            'people_count': self.people_count,
            'is_indoor': self.is_indoor,
            'is_crowded': self.is_crowded,
            'is_cluttered': self.is_cluttered,
            'brightness': float(self.brightness),
            'lighting_quality': self.lighting_quality,
            'anomaly_count': len(self.anomalies),
            'threat_count': len(self.threats),
            'frame_number': self.frame_number,
            'timestamp': self.timestamp.isoformat(),
            'processing_time': float(self.processing_time)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Scene':
        """Create Scene from dictionary"""
        return cls(
            scene_type=SceneType(data.get('scene_type', 'unknown')),
            confidence=data.get('confidence', 0.0),
            objects=[],  # Objects would need separate loading
            object_count=data.get('object_count', 0),
            people_count=data.get('people_count', 0),
            brightness=data.get('brightness', 0.0),
            lighting_quality=data.get('lighting_quality', 'normal'),
            is_indoor=data.get('is_indoor', True),
            is_crowded=data.get('is_crowded', False),
            is_cluttered=data.get('is_cluttered', False),
            frame_number=data.get('frame_number', 0),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            processing_time=data.get('processing_time', 0.0)
        )

class ObjectRecognizer:
    """
    Identifies objects in images/video; scene understanding.
    Wednesday's gaze is sharp - she catalogs everything in a glance.
    Nothing escapes her notice, no detail too small.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Load object detection model
        self.detection_model = None
        self.device = self._setup_device()
        self.coco_classes = []
        self._load_detection_model()
        
        # Load scene classification model
        self.scene_model = None
        self._load_scene_model()
        
        # Load face detection model
        self.face_model = None
        self._load_face_model()
        
        # Object tracking
        self.trackers = {}
        self.next_object_id = 0
        self.tracking_history = defaultdict(list)
        
        # Scene memory for change detection
        self.previous_scene = None
        self.scene_memory = deque(maxlen=10)  # Last 10 scenes
        
        # Object relationships database
        self.typical_relationships = self._load_typical_relationships()
        
        # Attributes classifiers
        self.attribute_classifiers = self._init_attribute_classifiers()
        
        # Performance settings
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.enable_tracking = self.config.get('enable_tracking', True)
        self.enable_face_detection = self.config.get('enable_face_detection', True)
        
        # Image dimensions for position classification
        self.image_width = self.config.get('image_width', 640)
        self.image_height = self.config.get('image_height', 480)
        
        # Performance tracking
        self.stats = {
            'total_frames': 0,
            'total_objects': 0,
            'avg_objects_per_frame': 0.0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0,
            'errors': 0,
            'categories_found': defaultdict(int)
        }
        
        logger.info(f"ObjectRecognizer initialized on device: {self.device}")
    
    def recognize(self, 
                 image: np.ndarray,
                 frame_number: int = 0,
                 context: Optional[Dict] = None) -> Scene:
        """
        Recognize objects and understand scene in image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            frame_number: Frame number for video sequences
            context: Optional context (location, time, etc.)
            
        Returns:
            Scene with detected objects and understanding
        """
        import time
        start_time = time.time()
        
        if not HAS_VISION:
            logger.error("Computer vision libraries not available")
            return Scene(scene_type=SceneType.UNKNOWN, confidence=0.0)
        
        if image is None or image.size == 0:
            logger.warning("Empty image provided")
            return Scene(scene_type=SceneType.UNKNOWN, confidence=0.0)
        
        try:
            # Store image dimensions for position classification
            if len(image.shape) >= 2:
                self.image_height, self.image_width = image.shape[:2]
            
            # Convert to RGB if needed (models expect RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Run object detection
            detections = self._detect_objects(rgb_image)
            
            # Filter by confidence
            detections = [d for d in detections if d['confidence'] >= self.confidence_threshold]
            
            # Convert to DetectedObject objects
            objects = []
            for det in detections:
                obj = self._create_detected_object(det, frame_number, rgb_image.shape)
                objects.append(obj)
                
                # Update stats
                self.stats['categories_found'][obj.category.value] += 1
            
            # Run face detection if enabled
            faces = []
            if self.enable_face_detection and self.face_model is not None:
                faces = self._detect_faces(rgb_image)
            
            # Update object tracking
            if self.enable_tracking and objects:
                objects = self._update_tracking(objects, frame_number)
            
            # Detect scene type
            scene_type, scene_conf = self._classify_scene(rgb_image, objects)
            
            # Analyze spatial layout
            layout = self._analyze_layout(image, objects)
            
            # Analyze lighting
            lighting = self._analyze_lighting(image)
            
            # Detect motion (if previous frame available)
            motion_areas = []
            motion_intensity = 0.0
            if self.previous_scene is not None and frame_number > 0:
                motion_areas, motion_intensity = self._detect_motion(
                    image, self.previous_scene
                )
            
            # Count people
            people_count = sum(1 for obj in objects if obj.category == ObjectCategory.PERSON)
            
            # Create scene object
            scene = Scene(
                scene_type=scene_type,
                confidence=scene_conf,
                objects=objects,
                object_count=len(objects),
                unique_categories=set(obj.category for obj in objects),
                layout=layout,
                brightness=lighting['brightness'],
                lighting_quality=lighting['quality'],
                light_sources=lighting['sources'],
                people_count=people_count,
                faces_detected=faces,
                motion_areas=motion_areas,
                motion_intensity=motion_intensity,
                is_indoor=scene_type.value.startswith('indoor') if scene_type else True,
                is_crowded=people_count > 5,
                is_cluttered=len(objects) > 20,
                is_orderly=self._assess_orderliness(objects, layout),
                frame_number=frame_number,
                processing_time=time.time() - start_time
            )
            
            # Wednesday's special analysis
            scene = self._apply_wednesday_analysis(scene, context)
            
            # Detect changes from previous scene
            if self.previous_scene is not None:
                scene.changes = self._detect_changes(scene, self.previous_scene)
            
            # Store in memory
            self.previous_scene = scene
            self.scene_memory.append(scene)
            
            # Update stats
            self._update_stats(scene)
            
            return scene
            
        except Exception as e:
            logger.error(f"Error in object recognition: {e}", exc_info=True)
            self.stats['errors'] += 1
            return Scene(scene_type=SceneType.UNKNOWN, confidence=0.0)
    
    def recognize_video_frame(self,
                             frame: np.ndarray,
                             frame_number: int,
                             previous_scene: Optional[Scene] = None) -> Scene:
        """
        Recognize objects in a video frame with temporal context.
        """
        if previous_scene is not None:
            self.previous_scene = previous_scene
        
        return self.recognize(frame, frame_number)
    
    def get_object_at_position(self, 
                              scene: Scene, 
                              x: int, 
                              y: int) -> Optional[DetectedObject]:
        """
        Find object at specific image coordinates.
        """
        for obj in scene.objects:
            x1, y1, x2, y2 = obj.bounding_box
            if x1 <= x <= x2 and y1 <= y <= y2:
                return obj
        return None
    
    def get_objects_by_category(self, 
                               scene: Scene, 
                               category: ObjectCategory) -> List[DetectedObject]:
        """
        Get all objects of a specific category.
        """
        return [obj for obj in scene.objects if obj.category == category]
    
    def _setup_device(self) -> str:
        """Setup computation device (CUDA/CPU)"""
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    def _load_detection_model(self):
        """Load object detection model"""
        try:
            # Use a pre-trained model (e.g., Faster R-CNN, YOLO, DETR)
            model_name = self.config.get('detection_model', 'fasterrcnn_resnet50_fpn')
            
            if model_name == 'fasterrcnn_resnet50_fpn':
                self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                    weights='DEFAULT'
                )
            elif model_name == 'retinanet_resnet50_fpn':
                self.detection_model = torchvision.models.detection.retinanet_resnet50_fpn(
                    weights='DEFAULT'
                )
            else:
                # Default to faster R-CNN
                self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                    weights='DEFAULT'
                )
            
            self.detection_model.eval()
            self.detection_model.to(self.device)
            
            # Load COCO class names
            self.coco_classes = self._load_coco_classes()
            
            logger.info(f"Loaded detection model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load detection model: {e}")
            self.detection_model = None
    
    def _load_scene_model(self):
        """Load scene classification model"""
        try:
            # Use a pre-trained scene classification model
            # Placeholder - would load actual model
            self.scene_model = {
                'name': 'places365',
                'classes': self._load_places_classes()
            }
            logger.info("Loaded scene classification model")
        except Exception as e:
            logger.error(f"Failed to load scene model: {e}")
            self.scene_model = None
    
    def _load_face_model(self):
        """Load face detection model"""
        try:
            # Use OpenCV's face detector
            self.face_model = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("Loaded face detection model")
        except Exception as e:
            logger.error(f"Failed to load face model: {e}")
            self.face_model = None
    
    def _load_coco_classes(self) -> List[str]:
        """Load COCO dataset class names"""
        return [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'N/A', 'backpack', 'umbrella', 'N/A', 'handbag', 'tie', 'suitcase',
            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'N/A', 'dining table', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def _load_places_classes(self) -> List[str]:
        """Load Places365 scene class names"""
        # Simplified list
        return [
            'airport', 'art_gallery', 'auditorium', 'bakery', 'bathroom',
            'bedroom', 'bookstore', 'bowling_alley', 'cafeteria', 'classroom',
            'closet', 'computer_room', 'conference_room', 'corridor', 'dining_room',
            'drugstore', 'elevator', 'garage', 'grocery_store', 'gym',
            'hallway', 'home_office', 'hospital', 'hotel_room', 'kitchen',
            'laundromat', 'library', 'living_room', 'lobby', 'movie_theater',
            'museum', 'office', 'park', 'parking_lot', 'patio',
            'restaurant', 'shop', 'staircase', 'street', 'subway'
        ]
    
    def _load_typical_relationships(self) -> Dict:
        """Load typical object relationships for context"""
        return {
            'chair': {'near': ['table', 'desk'], 'on': ['floor'], 'under': []},
            'book': {'on': ['table', 'shelf', 'desk'], 'in': ['bag', 'backpack']},
            'cup': {'on': ['table', 'desk'], 'in': ['sink', 'dishwasher']},
            'laptop': {'on': ['table', 'desk', 'lap'], 'near': ['mouse', 'keyboard']},
            'phone': {'on': ['table', 'bed', 'hand'], 'near': ['charger']},
            'tv': {'on': ['wall', 'stand'], 'facing': ['sofa', 'bed']},
            'knife': {'on': ['cutting board', 'counter'], 'in': ['drawer', 'block']},
            'scissors': {'in': ['drawer', 'desk'], 'on': ['table']}
        }
    
    def _init_attribute_classifiers(self) -> Dict:
        """Initialize attribute classifiers"""
        # Placeholder - would load ML models for attributes
        return {
            'color': self._classify_color,
            'size': self._classify_size,
            'position': self._classify_position
        }
    
    def _detect_objects(self, image: np.ndarray) -> List[Dict]:
        """Run object detection on image"""
        if self.detection_model is None:
            return []
        
        try:
            # Prepare image for model
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.detection_model(image_tensor)
            
            # Process predictions
            detections = []
            pred = predictions[0]
            
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            for i in range(len(boxes)):
                if scores[i] >= self.confidence_threshold:
                    # Convert COCO label to ObjectCategory
                    category = self._coco_to_category(int(labels[i]))
                    
                    detections.append({
                        'bbox': boxes[i].astype(int).tolist(),
                        'confidence': float(scores[i]),
                        'category': category,
                        'label': int(labels[i])
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def _coco_to_category(self, label: int) -> ObjectCategory:
        """Convert COCO label to ObjectCategory"""
        mapping = {
            1: ObjectCategory.PERSON,
            2: ObjectCategory.BICYCLE,
            3: ObjectCategory.CAR,
            4: ObjectCategory.MOTORCYCLE,
            6: ObjectCategory.BUS,
            8: ObjectCategory.TRUCK,
            16: ObjectCategory.PET,  # bird
            17: ObjectCategory.PET,  # cat
            18: ObjectCategory.PET,  # dog
            19: ObjectCategory.ANIMAL,  # horse
            21: ObjectCategory.ANIMAL,  # elephant
            24: ObjectCategory.ANIMAL,  # zebra
            25: ObjectCategory.ANIMAL,  # giraffe
            27: ObjectCategory.BAG,
            28: ObjectCategory.CLOTHING,  # tie
            30: ObjectCategory.CLOTHING,  # handbag
            31: ObjectCategory.CLOTHING,  # tie
            32: ObjectCategory.BAG,  # suitcase
            39: ObjectCategory.BOTTLE,
            40: ObjectCategory.CUP,  # wine glass
            41: ObjectCategory.CUP,
            42: ObjectCategory.UTENSIL,  # fork
            43: ObjectCategory.KNIFE,  # knife
            44: ObjectCategory.UTENSIL,  # spoon
            45: ObjectCategory.BOWL,
            46: ObjectCategory.FOOD,  # banana
            47: ObjectCategory.FOOD,  # apple
            48: ObjectCategory.FOOD,  # sandwich
            49: ObjectCategory.FOOD,  # orange
            50: ObjectCategory.FOOD,  # broccoli
            51: ObjectCategory.FOOD,  # carrot
            52: ObjectCategory.FOOD,  # hot dog
            53: ObjectCategory.FOOD,  # pizza
            54: ObjectCategory.FOOD,  # donut
            55: ObjectCategory.FOOD,  # cake
            56: ObjectCategory.CHAIR,
            57: ObjectCategory.SOFA,
            58: ObjectCategory.PLANT,
            59: ObjectCategory.BED,
            61: ObjectCategory.TABLE,
            62: ObjectCategory.TABLE,  # dining table
            63: ObjectCategory.APPLIANCE,  # toilet
            64: ObjectCategory.TV,  # tv
            65: ObjectCategory.LAPTOP,
            66: ObjectCategory.MOUSE,
            67: ObjectCategory.REMOTE,  # remote
            68: ObjectCategory.KEYBOARD,
            69: ObjectCategory.PHONE,
            70: ObjectCategory.APPLIANCE,  # microwave
            71: ObjectCategory.APPLIANCE,  # oven
            72: ObjectCategory.APPLIANCE,  # toaster
            73: ObjectCategory.APPLIANCE,  # sink
            74: ObjectCategory.APPLIANCE,  # refrigerator
            75: ObjectCategory.BOOK,
            76: ObjectCategory.CLOCK,
            77: ObjectCategory.VASE,
            78: ObjectCategory.SCISSORS,
            79: ObjectCategory.TOY,  # teddy bear
            80: ObjectCategory.APPLIANCE,  # hair drier
            81: ObjectCategory.APPLIANCE,  # toothbrush
        }
        
        return mapping.get(label, ObjectCategory.UNKNOWN)
    
    def _create_detected_object(self, detection: Dict, frame_number: int, 
                               image_shape: Tuple) -> DetectedObject:
        """Create DetectedObject from detection data"""
        x1, y1, x2, y2 = detection['bbox']
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        center = (x1 + width // 2, y1 + height // 2)
        area = width * height
        
        # Determine attributes
        attributes = []
        
        # Size attribute
        img_area = image_shape[0] * image_shape[1] if len(image_shape) >= 2 else 640*480
        area_ratio = area / img_area
        
        if area_ratio < 0.01:
            attributes.append((ObjectAttribute.TINY, 1.0))
        elif area_ratio < 0.05:
            attributes.append((ObjectAttribute.SMALL, 1.0))
        elif area_ratio < 0.15:
            attributes.append((ObjectAttribute.MEDIUM, 1.0))
        elif area_ratio < 0.3:
            attributes.append((ObjectAttribute.LARGE, 1.0))
        else:
            attributes.append((ObjectAttribute.HUGE, 1.0))
        
        # Position attribute
        img_center_x = self.image_width // 2
        margin = self.image_width // 4
        
        if center[0] < img_center_x - margin:
            attributes.append((ObjectAttribute.LEFT, 1.0))
        elif center[0] > img_center_x + margin:
            attributes.append((ObjectAttribute.RIGHT, 1.0))
        else:
            attributes.append((ObjectAttribute.CENTER, 1.0))
        
        # Foreground/background (simplified - based on size and position)
        if area_ratio > 0.1 and abs(center[0] - img_center_x) < self.image_width // 3:
            attributes.append((ObjectAttribute.FOREGROUND, 0.8))
        else:
            attributes.append((ObjectAttribute.BACKGROUND, 0.7))
        
        return DetectedObject(
            category=detection['category'],
            confidence=detection['confidence'],
            bounding_box=(x1, y1, x2, y2),
            center=center,
            size=(width, height),
            area=area,
            attributes=attributes,
            frame_number=frame_number
        )
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces in image"""
        if self.face_model is None:
            return []
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.face_model.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            results = []
            for (x, y, w, h) in faces:
                results.append({
                    'bbox': (x, y, x+w, y+h),
                    'center': (x + w//2, y + h//2),
                    'size': (w, h),
                    'confidence': 0.9  # OpenCV doesn't provide confidence
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def _classify_scene(self, 
                       image: np.ndarray, 
                       objects: List[DetectedObject]) -> Tuple[SceneType, float]:
        """Classify scene type"""
        if self.scene_model is None:
            # Guess based on objects
            return self._guess_scene_from_objects(objects)
        
        try:
            # Would run actual scene classification
            # Placeholder implementation
            return self._guess_scene_from_objects(objects)
            
        except Exception as e:
            logger.error(f"Scene classification failed: {e}")
            return SceneType.UNKNOWN, 0.0
    
    def _guess_scene_from_objects(self, 
                                 objects: List[DetectedObject]) -> Tuple[SceneType, float]:
        """Guess scene type based on detected objects"""
        categories = [obj.category for obj in objects]
        
        # Check for indoor indicators
        indoor_objects = {ObjectCategory.BED, ObjectCategory.SOFA, 
                          ObjectCategory.TABLE, ObjectCategory.CHAIR,
                          ObjectCategory.APPLIANCE, ObjectCategory.OVEN,
                          ObjectCategory.REFRIGERATOR}
        
        outdoor_objects = {ObjectCategory.CAR, ObjectCategory.BICYCLE,
                          ObjectCategory.TREE, ObjectCategory.BENCH,
                          ObjectCategory.MOTORCYCLE, ObjectCategory.BUS}
        
        indoor_count = sum(1 for cat in categories if cat in indoor_objects)
        outdoor_count = sum(1 for cat in categories if cat in outdoor_objects)
        
        if indoor_count > outdoor_count:
            # Specific indoor type
            if ObjectCategory.BED in categories:
                return SceneType.BEDROOM, 0.7
            elif ObjectCategory.OVEN in categories or ObjectCategory.REFRIGERATOR in categories:
                return SceneType.KITCHEN, 0.7
            elif ObjectCategory.SOFA in categories:
                return SceneType.LIVING_ROOM, 0.6
            elif ObjectCategory.DESK in categories or ObjectCategory.KEYBOARD in categories:
                return SceneType.OFFICE, 0.6
            elif ObjectCategory.TOILET in categories or ObjectCategory.SINK in categories:
                return SceneType.BATHROOM, 0.7
            else:
                return SceneType.INDOOR, 0.5
        else:
            # Outdoor
            if ObjectCategory.CAR in categories or ObjectCategory.BUS in categories:
                return SceneType.STREET, 0.6
            elif ObjectCategory.TREE in categories or ObjectCategory.BENCH in categories:
                return SceneType.PARK, 0.5
            else:
                return SceneType.OUTDOOR, 0.5
    
    def _analyze_layout(self, image: np.ndarray, objects: List[DetectedObject]) -> Dict:
        """Analyze spatial layout of scene"""
        layout = {
            'floor_area': None,
            'walls': [],
            'furniture_arrangement': []
        }
        
        # Would implement spatial analysis
        # Placeholder
        return layout
    
    def _analyze_lighting(self, image: np.ndarray) -> Dict:
        """Analyze lighting conditions"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate brightness
        brightness = float(np.mean(gray) / 255.0)
        
        # Determine lighting quality
        if brightness < 0.2:
            quality = "dark"
        elif brightness < 0.4:
            quality = "dim"
        elif brightness < 0.7:
            quality = "normal"
        elif brightness < 0.9:
            quality = "bright"
        else:
            quality = "harsh"
        
        # Detect light sources (bright spots)
        _, bright_spots = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        light_sources = []
        
        # Find contours of bright spots
        contours, _ = cv2.findContours(bright_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Minimum size
                x, y, w, h = cv2.boundingRect(contour)
                light_sources.append((x, y, x+w, y+h))
        
        return {
            'brightness': brightness,
            'quality': quality,
            'sources': light_sources
        }
    
    def _detect_motion(self, 
                      current_image: np.ndarray, 
                      previous_scene: Scene) -> Tuple[List, float]:
        """Detect motion between frames"""
        # Would implement motion detection
        # Placeholder
        return [], 0.0
    
    def _update_tracking(self, 
                        objects: List[DetectedObject], 
                        frame_number: int) -> List[DetectedObject]:
        """Update object tracking across frames"""
        if not self.trackers:
            # First frame, assign IDs
            for obj in objects:
                obj.object_id = f"obj_{self.next_object_id}"
                self.next_object_id += 1
                self.trackers[obj.object_id] = obj
            return objects
        
        # Match objects with existing trackers
        # Simplified IoU-based matching
        matched_objects = []
        
        for obj in objects:
            best_match = None
            best_iou = 0.3  # Threshold
            
            for tracker_id, tracker_obj in self.trackers.items():
                iou = self._calculate_iou(obj.bounding_box, tracker_obj.bounding_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = tracker_id
            
            if best_match:
                obj.object_id = best_match
                # Update tracker with latest object
                self.trackers[best_match] = obj
                # Store in history
                self.tracking_history[best_match].append((frame_number, obj.center))
            else:
                obj.object_id = f"obj_{self.next_object_id}"
                self.next_object_id += 1
                self.trackers[obj.object_id] = obj
                self.tracking_history[obj.object_id] = [(frame_number, obj.center)]
            
            matched_objects.append(obj)
        
        return matched_objects
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union of two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _classify_color(self, obj: DetectedObject, image: np.ndarray) -> Tuple[int, int, int]:
        """Classify average color of object"""
        x1, y1, x2, y2 = obj.bounding_box
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return (128, 128, 128)  # Gray
        
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return (128, 128, 128)
        
        avg_color = np.mean(roi, axis=(0, 1)).astype(int)
        return tuple(avg_color)
    
    def _classify_size(self, obj: DetectedObject) -> ObjectAttribute:
        """Classify size attribute"""
        area = obj.area
        img_area = self.image_width * self.image_height
        ratio = area / img_area
        
        if ratio < 0.01:
            return ObjectAttribute.TINY
        elif ratio < 0.05:
            return ObjectAttribute.SMALL
        elif ratio < 0.15:
            return ObjectAttribute.MEDIUM
        elif ratio < 0.3:
            return ObjectAttribute.LARGE
        else:
            return ObjectAttribute.HUGE
    
    def _classify_position(self, obj: DetectedObject) -> ObjectAttribute:
        """Classify position attribute"""
        center_x = obj.center[0]
        img_center = self.image_width // 2
        margin = self.image_width // 4
        
        if center_x < img_center - margin:
            return ObjectAttribute.LEFT
        elif center_x > img_center + margin:
            return ObjectAttribute.RIGHT
        else:
            return ObjectAttribute.CENTER
    
    def _assess_orderliness(self, 
                          objects: List[DetectedObject], 
                          layout: Dict) -> bool:
        """Assess if scene is orderly or cluttered"""
        # Simplified - would use spatial analysis
        return len(objects) < 15
    
    def _detect_changes(self, 
                       current_scene: Scene, 
                       previous_scene: Scene) -> List[Dict]:
        """Detect changes between scenes"""
        changes = []
        
        # Object appearance/disappearance
        current_ids = {obj.object_id for obj in current_scene.objects if obj.object_id}
        previous_ids = {obj.object_id for obj in previous_scene.objects if obj.object_id}
        
        new_objects = current_ids - previous_ids
        missing_objects = previous_ids - current_ids
        
        for obj_id in new_objects:
            obj = next(obj for obj in current_scene.objects if obj.object_id == obj_id)
            changes.append({
                'type': 'appeared',
                'object': obj.category.value,
                'object_id': obj_id,
                'position': obj.center,
                'confidence': obj.confidence
            })
        
        for obj_id in missing_objects:
            obj = next(obj for obj in previous_scene.objects if obj.object_id == obj_id)
            changes.append({
                'type': 'disappeared',
                'object': obj.category.value,
                'object_id': obj_id,
                'last_position': obj.center,
                'last_seen': obj.frame_number
            })
        
        # Object movement
        for obj_id in current_ids & previous_ids:
            current = next(obj for obj in current_scene.objects if obj.object_id == obj_id)
            previous = next(obj for obj in previous_scene.objects if obj.object_id == obj_id)
            
            # Calculate movement distance
            dist = np.sqrt(
                (current.center[0] - previous.center[0])**2 + 
                (current.center[1] - previous.center[1])**2
            )
            
            if dist > 50:  # Significant movement
                changes.append({
                    'type': 'moved',
                    'object': current.category.value,
                    'object_id': obj_id,
                    'from': previous.center,
                    'to': current.center,
                    'distance': float(dist)
                })
        
        return changes
    
    def _apply_wednesday_analysis(self, 
                                 scene: Scene, 
                                 context: Optional[Dict]) -> Scene:
        """
        Apply Wednesday's special analytical lens.
        She notices anomalies and potential threats.
        """
        # Detect anomalies (objects out of place)
        for obj in scene.objects:
            # Check if object is in expected location
            if context and 'expected_objects' in context:
                expected = context['expected_objects']
                if obj.category.value not in expected:
                    scene.anomalies.append(obj)
            
            # Check for threatening objects
            threat_categories = [
                ObjectCategory.KNIFE,
                ObjectCategory.SCISSORS,
                # Add more as appropriate
            ]
            
            if obj.category in threat_categories:
                # Check if in unusual location (not in drawer/block)
                # This would require more context
                scene.threats.append(obj)
            
            # Points of interest (unusual or rare objects)
            rare_categories = [
                ObjectCategory.JEWELRY,
                ObjectCategory.WATCH,
                ObjectCategory.WALLET,
                # Add more
            ]
            
            if obj.category in rare_categories:
                scene.points_of_interest.append(obj)
        
        # Check for people behaving unusually
        if scene.people_count > 0:
            # Would analyze pose/gait
            pass
        
        # Check for overall scene anomalies
        if context and context.get('expected_scene_type'):
            if scene.scene_type != context['expected_scene_type']:
                # Scene type changed unexpectedly
                scene.anomalies.append(None)  # Placeholder for scene-level anomaly
        
        return scene
    
    def _update_stats(self, scene: Scene):
        """Update recognition statistics"""
        self.stats['total_frames'] += 1
        self.stats['total_objects'] += scene.object_count
        
        # Update average objects per frame
        total = self.stats['total_frames']
        old_avg = self.stats['avg_objects_per_frame']
        self.stats['avg_objects_per_frame'] = old_avg + (scene.object_count - old_avg) / total
        
        # Update average processing time
        old_time = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = old_time + (scene.processing_time - old_time) / total
        
        # Update average confidence
        if scene.objects:
            avg_conf = float(np.mean([obj.confidence for obj in scene.objects]))
            old_conf = self.stats['avg_confidence']
            self.stats['avg_confidence'] = old_conf + (avg_conf - old_conf) / total
    
    def get_stats(self) -> Dict:
        """Return recognition statistics"""
        stats = dict(self.stats)
        stats['categories_found'] = dict(stats['categories_found'])
        return stats
    
    def reset_stats(self) -> None:
        """Reset recognition statistics"""
        self.stats = {
            'total_frames': 0,
            'total_objects': 0,
            'avg_objects_per_frame': 0.0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0,
            'errors': 0,
            'categories_found': defaultdict(int)
        }

# Connects to: perception/attention/salience.py (visual stimuli attract attention)
# Connects to: memory/episodic/ (stores visual scenes in episodic memory)
# Connects to: memory/semantic/ (object knowledge and relationships)
# Connects to: self/theory_of_mind.py (understanding what others see)
# Connects to: cognition/reasoning.py (spatial reasoning about objects)
# Connects to: emotion/appraisal.py (threatening objects trigger emotional response)