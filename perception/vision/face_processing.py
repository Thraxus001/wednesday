"""
Face detection; emotion from expressions; identity recognition.
Wednesday reads faces like others read books - every micro-expression,
every fleeting emotion, every carefully constructed mask.
The face is a window to truths people try to hide.
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
    logging.warning(f"Computer vision libraries not available: {e}. Install opencv-python, torch")

# Optional: face_recognition library
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False
    logging.warning("face_recognition library not available. Install face_recognition for better accuracy")

logger = logging.getLogger(__name__)

class FacialExpression(Enum):
    """Basic facial expressions"""
    # Primary emotions
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    CONTEMPT = "contempt"
    
    # Complex emotions
    INTEREST = "interest"
    BOREDOM = "boredom"
    CONFUSION = "confusion"
    SUSPICION = "suspicion"
    SKEPTICISM = "skepticism"
    AMUSEMENT = "amusement"
    EMBARRASSMENT = "embarrassment"
    PRIDE = "pride"
    SHAME = "shame"
    GUILT = "guilt"
    
    # Wednesday specials
    DARK_AMUSEMENT = "dark_amusement"  # When she finds something morbidly funny
    DEADPAN = "deadpan"  # Her signature expression
    MILD_INTEREST = "mild_interest"  # The most emotion she typically shows
    CALCULATING = "calculating"  # When she's thinking several steps ahead
    
    # Deception indicators
    MICRO_EXPRESSION = "micro_expression"  # Fleeting genuine emotion
    MASKED_EMOTION = "masked_emotion"  # Deliberately hidden feeling

class FacialFeature(Enum):
    """Specific facial features"""
    EYES = "eyes"
    EYEBROWS = "eyebrows"
    NOSE = "nose"
    MOUTH = "mouth"
    JAW = "jaw"
    FOREHEAD = "forehead"
    CHEEKS = "cheeks"
    CHIN = "chin"

class GazeDirection(Enum):
    """Direction of gaze"""
    TOWARDS_CAMERA = "towards_camera"
    AWAY_CAMERA = "away_camera"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    AVOIDING = "avoiding"  # Deliberately not looking
    SEARCHING = "searching"  # Looking around

class HeadPose(Enum):
    """Head orientation"""
    FRONTAL = "frontal"
    PROFILE_LEFT = "profile_left"
    PROFILE_RIGHT = "profile_right"
    TILTED_UP = "tilted_up"
    TILTED_DOWN = "tilted_down"
    TURNED_AWAY = "turned_away"

@dataclass
class FacialLandmarks:
    """Key points on the face"""
    # 68-point landmark model
    jaw: List[Tuple[int, int]] = field(default_factory=list)  # Points 0-16
    left_eyebrow: List[Tuple[int, int]] = field(default_factory=list)  # 17-21
    right_eyebrow: List[Tuple[int, int]] = field(default_factory=list)  # 22-26
    nose_bridge: List[Tuple[int, int]] = field(default_factory=list)  # 27-30
    nose_tip: List[Tuple[int, int]] = field(default_factory=list)  # 31-35
    left_eye: List[Tuple[int, int]] = field(default_factory=list)  # 36-41
    right_eye: List[Tuple[int, int]] = field(default_factory=list)  # 42-47
    outer_lips: List[Tuple[int, int]] = field(default_factory=list)  # 48-59
    inner_lips: List[Tuple[int, int]] = field(default_factory=list)  # 60-67
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'jaw': self.jaw[:10],  # Limit for storage
            'left_eyebrow': self.left_eyebrow,
            'right_eyebrow': self.right_eyebrow,
            'left_eye': self.left_eye,
            'right_eye': self.right_eye,
            'outer_lips': self.outer_lips[:10]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FacialLandmarks':
        """Create FacialLandmarks from dictionary"""
        return cls(
            jaw=data.get('jaw', []),
            left_eyebrow=data.get('left_eyebrow', []),
            right_eyebrow=data.get('right_eyebrow', []),
            left_eye=data.get('left_eye', []),
            right_eye=data.get('right_eye', []),
            outer_lips=data.get('outer_lips', [])
        )

@dataclass
class EyeAnalysis:
    """Detailed analysis of eyes"""
    left_eye_open: float = 0.0  # 0-1 how open
    right_eye_open: float = 0.0
    pupil_dilation: float = 0.0  # Normalized pupil size
    gaze_direction: GazeDirection = GazeDirection.TOWARDS_CAMERA
    gaze_vector: Tuple[float, float] = (0.0, 0.0)  # (x, y) direction
    blink_rate: float = 0.0  # blinks per minute
    last_blink: Optional[datetime] = None
    saccades: List[Tuple[float, float]] = field(default_factory=list)  # Rapid eye movements
    micro_expressions: List[Dict] = field(default_factory=list)  # Fleeting expressions
    
    # Deception indicators
    avoids_eye_contact: bool = False
    excessive_blinking: bool = False
    dilated_pupils: bool = False  # Can indicate interest or arousal
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'left_eye_open': float(self.left_eye_open),
            'right_eye_open': float(self.right_eye_open),
            'gaze_direction': self.gaze_direction.value,
            'avoids_eye_contact': self.avoids_eye_contact,
            'excessive_blinking': self.excessive_blinking
        }

@dataclass
class MicroExpression:
    """A fleeting facial expression"""
    expression: FacialExpression
    intensity: float
    duration: float  # seconds
    start_time: float
    confidence: float
    landmarks_delta: Dict[str, float] = field(default_factory=dict)  # Changes in landmark positions
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'expression': self.expression.value,
            'intensity': float(self.intensity),
            'duration': float(self.duration),
            'confidence': float(self.confidence)
        }

@dataclass
class FaceAnalysis:
    """Complete analysis of a detected face"""
    # Basic info
    face_id: str
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    
    # Identity
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    recognition_confidence: float = 0.0
    
    # Demographics (estimated)
    age: Optional[int] = None
    age_range: Tuple[int, int] = (0, 0)
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    
    # Facial features
    landmarks: Optional[FacialLandmarks] = None
    features: Dict[FacialFeature, Any] = field(default_factory=dict)
    
    # Expression analysis
    primary_expression: FacialExpression = FacialExpression.NEUTRAL
    expression_scores: Dict[FacialExpression, float] = field(default_factory=dict)
    expression_intensity: float = 0.0
    micro_expressions: List[MicroExpression] = field(default_factory=list)
    
    # Eye tracking
    eyes: EyeAnalysis = field(default_factory=EyeAnalysis)
    
    # Head pose
    head_pose: HeadPose = HeadPose.FRONTAL
    head_pose_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # yaw, pitch, roll
    
    # Action units (FACS - Facial Action Coding System)
    action_units: Dict[str, float] = field(default_factory=dict)
    # e.g., 'AU4': 0.5 (eyebrow lowerer), 'AU12': 0.8 (lip corner puller)
    
    # Temporal analysis
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    appearance_count: int = 1
    track_id: Optional[str] = None
    
    # Interaction
    speaking: bool = False
    listening: bool = True
    attention_target: Optional[str] = None  # What/who they're looking at
    
    # Wednesday's insights
    deception_indicators: Dict[str, float] = field(default_factory=dict)
    hidden_emotion: Optional[FacialExpression] = None
    trust_score: float = 0.5  # 0-1 how trustworthy they seem
    interesting: bool = False  # Does Wednesday find them interesting?
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    frame_number: int = 0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'face_id': self.face_id,
            'person_id': self.person_id,
            'person_name': self.person_name,
            'confidence': float(self.confidence),
            'recognition_confidence': float(self.recognition_confidence),
            'primary_expression': self.primary_expression.value,
            'expression_intensity': float(self.expression_intensity),
            'head_pose': self.head_pose.value,
            'gaze': self.eyes.gaze_direction.value,
            'age': self.age,
            'gender': self.gender,
            'speaking': self.speaking,
            'trust_score': float(self.trust_score),
            'interesting': self.interesting,
            'frame_number': self.frame_number,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FaceAnalysis':
        """Create FaceAnalysis from dictionary"""
        eyes = EyeAnalysis()
        eyes.gaze_direction = GazeDirection(data.get('gaze', 'towards_camera'))
        
        return cls(
            face_id=data.get('face_id', ''),
            bounding_box=(0, 0, 0, 0),  # Not stored in dict
            confidence=data.get('confidence', 0.0),
            person_id=data.get('person_id'),
            person_name=data.get('person_name'),
            recognition_confidence=data.get('recognition_confidence', 0.0),
            primary_expression=FacialExpression(data.get('primary_expression', 'neutral')),
            expression_intensity=data.get('expression_intensity', 0.0),
            head_pose=HeadPose(data.get('head_pose', 'frontal')),
            eyes=eyes,
            age=data.get('age'),
            gender=data.get('gender'),
            speaking=data.get('speaking', False),
            trust_score=data.get('trust_score', 0.5),
            interesting=data.get('interesting', False),
            frame_number=data.get('frame_number', 0),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now()
        )

@dataclass
class FaceMemory:
    """Memory of a known person"""
    person_id: str
    name: Optional[str]
    face_encodings: List[np.ndarray] = field(default_factory=list)
    first_encounter: datetime = field(default_factory=datetime.now)
    last_encounter: datetime = field(default_factory=datetime.now)
    encounter_count: int = 0
    average_expression: Dict[str, float] = field(default_factory=dict)  # Use string keys for serialization
    typical_gaze: GazeDirection = GazeDirection.TOWARDS_CAMERA
    notes: List[str] = field(default_factory=list)  # Wednesday's observations
    trust_level: float = 0.5  # Based on interactions
    relationship: str = "stranger"  # stranger, acquaintance, friend, etc.
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'person_id': self.person_id,
            'name': self.name,
            'first_encounter': self.first_encounter.isoformat(),
            'last_encounter': self.last_encounter.isoformat(),
            'encounter_count': self.encounter_count,
            'average_expression': self.average_expression,
            'typical_gaze': self.typical_gaze.value,
            'notes': self.notes,
            'trust_level': float(self.trust_level),
            'relationship': self.relationship
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FaceMemory':
        """Create FaceMemory from dictionary"""
        return cls(
            person_id=data['person_id'],
            name=data.get('name'),
            first_encounter=datetime.fromisoformat(data['first_encounter']),
            last_encounter=datetime.fromisoformat(data['last_encounter']),
            encounter_count=data.get('encounter_count', 0),
            average_expression=data.get('average_expression', {}),
            typical_gaze=GazeDirection(data.get('typical_gaze', 'towards_camera')),
            notes=data.get('notes', []),
            trust_level=data.get('trust_level', 0.5),
            relationship=data.get('relationship', 'stranger')
        )

class FaceProcessor:
    """
    Face detection; emotion from expressions; identity recognition.
    Wednesday reads faces with unnerving accuracy.
    She sees the micro-expression that betrays a lie,
    the pupil dilation that reveals interest,
    the subtle tension that masks fear.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Face detection models
        self.face_detector = None
        self.landmark_detector = None
        self.face_recognizer = None
        self.emotion_model = None
        self._load_models()
        
        # Face database
        self.known_faces: Dict[str, FaceMemory] = {}
        self.face_encodings_db = []  # For fast matching
        self.face_names_db = []
        
        # Active tracking
        self.active_faces: Dict[str, FaceAnalysis] = {}  # track_id -> analysis
        self.next_face_id = 0
        self.face_history = defaultdict(list)  # track_id -> list of analyses
        self.max_history_per_face = 100
        
        # Expression analysis
        self.expression_thresholds = self._load_expression_thresholds()
        self.facs_mapping = self._load_facs_mapping()
        
        # Deception detection patterns
        self.deception_patterns = self._load_deception_patterns()
        
        # Performance settings
        self.detection_interval = self.config.get('detection_interval', 1)  # frames
        self.recognition_threshold = self.config.get('recognition_threshold', 0.6)
        self.track_quality_threshold = self.config.get('track_quality_threshold', 0.5)
        self.face_match_threshold = self.config.get('face_match_threshold', 0.6)
        
        # Performance tracking
        self.stats = {
            'total_faces_detected': 0,
            'unique_persons': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0,
            'errors': 0,
            'expressions_detected': defaultdict(int)
        }
        
        logger.info("FaceProcessor initialized")
    
    def process_frame(self,
                     frame: np.ndarray,
                     frame_number: int = 0,
                     context: Optional[Dict] = None) -> List[FaceAnalysis]:
        """
        Process a video frame for faces.
        
        Args:
            frame: Input image (BGR format)
            frame_number: Frame number for video
            context: Optional context (location, conversation, etc.)
            
        Returns:
            List of FaceAnalysis for each detected face
        """
        import time
        start_time = time.time()
        
        if not HAS_VISION:
            logger.error("Computer vision libraries not available")
            return []
        
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided")
            return []
        
        try:
            # Convert to RGB for face_recognition
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            # Detect faces
            face_locations = self._detect_faces(rgb_frame)
            
            if not face_locations:
                # No faces, update tracking for faces that left
                self._cleanup_lost_faces()
                return []
            
            # Extract face encodings
            face_encodings = self._get_face_encodings(rgb_frame, face_locations)
            
            # Get facial landmarks
            landmarks_batch = self._get_face_landmarks(rgb_frame, face_locations)
            
            # Process each face
            faces = []
            for i, (loc, encoding, landmarks) in enumerate(zip(
                face_locations, face_encodings, landmarks_batch
            )):
                # Convert location to (x1, y1, x2, y2) format
                # face_recognition returns (top, right, bottom, left)
                bbox = (loc[3], loc[0], loc[1], loc[2])
                
                # Recognize person
                person_id, person_name, recog_conf = self._recognize_face(encoding)
                
                # Track face across frames
                track_id = self._track_face(bbox, encoding, person_id)
                
                # Analyze expression
                expression_scores = self._analyze_expression(landmarks, rgb_frame, bbox)
                primary_expr, expr_conf = self._get_primary_expression(expression_scores)
                
                # Analyze eyes and gaze
                eye_analysis = self._analyze_eyes(landmarks, rgb_frame, bbox)
                
                # Estimate head pose
                head_pose, pose_angles = self._estimate_head_pose(landmarks)
                
                # Extract action units
                action_units = self._extract_action_units(landmarks, expression_scores)
                
                # Detect micro-expressions
                micro_exprs = self._detect_micro_expressions(
                    track_id, landmarks, expression_scores
                )
                
                # Check for deception indicators
                deception = self._detect_deception(
                    eye_analysis, micro_exprs, action_units, context
                )
                
                # Estimate demographics
                age, gender = self._estimate_demographics(landmarks, rgb_frame, bbox)
                
                # Get or create face ID
                if track_id in self.active_faces:
                    face_id = self.active_faces[track_id].face_id
                else:
                    face_id = f"face_{self.next_face_id}"
                    self.next_face_id += 1
                
                # Create face analysis
                face = FaceAnalysis(
                    face_id=face_id,
                    bounding_box=bbox,
                    confidence=0.95,  # Default confidence
                    person_id=person_id,
                    person_name=person_name,
                    recognition_confidence=recog_conf,
                    age=age,
                    gender=gender,
                    landmarks=landmarks,
                    primary_expression=primary_expr,
                    expression_scores=expression_scores,
                    expression_intensity=expr_conf,
                    micro_expressions=micro_exprs,
                    eyes=eye_analysis,
                    head_pose=head_pose,
                    head_pose_angles=pose_angles,
                    action_units=action_units,
                    track_id=track_id,
                    speaking=self._detect_speaking(landmarks, action_units),
                    deception_indicators=deception,
                    hidden_emotion=self._infer_hidden_emotion(
                        expression_scores, micro_exprs, deception
                    ),
                    trust_score=self._calculate_trust_score(
                        person_id, deception, expression_scores
                    ),
                    interesting=self._is_interesting(expression_scores, deception, context),
                    frame_number=frame_number,
                    last_seen=datetime.now(),
                    timestamp=datetime.now(),
                    processing_time=time.time() - start_time
                )
                
                # Update tracking
                self.active_faces[track_id] = face
                
                # Add to history with size limit
                self.face_history[track_id].append(face)
                if len(self.face_history[track_id]) > self.max_history_per_face:
                    self.face_history[track_id] = self.face_history[track_id][-self.max_history_per_face:]
                
                # Update stats
                self._update_face_memory(person_id, face)
                
                faces.append(face)
            
            # Update stats
            self._update_stats(faces, time.time() - start_time)
            
            return faces
            
        except Exception as e:
            logger.error(f"Error processing faces: {e}", exc_info=True)
            self.stats['errors'] += 1
            return []
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face from an image.
        
        Args:
            face_image: Cropped face image
            
        Returns:
            (person_id, confidence)
        """
        if not HAS_FACE_RECOGNITION:
            return None, 0.0
        
        try:
            # Get face encoding
            encodings = face_recognition.face_encodings(face_image)
            if not encodings:
                return None, 0.0
            
            return self._match_encoding_to_db(encodings[0])
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return None, 0.0
    
    def learn_face(self, 
                  face_image: np.ndarray, 
                  person_id: str,
                  person_name: Optional[str] = None):
        """
        Learn a new face for future recognition.
        """
        if not HAS_FACE_RECOGNITION:
            return
        
        try:
            # Get face encoding
            encodings = face_recognition.face_encodings(face_image)
            if not encodings:
                logger.warning(f"No face found in image for {person_name or person_id}")
                return
            
            # Add to database
            self.face_encodings_db.append(encodings[0])
            self.face_names_db.append(person_id)
            
            # Create or update face memory
            if person_id not in self.known_faces:
                self.known_faces[person_id] = FaceMemory(
                    person_id=person_id,
                    name=person_name
                )
                self.stats['unique_persons'] = len(self.known_faces)
            
            memory = self.known_faces[person_id]
            memory.face_encodings.append(encodings[0])
            memory.last_encounter = datetime.now()
            memory.encounter_count += 1
            
            logger.info(f"Learned face for {person_name or person_id}")
            
        except Exception as e:
            logger.error(f"Failed to learn face: {e}")
    
    def get_person_history(self, person_id: str) -> Optional[FaceMemory]:
        """Get memory of a person"""
        return self.known_faces.get(person_id)
    
    def _load_models(self):
        """Load face processing models"""
        try:
            # Load face detection model
            if HAS_FACE_RECOGNITION:
                # face_recognition uses HOG + CNN by default
                logger.info("Using face_recognition library")
            else:
                # Fallback to OpenCV Haar cascades
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                if self.face_detector.empty():
                    logger.error("Failed to load OpenCV face detector")
                    self.face_detector = None
                else:
                    logger.info("Loaded OpenCV face detector")
            
            # Load facial landmark detector (using face_recognition)
            if HAS_FACE_RECOGNITION:
                # face_recognition includes landmark detection
                pass
            
            # Load emotion recognition model (simplified - would use proper model)
            self.emotion_model = {
                'model': 'placeholder',
                'classes': [e.value for e in FacialExpression]
            }
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def _detect_faces(self, image: np.ndarray) -> List[Tuple]:
        """Detect faces in image"""
        if HAS_FACE_RECOGNITION:
            # Use face_recognition (more accurate)
            try:
                return face_recognition.face_locations(image)
            except Exception as e:
                logger.warning(f"face_recognition detection failed: {e}")
                return []
        elif self.face_detector is not None:
            # Use OpenCV Haar cascades (faster, less accurate)
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                # Convert to face_recognition format (top, right, bottom, left)
                return [(y, x + w, y + h, x) for (x, y, w, h) in faces]
            except Exception as e:
                logger.warning(f"OpenCV face detection failed: {e}")
                return []
        return []
    
    def _get_face_encodings(self, image: np.ndarray, face_locations: List) -> List:
        """Get face encodings for recognition"""
        if HAS_FACE_RECOGNITION and face_locations:
            try:
                return face_recognition.face_encodings(image, face_locations)
            except Exception as e:
                logger.warning(f"Face encoding failed: {e}")
                return [None] * len(face_locations)
        return [None] * len(face_locations)
    
    def _get_face_landmarks(self, image: np.ndarray, face_locations: List) -> List:
        """Get facial landmarks for each face"""
        if HAS_FACE_RECOGNITION and face_locations:
            try:
                landmarks_batch = face_recognition.face_landmarks(image, face_locations)
                
                # Convert to our FacialLandmarks format
                result = []
                for landmarks in landmarks_batch:
                    fl = FacialLandmarks(
                        jaw=landmarks.get('chin', []),
                        left_eyebrow=landmarks.get('left_eyebrow', []),
                        right_eyebrow=landmarks.get('right_eyebrow', []),
                        nose_bridge=landmarks.get('nose_bridge', []),
                        nose_tip=landmarks.get('nose_tip', []),
                        left_eye=landmarks.get('left_eye', []),
                        right_eye=landmarks.get('right_eye', []),
                        outer_lips=landmarks.get('bottom_lip', []) + landmarks.get('top_lip', [])
                    )
                    result.append(fl)
                return result
            except Exception as e:
                logger.warning(f"Landmark detection failed: {e}")
                return [None] * len(face_locations)
        return [None] * len(face_locations)
    
    def _match_encoding_to_db(self, encoding: np.ndarray) -> Tuple[Optional[str], float]:
        """Match face encoding to database"""
        if not self.face_encodings_db or encoding is None:
            return None, 0.0
        
        try:
            # Compare with known faces
            distances = face_recognition.face_distance(self.face_encodings_db, encoding)
            
            if len(distances) > 0:
                best_idx = np.argmin(distances)
                confidence = 1 - distances[best_idx]
                
                if confidence > self.recognition_threshold:
                    return self.face_names_db[best_idx], confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Face matching error: {e}")
            return None, 0.0
    
    def _recognize_face(self, encoding: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """Recognize face from encoding"""
        if encoding is None:
            return None, None, 0.0
        
        person_id, confidence = self._match_encoding_to_db(encoding)
        
        if person_id:
            memory = self.known_faces.get(person_id)
            name = memory.name if memory else None
            return person_id, name, confidence
        
        return None, None, 0.0
    
    def _track_face(self, 
                   bbox: Tuple[int, int, int, int], 
                   encoding: np.ndarray,
                   person_id: Optional[str]) -> str:
        """Track face across frames using IoU and appearance"""
        # Simple IoU-based tracking
        best_match = None
        best_iou = 0.5  # Threshold
        best_encoding_sim = 0.0
        
        for track_id, face in self.active_faces.items():
            # Calculate IoU with existing track
            iou = self._calculate_iou(bbox, face.bounding_box)
            
            # Calculate encoding similarity if available
            encoding_sim = 0.0
            if encoding is not None and hasattr(face, 'encoding') and face.encoding is not None:
                try:
                    encoding_sim = 1 - face_recognition.face_distance([face.encoding], encoding)[0]
                except:
                    pass
            
            # Combined score
            combined_score = iou * 0.6 + encoding_sim * 0.4
            
            if combined_score > best_iou:
                best_iou = combined_score
                best_match = track_id
        
        if best_match:
            return best_match
        else:
            # New track
            return f"track_{int(time.time())}_{len(self.active_faces)}"
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _analyze_expression(self, 
                           landmarks: Optional[FacialLandmarks], 
                           image: np.ndarray,
                           bbox: Tuple) -> Dict[FacialExpression, float]:
        """Analyze facial expression"""
        scores = {expr: 0.0 for expr in FacialExpression}
        
        if not landmarks:
            return scores
        
        try:
            # Calculate facial metrics
            img_height, img_width = image.shape[:2]
            
            # Eye openness
            left_eye_height = self._eye_openness(landmarks.left_eye) if landmarks.left_eye else 1.0
            right_eye_height = self._eye_openness(landmarks.right_eye) if landmarks.right_eye else 1.0
            avg_eye_height = (left_eye_height + right_eye_height) / 2
            
            # Eyebrow position (relative to face)
            left_brow_y = np.mean([p[1] for p in landmarks.left_eyebrow]) if landmarks.left_eyebrow else 0
            right_brow_y = np.mean([p[1] for p in landmarks.right_eyebrow]) if landmarks.right_eyebrow else 0
            avg_brow_y = (left_brow_y + right_brow_y) / 2 if left_brow_y and right_brow_y else 0
            
            # Normalize brow height (relative to face height)
            face_height = bbox[3] - bbox[1]
            if face_height > 0:
                brow_height_norm = avg_brow_y / face_height
            else:
                brow_height_norm = 0.5
            
            # Mouth metrics
            mouth_width = 0
            mouth_openness = 0
            lip_corner_y = 0
            
            if landmarks.outer_lips:
                mouth_width = self._mouth_width(landmarks.outer_lips)
                mouth_openness = self._mouth_openness(landmarks.outer_lips, landmarks.inner_lips)
                lip_corner_y = self._lip_corner_position(landmarks.outer_lips)
            
            # Happiness: lip corners up, cheeks raised, eyes crinkled
            if lip_corner_y < -2:  # Corners up
                happiness_score = min(1.0, abs(lip_corner_y) / 10)
                if avg_eye_height < 0.7:  # Crinkled eyes
                    happiness_score *= 1.2
                scores[FacialExpression.HAPPINESS] = min(1.0, happiness_score)
            
            # Sadness: lip corners down, brows up and together
            if lip_corner_y > 2:  # Corners down
                sadness_score = min(1.0, lip_corner_y / 10)
                if brow_height_norm > 0.6:  # Brows raised
                    sadness_score *= 1.2
                scores[FacialExpression.SADNESS] = min(1.0, sadness_score)
            
            # Anger: brows down and together, eyes wide, lips tight
            if brow_height_norm < 0.4:  # Brows lowered
                anger_score = min(1.0, (0.5 - brow_height_norm) * 5)
                if avg_eye_height > 0.8:  # Eyes wide
                    anger_score *= 1.2
                scores[FacialExpression.ANGER] = min(1.0, anger_score)
            
            # Surprise: brows up, eyes wide, mouth open
            if brow_height_norm > 0.7 and avg_eye_height > 0.8 and mouth_openness > 0.5:
                surprise_score = min(1.0, (brow_height_norm - 0.5) * 3)
                scores[FacialExpression.SURPRISE] = surprise_score
            
            # Fear: brows up and together, eyes wide, mouth stretched
            if brow_height_norm > 0.6 and avg_eye_height > 0.8 and mouth_width > 1.2:
                fear_score = min(1.0, (brow_height_norm - 0.5) * 3)
                scores[FacialExpression.FEAR] = fear_score
            
            # Disgust: nose wrinkled, upper lip raised
            # Simplified detection
            
            # Neutral: relaxed features
            active_scores = [v for v in scores.values() if v > 0.3]
            if not active_scores:
                scores[FacialExpression.NEUTRAL] = 0.8
            
        except Exception as e:
            logger.warning(f"Expression analysis failed: {e}")
        
        return scores
    
    def _eye_openness(self, eye_points: List[Tuple[int, int]]) -> float:
        """Calculate eye openness ratio"""
        if len(eye_points) < 6:
            return 1.0
        
        # Eye aspect ratio (EAR)
        # Vertical distances / horizontal distance
        try:
            p1 = np.array(eye_points[1])
            p5 = np.array(eye_points[5])
            p2 = np.array(eye_points[2])
            p4 = np.array(eye_points[4])
            p0 = np.array(eye_points[0])
            p3 = np.array(eye_points[3])
            
            vert1 = np.linalg.norm(p1 - p5)
            vert2 = np.linalg.norm(p2 - p4)
            horiz = np.linalg.norm(p0 - p3)
            
            ear = (vert1 + vert2) / (2.0 * horiz) if horiz > 0 else 0
            return min(1.0, ear * 3)  # Normalize
        except:
            return 1.0
    
    def _mouth_width(self, mouth_points: List[Tuple[int, int]]) -> float:
        """Calculate mouth width relative to face"""
        if len(mouth_points) < 12:
            return 1.0
        
        try:
            left = np.array(mouth_points[0])
            right = np.array(mouth_points[6])
            width = np.linalg.norm(left - right)
            return width / 50  # Normalized
        except:
            return 1.0
    
    def _mouth_openness(self, outer: List, inner: List) -> float:
        """Calculate how open the mouth is"""
        if len(outer) < 18 or len(inner) < 8:
            return 0.0
        
        try:
            # Vertical distance between lips
            top_lip = np.mean([p[1] for p in outer[12:15]], axis=0)
            bottom_lip = np.mean([p[1] for p in outer[15:18]], axis=0)
            
            openness = abs(top_lip - bottom_lip)
            return min(1.0, openness / 20)  # Normalize
        except:
            return 0.0
    
    def _lip_corner_position(self, mouth_points: List) -> float:
        """Get relative position of lip corners (-1 to 1, negative = up)"""
        if len(mouth_points) < 12:
            return 0.0
        
        try:
            left_corner = np.array(mouth_points[0])
            right_corner = np.array(mouth_points[6])
            
            # Average y position of corners relative to mouth center
            corner_y = (left_corner[1] + right_corner[1]) / 2
            mouth_center_y = np.mean([p[1] for p in mouth_points[3:9]])
            
            return (corner_y - mouth_center_y)
        except:
            return 0.0
    
    def _get_primary_expression(self, 
                               scores: Dict[FacialExpression, float]) -> Tuple[FacialExpression, float]:
        """Get primary expression and confidence"""
        if not scores:
            return FacialExpression.NEUTRAL, 0.0
        
        # Filter by threshold
        active = [(expr, score) for expr, score in scores.items() if score > 0.3]
        
        if not active:
            return FacialExpression.NEUTRAL, 0.5
        
        # Get highest scoring expression
        active.sort(key=lambda x: x[1], reverse=True)
        return active[0]
    
    def _analyze_eyes(self, 
                     landmarks: Optional[FacialLandmarks],
                     image: np.ndarray,
                     bbox: Tuple) -> EyeAnalysis:
        """Detailed eye analysis"""
        eyes = EyeAnalysis()
        
        if not landmarks:
            return eyes
        
        try:
            # Eye openness
            if landmarks.left_eye:
                eyes.left_eye_open = self._eye_openness(landmarks.left_eye)
            if landmarks.right_eye:
                eyes.right_eye_open = self._eye_openness(landmarks.right_eye)
            
            # Gaze direction (simplified - based on head pose and eye position)
            # For now, assume towards camera if face is frontal
            if bbox[2] - bbox[0] > 50:  # Face is large enough
                # Very simplified gaze estimation
                if landmarks.left_eye and landmarks.right_eye:
                    left_eye_center = np.mean(landmarks.left_eye, axis=0)
                    right_eye_center = np.mean(landmarks.right_eye, axis=0)
                    
                    # Check if looking at camera (rough approximation)
                    img_center_x = image.shape[1] / 2
                    eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
                    
                    if abs(eye_center_x - img_center_x) < 50:
                        eyes.gaze_direction = GazeDirection.TOWARDS_CAMERA
                    else:
                        eyes.gaze_direction = GazeDirection.AWAY_CAMERA
            
            # Check for deception indicators
            eyes.avoids_eye_contact = eyes.gaze_direction != GazeDirection.TOWARDS_CAMERA
            eyes.excessive_blinking = (eyes.left_eye_open < 0.3 and eyes.right_eye_open < 0.3)
            
        except Exception as e:
            logger.warning(f"Eye analysis failed: {e}")
        
        return eyes
    
    def _estimate_head_pose(self, landmarks: Optional[FacialLandmarks]) -> Tuple[HeadPose, Tuple]:
        """Estimate head pose from landmarks"""
        if not landmarks or not landmarks.nose_bridge:
            return HeadPose.FRONTAL, (0, 0, 0)
        
        try:
            # Simplified pose estimation using nose position
            nose_tip = landmarks.nose_tip[0] if landmarks.nose_tip else None
            nose_bridge = landmarks.nose_bridge[0] if landmarks.nose_bridge else None
            
            if nose_tip and nose_bridge:
                # Calculate angle from nose orientation
                dx = nose_tip[0] - nose_bridge[0]
                dy = nose_tip[1] - nose_bridge[1]
                
                if abs(dx) < 5:
                    return HeadPose.FRONTAL, (0, 0, 0)
                elif dx > 10:
                    return HeadPose.PROFILE_RIGHT, (float(dx), float(dy), 0)
                elif dx < -10:
                    return HeadPose.PROFILE_LEFT, (float(dx), float(dy), 0)
            
            return HeadPose.FRONTAL, (0, 0, 0)
            
        except Exception as e:
            logger.warning(f"Head pose estimation failed: {e}")
            return HeadPose.FRONTAL, (0, 0, 0)
    
    def _extract_action_units(self, 
                            landmarks: Optional[FacialLandmarks],
                            expression_scores: Dict) -> Dict[str, float]:
        """Extract Facial Action Coding System units"""
        aus = {}
        
        if not landmarks:
            return aus
        
        try:
            # AU1: Inner brow raiser (simplified)
            # AU2: Outer brow raiser
            # AU4: Brow lowerer
            
            # AU5: Upper lid raiser
            if landmarks.left_eye:
                eye_open = self._eye_openness(landmarks.left_eye)
                aus['AU5'] = eye_open
            
            # AU12: Lip corner puller (smile)
            if expression_scores.get(FacialExpression.HAPPINESS, 0) > 0.3:
                aus['AU12'] = expression_scores[FacialExpression.HAPPINESS]
            
            # AU15: Lip corner depressor (frown)
            if expression_scores.get(FacialExpression.SADNESS, 0) > 0.3:
                aus['AU15'] = expression_scores[FacialExpression.SADNESS]
            
            # AU26: Jaw drop (surprise)
            if expression_scores.get(FacialExpression.SURPRISE, 0) > 0.3:
                aus['AU26'] = expression_scores[FacialExpression.SURPRISE]
            
        except Exception as e:
            logger.warning(f"Action unit extraction failed: {e}")
        
        return aus
    
    def _detect_micro_expressions(self,
                                 track_id: str,
                                 current_landmarks: Optional[FacialLandmarks],
                                 current_scores: Dict) -> List[MicroExpression]:
        """Detect micro-expressions from temporal patterns"""
        micro_exprs = []
        
        # Need history for this track
        if track_id not in self.face_history:
            return micro_exprs
        
        history = self.face_history[track_id][-10:]  # Last 10 frames
        if len(history) < 5:
            return micro_exprs
        
        try:
            # Look for brief, intense expressions
            # Compare current expression with recent average
            
            # Average recent expression scores
            recent_scores = defaultdict(list)
            for face in history[-5:-1]:  # Exclude current
                for expr, score in face.expression_scores.items():
                    recent_scores[expr.value].append(score)
            
            avg_recent = {
                expr: np.mean(scores) for expr, scores in recent_scores.items()
            }
            
            # Check for sudden changes
            for expr, score in current_scores.items():
                expr_str = expr.value
                if score > 0.7 and avg_recent.get(expr_str, 0) < 0.3:
                    # Possible micro-expression
                    micro = MicroExpression(
                        expression=expr,
                        intensity=score,
                        duration=0.1,  # Approximate
                        start_time=time.time(),
                        confidence=0.7
                    )
                    micro_exprs.append(micro)
                    
        except Exception as e:
            logger.warning(f"Micro-expression detection failed: {e}")
        
        return micro_exprs
    
    def _detect_speaking(self, 
                        landmarks: Optional[FacialLandmarks],
                        action_units: Dict) -> bool:
        """Detect if person is speaking"""
        if not landmarks:
            return False
        
        try:
            # Check for mouth movements
            if landmarks.outer_lips and landmarks.inner_lips:
                mouth_movement = self._mouth_openness(landmarks.outer_lips, landmarks.inner_lips)
                return mouth_movement > 0.3
            
        except Exception as e:
            logger.warning(f"Speaking detection failed: {e}")
        
        return False
    
    def _load_expression_thresholds(self) -> Dict:
        """Load thresholds for expression detection"""
        return {
            'happiness': {'au12': 0.5, 'au6': 0.3},
            'sadness': {'au1': 0.4, 'au4': 0.3, 'au15': 0.5},
            'anger': {'au4': 0.6, 'au5': 0.4, 'au7': 0.5},
            'surprise': {'au1': 0.6, 'au2': 0.6, 'au5': 0.5},
            'fear': {'au1': 0.5, 'au2': 0.5, 'au4': 0.3, 'au5': 0.5},
            'disgust': {'au9': 0.5, 'au10': 0.5},
        }
    
    def _load_facs_mapping(self) -> Dict:
        """Load FACS action units mapping"""
        return {
            'AU1': 'Inner Brow Raiser',
            'AU2': 'Outer Brow Raiser',
            'AU4': 'Brow Lowerer',
            'AU5': 'Upper Lid Raiser',
            'AU6': 'Cheek Raiser',
            'AU7': 'Lid Tightener',
            'AU9': 'Nose Wrinkler',
            'AU10': 'Upper Lip Raiser',
            'AU12': 'Lip Corner Puller',
            'AU15': 'Lip Corner Depressor',
            'AU17': 'Chin Raiser',
            'AU20': 'Lip Stretcher',
            'AU23': 'Lip Tightener',
            'AU25': 'Lips Part',
            'AU26': 'Jaw Drop',
            'AU27': 'Mouth Stretch',
        }
    
    def _load_deception_patterns(self) -> Dict:
        """Load patterns indicating possible deception"""
        return {
            'eye_contact_avoidance': 0.6,
            'excessive_blinking': 0.5,
            'micro_expressions': 0.8,
            'inconsistent_emotions': 0.7,
            'delayed_responses': 0.4,
            'touching_face': 0.3,
        }
    
    def _detect_deception(self,
                         eye_analysis: EyeAnalysis,
                         micro_expressions: List[MicroExpression],
                         action_units: Dict,
                         context: Optional[Dict]) -> Dict[str, float]:
        """Detect potential deception indicators"""
        indicators = {}
        
        # Eye contact avoidance
        if eye_analysis.avoids_eye_contact:
            indicators['eye_contact_avoidance'] = 0.6
        
        # Excessive blinking
        if eye_analysis.excessive_blinking:
            indicators['excessive_blinking'] = 0.5
        
        # Micro-expressions
        if micro_expressions:
            indicators['micro_expressions'] = min(1.0, len(micro_expressions) * 0.3)
        
        # Context-based adjustments
        if context and context.get('high_stakes', False):
            for key in indicators:
                indicators[key] = min(1.0, indicators[key] * 1.2)
        
        return indicators
    
    def _infer_hidden_emotion(self,
                             expression_scores: Dict,
                             micro_expressions: List[MicroExpression],
                             deception: Dict) -> Optional[FacialExpression]:
        """Infer hidden emotion from cues"""
        if micro_expressions:
            # Micro-expressions often reveal true emotion
            return micro_expressions[0].expression
        
        # Look for masked emotions
        if 'micro_expressions' in deception:
            # Deception detected, true emotion might be opposite
            primary = self._get_primary_expression(expression_scores)[0]
            
            # Map to opposite/hidden emotion
            opposites = {
                FacialExpression.HAPPINESS: FacialExpression.SADNESS,
                FacialExpression.ANGER: FacialExpression.FEAR,
                FacialExpression.SURPRISE: FacialExpression.NEUTRAL,
            }
            
            return opposites.get(primary)
        
        return None
    
    def _calculate_trust_score(self,
                              person_id: Optional[str],
                              deception: Dict,
                              expression_scores: Dict) -> float:
        """Calculate trust score for person"""
        # Start with baseline
        if person_id and person_id in self.known_faces:
            trust = self.known_faces[person_id].trust_level
        else:
            trust = 0.5  # Neutral for strangers
        
        # Adjust based on deception indicators
        deception_penalty = sum(deception.values()) * 0.3
        trust = max(0.1, trust - deception_penalty)
        
        # Adjust based on expression (angry expressions reduce trust)
        anger_score = expression_scores.get(FacialExpression.ANGER, 0)
        trust = max(0.1, trust - anger_score * 0.2)
        
        return min(1.0, trust)
    
    def _is_interesting(self,
                       expression_scores: Dict,
                       deception: Dict,
                       context: Optional[Dict]) -> bool:
        """Does Wednesday find this person interesting?"""
        # People showing deception are interesting
        if deception:
            return True
        
        # People with strong emotional expressions are interesting
        if any(v > 0.7 for v in expression_scores.values()):
            return True
        
        # People with unusual expressions
        unusual = [FacialExpression.SKEPTICISM, FacialExpression.SUSPICION]
        if any(expr in unusual and expression_scores.get(expr, 0) > 0.3 
               for expr in unusual):
            return True
        
        # Context-based interest
        if context and context.get('important_conversation', False):
            return True
        
        return False
    
    def _estimate_demographics(self,
                              landmarks: Optional[FacialLandmarks],
                              image: np.ndarray,
                              bbox: Tuple) -> Tuple[Optional[int], Optional[str]]:
        """Estimate age and gender"""
        # Simplified - would use dedicated models
        # This is a placeholder
        age = None
        gender = None
        
        try:
            # Very rough estimation based on facial proportions
            # This is not accurate - would use ML models in production
            
            # Placeholder values
            age = 30
            gender = "unknown"
            
        except Exception as e:
            logger.warning(f"Demographic estimation failed: {e}")
        
        return age, gender
    
    def _cleanup_lost_faces(self):
        """Remove faces that haven't been seen recently"""
        current_time = datetime.now()
        lost_faces = []
        
        for track_id, face in self.active_faces.items():
            if (current_time - face.last_seen).total_seconds() > 5:  # Lost for 5 seconds
                lost_faces.append(track_id)
        
        for track_id in lost_faces:
            del self.active_faces[track_id]
    
    def _update_face_memory(self, person_id: Optional[str], face: FaceAnalysis):
        """Update memory for a person"""
        if not person_id:
            return
        
        if person_id not in self.known_faces:
            self.known_faces[person_id] = FaceMemory(
                person_id=person_id,
                name=face.person_name
            )
            self.stats['unique_persons'] = len(self.known_faces)
        
        memory = self.known_faces[person_id]
        memory.last_encounter = datetime.now()
        memory.encounter_count += 1
        
        # Update average expression (using string keys)
        for expr, score in face.expression_scores.items():
            expr_str = expr.value
            if expr_str not in memory.average_expression:
                memory.average_expression[expr_str] = score
            else:
                # Moving average
                memory.average_expression[expr_str] = (
                    memory.average_expression[expr_str] * 0.9 + score * 0.1
                )
        
        # Update typical gaze
        if face.eyes.gaze_direction == GazeDirection.TOWARDS_CAMERA:
            memory.typical_gaze = GazeDirection.TOWARDS_CAMERA
        
        # Update trust level (moving average)
        memory.trust_level = memory.trust_level * 0.9 + face.trust_score * 0.1
    
    def _update_stats(self, faces: List[FaceAnalysis], processing_time: float):
        """Update processing statistics"""
        self.stats['total_faces_detected'] += len(faces)
        
        # Update average processing time
        total = self.stats['total_faces_detected']
        old_avg = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = old_avg + (processing_time - old_avg) / max(1, total)
        
        # Track expressions
        for face in faces:
            self.stats['expressions_detected'][face.primary_expression.value] += 1
    
    def get_stats(self) -> Dict:
        """Return processing statistics"""
        stats = dict(self.stats)
        stats['expressions_detected'] = dict(stats['expressions_detected'])
        return stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics"""
        self.stats = {
            'total_faces_detected': 0,
            'unique_persons': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0,
            'errors': 0,
            'expressions_detected': defaultdict(int)
        }

# Connects to: object_recognition.py (shares vision pipeline)
# Connects to: perception/attention/salience.py (faces attract attention)
# Connects to: memory/episodic/ (stores face encounters)
# Connects to: memory/semantic/ (person knowledge)
# Connects to: emotion/empathy.py (emotional expressions inform empathy)
# Connects to: self/theory_of_mind.py (understanding others' mental states)
# Connects to: cognition/reasoning.py (deception detection feeds reasoning)