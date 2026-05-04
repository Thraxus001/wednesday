"""
valence_arousal.py - Core dimensional model of emotion for Wednesday AI

This module implements the foundational valence-arousal (and optionally dominance)
dimensional model of emotion. It provides the basic building blocks for all
emotional processing in Wednesday's cognitive architecture.

The valence-arousal model is based on established psychological research:
- Russell's circumplex model of affect
- Mehrabian's PAD (Pleasure-Arousal-Dominance) emotional state model
- Ekman's basic emotions mapped to dimensional space

Key improvements:
- Fixed dimension range inconsistencies
- Added proper validation for all operations
- Removed numpy dependency for better portability
- Enhanced mathematical precision with decimal rounding
- Added comprehensive error handling
"""

import math
import random
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from functools import total_ordering

# Configure logging
logger = logging.getLogger(__name__)


class EmotionDimension(Enum):
    """Enumeration of emotional dimensions"""
    VALENCE = "valence"
    AROUSAL = "arousal"
    DOMINANCE = "dominance"


@dataclass
class PADVector:
    """
    Pleasure-Arousal-Dominance vector representation of emotion.
    
    This is the core data structure for dimensional emotion modeling.
    All three dimensions are continuous values within specific ranges.
    
    Attributes:
        valence: Pleasure-displeasure (-1.0 to 1.0)
        arousal: Calm-excitement (0.0 to 1.0)
        dominance: Control-submission (0.0 to 1.0)
    """
    
    valence: float  # -1.0 to 1.0
    arousal: float  # 0.0 to 1.0
    dominance: float  # 0.0 to 1.0
    
    # Class constants for validation
    VALENCE_MIN = -1.0
    VALENCE_MAX = 1.0
    AROUSAL_MIN = 0.0
    AROUSAL_MAX = 1.0
    DOMINANCE_MIN = 0.0
    DOMINANCE_MAX = 1.0
    
    def __post_init__(self):
        """Validate and clamp values to valid ranges"""
        self.valence = self._clamp(
            self.valence, self.VALENCE_MIN, self.VALENCE_MAX
        )
        self.arousal = self._clamp(
            self.arousal, self.AROUSAL_MIN, self.AROUSAL_MAX
        )
        self.dominance = self._clamp(
            self.dominance, self.DOMINANCE_MIN, self.DOMINANCE_MAX
        )
        
        # Round to 6 decimal places to avoid floating point issues
        self.valence = round(self.valence, 6)
        self.arousal = round(self.arousal, 6)
        self.dominance = round(self.dominance, 6)
    
    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple for hashable operations"""
        return (self.valence, self.arousal, self.dominance)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with rounded values"""
        return {
            'valence': round(self.valence, 4),
            'arousal': round(self.arousal, 4),
            'dominance': round(self.dominance, 4)
        }
    
    def distance_to(self, other: 'PADVector') -> float:
        """
        Calculate Euclidean distance to another PAD vector.
        
        Args:
            other: Another PADVector
            
        Returns:
            Euclidean distance (0 to ~2.0 maximum)
        """
        if not isinstance(other, PADVector):
            raise TypeError(f"Expected PADVector, got {type(other)}")
        
        squared_sum = (
            (self.valence - other.valence) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.dominance - other.dominance) ** 2
        )
        return math.sqrt(squared_sum)
    
    def cosine_similarity(self, other: 'PADVector') -> float:
        """
        Calculate cosine similarity to another PAD vector.
        
        Args:
            other: Another PADVector
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        if not isinstance(other, PADVector):
            raise TypeError(f"Expected PADVector, got {type(other)}")
        
        # Calculate dot product
        dot_product = (
            self.valence * other.valence +
            self.arousal * other.arousal +
            self.dominance * other.dominance
        )
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(
            self.valence ** 2 + self.arousal ** 2 + self.dominance ** 2
        )
        magnitude2 = math.sqrt(
            other.valence ** 2 + other.arousal ** 2 + other.dominance ** 2
        )
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Clamp to avoid floating point errors
        similarity = dot_product / (magnitude1 * magnitude2)
        return max(-1.0, min(1.0, similarity))
    
    def interpolate(self, other: 'PADVector', t: float) -> 'PADVector':
        """
        Linearly interpolate between this vector and another.
        
        Args:
            other: Target PADVector
            t: Interpolation factor (0 = this, 1 = other)
            
        Returns:
            Interpolated PADVector
            
        Raises:
            ValueError: If t is not between 0 and 1
        """
        if not 0 <= t <= 1:
            raise ValueError(f"Interpolation factor t must be between 0 and 1, got {t}")
        
        return PADVector(
            valence=self.valence * (1 - t) + other.valence * t,
            arousal=self.arousal * (1 - t) + other.arousal * t,
            dominance=self.dominance * (1 - t) + other.dominance * t
        )
    
    def __add__(self, other: 'PADVector') -> 'PADVector':
        """Vector addition"""
        if not isinstance(other, PADVector):
            return NotImplemented
        return PADVector(
            valence=self.valence + other.valence,
            arousal=self.arousal + other.arousal,
            dominance=self.dominance + other.dominance
        )
    
    def __sub__(self, other: 'PADVector') -> 'PADVector':
        """Vector subtraction"""
        if not isinstance(other, PADVector):
            return NotImplemented
        return PADVector(
            valence=self.valence - other.valence,
            arousal=self.arousal - other.arousal,
            dominance=self.dominance - other.dominance
        )
    
    def __mul__(self, scalar: float) -> 'PADVector':
        """Scalar multiplication"""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return PADVector(
            valence=self.valence * scalar,
            arousal=self.arousal * scalar,
            dominance=self.dominance * scalar
        )
    
    def __rmul__(self, scalar: float) -> 'PADVector':
        """Reverse scalar multiplication"""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'PADVector':
        """Scalar division"""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        return PADVector(
            valence=self.valence / scalar,
            arousal=self.arousal / scalar,
            dominance=self.dominance / scalar
        )
    
    def __eq__(self, other: object) -> bool:
        """Equality check with tolerance"""
        if not isinstance(other, PADVector):
            return False
        return (
            abs(self.valence - other.valence) < 1e-6 and
            abs(self.arousal - other.arousal) < 1e-6 and
            abs(self.dominance - other.dominance) < 1e-6
        )
    
    def __hash__(self) -> int:
        """Hash based on rounded values"""
        return hash((round(self.valence, 4), 
                     round(self.arousal, 4), 
                     round(self.dominance, 4)))
    
    def __repr__(self) -> str:
        return f"PAD(V={self.valence:.3f}, A={self.arousal:.3f}, D={self.dominance:.3f})"


@dataclass
class EmotionPrototype:
    """
    Prototypical representation of an emotion in PAD space.
    
    Each discrete emotion has a canonical PAD coordinate, plus
    variance information for realistic variation.
    """
    name: str
    pad: PADVector
    variance: float = 0.2  # Typical variance in emotional space
    description: str = ""
    
    def __post_init__(self):
        """Validate parameters"""
        if not 0 <= self.variance <= 1:
            raise ValueError(f"Variance must be between 0 and 1, got {self.variance}")
    
    def sample(self, variance_multiplier: float = 1.0, 
               random_gen: Optional[random.Random] = None) -> PADVector:
        """
        Sample a PAD vector near this prototype.
        
        Args:
            variance_multiplier: Scale factor for variance
            random_gen: Optional random generator for reproducibility
            
        Returns:
            Slightly varied PADVector
        """
        if random_gen is None:
            random_gen = random
        
        # Generate noise with appropriate scaling for each dimension
        valence_noise = random_gen.gauss(0, self.variance * variance_multiplier * 0.5)
        arousal_noise = random_gen.gauss(0, self.variance * variance_multiplier * 0.25)
        dominance_noise = random_gen.gauss(0, self.variance * variance_multiplier * 0.25)
        
        return PADVector(
            valence=self.pad.valence + valence_noise,
            arousal=self.pad.arousal + arousal_noise,
            dominance=self.pad.dominance + dominance_noise
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'pad': self.pad.to_dict(),
            'variance': self.variance,
            'description': self.description
        }


class EmotionLexicon:
    """
    Lexicon of emotion prototypes mapping discrete emotions to PAD space.
    
    Based on psychological research:
    - Russell's circumplex model (1980)
    - Mehrabian's PAD study (1995)
    - Plutchik's wheel of emotions (2001)
    """
    
    # Classic emotion prototypes (based on research)
    PROTOTYPES: Dict[str, EmotionPrototype] = {
        # Basic emotions (Ekman)
        'joy': EmotionPrototype(
            'joy', 
            PADVector(0.8, 0.7, 0.6),
            variance=0.15,
            description="High pleasure, high arousal, moderate dominance"
        ),
        'sadness': EmotionPrototype(
            'sadness',
            PADVector(-0.6, 0.2, 0.2),
            variance=0.2,
            description="Low pleasure, low arousal, low dominance"
        ),
        'anger': EmotionPrototype(
            'anger',
            PADVector(-0.5, 0.8, 0.7),
            variance=0.2,
            description="Low pleasure, high arousal, high dominance"
        ),
        'fear': EmotionPrototype(
            'fear',
            PADVector(-0.6, 0.8, 0.2),
            variance=0.25,
            description="Low pleasure, high arousal, low dominance"
        ),
        'surprise': EmotionPrototype(
            'surprise',
            PADVector(0.3, 0.8, 0.4),
            variance=0.3,
            description="Neutral-positive pleasure, high arousal, moderate dominance"
        ),
        'disgust': EmotionPrototype(
            'disgust',
            PADVector(-0.5, 0.5, 0.5),
            variance=0.2,
            description="Low pleasure, moderate arousal, moderate dominance"
        ),
        
        # Additional emotions (Plutchik)
        'trust': EmotionPrototype(
            'trust',
            PADVector(0.5, 0.3, 0.6),
            variance=0.2,
            description="High pleasure, low-moderate arousal, moderate-high dominance"
        ),
        'anticipation': EmotionPrototype(
            'anticipation',
            PADVector(0.3, 0.6, 0.5),
            variance=0.25,
            description="Mild pleasure, moderate-high arousal, moderate dominance"
        ),
        
        # Complex emotions
        'love': EmotionPrototype(
            'love',
            PADVector(0.9, 0.6, 0.5),
            variance=0.2,
            description="High pleasure, moderate arousal, moderate dominance"
        ),
        'grief': EmotionPrototype(
            'grief',
            PADVector(-0.8, 0.1, 0.1),
            variance=0.15,
            description="Very low pleasure, very low arousal, very low dominance"
        ),
        'contempt': EmotionPrototype(
            'contempt',
            PADVector(-0.3, 0.4, 0.8),
            variance=0.2,
            description="Slightly negative, moderate arousal, high dominance"
        ),
        'pride': EmotionPrototype(
            'pride',
            PADVector(0.6, 0.5, 0.8),
            variance=0.2,
            description="High pleasure, moderate arousal, high dominance"
        ),
        'shame': EmotionPrototype(
            'shame',
            PADVector(-0.5, 0.3, 0.2),
            variance=0.25,
            description="Low pleasure, low-moderate arousal, low dominance"
        ),
        'curiosity': EmotionPrototype(
            'curiosity',
            PADVector(0.4, 0.6, 0.5),
            variance=0.25,
            description="Mild pleasure, moderate-high arousal, moderate dominance"
        ),
        
        # Wednesday-specific emotional nuances
        'dark_amusement': EmotionPrototype(
            'dark_amusement',
            PADVector(0.2, 0.5, 0.7),
            variance=0.3,
            description="Slight pleasure, moderate arousal, high dominance - Wednesday's signature"
        ),
        'protective': EmotionPrototype(
            'protective',
            PADVector(0.1, 0.5, 0.8),
            variance=0.2,
            description="Neutral pleasure, moderate arousal, high dominance - defending trusted ones"
        ),
        'nostalgic': EmotionPrototype(
            'nostalgic',
            PADVector(0.1, 0.2, 0.4),
            variance=0.2,
            description="Slight pleasure, low arousal, moderate-low dominance - bittersweet memories"
        ),
        'satisfied': EmotionPrototype(
            'satisfied',
            PADVector(0.5, 0.2, 0.7),
            variance=0.15,
            description="Moderate pleasure, low arousal, high dominance - content and in control"
        ),
        'curiously_detached': EmotionPrototype(
            'curiously_detached',
            PADVector(0.2, 0.4, 0.6),
            variance=0.25,
            description="Slight pleasure, moderate-low arousal, moderate-high dominance"
        ),
        'wary': EmotionPrototype(
            'wary',
            PADVector(-0.2, 0.5, 0.5),
            variance=0.2,
            description="Slight negative, moderate arousal, moderate dominance - cautious and guarded"
        ),
        'pensive': EmotionPrototype(
            'pensive',
            PADVector(-0.1, 0.2, 0.4),
            variance=0.2,
            description="Slightly negative, low arousal, moderate-low dominance - deep in thought"
        ),
        'disdainful': EmotionPrototype(
            'disdainful',
            PADVector(-0.3, 0.3, 0.8),
            variance=0.2,
            description="Mildly negative, low-moderate arousal, high dominance - superior detachment"
        ),
    }
    
    @classmethod
    def get_prototype(cls, emotion_name: str) -> Optional[EmotionPrototype]:
        """Get emotion prototype by name (case-insensitive)"""
        if not emotion_name:
            return None
        return cls.PROTOTYPES.get(emotion_name.lower())
    
    @classmethod
    def get_pad(cls, emotion_name: str) -> Optional[PADVector]:
        """Get PAD vector for emotion prototype"""
        proto = cls.get_prototype(emotion_name)
        return proto.pad if proto else None
    
    @classmethod
    def has_emotion(cls, emotion_name: str) -> bool:
        """Check if emotion exists in lexicon"""
        return emotion_name.lower() in cls.PROTOTYPES
    
    @classmethod
    def get_all_names(cls) -> List[str]:
        """Get list of all emotion names"""
        return sorted(cls.PROTOTYPES.keys())
    
    @classmethod
    def closest_emotion(cls, pad: PADVector, 
                        candidates: Optional[List[str]] = None,
                        threshold: float = 0.5,
                        max_results: int = 5) -> List[Tuple[str, float]]:
        """
        Find closest emotion prototypes to a given PAD vector.
        
        Args:
            pad: Query PAD vector
            candidates: Optional list of emotion names to consider
            threshold: Similarity threshold for inclusion (0-1)
            max_results: Maximum number of results to return
            
        Returns:
            List of (emotion_name, similarity_score) sorted by similarity
            
        Raises:
            ValueError: If threshold is outside valid range
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        
        prototypes = cls.PROTOTYPES
        
        if candidates:
            # Filter to valid candidates (case-insensitive)
            candidates_lower = {c.lower() for c in candidates}
            prototypes = {k: v for k, v in prototypes.items() 
                         if k in candidates_lower}
            
            if not prototypes:
                logger.warning(f"No valid prototypes found for candidates: {candidates}")
                return []
        
        # Calculate similarities
        similarities = []
        for name, proto in prototypes.items():
            # Convert distance to similarity (closer = higher similarity)
            dist = pad.distance_to(proto.pad)
            # Normalize distance to 0-1 range (max possible distance ~2.0)
            normalized_dist = min(1.0, dist / 2.0)
            similarity = 1.0 - normalized_dist
            
            if similarity >= threshold:
                similarities.append((name, round(similarity, 4)))
        
        # Sort by similarity descending and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    @classmethod
    def add_prototype(cls, prototype: EmotionPrototype) -> bool:
        """
        Add a new emotion prototype to the lexicon.
        
        Args:
            prototype: EmotionPrototype to add
            
        Returns:
            True if added successfully, False if name already exists
        """
        if prototype.name in cls.PROTOTYPES:
            logger.warning(f"Prototype '{prototype.name}' already exists")
            return False
        
        cls.PROTOTYPES[prototype.name] = prototype
        logger.info(f"Added new emotion prototype: {prototype.name}")
        return True
    
    @classmethod
    def to_json(cls) -> str:
        """Export lexicon to JSON"""
        data = {
            name: proto.to_dict()
            for name, proto in cls.PROTOTYPES.items()
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> None:
        """Import lexicon from JSON"""
        try:
            data = json.loads(json_str)
            for name, proto_data in data.items():
                pad = PADVector(**proto_data['pad'])
                prototype = EmotionPrototype(
                    name=name,
                    pad=pad,
                    variance=proto_data.get('variance', 0.2),
                    description=proto_data.get('description', '')
                )
                cls.PROTOTYPES[name] = prototype
            logger.info(f"Loaded {len(data)} prototypes from JSON")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load lexicon from JSON: {e}")
            raise


class ValenceArousalSpace:
    """
    Core operations in valence-arousal (2D) or valence-arousal-dominance (3D) space.
    
    This class provides mathematical operations for working with emotional
    dimensions, including distance calculations, region definitions,
    and conversion utilities.
    """
    
    def __init__(self, use_dominance: bool = True):
        """
        Initialize emotional space.
        
        Args:
            use_dominance: Whether to include dominance dimension (3D vs 2D)
        """
        self.use_dominance = use_dominance
        self.dimensions = 3 if use_dominance else 2
        
        # Define emotional regions in 2D space (valence, arousal)
        self.regions_2d = {
            'happy': {
                'valence': (0.3, 1.0), 
                'arousal': (0.3, 1.0),
                'description': 'High pleasure, moderate to high arousal'
            },
            'relaxed': {
                'valence': (0.3, 1.0), 
                'arousal': (0.0, 0.4),
                'description': 'High pleasure, low arousal'
            },
            'sad': {
                'valence': (-1.0, -0.2), 
                'arousal': (0.0, 0.4),
                'description': 'Low pleasure, low arousal'
            },
            'angry': {
                'valence': (-1.0, -0.2), 
                'arousal': (0.5, 1.0),
                'description': 'Low pleasure, high arousal'
            },
            'neutral': {
                'valence': (-0.2, 0.3), 
                'arousal': (0.2, 0.6),
                'description': 'Near-neutral pleasure, moderate arousal'
            },
            'surprised': {
                'valence': (-0.2, 0.5), 
                'arousal': (0.6, 1.0),
                'description': 'Variable pleasure, high arousal'
            },
        }
        
        # Define emotional regions in 3D space (with dominance)
        self.regions_3d = {
            'confident': {
                'valence': (0.2, 1.0), 
                'arousal': (0.3, 1.0), 
                'dominance': (0.6, 1.0),
                'description': 'Positive, aroused, in control'
            },
            'vulnerable': {
                'valence': (-0.5, 0.2), 
                'arousal': (0.2, 0.7), 
                'dominance': (0.0, 0.3),
                'description': 'Slightly negative to neutral, moderate arousal, low control'
            },
            'dominant': {
                'valence': (-0.3, 0.8), 
                'arousal': (0.3, 0.9), 
                'dominance': (0.7, 1.0),
                'description': 'Variable valence, aroused, high control'
            },
            'submissive': {
                'valence': (-0.6, 0.3), 
                'arousal': (0.2, 0.6), 
                'dominance': (0.0, 0.3),
                'description': 'Slightly negative to neutral, moderate arousal, low control'
            },
        }
        
        logger.debug(f"ValenceArousalSpace initialized with {self.dimensions} dimensions")
    
    def create_vector(self, valence: float, arousal: float, 
                      dominance: Optional[float] = None) -> PADVector:
        """
        Create PAD vector with optional dominance.
        
        Args:
            valence: Valence value (-1 to 1)
            arousal: Arousal value (0 to 1)
            dominance: Optional dominance value (0 to 1)
            
        Returns:
            PADVector
        """
        if dominance is None:
            dominance = 0.5  # Default middle value
        return PADVector(valence, arousal, dominance)
    
    def euclidean_distance_2d(self, v1: Tuple[float, float], 
                              v2: Tuple[float, float]) -> float:
        """
        2D Euclidean distance in valence-arousal space.
        
        Args:
            v1: (valence, arousal) tuple
            v2: (valence, arousal) tuple
            
        Returns:
            Euclidean distance
        """
        return math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
    
    def manhattan_distance_2d(self, v1: Tuple[float, float],
                              v2: Tuple[float, float]) -> float:
        """
        2D Manhattan distance.
        
        Args:
            v1: (valence, arousal) tuple
            v2: (valence, arousal) tuple
            
        Returns:
            Manhattan distance
        """
        return abs(v1[0] - v2[0]) + abs(v1[1] - v2[1])
    
    def emotional_similarity(self, pad1: PADVector, pad2: PADVector) -> float:
        """
        Calculate emotional similarity between two PAD vectors.
        
        Returns value between 0 (dissimilar) and 1 (identical).
        
        Args:
            pad1: First PAD vector
            pad2: Second PAD vector
            
        Returns:
            Similarity score (0-1)
        """
        # Weight dimensions based on psychological importance
        if self.use_dominance:
            weights = [0.5, 0.3, 0.2]  # valence, arousal, dominance
            v1 = [pad1.valence, pad1.arousal, pad1.dominance]
            v2 = [pad2.valence, pad2.arousal, pad2.dominance]
        else:
            weights = [0.6, 0.4]  # valence, arousal
            v1 = [pad1.valence, pad1.arousal]
            v2 = [pad2.valence, pad2.arousal]
        
        # Calculate weighted Euclidean distance
        weighted_squared_sum = sum(
            w * (a - b) ** 2 
            for w, a, b in zip(weights, v1, v2)
        )
        distance = math.sqrt(weighted_squared_sum)
        
        # Maximum possible distance with these weights
        max_distances = [2.0, 1.0, 1.0]  # valence range 2, arousal range 1, dominance range 1
        max_weighted_squared = sum(
            w * (m) ** 2 
            for w, m in zip(weights, max_distances[:len(weights)])
        )
        max_distance = math.sqrt(max_weighted_squared)
        
        # Convert to similarity (1 - normalized distance)
        similarity = 1.0 - (distance / max_distance)
        return max(0.0, min(1.0, similarity))
    
    def get_emotional_region(self, pad: PADVector) -> List[Dict[str, Any]]:
        """
        Determine which emotional regions a PAD vector falls into.
        
        Args:
            pad: PAD vector to classify
            
        Returns:
            List of region information dictionaries
        """
        regions = []
        
        # Check 2D regions
        for name, bounds in self.regions_2d.items():
            if (bounds['valence'][0] <= pad.valence <= bounds['valence'][1] and
                bounds['arousal'][0] <= pad.arousal <= bounds['arousal'][1]):
                regions.append({
                    'name': name,
                    'type': '2d',
                    'description': bounds['description']
                })
        
        # Check 3D regions if using dominance
        if self.use_dominance:
            for name, bounds in self.regions_3d.items():
                if (bounds['valence'][0] <= pad.valence <= bounds['valence'][1] and
                    bounds['arousal'][0] <= pad.arousal <= bounds['arousal'][1] and
                    bounds['dominance'][0] <= pad.dominance <= bounds['dominance'][1]):
                    regions.append({
                        'name': name,
                        'type': '3d',
                        'description': bounds['description']
                    })
        
        return regions
    
    def interpolate_emotions(self, emotion1: str, emotion2: str, 
                             t: float) -> PADVector:
        """
        Interpolate between two emotion prototypes.
        
        Args:
            emotion1: First emotion name
            emotion2: Second emotion name
            t: Interpolation factor (0 = emotion1, 1 = emotion2)
            
        Returns:
            Interpolated PAD vector
            
        Raises:
            ValueError: If either emotion is unknown or t is invalid
        """
        if not 0 <= t <= 1:
            raise ValueError(f"Interpolation factor t must be between 0 and 1, got {t}")
        
        pad1 = EmotionLexicon.get_pad(emotion1)
        pad2 = EmotionLexicon.get_pad(emotion2)
        
        if pad1 is None:
            raise ValueError(f"Unknown emotion: {emotion1}")
        if pad2 is None:
            raise ValueError(f"Unknown emotion: {emotion2}")
        
        return pad1.interpolate(pad2, t)
    
    def blend_emotions(self, emotions: List[Tuple[str, float]]) -> PADVector:
        """
        Blend multiple emotions with weights.
        
        Args:
            emotions: List of (emotion_name, weight) tuples
                     Weights should be non-negative
            
        Returns:
            Weighted average PAD vector
            
        Raises:
            ValueError: If emotions list is empty or contains invalid weights
        """
        if not emotions:
            raise ValueError("Cannot blend empty emotion list")
        
        # Validate weights
        for name, weight in emotions:
            if weight < 0:
                raise ValueError(f"Weight must be non-negative, got {weight} for {name}")
            if not EmotionLexicon.has_emotion(name):
                raise ValueError(f"Unknown emotion: {name}")
        
        # Calculate weighted sum
        total_weight = sum(weight for _, weight in emotions)
        if total_weight == 0:
            raise ValueError("Total weight must be greater than 0")
        
        # Start with neutral vector
        result = PADVector(0.0, 0.5, 0.5)
        result = result * 0  # Zero out to start accumulation
        
        for name, weight in emotions:
            pad = EmotionLexicon.get_pad(name)
            if pad:
                result += pad * weight
        
        return result / total_weight
    
    def enhance_wednesday_style(self, pad: PADVector, 
                                 intensity: float = 0.3) -> PADVector:
        """
        Apply Wednesday-style emotional coloring to a PAD vector.
        
        This shifts emotions toward Wednesday's characteristic patterns:
        - Slightly lower valence (darker perspective)
        - Slightly higher dominance (in control)
        - More emotional control (lower arousal expression)
        
        Args:
            pad: Original PAD vector
            intensity: How strongly to apply Wednesday style (0-1)
            
        Returns:
            Wednesday-styled PAD vector
            
        Raises:
            ValueError: If intensity is outside valid range
        """
        if not 0 <= intensity <= 1:
            raise ValueError(f"Intensity must be between 0 and 1, got {intensity}")
        
        # Wednesday's signature emotional bias
        wednesday_bias = PADVector(
            valence=-0.1,  # Slightly darker
            arousal=-0.1,  # More controlled
            dominance=0.2   # More in control
        )
        
        # Apply bias weighted by intensity
        result = pad + (wednesday_bias * intensity)
        
        # Ensure we stay within reasonable bounds for Wednesday
        # (She rarely experiences extreme emotions)
        result.valence = max(-0.8, min(0.8, result.valence))
        result.arousal = max(0.1, min(0.9, result.arousal))
        result.dominance = max(0.2, min(0.9, result.dominance))
        
        return result
    
    def to_circumplex_angle(self, pad: PADVector) -> float:
        """
        Convert valence-arousal to angle on Russell's circumplex.
        
        Returns angle in radians (0 = high arousal/neutral valence,
        increasing counterclockwise)
        
        Args:
            pad: PAD vector
            
        Returns:
            Angle in radians (-π to π)
        """
        # Scale valence and arousal to range -1 to 1 for circle
        x = pad.valence  # Already -1 to 1
        y = 2 * pad.arousal - 1  # Map arousal 0-1 to -1-1
        
        return math.atan2(y, x)
    
    def to_circumplex_coords(self, pad: PADVector) -> Tuple[float, float]:
        """
        Convert to coordinates on Russell's circumplex.
        
        Returns (x, y) where:
        x = valence (right = positive)
        y = arousal (up = high arousal)
        
        Args:
            pad: PAD vector
            
        Returns:
            (x, y) coordinates in range -1 to 1
        """
        return (pad.valence, 2 * pad.arousal - 1)
    
    def from_circumplex_coords(self, x: float, y: float) -> PADVector:
        """
        Convert from circumplex coordinates to PAD vector.
        
        Args:
            x: Valence coordinate (-1 to 1)
            y: Arousal coordinate (-1 to 1)
            
        Returns:
            PADVector with default dominance (0.5)
        """
        valence = max(-1.0, min(1.0, x))
        arousal = max(0.0, min(1.0, (y + 1) / 2))
        return PADVector(valence, arousal, 0.5)
    
    def __repr__(self) -> str:
        return f"ValenceArousalSpace(dimensions={self.dimensions})"


# Utility functions for emotional math
def pad_distance_weighted(pad1: PADVector, pad2: PADVector, 
                          weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate weighted distance between two PAD vectors.
    
    Args:
        pad1: First PAD vector
        pad2: Second PAD vector
        weights: Dictionary with 'valence', 'arousal', 'dominance' weights
                (default: valence=0.5, arousal=0.3, dominance=0.2)
        
    Returns:
        Weighted Euclidean distance
        
    Raises:
        ValueError: If weights are invalid
    """
    if weights is None:
        weights = {'valence': 0.5, 'arousal': 0.3, 'dominance': 0.2}
    
    # Validate weights
    required_keys = {'valence', 'arousal', 'dominance'}
    if not all(key in weights for key in required_keys):
        raise ValueError(f"Weights must contain keys: {required_keys}")
    
    for key, value in weights.items():
        if not 0 <= value <= 1:
            raise ValueError(f"Weight for {key} must be between 0 and 1, got {value}")
    
    # Normalize weights to sum to 1
    total = sum(weights.values())
    if total == 0:
        raise ValueError("Weights cannot sum to zero")
    
    norm_weights = {k: v / total for k, v in weights.items()}
    
    # Calculate weighted squared distance
    squared_diff = (
        norm_weights['valence'] * (pad1.valence - pad2.valence) ** 2 +
        norm_weights['arousal'] * (pad1.arousal - pad2.arousal) ** 2 +
        norm_weights['dominance'] * (pad1.dominance - pad2.dominance) ** 2
    )
    
    return math.sqrt(squared_diff)


def emotional_contrast(pad: PADVector) -> PADVector:
    """
    Return the emotional opposite (contrasting emotion).
    
    Maps to roughly opposite position in emotional space.
    
    Args:
        pad: Input PAD vector
        
    Returns:
        Contrasting PAD vector
    """
    return PADVector(
        valence=-pad.valence,
        arousal=1.0 - pad.arousal,
        dominance=1.0 - pad.dominance
    )


def is_emotionally_congruent(pad1: PADVector, pad2: PADVector, 
                              threshold: float = 0.7) -> bool:
    """
    Check if two emotional states are congruent (similar).
    
    Uses a combination of cosine similarity and valence agreement.
    
    Args:
        pad1: First PAD vector
        pad2: Second PAD vector
        threshold: Similarity threshold (0-1)
        
    Returns:
        True if emotions are congruent
        
    Raises:
        ValueError: If threshold is outside valid range
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    
    # Calculate cosine similarity
    similarity = pad1.cosine_similarity(pad2)
    
    # Calculate valence agreement (normalized to 0-1)
    valence_agreement = 1.0 - (abs(pad1.valence - pad2.valence) / 2.0)
    
    # Combine measures (valence weighted more heavily)
    combined = (similarity * 0.4 + valence_agreement * 0.6)
    
    return combined >= threshold


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Valence-Arousal Module Test ===\n")
    
    # Test PADVector
    print("-- PADVector Operations --")
    v1 = PADVector(0.5, 0.6, 0.7)
    v2 = PADVector(-0.3, 0.8, 0.4)
    
    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"Distance: {v1.distance_to(v2):.3f}")
    print(f"Cosine similarity: {v1.cosine_similarity(v2):.3f}")
    
    # Test interpolation
    v3 = v1.interpolate(v2, 0.5)
    print(f"Interpolated (0.5): {v3}")
    
    # Test arithmetic
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 * 2 = {v1 * 2}")
    
    # Test EmotionLexicon
    print("\n-- Emotion Lexicon --")
    joy_pad = EmotionLexicon.get_pad('joy')
    print(f"Joy PAD: {joy_pad}")
    
    # List available emotions
    all_emotions = EmotionLexicon.get_all_names()
    print(f"Available emotions ({len(all_emotions)}): {all_emotions[:5]}...")
    
    # Find closest emotion to a point
    test_pad = PADVector(0.2, 0.5, 0.6)
    closest = EmotionLexicon.closest_emotion(test_pad, threshold=0.6)
    print(f"\nClosest emotions to {test_pad}:")
    for name, sim in closest:
        print(f"  {name}: similarity {sim:.3f}")
    
    # Test ValenceArousalSpace
    print("\n-- ValenceArousalSpace --")
    vas = ValenceArousalSpace(use_dominance=True)
    
    # Test emotional similarity
    sim = vas.emotional_similarity(
        PADVector(0.8, 0.7, 0.6),  # Joy
        PADVector(0.7, 0.6, 0.5)   # Slightly different joy
    )
    print(f"Similarity between similar emotions: {sim:.3f}")
    
    # Test region classification
    test_pad = PADVector(-0.4, 0.7, 0.6)  # Angry-like
    regions = vas.get_emotional_region(test_pad)
    print(f"\nRegions for {test_pad}:")
    for region in regions:
        print(f"  {region['name']} ({region['type']}): {region['description']}")
    
    # Test Wednesday style enhancement
    original = PADVector(0.3, 0.6, 0.5)  # Some emotion
    wednesday_style = vas.enhance_wednesday_style(original, intensity=0.7)
    print(f"\nWednesday style enhancement:")
    print(f"  Original: {original}")
    print(f"  Wednesday style: {wednesday_style}")
    
    # Test emotion blending
    print("\n-- Emotion Blending --")
    try:
        blended = vas.blend_emotions([
            ('joy', 0.3),
            ('trust', 0.5),
            ('dark_amusement', 0.2)
        ])
        print(f"Blended emotion: {blended}")
        
        # Find what emotion this is closest to
        closest_blend = EmotionLexicon.closest_emotion(blended, threshold=0.5)
        if closest_blend:
            print(f"Closest to: {closest_blend[0][0]} (sim: {closest_blend[0][1]:.3f})")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test circumplex conversion
    print("\n-- Circumplex Model --")
    test_pad = PADVector(0.6, 0.7, 0.5)
    x, y = vas.to_circumplex_coords(test_pad)
    angle = vas.to_circumplex_angle(test_pad)
    print(f"PAD {test_pad} -> Circumplex: ({x:.2f}, {y:.2f}), angle: {angle:.2f} rad")
    
    # Test utility functions
    print("\n-- Utility Functions --")
    contrast = emotional_contrast(test_pad)
    print(f"Contrast of {test_pad}: {contrast}")
    
    congruent = is_emotionally_congruent(
        PADVector(0.8, 0.7, 0.6),
        PADVector(0.7, 0.6, 0.5),
        threshold=0.7
    )
    print(f"Joy and similar joy congruent? {congruent}")
    
    print("\n=== Test Complete ===")