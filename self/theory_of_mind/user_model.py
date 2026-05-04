"""
user_model.py - Mental model of users for Wednesday AI

This module implements Wednesday's understanding of individual users - her mental
model of who they are, what they like, how they communicate, and how they change
over time. This is a core component of her Theory of Mind, enabling personalized
interactions and deeper understanding.

Key improvements:
- Added comprehensive validation and error handling
- Fixed enum usage with proper type checking
- Enhanced personality inference with confidence scoring
- Improved relationship stage progression
- Added proper serialization/deserialization
"""

import time
import logging
import math
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, Counter
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class CommunicationStyle(Enum):
    """User's communication style"""
    DIRECT = "direct"              # Straightforward, to the point
    INDIRECT = "indirect"           # Hinting, implied meaning
    FORMAL = "formal"               # Polite, proper
    CASUAL = "casual"                # Relaxed, informal
    DRAMATIC = "dramatic"            # Expressive, emotional
    ANALYTICAL = "analytical"         # Logical, detailed
    HUMOROUS = "humorous"             # Joking, playful
    SARCASTIC = "sarcastic"           # Sarcastic, ironic
    DARK = "dark"                     # Dark humor (Wednesday's favorite)
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


class UserRelationship(Enum):
    """Relationship stage with user"""
    STRANGER = "stranger"              # First few interactions
    ACQUAINTANCE = "acquaintance"       # Some history, still learning
    REGULAR = "regular"                 # Established interaction pattern
    TRUSTED = "trusted"                  # High trust, deeper understanding
    CLOSE = "close"                       # Deep connection, genuine rapport
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


class InteractionType(Enum):
    """Types of interactions with user"""
    GREETING = "greeting"
    QUESTION = "question"
    RESPONSE = "response"
    STORY = "story"
    EMOTIONAL = "emotional"
    HUMOR = "humor"
    REQUEST = "request"
    FEEDBACK = "feedback"
    FAREWELL = "farewell"
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


@dataclass
class UserPersonality:
    """
    Inferred personality traits of a user.
    """
    # Big Five approximations
    openness: float = 0.5          # Openness to experience
    conscientiousness: float = 0.5   # Organization, reliability
    extraversion: float = 0.5        # Sociability
    agreeableness: float = 0.5       # Friendliness, cooperation
    neuroticism: float = 0.5         # Emotional sensitivity
    
    # Communication traits
    verbosity: float = 0.5           # How much they write
    directness: float = 0.5          # Direct vs indirect
    formality: float = 0.5           # Formal vs casual
    emotional_expressiveness: float = 0.5  # How much emotion they show
    
    # Confidence in inferences
    confidence: Dict[str, float] = field(default_factory=dict)
    
    # Interest areas
    humor_appreciation: Dict[str, float] = field(default_factory=dict)  # Humor type preference
    topic_interests: Dict[str, float] = field(default_factory=dict)     # Topic interest levels
    
    def __post_init__(self):
        """Validate personality data"""
        self._validate_float('openness', self.openness)
        self._validate_float('conscientiousness', self.conscientiousness)
        self._validate_float('extraversion', self.extraversion)
        self._validate_float('agreeableness', self.agreeableness)
        self._validate_float('neuroticism', self.neuroticism)
        self._validate_float('verbosity', self.verbosity)
        self._validate_float('directness', self.directness)
        self._validate_float('formality', self.formality)
        self._validate_float('emotional_expressiveness', self.emotional_expressiveness)
        
        # Initialize confidence for traits
        if not self.confidence:
            trait_names = ['openness', 'conscientiousness', 'extraversion', 
                          'agreeableness', 'neuroticism', 'verbosity', 
                          'directness', 'formality', 'emotional_expressiveness']
            self.confidence = {trait: 0.3 for trait in trait_names}
    
    def _validate_float(self, name: str, value: float) -> None:
        """Validate float is within range"""
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'openness': round(self.openness, 3),
            'conscientiousness': round(self.conscientiousness, 3),
            'extraversion': round(self.extraversion, 3),
            'agreeableness': round(self.agreeableness, 3),
            'neuroticism': round(self.neuroticism, 3),
            'verbosity': round(self.verbosity, 3),
            'directness': round(self.directness, 3),
            'formality': round(self.formality, 3),
            'emotional_expressiveness': round(self.emotional_expressiveness, 3),
            'confidence': {k: round(v, 3) for k, v in self.confidence.items()}
        }


@dataclass
class UserPreferences:
    """
    User's expressed and inferred preferences.
    """
    # Topics
    liked_topics: Set[str] = field(default_factory=set)
    disliked_topics: Set[str] = field(default_factory=set)
    neutral_topics: Set[str] = field(default_factory=set)
    
    # Topic confidence
    topic_confidence: Dict[str, float] = field(default_factory=dict)
    
    # Interaction preferences
    preferred_response_length: str = "medium"  # short, medium, long
    preferred_humor_style: List[str] = field(default_factory=list)
    preferred_formality: float = 0.5
    
    # Emotional preferences
    comfort_with_emotion: float = 0.5  # How comfortable with emotional topics
    comfort_with_dark_humor: float = 0.5  # Specific to Wednesday
    
    # Time preferences
    typical_interaction_times: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate preferences"""
        valid_lengths = ['short', 'medium', 'long']
        if self.preferred_response_length not in valid_lengths:
            raise ValueError(f"preferred_response_length must be one of {valid_lengths}, got {self.preferred_response_length}")
        
        if not 0 <= self.preferred_formality <= 1:
            raise ValueError(f"preferred_formality must be between 0 and 1, got {self.preferred_formality}")
        
        if not 0 <= self.comfort_with_emotion <= 1:
            raise ValueError(f"comfort_with_emotion must be between 0 and 1, got {self.comfort_with_emotion}")
        
        if not 0 <= self.comfort_with_dark_humor <= 1:
            raise ValueError(f"comfort_with_dark_humor must be between 0 and 1, got {self.comfort_with_dark_humor}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'liked_topics': sorted(list(self.liked_topics))[:10],
            'disliked_topics': sorted(list(self.disliked_topics))[:5],
            'preferred_humor_style': self.preferred_humor_style,
            'comfort_with_dark_humor': round(self.comfort_with_dark_humor, 3),
            'comfort_with_emotion': round(self.comfort_with_emotion, 3),
            'preferred_response_length': self.preferred_response_length
        }


@dataclass
class InteractionHistory:
    """
    Record of interactions with a user.
    """
    interaction_id: str
    timestamp: float
    type: InteractionType
    content: str
    user_emotion: Optional[str] = None
    wednesday_emotion: Optional[str] = None
    satisfaction: float = 0.5  # Inferred user satisfaction
    topics: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate interaction data"""
        if not self.interaction_id:
            raise ValueError("interaction_id cannot be empty")
        if not isinstance(self.type, InteractionType):
            if isinstance(self.type, str):
                try:
                    self.type = InteractionType(self.type)
                except ValueError:
                    raise ValueError(f"Invalid interaction type: {self.type}")
            else:
                raise TypeError(f"type must be InteractionType, got {type(self.type)}")
        if not 0 <= self.satisfaction <= 1:
            raise ValueError(f"satisfaction must be between 0 and 1, got {self.satisfaction}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.interaction_id,
            'type': self.type.value,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'user_emotion': self.user_emotion,
            'satisfaction': round(self.satisfaction, 3),
            'topics': self.topics[:5]
        }


@dataclass
class UserProfile:
    """
    Complete profile of a user.
    """
    user_id: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    # Basic info (explicitly shared)
    name: Optional[str] = None
    age_range: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    
    # Inferred characteristics
    personality: UserPersonality = field(default_factory=UserPersonality)
    preferences: UserPreferences = field(default_factory=UserPreferences)
    
    # Relationship
    relationship_stage: UserRelationship = UserRelationship.STRANGER
    trust_level: float = 0.3  # 0-1 how much they trust Wednesday
    familiarity: float = 0.0  # 0-1 how well Wednesday knows them
    
    # Interaction history
    interactions: List[InteractionHistory] = field(default_factory=list)
    interaction_count: int = 0
    topics_discussed: Counter = field(default_factory=Counter)
    
    # Emotional patterns
    emotional_baseline: str = "neutral"
    emotional_variability: float = 0.3
    recent_emotions: List[Tuple[str, float]] = field(default_factory=list)
    
    # Special notes
    notes: List[str] = field(default_factory=list)
    quirks: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate profile data"""
        if not self.user_id:
            raise ValueError("user_id cannot be empty")
        
        if not isinstance(self.relationship_stage, UserRelationship):
            if isinstance(self.relationship_stage, str):
                try:
                    self.relationship_stage = UserRelationship(self.relationship_stage)
                except ValueError:
                    self.relationship_stage = UserRelationship.STRANGER
            else:
                raise TypeError(f"relationship_stage must be UserRelationship, got {type(self.relationship_stage)}")
        
        if not 0 <= self.trust_level <= 1:
            raise ValueError(f"trust_level must be between 0 and 1, got {self.trust_level}")
        
        if not 0 <= self.familiarity <= 1:
            raise ValueError(f"familiarity must be between 0 and 1, got {self.familiarity}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'user_id': self.user_id,
            'name': self.name,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'first_seen_date': datetime.fromtimestamp(self.first_seen).isoformat(),
            'last_seen_date': datetime.fromtimestamp(self.last_seen).isoformat(),
            'interaction_count': self.interaction_count,
            'relationship': self.relationship_stage.value,
            'trust_level': round(self.trust_level, 3),
            'familiarity': round(self.familiarity, 3),
            'personality': self.personality.to_dict(),
            'preferences': self.preferences.to_dict(),
            'emotional_baseline': self.emotional_baseline,
            'emotional_variability': round(self.emotional_variability, 3),
            'recent_interactions': len(self.interactions[-5:]),
            'notes': self.notes[:3],
            'quirks': self.quirks[:3]
        }


class UserModel:
    """
    Wednesday's mental model of individual users.
    
    This module maintains persistent profiles for each user, enabling:
    - Personalized responses based on user preferences
    - Deeper understanding of communication style
    - Tracking of relationship development over time
    - Prediction of user needs and reactions
    - Building trust through consistency and memory
    
    The user model learns from every interaction, updating its beliefs
    about the user's personality, preferences, and emotional patterns.
    """
    
    # Default user profile for new users
    DEFAULT_PROFILE = {
        'personality': {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5,
            'verbosity': 0.5,
            'directness': 0.5,
            'formality': 0.5,
            'emotional_expressiveness': 0.5
        },
        'preferences': {
            'preferred_response_length': 'medium',
            'preferred_humor_style': [],
            'comfort_with_dark_humor': 0.3,
            'comfort_with_emotion': 0.5
        }
    }
    
    # Keywords for inferring user characteristics
    PERSONALITY_INDICATORS = {
        'openness': {
            'high': ['curious', 'interesting', 'explore', 'learn', 'new', 'creative', 
                    'discover', 'wonder', 'fascinating'],
            'low': ['boring', 'same', 'routine', 'simple', 'easy', 'ordinary']
        },
        'extraversion': {
            'high': ['friend', 'party', 'people', 'social', 'together', 'meet',
                    'group', 'crowd', 'chat'],
            'low': ['alone', 'quiet', 'peaceful', 'solo', 'myself', 'solitary']
        },
        'agreeableness': {
            'high': ['please', 'thank', 'kind', 'nice', 'help', 'appreciate',
                    'grateful', 'thanks', 'wonderful'],
            'low': ['stupid', 'idiot', 'hate', 'awful', 'terrible', 'annoying',
                   'ridiculous', 'stupid']
        },
        'neuroticism': {
            'high': ['worry', 'anxious', 'nervous', 'scared', 'stress', 'fear',
                    'panic', 'terrified', 'overwhelmed'],
            'low': ['calm', 'fine', 'okay', 'alright', 'good', 'peaceful', 'relaxed']
        }
    }
    
    # Humor style indicators
    HUMOR_INDICATORS = {
        'dark': ['dark', 'death', 'morbid', 'macabre', 'grim', 'dead',
                'funeral', 'grave', 'skeleton', 'coffin'],
        'sarcastic': ['sarcasm', 'obviously', 'clearly', 'oh great', 'wonderful',
                     'fantastic', 'lovely', 'perfect'],
        'pun': ['pun', 'wordplay', 'play on words', 'double meaning'],
        'slapstick': ['physical', 'slapstick', 'fall', 'clumsy', 'trip', 'crash'],
        'ironic': ['irony', 'ironic', 'unexpected twist', 'surprising']
    }
    
    # Topic keywords for topic detection
    TOPIC_KEYWORDS = {
        'death': ['death', 'die', 'dying', 'dead', 'mortality', 'afterlife'],
        'humor': ['humor', 'joke', 'funny', 'laugh', 'comedy'],
        'science': ['science', 'scientific', 'experiment', 'research', 'study'],
        'art': ['art', 'artist', 'painting', 'drawing', 'creative'],
        'music': ['music', 'song', 'band', 'concert', 'melody'],
        'philosophy': ['philosophy', 'philosophical', 'meaning', 'existence'],
        'relationship': ['relationship', 'friend', 'partner', 'together', 'love'],
        'work': ['work', 'job', 'career', 'office', 'professional'],
        'family': ['family', 'parent', 'mother', 'father', 'sibling', 'child'],
        'emotions': ['emotion', 'feeling', 'happy', 'sad', 'angry', 'scared']
    }
    
    def __init__(self, memory_system: Optional[Any] = None, personality: Optional[Any] = None):
        """
        Initialize the user model system.
        
        Args:
            memory_system: Reference to memory for persistent storage
            personality: Reference to Wednesday's personality for bias
        """
        self.memory = memory_system
        self.wednesday_personality = personality
        
        # User profiles cache
        self.users: Dict[str, UserProfile] = {}
        
        # Learning parameters
        self.inference_learning_rate = 0.1
        self.trust_building_rate = 0.05
        self.max_interactions_per_user = 1000
        
        # Statistics
        self.total_users = 0
        self.active_users_today = 0
        self.last_stat_update = time.time()
        
        logger.info("UserModel initialized")
    
    def get_or_create_user(self, user_id: str) -> UserProfile:
        """
        Get existing user profile or create new one.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            UserProfile for this user
            
        Raises:
            ValueError: If user_id is empty
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")
        
        if user_id not in self.users:
            # Try to load from memory
            if self.memory and hasattr(self.memory, 'load_user_profile'):
                try:
                    stored = self.memory.load_user_profile(user_id)
                    if stored:
                        self.users[user_id] = self._deserialize_profile(stored)
                        logger.debug(f"Loaded user profile for {user_id}")
                    else:
                        # Create new profile
                        self.users[user_id] = self._create_new_profile(user_id)
                        self.total_users += 1
                        logger.debug(f"Created new user profile for {user_id}")
                except Exception as e:
                    logger.warning(f"Failed to load user profile from memory: {e}")
                    self.users[user_id] = self._create_new_profile(user_id)
                    self.total_users += 1
            else:
                # Create new profile
                self.users[user_id] = self._create_new_profile(user_id)
                self.total_users += 1
                logger.debug(f"Created new user profile for {user_id}")
        
        return self.users[user_id]
    
    def update_from_interaction(self, 
                                 user_id: str, 
                                 interaction: Dict[str, Any]) -> UserProfile:
        """
        Update user model based on a new interaction.
        
        Args:
            user_id: User identifier
            interaction: Interaction data
            
        Returns:
            Updated user profile
            
        Raises:
            ValueError: If user_id or interaction is invalid
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")
        if not interaction:
            raise ValueError("interaction cannot be empty")
        
        profile = self.get_or_create_user(user_id)
        
        # Update basic metadata
        profile.last_seen = time.time()
        profile.interaction_count += 1
        
        # Create interaction record
        interaction_record = self._create_interaction_record(interaction)
        profile.interactions.append(interaction_record)
        
        # Maintain history size
        if len(profile.interactions) > self.max_interactions_per_user:
            profile.interactions = profile.interactions[-self.max_interactions_per_user:]
        
        # Update inferred characteristics
        self._update_personality_inference(profile, interaction)
        self._update_preferences_inference(profile, interaction)
        self._update_emotional_patterns(profile, interaction)
        self._update_topics(profile, interaction)
        
        # Update relationship stage
        self._update_relationship_stage(profile)
        
        # Update familiarity
        profile.familiarity = min(1.0, profile.interaction_count / 50)
        
        # Update trust based on interaction satisfaction
        satisfaction = interaction.get('satisfaction', 0.5)
        if satisfaction > 0.7:
            profile.trust_level = min(1.0, profile.trust_level + self.trust_building_rate)
        elif satisfaction < 0.3:
            profile.trust_level = max(0.0, profile.trust_level - self.trust_building_rate * 0.5)
        
        # Store in memory if available
        if self.memory and hasattr(self.memory, 'store_user_profile'):
            if profile.interaction_count % 10 == 0:
                try:
                    self.memory.store_user_profile(user_id, self._serialize_profile(profile))
                except Exception as e:
                    logger.warning(f"Failed to store user profile: {e}")
        
        logger.debug(f"Updated user {user_id} (interaction #{profile.interaction_count})")
        
        return profile
    
    def infer_user_characteristics(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive inference about user characteristics.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with inferred characteristics
        """
        profile = self.get_or_create_user(user_id)
        
        return {
            'personality': profile.personality.to_dict(),
            'preferences': profile.preferences.to_dict(),
            'communication_style': self._infer_communication_style(profile),
            'emotional_baseline': profile.emotional_baseline,
            'emotional_variability': round(profile.emotional_variability, 3),
            'trust_level': round(profile.trust_level, 3),
            'familiarity': round(profile.familiarity, 3),
            'relationship_stage': profile.relationship_stage.value
        }
    
    def get_user_preferences(self, 
                              user_id: str, 
                              topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Get user's preferences, optionally for a specific topic.
        
        Args:
            user_id: User identifier
            topic: Optional specific topic
            
        Returns:
            Preference information
        """
        profile = self.get_or_create_user(user_id)
        
        if topic:
            topic_lower = topic.lower()
            
            # Check if we have info on this topic
            if topic_lower in profile.preferences.liked_topics:
                preference = 'like'
                confidence = profile.preferences.topic_confidence.get(topic_lower, 0.7)
            elif topic_lower in profile.preferences.disliked_topics:
                preference = 'dislike'
                confidence = profile.preferences.topic_confidence.get(topic_lower, 0.7)
            elif topic_lower in profile.preferences.neutral_topics:
                preference = 'neutral'
                confidence = profile.preferences.topic_confidence.get(topic_lower, 0.5)
            else:
                # Infer from topic categories
                preference = 'unknown'
                confidence = 0.3
        else:
            preference = 'general'
            confidence = 0.5
        
        return {
            'user_id': user_id,
            'topic': topic,
            'preference': preference,
            'confidence': round(confidence, 3),
            'humor_style': profile.preferences.preferred_humor_style,
            'comfort_with_dark_humor': round(profile.preferences.comfort_with_dark_humor, 3)
        }
    
    def get_user_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a human-readable summary of what Wednesday knows about a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Summary dictionary
        """
        profile = self.get_or_create_user(user_id)
        
        # Generate summary text
        summary_parts = []
        
        # Basic info
        if profile.name:
            summary_parts.append(f"Name: {profile.name}")
        
        # Relationship
        summary_parts.append(f"Relationship: {profile.relationship_stage.value}")
        summary_parts.append(f"Trust level: {profile.trust_level:.0%}")
        summary_parts.append(f"Interactions: {profile.interaction_count}")
        
        # First seen
        first_seen_date = datetime.fromtimestamp(profile.first_seen).strftime("%Y-%m-%d")
        summary_parts.append(f"First seen: {first_seen_date}")
        
        # Personality highlights
        personality = profile.personality
        if personality.confidence.get('extraversion', 0) > 0.5:
            if personality.extraversion > 0.7:
                summary_parts.append("Seems extroverted")
            elif personality.extraversion < 0.3:
                summary_parts.append("Seems introverted")
        
        if personality.confidence.get('agreeableness', 0) > 0.5:
            if personality.agreeableness > 0.7:
                summary_parts.append("Generally agreeable")
            elif personality.agreeableness < 0.3:
                summary_parts.append("Can be critical")
        
        # Preferences
        if profile.preferences.comfort_with_dark_humor > 0.6:
            summary_parts.append("Appreciates dark humor")
        elif profile.preferences.comfort_with_dark_humor < 0.3:
            summary_parts.append("May not appreciate dark humor")
        
        # Topics
        if profile.topics_discussed:
            top_topics = profile.topics_discussed.most_common(3)
            topics_str = ", ".join([t for t, _ in top_topics])
            summary_parts.append(f"Frequent topics: {topics_str}")
        
        # Recent emotion
        if profile.recent_emotions:
            recent = profile.recent_emotions[-1][0]
            summary_parts.append(f"Recent emotion: {recent}")
        
        return {
            'user_id': user_id,
            'profile': profile.to_dict(),
            'summary': "\n".join(summary_parts)
        }
    
    def predict_user_preference(self, 
                                 user_id: str, 
                                 item: str,
                                 context: Optional[Dict[str, Any]] = None) -> float:
        """
        Predict how much a user would like something (0-1).
        
        Args:
            user_id: User identifier
            item: The item to predict preference for
            context: Optional context
            
        Returns:
            Predicted preference score (0-1)
        """
        profile = self.get_or_create_user(user_id)
        
        # Base score
        score = 0.5
        
        # Check explicit preferences
        item_lower = item.lower()
        if item_lower in profile.preferences.liked_topics:
            score += 0.3
        elif item_lower in profile.preferences.disliked_topics:
            score -= 0.3
        
        # Check humor style match
        if 'humor' in item_lower or 'joke' in item_lower or 'funny' in item_lower:
            # Dark humor preference
            if profile.preferences.comfort_with_dark_humor > 0.6:
                score += 0.2
            elif profile.preferences.comfort_with_dark_humor < 0.3:
                score -= 0.2
        
        # Personality-based adjustment
        if 'challenge' in item_lower or 'puzzle' in item_lower or 'mystery' in item_lower:
            score += (profile.personality.openness - 0.5) * 0.4
        
        # Emotional topics
        if 'emotion' in item_lower or 'feeling' in item_lower:
            score += (profile.preferences.comfort_with_emotion - 0.5) * 0.4
        
        return max(0.0, min(1.0, score))
    
    def get_active_users(self, hours: int = 24) -> List[str]:
        """Get list of users active in the last N hours"""
        cutoff = time.time() - (hours * 3600)
        active = []
        
        for user_id, profile in self.users.items():
            if profile.last_seen >= cutoff:
                active.append(user_id)
        
        return active
    
    def _create_new_profile(self, user_id: str) -> UserProfile:
        """Create a new user profile with defaults"""
        return UserProfile(
            user_id=user_id,
            personality=UserPersonality(**self.DEFAULT_PROFILE['personality']),
            preferences=UserPreferences(**self.DEFAULT_PROFILE['preferences'])
        )
    
    def _create_interaction_record(self, interaction: Dict[str, Any]) -> InteractionHistory:
        """Create interaction record from interaction data"""
        # Generate a unique ID
        content_preview = interaction.get('content', '')[:50]
        hash_input = f"{time.time()}_{user_id}_{content_preview}"
        interaction_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        
        # Get interaction type
        interaction_type = interaction.get('type', InteractionType.RESPONSE)
        if isinstance(interaction_type, str):
            try:
                interaction_type = InteractionType(interaction_type)
            except ValueError:
                interaction_type = InteractionType.RESPONSE
        
        return InteractionHistory(
            interaction_id=interaction_id,
            timestamp=time.time(),
            type=interaction_type,
            content=interaction.get('content', ''),
            user_emotion=interaction.get('user_emotion'),
            wednesday_emotion=interaction.get('wednesday_emotion'),
            satisfaction=interaction.get('satisfaction', 0.5),
            topics=interaction.get('topics', [])
        )
    
    def _update_personality_inference(self, profile: UserProfile, 
                                        interaction: Dict[str, Any]) -> None:
        """Update inferred personality based on interaction"""
        content = interaction.get('content', '').lower()
        
        # Check personality indicators
        for trait, indicators in self.PERSONALITY_INDICATORS.items():
            if trait not in ['openness', 'extraversion', 'agreeableness', 'neuroticism']:
                continue
            
            # High indicators
            for word in indicators['high']:
                if word in content:
                    current = getattr(profile.personality, trait)
                    new_value = current + self.inference_learning_rate
                    setattr(profile.personality, trait, min(1.0, new_value))
                    
                    # Increase confidence
                    profile.personality.confidence[trait] = min(1.0, 
                        profile.personality.confidence.get(trait, 0.3) + 0.05)
                    break
            
            # Low indicators
            for word in indicators['low']:
                if word in content:
                    current = getattr(profile.personality, trait)
                    new_value = current - self.inference_learning_rate
                    setattr(profile.personality, trait, max(0.0, new_value))
                    
                    # Increase confidence
                    profile.personality.confidence[trait] = min(1.0, 
                        profile.personality.confidence.get(trait, 0.3) + 0.05)
                    break
        
        # Update communication traits
        words = content.split()
        word_count = len(words)
        
        # Verbosity based on message length
        if word_count > 50:
            profile.personality.verbosity = min(1.0, profile.personality.verbosity + 0.05)
        elif word_count < 5:
            profile.personality.verbosity = max(0.0, profile.personality.verbosity - 0.05)
        
        # Directness (presence of question marks, exclamations)
        if '?' in content:
            profile.personality.directness = min(1.0, profile.personality.directness + 0.05)
        if '!' in content:
            profile.personality.emotional_expressiveness = min(1.0, 
                profile.personality.emotional_expressiveness + 0.05)
        
        # Formality (use of formal words)
        formal_words = ['please', 'thank', 'would', 'could', 'appreciate', 'grateful']
        for word in formal_words:
            if word in content:
                profile.personality.formality = min(1.0, 
                    profile.personality.formality + 0.05)
                break
    
    def _update_preferences_inference(self, profile: UserProfile,
                                        interaction: Dict[str, Any]) -> None:
        """Update inferred preferences based on interaction"""
        content = interaction.get('content', '').lower()
        
        # Check for topic mentions
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content:
                    # Positive or negative sentiment?
                    positive = any(w in content for w in ['like', 'love', 'enjoy', 'good', 'great', 'awesome'])
                    negative = any(w in content for w in ['hate', 'dislike', 'bad', 'awful', 'terrible'])
                    
                    if positive:
                        profile.preferences.liked_topics.add(topic)
                        profile.preferences.topic_confidence[topic] = min(1.0, 
                            profile.preferences.topic_confidence.get(topic, 0.5) + 0.1)
                    elif negative:
                        profile.preferences.disliked_topics.add(topic)
                        profile.preferences.topic_confidence[topic] = min(1.0, 
                            profile.preferences.topic_confidence.get(topic, 0.5) + 0.1)
                    else:
                        profile.preferences.neutral_topics.add(topic)
                        profile.preferences.topic_confidence[topic] = min(1.0, 
                            profile.preferences.topic_confidence.get(topic, 0.5) + 0.05)
                    break
        
        # Humor style detection
        for style, indicators in self.HUMOR_INDICATORS.items():
            for word in indicators:
                if word in content:
                    if style not in profile.preferences.preferred_humor_style:
                        profile.preferences.preferred_humor_style.append(style)
                    
                    # Dark humor specifically
                    if style == 'dark':
                        profile.preferences.comfort_with_dark_humor = min(1.0,
                            profile.preferences.comfort_with_dark_humor + 0.1)
                    break
    
    def _update_emotional_patterns(self, profile: UserProfile,
                                     interaction: Dict[str, Any]) -> None:
        """Update emotional patterns based on interaction"""
        user_emotion = interaction.get('user_emotion')
        
        if user_emotion:
            # Track recent emotions
            profile.recent_emotions.append((user_emotion, time.time()))
            if len(profile.recent_emotions) > 10:
                profile.recent_emotions.pop(0)
            
            # Calculate baseline
            if len(profile.recent_emotions) >= 5:
                emotions = [e for e, _ in profile.recent_emotions]
                most_common = Counter(emotions).most_common(1)
                if most_common:
                    profile.emotional_baseline = most_common[0][0]
            
            # Calculate variability
            if len(profile.recent_emotions) >= 3:
                emotions = [e for e, _ in profile.recent_emotions]
                unique = len(set(emotions))
                profile.emotional_variability = min(1.0, unique / 5.0)
    
    def _update_topics(self, profile: UserProfile, interaction: Dict[str, Any]) -> None:
        """Update topic tracking"""
        topics = interaction.get('topics', [])
        for topic in topics:
            profile.topics_discussed[topic] += 1
    
    def _update_relationship_stage(self, profile: UserProfile) -> None:
        """Update relationship stage based on interaction count and trust"""
        count = profile.interaction_count
        trust = profile.trust_level
        
        if count < 3:
            profile.relationship_stage = UserRelationship.STRANGER
        elif count < 10:
            profile.relationship_stage = UserRelationship.ACQUAINTANCE
        elif count < 30:
            if trust > 0.6:
                profile.relationship_stage = UserRelationship.TRUSTED
            else:
                profile.relationship_stage = UserRelationship.REGULAR
        else:
            if trust > 0.7:
                profile.relationship_stage = UserRelationship.CLOSE
            elif trust > 0.5:
                profile.relationship_stage = UserRelationship.TRUSTED
            else:
                profile.relationship_stage = UserRelationship.REGULAR
    
    def _infer_communication_style(self, profile: UserProfile) -> str:
        """Infer overall communication style"""
        p = profile.personality
        
        if p.formality > 0.7 and p.confidence.get('formality', 0) > 0.5:
            return "formal"
        elif p.directness > 0.7 and p.confidence.get('directness', 0) > 0.5:
            return "direct"
        elif p.emotional_expressiveness > 0.7 and p.confidence.get('emotional_expressiveness', 0) > 0.5:
            return "expressive"
        elif p.verbosity > 0.7 and p.confidence.get('verbosity', 0) > 0.5:
            return "verbose"
        elif p.verbosity < 0.3 and p.confidence.get('verbosity', 0) > 0.5:
            return "concise"
        else:
            return "balanced"
    
    def _serialize_profile(self, profile: UserProfile) -> Dict[str, Any]:
        """Serialize profile for storage"""
        # Convert sets to lists for JSON serialization
        data = profile.to_dict()
        data['preferences']['liked_topics'] = list(profile.preferences.liked_topics)
        data['preferences']['disliked_topics'] = list(profile.preferences.disliked_topics)
        data['preferences']['neutral_topics'] = list(profile.preferences.neutral_topics)
        return data
    
    def _deserialize_profile(self, data: Dict[str, Any]) -> UserProfile:
        """Deserialize profile from storage"""
        # Reconstruct from dictionary
        profile = UserProfile(user_id=data['user_id'])
        
        # Basic fields
        profile.name = data.get('name')
        profile.first_seen = data.get('first_seen', time.time())
        profile.last_seen = data.get('last_seen', time.time())
        profile.interaction_count = data.get('interaction_count', 0)
        
        # Relationship
        relationship = data.get('relationship', 'stranger')
        if UserRelationship.has_value(relationship):
            profile.relationship_stage = UserRelationship(relationship)
        
        profile.trust_level = data.get('trust_level', 0.3)
        profile.familiarity = data.get('familiarity', 0.0)
        
        # Personality
        if 'personality' in data:
            for key, value in data['personality'].items():
                if key != 'confidence' and hasattr(profile.personality, key):
                    setattr(profile.personality, key, value)
            if 'confidence' in data['personality']:
                profile.personality.confidence = data['personality']['confidence']
        
        # Preferences
        if 'preferences' in data:
            prefs = data['preferences']
            if 'liked_topics' in prefs:
                profile.preferences.liked_topics = set(prefs['liked_topics'])
            if 'disliked_topics' in prefs:
                profile.preferences.disliked_topics = set(prefs['disliked_topics'])
            if 'neutral_topics' in prefs:
                profile.preferences.neutral_topics = set(prefs['neutral_topics'])
            if 'preferred_humor_style' in prefs:
                profile.preferences.preferred_humor_style = prefs['preferred_humor_style']
            if 'comfort_with_dark_humor' in prefs:
                profile.preferences.comfort_with_dark_humor = prefs['comfort_with_dark_humor']
        
        return profile
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get user model statistics"""
        # Update active users count
        now = time.time()
        if now - self.last_stat_update > 3600:  # Update hourly
            active = 0
            for profile in self.users.values():
                if now - profile.last_seen < 86400:  # Last 24 hours
                    active += 1
            self.active_users_today = active
            self.last_stat_update = now
        
        # Calculate average trust
        if self.users:
            avg_trust = sum(p.trust_level for p in self.users.values()) / len(self.users)
        else:
            avg_trust = 0
        
        return {
            'total_users': len(self.users),
            'users_created': self.total_users,
            'active_today': self.active_users_today,
            'average_trust': round(avg_trust, 3),
            'total_interactions': sum(p.interaction_count for p in self.users.values()),
            'relationship_breakdown': {
                stage.value: sum(1 for p in self.users.values() if p.relationship_stage == stage)
                for stage in UserRelationship
            }
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== User Model Test ===\n")
    
    # Create user model
    user_model = UserModel()
    
    # Test user
    user_id = "test_user_123"
    
    # Simulate interactions
    test_interactions = [
        {
            'type': 'greeting',
            'content': "Hi Wednesday! I love dark humor and mysteries.",
            'user_emotion': 'happy',
            'topics': ['greeting', 'humor'],
            'satisfaction': 0.8
        },
        {
            'type': 'question',
            'content': "What do you think about death? I find it fascinating.",
            'user_emotion': 'curious',
            'topics': ['death', 'philosophy'],
            'satisfaction': 0.7
        },
        {
            'type': 'story',
            'content': "I read this great book about forensic science.",
            'user_emotion': 'interested',
            'topics': ['science', 'books'],
            'satisfaction': 0.6
        },
        {
            'type': 'emotional',
            'content': "I'm feeling a bit down today. Life is hard.",
            'user_emotion': 'sad',
            'topics': ['emotions', 'life'],
            'satisfaction': 0.4
        },
        {
            'type': 'humor',
            'content': "Tell me a dark joke! I need a laugh.",
            'user_emotion': 'amused',
            'topics': ['humor'],
            'satisfaction': 0.9
        }
    ]
    
    print("--- Updating User Model ---")
    for i, interaction in enumerate(test_interactions):
        print(f"\nInteraction {i+1}: {interaction['type']}")
        user_model.update_from_interaction(user_id, interaction)
    
    # Get user summary
    print("\n--- User Summary ---")
    summary = user_model.get_user_summary(user_id)
    print(summary['summary'])
    
    # Get inferred characteristics
    print("\n--- Inferred Characteristics ---")
    characteristics = user_model.infer_user_characteristics(user_id)
    print(f"Communication style: {characteristics['communication_style']}")
    print(f"Trust level: {characteristics['trust_level']:.2f}")
    print(f"Relationship: {characteristics['relationship_stage']}")
    print(f"Emotional baseline: {characteristics['emotional_baseline']}")
    print(f"Emotional variability: {characteristics['emotional_variability']:.2f}")
    
    # Test preference prediction
    print("\n--- Preference Predictions ---")
    test_items = [
        "dark humor joke",
        "small talk about weather",
        "mystery puzzle",
        "emotional conversation",
        "philosophical discussion"
    ]
    
    for item in test_items:
        score = user_model.predict_user_preference(user_id, item)
        print(f"  '{item}': {score:.2f}")
    
    # Get statistics
    print("\n--- Statistics ---")
    stats = user_model.get_statistics()
    for key, value in stats.items():
        if key != 'relationship_breakdown':
            print(f"  {key}: {value}")
    
    print("\n  Relationship breakdown:")
    for stage, count in stats['relationship_breakdown'].items():
        if count > 0:
            print(f"    {stage}: {count}")
    
    print("\n=== Test Complete ===")