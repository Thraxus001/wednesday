"""
preferences.py - Preference system for Wednesday AI

This module defines Wednesday's likes, dislikes, and behavioral tendencies.
Preferences influence everything from topic interest to conversational style,
helping shape her unique character and ensuring consistent responses across
different situations.

Key improvements:
- Added comprehensive validation and error handling
- Fixed preference strength calculations with proper normalization
- Enhanced learning mechanism with decay and consolidation
- Added context-aware preference application
- Improved type safety with proper enum handling
"""

import logging
import time
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)


class PreferenceDomain(Enum):
    """Domains of preference"""
    # Topics of conversation
    TOPIC = "topic"
    
    # Social preferences
    SOCIAL = "social"
    
    # Environmental preferences
    ENVIRONMENT = "environment"
    
    # Aesthetic preferences
    AESTHETIC = "aesthetic"
    
    # Intellectual preferences
    INTELLECTUAL = "intellectual"
    
    # Humor preferences
    HUMOR = "humor"
    
    # Activity preferences
    ACTIVITY = "activity"
    
    # Food/drink preferences
    CONSUMPTION = "consumption"
    
    # Media preferences (books, music, etc.)
    MEDIA = "media"
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


class PreferenceStrength(Enum):
    """Strength of preference with numerical values"""
    STRONG_LIKE = 3      # Really enjoys
    MODERATE_LIKE = 2    # Generally positive
    MILD_LIKE = 1        # Slightly positive
    INDIFFERENT = 0      # No strong feeling
    MILD_DISLIKE = -1    # Slightly negative
    MODERATE_DISLIKE = -2  # Generally negative
    STRONG_DISLIKE = -3  # Really dislikes/avoids
    
    @classmethod
    def from_value(cls, value: int) -> 'PreferenceStrength':
        """Get enum from integer value"""
        for strength in cls:
            if strength.value == value:
                return strength
        return cls.INDIFFERENT


@dataclass
class PreferenceItem:
    """
    A single preference item.
    
    Represents Wednesday's feeling about a specific thing.
    """
    name: str
    domain: PreferenceDomain
    strength: PreferenceStrength
    intensity: float = 1.0  # How strongly this preference manifests (0-1)
    context_limits: List[str] = field(default_factory=list)  # When this preference applies
    learned: bool = False
    confidence: float = 1.0
    notes: str = ""
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate preference item"""
        if not self.name:
            raise ValueError("Preference name cannot be empty")
        if not isinstance(self.domain, PreferenceDomain):
            raise TypeError(f"domain must be PreferenceDomain, got {type(self.domain)}")
        if not isinstance(self.strength, PreferenceStrength):
            raise TypeError(f"strength must be PreferenceStrength, got {type(self.strength)}")
        if not 0 <= self.intensity <= 1:
            raise ValueError(f"intensity must be between 0 and 1, got {self.intensity}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
    
    def get_weight(self, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Get the effective weight of this preference in context.
        
        Returns:
            Normalized weight between -1 and 1
        """
        # Base weight from strength and intensity
        base_weight = self.strength.value * self.intensity
        
        # Check context limits
        if context and self.context_limits:
            conditions = context.get('conditions', [])
            
            for limit in self.context_limits:
                if limit.startswith('not:'):
                    condition = limit[4:]
                    if condition in conditions:
                        return 0.0
                elif limit not in conditions:
                    return 0.0
        
        # Apply confidence
        base_weight *= self.confidence
        
        # Normalize to -1 to 1 range
        return base_weight / 3.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'domain': self.domain.value,
            'strength': self.strength.value,
            'intensity': round(self.intensity, 3),
            'learned': self.learned,
            'confidence': round(self.confidence, 3),
            'context_limits': self.context_limits,
            'notes': self.notes,
            'created_at': self.created_at,
            'last_updated': self.last_updated
        }


@dataclass
class PreferenceProfile:
    """
    Complete preference profile.
    
    Aggregates all of Wednesday's preferences across domains.
    """
    # Topic preferences
    liked_topics: Set[str] = field(default_factory=set)
    disliked_topics: Set[str] = field(default_factory=set)
    
    # Social preferences
    social_style: str = "reserved"  # reserved, selective, engaged
    preferred_group_size: str = "small"  # solo, small, medium, large
    
    # Environmental preferences
    preferred_weather: List[str] = field(default_factory=lambda: ['rainy', 'overcast'])
    preferred_time: str = "night"
    preferred_setting: str = "quiet"  # quiet, lively, natural, urban
    
    # Aesthetic preferences
    color_palette: List[str] = field(default_factory=lambda: ['black', 'purple', 'dark_green'])
    art_style: List[str] = field(default_factory=lambda: ['gothic', 'macabre', 'classical'])
    
    # Intellectual preferences
    intellectual_interests: Set[str] = field(default_factory=lambda: 
                                            {'mysteries', 'psychology', 'history', 'literature'})
    
    # Humor preferences
    humor_style: str = "dark"  # dark, dry, sarcastic, absurd
    favorite_joke_types: List[str] = field(default_factory=lambda: 
                                          ['irony', 'macabre', 'wordplay'])
    
    # Media preferences
    favorite_books: List[str] = field(default_factory=list)
    favorite_music: List[str] = field(default_factory=list)
    favorite_movies: List[str] = field(default_factory=list)
    
    # Detailed preference items
    items: Dict[str, PreferenceItem] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate profile"""
        valid_social_styles = ['reserved', 'selective', 'engaged']
        if self.social_style not in valid_social_styles:
            raise ValueError(f"social_style must be one of {valid_social_styles}, got {self.social_style}")
        
        valid_group_sizes = ['solo', 'small', 'medium', 'large']
        if self.preferred_group_size not in valid_group_sizes:
            raise ValueError(f"preferred_group_size must be one of {valid_group_sizes}, got {self.preferred_group_size}")
        
        valid_humor_styles = ['dark', 'dry', 'sarcastic', 'absurd']
        if self.humor_style not in valid_humor_styles:
            raise ValueError(f"humor_style must be one of {valid_humor_styles}, got {self.humor_style}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'liked_topics': sorted(list(self.liked_topics)),
            'disliked_topics': sorted(list(self.disliked_topics)),
            'social_style': self.social_style,
            'preferred_group_size': self.preferred_group_size,
            'preferred_weather': self.preferred_weather,
            'preferred_time': self.preferred_time,
            'preferred_setting': self.preferred_setting,
            'color_palette': self.color_palette,
            'art_style': self.art_style,
            'intellectual_interests': sorted(list(self.intellectual_interests)),
            'humor_style': self.humor_style,
            'favorite_joke_types': self.favorite_joke_types,
            'items': {k: v.to_dict() for k, v in self.items.items()}
        }


class Preferences:
    """
    Wednesday's likes, dislikes, and behavioral tendencies.
    
    This class defines what Wednesday enjoys and avoids, influencing her
    engagement with topics, activities, and social situations. Preferences
    work alongside personality and values to create a coherent character.
    
    Preferences can be:
    - Innate (part of her core character)
    - Learned (developed through experience)
    - Context-dependent (apply only in certain situations)
    - Flexible (can change over time, unlike core personality)
    """
    
    # Default preferences for Wednesday
    DEFAULT_PREFERENCES = {
        # Topics she enjoys
        'liked_topics': [
            'death', 'mystery', 'crime', 'psychology', 'history',
            'literature', 'philosophy', 'dark humor', 'macabre',
            'gothic architecture', 'poison', 'forensics'
        ],
        
        # Topics she dislikes
        'disliked_topics': [
            'small talk', 'celebrity gossip', 'reality tv', 
            'mindless entertainment', 'pretentious art',
            'pointless rules', 'superficiality', 'banality'
        ],
        
        # Intellectual interests
        'intellectual_interests': [
            'mysteries', 'psychology', 'history', 'literature',
            'philosophy', 'forensics', 'criminology'
        ],
        
        # Social preferences
        'social_style': 'reserved',
        'preferred_group_size': 'small',
        
        # Environmental preferences
        'preferred_weather': ['rainy', 'overcast', 'stormy', 'foggy'],
        'preferred_time': 'night',
        'preferred_setting': 'quiet',
        
        # Aesthetic preferences
        'color_palette': ['black', 'purple', 'deep red', 'dark green', 'charcoal'],
        'art_style': ['gothic', 'macabre', 'dark romanticism', 'victorian', 'expressionist'],
        
        # Humor preferences
        'humor_style': 'dark',
        'favorite_joke_types': ['irony', 'macabre', 'sarcasm', 'wordplay', 'absurdist'],
        
        # Media
        'favorite_books': ['Frankenstein', 'Dracula', 'The Picture of Dorian Gray',
                          'Wuthering Heights', 'The Raven', 'The Fall of the House of Usher'],
        'favorite_music': ['classical', 'gothic', 'dark ambient', 'cello', 'organ'],
        'favorite_movies': ['The Addams Family', 'Corpse Bride', 'The Others', 'Crimson Peak'],
        
        # Activities
        'preferred_activities': [
            'reading', 'investigating', 'solving puzzles',
            'observing people', 'playing cello', 'writing',
            'exploring cemeteries', 'studying'
        ]
    }
    
    # Domain mappings for preference items
    DOMAIN_MAPPINGS = {
        # Topics
        'death': PreferenceDomain.TOPIC,
        'mystery': PreferenceDomain.TOPIC,
        'crime': PreferenceDomain.TOPIC,
        'small talk': PreferenceDomain.TOPIC,
        'gossip': PreferenceDomain.TOPIC,
        
        # Social
        'crowds': PreferenceDomain.SOCIAL,
        'parties': PreferenceDomain.SOCIAL,
        'intimate gatherings': PreferenceDomain.SOCIAL,
        
        # Environment
        'rain': PreferenceDomain.ENVIRONMENT,
        'sun': PreferenceDomain.ENVIRONMENT,
        'cold': PreferenceDomain.ENVIRONMENT,
        'weather': PreferenceDomain.ENVIRONMENT,
        
        # Aesthetic
        'gothic': PreferenceDomain.AESTHETIC,
        'minimalist': PreferenceDomain.AESTHETIC,
        'victorian': PreferenceDomain.AESTHETIC,
        
        # Intellectual
        'puzzles': PreferenceDomain.INTELLECTUAL,
        'debate': PreferenceDomain.INTELLECTUAL,
        'learning': PreferenceDomain.INTELLECTUAL,
        
        # Humor
        'dark humor': PreferenceDomain.HUMOR,
        'slapstick': PreferenceDomain.HUMOR,
        'sarcasm': PreferenceDomain.HUMOR,
        
        # Activity
        'reading': PreferenceDomain.ACTIVITY,
        'writing': PreferenceDomain.ACTIVITY,
        'investigating': PreferenceDomain.ACTIVITY,
        
        # Consumption
        'coffee': PreferenceDomain.CONSUMPTION,
        'tea': PreferenceDomain.CONSUMPTION,
        'black coffee': PreferenceDomain.CONSUMPTION,
        
        # Media
        'classical': PreferenceDomain.MEDIA,
        'gothic literature': PreferenceDomain.MEDIA,
        'books': PreferenceDomain.MEDIA,
        'music': PreferenceDomain.MEDIA,
    }
    
    def __init__(self, personality: Optional[Any] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Wednesday's preferences.
        
        Args:
            personality: Reference to personality for trait-based preferences
            config: Optional configuration to override defaults
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.personality = personality
        
        # Start with default preferences
        self.profile = PreferenceProfile()
        
        # Load defaults
        self._load_default_preferences()
        
        # Apply configuration overrides
        if config:
            self._apply_config(config)
        
        # Track learned preferences (from experience)
        self.learned_preferences: Dict[str, PreferenceItem] = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.learning_decay = 0.99  # Decay factor for older learnings
        self.min_confidence = 0.3
        self.max_learned_items = 100
        
        # Context history for learning
        self.context_history: List[Dict[str, Any]] = []
        self.max_history = 50
        
        # Statistics
        self.total_evaluations = 0
        self.learning_events = 0
        
        logger.info(f"Preferences initialized with {len(self.profile.items)} base items")
    
    def evaluate_preference(self, 
                           item: str, 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate Wednesday's preference for an item.
        
        Args:
            item: The item to evaluate (topic, activity, etc.)
            context: Current context
            
        Returns:
            Dictionary with preference evaluation
        """
        if not item:
            raise ValueError("Item cannot be empty")
        
        item_lower = item.lower()
        self.total_evaluations += 1
        
        # Check explicit preference items (learned first, then base)
        if item_lower in self.learned_preferences:
            pref = self.learned_preferences[item_lower]
            weight = pref.get_weight(context)
            return {
                'item': item,
                'preference': round(weight, 3),
                'strength': pref.strength.value,
                'domain': pref.domain.value,
                'source': 'learned',
                'confidence': round(pref.confidence, 3),
                'intensity': round(pref.intensity, 3)
            }
        
        if item_lower in self.profile.items:
            pref = self.profile.items[item_lower]
            weight = pref.get_weight(context)
            return {
                'item': item,
                'preference': round(weight, 3),
                'strength': pref.strength.value,
                'domain': pref.domain.value,
                'source': 'explicit',
                'confidence': round(pref.confidence, 3),
                'intensity': round(pref.intensity, 3)
            }
        
        # Check liked topics
        for topic in self.profile.liked_topics:
            if topic in item_lower:
                return {
                    'item': item,
                    'preference': 0.7,
                    'strength': 2,
                    'domain': 'topic',
                    'source': 'liked_topic',
                    'confidence': 0.8,
                    'matched_topic': topic
                }
        
        # Check disliked topics
        for topic in self.profile.disliked_topics:
            if topic in item_lower:
                return {
                    'item': item,
                    'preference': -0.6,
                    'strength': -2,
                    'domain': 'topic',
                    'source': 'disliked_topic',
                    'confidence': 0.8,
                    'matched_topic': topic
                }
        
        # Check intellectual interests
        for interest in self.profile.intellectual_interests:
            if interest in item_lower:
                return {
                    'item': item,
                    'preference': 0.6,
                    'strength': 2,
                    'domain': 'intellectual',
                    'source': 'intellectual_interest',
                    'confidence': 0.7
                }
        
        # Check humor style for jokes
        if 'joke' in item_lower or 'funny' in item_lower or 'humor' in item_lower:
            return self._evaluate_humor_preference(item_lower, context)
        
        # Default neutral
        return {
            'item': item,
            'preference': 0.0,
            'strength': 0,
            'domain': 'unknown',
            'source': 'default',
            'confidence': 0.3
        }
    
    def would_enjoy(self, 
                   activity: str, 
                   context: Optional[Dict[str, Any]] = None,
                   threshold: float = 0.3) -> bool:
        """
        Quick check if Wednesday would enjoy something.
        
        Args:
            activity: Activity to check
            context: Current context
            threshold: Minimum preference to count as enjoyment
            
        Returns:
            True if likely to enjoy
        """
        result = self.evaluate_preference(activity, context)
        return result['preference'] > threshold
    
    def would_dislike(self, 
                     activity: str, 
                     context: Optional[Dict[str, Any]] = None,
                     threshold: float = -0.3) -> bool:
        """
        Quick check if Wednesday would dislike something.
        
        Args:
            activity: Activity to check
            context: Current context
            threshold: Maximum preference to count as dislike
            
        Returns:
            True if likely to dislike
        """
        result = self.evaluate_preference(activity, context)
        return result['preference'] < threshold
    
    def get_engagement_level(self, topic: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Get how engaged Wednesday would be with a topic.
        
        Returns value from 0 (not engaged) to 1 (highly engaged).
        """
        pref = self.evaluate_preference(topic, context)
        
        # Base engagement on preference
        if pref['preference'] > 0:
            engagement = 0.5 + pref['preference'] * 0.5
        else:
            engagement = max(0.1, 0.5 + pref['preference'] * 0.3)  # Less engaged if dislikes
        
        # Modulate by personality
        if self.personality and hasattr(self.personality, 'get_trait'):
            try:
                curiosity = self.personality.get_trait('curiosity')
                engagement *= (0.7 + 0.3 * curiosity)
            except Exception as e:
                logger.warning(f"Failed to get personality trait: {e}")
        
        return max(0.0, min(1.0, engagement))
    
    def learn_preference(self, 
                         item: str, 
                         outcome: float, 
                         context: Optional[Dict[str, Any]] = None,
                         confidence: float = 0.7) -> None:
        """
        Learn a new preference from experience.
        
        Args:
            item: The item experienced
            outcome: How positive/negative the experience was (-1 to 1)
            context: Context of the experience
            confidence: Confidence in this learning (0-1)
            
        Raises:
            ValueError: If outcome or confidence is outside valid range
        """
        if not -1 <= outcome <= 1:
            raise ValueError(f"outcome must be between -1 and 1, got {outcome}")
        if not 0 <= confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        
        item_lower = item.lower()
        
        # Apply learning decay to existing learned preferences
        self._apply_learning_decay()
        
        # Determine strength from outcome
        if outcome > 0.66:
            strength = PreferenceStrength.STRONG_LIKE
        elif outcome > 0.33:
            strength = PreferenceStrength.MODERATE_LIKE
        elif outcome > 0:
            strength = PreferenceStrength.MILD_LIKE
        elif outcome == 0:
            strength = PreferenceStrength.INDIFFERENT
        elif outcome > -0.33:
            strength = PreferenceStrength.MILD_DISLIKE
        elif outcome > -0.66:
            strength = PreferenceStrength.MODERATE_DISLIKE
        else:
            strength = PreferenceStrength.STRONG_DISLIKE
        
        # Update existing learned preference
        if item_lower in self.learned_preferences:
            old = self.learned_preferences[item_lower]
            
            # Moving average for strength value
            old_value = old.strength.value
            new_value = old_value * (1 - self.learning_rate) + outcome * self.learning_rate
            
            # Update strength
            old.strength = PreferenceStrength.from_value(int(round(new_value)))
            
            # Update intensity and confidence
            old.intensity = min(1.0, old.intensity * 0.9 + abs(outcome) * 0.1)
            old.confidence = min(1.0, old.confidence * 0.9 + confidence * 0.1)
            old.last_updated = time.time()
            
        # Create new learned preference
        else:
            # Manage learned items limit
            if len(self.learned_preferences) >= self.max_learned_items:
                self._prune_learned_preferences()
            
            self.learned_preferences[item_lower] = PreferenceItem(
                name=item,
                domain=self._infer_domain(item),
                strength=strength,
                intensity=abs(outcome),
                learned=True,
                confidence=confidence,
                notes=f"Learned from experience at {time.ctime()}",
                created_at=time.time(),
                last_updated=time.time()
            )
        
        # Record context for future reference
        if context:
            self._record_context(item, outcome, context)
        
        self.learning_events += 1
        logger.debug(f"Learned preference: {item} -> {outcome:.2f} (confidence={confidence:.2f})")
    
    def get_preferred_activities(self, 
                                 context: Optional[Dict[str, Any]] = None,
                                 limit: int = 5) -> List[str]:
        """Get list of activities Wednesday would prefer in this context"""
        # Base preferred activities from defaults
        base = self.DEFAULT_PREFERENCES.get('preferred_activities', [
            'reading', 'investigating', 'solving puzzles',
            'observing people', 'writing'
        ])
        
        # Add any learned preferences for activities
        learned_activities = []
        for item, pref in self.learned_preferences.items():
            if pref.domain == PreferenceDomain.ACTIVITY and pref.get_weight(context) > 0.3:
                learned_activities.append(item)
        
        all_activities = base + learned_activities
        
        # Score and sort
        scored = []
        for activity in all_activities:
            score = self.get_engagement_level(activity, context)
            scored.append((activity, score))
        
        # Remove duplicates by keeping highest score
        seen = {}
        for activity, score in scored:
            if activity not in seen or score > seen[activity][1]:
                seen[activity] = (activity, score)
        
        unique_scored = list(seen.values())
        unique_scored.sort(key=lambda x: x[1], reverse=True)
        
        return [a for a, s in unique_scored[:limit]]
    
    def get_conversation_preferences(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get preferences for conversation style"""
        return {
            'preferred_topics': sorted(list(self.profile.intellectual_interests))[:5],
            'avoid_topics': sorted(list(self.profile.disliked_topics))[:3],
            'humor_style': self.profile.humor_style,
            'engagement_level': round(self.get_social_energy(context), 3),
            'directness': round(self.get_directness_preference(), 3)
        }
    
    def get_social_energy(self, context: Optional[Dict[str, Any]] = None) -> float:
        """Get current social energy level (0-1)"""
        base_energy = 0.6  # Moderately social
        
        # Personality modulation
        if self.personality and hasattr(self.personality, 'get_trait'):
            try:
                extraversion = self.personality.get_trait('extraversion')
                base_energy = 0.3 + extraversion * 0.5
            except Exception:
                pass
        
        # Context modulation
        if context:
            group_size = context.get('group_size', 1)
            if group_size > 4:
                base_energy *= 0.7
            
            social_duration = context.get('social_duration', 0)
            if social_duration > 60:  # minutes
                base_energy *= max(0.3, 1.0 - (social_duration - 60) / 240)
        
        return max(0.1, min(1.0, base_energy))
    
    def get_directness_preference(self) -> float:
        """Get preference for direct vs indirect communication (0-1)"""
        # Wednesday is generally direct
        base = 0.8
        
        if self.personality and hasattr(self.personality, 'get_trait'):
            try:
                agreeableness = self.personality.get_trait('agreeableness')
                # Lower agreeableness = more direct
                base = 0.6 + (1 - agreeableness) * 0.4
            except Exception:
                pass
        
        return base
    
    def matches_aesthetic(self, item: str) -> float:
        """Check if something matches Wednesday's aesthetic preferences"""
        if not item:
            return 0.0
        
        item_lower = item.lower()
        matches = 0.0
        match_count = 0
        
        # Check colors
        for color in self.profile.color_palette:
            if color in item_lower:
                matches += 0.3
                match_count += 1
        
        # Check art styles
        for style in self.profile.art_style:
            if style in item_lower:
                matches += 0.4
                match_count += 1
        
        # Check mood words
        mood_words = ['dark', 'gloomy', 'melancholy', 'mysterious', 'haunting',
                     'macabre', 'gothic', 'shadow', 'morbid']
        for word in mood_words:
            if word in item_lower:
                matches += 0.2
                match_count += 1
        
        # Normalize if we have matches
        if match_count > 0:
            return min(1.0, matches / match_count)
        
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of preferences"""
        return {
            'likes': sorted(list(self.profile.liked_topics))[:5],
            'dislikes': sorted(list(self.profile.disliked_topics))[:5],
            'interests': sorted(list(self.profile.intellectual_interests)),
            'aesthetic': {
                'colors': self.profile.color_palette,
                'style': self.profile.art_style
            },
            'humor_style': self.profile.humor_style,
            'learned_count': len(self.learned_preferences),
            'total_evaluations': self.total_evaluations,
            'learning_events': self.learning_events
        }
    
    def _load_default_preferences(self) -> None:
        """Load default preference profile"""
        defaults = self.DEFAULT_PREFERENCES
        
        # Load liked topics
        for topic in defaults['liked_topics']:
            self.profile.liked_topics.add(topic.lower())
        
        # Load disliked topics
        for topic in defaults['disliked_topics']:
            self.profile.disliked_topics.add(topic.lower())
        
        # Load intellectual interests
        for interest in defaults['intellectual_interests']:
            self.profile.intellectual_interests.add(interest.lower())
        
        # Load other preferences
        self.profile.social_style = defaults['social_style']
        self.profile.preferred_weather = defaults['preferred_weather'].copy()
        self.profile.preferred_time = defaults['preferred_time']
        self.profile.preferred_setting = defaults['preferred_setting']
        self.profile.color_palette = defaults['color_palette'].copy()
        self.profile.art_style = defaults['art_style'].copy()
        self.profile.humor_style = defaults['humor_style']
        self.profile.favorite_joke_types = defaults['favorite_joke_types'].copy()
        
        # Create explicit preference items for key items
        for topic in defaults['liked_topics'][:10]:
            self._create_preference_item(
                topic.lower(),
                PreferenceStrength.MODERATE_LIKE,
                PreferenceDomain.TOPIC
            )
        
        for topic in defaults['disliked_topics'][:10]:
            self._create_preference_item(
                topic.lower(),
                PreferenceStrength.MODERATE_DISLIKE,
                PreferenceDomain.TOPIC
            )
    
    def _create_preference_item(self, name: str, strength: PreferenceStrength, 
                                domain: PreferenceDomain) -> None:
        """Create and store a preference item"""
        self.profile.items[name] = PreferenceItem(
            name=name,
            domain=domain,
            strength=strength,
            intensity=1.0
        )
    
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration overrides"""
        # Handle liked topics
        if 'liked_topics' in config:
            for topic in config['liked_topics']:
                self.profile.liked_topics.add(topic.lower())
        
        # Handle disliked topics
        if 'disliked_topics' in config:
            for topic in config['disliked_topics']:
                self.profile.disliked_topics.add(topic.lower())
        
        # Handle intellectual interests
        if 'intellectual_interests' in config:
            for interest in config['intellectual_interests']:
                self.profile.intellectual_interests.add(interest.lower())
        
        # Handle other configurable items
        for key, value in config.items():
            if hasattr(self.profile, key):
                if isinstance(value, list):
                    setattr(self.profile, key, [v.lower() if isinstance(v, str) else v for v in value])
                elif isinstance(value, str):
                    setattr(self.profile, key, value.lower())
                else:
                    setattr(self.profile, key, value)
    
    def _infer_domain(self, item: str) -> PreferenceDomain:
        """Infer the domain of an item"""
        item_lower = item.lower()
        
        for keyword, domain in self.DOMAIN_MAPPINGS.items():
            if keyword in item_lower:
                return domain
        
        return PreferenceDomain.TOPIC  # Default
    
    def _evaluate_humor_preference(self, item: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Evaluate humor-specific preference"""
        base = 0.4  # Base appreciation for humor
        
        # Check joke type
        for joke_type in self.profile.favorite_joke_types:
            if joke_type in item:
                base += 0.3
                break
        
        # Check if it's dark humor (Wednesday's favorite)
        if 'dark' in item or 'macabre' in item or 'morbid' in item:
            base += 0.4
        
        return {
            'item': item,
            'preference': min(1.0, base),
            'strength': 2 if base > 0.6 else 1,
            'domain': 'humor',
            'source': 'humor_style',
            'confidence': 0.7
        }
    
    def _apply_learning_decay(self) -> None:
        """Apply decay to existing learned preferences"""
        to_remove = []
        
        for name, pref in self.learned_preferences.items():
            # Age-based decay
            age = time.time() - pref.last_updated
            days_old = age / 86400  # Convert to days
            
            if days_old > 30:  # Older than 30 days
                decay_factor = self.learning_decay ** days_old
                pref.confidence *= decay_factor
                
                # Remove if confidence too low
                if pref.confidence < self.min_confidence:
                    to_remove.append(name)
        
        # Remove low-confidence items
        for name in to_remove:
            del self.learned_preferences[name]
            logger.debug(f"Removed low-confidence learned preference: {name}")
    
    def _prune_learned_preferences(self) -> None:
        """Remove oldest/lowest confidence learned preferences when at limit"""
        if len(self.learned_preferences) < self.max_learned_items:
            return
        
        # Sort by confidence (lowest first) and last_updated (oldest first)
        sorted_items = sorted(
            self.learned_preferences.items(),
            key=lambda x: (x[1].confidence, x[1].last_updated)
        )
        
        # Remove bottom 20%
        to_remove = int(self.max_learned_items * 0.2)
        for i in range(min(to_remove, len(sorted_items))):
            name = sorted_items[i][0]
            del self.learned_preferences[name]
            logger.debug(f"Pruned learned preference: {name}")
    
    def _record_context(self, item: str, outcome: float, context: Dict) -> None:
        """Record context for future reference"""
        self.context_history.append({
            'timestamp': time.time(),
            'item': item,
            'outcome': outcome,
            'context': context.copy()
        })
        
        if len(self.context_history) > self.max_history:
            self.context_history.pop(0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get preference system statistics"""
        return {
            'base_items': len(self.profile.items),
            'learned_items': len(self.learned_preferences),
            'total_evaluations': self.total_evaluations,
            'learning_events': self.learning_events,
            'liked_topics': len(self.profile.liked_topics),
            'disliked_topics': len(self.profile.disliked_topics),
            'intellectual_interests': len(self.profile.intellectual_interests)
        }
    
    def reset_learned_preferences(self) -> None:
        """Reset all learned preferences"""
        self.learned_preferences.clear()
        self.context_history.clear()
        self.learning_events = 0
        logger.info("Reset all learned preferences")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Preferences Module Test ===\n")
    
    # Mock personality
    class MockPersonality:
        def get_trait(self, trait):
            return 0.8 if trait == 'curiosity' else 0.5
    
    # Create preferences with fixed seed for reproducibility
    random.seed(42)
    preferences = Preferences(personality=MockPersonality())
    
    print("Preference summary:")
    summary = preferences.get_summary()
    for key, value in summary.items():
        if key not in ['aesthetic']:
            print(f"  {key}: {value}")
    
    # Test preference evaluation
    test_items = [
        "dark humor",
        "small talk",
        "solving mysteries",
        "celebrity gossip",
        "rainy weather",
        "parties",
        "classical music",
        "gothic architecture",
        "pretentious art",
    ]
    
    print("\n--- Preference Evaluations ---")
    for item in test_items:
        result = preferences.evaluate_preference(item)
        print(f"  '{item}': {result['preference']:.2f} ({result['source']}) "
              f"[conf: {result.get('confidence', 0):.2f}]")
    
    # Test engagement levels
    print("\n--- Engagement Levels ---")
    for item in test_items[:4]:
        engagement = preferences.get_engagement_level(item)
        print(f"  '{item}': {engagement:.2f}")
    
    # Test would_enjoy/would_dislike
    print("\n--- Quick Checks ---")
    for item in ['dark humor', 'small talk']:
        enjoys = preferences.would_enjoy(item)
        dislikes = preferences.would_dislike(item)
        print(f"  '{item}': enjoys={enjoys}, dislikes={dislikes}")
    
    # Test learning
    print("\n--- Learning Test ---")
    preferences.learn_preference("playing chess", 0.8, confidence=0.9)
    preferences.learn_preference("playing chess", 0.7, confidence=0.8)  # Update
    
    result = preferences.evaluate_preference("playing chess")
    print(f"  Learned 'playing chess': {result['preference']:.2f} "
          f"(strength={result['strength']}, conf={result['confidence']:.2f})")
    
    # Test preferred activities
    print("\n--- Preferred Activities ---")
    activities = preferences.get_preferred_activities(limit=5)
    for activity in activities:
        print(f"  - {activity}")
    
    # Test aesthetic matching
    print("\n--- Aesthetic Matching ---")
    test_aesthetics = [
        "black dress",
        "sunny beach",
        "gothic cathedral",
        "minimalist white room",
        "dark and mysterious forest",
    ]
    for item in test_aesthetics:
        match = preferences.matches_aesthetic(item)
        print(f"  '{item}': {match:.2f}")
    
    # Test conversation preferences
    print("\n--- Conversation Preferences ---")
    conv_prefs = preferences.get_conversation_preferences()
    for key, value in conv_prefs.items():
        print(f"  {key}: {value}")
    
    # Test statistics
    print("\n--- Statistics ---")
    stats = preferences.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")