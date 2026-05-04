"""
perspective_taking.py - Emotional perspective taking for Wednesday AI

This module implements Wednesday's ability to understand and imagine the emotional
perspective of others. It's a key component of her Theory of Mind, allowing her to
infer what others are feeling and how they might experience situations differently
than she would.

Key improvements:
- Removed numpy dependency (using pure Python math)
- Added comprehensive validation and error handling
- Fixed regex pattern handling
- Enhanced user profile management
- Added proper type hints and documentation
"""

import time
import logging
import re
import math
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

# Configure logging
logger = logging.getLogger(__name__)


class UserEmotionState(Enum):
    """Possible emotional states for users"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    AMUSED = "amused"
    CURIOUS = "curious"
    HURT = "hurt"
    DEFENSIVE = "defensive"
    TRUSTING = "trusting"
    SUSPICIOUS = "suspicious"
    LONELY = "lonely"
    HOPEFUL = "hopeful"
    NOSTALGIC = "nostalgic"
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


@dataclass
class UserEmotionInference:
    """
    Inference about a user's emotional state.
    
    This includes both the inferred emotion and the confidence
    in that inference, along with supporting evidence.
    """
    primary_emotion: UserEmotionState
    confidence: float  # 0-1
    emotion_intensities: Dict[UserEmotionState, float] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)  # What led to this inference
    timestamp: float = field(default_factory=time.time)
    context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate inference data"""
        if not isinstance(self.primary_emotion, UserEmotionState):
            raise TypeError(f"primary_emotion must be UserEmotionState, got {type(self.primary_emotion)}")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        # Validate emotion intensities
        for emotion, intensity in self.emotion_intensities.items():
            if not isinstance(emotion, UserEmotionState):
                raise TypeError(f"Emotion key must be UserEmotionState, got {type(emotion)}")
            if not 0 <= intensity <= 1:
                raise ValueError(f"Intensity must be between 0 and 1, got {intensity}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'primary_emotion': self.primary_emotion.value,
            'confidence': round(self.confidence, 3),
            'intensities': {k.value: round(v, 3) for k, v in self.emotion_intensities.items()},
            'evidence': self.evidence,
            'timestamp': self.timestamp
        }


@dataclass
class UserEmotionProfile:
    """
    Long-term profile of a user's emotional patterns.
    
    This helps Wednesday understand individual differences in
    emotional expression and respond appropriately to each user.
    """
    user_id: str
    interaction_count: int = 0
    
    # Emotional baselines for this user
    baseline_mood: str = "neutral"
    emotional_expressiveness: float = 0.5  # 0-1 how much they show emotions
    emotional_volatility: float = 0.5  # 0-1 how quickly they change
    
    # History of observed emotions
    emotion_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Typical emotional patterns
    common_triggers: Dict[str, List[str]] = field(default_factory=dict)  # Trigger -> emotions
    response_patterns: Dict[str, int] = field(default_factory=dict)  # Emotion -> frequency
    
    # Special considerations
    trust_level: float = 0.5  # 0-1 how much they trust Wednesday
    communication_style: str = "direct"  # direct, indirect, dramatic, etc.
    
    # Last interaction
    last_emotion: Optional[str] = None
    last_interaction_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate profile data"""
        if not 0 <= self.emotional_expressiveness <= 1:
            raise ValueError(f"emotional_expressiveness must be between 0 and 1, got {self.emotional_expressiveness}")
        if not 0 <= self.emotional_volatility <= 1:
            raise ValueError(f"emotional_volatility must be between 0 and 1, got {self.emotional_volatility}")
        if not 0 <= self.trust_level <= 1:
            raise ValueError(f"trust_level must be between 0 and 1, got {self.trust_level}")


class PerspectiveTaking:
    """
    Enables Wednesday to understand and imagine others' emotional perspectives.
    
    This module builds and maintains models of users' emotional states and patterns,
    allowing Wednesday to:
    - Infer what users are feeling from their input
    - Predict how users would feel in different situations
    - Adjust her responses based on individual differences
    - Build deeper understanding over time
    
    Wednesday's perspective taking is analytical and precise rather than
    emotionally immersive - she understands emotions without necessarily
    feeling them herself.
    """
    
    # Linguistic markers for different emotions
    EMOTION_MARKERS = {
        UserEmotionState.HAPPY: {
            'keywords': ['happy', 'glad', 'great', 'wonderful', 'love', 'excited', 
                        'awesome', 'fantastic', 'good', 'delighted', 'pleased'],
            'punctuation': ['!', '!!', '!!!'],
            'patterns': [
                re.compile(r'\bso\s+(happy|glad|excited)\b', re.IGNORECASE),
                re.compile(r'\bthis\s+is\s+(great|amazing|wonderful)\b', re.IGNORECASE),
                re.compile(r'\bi\s+(love|adore)\b', re.IGNORECASE)
            ]
        },
        UserEmotionState.SAD: {
            'keywords': ['sad', 'unhappy', 'depressed', 'down', 'gloomy', 'heartbroken',
                        'miserable', 'crying', 'upset', 'disappointed', 'grief'],
            'punctuation': ['.', '...'],
            'patterns': [
                re.compile(r'\bi\s+(feel|am)\s+(sad|down|depressed)\b', re.IGNORECASE),
                re.compile(r'\bthis\s+(sucks|blows|is awful)\b', re.IGNORECASE),
                re.compile(r'\bi\s+miss\b', re.IGNORECASE)
            ]
        },
        UserEmotionState.ANGRY: {
            'keywords': ['angry', 'mad', 'furious', 'pissed', 'annoyed', 'frustrated',
                        'hate', 'stupid', 'ridiculous', 'outrageous', 'irritated'],
            'punctuation': ['!', '!!'],
            'patterns': [
                re.compile(r'\bi\s+(hate|cannot stand|can\'t stand)\b', re.IGNORECASE),
                re.compile(r'\bthis\s+is\s+(ridiculous|unacceptable|outrageous)\b', re.IGNORECASE),
                re.compile(r'\bwhat\s+the\s+(hell|heck)\b', re.IGNORECASE)
            ]
        },
        UserEmotionState.FEARFUL: {
            'keywords': ['afraid', 'scared', 'terrified', 'worried', 'anxious', 'nervous',
                        'fear', 'panic', 'dread', 'frightened', 'alarmed'],
            'punctuation': ['.', '...', '?'],
            'patterns': [
                re.compile(r'\bi\s+am\s+(scared|afraid|worried|nervous)\b', re.IGNORECASE),
                re.compile(r'\bwhat\s+if\b', re.IGNORECASE),
                re.compile(r'\bi\s+worry\b', re.IGNORECASE)
            ]
        },
        UserEmotionState.SURPRISED: {
            'keywords': ['surprised', 'shocked', 'astonished', 'amazed', 'wow', 'unexpected',
                        'can\'t believe', 'no way', 'oh my', 'gosh'],
            'punctuation': ['!', '?', '!!'],
            'patterns': [
                re.compile(r'\boh\s+(my|wow|god)\b', re.IGNORECASE),
                re.compile(r'\bare\s+you\s+(serious|kidding)\b', re.IGNORECASE),
                re.compile(r'\bi\s+can\'t\s+believe\b', re.IGNORECASE)
            ]
        },
        UserEmotionState.CONFUSED: {
            'keywords': ['confused', 'don\'t understand', 'unclear', 'puzzled', 'perplexed',
                        'what', 'huh', 'lost', 'bewildered', 'uncertain'],
            'punctuation': ['?', '...', '?'],
            'patterns': [
                re.compile(r'\bi\s+don\'t\s+(understand|get|follow)\b', re.IGNORECASE),
                re.compile(r'\bwhat\s+do\s+you\s+mean\b', re.IGNORECASE),
                re.compile(r'\bi\'m\s+(confused|lost)\b', re.IGNORECASE)
            ]
        },
        UserEmotionState.AMUSED: {
            'keywords': ['funny', 'hilarious', 'laugh', 'amusing', 'humor', 'joke',
                        'lol', 'haha', '😂', '😄', 'comedy'],
            'punctuation': ['!', '!', '!'],
            'patterns': [
                re.compile(r'\bthat\'s\s+(funny|hilarious|amusing)\b', re.IGNORECASE),
                re.compile(r'\bi\s+laughed\b', re.IGNORECASE),
                re.compile(r'\bmade\s+me\s+laugh\b', re.IGNORECASE)
            ]
        },
        UserEmotionState.CURIOUS: {
            'keywords': ['curious', 'wonder', 'interested', 'fascinating', 'tell me more',
                        'how', 'why', 'explain', 'intrigued', 'captivated'],
            'punctuation': ['?', '?', '...'],
            'patterns': [
                re.compile(r'\bi\s+wonder\b', re.IGNORECASE),
                re.compile(r'\btell\s+me\s+more\b', re.IGNORECASE),
                re.compile(r'\bwhat\s+happens\b', re.IGNORECASE),
                re.compile(r'\bhow\s+does\s+that\s+work\b', re.IGNORECASE)
            ]
        },
        UserEmotionState.HURT: {
            'keywords': ['hurt', 'painful', 'ouch', 'wounded', 'offended', 'insulted',
                        'betrayed', 'ouch', 'wounded', 'injured'],
            'punctuation': ['.', '...', '.'],
            'patterns': [
                re.compile(r'\bthat\s+(hurt|wounded|offended)\b', re.IGNORECASE),
                re.compile(r'\bwhy\s+would\s+you\b', re.IGNORECASE),
                re.compile(r'\bi\s+feel\s+(hurt|betrayed)\b', re.IGNORECASE)
            ]
        },
        UserEmotionState.DEFENSIVE: {
            'keywords': ['defensive', 'not my fault', 'didn\'t mean', 'it\'s not',
                        'you\'re wrong', 'actually', 'but', 'however'],
            'punctuation': ['.', '!', '.'],
            'patterns': [
                re.compile(r'\bthat\'s\s+not\s+(true|fair|right)\b', re.IGNORECASE),
                re.compile(r'\bi\s+didn\'t\b', re.IGNORECASE),
                re.compile(r'\bit\'s\s+not\s+my\s+fault\b', re.IGNORECASE)
            ]
        }
    }
    
    # Default emotional baseline for new users
    DEFAULT_USER_PROFILE = {
        'baseline_mood': 'neutral',
        'emotional_expressiveness': 0.6,
        'emotional_volatility': 0.5,
        'communication_style': 'direct'
    }
    
    def __init__(self, user_model: Optional[Any] = None, personality: Optional[Dict[str, Any]] = None):
        """
        Initialize the perspective taking system.
        
        Args:
            user_model: Reference to user model for user profiles
            personality: Optional personality parameters
            
        Raises:
            ValueError: If personality parameters are invalid
        """
        self.user_model = user_model
        
        # Personality influences on perspective taking
        default_personality = {
            'empathy_style': 'analytical',  # 'analytical', 'intuitive', or 'detached'
            'emotional_intelligence': 0.8,   # How well she reads emotions (0-1)
            'bias_toward_negative': 0.6,     # Tendency to expect negative emotions (0-1)
            'skepticism': 0.5,                # Questions inferred emotions (0-1)
            'trust_building': 0.6,             # Uses perspective to build trust (0-1)
        }
        
        self.personality = default_personality.copy()
        if personality:
            self._validate_personality(personality)
            self.personality.update(personality)
        
        # User profiles cache
        self.user_profiles: Dict[str, UserEmotionProfile] = {}
        
        # Current interaction context
        self.current_inference: Optional[UserEmotionInference] = None
        
        logger.info("PerspectiveTaking initialized with empathy style: %s", 
                   self.personality['empathy_style'])
    
    def _validate_personality(self, personality: Dict[str, Any]) -> None:
        """Validate personality parameters"""
        valid_styles = {'analytical', 'intuitive', 'detached'}
        
        for key, value in personality.items():
            if key not in self.personality:
                raise ValueError(f"Unknown personality parameter: {key}")
            
            if key == 'empathy_style':
                if value not in valid_styles:
                    raise ValueError(f"empathy_style must be one of {valid_styles}, got {value}")
            else:
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise ValueError(f"{key} must be between 0 and 1, got {value}")
    
    def infer_user_emotion(self, 
                           user_input: str, 
                           user_id: str,
                           context: Optional[Dict[str, Any]] = None) -> UserEmotionInference:
        """
        Infer the user's current emotional state from their input.
        
        Args:
            user_input: The user's message or input
            user_id: Identifier for the user
            context: Conversation context
            
        Returns:
            Inference about user's emotional state
            
        Raises:
            ValueError: If user_input is empty or user_id is invalid
        """
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not user_id:
            raise ValueError("user_id cannot be empty")
        
        # Get or create user profile
        profile = self._get_user_profile(user_id)
        
        # Analyze linguistic markers
        marker_scores = self._analyze_emotion_markers(user_input)
        
        # Consider context
        context_scores = self._analyze_context_emotion(context)
        
        # Apply user profile adjustments
        adjusted_scores = self._apply_user_profile(marker_scores, profile, context_scores)
        
        # Apply personality bias
        final_scores = self._apply_personality_bias(adjusted_scores)
        
        # Determine primary emotion and confidence
        if final_scores:
            primary_emotion = max(final_scores.items(), key=lambda x: x[1])[0]
            confidence = final_scores[primary_emotion]
            
            # Normalize confidence based on evidence strength
            evidence = self._collect_evidence(user_input, primary_emotion)
            confidence = min(1.0, confidence * (0.5 + 0.5 * len(evidence) / 3))
        else:
            primary_emotion = UserEmotionState.NEUTRAL
            confidence = 0.3
            evidence = []
        
        # Create inference
        inference = UserEmotionInference(
            primary_emotion=primary_emotion,
            confidence=confidence,
            emotion_intensities={k: v for k, v in final_scores.items() if v > 0.1},
            evidence=evidence,
            context=context
        )
        
        # Update user profile with this observation
        self._update_user_profile(user_id, inference)
        
        # Store current inference
        self.current_inference = inference
        
        logger.debug("Inferred user emotion: %s (confidence=%.2f)", 
                    primary_emotion.value, confidence)
        
        return inference
    
    def imagine_user_perspective(self, 
                                 situation: str,
                                 user_id: str,
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Imagine how a user would feel in a given situation.
        
        Args:
            situation: Description of the situation
            user_id: Identifier for the user
            context: Additional context
            
        Returns:
            Predicted emotional perspective of the user
            
        Raises:
            ValueError: If situation is empty or user_id is invalid
        """
        if not situation:
            raise ValueError("situation cannot be empty")
        if not user_id:
            raise ValueError("user_id cannot be empty")
        
        profile = self._get_user_profile(user_id)
        
        # Analyze the situation
        situation_lower = situation.lower()
        
        # Determine likely emotional response based on situation type
        predicted_emotions = {}
        
        # Check for positive situations
        positive_indicators = ['good', 'great', 'happy', 'win', 'success', 'celebrate',
                              'gift', 'surprise', 'love', 'friend', 'joy', 'wonderful']
        positive_score = sum(1 for word in positive_indicators if word in situation_lower)
        if positive_score > 0:
            predicted_emotions[UserEmotionState.HAPPY] = min(1.0, 0.3 + 0.1 * positive_score)
        
        # Check for negative situations
        negative_indicators = ['bad', 'sad', 'loss', 'fail', 'hurt', 'pain',
                              'danger', 'threat', 'problem', 'difficult', 'tragedy', 'death']
        negative_score = sum(1 for word in negative_indicators if word in situation_lower)
        if negative_score > 0:
            predicted_emotions[UserEmotionState.SAD] = min(1.0, 0.2 + 0.1 * negative_score)
            predicted_emotions[UserEmotionState.FRUSTRATED] = min(1.0, 0.2 + 0.1 * negative_score)
        
        # Check for challenging situations
        challenge_indicators = ['challenge', 'problem', 'puzzle', 'difficult',
                               'hard', 'complex', 'figure out', 'mystery']
        if any(word in situation_lower for word in challenge_indicators):
            predicted_emotions[UserEmotionState.CURIOUS] = 0.4
            predicted_emotions[UserEmotionState.CONFUSED] = 0.3
        
        # Check for social situations
        social_indicators = ['friend', 'family', 'together', 'relationship', 
                            'partner', 'colleague', 'team']
        if any(word in situation_lower for word in social_indicators):
            predicted_emotions[UserEmotionState.TRUSTING] = 0.3
            predicted_emotions[UserEmotionState.LONELY] = 0.2
        
        # Apply user profile adjustments
        for emotion in list(predicted_emotions.keys()):
            # Adjust for user's baseline mood
            if profile.baseline_mood == emotion.value:
                predicted_emotions[emotion] *= 1.2
            
            # Adjust for expressiveness
            predicted_emotions[emotion] *= profile.emotional_expressiveness
            
            # Cap at 1.0
            predicted_emotions[emotion] = min(1.0, predicted_emotions[emotion])
        
        # Determine primary predicted emotion
        if predicted_emotions:
            primary = max(predicted_emotions.items(), key=lambda x: x[1])[0]
        else:
            primary = UserEmotionState.NEUTRAL
            predicted_emotions[UserEmotionState.NEUTRAL] = 0.5
        
        # Calculate confidence based on how well we know the user
        familiarity = min(1.0, profile.interaction_count / 50)
        confidence = 0.5 * (0.5 + 0.5 * familiarity)
        
        return {
            'primary_emotion': primary.value,
            'emotion_intensities': {k.value: round(v, 3) for k, v in predicted_emotions.items()},
            'situation': situation,
            'confidence': round(confidence, 3),
            'based_on_profile': profile.user_id,
            'familiarity': round(familiarity, 3)
        }
    
    def adjust_for_individual(self, user_id: str, inference: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust emotional inference based on individual user characteristics.
        
        Args:
            user_id: User identifier
            inference: Raw emotional inference
            
        Returns:
            User-adjusted inference
            
        Raises:
            ValueError: If user_id is invalid or inference is malformed
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")
        
        profile = self._get_user_profile(user_id)
        
        adjusted = inference.copy()
        
        # Adjust confidence based on how well we know the user
        familiarity = min(1.0, profile.interaction_count / 50)
        adjusted['confidence'] = adjusted.get('confidence', 0.5) * (0.7 + 0.3 * familiarity)
        
        # Adjust for user's expressiveness
        if 'intensities' in adjusted:
            for emotion, intensity in adjusted['intensities'].items():
                # More expressive users show stronger signals
                adjusted['intensities'][emotion] = min(1.0, 
                    intensity * (0.8 + 0.4 * profile.emotional_expressiveness))
        
        # Consider communication style
        if profile.communication_style == 'indirect':
            # Indirect users need more inference
            adjusted['confidence'] *= 0.8
        
        return adjusted
    
    def _get_user_profile(self, user_id: str) -> UserEmotionProfile:
        """Get or create user emotional profile"""
        if user_id not in self.user_profiles:
            # Check user model if available
            if self.user_model and hasattr(self.user_model, 'get_user_profile'):
                try:
                    model_profile = self.user_model.get_user_profile(user_id)
                    if model_profile:
                        # Convert to emotion profile
                        self.user_profiles[user_id] = UserEmotionProfile(
                            user_id=user_id,
                            baseline_mood=model_profile.get('baseline_mood', 'neutral'),
                            emotional_expressiveness=model_profile.get('expressiveness', 0.6),
                            trust_level=model_profile.get('trust_level', 0.5)
                        )
                    else:
                        # Create default
                        self.user_profiles[user_id] = UserEmotionProfile(
                            user_id=user_id,
                            **self.DEFAULT_USER_PROFILE
                        )
                except Exception as e:
                    logger.warning("Failed to get user profile from model: %s", e)
                    self.user_profiles[user_id] = UserEmotionProfile(
                        user_id=user_id,
                        **self.DEFAULT_USER_PROFILE
                    )
            else:
                # Create default
                self.user_profiles[user_id] = UserEmotionProfile(
                    user_id=user_id,
                    **self.DEFAULT_USER_PROFILE
                )
        
        return self.user_profiles[user_id]
    
    def _analyze_emotion_markers(self, text: str) -> Dict[UserEmotionState, float]:
        """Analyze text for emotional markers"""
        if not text:
            return {}
        
        text_lower = text.lower()
        scores = {}
        
        for emotion, markers in self.EMOTION_MARKERS.items():
            score = 0.0
            
            # Keyword matching
            for keyword in markers.get('keywords', []):
                if keyword in text_lower:
                    score += 0.2
                    # Break early if we've found enough keywords
                    if score >= 0.8:
                        break
            
            # Punctuation patterns
            for punct in markers.get('punctuation', []):
                if punct in text:
                    score += 0.1
            
            # Regular expression patterns
            for pattern in markers.get('patterns', []):
                if pattern.search(text):
                    score += 0.3
            
            if score > 0:
                scores[emotion] = min(1.0, score)
        
        return scores
    
    def _analyze_context_emotion(self, context: Optional[Dict]) -> Dict[UserEmotionState, float]:
        """Analyze context for emotional information"""
        if not context:
            return {}
        
        scores = {}
        
        # Check for explicit emotional context
        if 'user_emotion' in context:
            emotion_val = context['user_emotion']
            if isinstance(emotion_val, str) and UserEmotionState.has_value(emotion_val):
                try:
                    emotion = UserEmotionState(emotion_val)
                    scores[emotion] = 0.8
                except ValueError:
                    pass
            elif isinstance(emotion_val, UserEmotionState):
                scores[emotion_val] = 0.8
        
        # Consider conversation topic
        if 'topic' in context:
            topic = context['topic'].lower() if isinstance(context['topic'], str) else ""
            
            # Sad topics
            sad_indicators = ['death', 'loss', 'tragedy', 'grief', 'funeral', 'sad']
            if any(word in topic for word in sad_indicators):
                scores[UserEmotionState.SAD] = max(scores.get(UserEmotionState.SAD, 0), 0.4)
                scores[UserEmotionState.NOSTALGIC] = max(scores.get(UserEmotionState.NOSTALGIC, 0), 0.3)
            
            # Happy topics
            happy_indicators = ['joke', 'funny', 'humor', 'celebration', 'party', 'happy']
            if any(word in topic for word in happy_indicators):
                scores[UserEmotionState.AMUSED] = max(scores.get(UserEmotionState.AMUSED, 0), 0.5)
                scores[UserEmotionState.HAPPY] = max(scores.get(UserEmotionState.HAPPY, 0), 0.4)
            
            # Suspicious topics
            suspicious_indicators = ['betray', 'lie', 'deceive', 'secret', 'hidden']
            if any(word in topic for word in suspicious_indicators):
                scores[UserEmotionState.SUSPICIOUS] = max(scores.get(UserEmotionState.SUSPICIOUS, 0), 0.5)
        
        return scores
    
    def _apply_user_profile(self, 
                           marker_scores: Dict[UserEmotionState, float],
                           profile: UserEmotionProfile,
                           context_scores: Dict[UserEmotionState, float]) -> Dict[UserEmotionState, float]:
        """Apply user profile to adjust emotion scores"""
        adjusted = marker_scores.copy()
        
        # Combine with context
        for emotion, score in context_scores.items():
            adjusted[emotion] = max(adjusted.get(emotion, 0), score)
        
        # Adjust based on user's baseline
        if UserEmotionState.has_value(profile.baseline_mood):
            try:
                baseline = UserEmotionState(profile.baseline_mood)
                adjusted[baseline] = adjusted.get(baseline, 0) * 1.2
            except ValueError:
                pass
        
        # Consider recent emotion history for continuity
        if profile.last_emotion and UserEmotionState.has_value(profile.last_emotion):
            try:
                last = UserEmotionState(profile.last_emotion)
                adjusted[last] = adjusted.get(last, 0) * 1.1
            except ValueError:
                pass
        
        # Normalize if we have scores
        if adjusted:
            max_score = max(adjusted.values())
            if max_score > 0:
                adjusted = {k: v / max_score for k, v in adjusted.items()}
        
        return adjusted
    
    def _apply_personality_bias(self, scores: Dict[UserEmotionState, float]) -> Dict[UserEmotionState, float]:
        """Apply personality-based biases to emotion inference"""
        if not scores:
            return {}
        
        biased = scores.copy()
        
        # Bias toward negative emotions if personality trait high
        if self.personality['bias_toward_negative'] > 0.6:
            negative_emotions = [UserEmotionState.SAD, UserEmotionState.ANGRY, 
                                UserEmotionState.FEARFUL, UserEmotionState.FRUSTRATED,
                                UserEmotionState.HURT]
            bias_factor = 1 + self.personality['bias_toward_negative'] * 0.2
            
            for emotion in negative_emotions:
                if emotion in biased:
                    biased[emotion] = min(1.0, biased[emotion] * bias_factor)
        
        # Analytical style reduces confidence in extreme emotions
        if self.personality['empathy_style'] == 'analytical':
            for emotion in biased:
                # Slightly dampen high scores
                if biased[emotion] > 0.7:
                    biased[emotion] = 0.7 + (biased[emotion] - 0.7) * 0.5
        
        # Skepticism slightly reduces all scores
        if self.personality['skepticism'] > 0.5:
            skepticism_factor = 1 - (self.personality['skepticism'] - 0.5) * 0.2
            for emotion in biased:
                biased[emotion] *= skepticism_factor
        
        return biased
    
    def _collect_evidence(self, text: str, primary_emotion: UserEmotionState) -> List[str]:
        """Collect evidence supporting the primary emotion inference"""
        evidence = []
        text_lower = text.lower()
        
        # Find matching keywords
        markers = self.EMOTION_MARKERS.get(primary_emotion, {})
        for keyword in markers.get('keywords', []):
            if keyword in text_lower:
                evidence.append(f"keyword: '{keyword}'")
                if len(evidence) >= 3:
                    break
        
        # Check punctuation
        for punct in markers.get('punctuation', []):
            if punct in text:
                evidence.append(f"punctuation: {punct}")
                break
        
        # Check patterns
        for pattern in markers.get('patterns', []):
            if pattern.search(text):
                evidence.append("pattern match")
                break
        
        return evidence
    
    def _update_user_profile(self, user_id: str, inference: UserEmotionInference) -> None:
        """Update user profile with new emotional observation"""
        profile = self._get_user_profile(user_id)
        
        profile.interaction_count += 1
        profile.last_emotion = inference.primary_emotion.value
        profile.last_interaction_time = time.time()
        
        # Update emotion history
        profile.emotion_history.append({
            'emotion': inference.primary_emotion.value,
            'confidence': inference.confidence,
            'timestamp': inference.timestamp
        })
        
        # Keep history manageable
        if len(profile.emotion_history) > 100:
            profile.emotion_history = profile.emotion_history[-100:]
        
        # Update emotion frequency patterns
        emotion_str = inference.primary_emotion.value
        profile.response_patterns[emotion_str] = profile.response_patterns.get(emotion_str, 0) + 1
        
        # Update baseline mood (most frequent emotion)
        if profile.response_patterns:
            most_common = max(profile.response_patterns.items(), key=lambda x: x[1])
            profile.baseline_mood = most_common[0]
        
        # Update trust level based on interactions (simplified)
        # More interactions with positive emotions increase trust
        if inference.primary_emotion in [UserEmotionState.HAPPY, UserEmotionState.TRUSTING, 
                                         UserEmotionState.AMUSED, UserEmotionState.CURIOUS]:
            profile.trust_level = min(1.0, profile.trust_level + 0.01)
        elif inference.primary_emotion in [UserEmotionState.ANGRY, UserEmotionState.HURT,
                                           UserEmotionState.DEFENSIVE, UserEmotionState.SUSPICIOUS]:
            profile.trust_level = max(0.0, profile.trust_level - 0.02)
        
        # Save to user model if available
        if self.user_model and hasattr(self.user_model, 'update_emotional_patterns'):
            try:
                self.user_model.update_emotional_patterns(user_id, {
                    'last_emotion': profile.last_emotion,
                    'baseline_mood': profile.baseline_mood,
                    'emotional_expressiveness': profile.emotional_expressiveness,
                    'trust_level': profile.trust_level
                })
            except Exception as e:
                logger.warning("Failed to update user model: %s", e)
    
    def get_user_profile_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of user's emotional profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with profile summary
            
        Raises:
            ValueError: If user_id is invalid
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")
        
        profile = self._get_user_profile(user_id)
        
        # Calculate recent emotional trends (last 20 interactions)
        recent = profile.emotion_history[-20:] if profile.emotion_history else []
        recent_emotions = Counter()
        for entry in recent:
            recent_emotions[entry['emotion']] += 1
        
        # Calculate emotional volatility (simplified)
        if len(profile.emotion_history) >= 5:
            # Look at emotion changes
            changes = 0
            for i in range(1, len(profile.emotion_history[-10:])):
                if profile.emotion_history[i]['emotion'] != profile.emotion_history[i-1]['emotion']:
                    changes += 1
            volatility = changes / max(1, len(profile.emotion_history[-10:]) - 1)
            profile.emotional_volatility = volatility
        
        return {
            'user_id': profile.user_id,
            'baseline_mood': profile.baseline_mood,
            'emotional_expressiveness': round(profile.emotional_expressiveness, 3),
            'emotional_volatility': round(profile.emotional_volatility, 3),
            'trust_level': round(profile.trust_level, 3),
            'interaction_count': profile.interaction_count,
            'communication_style': profile.communication_style,
            'recent_emotional_trend': {k: v for k, v in recent_emotions.most_common(5)},
            'common_emotions': dict(sorted(profile.response_patterns.items(), 
                                          key=lambda x: x[1], reverse=True)[:5])
        }
    
    def reset_user_profile(self, user_id: str) -> None:
        """
        Reset a user's emotional profile to default.
        
        Args:
            user_id: User identifier
        """
        if user_id in self.user_profiles:
            del self.user_profiles[user_id]
            logger.info("Reset emotional profile for user: %s", user_id)
    
    def __repr__(self) -> str:
        return f"PerspectiveTaking(users={len(self.user_profiles)}, style={self.personality['empathy_style']})"


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Perspective Taking Test ===\n")
    
    # Create perspective taking system
    perspective = PerspectiveTaking(personality={
        'empathy_style': 'analytical',
        'emotional_intelligence': 0.8,
        'bias_toward_negative': 0.4,
        'skepticism': 0.5,
        'trust_building': 0.6
    })
    
    # Test user ID
    user_id = "test_user_1"
    
    # Test inputs with different emotional content
    test_inputs = [
        "I'm so happy today! Everything is going great!",
        "I'm really sad. I miss my friend.",
        "This is ridiculous! I can't believe this happened!",
        "I'm curious about how this works. Can you explain?",
        "That's hilarious! I laughed so hard.",
        "I don't understand what's happening...",
        "I'm a bit worried about the test tomorrow.",
        "The weather is nice.",
    ]
    
    for i, user_input in enumerate(test_inputs):
        print(f"\n--- Input {i+1}: \"{user_input}\" ---")
        
        inference = perspective.infer_user_emotion(
            user_input=user_input,
            user_id=user_id,
            context={'topic': 'general'}
        )
        
        print(f"Inferred emotion: {inference.primary_emotion.value}")
        print(f"Confidence: {inference.confidence:.2f}")
        print(f"Evidence: {inference.evidence}")
        
        # Test perspective imagining
        if i % 2 == 0:
            imagined = perspective.imagine_user_perspective(
                situation="They just received some unexpected news",
                user_id=user_id
            )
            print(f"Imagined perspective: {imagined['primary_emotion']} "
                  f"(confidence: {imagined['confidence']:.2f})")
    
    print("\n--- User Profile Summary ---")
    summary = perspective.get_user_profile_summary(user_id)
    for key, value in summary.items():
        if key not in ['recent_emotional_trend', 'common_emotions']:
            print(f"  {key}: {value}")
    
    print("\n  Recent emotional trend:")
    for emotion, count in summary['recent_emotional_trend'].items():
        print(f"    {emotion}: {count}")
    
    print("\n=== Test Complete ===")