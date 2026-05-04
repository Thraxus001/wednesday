"""
Detects emotional tone in text - happy, sad, angry, etc.
Wednesday may seem detached, but she notices every emotional undercurrent.
She just processes them differently.
"""
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import re
import logging
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

class EmotionCategory(Enum):
    """Primary emotion categories"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    
    # Complex emotions
    SARCASM = "sarcasm"
    IRONY = "irony"
    CYNICISM = "cynicism"  # Wednesday appreciates this
    RESIGNATION = "resignation"
    NOSTALGIA = "nostalgia"
    CONFUSION = "confusion"
    
    # Social emotions
    EMBARRASSMENT = "embarrassment"
    PRIDE = "pride"
    SHAME = "shame"
    GUILT = "guilt"
    
    # Wednesday specials
    DARK_HUMOR = "dark_humor"
    DEADPAN = "deadpan"
    MORBID_CURIOSITY = "morbid_curiosity"
    
    NEUTRAL = "neutral"

@dataclass
class EmotionalTone:
    """Complete emotional analysis of text"""
    # Core dimensions
    valence: float  # -1 (negative) to 1 (positive)
    arousal: float  # 0 (calm) to 1 (intense)
    dominance: float  # 0 (submissive) to 1 (dominant)
    
    # Specific emotions detected
    primary_emotion: EmotionCategory
    secondary_emotions: List[Tuple[EmotionCategory, float]] = field(default_factory=list)
    emotion_scores: Dict[EmotionCategory, float] = field(default_factory=dict)
    
    # Contextual markers
    sarcasm_score: float = 0.0  # 0-1 likelihood of sarcasm
    sincerity_score: float = 1.0  # 0-1 how genuine the emotion appears
    intensity: float = 0.5  # Overall emotional intensity
    
    # Linguistic markers
    emotional_words: List[str] = field(default_factory=list)
    intensifiers: List[str] = field(default_factory=list)  # very, extremely, etc.
    hedges: List[str] = field(default_factory=list)  # maybe, perhaps, etc.
    
    # Temporal aspects
    timestamp: datetime = field(default_factory=datetime.now)
    duration: Optional[float] = None  # Expected duration of this emotional state
    
    # Wednesday's analysis
    hidden_emotion: Optional[EmotionCategory] = None  # What user might really feel
    manipulation_attempt: bool = False  # Is user trying to manipulate?
    authenticity_score: float = 1.0  # How authentic the emotion seems
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance,
            'primary': self.primary_emotion.value,
            'secondary': [(e.value, s) for e, s in self.secondary_emotions],
            'sarcasm': self.sarcasm_score,
            'intensity': self.intensity,
            'timestamp': self.timestamp.isoformat(),
            'manipulation_attempt': self.manipulation_attempt,
            'authenticity': self.authenticity_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EmotionalTone':
        """Create EmotionalTone from dictionary"""
        return cls(
            valence=data.get('valence', 0.0),
            arousal=data.get('arousal', 0.5),
            dominance=data.get('dominance', 0.5),
            primary_emotion=EmotionCategory(data.get('primary', 'neutral')),
            secondary_emotions=[(EmotionCategory(e), s) for e, s in data.get('secondary', [])],
            sarcasm_score=data.get('sarcasm', 0.0),
            intensity=data.get('intensity', 0.5),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            manipulation_attempt=data.get('manipulation_attempt', False),
            authenticity_score=data.get('authenticity', 1.0)
        )

@dataclass
class EmotionalTrend:
    """Temporal trend in emotional states"""
    conversation_id: str
    emotions: List[EmotionalTone]
    start_time: datetime
    end_time: datetime
    trend_direction: str  # improving, worsening, stable, volatile
    volatility: float  # 0-1 how much emotions fluctuate
    dominant_pattern: Optional[EmotionCategory] = None
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'conversation_id': self.conversation_id,
            'emotion_count': len(self.emotions),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'trend_direction': self.trend_direction,
            'volatility': self.volatility,
            'dominant_pattern': self.dominant_pattern.value if self.dominant_pattern else None
        }

class SentimentAnalyzer:
    """
    Detects emotional tone in text - happy, sad, angry, etc.
    Wednesday is an expert at reading between the lines.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Emotion lexicons
        self.emotion_lexicons = self._load_emotion_lexicons()
        
        # Sarcasm detection
        self.sarcasm_detector = SarcasmDetector(config.get('sarcasm_config', {}))
        
        # ML models (if available)
        self.emotion_model = self._load_emotion_model()
        
        # Intensifiers and modifiers
        self.intensifiers = self._load_intensifiers()
        self.hedges = self._load_hedges()
        self.negations = self._load_negations()
        
        # Context tracking
        self.conversation_emotions: Dict[str, List[EmotionalTone]] = defaultdict(list)
        self.user_emotional_profiles: Dict[str, Dict] = defaultdict(dict)
        
        # Wednesday's emotional intelligence parameters
        self.wednesday_insight = {
            'sarcasm_sensitivity': 0.85,  # High sensitivity to sarcasm
            'manipulation_detection': 0.9,  # Very good at detecting manipulation
            'hidden_emotion_acuity': 0.8,  # Sees through facades
            'cynicism_bias': 0.3,  # Slight bias toward cynical interpretations
        }
        
        # Performance tracking
        self.analysis_stats = {
            'total_analyses': 0,
            'avg_sarcasm_detected': 0.0,
            'emotion_distribution': defaultdict(int),
            'avg_confidence': 0.0
        }
        
        logger.info(f"SentimentAnalyzer initialized with {len(self.emotion_lexicons)} emotion categories")
    
    def analyze(self, 
               text: Union[str, Any], 
               parsed: Optional[Any] = None,
               context: Optional[Dict] = None) -> EmotionalTone:
        """
        Analyze emotional content of text.
        
        Args:
            text: Raw text to analyze
            parsed: Optional parsed text from TextParser
            context: Optional conversation context
            
        Returns:
            EmotionalTone with complete emotional analysis
        """
        # Handle non-string input
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        if not text.strip():
            return self._create_neutral_tone()
        
        # Use parsed text if provided, otherwise do minimal parsing
        if not parsed:
            tokens = text.split()
            pos_tags = []  # Can't do POS without proper parser
        else:
            tokens = getattr(parsed, 'tokens', text.split())
            pos_tags = getattr(parsed, 'pos_tags', [])
        
        # Initialize scores
        emotion_scores = {emotion: 0.0 for emotion in EmotionCategory}
        
        # 1. Lexicon-based scoring
        emotional_words = []
        intensifiers = []
        hedges = []
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            # Check emotion lexicons
            for emotion, words in self.emotion_lexicons.items():
                if token_lower in words:
                    # Check if this is a multi-word emotion (simplified)
                    emotion_scores[emotion] += 1.0
                    emotional_words.append((token, emotion.value))
            
            # Check intensifiers
            if token_lower in self.intensifiers:
                intensifiers.append(token)
                # Boost surrounding emotion words
                emotion_scores = self._apply_intensifier(emotion_scores, token, i, tokens)
            
            # Check hedges
            if token_lower in self.hedges:
                hedges.append(token)
        
        # 2. Handle negations (reverse sentiment)
        emotion_scores = self._apply_negations(emotion_scores, tokens)
        
        # 3. Calculate core dimensions (valence, arousal, dominance)
        valence = self._calculate_valence(emotion_scores)
        arousal = self._calculate_arousal(emotion_scores, intensifiers)
        dominance = self._calculate_dominance(emotion_scores, tokens)
        
        # 4. Detect sarcasm
        sarcasm_score = self.sarcasm_detector.detect(text, parsed, context)
        
        # 5. Get primary and secondary emotions
        primary, secondary = self._get_top_emotions(emotion_scores)
        
        # 6. Calculate intensity
        intensity = self._calculate_intensity(emotion_scores, intensifiers)
        
        # 7. Check sincerity (genuineness of emotion)
        sincerity = self._calculate_sincerity(emotion_scores, sarcasm_score, context)
        
        # 8. Wednesday's deep analysis
        hidden_emotion = self._detect_hidden_emotion(emotion_scores, context)
        manipulation_attempt = self._detect_manipulation(text, emotion_scores, context)
        authenticity = self._assess_authenticity(emotion_scores, context, text)
        
        # Create result
        tone = EmotionalTone(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            primary_emotion=primary,
            secondary_emotions=secondary,
            emotion_scores={k: v for k, v in emotion_scores.items() if v > 0},
            sarcasm_score=sarcasm_score,
            sincerity_score=sincerity,
            intensity=intensity,
            emotional_words=[w[0] for w in emotional_words],
            intensifiers=intensifiers,
            hedges=hedges,
            hidden_emotion=hidden_emotion,
            manipulation_attempt=manipulation_attempt,
            authenticity_score=authenticity
        )
        
        # Store in conversation context if provided
        if context and 'conversation_id' in context:
            conv_id = context['conversation_id']
            self.conversation_emotions[conv_id].append(tone)
            
            # Trim conversation history if too long
            if len(self.conversation_emotions[conv_id]) > 100:
                self.conversation_emotions[conv_id] = self.conversation_emotions[conv_id][-100:]
            
            # Update user profile
            if 'user_id' in context:
                self._update_user_profile(context['user_id'], tone)
        
        # Update stats
        self._update_stats(tone)
        
        return tone
    
    def detect_sarcasm(self, text: str, parsed: Optional[Any] = None) -> float:
        """
        Check for ironic/sarcastic statements.
        Wednesday appreciates good sarcasm.
        """
        return self.sarcasm_detector.detect(text, parsed)
    
    def get_emotional_trend(self, 
                           conversation_id: str, 
                           window: Optional[timedelta] = None) -> Optional[EmotionalTrend]:
        """
        Track emotional arc over time in a conversation.
        """
        if conversation_id not in self.conversation_emotions:
            return None
        
        emotions = self.conversation_emotions[conversation_id]
        
        # Filter by time window if specified
        if window:
            cutoff = datetime.now() - window
            emotions = [e for e in emotions if e.timestamp > cutoff]
        
        if not emotions:
            return None
        
        # Calculate trend metrics
        valences = [e.valence for e in emotions]
        
        # Trend direction
        if len(valences) > 1:
            # Simple linear regression for slope
            x = list(range(len(valences)))
            n = len(valences)
            if n > 1:
                slope = (n * sum(x[i] * valences[i] for i in range(n)) - 
                        sum(x) * sum(valences)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
                
                if slope > 0.05:
                    trend = "improving"
                elif slope < -0.05:
                    trend = "worsening"
                elif np.std(valences) > 0.3:
                    trend = "volatile"
                else:
                    trend = "stable"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Volatility
        volatility = np.std(valences) if len(valences) > 1 else 0.0
        
        # Dominant emotion
        emotion_counts = Counter([e.primary_emotion.value for e in emotions])
        dominant_value = emotion_counts.most_common(1)[0][0] if emotion_counts else None
        dominant = EmotionCategory(dominant_value) if dominant_value else None
        
        return EmotionalTrend(
            conversation_id=conversation_id,
            emotions=emotions,
            start_time=emotions[0].timestamp,
            end_time=emotions[-1].timestamp,
            trend_direction=trend,
            volatility=min(1.0, volatility),
            dominant_pattern=dominant
        )
    
    def _load_emotion_lexicons(self) -> Dict[EmotionCategory, Set[str]]:
        """Load emotion word lexicons"""
        # This would load from files in production
        # For now, provide basic examples
        return {
            EmotionCategory.JOY: {'happy', 'joy', 'delighted', 'pleased', 'glad', 
                                  'wonderful', 'great', 'excellent', 'fantastic', 'thrilled'},
            EmotionCategory.SADNESS: {'sad', 'unhappy', 'depressed', 'gloomy', 
                                     'miserable', 'down', 'heartbroken', 'grief', 'sorrow'},
            EmotionCategory.ANGER: {'angry', 'mad', 'furious', 'irritated', 
                                   'annoyed', 'outraged', 'frustrated', 'livid', 'enraged'},
            EmotionCategory.FEAR: {'afraid', 'scared', 'terrified', 'anxious', 
                                  'worried', 'nervous', 'frightened', 'panicked', 'dread'},
            EmotionCategory.SURPRISE: {'surprised', 'shocked', 'amazed', 'astonished',
                                       'startled', 'stunned', 'astounded'},
            EmotionCategory.DISGUST: {'disgusted', 'gross', 'revolting', 'repulsed',
                                     'nauseated', 'appalled', 'distaste'},
            EmotionCategory.TRUST: {'trust', 'believe', 'confident', 'sure',
                                   'certain', 'faith', 'rely'},
            EmotionCategory.ANTICIPATION: {'anticipate', 'expect', 'await', 'look forward',
                                          'hope', 'excited'},
            EmotionCategory.SARCASM: {'sure', 'obviously', 'clearly', 'great',
                                     'wonderful', 'fantastic'},  # Often used sarcastically
            EmotionCategory.CYNICISM: {'pointless', 'meaningless', 'futile',
                                      'hopeless', 'useless', 'vain'},
            EmotionCategory.CONFUSION: {'confused', 'puzzled', 'perplexed', 'baffled',
                                       'bewildered', 'uncertain'},
            EmotionCategory.DARK_HUMOR: {'death', 'dead', 'kill', 'murder',
                                        'corpse', 'grave', 'morbid'},  # Wednesday territory
        }
    
    def _load_intensifiers(self) -> Set[str]:
        """Load words that intensify emotion"""
        return {
            'very', 'extremely', 'incredibly', 'unbelievably', 'absolutely',
            'completely', 'totally', 'utterly', 'deeply', 'profoundly',
            'terribly', 'awfully', 'so', 'such', 'really', 'quite',
            'exceptionally', 'remarkably', 'extraordinarily'
        }
    
    def _load_hedges(self) -> Set[str]:
        """Load hedging words that soften emotion"""
        return {
            'maybe', 'perhaps', 'possibly', 'probably', 'kinda', 'sorta',
            'somewhat', 'slightly', 'a bit', 'a little', 'quite', 'rather',
            'seems', 'appears', 'might', 'could', 'may', 'arguably'
        }
    
    def _load_negations(self) -> Set[str]:
        """Load negation words"""
        return {
            'not', 'no', "n't", 'never', 'none', 'nobody', 'nothing',
            'neither', 'nor', 'cannot', "can't", "don't", "doesn't",
            "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
            "wasn't", "weren't", "haven't", "hasn't", "hadn't"
        }
    
    def _load_emotion_model(self):
        """Load ML emotion detection model"""
        # Placeholder for transformer-based model
        try:
            # Would load from config path
            model_path = self.config.get('emotion_model_path')
            if model_path:
                # In production, you'd load your model here
                # model = load_model(model_path)
                logger.info(f"ML emotion model would be loaded from {model_path}")
            else:
                logger.info("No ML emotion model configured")
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
        return None
    
    def _apply_intensifier(self, 
                          emotion_scores: Dict[EmotionCategory, float], 
                          intensifier: str,
                          position: int,
                          tokens: List[str]) -> Dict[EmotionCategory, float]:
        """Boost nearby emotion scores based on intensifier"""
        # Look at next 2 tokens for emotion words
        multiplier = 1.5
        for i in range(position + 1, min(position + 3, len(tokens))):
            token = tokens[i].lower()
            for emotion, words in self.emotion_lexicons.items():
                if token in words:
                    emotion_scores[emotion] *= multiplier
        return emotion_scores
    
    def _apply_negations(self, 
                        emotion_scores: Dict[EmotionCategory, float], 
                        tokens: List[str]) -> Dict[EmotionCategory, float]:
        """Reverse sentiment for negated words"""
        negation_active = False
        negation_distance = 0
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            if token_lower in self.negations:
                negation_active = True
                negation_distance = 0
            elif negation_active and negation_distance < 3:
                # Check if this token is an emotion word
                for emotion, words in self.emotion_lexicons.items():
                    if token_lower in words:
                        # Reverse the sentiment (but keep magnitude)
                        emotion_scores[emotion] *= -1
                negation_distance += 1
            else:
                negation_active = False
                negation_distance = 0
        
        # Make all scores non-negative again (we use absolute values for magnitude)
        for emotion in emotion_scores:
            emotion_scores[emotion] = abs(emotion_scores[emotion])
        
        return emotion_scores
    
    def _calculate_valence(self, emotion_scores: Dict[EmotionCategory, float]) -> float:
        """Calculate valence (positivity/negativity)"""
        positive_emotions = {EmotionCategory.JOY, EmotionCategory.TRUST, EmotionCategory.ANTICIPATION}
        negative_emotions = {EmotionCategory.SADNESS, EmotionCategory.ANGER, 
                            EmotionCategory.FEAR, EmotionCategory.DISGUST}
        
        positive_score = sum(emotion_scores.get(e, 0) for e in positive_emotions)
        negative_score = sum(emotion_scores.get(e, 0) for e in negative_emotions)
        
        total = positive_score + negative_score
        if total == 0:
            return 0.0
        
        return (positive_score - negative_score) / total
    
    def _calculate_arousal(self, 
                          emotion_scores: Dict[EmotionCategory, float], 
                          intensifiers: List[str]) -> float:
        """Calculate arousal (emotional intensity)"""
        # Base arousal from emotion scores
        total_score = sum(abs(score) for score in emotion_scores.values())
        base_arousal = min(1.0, total_score / 15)  # Normalize (higher threshold)
        
        # Boost from intensifiers
        intensifier_boost = min(0.4, len(intensifiers) * 0.1)
        
        # High-arousal emotions get extra boost
        high_arousal = {EmotionCategory.ANGER, EmotionCategory.FEAR, EmotionCategory.SURPRISE}
        high_arousal_score = sum(emotion_scores.get(e, 0) for e in high_arousal)
        if high_arousal_score > 0:
            base_arousal *= 1.2
        
        return min(1.0, base_arousal + intensifier_boost)
    
    def _calculate_dominance(self, 
                            emotion_scores: Dict[EmotionCategory, float], 
                            tokens: List[str]) -> float:
        """Calculate dominance (control/power dimension)"""
        # Words indicating dominance
        dominant_words = {'must', 'will', 'command', 'demand', 'insist',
                         'control', 'power', 'strong', 'confident', 'force',
                         'require', 'order', 'dominate', 'win'}
        submissive_words = {'please', 'maybe', 'perhaps', 'if possible',
                           'sorry', 'apologize', 'helpless', 'weak',
                           'dependent', 'submit', 'yield', 'beg'}
        
        text = ' '.join(tokens).lower()
        
        dominant_count = sum(1 for w in dominant_words if w in text.split())
        submissive_count = sum(1 for w in submissive_words if w in text.split())
        
        if dominant_count + submissive_count == 0:
            return 0.5  # Neutral
        
        # Emotions associated with dominance
        if emotion_scores.get(EmotionCategory.ANGER, 0) > 0.5:
            return 0.7  # Anger often indicates dominance
        
        if emotion_scores.get(EmotionCategory.FEAR, 0) > 0.5:
            return 0.3  # Fear often indicates submission
        
        return dominant_count / (dominant_count + submissive_count)
    
    def _get_top_emotions(self, 
                         emotion_scores: Dict[EmotionCategory, float]
                         ) -> Tuple[EmotionCategory, List[Tuple[EmotionCategory, float]]]:
        """Get primary and secondary emotions"""
        # Filter zero scores and negative (shouldn't happen after abs)
        active = [(e, s) for e, s in emotion_scores.items() if s > 0.1]
        
        if not active:
            return EmotionCategory.NEUTRAL, []
        
        # Sort by score
        active.sort(key=lambda x: x[1], reverse=True)
        
        primary = active[0][0]
        secondary = active[1:4]  # Top 3 secondary
        
        return primary, secondary
    
    def _calculate_intensity(self, 
                           emotion_scores: Dict[EmotionCategory, float], 
                           intensifiers: List[str]) -> float:
        """Calculate overall emotional intensity"""
        # Average of non-zero scores
        non_zero = [s for s in emotion_scores.values() if s > 0]
        if not non_zero:
            return 0.0
        
        base_intensity = sum(non_zero) / len(non_zero)
        
        # Normalize and apply intensifier boost
        normalized = min(1.0, base_intensity / 3)  # Assume strong emotion score of 3
        intensifier_boost = min(0.3, len(intensifiers) * 0.1)
        
        return min(1.0, normalized + intensifier_boost)
    
    def _calculate_sincerity(self, 
                           emotion_scores: Dict[EmotionCategory, float], 
                           sarcasm_score: float,
                           context: Optional[Dict]) -> float:
        """Calculate how sincere the emotion appears"""
        # Start with assumption of sincerity
        sincerity = 1.0
        
        # Reduce for sarcasm
        sincerity -= sarcasm_score * 0.5
        
        # Check for contradictions in emotion scores
        contradictory_pairs = [
            (EmotionCategory.JOY, EmotionCategory.SADNESS),
            (EmotionCategory.ANGER, EmotionCategory.TRUST),
            (EmotionCategory.FEAR, EmotionCategory.ANTICIPATION)
        ]
        
        for e1, e2 in contradictory_pairs:
            if emotion_scores.get(e1, 0) > 0.5 and emotion_scores.get(e2, 0) > 0.5:
                sincerity -= 0.2  # Mixed emotions can be less sincere
        
        # Check context for previous insincerity
        if context and context.get('user_id'):
            user_profile = self.user_emotional_profiles.get(context['user_id'], {})
            if user_profile.get('sarcasm_tendency', 0) > 0.7:
                sincerity *= 0.8  # User is often sarcastic
        
        return max(0.0, min(1.0, sincerity))
    
    def _detect_hidden_emotion(self, 
                              emotion_scores: Dict[EmotionCategory, float], 
                              context: Optional[Dict]) -> Optional[EmotionCategory]:
        """
        Detect what emotion user might really be feeling beneath the surface.
        Wednesday is good at this.
        """
        # Look for common masking patterns
        masking_patterns = [
            # Anger masking fear
            (EmotionCategory.ANGER, EmotionCategory.FEAR, 0.7),
            # Joy masking sadness
            (EmotionCategory.JOY, EmotionCategory.SADNESS, 0.6),
            # Sarcasm masking genuine emotion
            (EmotionCategory.SARCASM, EmotionCategory.ANGER, 0.8),
            (EmotionCategory.SARCASM, EmotionCategory.SADNESS, 0.8),
            # Cynicism masking hurt
            (EmotionCategory.CYNICISM, EmotionCategory.SADNESS, 0.7),
        ]
        
        for surface, hidden, threshold in masking_patterns:
            if emotion_scores.get(surface, 0) > threshold:
                # Check context for clues
                if context and self._has_context_clues(hidden, context):
                    return hidden
        
        return None
    
    def _detect_manipulation(self, 
                           text: str, 
                           emotion_scores: Dict[EmotionCategory, float], 
                           context: Optional[Dict]) -> bool:
        """
        Detect if user is trying to manipulate Wednesday emotionally.
        She's immune to most manipulation.
        """
        manipulation_indicators = [
            # Excessive flattery
            (r'you\'?re (so|the most) (smart|brilliant|amazing|genius|wonderful)', 0.7),
            # Guilt tripping
            (r'if you (really|cared|understood|knew)', 0.8),
            # Emotional blackmail
            (r'after (everything|all) I\'?ve done', 0.8),
            # Playing victim
            (r'poor me|woe is me|nobody understands|everyone hates me', 0.7),
            # Exaggerated emotion
            (r'I\'?ve never been so (hurt|upset|angry|happy)', 0.6),
            # Threatening withdrawal
            (r'if you don\'?t,? I\'?ll (never|not)', 0.9),
        ]
        
        for pattern, threshold in manipulation_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                # Check if emotion scores align with manipulation
                # Manipulators often use extreme emotions
                if max(emotion_scores.values(), default=0) > 2.0:
                    return True
        
        # Check for pattern of manipulation in context
        if context and context.get('user_id'):
            user_profile = self.user_emotional_profiles.get(context['user_id'], {})
            if user_profile.get('manipulation_attempts', 0) > 3:
                # User has history of manipulation
                if emotion_scores.get(EmotionCategory.SADNESS, 0) > 1.0:
                    return True
        
        return False
    
    def _assess_authenticity(self, 
                            emotion_scores: Dict[EmotionCategory, float], 
                            context: Optional[Dict],
                            text: str) -> float:
        """Assess how authentic the emotional expression seems"""
        authenticity = 1.0
        
        # Check for emotional consistency with context
        if context and 'user_id' in context:
            user_profile = self.user_emotional_profiles.get(context['user_id'], {})
            if user_profile:
                # Compare with user's typical emotional range
                typical_emotions = user_profile.get('typical_emotions', {})
                for emotion, score in emotion_scores.items():
                    if emotion in typical_emotions:
                        typical = typical_emotions[emotion]
                        if score > typical * 2:  # More than double typical
                            authenticity -= 0.2
        
        # Sarcasm reduces authenticity
        if emotion_scores.get(EmotionCategory.SARCASM, 0) > 0.5:
            authenticity -= 0.3
        
        # Check for contradictory emotional signals
        if '!' in text and '?' in text:  # Mixed punctuation
            authenticity -= 0.1
        if '...' in text:  # Ellipsis can indicate hesitation
            authenticity -= 0.1
        
        # Check for clichés (often less authentic)
        cliches = ['heart of gold', 'crying my eyes out', 'over the moon',
                  'sick and tired', 'scared to death']
        for cliche in cliches:
            if cliche in text.lower():
                authenticity -= 0.2
        
        return max(0.0, min(1.0, authenticity))
    
    def _has_context_clues(self, emotion: EmotionCategory, context: Dict) -> bool:
        """Check if context supports hidden emotion hypothesis"""
        # Would check conversation history for clues
        # Simplified version
        if 'conversation_history' in context:
            # Look for recent mentions of situations that might trigger this emotion
            return True
        return False
    
    def _create_neutral_tone(self) -> EmotionalTone:
        """Create a neutral emotional tone for empty input"""
        return EmotionalTone(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            primary_emotion=EmotionCategory.NEUTRAL,
            emotion_scores={},
            sincerity_score=1.0,
            intensity=0.0
        )
    
    def _update_user_profile(self, user_id: str, tone: EmotionalTone):
        """Update user's emotional profile"""
        profile = self.user_emotional_profiles.get(user_id, {
            'typical_emotions': defaultdict(float),
            'emotional_range': {'min_valence': 0.0, 'max_valence': 0.0},
            'sarcasm_tendency': 0.0,
            'manipulation_attempts': 0,
            'interaction_count': 0
        })
        
        # Update typical emotions (moving average)
        for emotion, score in tone.emotion_scores.items():
            current = profile['typical_emotions'][emotion.value]
            profile['typical_emotions'][emotion.value] = (
                current * 0.9 + score * 0.1
            )
        
        # Update valence range
        profile['emotional_range']['min_valence'] = min(
            profile['emotional_range']['min_valence'], tone.valence
        )
        profile['emotional_range']['max_valence'] = max(
            profile['emotional_range']['max_valence'], tone.valence
        )
        
        # Update sarcasm tendency
        profile['sarcasm_tendency'] = (
            profile['sarcasm_tendency'] * 0.9 + tone.sarcasm_score * 0.1
        )
        
        # Update manipulation attempts
        if tone.manipulation_attempt:
            profile['manipulation_attempts'] += 1
        
        profile['interaction_count'] += 1
        
        self.user_emotional_profiles[user_id] = profile
    
    def _update_stats(self, tone: EmotionalTone):
        """Update analysis statistics"""
        self.analysis_stats['total_analyses'] += 1
        self.analysis_stats['avg_sarcasm_detected'] = (
            self.analysis_stats['avg_sarcasm_detected'] * 0.99 + 
            tone.sarcasm_score * 0.01
        )
        self.analysis_stats['emotion_distribution'][tone.primary_emotion.value] += 1
        
        # Update average confidence (using intensity as proxy)
        total = self.analysis_stats['total_analyses']
        old_avg = self.analysis_stats['avg_confidence']
        self.analysis_stats['avg_confidence'] = old_avg + (tone.intensity - old_avg) / total
    
    def get_user_emotional_profile(self, user_id: str) -> Optional[Dict]:
        """Get emotional profile for a user"""
        return self.user_emotional_profiles.get(user_id)
    
    def get_stats(self) -> Dict:
        """Return analysis statistics"""
        stats = dict(self.analysis_stats)
        stats['emotion_distribution'] = dict(stats['emotion_distribution'])
        return stats
    
    def reset_stats(self) -> None:
        """Reset analysis statistics"""
        self.analysis_stats = {
            'total_analyses': 0,
            'avg_sarcasm_detected': 0.0,
            'emotion_distribution': defaultdict(int),
            'avg_confidence': 0.0
        }


class SarcasmDetector:
    """
    Specialized sarcasm detection for Wednesday.
    She has a PhD in recognizing sarcasm.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Sarcasm patterns
        self.patterns = self._load_sarcasm_patterns()
        
        # Contradiction detection
        self.contradiction_words = {
            'but', 'however', 'although', 'yet', 'though'
        }
        
        logger.info("SarcasmDetector initialized")
    
    def detect(self, 
              text: str, 
              parsed: Optional[Any] = None,
              context: Optional[Dict] = None) -> float:
        """
        Detect likelihood of sarcasm in text.
        
        Returns score from 0 (no sarcasm) to 1 (definitely sarcastic)
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        score = 0.0
        
        # 1. Pattern matching
        for pattern, weight in self.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += weight
        
        # 2. Check for positive words in negative context
        if parsed and hasattr(parsed, 'tokens'):
            score += self._check_positive_negative_contradiction(
                parsed.tokens, getattr(parsed, 'pos_tags', None)
            )
        
        # 3. Check for exaggeration markers
        exaggeration_score = self._detect_exaggeration(text)
        score += exaggeration_score
        
        # 4. Context-based (if previous turns were sarcastic)
        if context and 'previous_sarcasm' in context:
            if context['previous_sarcasm'] > 0.5:
                score += 0.2  # Sarcasm often continues
        
        # 5. Punctuation and formatting
        if '!' in text and '?' in text:  # Interrobang effect
            score += 0.15
        if text.endswith('...') or '...' in text:  # Ellipsis
            score += 0.1
        if '"' in text:  # Scare quotes
            score += 0.15
        if text.isupper():  # ALL CAPS (often sarcastic emphasis)
            score += 0.1
        
        # Normalize to 0-1
        return min(1.0, score)
    
    def _load_sarcasm_patterns(self) -> List[Tuple[str, float]]:
        """Load regex patterns for sarcasm detection"""
        return [
            (r'sure,? sure', 0.6),
            (r'oh (great|wonderful|fantastic|brilliant)', 0.7),
            (r'yeah,? right', 0.7),
            (r'as if', 0.6),
            (r'obviously', 0.4),  # Can be sincere or sarcastic
            (r'clearly', 0.4),
            (r'literally', 0.5),  # Often used for exaggeration
            (r'whatever you say', 0.8),
            (r'if you say so', 0.7),
            (r'that\'s (just )?great', 0.6),
            (r'just what I needed', 0.7),
            (r'big (deal|whoop)', 0.8),
            (r'like I care', 0.9),
            (r'as if that matters', 0.8),
            (r'well,? well,? well', 0.5),
            (r'how (nice|lovely|wonderful) for you', 0.7),
            (r'I\'?m (so )?(happy|thrilled|delighted) to hear', 0.6),
        ]
    
    def _check_positive_negative_contradiction(self, 
                                              tokens: List[str], 
                                              pos_tags: Optional[List]) -> float:
        """
        Check for positive words used in negative context.
        E.g., "Great, another problem."
        """
        positive_words = {'great', 'wonderful', 'fantastic', 'excellent', 
                         'perfect', 'lovely', 'amazing', 'brilliant'}
        negative_context_words = {'problem', 'issue', 'trouble', 'wrong',
                                 'broken', 'failed', 'mistake', 'error',
                                 'terrible', 'awful', 'horrible'}
        
        text = ' '.join(tokens).lower()
        
        has_positive = any(word in text for word in positive_words)
        has_negative = any(word in text for word in negative_context_words)
        
        if has_positive and has_negative:
            return 0.6
        
        return 0.0
    
    def _detect_exaggeration(self, text: str) -> float:
        """Detect exaggerated language that might indicate sarcasm"""
        exaggeration_patterns = [
            (r'the (best|worst|greatest|most) (ever|in history|of all time)', 0.5),
            (r'every single (person|time|day|thing)', 0.4),
            (r'literally (every|all|no|nothing|nobody)', 0.5),
            (r'absolutely (nothing|everything|everyone|no one)', 0.4),
            (r'totally (not|didn\'t|never|won\'t)', 0.4),
            (r'!{3,}', 0.3),  # Multiple exclamation marks
            (r'\bnever\s+ever\b', 0.3),
            (r'\balways\b.*\bnever\b', 0.4),  # Contradictory absolutes
            (r'(million|billion|trillion) (times|percent|dollars)', 0.3),
        ]
        
        score = 0.0
        for pattern, weight in exaggeration_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += weight
        
        return min(0.8, score)  # Cap exaggeration contribution

# Connects to: parser.py (gets tokens and linguistic structure)
# Connects to: memory/working/ (stores emotional context for conversations)
# Connects to: emotion/empathy.py (provides emotional input for empathetic responses)
# Connects to: emotion/appraisal.py (emotional context influences appraisal)
# Connects to: language/generation/style_adapter.py (emotional tone influences response style)
# Connects to: self/theory_of_mind.py (understanding others' emotional states)
# Connects to: perception/attention/salience.py (emotional content influences salience)