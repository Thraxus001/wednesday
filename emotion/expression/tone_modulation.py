"""
tone_modulation.py - Emotional tone modulation for Wednesday AI

This module applies fine-grained emotional coloring to Wednesday's output,
adjusting textual parameters to reflect her emotional state. It handles
the subtle cues that make text feel authentic: punctuation, emphasis,
pacing, and word choice.

Key improvements:
- Fixed missing time import
- Added comprehensive validation and error handling
- Improved punctuation modulation with context awareness
- Enhanced emphasis application with proper escaping
- Added configuration validation
- Made random operations reproducible with seed support
"""

import re
import time
import logging
import random
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class PunctuationStyle(Enum):
    """Styles of punctuation usage"""
    NEUTRAL = "neutral"           # Standard punctuation
    EMPHATIC = "emphatic"          # More exclamations, emphasis
    HESITANT = "hesitant"          # More ellipses, pauses
    PRECISE = "precise"            # Proper, formal punctuation
    MINIMAL = "minimal"             # Minimal punctuation, run-on
    DRY = "dry"                      # Period-focused, understated
    DRAMATIC = "dramatic"            # Theatrical punctuation


class EmphasisPattern(Enum):
    """Patterns for emphasizing text"""
    NONE = "none"
    ITALICS = "italics"           # *around words*
    CAPS = "caps"                  # ALL CAPS for emphasis
    BOLD = "bold"                   # **around words**
    QUOTES = "quotes"               # "around words"
    DASHES = "dashes"               # - around words -


@dataclass
class ToneParameters:
    """
    Parameters controlling tonal modulation of text.
    
    These are granular controls for text production mechanics.
    All float values should be between 0 and 1 unless otherwise noted.
    """
    # Text parameters
    punctuation_style: PunctuationStyle = PunctuationStyle.NEUTRAL
    exclamation_frequency: float = 0.1  # 0-1 how often to use !
    ellipsis_frequency: float = 0.1      # 0-1 how often to use ...
    question_frequency: float = 0.2       # 0-1 for rhetorical questions
    
    # Emphasis parameters
    emphasis_pattern: EmphasisPattern = EmphasisPattern.NONE
    emphasis_frequency: float = 0.1       # 0-1 how often to emphasize
    emphasis_words: List[str] = field(default_factory=list)  # Words to emphasize
    
    # Voice parameters (for TTS)
    speech_rate: float = 1.0              # Multiplier for speaking speed (0.5-2.0)
    pitch_mean: float = 1.0                # Base pitch (0.5-2.0)
    pitch_variation: float = 0.1            # How much pitch varies (0-1)
    volume: float = 1.0                     # Relative volume (0-2)
    breathiness: float = 0.0                 # Breathy quality (0-1)
    
    # Word choice parameters
    word_complexity: float = 0.5            # 0-1 simple to complex
    contraction_ratio: float = 0.7           # 0-1 formal to contractions
    repetition_tendency: float = 0.1         # 0-1 how likely to repeat
    
    # Timing parameters
    pause_frequency: float = 0.1             # How often to insert pauses
    pause_duration: float = 0.3               # Base pause duration in seconds
    
    def __post_init__(self):
        """Validate parameters"""
        # Validate enum types
        if not isinstance(self.punctuation_style, PunctuationStyle):
            raise TypeError(f"punctuation_style must be PunctuationStyle, got {type(self.punctuation_style)}")
        if not isinstance(self.emphasis_pattern, EmphasisPattern):
            raise TypeError(f"emphasis_pattern must be EmphasisPattern, got {type(self.emphasis_pattern)}")
        
        # Validate float ranges
        self._validate_float(self.exclamation_frequency, 0, 1, "exclamation_frequency")
        self._validate_float(self.ellipsis_frequency, 0, 1, "ellipsis_frequency")
        self._validate_float(self.question_frequency, 0, 1, "question_frequency")
        self._validate_float(self.emphasis_frequency, 0, 1, "emphasis_frequency")
        self._validate_float(self.speech_rate, 0.5, 2.0, "speech_rate")
        self._validate_float(self.pitch_mean, 0.5, 2.0, "pitch_mean")
        self._validate_float(self.pitch_variation, 0, 1, "pitch_variation")
        self._validate_float(self.volume, 0, 2, "volume")
        self._validate_float(self.breathiness, 0, 1, "breathiness")
        self._validate_float(self.word_complexity, 0, 1, "word_complexity")
        self._validate_float(self.contraction_ratio, 0, 1, "contraction_ratio")
        self._validate_float(self.repetition_tendency, 0, 1, "repetition_tendency")
        self._validate_float(self.pause_frequency, 0, 1, "pause_frequency")
        self._validate_float(self.pause_duration, 0, 10, "pause_duration")
    
    def _validate_float(self, value: float, min_val: float, max_val: float, name: str):
        """Validate float is within range"""
        if not min_val <= value <= max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'punctuation_style': self.punctuation_style.value,
            'exclamation_frequency': round(self.exclamation_frequency, 3),
            'ellipsis_frequency': round(self.ellipsis_frequency, 3),
            'question_frequency': round(self.question_frequency, 3),
            'emphasis_pattern': self.emphasis_pattern.value,
            'emphasis_frequency': round(self.emphasis_frequency, 3),
            'speech_rate': round(self.speech_rate, 3),
            'pitch_mean': round(self.pitch_mean, 3),
            'pitch_variation': round(self.pitch_variation, 3),
            'volume': round(self.volume, 3),
            'breathiness': round(self.breathiness, 3)
        }


class ToneModulator:
    """
    Applies emotional tone modulation to text and voice parameters.
    
    This module handles the micro-adjustments that make Wednesday's
    speech feel emotionally authentic: when to use an exclamation,
    when to pause, which words to emphasize, and how her voice should sound.
    
    The modulator is stateless between modulations but maintains a
    history for consistency tracking.
    """
    
    # Emotion to punctuation style mapping
    EMOTION_PUNCTUATION = {
        'joy': PunctuationStyle.EMPHATIC,
        'sadness': PunctuationStyle.HESITANT,
        'anger': PunctuationStyle.DRY,
        'fear': PunctuationStyle.HESITANT,
        'surprise': PunctuationStyle.DRAMATIC,
        'trust': PunctuationStyle.NEUTRAL,
        'disgust': PunctuationStyle.DRY,
        'anticipation': PunctuationStyle.PRECISE,
        
        # Wednesday-specific
        'dark_amusement': PunctuationStyle.DRY,
        'protective': PunctuationStyle.PRECISE,
        'nostalgic': PunctuationStyle.HESITANT,
        'satisfied': PunctuationStyle.MINIMAL,
        'curiously_detached': PunctuationStyle.PRECISE,
        'wary': PunctuationStyle.MINIMAL,
        'pensive': PunctuationStyle.HESITANT,
        'disdainful': PunctuationStyle.DRY,
        'neutral': PunctuationStyle.NEUTRAL
    }
    
    # Emotion to emphasis pattern mapping
    EMOTION_EMPHASIS = {
        'joy': EmphasisPattern.ITALICS,
        'anger': EmphasisPattern.CAPS,
        'fear': EmphasisPattern.NONE,
        'surprise': EmphasisPattern.CAPS,
        'dark_amusement': EmphasisPattern.ITALICS,
        'protective': EmphasisPattern.BOLD,
        'satisfied': EmphasisPattern.NONE,
        'disdainful': EmphasisPattern.ITALICS,
        'pensive': EmphasisPattern.NONE,
        'neutral': EmphasisPattern.NONE
    }
    
    # Words commonly emphasized for each emotion
    EMPHASIS_WORDS = {
        'anger': ['never', 'always', 'completely', 'utterly', 'absolutely', 'entirely'],
        'joy': ['wonderful', 'perfect', 'excellent', 'brilliant', 'fantastic'],
        'dark_amusement': ['delicious', 'perfect', 'exquisite', 'ironic', 'fascinating', 'amusing'],
        'protective': ['mine', 'enough', 'stop', 'no', 'leave', 'don\'t'],
        'disdainful': ['obviously', 'clearly', 'naturally', 'indeed', 'quite', 'certainly'],
        'surprise': ['what', 'how', 'unbelievable', 'incredible', 'amazing'],
    }
    
    # Punctuation frequency baselines by emotion
    PUNCTUATION_FREQUENCIES = {
        'joy': {'exclamation': 0.4, 'ellipsis': 0.05, 'question': 0.2},
        'sadness': {'exclamation': 0.0, 'ellipsis': 0.4, 'question': 0.1},
        'anger': {'exclamation': 0.2, 'ellipsis': 0.1, 'question': 0.1},
        'fear': {'exclamation': 0.1, 'ellipsis': 0.3, 'question': 0.2},
        'surprise': {'exclamation': 0.5, 'ellipsis': 0.2, 'question': 0.3},
        'dark_amusement': {'exclamation': 0.1, 'ellipsis': 0.2, 'question': 0.15},
        'protective': {'exclamation': 0.2, 'ellipsis': 0.05, 'question': 0.1},
        'nostalgic': {'exclamation': 0.0, 'ellipsis': 0.4, 'question': 0.2},
        'satisfied': {'exclamation': 0.0, 'ellipsis': 0.1, 'question': 0.05},
        'pensive': {'exclamation': 0.0, 'ellipsis': 0.3, 'question': 0.3},
        'disdainful': {'exclamation': 0.0, 'ellipsis': 0.15, 'question': 0.1},
        'neutral': {'exclamation': 0.05, 'ellipsis': 0.1, 'question': 0.15}
    }
    
    # Voice parameter baselines by emotion
    VOICE_PARAMETERS = {
        'joy': {'rate': 1.1, 'pitch': 1.1, 'variation': 0.2, 'volume': 1.1},
        'sadness': {'rate': 0.8, 'pitch': 0.9, 'variation': 0.05, 'volume': 0.8},
        'anger': {'rate': 1.0, 'pitch': 1.0, 'variation': 0.1, 'volume': 1.2},
        'fear': {'rate': 1.2, 'pitch': 1.2, 'variation': 0.25, 'volume': 0.7},
        'surprise': {'rate': 1.2, 'pitch': 1.3, 'variation': 0.3, 'volume': 1.2},
        'dark_amusement': {'rate': 0.9, 'pitch': 0.95, 'variation': 0.15, 'volume': 0.9},
        'protective': {'rate': 0.9, 'pitch': 0.95, 'variation': 0.1, 'volume': 1.1},
        'nostalgic': {'rate': 0.8, 'pitch': 0.9, 'variation': 0.1, 'volume': 0.8},
        'satisfied': {'rate': 0.9, 'pitch': 1.0, 'variation': 0.05, 'volume': 0.9},
        'pensive': {'rate': 0.8, 'pitch': 0.95, 'variation': 0.1, 'volume': 0.8},
        'disdainful': {'rate': 0.9, 'pitch': 0.95, 'variation': 0.05, 'volume': 0.9},
        'neutral': {'rate': 1.0, 'pitch': 1.0, 'variation': 0.1, 'volume': 1.0}
    }
    
    def __init__(self, personality: Optional[Dict[str, float]] = None, random_seed: Optional[int] = None):
        """
        Initialize the tone modulator.
        
        Args:
            personality: Optional personality parameters
            random_seed: Optional seed for reproducible random operations
            
        Raises:
            ValueError: If personality parameters are invalid
        """
        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
        
        # Personality influences on tone
        default_personality = {
            'vocal_expressiveness': 0.4,      # How much voice varies (0-1)
            'punctuation_tendency': 0.3,        # How much punctuation to use (0-1)
            'emphasis_tendency': 0.3,            # How often to emphasize (0-1)
            'dryness': 0.8,                       # Preference for dry delivery (0-1)
            'formality': 0.5,                     # Formal vs casual speech (0-1)
        }
        
        self.personality = default_personality.copy()
        if personality:
            self._validate_personality(personality)
            self.personality.update(personality)
        
        # Current tone parameters
        self.current_params = ToneParameters()
        
        # Track last few modulations for consistency
        self.recent_modulations: List[Dict[str, Any]] = []
        self.max_history = 20
        
        logger.info("ToneModulator initialized")
    
    def _validate_personality(self, personality: Dict[str, float]) -> None:
        """Validate personality parameters"""
        for key, value in personality.items():
            if key not in self.personality:
                raise ValueError(f"Unknown personality parameter: {key}")
            if not 0 <= value <= 1:
                raise ValueError(f"Personality parameter {key} must be between 0 and 1, got {value}")
    
    def apply_to_text(self, 
                      text: str, 
                      emotional_state: Dict[str, Any],
                      expression_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply emotional tone modulation to text.
        
        Args:
            text: Original text to modulate
            emotional_state: Current emotional state (must contain 'dominant' and 'emotions')
            expression_params: Optional expression parameters from EmotionalResponse
            
        Returns:
            Modulated text with emotional coloring
            
        Raises:
            ValueError: If emotional_state is missing required fields
        """
        if not text:
            return text
        
        # Validate emotional state
        if 'dominant' not in emotional_state:
            raise ValueError("emotional_state must contain 'dominant' emotion")
        if 'emotions' not in emotional_state:
            raise ValueError("emotional_state must contain 'emotions' dictionary")
        
        # Get dominant emotion
        dominant = emotional_state.get('dominant', 'neutral')
        intensities = emotional_state.get('emotions', {})
        intensity = intensities.get(dominant, 0.5)
        
        # Clamp intensity
        intensity = max(0.0, min(1.0, intensity))
        
        # Get tone parameters for this emotion
        params = self._generate_tone_parameters(dominant, intensity, expression_params)
        self.current_params = params
        
        # Apply modulations
        modulated = text
        
        # 1. Adjust punctuation
        modulated = self._modulate_punctuation(modulated, dominant, intensity, params)
        
        # 2. Add emphasis
        modulated = self._add_emphasis(modulated, dominant, intensity, params)
        
        # 3. Adjust word choice (simplified)
        modulated = self._modulate_word_choice(modulated, dominant, intensity, params)
        
        # 4. Add pauses/hesitations
        modulated = self._add_pauses(modulated, dominant, intensity, params)
        
        # Record modulation
        self._record_modulation(dominant, intensity, params)
        
        logger.debug(f"Applied tone modulation to text: {dominant} (intensity={intensity:.2f})")
        
        return modulated
    
    def get_voice_params(self, 
                         emotional_state: Optional[Dict[str, Any]] = None,
                         expression_params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Get voice synthesis parameters based on emotional state.
        
        Args:
            emotional_state: Current emotional state
            expression_params: Optional expression parameters
            
        Returns:
            Dictionary of voice parameters for TTS
            
        Raises:
            ValueError: If emotional_state is provided but invalid
        """
        if not emotional_state:
            # Return current parameters if no state provided
            return self._voice_params_to_dict(self.current_params)
        
        # Validate emotional state if provided
        if 'dominant' not in emotional_state:
            raise ValueError("emotional_state must contain 'dominant' emotion")
        
        dominant = emotional_state.get('dominant', 'neutral')
        intensities = emotional_state.get('emotions', {})
        intensity = intensities.get(dominant, 0.5)
        intensity = max(0.0, min(1.0, intensity))
        
        # Get base voice parameters for this emotion
        voice_params = self.VOICE_PARAMETERS.get(dominant, self.VOICE_PARAMETERS['neutral']).copy()
        
        # Scale by intensity
        neutral_params = self.VOICE_PARAMETERS['neutral']
        for param in ['rate', 'pitch', 'variation', 'volume']:
            if param in voice_params:
                # Interpolate between neutral and target based on intensity
                neutral_val = neutral_params.get(param, 1.0)
                target_val = voice_params.get(param, neutral_val)
                voice_params[param] = neutral_val + (target_val - neutral_val) * intensity
        
        # Apply personality modifiers
        voice_params['rate'] *= (1 + (self.personality['vocal_expressiveness'] - 0.5) * 0.2)
        voice_params['variation'] *= self.personality['vocal_expressiveness']
        
        # Wednesday-specific: generally more controlled
        voice_params['variation'] *= 0.8
        voice_params['pitch'] = 1.0 + (voice_params['pitch'] - 1.0) * 0.7
        
        # Add breathiness for certain emotions
        if dominant in ['sadness', 'nostalgic', 'pensive']:
            voice_params['breathiness'] = 0.2 * intensity
        else:
            voice_params['breathiness'] = 0.05
        
        return self._voice_params_to_dict_from_raw(voice_params)
    
    def _generate_tone_parameters(self, 
                                   emotion: str, 
                                   intensity: float,
                                   expression_params: Optional[Dict]) -> ToneParameters:
        """Generate tone parameters based on emotion and intensity"""
        params = ToneParameters()
        
        # Set punctuation style
        params.punctuation_style = self.EMOTION_PUNCTUATION.get(
            emotion, PunctuationStyle.NEUTRAL
        )
        
        # Set punctuation frequencies
        freq = self.PUNCTUATION_FREQUENCIES.get(
            emotion, self.PUNCTUATION_FREQUENCIES['neutral']
        )
        params.exclamation_frequency = freq['exclamation'] * intensity
        params.ellipsis_frequency = freq['ellipsis'] * intensity
        params.question_frequency = freq['question'] * intensity
        
        # Apply dryness personality (reduces exclamation, increases ellipsis)
        if self.personality['dryness'] > 0.6:
            params.exclamation_frequency *= (1 - self.personality['dryness'])
            params.ellipsis_frequency *= (1 + self.personality['dryness'] * 0.2)
        
        # Apply punctuation tendency
        params.exclamation_frequency *= self.personality['punctuation_tendency']
        params.ellipsis_frequency *= self.personality['punctuation_tendency']
        params.question_frequency *= self.personality['punctuation_tendency']
        
        # Clamp frequencies
        params.exclamation_frequency = min(0.8, params.exclamation_frequency)
        params.ellipsis_frequency = min(0.8, params.ellipsis_frequency)
        params.question_frequency = min(0.8, params.question_frequency)
        
        # Set emphasis pattern
        params.emphasis_pattern = self.EMOTION_EMPHASIS.get(
            emotion, EmphasisPattern.NONE
        )
        
        # Set emphasis frequency
        if params.emphasis_pattern != EmphasisPattern.NONE:
            params.emphasis_frequency = 0.2 * intensity * self.personality['emphasis_tendency']
        
        # Set emphasis words
        words = self.EMPHASIS_WORDS.get(emotion, [])
        if words and intensity > 0.5:
            num_words = max(1, min(len(words), int(len(words) * intensity * 0.5)))
            params.emphasis_words = words[:num_words]
        
        # Set voice parameters
        voice = self.VOICE_PARAMETERS.get(emotion, self.VOICE_PARAMETERS['neutral'])
        params.speech_rate = voice['rate']
        params.pitch_mean = voice['pitch']
        params.pitch_variation = voice['variation'] * intensity
        params.volume = voice['volume']
        
        # Apply expression parameters if provided
        if expression_params:
            if 'speech_rate' in expression_params:
                params.speech_rate = float(expression_params['speech_rate'])
            if 'intensity' in expression_params:
                params.emphasis_frequency *= float(expression_params['intensity'])
        
        return params
    
    def _modulate_punctuation(self, 
                              text: str, 
                              emotion: str, 
                              intensity: float,
                              params: ToneParameters) -> str:
        """Modulate punctuation in text based on emotional parameters"""
        # Split into sentences more carefully
        # This regex handles common punctuation patterns
        sentence_pattern = r'([^.!?…]+[.!?…]+)'
        sentences = re.findall(sentence_pattern, text)
        
        if not sentences:
            # If no sentences found, treat the whole text as one
            sentences = [text]
        
        modulated_sentences = []
        
        for sentence in sentences:
            # Remove trailing punctuation for processing
            match = re.search(r'(.*?)([.!?…]+)$', sentence.strip())
            if match:
                content = match.group(1).strip()
                current_punct = match.group(2)
            else:
                content = sentence.strip()
                current_punct = '.'
            
            # Determine new punctuation
            new_punct = self._choose_punctuation(emotion, intensity, params)
            
            # Build modulated sentence
            modulated_sentences.append(f"{content}{new_punct}")
        
        # Add ellipses in the middle of sentences for hesitation
        if params.ellipsis_frequency > 0.1 and intensity > 0.4:
            result = ' '.join(modulated_sentences)
            words = result.split()
            if len(words) > 5:
                # Insert ellipsis at a random position in the middle
                if random.random() < params.ellipsis_frequency * 0.3:
                    insert_pos = random.randint(max(1, len(words) // 4), 
                                               min(len(words) - 2, len(words) * 3 // 4))
                    words.insert(insert_pos, '...')
                    result = ' '.join(words)
            return result
        
        return ' '.join(modulated_sentences)
    
    def _choose_punctuation(self, 
                            emotion: str, 
                            intensity: float,
                            params: ToneParameters) -> str:
        """Choose appropriate sentence-ending punctuation"""
        # Adjust probabilities based on emotion
        if emotion == 'neutral':
            return '.'  # Neutral defaults to period
        
        r = random.random()
        
        # Normalize frequencies to sum <= 1
        total = params.exclamation_frequency + params.ellipsis_frequency + params.question_frequency
        if total <= 0:
            return '.'
        
        # Scale probabilities
        exclamation_prob = params.exclamation_frequency / total if total > 0 else 0
        ellipsis_prob = params.ellipsis_frequency / total if total > 0 else 0
        question_prob = params.question_frequency / total if total > 0 else 0
        
        if r < exclamation_prob:
            return '!'
        elif r < exclamation_prob + ellipsis_prob:
            return '...'
        elif r < exclamation_prob + ellipsis_prob + question_prob:
            return '?'
        else:
            return '.'
    
    def _add_emphasis(self, 
                      text: str, 
                      emotion: str, 
                      intensity: float,
                      params: ToneParameters) -> str:
        """Add emphasis to text based on emotional parameters"""
        if params.emphasis_pattern == EmphasisPattern.NONE or params.emphasis_frequency <= 0:
            return text
        
        words = text.split()
        if len(words) < 3:
            return text
        
        modulated_words = []
        emphasis_patterns = {
            EmphasisPattern.ITALICS: ('*', '*'),
            EmphasisPattern.CAPS: ('', ''),  # Handled by uppercasing
            EmphasisPattern.BOLD: ('**', '**'),
            EmphasisPattern.QUOTES: ('"', '"'),
            EmphasisPattern.DASHES: (' -', '- '),
        }
        
        emphasis_set = {w.lower() for w in params.emphasis_words}
        
        for word in words:
            # Check if this word should be emphasized
            should_emphasize = False
            
            # Skip if word is already punctuation or empty
            if not word or all(c in '.,!?;:' for c in word):
                modulated_words.append(word)
                continue
            
            # Check emphasis words list
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in emphasis_set:
                should_emphasize = True
            # Random chance based on frequency
            elif random.random() < params.emphasis_frequency:
                should_emphasize = True
            
            if should_emphasize:
                if params.emphasis_pattern == EmphasisPattern.CAPS:
                    modulated_words.append(word.upper())
                else:
                    prefix, suffix = emphasis_patterns.get(params.emphasis_pattern, ('', ''))
                    # Handle punctuation at the end of words
                    punct = ''
                    clean_word = word
                    if word and word[-1] in '.,!?;:':
                        punct = word[-1]
                        clean_word = word[:-1]
                    modulated_words.append(f"{prefix}{clean_word}{suffix}{punct}")
            else:
                modulated_words.append(word)
        
        return ' '.join(modulated_words)
    
    def _modulate_word_choice(self, 
                              text: str, 
                              emotion: str, 
                              intensity: float,
                              params: ToneParameters) -> str:
        """Modulate word choice based on emotional parameters"""
        # Handle contractions based on formality
        if self.personality['formality'] < 0.4:
            # Make more contractions
            contractions = {
                r'\bI am\b': "I'm",
                r'\byou are\b': "you're",
                r'\bit is\b': "it's",
                r'\bthat is\b': "that's",
                r'\bthere is\b': "there's",
                r'\bcannot\b': "can't",
                r'\bwill not\b': "won't",
                r'\bdo not\b': "don't",
                r'\bdid not\b': "didn't",
                r'\bhave not\b': "haven't",
            }
            
            for pattern, replacement in contractions.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _add_pauses(self, 
                    text: str, 
                    emotion: str, 
                    intensity: float,
                    params: ToneParameters) -> str:
        """Add pause indicators to text"""
        # For text output, pauses are represented by ellipses or commas
        if params.ellipsis_frequency > 0.2 and intensity > 0.5:
            # Add comma pauses in longer sentences
            sentences = text.split('. ')
            modulated_sentences = []
            
            for sentence in sentences:
                words = sentence.split()
                if len(words) > 8 and random.random() < 0.2:
                    # Insert comma after first few words
                    insert_pos = random.randint(2, min(4, len(words) - 2))
                    words[insert_pos] = words[insert_pos] + ','
                    modulated_sentences.append(' '.join(words))
                else:
                    modulated_sentences.append(sentence)
            
            return '. '.join(modulated_sentences)
        
        return text
    
    def _voice_params_to_dict(self, params: ToneParameters) -> Dict[str, float]:
        """Convert tone parameters to voice parameter dictionary"""
        return {
            'rate': round(params.speech_rate, 3),
            'pitch': round(params.pitch_mean, 3),
            'pitch_variation': round(params.pitch_variation, 3),
            'volume': round(params.volume, 3),
            'breathiness': round(params.breathiness, 3)
        }
    
    def _voice_params_to_dict_from_raw(self, params: Dict[str, float]) -> Dict[str, float]:
        """Convert raw voice parameters to standardized dict"""
        return {
            'rate': round(params.get('rate', 1.0), 3),
            'pitch': round(params.get('pitch', 1.0), 3),
            'pitch_variation': round(params.get('variation', 0.1), 3),
            'volume': round(params.get('volume', 1.0), 3),
            'breathiness': round(params.get('breathiness', 0.05), 3)
        }
    
    def _record_modulation(self, emotion: str, intensity: float, 
                           params: ToneParameters) -> None:
        """Record modulation for consistency tracking"""
        self.recent_modulations.append({
            'emotion': emotion,
            'intensity': round(intensity, 3),
            'punctuation_style': params.punctuation_style.value,
            'emphasis_pattern': params.emphasis_pattern.value,
            'timestamp': time.time()
        })
        
        if len(self.recent_modulations) > self.max_history:
            self.recent_modulations.pop(0)
    
    def get_current_voice_params(self) -> Dict[str, float]:
        """Get current voice parameters"""
        return self._voice_params_to_dict(self.current_params)
    
    def get_modulation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent modulations"""
        if limit <= 0:
            return []
        return self.recent_modulations[-min(limit, len(self.recent_modulations)):]
    
    def reset(self) -> None:
        """Reset modulator state"""
        self.current_params = ToneParameters()
        self.recent_modulations.clear()
        logger.info("ToneModulator reset")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Tone Modulator Test ===\n")
    
    # Create modulator with fixed seed for reproducible tests
    modulator = ToneModulator(
        personality={
            'vocal_expressiveness': 0.4,
            'punctuation_tendency': 0.3,
            'emphasis_tendency': 0.3,
            'dryness': 0.8,
            'formality': 0.5
        },
        random_seed=42
    )
    
    # Test text with multiple sentences
    base_text = "I see. That is an interesting development. I will consider it carefully."
    
    # Test emotional states
    test_emotional_states = [
        {
            'dominant': 'neutral',
            'emotions': {'neutral': 0.8},
            'pad': {'valence': 0.0, 'arousal': 0.3}
        },
        {
            'dominant': 'dark_amusement',
            'emotions': {'dark_amusement': 0.7},
            'pad': {'valence': 0.2, 'arousal': 0.5}
        },
        {
            'dominant': 'anger',
            'emotions': {'anger': 0.6},
            'pad': {'valence': -0.4, 'arousal': 0.7}
        },
        {
            'dominant': 'sadness',
            'emotions': {'sadness': 0.6},
            'pad': {'valence': -0.4, 'arousal': 0.2}
        },
        {
            'dominant': 'surprise',
            'emotions': {'surprise': 0.7},
            'pad': {'valence': 0.3, 'arousal': 0.8}
        },
        {
            'dominant': 'protective',
            'emotions': {'protective': 0.8},
            'pad': {'valence': 0.1, 'arousal': 0.5}
        }
    ]
    
    print(f"Base text: {base_text}\n")
    
    for i, emotional_state in enumerate(test_emotional_states):
        print(f"--- Test {i+1}: {emotional_state['dominant']} ---")
        
        # Apply tone modulation
        modulated = modulator.apply_to_text(base_text, emotional_state)
        
        print(f"Original: {base_text}")
        print(f"Modulated: {modulated}")
        
        # Get voice parameters
        voice_params = modulator.get_voice_params(emotional_state)
        print(f"Voice params: {voice_params}")
        print()
    
    print("--- Modulation History ---")
    history = modulator.get_modulation_history()
    for h in history:
        print(f"  {h['emotion']} at {time.ctime(h['timestamp'])}: {h['punctuation_style']}")
    
    print("\n=== Test Complete ===")