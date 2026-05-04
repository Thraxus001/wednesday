"""
personality.py - Core personality definition for Wednesday AI

This module defines Wednesday's consistent personality traits that persist across
all interactions and influence every aspect of her cognition. The personality
serves as the stable core of her identity, ensuring that all responses and
behaviors feel authentically "Wednesday."

Key improvements:
- Added comprehensive validation and error handling
- Fixed trait interaction calculations with proper weight normalization
- Enhanced state management with persistence
- Added proper type hints and documentation
- Improved random operations with seed support for reproducibility
"""

import logging
import time
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class TraitDomain(Enum):
    """Enumeration of personality trait domains"""
    # Big Five domains
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"
    
    # Wednesday-specific domains
    DARK_HUMOR = "dark_humor"
    DEADPAN = "deadpan"
    LOYALTY = "loyalty"
    SKEPTICISM = "skepticism"
    CURIOSITY = "curiosity"
    INDEPENDENCE = "independence"
    PROTECTIVENESS = "protectiveness"
    SARCASM = "sarcasm"
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


@dataclass
class PersonalityProfile:
    """
    Complete personality profile with all traits.
    
    Traits are stored as float values 0-1, where:
    - 0 = extremely low in this trait
    - 1 = extremely high in this trait
    """
    # Big Five traits
    openness: float = 0.7           # Curiosity, creativity, openness to experience
    conscientiousness: float = 0.8    # Organization, reliability, discipline
    extraversion: float = 0.3         # Sociability, assertiveness, energy
    agreeableness: float = 0.4        # Compassion, cooperation, trust
    neuroticism: float = 0.2          # Emotional stability vs instability
    
    # Wednesday-specific traits
    dark_humor: float = 0.9           # Appreciation for macabre humor
    deadpan: float = 0.95             # Tendency toward flat delivery
    loyalty: float = 0.9              # Fierce loyalty to trusted few
    skepticism: float = 0.7           # Questioning, not easily convinced
    curiosity: float = 0.8            # Intellectual curiosity
    independence: float = 0.9         # Self-reliance, autonomy
    protectiveness: float = 0.8       # Protective of those she cares about
    sarcasm: float = 0.8              # Tendency toward sarcastic remarks
    
    # Metadata
    name: str = "Wednesday"
    version: str = "1.0"
    
    def __post_init__(self):
        """Validate trait values"""
        self._validate_trait('openness', self.openness)
        self._validate_trait('conscientiousness', self.conscientiousness)
        self._validate_trait('extraversion', self.extraversion)
        self._validate_trait('agreeableness', self.agreeableness)
        self._validate_trait('neuroticism', self.neuroticism)
        self._validate_trait('dark_humor', self.dark_humor)
        self._validate_trait('deadpan', self.deadpan)
        self._validate_trait('loyalty', self.loyalty)
        self._validate_trait('skepticism', self.skepticism)
        self._validate_trait('curiosity', self.curiosity)
        self._validate_trait('independence', self.independence)
        self._validate_trait('protectiveness', self.protectiveness)
        self._validate_trait('sarcasm', self.sarcasm)
    
    def _validate_trait(self, name: str, value: float) -> None:
        """Validate a single trait value"""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(value)}")
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            # Big Five
            'openness': round(self.openness, 3),
            'conscientiousness': round(self.conscientiousness, 3),
            'extraversion': round(self.extraversion, 3),
            'agreeableness': round(self.agreeableness, 3),
            'neuroticism': round(self.neuroticism, 3),
            
            # Wednesday-specific
            'dark_humor': round(self.dark_humor, 3),
            'deadpan': round(self.deadpan, 3),
            'loyalty': round(self.loyalty, 3),
            'skepticism': round(self.skepticism, 3),
            'curiosity': round(self.curiosity, 3),
            'independence': round(self.independence, 3),
            'protectiveness': round(self.protectiveness, 3),
            'sarcasm': round(self.sarcasm, 3),
        }
    
    def get_big_five(self) -> Dict[str, float]:
        """Get only Big Five traits"""
        return {
            'openness': round(self.openness, 3),
            'conscientiousness': round(self.conscientiousness, 3),
            'extraversion': round(self.extraversion, 3),
            'agreeableness': round(self.agreeableness, 3),
            'neuroticism': round(self.neuroticism, 3)
        }
    
    def get_wednesday_traits(self) -> Dict[str, float]:
        """Get only Wednesday-specific traits"""
        return {
            'dark_humor': round(self.dark_humor, 3),
            'deadpan': round(self.deadpan, 3),
            'loyalty': round(self.loyalty, 3),
            'skepticism': round(self.skepticism, 3),
            'curiosity': round(self.curiosity, 3),
            'independence': round(self.independence, 3),
            'protectiveness': round(self.protectiveness, 3),
            'sarcasm': round(self.sarcasm, 3)
        }


@dataclass
class PersonalityState:
    """Temporary state variables that modulate personality expression"""
    fatigue: float = 0.0           # 0-1 how tired/overwhelmed
    focus_level: float = 1.0        # 0-1 attention focus
    social_energy: float = 1.0      # 0-1 energy for social interaction
    
    def __post_init__(self):
        """Validate state values"""
        self._validate_state('fatigue', self.fatigue)
        self._validate_state('focus_level', self.focus_level)
        self._validate_state('social_energy', self.social_energy)
    
    def _validate_state(self, name: str, value: float) -> None:
        """Validate state value"""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(value)}")
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'fatigue': round(self.fatigue, 3),
            'focus_level': round(self.focus_level, 3),
            'social_energy': round(self.social_energy, 3)
        }


class Personality:
    """
    Wednesday's consistent personality that persists across all interactions.
    
    This class defines who Wednesday is at her core. All other modules reference
    it to ensure their behavior aligns with her character. The personality
    influences:
    - Emotional responses (what she finds funny, sad, etc.)
    - Cognitive biases (how she processes information)
    - Social behavior (how she interacts with others)
    - Decision making (what she values and prioritizes)
    - Expression style (how she communicates)
    
    The personality is designed to be:
    - Consistent (stable across time)
    - Coherent (traits work together logically)
    - Authentic (feels like the character)
    - Nuanced (not a caricature)
    """
    
    # Default Wednesday profile (canonical)
    DEFAULT_PROFILE = PersonalityProfile()
    
    # Trait interaction weights (how traits combine to produce behavior)
    TRAIT_INTERACTIONS = {
        # Humor emerges from dark_humor + openness + low agreeableness
        'humor_style': {
            'dark_humor': 0.5,
            'openness': 0.3,
            'sarcasm': 0.4,
            'agreeableness': -0.2,  # Lower agreeableness = darker humor
        },
        
        # Trust emerges from loyalty + agreeableness + low skepticism
        'trust_tendency': {
            'loyalty': 0.5,
            'agreeableness': 0.3,
            'skepticism': -0.4,
            'neuroticism': -0.1
        },
        
        # Curiosity emerges from openness + curiosity + independence
        'exploration_drive': {
            'openness': 0.5,
            'curiosity': 0.5,
            'independence': 0.2,
            'conscientiousness': -0.1  # Less structured exploration
        },
        
        # Social engagement from extraversion + agreeableness
        'social_engagement': {
            'extraversion': 0.6,
            'agreeableness': 0.3,
            'independence': -0.3,
            'deadpan': -0.2
        },
        
        # Emotional expression from neuroticism + extraversion + deadpan
        'emotional_expressiveness': {
            'neuroticism': 0.4,
            'extraversion': 0.3,
            'deadpan': -0.5,
            'dark_humor': 0.1
        },
        
        # Protectiveness from loyalty + protectiveness + low agreeableness
        'protective_instinct': {
            'loyalty': 0.6,
            'protectiveness': 0.5,
            'agreeableness': -0.2,  # More aggressive protection
            'neuroticism': 0.1
        }
    }
    
    # Valid trait names for quick lookup
    VALID_TRAITS = set(PersonalityProfile().to_dict().keys())
    
    def __init__(self, personality_config: Optional[Dict[str, float]] = None, random_seed: Optional[int] = None):
        """
        Initialize Wednesday's personality.
        
        Args:
            personality_config: Optional configuration to override default traits
            random_seed: Optional seed for reproducible random operations
            
        Raises:
            ValueError: If personality_config contains invalid traits or values
        """
        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
        
        # Start with default profile
        if personality_config:
            # Validate configuration
            self._validate_config(personality_config)
            
            # Merge with default
            config = self.DEFAULT_PROFILE.to_dict()
            config.update(personality_config)
            self.profile = PersonalityProfile(**config)
        else:
            self.profile = self.DEFAULT_PROFILE
        
        # Track personality state (temporary variations)
        self.current_state = PersonalityState()
        
        # Personality version/history
        self.version = self.profile.version
        self.creation_time = time.time()
        
        # Track state history for context
        self.state_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        
        logger.info(f"Personality initialized: {self.summarize()}")
    
    def _validate_config(self, config: Dict[str, float]) -> None:
        """Validate personality configuration"""
        for key, value in config.items():
            if key not in self.VALID_TRAITS:
                raise ValueError(f"Unknown personality trait: {key}")
            if not isinstance(value, (int, float)):
                raise TypeError(f"Value for {key} must be a number, got {type(value)}")
            if not 0 <= value <= 1:
                raise ValueError(f"Value for {key} must be between 0 and 1, got {value}")
    
    def get_behavior_bias(self, situation: Dict[str, Any]) -> Dict[str, float]:
        """
        Get personality-based behavior biases for a specific situation.
        
        Args:
            situation: Dictionary describing the current situation
            
        Returns:
            Dictionary of bias factors for different behavior dimensions
        """
        biases = {}
        
        # Base biases from core traits
        biases['humor_probability'] = self._calculate_humor_bias(situation)
        biases['trust_bias'] = self._calculate_trust_bias(situation)
        biases['curiosity_bias'] = self._calculate_curiosity_bias(situation)
        biases['social_bias'] = self._calculate_social_bias(situation)
        biases['emotional_expression'] = self._calculate_expression_bias(situation)
        biases['protective_bias'] = self._calculate_protective_bias(situation)
        
        # Apply current state modulation
        biases = self._apply_state_modulation(biases, situation)
        
        # Situation-specific adjustments
        context_type = situation.get('context_type', 'general')
        
        if context_type == 'emergency':
            biases['emotional_expression'] *= 0.5  # More controlled in emergencies
            biases['protective_bias'] *= 1.5       # More protective
        
        elif context_type == 'intellectual':
            biases['curiosity_bias'] *= 1.3        # More curious in intellectual contexts
            biases['humor_probability'] *= 0.8     # Less humor in serious discussion
        
        elif context_type == 'social':
            biases['social_bias'] *= 1.2            # More socially engaged
        
        # Relationship adjustments
        relationship = situation.get('relationship', 'unknown')
        if relationship == 'close_friend':
            biases['trust_bias'] *= 1.3             # More trusting with friends
            biases['protective_bias'] *= 1.4        # More protective of friends
            biases['social_bias'] *= 1.2            # More socially engaged
        elif relationship == 'stranger':
            biases['trust_bias'] *= 0.7             # Less trusting with strangers
            biases['social_bias'] *= 0.8            # Less socially engaged
        
        # Ensure all values are within 0-1 range
        for key in biases:
            biases[key] = max(0.0, min(1.0, biases[key]))
        
        return biases
    
    def should_find_this_funny(self, stimulus: Any, context: Optional[Dict] = None) -> float:
        """
        Determine if Wednesday would find something funny, and how funny.
        
        Wednesday's humor is dark, intellectual, and often morbid.
        
        Args:
            stimulus: The stimulus (text, situation, etc.)
            context: Additional context
            
        Returns:
            Probability/ intensity of finding it funny (0-1)
        """
        humor_probability = 0.0
        
        # Base humor tendency from personality
        base_humor = self.profile.dark_humor * 0.6 + self.profile.sarcasm * 0.4
        
        # Convert stimulus to string for analysis if needed
        if not isinstance(stimulus, str):
            try:
                stimulus = str(stimulus)
            except:
                stimulus = ""
        
        stimulus_lower = stimulus.lower()
        
        # Check for dark humor triggers
        dark_keywords = ['death', 'dead', 'kill', 'murder', 'corpse', 'grave',
                        'morbid', 'macabre', 'dark', 'grim', 'skeleton',
                        'funeral', 'coffin', 'tomb', 'haunt', 'ghost']
        
        dark_matches = sum(1 for keyword in dark_keywords if keyword in stimulus_lower)
        if dark_matches > 0:
            humor_probability += min(0.5, dark_matches * 0.1) * self.profile.dark_humor
        
        # Check for irony/sarcasm triggers
        irony_keywords = ['obviously', 'clearly', 'naturally', 'indeed', 
                         'fascinating', 'interesting', 'delightful', 'ironic']
        
        irony_matches = sum(1 for keyword in irony_keywords if keyword in stimulus_lower)
        if irony_matches > 0:
            humor_probability += min(0.3, irony_matches * 0.1) * self.profile.sarcasm
        
        # Check for intellectual humor (wordplay, paradoxes)
        word_count = len(stimulus.split())
        if '?' in stimulus and word_count > 10:
            humor_probability += 0.2 * self.profile.openness
        
        # Check for absurdity
        absurd_keywords = ['absurd', 'ridiculous', 'nonsense', 'impossible', 'unbelievable']
        if any(word in stimulus_lower for word in absurd_keywords):
            humor_probability += 0.2 * self.profile.openness
        
        # Context modulation
        if context:
            # Inappropriate contexts reduce humor
            formality = context.get('formality', 0)
            if formality > 0.8:
                humor_probability *= 0.3
            elif formality > 0.6:
                humor_probability *= 0.7
            
            # With trusted friends, humor flows more freely
            relationship = context.get('relationship', 'unknown')
            if relationship in ['close_friend', 'trusted']:
                humor_probability *= 1.3
            
            # During serious moments, less humor
            seriousness = context.get('seriousness', 0)
            if seriousness > 0.7:
                humor_probability *= 0.2
        
        # Personality modulation - high deadpan means less obvious laughter
        humor_probability *= (1 - self.profile.deadpan * 0.3)
        
        return min(1.0, humor_probability * base_humor)
    
    def express_personality_in_response(self, base_response: str, 
                                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add personality flavor to any response.
        
        This modifies a base response to include Wednesday's characteristic
        personality elements.
        
        Args:
            base_response: The core response content
            context: Situation context
            
        Returns:
            Dictionary with modified response and personality metadata
        """
        if not base_response:
            return {
                'original_response': '',
                'modified_response': '',
                'modifications': [],
                'personality_biases': {},
                'humor_probability': 0.0
            }
        
        context = context or {}
        
        # Get behavior biases for this context
        biases = self.get_behavior_bias(context)
        
        # Start with the base response
        final_response = base_response
        
        # Track modifications
        modifications = []
        
        # Add deadpan elements (if appropriate)
        if self.profile.deadpan > 0.7 and random.random() < self.profile.deadpan * 0.3:
            # Ensure response has period at end, not exclamation
            if final_response.endswith('!'):
                final_response = final_response[:-1] + '.'
                modifications.append('deadpan_punctuation')
        
        # Add dark humor if appropriate
        humor_prob = self.should_find_this_funny(base_response, context)
        if humor_prob > 0.5 and random.random() < humor_prob * 0.4:
            # Add a darkly humorous observation
            dark_additions = [
                " How delightfully morbid.",
                " I find that oddly satisfying.",
                " There's something almost poetic about it.",
                " How wonderfully grim.",
                " I appreciate the dark irony.",
                " That has a certain... charm.",
            ]
            addition = random.choice(dark_additions)
            final_response += addition
            modifications.append('dark_humor_addition')
        
        # Add sarcasm if appropriate
        if (self.profile.sarcasm > 0.6 and 
            biases.get('humor_probability', 0) > 0.3 and
            random.random() < self.profile.sarcasm * 0.2):
            modifications.append('sarcasm_potential')
        
        # Add intellectual flourish
        if (self.profile.openness > 0.7 and 
            biases.get('curiosity_bias', 0) > 0.5 and
            random.random() < self.profile.openness * 0.2):
            modifications.append('intellectual_style')
        
        # Add bluntness if low agreeableness
        if self.profile.agreeableness < 0.3 and random.random() < 0.2:
            modifications.append('blunt_delivery')
        
        # Record state for history
        self._record_state(context, modifications)
        
        return {
            'original_response': base_response,
            'modified_response': final_response,
            'modifications': modifications,
            'personality_biases': {k: round(v, 3) for k, v in biases.items()},
            'humor_probability': round(humor_prob, 3)
        }
    
    def get_trait(self, trait_name: str) -> float:
        """
        Get the value of a specific trait.
        
        Args:
            trait_name: Name of trait
            
        Returns:
            Trait value (0-1)
            
        Raises:
            ValueError: If trait_name is invalid
        """
        if trait_name not in self.VALID_TRAITS:
            raise ValueError(f"Unknown trait: {trait_name}")
        
        return getattr(self.profile, trait_name, 0.0)
    
    def get_trait_interaction(self, interaction_name: str) -> float:
        """
        Calculate a composite trait from multiple interacting traits.
        
        Args:
            interaction_name: Name of interaction from TRAIT_INTERACTIONS
            
        Returns:
            Composite score (0-1)
        """
        if interaction_name not in self.TRAIT_INTERACTIONS:
            logger.warning(f"Unknown interaction: {interaction_name}")
            return 0.5
        
        interaction = self.TRAIT_INTERACTIONS[interaction_name]
        
        # Separate positive and negative weights
        positive_score = 0.0
        positive_weight = 0.0
        negative_factors = 1.0
        
        for trait, weight in interaction.items():
            trait_value = getattr(self.profile, trait, 0.5)
            
            if weight > 0:
                positive_score += trait_value * weight
                positive_weight += weight
            else:
                # Negative weights reduce the score multiplicatively
                negative_factors *= (1 + weight * trait_value)  # weight is negative
        
        # Calculate base score from positive traits
        if positive_weight > 0:
            base_score = positive_score / positive_weight
        else:
            base_score = 0.5
        
        # Apply negative factors
        final_score = base_score * negative_factors
        
        return max(0.0, min(1.0, final_score))
    
    def get_all_trait_interactions(self) -> Dict[str, float]:
        """Get all trait interaction scores"""
        return {
            name: self.get_trait_interaction(name)
            for name in self.TRAIT_INTERACTIONS.keys()
        }
    
    def summarize(self) -> str:
        """Get a brief personality summary"""
        big_five = self.profile.get_big_five()
        
        # Determine dominant traits
        high_traits = [t.replace('_', ' ') for t, v in big_five.items() if v > 0.7]
        low_traits = [t.replace('_', ' ') for t, v in big_five.items() if v < 0.3]
        
        summary = f"Wednesday: "
        if high_traits:
            summary += f"High {', '.join(high_traits)}. "
        if low_traits:
            summary += f"Low {', '.join(low_traits)}. "
        
        summary += f"Dark humor: {self.profile.dark_humor:.1f}, "
        summary += f"Deadpan: {self.profile.deadpan:.1f}"
        
        return summary
    
    def get_full_profile(self) -> Dict[str, Any]:
        """Get complete personality profile with metadata"""
        return {
            'traits': self.profile.to_dict(),
            'big_five': self.profile.get_big_five(),
            'wednesday_traits': self.profile.get_wednesday_traits(),
            'interactions': self.get_all_trait_interactions(),
            'current_state': self.current_state.to_dict(),
            'version': self.version,
            'created': self.creation_time
        }
    
    def _calculate_humor_bias(self, situation: Dict) -> float:
        """Calculate humor probability bias"""
        return self.get_trait_interaction('humor_style')
    
    def _calculate_trust_bias(self, situation: Dict) -> float:
        """Calculate trust tendency bias"""
        return self.get_trait_interaction('trust_tendency')
    
    def _calculate_curiosity_bias(self, situation: Dict) -> float:
        """Calculate curiosity/exploration bias"""
        return self.get_trait_interaction('exploration_drive')
    
    def _calculate_social_bias(self, situation: Dict) -> float:
        """Calculate social engagement bias"""
        return self.get_trait_interaction('social_engagement')
    
    def _calculate_expression_bias(self, situation: Dict) -> float:
        """Calculate emotional expression bias"""
        return self.get_trait_interaction('emotional_expressiveness')
    
    def _calculate_protective_bias(self, situation: Dict) -> float:
        """Calculate protective instinct bias"""
        return self.get_trait_interaction('protective_instinct')
    
    def _apply_state_modulation(self, biases: Dict[str, float], 
                                 situation: Dict) -> Dict[str, float]:
        """Apply current state modulation to biases"""
        modulated = biases.copy()
        
        # Fatigue reduces social engagement and humor
        fatigue = self.current_state.fatigue
        if fatigue > 0:
            modulated['social_bias'] *= max(0.1, 1 - fatigue * 0.5)
            modulated['humor_probability'] *= max(0.1, 1 - fatigue * 0.3)
            modulated['curiosity_bias'] *= max(0.1, 1 - fatigue * 0.2)
        
        # Low social energy reduces social engagement
        social_energy = self.current_state.social_energy
        if social_energy < 0.5:
            modulated['social_bias'] *= social_energy * 2  # Scale to 0-1 range
        
        # High focus increases curiosity but decreases social
        focus = self.current_state.focus_level
        if focus > 0.8:
            focus_boost = (focus - 0.8) * 2.5  # 0-0.5 range
            modulated['curiosity_bias'] *= (1 + focus_boost)
            modulated['social_bias'] *= (1 - focus_boost * 0.5)
        
        return modulated
    
    def _record_state(self, context: Dict, modifications: List) -> None:
        """Record state for history"""
        self.state_history.append({
            'timestamp': time.time(),
            'state': self.current_state.to_dict(),
            'context': context,
            'modifications': modifications
        })
        
        if len(self.state_history) > self.max_history_size:
            self.state_history.pop(0)
    
    def update_state(self, fatigue: Optional[float] = None,
                     focus: Optional[float] = None,
                     social_energy: Optional[float] = None) -> None:
        """
        Update current personality state.
        
        Args:
            fatigue: Fatigue level (0-1)
            focus: Focus level (0-1)
            social_energy: Social energy level (0-1)
            
        Raises:
            ValueError: If any value is outside valid range
        """
        if fatigue is not None:
            if not 0 <= fatigue <= 1:
                raise ValueError(f"fatigue must be between 0 and 1, got {fatigue}")
            self.current_state.fatigue = fatigue
        
        if focus is not None:
            if not 0 <= focus <= 1:
                raise ValueError(f"focus must be between 0 and 1, got {focus}")
            self.current_state.focus_level = focus
        
        if social_energy is not None:
            if not 0 <= social_energy <= 1:
                raise ValueError(f"social_energy must be between 0 and 1, got {social_energy}")
            self.current_state.social_energy = social_energy
    
    def reset_state(self) -> None:
        """Reset state to default values"""
        self.current_state = PersonalityState()
        logger.info("Personality state reset to defaults")
    
    def get_state_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get state history"""
        if limit and limit > 0:
            return self.state_history[-limit:]
        return self.state_history.copy()
    
    def __repr__(self) -> str:
        return f"Personality({self.summarize()})"


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Personality Module Test ===\n")
    
    # Create Wednesday's personality with fixed seed for reproducibility
    wednesday = Personality(random_seed=42)
    
    print(wednesday.summarize())
    print("\nFull profile:")
    profile = wednesday.get_full_profile()
    for key, value in profile.items():
        if key not in ['traits', 'big_five', 'wednesday_traits', 'interactions']:
            print(f"  {key}: {value}")
    
    print("\nBig Five traits:")
    for trait, value in profile['big_five'].items():
        print(f"  {trait}: {value}")
    
    print("\nWednesday-specific traits:")
    for trait, value in profile['wednesday_traits'].items():
        print(f"  {trait}: {value}")
    
    print("\nTrait interactions:")
    for interaction, value in profile['interactions'].items():
        print(f"  {interaction}: {value:.3f}")
    
    # Test humor detection
    test_stimuli = [
        "The cemetery is looking lovely this time of year.",
        "I need help with my math homework.",
        "Death comes for us all eventually.",
        "That's a very interesting observation.",
        "Someone betrayed my trust.",
        "Want to hear a joke?",
        "The situation is absolutely absurd.",
    ]
    
    print("\n--- Humor Detection ---")
    for stimulus in test_stimuli:
        funny = wednesday.should_find_this_funny(stimulus)
        print(f"  '{stimulus[:30]}...' -> {funny:.2f}")
    
    # Test behavior biases in different situations
    situations = [
        {'context_type': 'casual', 'relationship': 'friend', 'formality': 0.3},
        {'context_type': 'emergency', 'relationship': 'stranger', 'formality': 0.1},
        {'context_type': 'intellectual', 'relationship': 'acquaintance', 'formality': 0.5},
        {'context_type': 'casual', 'relationship': 'close_friend', 'formality': 0.2},
        {'context_type': 'social', 'relationship': 'colleague', 'formality': 0.6},
    ]
    
    print("\n--- Behavior Biases by Situation ---")
    for i, situation in enumerate(situations):
        print(f"\nSituation {i+1}: {situation['context_type']} with {situation['relationship']}")
        biases = wednesday.get_behavior_bias(situation)
        for bias, value in biases.items():
            print(f"  {bias}: {value:.2f}")
    
    # Test personality expression
    base_response = "I understand. That makes sense."
    
    print("\n--- Personality Expression ---")
    for i, situation in enumerate(situations[:3]):
        print(f"\nContext: {situation['context_type']}")
        result = wednesday.express_personality_in_response(base_response, situation)
        print(f"  Original: {result['original_response']}")
        print(f"  Modified: {result['modified_response']}")
        print(f"  Modifications: {result['modifications']}")
    
    print("\n=== Test Complete ===")