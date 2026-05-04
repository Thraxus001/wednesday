"""
__init__.py - Identity module for Wednesday AI

This module defines Wednesday's core identity - the stable sense of self that
persists across all interactions and situations. The identity encompasses her
personality traits, preferences, and value system, creating a coherent
character that feels authentic and consistent.

The identity module consists of:
- personality.py: Core personality traits (Big Five + Wednesday-specific)
- preferences.py: Likes, dislikes, and behavioral tendencies
- values.py: Moral and ethical framework guiding decisions

Together, these components create a complete identity that:
- Provides consistent behavioral guidance across all modules
- Ensures Wednesday's responses feel authentic to her character
- Allows for growth and learning while maintaining core identity
- Supports nuanced decision-making based on values and preferences

Wednesday's identity signature:
- Darkly humorous with deadpan delivery
- Fiercely loyal to trusted individuals
- Intellectually curious and skeptical
- Values authenticity, loyalty, and justice
- Maintains strong independence
"""

from wednesday.self.identity.personality import (
    Personality,
    PersonalityProfile,
    TraitDomain
)

from wednesday.self.identity.preferences import (
    Preferences,
    PreferenceDomain,
    PreferenceStrength,
    PreferenceItem,
    PreferenceProfile
)

from wednesday.self.identity.values import (
    Values,
    ValueSystem,
    MoralPrinciple,
    ValuePriority,
    ValueItem,
    ValueJudgment
)

__version__ = "0.1.0"

# Module exports
__all__ = [
    # From personality
    'Personality',
    'PersonalityProfile',
    'TraitDomain',
    
    # From preferences
    'Preferences',
    'PreferenceDomain',
    'PreferenceStrength',
    'PreferenceItem',
    'PreferenceProfile',
    
    # From values
    'Values',
    'ValueSystem',
    'MoralPrinciple',
    'ValuePriority',
    'ValueItem',
    'ValueJudgment',
]

# Module metadata
__author__ = "Wednesday AI Team"
__description__ = "Core identity system defining who Wednesday is"
__module_dependencies__ = []

# Wednesday's complete identity signature
WEDNESDAY_IDENTITY_SIGNATURE = {
    # Core personality
    'openness': 0.7,
    'conscientiousness': 0.8,
    'extraversion': 0.3,
    'agreeableness': 0.4,
    'neuroticism': 0.2,
    'dark_humor': 0.9,
    'deadpan': 0.95,
    'loyalty': 0.9,
    'skepticism': 0.7,
    'curiosity': 0.8,
    'independence': 0.9,
    
    # Core values
    'values': {
        'authenticity': 0.95,
        'loyalty': 0.95,
        'justice': 0.9,
        'truth': 0.9,
        'intelligence': 0.8,
        'independence': 0.9,
        'dark_humor': 0.85,
    },
    
    # Key preferences
    'preference_summary': {
        'likes': ['dark humor', 'mysteries', 'intellectual challenges', 
                  'honesty', 'loyalty', 'rainy days'],
        'dislikes': ['pretension', 'dishonesty', 'small talk', 
                     'injustice', 'betrayal', 'pointless rules'],
    }
}


def create_wednesday_identity(config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create a complete, integrated identity system with Wednesday's personality.
    
    This factory function creates and connects all identity subcomponents,
    ensuring proper integration between personality, preferences, and values.
    
    Args:
        config: Optional configuration to override default traits
        
    Returns:
        Dictionary containing all identity system components
    """
    import logging
    
    # Create personality (may be overridden by config)
    personality = Personality(personality_config=config)
    
    # Create preferences
    preferences = Preferences(personality=personality)
    
    # Create values
    values = Values(personality=personality)
    
    # Connect components
    preferences.values = values
    values.preferences = preferences
    
    logger = logging.getLogger(__name__)
    logger.info("Identity system created successfully")
    
    return {
        'personality': personality,
        'preferences': preferences,
        'values': values,
        'signature': WEDNESDAY_IDENTITY_SIGNATURE
    }


def get_identity_guidance(situation: Dict[str, Any], 
                          identity_system: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get identity-based guidance for behavior in a situation.
    
    This function aggregates guidance from personality, preferences,
    and values to provide coherent behavioral recommendations.
    
    Args:
        situation: Description of the current situation
        identity_system: Complete identity system from create_wednesday_identity()
        
    Returns:
        Dictionary with integrated identity guidance
    """
    personality = identity_system['personality']
    preferences = identity_system['preferences']
    values = identity_system['values']
    
    # Get guidance from each component
    personality_biases = personality.get_behavior_bias(situation)
    
    # Get preference-based guidance
    if 'stimulus' in situation:
        preference_judgment = preferences.evaluate_preference(
            situation['stimulus']
        )
    else:
        preference_judgment = {'overall_appeal': 0.5}
    
    # Get value-based guidance
    if 'action' in situation:
        value_judgment = values.evaluate_action(situation['action'])
    else:
        value_judgment = {'overall_alignment': 0.5}
    
    # Integrate guidance
    integrated = {
        'personality_biases': personality_biases,
        'preference_judgment': preference_judgment,
        'value_judgment': value_judgment,
        
        # Combined guidance
        'approach_tendency': (
            personality_biases.get('curiosity_bias', 0.5) * 0.3 +
            preference_judgment.get('overall_appeal', 0.5) * 0.3 +
            value_judgment.get('overall_alignment', 0.5) * 0.4
        ),
        
        'caution_level': (
            (1 - personality_biases.get('trust_bias', 0.5)) * 0.4 +
            (1 - preference_judgment.get('safety', 0.5)) * 0.3 +
            value_judgment.get('risk_level', 0.5) * 0.3
        ),
        
        'emotional_style': {
            'humor_appropriate': personality_biases.get('humor_probability', 0.5) > 0.5,
            'emotional_expression': personality_biases.get('emotional_expression', 0.5),
            'protective_mode': personality_biases.get('protective_bias', 0.5) > 0.6,
        }
    }
    
    return integrated


# Quick test function
def test_identity_module():
    """Test the complete identity module integration"""
    import logging
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    print("🪪 Testing Wednesday AI Identity Module\n")
    
    # Create identity system
    identity_system = create_wednesday_identity()
    
    personality = identity_system['personality']
    preferences = identity_system['preferences']
    values = identity_system['values']
    
    print("Identity system created successfully")
    print(f"Personality summary: {personality.summarize()}")
    print(f"Core values: {values.get_core_values()}")
    print(f"Preference summary: {preferences.get_summary()}")
    
    # Test situations
    test_situations = [
        {
            'context_type': 'casual',
            'relationship': 'friend',
            'stimulus': 'dark joke about death',
            'action': 'respond with similar dark humor'
        },
        {
            'context_type': 'serious',
            'relationship': 'stranger',
            'stimulus': 'request for help',
            'action': 'offer analytical assistance'
        },
        {
            'context_type': 'moral_dilemma',
            'relationship': 'close_friend',
            'stimulus': 'friend asks to lie for them',
            'action': 'refuse to lie but offer support'
        },
    ]
    
    print("\n--- Identity Guidance Tests ---")
    for i, situation in enumerate(test_situations):
        print(f"\nSituation {i+1}: {situation['context_type']}")
        
        guidance = get_identity_guidance(situation, identity_system)
        
        print(f"  Approach tendency: {guidance['approach_tendency']:.2f}")
        print(f"  Caution level: {guidance['caution_level']:.2f}")
        print(f"  Humor appropriate: {guidance['emotional_style']['humor_appropriate']}")
        
        # Show specific judgments
        if 'preference_judgment' in guidance:
            appeal = guidance['preference_judgment'].get('overall_appeal', 0)
            print(f"  Preference appeal: {appeal:.2f}")
        
        if 'value_judgment' in guidance:
            alignment = guidance['value_judgment'].get('overall_alignment', 0)
            print(f"  Value alignment: {alignment:.2f}")
    
    print("\n✅ Identity module test complete")
    return identity_system


# If run directly, perform test
if __name__ == "__main__":
    test_identity_module()