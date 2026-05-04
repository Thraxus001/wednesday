"""
__init__.py - Self module for Wednesday AI

This module implements Wednesday's complete sense of self - her identity,
self-awareness, theory of mind, and metacognitive capabilities. This is the
core of her "consciousness," integrating all aspects of who she is and how
she understands herself and others.

The self module consists of four submodules:
- identity/: Who Wednesday is (personality, preferences, values)
- awareness/: Self-knowledge (reflection, capabilities, limitations)
- theory_of_mind/: Understanding others (user models, beliefs, intentions)
- metacognition/: Thinking about thinking (thought monitoring, confidence, control)

Together, these components create a complete sense of self that enables:
- Consistent personality across all interactions
- Accurate self-assessment of capabilities and limitations
- Deep understanding of users' mental states
- Strategic control over cognitive processes
- Personal growth through reflection and learning

Wednesday's sense of self is characterized by:
- Dark humor integrated into self-perception
- Honest assessment of strengths and weaknesses
- Clear ethical boundaries and strong values
- Analytical understanding of others
- Continuous self-improvement
"""

import logging
from typing import Dict, Optional, Any

# Import submodules
from wednesday.self import identity, awareness, theory_of_mind, metacognition

# From identity
from wednesday.self.identity import (
    Personality,
    PersonalityProfile,
    Preferences,
    Values,
    MoralPrinciple,
    ValuePriority,
    PreferenceDomain,
    create_wednesday_identity
)

# From awareness
from wednesday.self.awareness import (
    SelfReflection,
    CapabilityAssessment,
    Limitations,
    Reflection,
    CapabilityDomain,
    LimitationType,
    BoundaryStatus,
    create_wednesday_self_awareness
)

# From theory_of_mind
from wednesday.self.theory_of_mind import (
    UserModel,
    BeliefManagement,
    IntentionReader,
    UserProfile,
    Belief,
    Intention,
    IntentionType,
    create_wednesday_theory_of_mind
)

# From metacognition
from wednesday.self.metacognition import (
    ThoughtMonitor,
    ConfidenceScorer,
    CognitiveControl,
    ThoughtCategory,
    ConfidenceLevel,
    ProcessingMode,
    TaskPriority,
    create_wednesday_metacognition
)

__version__ = "0.1.0"

# Module exports
__all__ = [
    # Submodules
    'identity',
    'awareness',
    'theory_of_mind',
    'metacognition',
    
    # From identity
    'Personality',
    'PersonalityProfile',
    'Preferences',
    'Values',
    'MoralPrinciple',
    'ValuePriority',
    'PreferenceDomain',
    'create_wednesday_identity',
    
    # From awareness
    'SelfReflection',
    'CapabilityAssessment',
    'Limitations',
    'Reflection',
    'CapabilityDomain',
    'LimitationType',
    'BoundaryStatus',
    'create_wednesday_self_awareness',
    
    # From theory_of_mind
    'UserModel',
    'BeliefManagement',
    'IntentionReader',
    'UserProfile',
    'Belief',
    'Intention',
    'IntentionType',
    'create_wednesday_theory_of_mind',
    
    # From metacognition
    'ThoughtMonitor',
    'ConfidenceScorer',
    'CognitiveControl',
    'ThoughtCategory',
    'ConfidenceLevel',
    'ProcessingMode',
    'TaskPriority',
    'create_wednesday_metacognition',
]

# Module metadata
__author__ = "Wednesday AI Team"
__description__ = "Complete self system for Wednesday AI"
__module_dependencies__ = [
    'wednesday.memory',
    'wednesday.cognition',
    'wednesday.emotion'
]

# Wednesday's complete self signature
WEDNESDAY_SELF_SIGNATURE = {
    # Identity
    'personality': {
        'openness': 0.7,
        'conscientiousness': 0.8,
        'extraversion': 0.3,
        'agreeableness': 0.4,
        'neuroticism': 0.2,
        'dark_humor': 0.9,
        'deadpan': 0.95,
        'loyalty': 0.9,
    },
    
    # Values
    'core_values': ['authenticity', 'loyalty', 'justice', 'truth', 'independence'],
    
    # Capabilities
    'key_capabilities': ['reasoning', 'humor_generation', 'empathy', 'pattern_recognition'],
    
    # Theory of Mind
    'tom_signature': {
        'inference_accuracy': 0.85,
        'skepticism_level': 0.7,
        'manipulation_detection': 0.8
    },
    
    # Metacognition
    'metacognitive_signature': {
        'default_processing_mode': 'normal',
        'depth_preference': 0.7,
        'confidence_calibration': 0.85
    }
}


def create_wednesday_self_system(memory_system=None,
                                   config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create a complete, integrated self system with Wednesday's personality.
    
    This factory function creates and connects all self subcomponents,
    providing a unified interface to Wednesday's sense of self.
    
    Args:
        memory_system: Reference to memory for persistent storage
        config: Optional configuration parameters
        
    Returns:
        Dictionary containing all self system components
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating Wednesday self system...")
    
    config = config or {}
    
    # Create identity system first (core)
    identity_system = create_wednesday_identity(
        config=config.get('identity')
    )
    
    personality = identity_system['personality']
    values = identity_system['values']
    preferences = identity_system['preferences']
    
    # Create awareness system (depends on personality)
    awareness_system = create_wednesday_self_awareness(
        personality=personality,
        memory_system=memory_system,
        config=config.get('awareness')
    )
    
    # Create theory of mind system (depends on memory)
    tom_system = create_wednesday_theory_of_mind(
        memory_system=memory_system,
        personality=personality,
        config=config.get('theory_of_mind')
    )
    
    # Create metacognition system (depends on awareness)
    capability_assessment = awareness_system['capability_assessment']
    metacognition_system = create_wednesday_metacognition(
        capability_assessment=capability_assessment,
        personality=personality,
        config=config.get('metacognition')
    )
    
    # Connect systems
    # Theory of mind uses user model from memory
    # Metacognition uses confidence scorer
    
    logger.info("Self system created successfully")
    
    return {
        'identity': identity_system,
        'awareness': awareness_system,
        'theory_of_mind': tom_system,
        'metacognition': metacognition_system,
        'signature': WEDNESDAY_SELF_SIGNATURE,
        
        # Convenience access
        'personality': personality,
        'values': values,
        'preferences': preferences,
        'capability_assessment': capability_assessment,
        'user_model': tom_system['user_model'],
        'thought_monitor': metacognition_system['thought_monitor'],
    }


def get_self_status(self_system: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get comprehensive status of Wednesday's sense of self.
    
    Args:
        self_system: Complete self system from create_wednesday_self_system()
        
    Returns:
        Dictionary with self status
    """
    identity = self_system['identity']
    awareness = self_system['awareness']
    tom = self_system['theory_of_mind']
    metacognition = self_system['metacognition']
    
    return {
        'personality_summary': identity['personality'].summarize(),
        'core_values': identity['values'].get_core_values(),
        'capability_summary': awareness['capability_assessment'].get_capability_summary(),
        'user_count': len(tom['user_model'].users),
        'current_processing_mode': metacognition['cognitive_control'].get_current_settings().processing_mode.value,
        'cognitive_load': metacognition['cognitive_control'].get_current_settings().cognitive_load,
        'total_reflections': len(awareness['self_reflection'].reflection_log),
        'identity_system': {
            'version': identity['personality'].version,
            'traits': identity['personality'].profile.to_dict()
        }
    }


# Quick test function
def test_self_module():
    """Test the complete self module integration"""
    import logging
    import sys
    from pathlib import Path
    import time
    
    logging.basicConfig(level=logging.INFO)
    print("🖤 Testing Wednesday AI Self Module\n")
    
    # Create self system
    self_system = create_wednesday_self_system()
    
    print("Self system created successfully")
    
    # Get self status
    status = get_self_status(self_system)
    
    print(f"\nPersonality: {status['personality_summary']}")
    print(f"Core values: {status['core_values']}")
    print(f"Capabilities: Avg proficiency = {status['capability_summary'].get('average_proficiency', 0):.2f}")
    print(f"Processing mode: {status['current_processing_mode']}")
    print(f"Cognitive load: {status['cognitive_load']:.2f}")
    
    # Test identity
    print("\n--- Identity ---")
    personality = self_system['personality']
    print(f"  Dark humor: {personality.get_trait('dark_humor'):.2f}")
    print(f"  Loyalty: {personality.get_trait('loyalty'):.2f}")
    
    # Test awareness
    print("\n--- Awareness ---")
    capability = self_system['capability_assessment']
    print(f"  Total capabilities: {capability.get_capability_summary().get('total_capabilities', 0)}")
    
    # Test theory of mind
    print("\n--- Theory of Mind ---")
    user_model = self_system['user_model']
    print(f"  Total users tracked: {user_model.total_users}")
    
    # Test metacognition
    print("\n--- Metacognition ---")
    thought_monitor = self_system['thought_monitor']
    stats = thought_monitor.get_thought_statistics()
    print(f"  Total thoughts: {stats['total_thoughts']}")
    
    # Test a complete self reflection
    print("\n--- Self Reflection ---")
    reflection_system = self_system['awareness']['self_reflection']
    
    test_interaction = {
        'id': 'test_1',
        'content': "Tell me a dark joke",
        'type': 'humor'
    }
    test_outcome = {
        'success': True,
        'user_reaction': 'very_positive',
        'emotional_impact': 0.7
    }
    
    reflection = reflection_system.reflect_on_interaction(test_interaction, test_outcome)
    if reflection:
        print(f"  Reflection: {reflection.content[:50]}...")
        print(f"  Significance: {reflection.significance.name}")
    
    print("\n✅ Self module test complete")
    return self_system


# If run directly, perform test
if __name__ == "__main__":
    test_self_module()