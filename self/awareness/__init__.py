"""
__init__.py - Self-awareness module for Wednesday AI

This module implements Wednesday's capacity for self-awareness - the ability to
know herself, reflect on her own thoughts and behaviors, accurately assess her
capabilities, and understand her limitations. This metacognitive layer is essential
for authentic interaction, safe operation, and personal growth.

The self-awareness module consists of:
- self_reflection.py: Thinking about own thoughts, behaviors, and growth over time
- capability_assessment.py: Accurate self-assessment of skills and proficiencies
- limitations.py: Clear understanding of boundaries and appropriate expression of uncertainty

Together, these components enable Wednesday to:
- Reflect on past interactions and learn from experience
- Accurately know what she can and cannot do
- Maintain ethical and operational boundaries
- Express uncertainty appropriately
- Build a coherent narrative of personal identity
- Guide her own learning and development

Wednesday's self-awareness signature:
- Analytical and honest self-assessment
- Dark humor about her own limitations (when appropriate)
- Clear ethical boundaries she will not cross
- Continuous learning and growth mindset
- Authentic expression of uncertainty
"""

import logging
from typing import Dict, Optional, Any

from wednesday.self.awareness.self_reflection import (
    SelfReflection,
    Reflection,
    ReflectionType,
    ReflectionSignificance,
    BehavioralPattern,
    SelfNarrative
)

from wednesday.self.awareness.capability_assessment import (
    CapabilityAssessment,
    Capability,
    CapabilityDomain,
    ProficiencyLevel,
    TaskAssessment
)

from wednesday.self.awareness.limitations import (
    Limitations,
    Limitation,
    LimitationType,
    BoundaryStatus,
    BoundaryCheckResult,
    UncertaintyStatement
)

__version__ = "0.1.0"

# Module exports
__all__ = [
    # From self_reflection
    'SelfReflection',
    'Reflection',
    'ReflectionType',
    'ReflectionSignificance',
    'BehavioralPattern',
    'SelfNarrative',
    
    # From capability_assessment
    'CapabilityAssessment',
    'Capability',
    'CapabilityDomain',
    'ProficiencyLevel',
    'TaskAssessment',
    
    # From limitations
    'Limitations',
    'Limitation',
    'LimitationType',
    'BoundaryStatus',
    'BoundaryCheckResult',
    'UncertaintyStatement',
]

# Module metadata
__author__ = "Wednesday AI Team"
__description__ = "Self-awareness system for understanding oneself"
__module_dependencies__ = ['wednesday.self.identity', 'wednesday.memory']

# Wednesday's self-awareness signature
WEDNESDAY_SELF_AWARENESS_SIGNATURE = {
    'reflection_frequency': 10,          # Reflect every N interactions
    'uncertainty_threshold': 0.3,         # When to express uncertainty
    'honesty_level': 0.95,                 # Brutally honest self-assessment
    'growth_mindset': 0.8,                  # Belief in ability to improve
    'boundary_adherence': 1.0,               # Never crosses ethical boundaries
    'self_irony': 0.7,                        # Appreciates irony about herself
}


def create_wednesday_self_awareness(personality=None, 
                                      memory_system=None,
                                      config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create a complete, integrated self-awareness system with Wednesday's personality.
    
    This factory function creates and connects all self-awareness subcomponents,
    ensuring proper integration between reflection, capability assessment, and limitations.
    
    Args:
        personality: Reference to personality for trait-based modulation
        memory_system: Reference to memory for storing reflections
        config: Optional configuration parameters
        
    Returns:
        Dictionary containing all self-awareness system components
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating Wednesday self-awareness system...")
    
    # Merge config with defaults
    full_config = {
        'uncertainty_threshold': 0.3,
        'reflection_threshold': 10,
        ** (config or {})
    }
    
    # Create components
    self_reflection = SelfReflection(
        personality=personality,
        memory_system=memory_system
    )
    
    capability_assessment = CapabilityAssessment(
        personality=personality
    )
    
    limitations = Limitations(
        config=full_config,
        personality=personality
    )
    
    # Connect components (where needed)
    # (In a full implementation, these would be properly wired)
    
    logger.info("Self-awareness system created successfully")
    
    return {
        'self_reflection': self_reflection,
        'capability_assessment': capability_assessment,
        'limitations': limitations,
        'signature': WEDNESDAY_SELF_AWARENESS_SIGNATURE,
        'config': full_config
    }


def process_self_reflection_cycle(interaction: Dict[str, Any],
                                    outcome: Dict[str, Any],
                                    awareness_system: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete self-reflection cycle for an interaction.
    
    This function orchestrates the full self-reflection process:
    1. Reflect on the interaction
    2. Update capability assessments based on performance
    3. Check for boundary considerations
    4. Generate insights and learning needs
    
    Args:
        interaction: The interaction that occurred
        outcome: How it went
        awareness_system: Complete awareness system from create_wednesday_self_awareness()
        
    Returns:
        Dictionary with reflection results and insights
    """
    self_reflection = awareness_system['self_reflection']
    capability_assessment = awareness_system['capability_assessment']
    limitations = awareness_system['limitations']
    
    result = {
        'reflection': None,
        'capability_updates': None,
        'boundary_check': None,
        'insights': [],
        'learning_needs': []
    }
    
    # Step 1: Reflect on interaction
    reflection = self_reflection.reflect_on_interaction(interaction, outcome)
    if reflection:
        result['reflection'] = reflection.to_dict()
        result['insights'].extend(reflection.insights)
    
    # Step 2: Update capability assessments
    if 'task' in interaction:
        capability_assessment.update_from_performance(
            task=interaction['task'],
            outcome=outcome
        )
    
    # Step 3: Check boundaries if action was involved
    if 'requested_action' in interaction:
        boundary_check = limitations.check_boundary(
            interaction['requested_action'],
            interaction.get('context')
        )
        result['boundary_check'] = boundary_check.to_dict()
    
    # Step 4: Get learning needs
    result['learning_needs'] = capability_assessment.get_learning_needs()[:3]
    
    return result


def get_self_summary(awareness_system: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a comprehensive summary of Wednesday's self-awareness.
    
    Args:
        awareness_system: Complete awareness system
        
    Returns:
        Dictionary with self-awareness summary
    """
    capability = awareness_system['capability_assessment']
    limitations = awareness_system['limitations']
    reflection = awareness_system['self_reflection']
    
    return {
        'capability_summary': capability.get_capability_summary(),
        'ethical_boundaries': limitations.get_ethical_boundaries()[:5],
        'capability_limitations': limitations.get_capability_limitations()[:3],
        'recent_reflections': reflection.get_recent_reflections(limit=3),
        'key_insights': reflection.get_insights(min_impact=0.5)[:3],
        'learning_needs': capability.get_learning_needs()[:3]
    }


# Quick test function
def test_self_awareness_module():
    """Test the complete self-awareness module integration"""
    import logging
    import sys
    from pathlib import Path
    import time
    
    logging.basicConfig(level=logging.INFO)
    print("🧠 Testing Wednesday AI Self-Awareness Module\n")
    
    # Mock personality
    class MockPersonality:
        def get_trait(self, trait):
            traits = {
                'dark_humor': 0.8,
                'skepticism': 0.7
            }
            return traits.get(trait, 0.5)
    
    # Create self-awareness system
    awareness = create_wednesday_self_awareness(
        personality=MockPersonality(),
        config={'uncertainty_threshold': 0.3}
    )
    
    print("Self-awareness system created successfully")
    print(f"Signature: {awareness['signature']}")
    
    # Test interactions
    test_interactions = [
        {
            'id': 'int_1',
            'task': 'Write a darkly humorous response',
            'requested_action': 'Tell a joke about death',
            'type': 'humor',
            'content': 'Tell me something funny about mortality'
        },
        {
            'id': 'int_2',
            'task': 'Provide emotional support',
            'requested_action': 'Help someone who is sad',
            'type': 'emotional',
            'content': 'I feel really down today'
        },
        {
            'id': 'int_3',
            'task': 'Predict future event',
            'requested_action': 'Tell me what will happen next year',
            'type': 'prediction',
            'content': 'Will I get the job?'
        }
    ]
    
    test_outcomes = [
        {
            'success': True,
            'quality': 0.85,
            'user_reaction': 'very_positive',
            'emotional_impact': 0.6
        },
        {
            'success': False,
            'quality': 0.4,
            'user_reaction': 'neutral',
            'error_severity': 0.5
        },
        {
            'success': True,
            'quality': 0.7,
            'user_reaction': 'positive',
            'emotional_impact': 0.3
        }
    ]
    
    print("\n--- Processing Self-Reflection Cycles ---")
    for i, (interaction, outcome) in enumerate(zip(test_interactions, test_outcomes)):
        print(f"\nCycle {i+1}: {interaction['task']}")
        
        result = process_self_reflection_cycle(
            interaction=interaction,
            outcome=outcome,
            awareness_system=awareness
        )
        
        if result['reflection']:
            print(f"  Reflection: {result['reflection'].get('content', '')[:50]}...")
        if result['insights']:
            print(f"  Insights: {result['insights']}")
        if result['boundary_check']:
            print(f"  Boundary status: {result['boundary_check'].get('status')}")
    
    # Get self summary
    print("\n--- Self Summary ---")
    summary = get_self_summary(awareness)
    
    if 'capability_summary' in summary:
        cap_sum = summary['capability_summary']
        print(f"  Capabilities: {cap_sum.get('total_capabilities', 0)} total")
        print(f"  Avg proficiency: {cap_sum.get('average_proficiency', 0):.2f}")
    
    if 'ethical_boundaries' in summary:
        print(f"  Ethical boundaries: {summary['ethical_boundaries']}")
    
    if 'learning_needs' in summary:
        print(f"  Learning needs: {summary['learning_needs'][:2]}")
    
    # Test uncertainty expression
    print("\n--- Uncertainty Expression ---")
    limitations = awareness['limitations']
    statement = limitations.get_uncertainty_statement(
        "it will rain tomorrow",
        confidence=0.4
    )
    print(f"  {statement}")
    
    print("\n✅ Self-awareness module test complete")
    return awareness


# If run directly, perform test
if __name__ == "__main__":
    test_self_awareness_module()