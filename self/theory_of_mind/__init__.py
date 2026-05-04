"""
__init__.py - Theory of Mind module for Wednesday AI

This module implements Wednesday's Theory of Mind - her ability to understand
that others have their own minds, with unique beliefs, intentions, and
perspectives. This is a cornerstone of social intelligence, enabling genuine
understanding and personalized interaction.

The Theory of Mind module consists of:
- user_model.py: Persistent mental models of individual users
- belief_management.py: Tracking what users believe about the world
- intention_reader.py: Inferring what users are trying to accomplish

Together, these components enable Wednesday to:
- Build deep, personalized understanding of each user
- Track how user beliefs evolve over time
- Infer both stated and hidden intentions
- Predict user behavior and needs
- Maintain coherent models across interactions
- Understand when users have false or incomplete beliefs

Wednesday's Theory of Mind signature:
- Analytical and pattern-based understanding
- Recognizes manipulation and hidden agendas
- Appreciates clever users who challenge her
- Builds trust through consistency and accuracy
- Dark humor about human cognitive biases
"""

import logging
from typing import Dict, Optional, Any

from wednesday.self.theory_of_mind.user_model import (
    UserModel,
    UserProfile,
    UserPersonality,
    UserPreferences,
    UserRelationship,
    CommunicationStyle,
    InteractionHistory
)

from wednesday.self.theory_of_mind.belief_management import (
    BeliefManagement,
    Belief,
    BeliefType,
    BeliefConflict,
    BeliefConfidence
)

from wednesday.self.theory_of_mind.intention_reader import (
    IntentionReader,
    Intention,
    IntentionType,
    IntentionConfidence,
    GoalHierarchy
)

__version__ = "0.1.0"

# Module exports
__all__ = [
    # From user_model
    'UserModel',
    'UserProfile',
    'UserPersonality',
    'UserPreferences',
    'UserRelationship',
    'CommunicationStyle',
    'InteractionHistory',
    
    # From belief_management
    'BeliefManagement',
    'Belief',
    'BeliefType',
    'BeliefConflict',
    'BeliefConfidence',
    
    # From intention_reader
    'IntentionReader',
    'Intention',
    'IntentionType',
    'IntentionConfidence',
    'GoalHierarchy',
]

# Module metadata
__author__ = "Wednesday AI Team"
__description__ = "Theory of Mind system for understanding others' minds"
__module_dependencies__ = ['wednesday.memory', 'wednesday.self.identity']

# Wednesday's Theory of Mind signature
WEDNESDAY_THEORY_OF_MIND_SIGNATURE = {
    'inference_accuracy': 0.85,           # Generally accurate in inferences
    'skepticism_level': 0.7,                # Questions easy inferences
    'pattern_recognition': 0.8,              # Good at behavioral patterns
    'trust_building_rate': 0.05,              # Builds trust gradually
    'manipulation_detection': 0.8,            # Good at detecting manipulation
    'belief_update_rate': 0.1,                 # Updates beliefs appropriately
    'max_users_tracked': 1000,                  # Scales to many users
}


def create_wednesday_theory_of_mind(memory_system=None,
                                      personality=None,
                                      config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create a complete, integrated Theory of Mind system with Wednesday's personality.
    
    This factory function creates and connects all Theory of Mind subcomponents,
    ensuring proper integration between user modeling, belief management, and
    intention reading.
    
    Args:
        memory_system: Reference to memory for persistent storage
        personality: Reference to Wednesday's personality for bias
        config: Optional configuration parameters
        
    Returns:
        Dictionary containing all Theory of Mind system components
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating Wednesday Theory of Mind system...")
    
    # Create components in dependency order
    user_model = UserModel(
        memory_system=memory_system,
        personality=personality
    )
    
    belief_management = BeliefManagement(
        user_model=user_model
    )
    
    intention_reader = IntentionReader(
        user_model=user_model,
        belief_management=belief_management,
        personality=personality
    )
    
    # Connect components (bidirectional where needed)
    # belief_management already has user_model reference
    # intention_reader has both references
    
    logger.info("Theory of Mind system created successfully")
    
    return {
        'user_model': user_model,
        'belief_management': belief_management,
        'intention_reader': intention_reader,
        'signature': WEDNESDAY_THEORY_OF_MIND_SIGNATURE,
        'config': config or {}
    }


def understand_user(user_id: str,
                     utterance: str,
                     tom_system: Dict[str, Any],
                     context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Complete Theory of Mind processing for a user utterance.
    
    This function orchestrates the full Theory of Mind pipeline:
    1. Update user model with interaction
    2. Infer and update user beliefs
    3. Read user's current intention
    
    Args:
        user_id: User identifier
        utterance: What the user said
        tom_system: Complete Theory of Mind system from create_wednesday_theory_of_mind()
        context: Current conversation context
        
    Returns:
        Dictionary with comprehensive user understanding
    """
    user_model = tom_system['user_model']
    belief_management = tom_system['belief_management']
    intention_reader = tom_system['intention_reader']
    
    # Step 1: Update user model with this interaction
    interaction_data = {
        'content': utterance,
        'type': context.get('interaction_type', 'unknown') if context else 'unknown',
        'topics': context.get('topics', []) if context else [],
        'timestamp': __import__('time').time()
    }
    
    updated_profile = user_model.update_from_interaction(user_id, interaction_data)
    
    # Step 2: Infer and update beliefs
    inferred_belief = belief_management.infer_belief_from_statement(
        user_id=user_id,
        statement=utterance,
        context=context
    )
    
    # Check for belief conflicts with new information
    if context and 'new_information' in context:
        conflicts = belief_management.detect_belief_conflict(
            user_id=user_id,
            new_information=context['new_information']
        )
    else:
        conflicts = []
    
    # Step 3: Read user's intention
    intention = intention_reader.infer_intention(
        user_id=user_id,
        utterance=utterance,
        context=context
    )
    
    # Step 4: Predict next actions
    predictions = intention_reader.predict_next_action(
        user_id=user_id,
        current_state=context or {}
    )
    
    return {
        'user_id': user_id,
        'user_profile': updated_profile.to_dict(),
        'inferred_belief': inferred_belief.to_dict() if inferred_belief else None,
        'belief_conflicts': [c.to_dict() for c in conflicts],
        'intention': intention.to_dict(),
        'next_action_predictions': predictions,
        'goal_hierarchy': intention_reader.get_goal_hierarchy(user_id).to_dict()
    }


def get_user_understanding_summary(user_id: str,
                                    tom_system: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a comprehensive summary of Wednesday's understanding of a user.
    
    Args:
        user_id: User identifier
        tom_system: Complete Theory of Mind system
        
    Returns:
        Dictionary with understanding summary
    """
    user_model = tom_system['user_model']
    belief_management = tom_system['belief_management']
    intention_reader = tom_system['intention_reader']
    
    # Get user profile summary
    profile_summary = user_model.get_user_summary(user_id)
    
    # Get belief summary
    belief_summary = belief_management.get_belief_summary(user_id)
    
    # Get goal hierarchy
    goal_hierarchy = intention_reader.get_goal_hierarchy(user_id)
    
    # Get intention history
    intention_history = intention_reader.get_intention_history(user_id, limit=5)
    
    return {
        'user_id': user_id,
        'profile': profile_summary.get('profile', {}),
        'summary_text': profile_summary.get('summary', ''),
        'beliefs': belief_summary,
        'current_goals': goal_hierarchy.to_dict(),
        'recent_intentions': intention_history,
        'understanding_confidence': _calculate_understanding_confidence(
            profile_summary, belief_summary, goal_hierarchy
        )
    }


def _calculate_understanding_confidence(profile_summary: Dict,
                                         belief_summary: Dict,
                                         goal_hierarchy: GoalHierarchy) -> float:
    """Calculate confidence in overall user understanding"""
    confidence = 0.5  # Base
    
    # More interactions = higher confidence
    if 'profile' in profile_summary:
        interaction_count = profile_summary['profile'].get('interaction_count', 0)
        confidence += min(0.3, interaction_count / 100)
    
    # More beliefs tracked = higher confidence
    if 'total_beliefs' in belief_summary:
        confidence += min(0.2, belief_summary['total_beliefs'] / 20)
    
    # Having current intention = higher confidence
    if goal_hierarchy.current_intention:
        confidence += 0.1
    
    return min(1.0, confidence)


# Quick test function
def test_theory_of_mind_module():
    """Test the complete Theory of Mind module integration"""
    import logging
    import sys
    from pathlib import Path
    import time
    
    logging.basicConfig(level=logging.INFO)
    print("🧠 Testing Wednesday AI Theory of Mind Module\n")
    
    # Mock personality
    class MockPersonality:
        def get_trait(self, trait):
            traits = {
                'skepticism': 0.7,
                'dark_humor': 0.8
            }
            return traits.get(trait, 0.5)
    
    # Create Theory of Mind system
    tom_system = create_wednesday_theory_of_mind(
        memory_system=None,
        personality=MockPersonality()
    )
    
    print("Theory of Mind system created successfully")
    print(f"Signature: {tom_system['signature']}")
    
    # Test user
    user_id = "test_user_tom"
    
    # Test interactions
    test_interactions = [
        {
            'utterance': "Hi Wednesday! I love dark humor and mysteries.",
            'context': {
                'interaction_type': 'greeting',
                'topics': ['greeting', 'humor']
            }
        },
        {
            'utterance': "I think vaccines are dangerous.",
            'context': {
                'interaction_type': 'statement',
                'topics': ['health', 'science'],
                'new_information': {
                    'vaccines': "Vaccines are extensively tested and safe"
                }
            }
        },
        {
            'utterance': "What do you think about death? I find it fascinating.",
            'context': {
                'interaction_type': 'question',
                'topics': ['philosophy', 'death']
            }
        },
        {
            'utterance': "Can you help me solve a mystery?",
            'context': {
                'interaction_type': 'request',
                'topics': ['mystery', 'help']
            }
        }
    ]
    
    print("\n--- Processing Interactions ---")
    for i, interaction in enumerate(test_interactions):
        print(f"\nInteraction {i+1}: '{interaction['utterance']}'")
        
        understanding = understand_user(
            user_id=user_id,
            utterance=interaction['utterance'],
            tom_system=tom_system,
            context=interaction['context']
        )
        
        if understanding['intention']:
            print(f"  Intention: {understanding['intention']['type']} "
                  f"(confidence: {understanding['intention']['confidence']:.2f})")
        
        if understanding['inferred_belief']:
            print(f"  Inferred belief: {understanding['inferred_belief']['topic']}")
        
        if understanding['belief_conflicts']:
            print(f"  Belief conflicts: {len(understanding['belief_conflicts'])}")
    
    # Get understanding summary
    print("\n--- User Understanding Summary ---")
    summary = get_user_understanding_summary(user_id, tom_system)
    
    print(f"  Summary: {summary['summary_text']}")
    print(f"  Understanding confidence: {summary['understanding_confidence']:.2f}")
    
    if 'beliefs' in summary:
        print(f"  Total beliefs tracked: {summary['beliefs'].get('total_beliefs', 0)}")
    
    if 'current_goals' in summary:
        current = summary['current_goals'].get('current_intention')
        if current:
            print(f"  Current intention: {current['type']}")
    
    print("\n✅ Theory of Mind module test complete")
    return tom_system


# If run directly, perform test
if __name__ == "__main__":
    test_theory_of_mind_module()