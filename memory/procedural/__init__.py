"""
Procedural Memory Package - How to do things.
Stores skills, muscle memory, and learned behaviors.
Like knowing how to ride a bike or make a witty comeback.
"""
# Deferred imports to avoid package context issues when running as module
# from .skills import SkillLibrary, Skill, SkillLevel, SkillType
# from .muscle_memory import MuscleMemorySystem, MuscleMemoryPattern, MuscleMemoryType, ActivationTrigger
# from .learned_behaviors import BehaviorLibrary, LearnedBehavior, BehaviorContext, BehaviorComplexity

__all__ = [
    # Main classes
    'SkillLibrary',
    'MuscleMemorySystem',
    'BehaviorLibrary',
    
    # Skill related
    'Skill',
    'SkillLevel',
    'SkillType',
    
    # Muscle memory related
    'MuscleMemoryPattern',
    'MuscleMemoryType',
    'ActivationTrigger',
    
    # Behavior related
    'LearnedBehavior',
    'BehaviorContext',
    'BehaviorComplexity'
]