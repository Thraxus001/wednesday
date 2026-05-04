"""
__init__.py - Metacognition module for Wednesday AI

This module implements Wednesday's capacity for metacognition - thinking about
her own thinking. This higher-order cognitive ability enables self-awareness,
self-regulation, and continuous improvement of her cognitive processes.

The metacognition module consists of:
- thought_monitor.py: Real-time observation of thought processes
- confidence_scoring.py: Self-assessment of confidence in outputs
- cognitive_control.py: Executive control over thinking processes

Together, these components enable Wednesday to:
- Observe and log her own thoughts as they occur
- Assess how confident she is in her responses
- Regulate her thinking speed and depth based on task demands
- Detect when she's ruminating or stuck
- Calibrate confidence based on feedback
- Allocate cognitive resources appropriately
- Handle interruptions and task switching

Wednesday's metacognitive style:
- Analytical observation of her own mind
- Honest self-assessment (with dark humor)
- Preference for depth over speed when appropriate
- Awareness of cognitive limitations
- Dark humor about overthinking
"""

import logging
from typing import Dict, Optional, Any

from wednesday.self.metacognition.thought_monitor import (
    ThoughtMonitor,
    Thought,
    ThoughtCategory,
    ThoughtImportance,
    AttentionFocus
)

from wednesday.self.metacognition.confidence_scoring import (
    ConfidenceScorer,
    ConfidenceScore,
    ConfidenceLevel,
    CalibrationRecord
)

from wednesday.self.metacognition.cognitive_control import (
    CognitiveControl,
    ControlSettings,
    ProcessingMode,
    TaskPriority,
    ResourceAllocation,
    InterruptionDecision,
    InterruptionRecord
)

__version__ = "0.1.0"

# Module exports
__all__ = [
    # From thought_monitor
    'ThoughtMonitor',
    'Thought',
    'ThoughtCategory',
    'ThoughtImportance',
    'AttentionFocus',
    
    # From confidence_scoring
    'ConfidenceScorer',
    'ConfidenceScore',
    'ConfidenceLevel',
    'CalibrationRecord',
    
    # From cognitive_control
    'CognitiveControl',
    'ControlSettings',
    'ProcessingMode',
    'TaskPriority',
    'ResourceAllocation',
    'InterruptionDecision',
    'InterruptionRecord',
]

# Module metadata
__author__ = "Wednesday AI Team"
__description__ = "Metacognition system for thinking about thinking"
__module_dependencies__ = ['wednesday.self.awareness', 'wednesday.executive']

# Wednesday's metacognitive signature
WEDNESDAY_METACOGNITIVE_SIGNATURE = {
    'default_processing_mode': 'normal',          # Balanced by default
    'depth_preference': 0.7,                       # Prefers deeper thinking
    'confidence_calibration': 0.85,                # Well-calibrated confidence
    'rumination_threshold': 3,                      # Notices repetitive thoughts
    'cognitive_load_tolerance': 0.8,                # Manages load well
    'interruption_handling': 'priority_based',      # Handles interruptions strategically
}


def create_wednesday_metacognition(capability_assessment,
                                     personality=None,
                                     config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create a complete, integrated metacognition system with Wednesday's style.
    
    This factory function creates and connects all metacognition subcomponents,
    ensuring proper integration between thought monitoring, confidence scoring,
    and cognitive control.
    
    Args:
        capability_assessment: Reference to capability assessment system
        personality: Reference to Wednesday's personality for expression
        config: Optional configuration parameters
        
    Returns:
        Dictionary containing all metacognition system components
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating Wednesday metacognition system...")
    
    # Create components in dependency order
    thought_monitor = ThoughtMonitor(
        config=config.get('thought_monitor') if config else None
    )
    
    confidence_scorer = ConfidenceScorer(
        capability_assessment=capability_assessment,
        personality=personality,
        config=config.get('confidence_scorer') if config else None
    )
    
    cognitive_control = CognitiveControl(
        thought_monitor=thought_monitor,
        confidence_scorer=confidence_scorer,
        personality=personality
    )
    
    logger.info("Metacognition system created successfully")
    
    return {
        'thought_monitor': thought_monitor,
        'confidence_scorer': confidence_scorer,
        'cognitive_control': cognitive_control,
        'signature': WEDNESDAY_METACOGNITIVE_SIGNATURE,
        'config': config or {}
    }


def process_metacognitive_cycle(task: str,
                                  metacognition_system: Dict[str, Any],
                                  priority: TaskPriority = TaskPriority.NORMAL,
                                  complexity: float = 0.5,
                                  time_available: Optional[float] = None) -> Dict[str, Any]:
    """
    Complete metacognitive processing cycle for a task.
    
    This function orchestrates the full metacognitive pipeline:
    1. Allocate cognitive resources based on task characteristics
    2. Monitor thoughts during processing (ongoing)
    3. Score confidence after completion
    4. Record performance for calibration
    
    Args:
        task: Description of the task
        metacognition_system: Complete metacognition system
        priority: Task priority
        complexity: Task complexity (0-1)
        time_available: Time available in seconds
        
    Returns:
        Dictionary with metacognitive processing results
    """
    thought_monitor = metacognition_system['thought_monitor']
    confidence_scorer = metacognition_system['confidence_scorer']
    cognitive_control = metacognition_system['cognitive_control']
    
    # Step 1: Allocate resources
    allocation = cognitive_control.allocate_resources(
        task=task,
        priority=priority,
        complexity=complexity,
        time_available=time_available
    )
    
    # Step 2: Log the start of processing
    thought_monitor.log_thought(
        content=f"Processing task: {task[:50]}",
        category=ThoughtCategory.PLANNING,
        importance=ThoughtImportance.MODERATE,
        intensity=allocation.allocated_depth
    )
    
    # Step 3: Adjust control settings for task difficulty
    cognitive_control.adjust_for_difficulty(complexity)
    
    # Step 4: Start task in cognitive control
    cognitive_control.start_task(task, allocation)
    
    # Note: Actual task execution would happen here
    # This is a placeholder - in production, the task would be executed
    
    # Step 5: Score confidence after processing
    confidence_score = confidence_scorer.score_confidence(
        task=task,
        context={'complexity': complexity, 'priority': priority.value}
    )
    
    # Step 6: Complete task in cognitive control
    cognitive_control.complete_task(
        success=True,
        confidence_achieved=confidence_score.overall_confidence
    )
    
    # Step 7: Check for rumination
    rumination = thought_monitor.detect_rumination()
    
    # Step 8: Get current focus
    focus = thought_monitor.get_current_focus()
    
    return {
        'allocation': allocation.to_dict(),
        'confidence_score': confidence_score.to_dict(),
        'rumination_detected': rumination is not None,
        'rumination_details': rumination,
        'current_focus': focus.to_dict(),
        'control_settings': cognitive_control.get_current_settings().to_dict(),
        'should_hedge': confidence_score.should_hedge(),
        'hedging_phrase': confidence_scorer.get_hedging_phrase(
            confidence_score, 
            "my assessment of this task",
            use_wednesday_style=True
        ) if confidence_score.should_hedge() else None
    }


def get_metacognitive_status(metacognition_system: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get comprehensive status of metacognitive state.
    
    Args:
        metacognition_system: Complete metacognition system
        
    Returns:
        Dictionary with metacognitive status
    """
    thought_monitor = metacognition_system['thought_monitor']
    confidence_scorer = metacognition_system['confidence_scorer']
    cognitive_control = metacognition_system['cognitive_control']
    
    return {
        'thought_statistics': thought_monitor.get_thought_statistics(),
        'attention_allocation': thought_monitor.get_attention_allocation(),
        'calibration_metrics': confidence_scorer.get_calibration_metrics(),
        'confidence_trend': confidence_scorer.get_confidence_trend(),
        'control_settings': cognitive_control.get_current_settings().to_dict(),
        'performance_stats': cognitive_control.get_performance_stats(),
        'current_rumination': thought_monitor.detect_rumination()
    }


# Quick test function
def test_metacognition_module():
    """Test the complete metacognition module integration"""
    import logging
    import sys
    from pathlib import Path
    import time
    
    logging.basicConfig(level=logging.INFO)
    print("🧠 Testing Wednesday AI Metacognition Module\n")
    
    # Mock capability assessment
    class MockCapabilityAssessment:
        def assess_capability(self, task, context=None):
            class Result:
                def __init__(self):
                    self.estimated_proficiency = 0.7 if 'humor' in task.lower() else 0.5
            return Result()
        
        def _score_performance_history(self, task):
            return 0.65
    
    # Create metacognition system
    metacognition = create_wednesday_metacognition(
        capability_assessment=MockCapabilityAssessment(),
        config={
            'thought_monitor': {},
            'confidence_scorer': {'weights': {'capability': 0.4, 'information': 0.3, 'context': 0.1, 'performance': 0.2}}
        }
    )
    
    print("Metacognition system created successfully")
    print(f"Signature: {metacognition['signature']}")
    
    # Test different tasks
    test_tasks = [
        ("Respond with dark humor", TaskPriority.NORMAL, 0.5),
        ("Solve complex logical puzzle", TaskPriority.HIGH, 0.9),
        ("Engage in small talk", TaskPriority.LOW, 0.2),
        ("Handle user emotional support request", TaskPriority.HIGH, 0.6),
    ]
    
    print("\n--- Processing Tasks ---")
    for task, priority, complexity in test_tasks:
        print(f"\nTask: {task}")
        
        result = process_metacognitive_cycle(
            task=task,
            metacognition_system=metacognition,
            priority=priority,
            complexity=complexity,
            time_available=3.0
        )
        
        print(f"  Processing mode: {result['allocation']['mode']}")
        print(f"  Confidence: {result['confidence_score']['overall']:.2f} ({result['confidence_score']['level']})")
        if result['should_hedge']:
            print(f"  Hedge: {result['hedging_phrase'][:60]}...")
        if result['rumination_detected']:
            print(f"  ⚠️ Rumination: {result['rumination_details']['suggestion']}")
        
        time.sleep(0.1)
    
    # Get metacognitive status
    print("\n--- Metacognitive Status ---")
    status = get_metacognitive_status(metacognition)
    
    print(f"Thinking speed: {status['thought_statistics']['thinking_speed']:.2f}")
    print(f"Cognitive load: {status['control_settings']['cognitive_load']:.2f}")
    print(f"Processing mode: {status['control_settings']['mode']}")
    
    if status['calibration_metrics'].get('has_data', False):
        print(f"Calibration bias: {status['calibration_metrics']['calibration_bias']:.3f}")
    
    print(f"Total tasks: {status['performance_stats']['total_tasks']}")
    print(f"Success rate: {status['performance_stats']['success_rate']:.2%}")
    
    print("\n--- Thought Category Distribution ---")
    for cat, count in status['thought_statistics']['category_distribution'].items():
        if count > 0:
            print(f"  {cat}: {count}")
    
    print("\n✅ Metacognition module test complete")
    return metacognition


# If run directly, perform test
if __name__ == "__main__":
    test_metacognition_module()