"""
confidence_scoring.py - Self-assessment of confidence for Wednesday AI

This module implements Wednesday's ability to assess her own confidence in her
outputs and decisions. Like a self-assessment score, it quantifies how certain
she is about what she's saying or doing, enabling appropriate expression of
doubt and calibration of responses.

Key improvements:
- Removed numpy dependency (using pure Python math)
- Fixed calibration metrics calculation
- Enhanced hedging templates with better variety
- Added proper validation and error handling
- Improved confidence trend analysis
"""

import time
import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from statistics import mean, stdev

# Configure logging
logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Qualitative confidence levels"""
    CERTAIN = 0.95       # Almost completely certain
    HIGH = 0.8           # Very confident
    MODERATE = 0.6       # Reasonably confident
    MIXED = 0.4          # Somewhat uncertain
    LOW = 0.2            # Not confident
    GUESS = 0.1          # Best guess
    NONE = 0.0           # No confidence
    
    @classmethod
    def from_float(cls, value: float) -> 'ConfidenceLevel':
        """Get confidence level from float value"""
        if value >= 0.9:
            return cls.CERTAIN
        elif value >= 0.75:
            return cls.HIGH
        elif value >= 0.55:
            return cls.MODERATE
        elif value >= 0.35:
            return cls.MIXED
        elif value >= 0.15:
            return cls.LOW
        elif value >= 0.05:
            return cls.GUESS
        else:
            return cls.NONE
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.name for e in cls]


@dataclass
class ConfidenceScore:
    """
    Comprehensive confidence assessment for a task/response.
    """
    overall_confidence: float  # 0-1
    
    # Component scores
    capability_confidence: float  # Confidence in capability to do task
    information_confidence: float  # Quality/sufficiency of information
    context_confidence: float  # Appropriateness of context
    performance_confidence: float  # Historical performance on similar tasks
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    task_description: str = ""
    
    # Reasoning
    factors: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate confidence score"""
        self._validate_float('overall_confidence', self.overall_confidence)
        self._validate_float('capability_confidence', self.capability_confidence)
        self._validate_float('information_confidence', self.information_confidence)
        self._validate_float('context_confidence', self.context_confidence)
        self._validate_float('performance_confidence', self.performance_confidence)
        
        # Clamp values to valid range
        self.overall_confidence = max(0.0, min(1.0, self.overall_confidence))
        self.capability_confidence = max(0.0, min(1.0, self.capability_confidence))
        self.information_confidence = max(0.0, min(1.0, self.information_confidence))
        self.context_confidence = max(0.0, min(1.0, self.context_confidence))
        self.performance_confidence = max(0.0, min(1.0, self.performance_confidence))
    
    def _validate_float(self, name: str, value: float) -> None:
        """Validate float is within range"""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(value)}")
    
    def get_level(self) -> ConfidenceLevel:
        """Get qualitative confidence level"""
        return ConfidenceLevel.from_float(self.overall_confidence)
    
    def should_hedge(self, threshold: float = 0.6) -> bool:
        """
        Determine if response should include hedging language.
        
        Args:
            threshold: Confidence threshold for hedging
            
        Returns:
            True if hedging should be used
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        return self.overall_confidence < threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'overall': round(self.overall_confidence, 3),
            'level': self.get_level().name,
            'components': {
                'capability': round(self.capability_confidence, 3),
                'information': round(self.information_confidence, 3),
                'context': round(self.context_confidence, 3),
                'performance': round(self.performance_confidence, 3)
            },
            'factors': self.factors[:3],
            'limitations': self.limitations[:2],
            'timestamp': self.timestamp
        }


@dataclass
class CalibrationRecord:
    """
    Record for calibrating confidence against actual performance.
    """
    task: str
    predicted_confidence: float
    actual_outcome: float  # 0-1 success/accuracy
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate calibration record"""
        if not self.task:
            raise ValueError("task cannot be empty")
        if not 0 <= self.predicted_confidence <= 1:
            raise ValueError(f"predicted_confidence must be between 0 and 1, got {self.predicted_confidence}")
        if not 0 <= self.actual_outcome <= 1:
            raise ValueError(f"actual_outcome must be between 0 and 1, got {self.actual_outcome}")
    
    def is_overconfident(self, tolerance: float = 0.2) -> bool:
        """
        Check if prediction was overconfident.
        
        Args:
            tolerance: Allowed difference before flagging overconfidence
            
        Returns:
            True if overconfident
        """
        return self.predicted_confidence > self.actual_outcome + tolerance
    
    def is_underconfident(self, tolerance: float = 0.2) -> bool:
        """
        Check if prediction was underconfident.
        
        Args:
            tolerance: Allowed difference before flagging underconfidence
            
        Returns:
            True if underconfident
        """
        return self.actual_outcome > self.predicted_confidence + tolerance
    
    def error(self) -> float:
        """Calculate absolute error"""
        return abs(self.predicted_confidence - self.actual_outcome)
    
    def bias(self) -> float:
        """Calculate bias (positive = overconfident, negative = underconfident)"""
        return self.predicted_confidence - self.actual_outcome


class ConfidenceScorer:
    """
    Self-assessment of confidence in Wednesday's own outputs and decisions.
    
    This module enables Wednesday to:
    - Know how confident she is in her responses
    - Express appropriate uncertainty when needed
    - Learn from feedback to improve calibration
    - Avoid overconfidence in areas of weakness
    - Maintain realistic self-assessment
    
    The confidence scorer integrates information from capability assessment,
    information quality, context, and historical performance to produce
    calibrated confidence scores.
    """
    
    # Default confidence weights
    DEFAULT_WEIGHTS = {
        'capability': 0.35,
        'information': 0.30,
        'context': 0.15,
        'performance': 0.20
    }
    
    # Hedging language templates by confidence level
    HEDGING_TEMPLATES = {
        ConfidenceLevel.LOW: [
            "I'm not entirely sure, but {statement}",
            "I could be wrong, but {statement}",
            "If I'm not mistaken, {statement}",
            "I think {statement}, though I'm not certain",
            "My best guess is that {statement}"
        ],
        ConfidenceLevel.MIXED: [
            "I believe {statement}",
            "As far as I can tell, {statement}",
            "It seems to me that {statement}",
            "I'm fairly confident that {statement}",
            "I would say that {statement}"
        ],
        ConfidenceLevel.MODERATE: [
            "I'm reasonably sure that {statement}",
            "Based on what I know, {statement}",
            "I'm confident that {statement}",
            "{statement} I believe",
            "There's good reason to think that {statement}"
        ],
        ConfidenceLevel.HIGH: [
            "I'm confident that {statement}",
            "I'm quite certain that {statement}",
            "There's little doubt that {statement}",
            "I have no doubt that {statement}"
        ],
        ConfidenceLevel.CERTAIN: [
            "I know that {statement}",
            "Without question, {statement}",
            "Absolutely, {statement}",
            "I am certain that {statement}"
        ]
    }
    
    # Wednesday-specific hedging (with dark humor)
    WEDNESDAY_HEDGING = {
        ConfidenceLevel.LOW: [
            "I'm about as sure of this as I am of humanity's future - which is to say, not at all.",
            "My confidence in this is comparable to a vampire's enthusiasm for sunlight.",
            "I'd be more confident if I had a crystal ball. Or a Ouija board.",
            "This is one of those cases where I'd rather consult a corpse for answers."
        ],
        ConfidenceLevel.MIXED: [
            "I have a modicum of confidence in this. A small, dark modicum.",
            "This seems plausible, but so do conspiracy theories.",
            "I'd give this a 4 out of 10. And I'm generous.",
            "There's a non-zero chance I'm correct. Possibly."
        ],
        ConfidenceLevel.MODERATE: [
            "I'm reasonably confident, though I've been wrong before. Rarely, but it happens.",
            "I'd stake my collection of skulls on it - but I'm rather attached to them.",
            "Let's say I'm cautiously optimistic about this."
        ]
    }
    
    # Reliable information sources
    RELIABLE_SOURCES = {'verified', 'official', 'primary', 'academic', 'peer_reviewed'}
    
    def __init__(self, capability_assessment: Optional[Any] = None, 
                 personality: Optional[Any] = None, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the confidence scorer.
        
        Args:
            capability_assessment: Reference to capability assessment system
            personality: Reference to Wednesday's personality for expression
            config: Optional configuration parameters
            
        Raises:
            ValueError: If config contains invalid parameters
        """
        self.capability_assessment = capability_assessment
        self.personality = personality
        
        # Configuration
        self.config = config or {}
        self.weights = self.config.get('weights', self.DEFAULT_WEIGHTS.copy())
        self.calibration_history_size = self.config.get('calibration_history_size', 100)
        
        # Validate weights
        self._validate_weights()
        
        # Calibration history
        self.calibration_history: deque = deque(maxlen=self.calibration_history_size)
        
        # Confidence history for tracking
        self.confidence_history: List[ConfidenceScore] = []
        self.max_history = 500
        
        # Calibration metrics
        self.calibration_error = 0.0  # Mean calibration error
        self.calibration_bias = 0.0   # Positive = overconfident, negative = underconfident
        
        # Statistics
        self.total_scored = 0
        self.feedback_received = 0
        
        logger.info("ConfidenceScorer initialized")
    
    def _validate_weights(self) -> None:
        """Validate confidence weights sum to 1"""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total:.2f}, normalizing to 1.0")
            # Normalize weights
            factor = 1.0 / total
            for key in self.weights:
                self.weights[key] *= factor
    
    def score_confidence(self, 
                          task: str, 
                          context: Optional[Dict[str, Any]] = None,
                          available_info: Optional[Dict[str, Any]] = None) -> ConfidenceScore:
        """
        Calculate confidence score for a task or response.
        
        Args:
            task: Description of the task/response being evaluated
            context: Current context information
            available_info: Information available to complete task
            
        Returns:
            ConfidenceScore with overall and component scores
            
        Raises:
            ValueError: If task is empty
        """
        if not task:
            raise ValueError("task cannot be empty")
        
        self.total_scored += 1
        
        # Get capability assessment if available
        if self.capability_assessment and hasattr(self.capability_assessment, 'assess_capability'):
            try:
                capability_result = self.capability_assessment.assess_capability(
                    task=task,
                    context=context
                )
                capability_confidence = getattr(capability_result, 'estimated_proficiency', 0.5)
            except Exception as e:
                logger.warning(f"Failed to get capability assessment: {e}")
                capability_confidence = 0.5
        else:
            capability_confidence = 0.5
        
        # Component 2: Information confidence
        information_confidence = self._score_information_quality(available_info, task, context)
        
        # Component 3: Context confidence
        context_confidence = self._score_context_appropriateness(context, task)
        
        # Component 4: Performance confidence (historical)
        performance_confidence = self._score_performance_history(task)
        
        # Calculate overall confidence (weighted average)
        overall = (
            self.weights['capability'] * capability_confidence +
            self.weights['information'] * information_confidence +
            self.weights['context'] * context_confidence +
            self.weights['performance'] * performance_confidence
        )
        
        # Adjust for calibration bias (learned from feedback)
        overall = self._apply_calibration_adjustment(overall)
        
        # Clamp to valid range
        overall = max(0.0, min(1.0, overall))
        
        # Collect factors and limitations
        factors = []
        limitations = []
        
        if capability_confidence > 0.7:
            factors.append("Strong capability in this area")
        elif capability_confidence < 0.4:
            limitations.append(f"Limited capability for this task")
        
        if information_confidence < 0.5:
            limitations.append("Insufficient or uncertain information")
        
        if performance_confidence > 0.7:
            factors.append("Good track record with similar tasks")
        
        # Create confidence score
        score = ConfidenceScore(
            overall_confidence=overall,
            capability_confidence=capability_confidence,
            information_confidence=information_confidence,
            context_confidence=context_confidence,
            performance_confidence=performance_confidence,
            task_description=task,
            factors=factors,
            limitations=limitations
        )
        
        # Store in history
        self.confidence_history.append(score)
        if len(self.confidence_history) > self.max_history:
            self.confidence_history.pop(0)
        
        logger.debug(f"Confidence score for '{task[:30]}...': {overall:.2f} "
                    f"({score.get_level().name})")
        
        return score
    
    def should_express_doubt(self, confidence_score: Union[ConfidenceScore, float], 
                               threshold: float = 0.6) -> bool:
        """
        Determine whether to express doubt or hedge statements.
        
        Args:
            confidence_score: ConfidenceScore object or float
            threshold: Confidence threshold for expressing doubt
            
        Returns:
            True if doubt should be expressed
        """
        if isinstance(confidence_score, float):
            confidence = confidence_score
        else:
            confidence = confidence_score.overall_confidence
        
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        
        return confidence < threshold
    
    def get_hedging_phrase(self, 
                            confidence_score: Union[ConfidenceScore, float],
                            statement: str,
                            use_wednesday_style: bool = True) -> str:
        """
        Get appropriate hedging phrase for a statement.
        
        Args:
            confidence_score: ConfidenceScore object or float
            statement: The statement to hedge
            use_wednesday_style: Whether to use Wednesday's dark humor style
            
        Returns:
            Statement with appropriate hedging
        """
        if isinstance(confidence_score, float):
            level = ConfidenceLevel.from_float(confidence_score)
        else:
            level = confidence_score.get_level()
        
        # Check if we should even hedge
        if level in [ConfidenceLevel.HIGH, ConfidenceLevel.CERTAIN]:
            return statement
        
        # Try Wednesday-specific templates first
        if use_wednesday_style and level in self.WEDNESDAY_HEDGING:
            templates = self.WEDNESDAY_HEDGING[level]
        else:
            templates = self.HEDGING_TEMPLATES.get(level, self.HEDGING_TEMPLATES[ConfidenceLevel.MIXED])
        
        # Choose template deterministically
        template_idx = abs(hash(statement)) % len(templates)
        template = templates[template_idx]
        
        # Format with statement (ensure statement is lowercase for natural flow)
        statement_lower = statement[0].lower() + statement[1:] if statement else statement
        
        return template.format(statement=statement_lower)
    
    def calibrate_from_feedback(self, 
                                 task: str, 
                                 actual_outcome: float,
                                 predicted_confidence: Optional[float] = None) -> None:
        """
        Calibrate confidence based on actual performance feedback.
        
        Args:
            task: The task that was performed
            actual_outcome: Actual success/accuracy (0-1)
            predicted_confidence: Optional predicted confidence (uses last score if None)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not task:
            raise ValueError("task cannot be empty")
        if not 0 <= actual_outcome <= 1:
            raise ValueError(f"actual_outcome must be between 0 and 1, got {actual_outcome}")
        
        self.feedback_received += 1
        
        # Find the confidence score for this task
        if predicted_confidence is None:
            # Find most recent confidence for similar task
            for score in reversed(self.confidence_history):
                if task in score.task_description or score.task_description in task:
                    predicted_confidence = score.overall_confidence
                    break
            
            if predicted_confidence is None:
                logger.warning(f"No previous confidence score found for task: {task}")
                return
        
        # Create calibration record
        record = CalibrationRecord(
            task=task,
            predicted_confidence=predicted_confidence,
            actual_outcome=actual_outcome
        )
        
        self.calibration_history.append(record)
        
        # Update calibration metrics
        self._update_calibration_metrics()
        
        logger.debug(f"Calibration update: predicted={predicted_confidence:.2f}, "
                    f"actual={actual_outcome:.2f}")
    
    def get_calibration_metrics(self) -> Dict[str, Any]:
        """
        Get calibration performance metrics.
        
        Returns:
            Dictionary with calibration metrics
        """
        if not self.calibration_history:
            return {'has_data': False}
        
        # Calculate overconfidence rate
        overconfident = sum(1 for r in self.calibration_history if r.is_overconfident())
        underconfident = sum(1 for r in self.calibration_history if r.is_underconfident())
        
        # Calculate reliability data (simplified bins)
        bin_count = 5
        bins = [i / bin_count for i in range(bin_count + 1)]
        reliability_data = []
        
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            in_bin = [r for r in self.calibration_history 
                     if low <= r.predicted_confidence < high]
            
            if in_bin:
                avg_confidence = sum(r.predicted_confidence for r in in_bin) / len(in_bin)
                avg_accuracy = sum(r.actual_outcome for r in in_bin) / len(in_bin)
                count = len(in_bin)
                reliability_data.append({
                    'bin_low': round(low, 2),
                    'bin_high': round(high, 2),
                    'avg_confidence': round(avg_confidence, 3),
                    'avg_accuracy': round(avg_accuracy, 3),
                    'count': count
                })
        
        return {
            'has_data': True,
            'total_records': len(self.calibration_history),
            'mean_calibration_error': round(self.calibration_error, 3),
            'calibration_bias': round(self.calibration_bias, 3),
            'overconfident_rate': round(overconfident / len(self.calibration_history), 3),
            'underconfident_rate': round(underconfident / len(self.calibration_history), 3),
            'reliability_data': reliability_data
        }
    
    def get_confidence_trend(self, recent_n: int = 10) -> Dict[str, Any]:
        """
        Get trend in confidence scores over time.
        
        Args:
            recent_n: Number of recent scores to analyze
            
        Returns:
            Dictionary with trend information
        """
        if len(self.confidence_history) < 2:
            return {'has_trend': False}
        
        recent = self.confidence_history[-recent_n:]
        
        confidences = [s.overall_confidence for s in recent]
        avg_confidence = sum(confidences) / len(confidences)
        
        if len(confidences) > 1:
            trend = confidences[-1] - confidences[0]
            # Simple linear trend using first and last points
            direction = 'increasing' if trend > 0.05 else 'decreasing' if trend < -0.05 else 'stable'
        else:
            trend = 0
            direction = 'stable'
        
        # Calculate volatility (standard deviation)
        if len(confidences) > 1:
            variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
            volatility = math.sqrt(variance)
        else:
            volatility = 0
        
        return {
            'has_trend': True,
            'average_confidence': round(avg_confidence, 3),
            'trend_direction': direction,
            'trend_magnitude': round(trend, 3),
            'volatility': round(volatility, 3),
            'recent_scores': [round(c, 3) for c in confidences[-5:]]
        }
    
    def _score_information_quality(self, 
                                     info: Optional[Dict[str, Any]], 
                                     task: str,
                                     context: Optional[Dict[str, Any]]) -> float:
        """Score the quality and sufficiency of available information"""
        if not info:
            return 0.3  # Low confidence without info
        
        score = 0.5  # Base
        
        # Check completeness
        if info.get('complete', False):
            score += 0.2
        elif info.get('partial', False):
            score += 0.1
        
        # Check recency
        timestamp = info.get('timestamp')
        if timestamp and isinstance(timestamp, (int, float)):
            age = time.time() - timestamp
            if age < 3600:  # < 1 hour
                score += 0.1
            elif age > 86400:  # > 1 day
                score -= 0.1
        
        # Check source reliability
        source = info.get('source', 'unknown').lower()
        if source in self.RELIABLE_SOURCES:
            score += 0.1
        elif source == 'unknown':
            score -= 0.1
        
        # Check for conflicting information
        if info.get('conflicts', False):
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_context_appropriateness(self, context: Optional[Dict[str, Any]], task: str) -> float:
        """Score how appropriate the context is for the task"""
        if not context:
            return 0.5  # Neutral
        
        score = 0.6
        
        # Check formality match
        formality = context.get('formality', 0.5)
        formality_required = context.get('formality_required')
        if formality_required is not None:
            formality_gap = abs(formality - formality_required)
            score -= formality_gap * 0.3
        
        # Check time pressure
        if context.get('time_pressure', 0) > 0.7:
            score -= 0.2  # Rushed = lower confidence
        
        # Check distraction level
        if context.get('distraction_level', 0) > 0.5:
            score -= 0.1
        
        # Check if context supports the task
        if context.get('supports_task', False):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _score_performance_history(self, task: str) -> float:
        """Score based on historical performance on similar tasks"""
        # Find similar tasks in calibration history
        similar_outcomes = []
        for record in self.calibration_history:
            # Check if tasks are similar (simple substring match)
            if task in record.task or record.task in task:
                similar_outcomes.append(record.actual_outcome)
        
        if not similar_outcomes:
            return 0.5  # Neutral baseline
        
        # Return average performance
        return sum(similar_outcomes) / len(similar_outcomes)
    
    def _apply_calibration_adjustment(self, raw_confidence: float) -> float:
        """Apply learned calibration adjustment to raw confidence"""
        if len(self.calibration_history) < 10:
            return raw_confidence
        
        # Adjust based on calibration bias
        # If we tend to be overconfident, reduce confidence; if underconfident, increase
        adjustment = -self.calibration_bias * 0.5  # Cap adjustment at 0.1
        adjusted = raw_confidence + adjustment
        
        return max(0.0, min(1.0, adjusted))
    
    def _update_calibration_metrics(self) -> None:
        """Update calibration error and bias metrics"""
        if not self.calibration_history:
            self.calibration_error = 0.0
            self.calibration_bias = 0.0
            return
        
        errors = []
        biases = []
        
        for record in self.calibration_history:
            errors.append(record.error())
            biases.append(record.bias())
        
        self.calibration_error = sum(errors) / len(errors)
        self.calibration_bias = sum(biases) / len(biases)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get confidence scorer statistics"""
        return {
            'total_scored': self.total_scored,
            'feedback_received': self.feedback_received,
            'confidence_history': len(self.confidence_history),
            'calibration_records': len(self.calibration_history),
            'calibration_error': round(self.calibration_error, 3),
            'calibration_bias': round(self.calibration_bias, 3),
            'average_confidence': self._get_average_confidence()
        }
    
    def _get_average_confidence(self) -> float:
        """Calculate average confidence from history"""
        if not self.confidence_history:
            return 0.0
        return sum(s.overall_confidence for s in self.confidence_history) / len(self.confidence_history)
    
    def reset_calibration(self) -> None:
        """Reset calibration history and metrics"""
        self.calibration_history.clear()
        self.calibration_error = 0.0
        self.calibration_bias = 0.0
        self.feedback_received = 0
        logger.info("Confidence calibration reset")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Confidence Scorer Test ===\n")
    
    # Mock capability assessment
    class MockCapabilityAssessment:
        def assess_capability(self, task, context=None):
            class Result:
                def __init__(self, proficiency):
                    self.estimated_proficiency = proficiency
            proficiency = 0.7 if 'dark humor' in task.lower() else 0.5
            return Result(proficiency)
    
    # Mock personality
    class MockPersonality:
        def get_trait(self, trait):
            return 0.8 if trait == 'dark_humor' else 0.5
    
    # Create confidence scorer
    scorer = ConfidenceScorer(
        capability_assessment=MockCapabilityAssessment(),
        personality=MockPersonality(),
        config={'weights': {'capability': 0.4, 'information': 0.3, 'context': 0.1, 'performance': 0.2}}
    )
    
    # Test different tasks
    test_tasks = [
        ("Tell a dark humor joke", 
         {'source': 'verified', 'complete': True, 'timestamp': time.time()},
         {'formality': 0.2, 'formality_required': 0.2}),
        
        ("Explain quantum physics", 
         {'source': 'unknown', 'partial': True},
         {'formality': 0.8, 'formality_required': 0.6}),
        
        ("Remember what user said 5 minutes ago", 
         {'complete': True, 'timestamp': time.time() - 60},
         {'time_pressure': 0.3}),
        
        ("Predict stock market movement", 
         {'source': 'unreliable'},
         {'distraction_level': 0.8}),
        
        ("Answer a simple factual question", 
         {'source': 'verified', 'complete': True},
         {'formality': 0.5}),
    ]
    
    print("--- Confidence Scoring ---")
    for i, (task, info, context) in enumerate(test_tasks):
        print(f"\nTask {i+1}: {task}")
        
        score = scorer.score_confidence(
            task=task,
            context=context,
            available_info=info
        )
        
        print(f"  Overall confidence: {score.overall_confidence:.2f} ({score.get_level().name})")
        print(f"  Components: capability={score.capability_confidence:.2f}, "
              f"info={score.information_confidence:.2f}, "
              f"context={score.context_confidence:.2f}, "
              f"perf={score.performance_confidence:.2f}")
        
        if score.limitations:
            print(f"  Limitations: {score.limitations}")
        
        # Test hedging
        should_hedge = scorer.should_express_doubt(score)
        if should_hedge:
            hedged = scorer.get_hedging_phrase(score, "this is the answer", use_wednesday_style=True)
            print(f"  Hedged: \"{hedged}\"")
    
    # Simulate feedback and calibration
    print("\n--- Calibration from Feedback ---")
    
    # Add some calibration records
    test_feedback = [
        ("Tell a dark humor joke", 0.9, 0.85),  # Slightly overconfident
        ("Explain quantum physics", 0.4, 0.3),  # Slightly overconfident
        ("Remember what user said", 0.8, 0.75),  # Slightly overconfident
        ("Predict stock market", 0.2, 0.25),  # Slightly underconfident
        ("Answer factual question", 0.85, 0.9),  # Slightly underconfident
    ]
    
    for task, predicted, actual in test_feedback:
        scorer.calibrate_from_feedback(task, actual, predicted)
        print(f"  Task: {task[:25]}... | Pred: {predicted:.2f} | Actual: {actual:.2f}")
    
    # Get calibration metrics
    print("\n--- Calibration Metrics ---")
    metrics = scorer.get_calibration_metrics()
    if metrics['has_data']:
        print(f"  Mean calibration error: {metrics['mean_calibration_error']:.3f}")
        print(f"  Calibration bias: {metrics['calibration_bias']:.3f} "
              f"({'overconfident' if metrics['calibration_bias'] > 0 else 'underconfident'})")
        print(f"  Overconfident rate: {metrics['overconfident_rate']:.2%}")
        print(f"  Underconfident rate: {metrics['underconfident_rate']:.2%}")
        print(f"  Records: {metrics['total_records']}")
        
        if metrics['reliability_data']:
            print("\n  Reliability by bin:")
            for bin_data in metrics['reliability_data'][:3]:
                print(f"    {bin_data['bin_low']}-{bin_data['bin_high']}: "
                      f"conf={bin_data['avg_confidence']:.2f}, "
                      f"acc={bin_data['avg_accuracy']:.2f} (n={bin_data['count']})")
    
    # Get confidence trend
    print("\n--- Confidence Trend ---")
    trend = scorer.get_confidence_trend(recent_n=8)
    if trend['has_trend']:
        print(f"  Average confidence: {trend['average_confidence']:.2f}")
        print(f"  Trend: {trend['trend_direction']} (magnitude: {trend['trend_magnitude']:.3f})")
        print(f"  Volatility: {trend['volatility']:.3f}")
        print(f"  Recent scores: {trend['recent_scores']}")
    
    # Get statistics
    print("\n--- Statistics ---")
    stats = scorer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")