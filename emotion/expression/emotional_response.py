"""
relevance_detection.py - Initial emotional relevance filtering for Wednesday AI

This module implements the first stage of emotional processing: quickly determining
whether a stimulus warrants full emotional appraisal. Based on theories of
preattentive processing and emotional relevance detection, it acts as a gatekeeper
to prevent unnecessary cognitive load from emotionally irrelevant stimuli.

Key improvements:
- Added proper validation and error handling
- Fixed memory management issues
- Enhanced type safety with comprehensive type hints
- Improved pattern matching efficiency
- Added configurable thresholds and learning capabilities
"""

import time
import logging
import math
import re
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class RelevanceCategory(Enum):
    """Categories of emotional relevance"""
    THREAT = "threat"               # Potential harm
    OPPORTUNITY = "opportunity"      # Potential gain
    VALUE_VIOLATION = "value_violation"  # Clashes with values
    VALUE_ALIGNMENT = "value_alignment"  # Supports values
    GOAL_RELEVANT = "goal_relevant"   # Affects current goals
    SOCIAL = "social"                 # Social significance
    NOVEL = "novel"                   # Novel/interesting
    URGENT = "urgent"                  # Requires immediate attention
    DARK_HUMOR = "dark_humor"          # Wednesday-specific


@dataclass
class RelevanceResult:
    """
    Result of relevance detection for a stimulus.
    
    This is a lightweight structure used to decide whether to
    trigger full emotional appraisal.
    """
    is_relevant: bool
    relevance_score: float  # 0-1 overall score
    primary_category: Optional[RelevanceCategory] = None
    categories: Dict[RelevanceCategory, float] = field(default_factory=dict)
    triggers_full_appraisal: bool = False
    processing_time_ms: float = 0.0
    stimulus_hash: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate result data"""
        if not 0 <= self.relevance_score <= 1:
            raise ValueError(f"Relevance score must be between 0 and 1, got {self.relevance_score}")
        
        # Validate category scores
        for category, score in self.categories.items():
            if not isinstance(category, RelevanceCategory):
                raise TypeError(f"Category must be RelevanceCategory, got {type(category)}")
            if not 0 <= score <= 1:
                raise ValueError(f"Category score must be between 0 and 1, got {score}")
    
    def should_appraise(self, threshold: float = 0.5) -> bool:
        """
        Determine if full appraisal should be triggered.
        
        Args:
            threshold: Minimum score to trigger appraisal
            
        Returns:
            True if appraisal should be triggered
            
        Raises:
            ValueError: If threshold is outside valid range
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        
        return self.is_relevant and self.relevance_score >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'is_relevant': self.is_relevant,
            'score': round(self.relevance_score, 3),
            'primary_category': self.primary_category.value if self.primary_category else None,
            'categories': {k.value: round(v, 3) for k, v in self.categories.items()},
            'triggers_appraisal': self.triggers_full_appraisal,
            'processing_time_ms': round(self.processing_time_ms, 2)
        }


@dataclass
class MemoryEntry:
    """Entry in relevance memory"""
    hash: str
    score: float
    timestamp: float
    count: int = 1
    
    def __post_init__(self):
        """Validate entry"""
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")
        if self.count <= 0:
            raise ValueError(f"Count must be positive, got {self.count}")


class RelevanceDetector:
    """
    Rapid, low-cost emotional relevance detector.
    
    Acts as the first gate in emotional processing, quickly filtering
    stimuli to determine if they warrant full cognitive appraisal.
    
    The detector uses:
    - Pattern matching against known relevance patterns
    - Personal value checking
    - Current need/goal state
    - Learned relevance from experience
    - Wednesday-specific sensitivity patterns
    
    Performance characteristics:
    - O(n) where n is number of patterns
    - Typically < 1ms processing time
    - Minimal memory footprint
    """
    
    # Default importance threshold (can be adjusted by context)
    DEFAULT_THRESHOLD = 0.3
    
    # Wednesday's core values (with weights)
    CORE_VALUES = {
        'loyalty': 0.9,
        'truth': 0.8,
        'justice': 0.8,
        'authenticity': 0.9,
        'intelligence': 0.7,
        'independence': 0.8,
        'curiosity': 0.7,
        'dark_humor': 0.8  # Yes, this is a value for Wednesday
    }
    
    # Keywords and patterns for rapid relevance detection
    # Compiled into regex patterns for efficiency
    RELEVANCE_PATTERNS = {
        RelevanceCategory.THREAT: {
            'keywords': ['danger', 'threat', 'risk', 'harm', 'hurt', 'attack', 
                        'betray', 'trap', 'warning', 'careful', 'enemy'],
            'weight': 0.8,
            'patterns': []  # Will be compiled in __init__
        },
        RelevanceCategory.OPPORTUNITY: {
            'keywords': ['opportunity', 'chance', 'advantage', 'benefit', 'gain',
                        'useful', 'helpful', 'leverage', 'profit'],
            'weight': 0.6,
            'patterns': []
        },
        RelevanceCategory.VALUE_VIOLATION: {
            'keywords': ['lie', 'betray', 'fake', 'dishonest', 'unfair', 'injustice',
                        'cruel', 'manipulate', 'exploit', 'deceive'],
            'weight': 0.9,
            'patterns': []
        },
        RelevanceCategory.VALUE_ALIGNMENT: {
            'keywords': ['truth', 'honest', 'loyal', 'fair', 'just', 'authentic',
                        'real', 'genuine', 'integrity'],
            'weight': 0.7,
            'patterns': []
        },
        RelevanceCategory.SOCIAL: {
            'keywords': ['friend', 'family', 'trust', 'relationship', 'together',
                        'alone', 'reject', 'accept', 'belong', 'social'],
            'weight': 0.6,
            'patterns': []
        },
        RelevanceCategory.NOVEL: {
            'keywords': ['strange', 'unusual', 'odd', 'peculiar', 'weird', 'mystery',
                        'curious', 'interesting', 'unexpected', 'surprising', 'novel'],
            'weight': 0.5,
            'patterns': []
        },
        RelevanceCategory.URGENT: {
            'keywords': ['now', 'immediate', 'urgent', 'emergency', 'quick', 'hurry',
                        'asap', 'critical', 'deadline', 'rush'],
            'weight': 0.9,
            'patterns': []
        },
        RelevanceCategory.DARK_HUMOR: {
            'keywords': ['death', 'dark', 'macabre', 'grim', 'morbid', 'twisted',
                        'ironic', 'absurd', 'sarcasm', 'joke', 'funny', 'humor'],
            'weight': 0.8,
            'patterns': [],
            'special': 'wednesday'  # Wednesday-specific category
        }
    }
    
    def __init__(self, 
                 values_system: Optional[Any] = None, 
                 needs_system: Optional[Any] = None, 
                 personality: Optional[Dict[str, float]] = None):
        """
        Initialize the relevance detector.
        
        Args:
            values_system: Reference to values system for value checking
            needs_system: Reference to needs system for current needs
            personality: Personality parameters affecting relevance
            
        Raises:
            ValueError: If personality parameters are invalid
        """
        self.values_system = values_system
        self.needs_system = needs_system
        
        # Personality influences on relevance
        default_personality = {
            'importance_threshold': 0.3,
            'curiosity': 0.7,              # How much novel things matter
            'loyalty_sensitivity': 0.9,      # How much loyalty matters
            'threat_sensitivity': 0.7,       # How much threats matter
            'social_sensitivity': 0.4,        # How much social things matter
            'dark_humor_sensitivity': 0.8,    # How much dark humor matters
        }
        
        self.personality = default_personality.copy()
        if personality:
            self._validate_personality(personality)
            self.personality.update(personality)
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Adaptive threshold based on cognitive load
        self.base_threshold = self.personality['importance_threshold']
        self.current_threshold = self.base_threshold
        
        # Learning from past relevance
        self.relevance_memory: Dict[str, MemoryEntry] = {}
        self.stimulus_history: deque = deque(maxlen=1000)  # (hash, score, timestamp)
        
        # Category statistics for adaptation
        self.category_stats: Dict[RelevanceCategory, Dict[str, float]] = defaultdict(
            lambda: {'true_positives': 0, 'false_positives': 0, 'avg_score': 0.0}
        )
        
        # Statistics
        self.stimuli_processed = 0
        self.relevant_count = 0
        self.appraisal_triggered = 0
        
        logger.info(f"RelevanceDetector initialized with threshold {self.base_threshold}")
    
    def _validate_personality(self, personality: Dict[str, float]) -> None:
        """Validate personality parameters"""
        for key, value in personality.items():
            if key not in self.personality:
                raise ValueError(f"Unknown personality parameter: {key}")
            if not 0 <= value <= 1:
                raise ValueError(f"Personality parameter {key} must be between 0 and 1, got {value}")
    
    def _compile_patterns(self) -> None:
        """Compile keyword patterns into regex for efficient matching"""
        for category, pattern_data in self.RELEVANCE_PATTERNS.items():
            keywords = pattern_data.get('keywords', [])
            if keywords:
                # Create regex pattern that matches whole words
                pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
                pattern_data['compiled'] = re.compile(pattern, re.IGNORECASE)
    
    def check_relevance(self, 
                        stimulus: Union[str, Dict, Any], 
                        context: Optional[Dict[str, Any]] = None,
                        current_needs: Optional[Dict[str, float]] = None,
                        cognitive_load: float = 0.0) -> RelevanceResult:
        """
        Rapidly check if a stimulus is emotionally relevant.
        
        Args:
            stimulus: The stimulus to check (text, dict, or other)
            context: Current context information
            current_needs: Current needs state (if None, fetched from needs_system)
            cognitive_load: Current cognitive load (0-1) affecting threshold
            
        Returns:
            RelevanceResult indicating if stimulus matters
            
        Raises:
            ValueError: If cognitive_load is outside valid range
        """
        start_time = time.time()
        
        # Validate inputs
        if not 0 <= cognitive_load <= 1:
            raise ValueError(f"Cognitive load must be between 0 and 1, got {cognitive_load}")
        
        # Generate stimulus hash for memory
        stimulus_hash = self._hash_stimulus(stimulus)
        
        # Adjust threshold based on cognitive load
        self._adjust_threshold(cognitive_load)
        
        # Get current needs if not provided
        if current_needs is None and self.needs_system:
            try:
                current_needs = self.needs_system.get_current_needs()
            except AttributeError:
                logger.warning("needs_system has no get_current_needs method")
                current_needs = {}
        else:
            current_needs = current_needs or {}
        
        # Calculate relevance scores by category
        category_scores = {}
        total_score = 0.0
        
        # Check each relevance category
        for category, pattern in self.RELEVANCE_PATTERNS.items():
            score = self._check_category_relevance(
                stimulus, category, pattern, context, current_needs
            )
            
            # Apply personality weighting
            score = self._apply_personality_weight(category, score)
            
            if score > 0:
                category_scores[category] = score
                total_score += score
        
        # Check against personal values if available
        values_score = self._check_values_relevance(stimulus)
        if values_score > 0:
            category_scores[RelevanceCategory.VALUE_ALIGNMENT] = values_score
            total_score += values_score
        
        # Check memory for learned relevance
        memory_score = self._check_memory_relevance(stimulus_hash)
        if memory_score > 0:
            # Add as a general boost
            total_score += memory_score * 0.5
        
        # Normalize score (0-1 range)
        # Maximum reasonable score from 3 categories
        normalized_score = min(1.0, total_score / 3.0)
        
        # Determine primary category
        primary_category = None
        if category_scores:
            primary_category = max(category_scores.items(), key=lambda x: x[1])[0]
        
        # Determine if relevant (above threshold)
        is_relevant = normalized_score >= self.current_threshold
        
        # Decide if triggers full appraisal
        triggers_appraisal = self._should_trigger_appraisal(
            is_relevant, normalized_score, primary_category, memory_score
        )
        
        # Update memory if stimulus was significant
        if normalized_score > 0.2:
            self._update_memory(stimulus_hash, normalized_score)
        
        # Update statistics
        self.stimuli_processed += 1
        if is_relevant:
            self.relevant_count += 1
        if triggers_appraisal:
            self.appraisal_triggered += 1
        
        # Update category statistics
        if primary_category and normalized_score > 0:
            stats = self.category_stats[primary_category]
            stats['avg_score'] = (stats['avg_score'] * stats.get('count', 0) + normalized_score) / (stats.get('count', 0) + 1)
            stats['count'] = stats.get('count', 0) + 1
        
        # Create result
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        result = RelevanceResult(
            is_relevant=is_relevant,
            relevance_score=normalized_score,
            primary_category=primary_category,
            categories=category_scores,
            triggers_full_appraisal=triggers_appraisal,
            processing_time_ms=processing_time,
            stimulus_hash=stimulus_hash
        )
        
        if is_relevant:
            logger.debug(f"Relevant stimulus detected: {primary_category.value if primary_category else 'unknown'} "
                        f"(score={normalized_score:.2f}, time={processing_time:.1f}ms)")
        
        return result
    
    def _check_category_relevance(self, 
                                  stimulus: Union[str, Dict, Any],
                                  category: RelevanceCategory,
                                  pattern: Dict,
                                  context: Optional[Dict],
                                  needs: Dict) -> float:
        """Check relevance for a specific category"""
        score = 0.0
        
        # Text-based stimuli
        if isinstance(stimulus, str):
            text = stimulus.lower()
            
            # Use compiled regex for keyword matching
            compiled = pattern.get('compiled')
            if compiled:
                matches = compiled.findall(text)
                if matches:
                    # Score based on number of matches, capped
                    match_score = min(1.0, len(matches) * 0.3)
                    score += pattern.get('weight', 0.5) * match_score
            
            # Check for special patterns
            if category == RelevanceCategory.DARK_HUMOR:
                # Check for combinations that might indicate dark humor
                dark_words = {'death', 'dead', 'kill', 'murder'}
                humor_words = {'joke', 'funny', 'humor', 'laugh'}
                
                has_dark = any(word in text for word in dark_words)
                has_humor = any(word in text for word in humor_words)
                
                if has_dark and has_humor:
                    score += 0.8
        
        # Dictionary-based stimuli (structured events)
        elif isinstance(stimulus, dict):
            # Check explicit category indicators
            if 'category' in stimulus:
                if stimulus['category'] == category.value:
                    score += float(stimulus.get('intensity', 0.5)) * 0.8
                elif isinstance(stimulus['category'], list) and category.value in stimulus['category']:
                    score += float(stimulus.get('intensity', 0.5)) * 0.6
            
            # Check for threat indicators
            if category == RelevanceCategory.THREAT:
                threat_level = float(stimulus.get('threat_level', 0))
                if threat_level > 0:
                    score += threat_level * 0.9
            
            # Check for goal relevance
            if category == RelevanceCategory.GOAL_RELEVANT:
                if 'goal_id' in stimulus:
                    score += 0.7
        
        # Context-based relevance
        if context:
            # If we're in a sensitive context, lower threshold for threats
            if category == RelevanceCategory.THREAT and context.get('high_alert', False):
                score += 0.2
            
            # If we're in a social context, increase social relevance
            if category == RelevanceCategory.SOCIAL and context.get('social_situation', False):
                score += 0.2
        
        # Need-based relevance
        if needs:
            # If we need safety, threats are more relevant
            if category == RelevanceCategory.THREAT and needs.get('safety', 0.5) > 0.7:
                score += 0.3
            
            # If we need information, novel things are more relevant
            if category == RelevanceCategory.NOVEL and needs.get('knowledge', 0.5) > 0.7:
                score += 0.3
        
        return min(1.0, score)
    
    def _check_values_relevance(self, stimulus: Union[str, Dict, Any]) -> float:
        """Check relevance based on personal values"""
        if not self.values_system and not isinstance(stimulus, str):
            # Without values system, do simple text matching
            return 0.0
        
        score = 0.0
        
        # Quick check against core values
        if isinstance(stimulus, str):
            text = stimulus.lower()
            
            for value, weight in self.CORE_VALUES.items():
                if value in text:
                    # Check if value is being supported or violated
                    violation_words = {'betray', 'lie', 'fake', 'deceive', 'dishonest'}
                    support_words = {'truth', 'loyal', 'real', 'honest', 'genuine'}
                    
                    if any(word in text for word in violation_words):
                        score += weight * 0.8  # Violation is highly relevant
                    elif any(word in text for word in support_words):
                        score += weight * 0.6  # Support is relevant
        
        return min(1.0, score)
    
    def _apply_personality_weight(self, category: RelevanceCategory, score: float) -> float:
        """Apply personality-based weighting to category scores"""
        weights = {
            RelevanceCategory.THREAT: self.personality['threat_sensitivity'],
            RelevanceCategory.SOCIAL: self.personality['social_sensitivity'],
            RelevanceCategory.NOVEL: self.personality['curiosity'],
            RelevanceCategory.DARK_HUMOR: self.personality['dark_humor_sensitivity'],
            RelevanceCategory.VALUE_VIOLATION: self.personality['loyalty_sensitivity'],
            RelevanceCategory.VALUE_ALIGNMENT: self.personality['loyalty_sensitivity'] * 0.7,
        }
        
        weight = weights.get(category, 1.0)
        return score * weight
    
    def _check_memory_relevance(self, stimulus_hash: str) -> float:
        """Check if we've learned this stimulus is relevant"""
        if stimulus_hash in self.relevance_memory:
            entry = self.relevance_memory[stimulus_hash]
            
            # Calculate recency factor (decay over time)
            time_diff = time.time() - entry.timestamp
            hours_passed = time_diff / 3600.0
            recency = math.exp(-hours_passed / 24.0)  # Decay over days
            
            # Frequency boost
            frequency_boost = min(1.0, entry.count / 10.0)
            
            return entry.score * recency * (1 + frequency_boost * 0.3)
        
        return 0.0
    
    def _update_memory(self, stimulus_hash: str, score: float) -> None:
        """Update relevance memory with new experience"""
        if stimulus_hash in self.relevance_memory:
            # Update existing entry
            entry = self.relevance_memory[stimulus_hash]
            entry.score = entry.score * 0.7 + score * 0.3  # Moving average
            entry.timestamp = time.time()
            entry.count += 1
        else:
            # Create new entry
            self.relevance_memory[stimulus_hash] = MemoryEntry(
                hash=stimulus_hash,
                score=score,
                timestamp=time.time()
            )
        
        # Add to history
        self.stimulus_history.append((stimulus_hash, score, time.time()))
    
    def _should_trigger_appraisal(self, 
                                  is_relevant: bool,
                                  score: float, 
                                  primary_category: Optional[RelevanceCategory],
                                  memory_score: float) -> bool:
        """Determine if stimulus should trigger full appraisal"""
        if not is_relevant:
            return False
        
        # Always appraise high scores
        if score >= 0.6:
            return True
        
        # Always appraise certain categories
        if primary_category in [RelevanceCategory.THREAT, 
                               RelevanceCategory.URGENT,
                               RelevanceCategory.VALUE_VIOLATION]:
            return True
        
        # Appraise if above threshold AND has history of being significant
        if memory_score > 0.5:
            return True
        
        # Appraise if multiple categories indicate relevance
        if len([s for s in self.category_stats if s > 0.3]) >= 2:
            return True
        
        return False
    
    def _adjust_threshold(self, cognitive_load: float) -> None:
        """
        Adjust relevance threshold based on cognitive load.
        
        Higher cognitive load = higher threshold (filter more aggressively)
        """
        # Cognitive load increases threshold, but never above 0.8
        load_factor = 1.0 + cognitive_load * 0.5
        self.current_threshold = min(0.8, self.base_threshold * load_factor)
    
    def _hash_stimulus(self, stimulus: Any) -> str:
        """Create a hash of the stimulus for memory lookup"""
        import hashlib
        
        if isinstance(stimulus, str):
            # For text, use first 100 chars as content
            content = stimulus[:100].lower()
        elif isinstance(stimulus, dict):
            # For dicts, extract relevant fields
            relevant_keys = ['content', 'text', 'message', 'event_type', 'id']
            content_parts = []
            for key in relevant_keys:
                if key in stimulus:
                    value = stimulus[key]
                    if isinstance(value, str):
                        content_parts.append(value[:50])
                    else:
                        content_parts.append(str(value))
            content = '|'.join(content_parts)
        else:
            content = str(stimulus)[:100]
        
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def set_threshold(self, threshold: float) -> None:
        """Manually set relevance threshold"""
        if not 0.1 <= threshold <= 0.9:
            raise ValueError(f"Threshold must be between 0.1 and 0.9, got {threshold}")
        
        self.base_threshold = threshold
        self.current_threshold = threshold
        logger.info(f"Relevance threshold set to {self.base_threshold}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            'stimuli_processed': self.stimuli_processed,
            'relevant_count': self.relevant_count,
            'appraisal_triggered': self.appraisal_triggered,
            'relevance_rate': round(self.relevant_count / max(1, self.stimuli_processed), 3),
            'appraisal_rate': round(self.appraisal_triggered / max(1, self.stimuli_processed), 3),
            'current_threshold': round(self.current_threshold, 3),
            'base_threshold': round(self.base_threshold, 3),
            'memory_size': len(self.relevance_memory),
            'category_stats': {
                k.value: {
                    'avg_score': round(v.get('avg_score', 0), 3),
                    'count': v.get('count', 0)
                }
                for k, v in self.category_stats.items()
            }
        }
    
    def learn_relevance_pattern(self, stimulus: Any, was_relevant: bool, 
                                 actual_importance: float) -> None:
        """
        Learn from feedback about relevance judgments.
        
        Args:
            stimulus: The stimulus that was judged
            was_relevant: Whether we thought it was relevant
            actual_importance: Actual importance (0-1) from full appraisal
            
        Raises:
            ValueError: If actual_importance is outside valid range
        """
        if not 0 <= actual_importance <= 1:
            raise ValueError(f"Actual importance must be between 0 and 1, got {actual_importance}")
        
        stimulus_hash = self._hash_stimulus(stimulus)
        
        # Update memory with actual importance
        self._update_memory(stimulus_hash, actual_importance)
        
        # Update category statistics based on feedback
        if was_relevant and actual_importance < 0.2:
            # False positive - we thought it was relevant but it wasn't
            logger.debug("Learning from false positive relevance detection")
            
            # Find which categories were active for this stimulus
            if stimulus_hash in self.relevance_memory:
                # Could adjust keyword weights here in future versions
                pass
            
        elif not was_relevant and actual_importance > 0.6:
            # False negative - we missed something important
            logger.debug("Learning from false negative relevance detection")
            
            # Could increase sensitivity for similar patterns in future
    
    def get_memory_insights(self) -> Dict[str, Any]:
        """Get insights from relevance memory"""
        if not self.relevance_memory:
            return {'message': 'No memory data available'}
        
        # Calculate average scores
        scores = [entry.score for entry in self.relevance_memory.values()]
        avg_score = sum(scores) / len(scores)
        
        # Find most relevant memories
        top_memories = sorted(
            self.relevance_memory.items(),
            key=lambda x: x[1].score,
            reverse=True
        )[:5]
        
        return {
            'total_memories': len(self.relevance_memory),
            'average_score': round(avg_score, 3),
            'top_memories': [
                {'hash': h, 'score': round(e.score, 3), 'count': e.count}
                for h, e in top_memories
            ]
        }
    
    def reset(self) -> None:
        """Reset detector to initial state"""
        self.current_threshold = self.base_threshold
        self.relevance_memory.clear()
        self.stimulus_history.clear()
        self.category_stats.clear()
        self.stimuli_processed = 0
        self.relevant_count = 0
        self.appraisal_triggered = 0
        logger.info("RelevanceDetector reset")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Relevance Detector Test ===\n")
    
    # Create detector with Wednesday-like personality
    detector = RelevanceDetector(personality={
        'importance_threshold': 0.3,
        'curiosity': 0.8,
        'loyalty_sensitivity': 0.9,
        'threat_sensitivity': 0.7,
        'social_sensitivity': 0.4,
        'dark_humor_sensitivity': 0.9,
    })
    
    # Test stimuli
    test_stimuli = [
        "The weather is nice today",
        "Someone betrayed my trust",
        "Want to hear a joke about death?",
        "There's a threat to your friend's safety",
        "The clock is ticking",
        "This is a completely mundane statement about nothing in particular",
        "I found something strange and unusual in the basement",
        "Your loyalty means everything to me",
        "The murderer left a darkly humorous note",
        "URGENT: Please respond immediately",
    ]
    
    print("Testing relevance detection on various stimuli:\n")
    
    for i, stimulus in enumerate(test_stimuli):
        result = detector.check_relevance(stimulus)
        
        print(f"Stimulus {i+1}: {stimulus[:50]}..." if len(stimulus) > 50 else f"Stimulus {i+1}: {stimulus}")
        print(f"  Relevant: {result.is_relevant} (score: {result.relevance_score:.2f})")
        print(f"  Primary category: {result.primary_category.value if result.primary_category else 'None'}")
        print(f"  Categories: {', '.join([f'{c.value}:{s:.2f}' for c, s in result.categories.items()])}")
        print(f"  Triggers appraisal: {result.triggers_full_appraisal}")
        print(f"  Processing time: {result.processing_time_ms:.2f}ms")
        print()
    
    print("--- Statistics ---")
    stats = detector.get_statistics()
    for key, value in stats.items():
        if key != 'category_stats':
            print(f"  {key}: {value}")
    
    print("\n--- Memory Insights ---")
    insights = detector.get_memory_insights()
    for key, value in insights.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")