"""relevance_detection.py - Initial emotional relevance filtering for Wednesday AI

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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class RelevanceCategory(Enum):
    """Categories of emotional relevance"""
    THREAT = "threat"
    OPPORTUNITY = "opportunity"
    VALUE_VIOLATION = "value_violation"
    VALUE_ALIGNMENT = "value_alignment"
    GOAL_RELEVANT = "goal_relevant"
    SOCIAL = "social"
    NOVEL = "novel"
    URGENT = "urgent"
    DARK_HUMOR = "dark_humor"


@dataclass
class RelevanceResult:
    """Result of relevance detection for a stimulus."""
    is_relevant: bool
    relevance_score: float  # 0-1 overall score
    primary_category: Optional[RelevanceCategory] = None
    categories: Dict[RelevanceCategory, float] = field(default_factory=dict)
    triggers_full_appraisal: bool = False
    processing_time_ms: float = 0.0
    stimulus_hash: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Validate result data."""
        if not 0 <= self.relevance_score <= 1:
            raise ValueError(f"Relevance score must be between 0 and 1, got {self.relevance_score}")

        # Validate category scores
        for category, score in self.categories.items():
            if not isinstance(category, RelevanceCategory):
                raise TypeError(f"Category must be RelevanceCategory, got {type(category)}")
            if not 0 <= score <= 1:
                raise ValueError(f"Category score must be between 0 and 1, got {score}")

    def should_appraise(self, threshold: float = 0.5) -> bool:
        """Determine if full appraisal should be triggered."""
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        return self.is_relevant and self.relevance_score >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
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
    """Entry in relevance memory."""
    hash: str
    score: float
    timestamp: float
    count: int = 1

    def __post_init__(self):
        """Validate entry."""
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")
        if self.count <= 0:
            raise ValueError(f"Count must be positive, got {self.count}")


class RelevanceDetector:
    """Rapid, low-cost emotional relevance detector."""

    DEFAULT_THRESHOLD = 0.3

    CORE_VALUES = {
        'loyalty': 0.9,
        'truth': 0.8,
        'justice': 0.8,
        'authenticity': 0.9,
        'intelligence': 0.7,
        'independence': 0.8,
        'curiosity': 0.7,
        'dark_humor': 0.8
    }

    RELEVANCE_PATTERNS = {
        RelevanceCategory.THREAT: {
            'keywords': ['danger', 'threat', 'risk', 'harm', 'hurt', 'attack', 
                        'betray', 'trap', 'warning', 'careful', 'enemy'],
            'weight': 0.8,
        },
        RelevanceCategory.OPPORTUNITY: {
            'keywords': ['opportunity', 'chance', 'advantage', 'benefit', 'gain',
                        'useful', 'helpful', 'leverage', 'profit'],
            'weight': 0.6,
        },
        RelevanceCategory.VALUE_VIOLATION: {
            'keywords': ['lie', 'betray', 'fake', 'dishonest', 'unfair', 'injustice',
                        'cruel', 'manipulate', 'exploit', 'deceive'],
            'weight': 0.9,
        },
        RelevanceCategory.VALUE_ALIGNMENT: {
            'keywords': ['truth', 'honest', 'loyal', 'fair', 'just', 'authentic',
                        'real', 'genuine', 'integrity'],
            'weight': 0.7,
        },
        RelevanceCategory.SOCIAL: {
            'keywords': ['friend', 'family', 'trust', 'relationship', 'together',
                        'alone', 'reject', 'accept', 'belong', 'social'],
            'weight': 0.6,
        },
        RelevanceCategory.NOVEL: {
            'keywords': ['strange', 'unusual', 'odd', 'peculiar', 'weird', 'mystery',
                        'curious', 'interesting', 'unexpected', 'surprising', 'novel'],
            'weight': 0.5,
        },
        RelevanceCategory.URGENT: {
            'keywords': ['now', 'immediate', 'urgent', 'emergency', 'quick', 'hurry',
                        'asap', 'critical', 'deadline', 'rush'],
            'weight': 0.9,
        },
        RelevanceCategory.DARK_HUMOR: {
            'keywords': ['death', 'dark', 'macabre', 'grim', 'morbid', 'twisted',
                        'ironic', 'absurd', 'sarcasm', 'joke', 'funny', 'humor'],
            'weight': 0.8,
        }
    }

    def __init__(self, personality: Optional[Dict[str, float]] = None):
        self.personality = {
            'importance_threshold': 0.3,
            'curiosity': 0.7,
            'loyalty_sensitivity': 0.9,
            'threat_sensitivity': 0.7,
            'social_sensitivity': 0.4,
            'dark_humor_sensitivity': 0.8,
        }
        if personality:
            self._validate_personality(personality)
            self.personality.update(personality)
        
        self._compile_patterns()
        self.base_threshold = self.personality['importance_threshold']
        self.current_threshold = self.base_threshold
        
        self.relevance_memory = {}
        self.stimulus_history = deque(maxlen=1000)
        
        self.category_stats = defaultdict(lambda: {'avg_score': 0.0, 'count': 0})
        
        self.stimuli_processed = 0
        self.relevant_count = 0
        self.appraisal_triggered = 0
        
        logger.info(f"RelevanceDetector initialized with threshold {self.base_threshold}")
    
    def _validate_personality(self, personality: Dict[str, float]):
        for key, value in personality.items():
            if key not in self.personality:
                raise ValueError(f"Unknown personality parameter: {key}")
            if not 0 <= value <= 1:
                raise ValueError(f"Personality parameter {key} must be between 0 and 1, got {value}")

    def _compile_patterns(self):
        for category, pattern_data in self.RELEVANCE_PATTERNS.items():
            keywords = pattern_data.get('keywords', [])
            if keywords:
                pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
                pattern_data['compiled'] = re.compile(pattern, re.IGNORECASE)

    def check_relevance(self, stimulus: str, context=None, cognitive_load=0.0):
        start_time = time.time()
        
        if not 0 <= cognitive_load <= 1:
            raise ValueError(f"Cognitive load must be between 0 and 1, got {cognitive_load}")
        
        stimulus_hash = self._hash_stimulus(stimulus)
        self._adjust_threshold(cognitive_load)
        
        category_scores = {}
        total_score = 0.0
        
        for category, pattern in self.RELEVANCE_PATTERNS.items():
            score = self._check_category_relevance(stimulus, category)
            score = self._apply_personality_weight(category, score)
            if score > 0:
                category_scores[category] = score
                total_score += score
        
        normalized_score = min(1.0, total_score / 3.0)
        primary_category = max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else None
        
        is_relevant = normalized_score >= self.current_threshold
        triggers_appraisal = self._should_trigger_appraisal(is_relevant, normalized_score, primary_category)
        
        if normalized_score > 0.2:
            self._update_memory(stimulus_hash, normalized_score)
        
        self.stimuli_processed += 1
        if is_relevant:
            self.relevant_count += 1
        if triggers_appraisal:
            self.appraisal_triggered += 1
        
        processing_time = (time.time() - start_time) * 1000
        
        return RelevanceResult(
            is_relevant=is_relevant,
            relevance_score=normalized_score,
            primary_category=primary_category,
            categories=category_scores,
            triggers_full_appraisal=triggers_appraisal,
            processing_time_ms=processing_time,
            stimulus_hash=stimulus_hash
        )
    
    def _check_category_relevance(self, stimulus, category):
        score = 0.0
        text = stimulus.lower()
        compiled = self.RELEVANCE_PATTERNS[category].get('compiled')
        if compiled:
            matches = compiled.findall(text)
            if matches:
                score = min(1.0, len(matches) * 0.3) * self.RELEVANCE_PATTERNS[category].get('weight', 0.5)
        
        if category == RelevanceCategory.DARK_HUMOR:
            dark_words = {'death', 'dead', 'kill', 'murder'}
            humor_words = {'joke', 'funny', 'humor', 'laugh'}
            if any(word in text for word in dark_words) and any(word in text for word in humor_words):
                score += 0.8
        
        return min(1.0, score)
    
    def _apply_personality_weight(self, category, score):
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
    
    def _should_trigger_appraisal(self, is_relevant, score, primary_category):
        if not is_relevant:
            return False
        if score >= 0.6:
            return True
        if primary_category in [RelevanceCategory.THREAT, RelevanceCategory.URGENT, RelevanceCategory.VALUE_VIOLATION]:
            return True
        if len([stats for stats in self.category_stats.values() if stats['avg_score'] > 0.3]) >= 2:
            return True
        return False
    
    def _adjust_threshold(self, cognitive_load):
        load_factor = 1.0 + cognitive_load * 0.5
        self.current_threshold = min(0.8, self.base_threshold * load_factor)
    
    def _hash_stimulus(self, stimulus):
        import hashlib
        content = stimulus[:100].lower()
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _update_memory(self, stimulus_hash, score):
        if stimulus_hash in self.relevance_memory:
            entry = self.relevance_memory[stimulus_hash]
            entry.score = entry.score * 0.7 + score * 0.3
            entry.timestamp = time.time()
            entry.count += 1
        else:
            self.relevance_memory[stimulus_hash] = MemoryEntry(
                hash=stimulus_hash,
                score=score,
                timestamp=time.time()
            )
    
    def get_statistics(self):
        return {
            'stimuli_processed': self.stimuli_processed,
            'relevant_count': self.relevant_count,
            'appraisal_triggered': self.appraisal_triggered,
        }

if __name__ == "__main__":
    detector = RelevanceDetector()
    test_stimuli = [
        "The weather is nice today",
        "Someone betrayed my trust",
    ]
    for stimulus in test_stimuli:
        result = detector.check_relevance(stimulus)
        print(f"Stimulus: {stimulus}")
        print(f"Relevant: {result.is_relevant} (score: {result.relevance_score:.2f})")
        print()
