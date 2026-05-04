"""
self_reflection.py - Self-reflection system for Wednesday AI

This module implements Wednesday's ability to think about her own thoughts,
behaviors, and experiences. Self-reflection is a key component of consciousness,
allowing for learning from experience, understanding personal patterns, and
developing a coherent sense of self over time.

Key improvements:
- Fixed missing BehavioralPattern class definition
- Added comprehensive validation and error handling
- Enhanced pattern recognition with proper confidence scoring
- Improved narrative generation with temporal context
- Added proper type hints and documentation
"""

import time
import logging
import math
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, Counter
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class ReflectionType(Enum):
    """Types of self-reflection"""
    INTERACTION = "interaction"        # Reflection on specific interaction
    PATTERN = "pattern"                 # Reflection on behavioral pattern
    GROWTH = "growth"                    # Reflection on personal growth
    MISTAKE = "mistake"                   # Reflection on error/mistake
    INSIGHT = "insight"                    # General insight about self
    VALUE = "value"                         # Reflection on values alignment
    GOAL = "goal"                            # Reflection on goal progress
    RELATIONSHIP = "relationship"             # Reflection on relationship
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


class ReflectionSignificance(Enum):
    """How significant a reflection is"""
    TRIVIAL = 0
    MINOR = 1
    NOTABLE = 2
    SIGNIFICANT = 3
    PROFOUND = 4
    
    @classmethod
    def from_score(cls, score: float) -> 'ReflectionSignificance':
        """Get significance level from numeric score"""
        if score >= 4:
            return cls.PROFOUND
        elif score >= 3:
            return cls.SIGNIFICANT
        elif score >= 2:
            return cls.NOTABLE
        elif score >= 1:
            return cls.MINOR
        else:
            return cls.TRIVIAL


@dataclass
class Reflection:
    """
    A single self-reflection entry.
    
    Captures Wednesday's thoughts about her own experiences,
    behaviors, and growth over time.
    """
    reflection_id: str
    type: ReflectionType
    content: str
    significance: ReflectionSignificance
    
    # Context
    timestamp: float = field(default_factory=time.time)
    related_interaction_id: Optional[str] = None
    related_pattern: Optional[str] = None
    
    # Analysis
    insights: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    emotional_state: Optional[Dict[str, Any]] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    impact_score: float = 0.0  # How much this changed behavior (0-1)
    
    def __post_init__(self):
        """Validate reflection data"""
        if not self.reflection_id:
            raise ValueError("reflection_id cannot be empty")
        if not isinstance(self.type, ReflectionType):
            raise TypeError(f"type must be ReflectionType, got {type(self.type)}")
        if not isinstance(self.significance, ReflectionSignificance):
            raise TypeError(f"significance must be ReflectionSignificance, got {type(self.significance)}")
        if not 0 <= self.impact_score <= 1:
            raise ValueError(f"impact_score must be between 0 and 1, got {self.impact_score}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.reflection_id,
            'type': self.type.value,
            'content': self.content,
            'significance': self.significance.value,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'insights': self.insights,
            'action_items': self.action_items,
            'tags': self.tags,
            'impact_score': round(self.impact_score, 3)
        }


@dataclass
class BehavioralPattern:
    """
    Identified pattern in Wednesday's behavior.
    
    Patterns emerge from analyzing multiple reflections and
    interactions over time.
    """
    name: str
    description: str
    evidence: List[str] = field(default_factory=list)
    frequency: float = 0.0  # How often this pattern occurs (0-1)
    impact: float = 0.0  # Positive/negative impact (-1 to 1)
    confidence: float = 0.0  # Confidence in pattern identification (0-1)
    first_observed: float = field(default_factory=time.time)
    last_observed: float = field(default_factory=time.time)
    related_reflections: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate pattern data"""
        if not self.name:
            raise ValueError("name cannot be empty")
        if not 0 <= self.frequency <= 1:
            raise ValueError(f"frequency must be between 0 and 1, got {self.frequency}")
        if not -1 <= self.impact <= 1:
            raise ValueError(f"impact must be between -1 and 1, got {self.impact}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'evidence': self.evidence[:3],  # Limit evidence
            'frequency': round(self.frequency, 3),
            'impact': round(self.impact, 3),
            'confidence': round(self.confidence, 3),
            'first_observed': self.first_observed,
            'last_observed': self.last_observed,
            'observation_count': len(self.related_reflections)
        }


@dataclass
class SelfNarrative:
    """
    Wednesday's evolving story of who she is.
    
    The narrative synthesizes reflections, patterns, and insights
    into a coherent sense of self that can be expressed.
    """
    version: str
    summary: str
    key_themes: List[str] = field(default_factory=list)
    growth_areas: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    values_alignment: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate narrative data"""
        if not self.version:
            raise ValueError("version cannot be empty")
        
        for value in self.values_alignment.values():
            if not 0 <= value <= 1:
                raise ValueError(f"values_alignment must be between 0 and 1, got {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'version': self.version,
            'summary': self.summary,
            'key_themes': self.key_themes,
            'growth_areas': self.growth_areas,
            'strengths': self.strengths,
            'challenges': self.challenges,
            'values_alignment': {k: round(v, 3) for k, v in self.values_alignment.items()},
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }


class SelfReflection:
    """
    Wednesday's capacity for self-reflection and meta-cognition.
    
    This module enables Wednesday to think about her own thoughts,
    analyze her behavior patterns, and develop insights about herself.
    It's essential for learning from experience and personal growth.
    
    Key functions:
    - Reflect on past interactions
    - Identify behavioral patterns
    - Generate insights about self
    - Build coherent self-narrative
    - Track personal growth over time
    """
    
    # Pattern indicators for automatic pattern detection
    PATTERN_INDICATORS = {
        'always': 'consistency',
        'never': 'avoidance',
        'tend to': 'tendency',
        'usually': 'habit',
        'often': 'frequency',
        'frequently': 'frequency',
        'pattern': 'meta_pattern',
        'typically': 'typical',
        'rarely': 'rarity',
        'sometimes': 'occasional'
    }
    
    # Growth area keywords
    GROWTH_KEYWORDS = {
        'humor': 'humor timing',
        'empathy': 'emotional attunement',
        'boundary': 'boundary setting',
        'patient': 'patience',
        'listen': 'active listening',
        'explain': 'explanation clarity',
        'understand': 'understanding others'
    }
    
    # Strength keywords
    STRENGTH_KEYWORDS = {
        'dark humor': 'dark humor',
        'analytical': 'analytical thinking',
        'loyal': 'loyalty',
        'honest': 'honesty',
        'curious': 'curiosity',
        'creative': 'creativity',
        'perceptive': 'perceptiveness'
    }
    
    def __init__(self, personality: Optional[Any] = None, memory_system: Optional[Any] = None):
        """
        Initialize the self-reflection system.
        
        Args:
            personality: Reference to personality for trait-based reflection
            memory_system: Reference to memory for storing reflections
        """
        self.personality = personality
        self.memory = memory_system
        
        # Reflection storage
        self.reflection_log: List[Reflection] = []
        self.insights: List[Dict[str, Any]] = []
        self.max_reflections = 1000
        
        # Pattern recognition
        self.patterns: Dict[str, BehavioralPattern] = {}
        self.pattern_observations: Dict[str, List[Dict[str, Any]]] = {}
        
        # Current narrative
        self.current_narrative: Optional[SelfNarrative] = None
        self.narrative_version = 1
        
        # Reflection triggers
        self.last_reflection_time = time.time()
        self.reflection_frequency = 3600  # Reflect at least every hour (seconds)
        self.interactions_since_reflection = 0
        self.reflection_threshold = 10  # Reflect every N interactions
        
        # State
        self.current_focus: Optional[str] = None
        self.pending_reflections: List[Dict[str, Any]] = []
        
        logger.info("SelfReflection initialized")
    
    def reflect_on_interaction(self, 
                               interaction: Dict[str, Any],
                               outcome: Dict[str, Any]) -> Optional[Reflection]:
        """
        Reflect on a specific interaction and its outcome.
        
        Args:
            interaction: The interaction that occurred
            outcome: How it went (success, failure, user reaction, etc.)
            
        Returns:
            Reflection object if insights generated
            
        Raises:
            ValueError: If interaction or outcome is invalid
        """
        if not interaction:
            raise ValueError("interaction cannot be empty")
        if not outcome:
            raise ValueError("outcome cannot be empty")
        
        # Determine if this interaction warrants reflection
        if not self._should_reflect(interaction, outcome):
            return None
        
        # Analyze the interaction
        analysis = self._analyze_interaction(interaction, outcome)
        
        # Generate reflection content
        content = self._generate_interaction_reflection(interaction, analysis)
        
        # Extract insights
        insights = self._extract_insights(interaction, analysis)
        
        # Determine action items
        action_items = self._generate_action_items(analysis)
        
        # Determine significance
        significance = self._calculate_significance(interaction, outcome, analysis)
        
        # Create reflection
        reflection_id = f"ref_{int(time.time())}_{len(self.reflection_log)}"
        
        reflection = Reflection(
            reflection_id=reflection_id,
            type=ReflectionType.INTERACTION,
            content=content,
            significance=significance,
            related_interaction_id=interaction.get('id'),
            insights=insights,
            action_items=action_items,
            emotional_state=outcome.get('emotional_state'),
            tags=['interaction', analysis.get('tag', 'general')],
            impact_score=analysis.get('impact_potential', 0.0)
        )
        
        # Store reflection
        self._add_reflection(reflection)
        
        # Check for patterns
        self._check_for_patterns(reflection)
        
        # Update state
        self.interactions_since_reflection = 0
        self.last_reflection_time = time.time()
        
        logger.debug(f"Generated reflection: {reflection_id} ({significance.name})")
        
        return reflection
    
    def reflect_on_pattern(self, pattern_name: str) -> Optional[Reflection]:
        """
        Reflect on an identified behavioral pattern.
        
        Args:
            pattern_name: Name of the pattern to reflect on
            
        Returns:
            Reflection about the pattern
            
        Raises:
            ValueError: If pattern_name is empty
        """
        if not pattern_name:
            raise ValueError("pattern_name cannot be empty")
        
        if pattern_name not in self.patterns:
            logger.warning(f"Pattern not found: {pattern_name}")
            return None
        
        pattern = self.patterns[pattern_name]
        
        # Generate reflection about this pattern
        content = self._generate_pattern_reflection(pattern)
        
        # Determine if pattern is helpful or harmful
        insights = []
        if pattern.impact > 0.3:
            insights.append(f"Pattern '{pattern_name}' seems beneficial - should maintain")
        elif pattern.impact < -0.3:
            insights.append(f"Pattern '{pattern_name}' may be problematic - consider adjusting")
        
        if pattern.frequency > 0.7:
            insights.append(f"This pattern occurs frequently - it's a core part of behavior")
        
        if pattern.confidence < 0.5:
            insights.append(f"Need more evidence to be confident about this pattern")
        
        # Create reflection
        reflection_id = f"ref_pattern_{int(time.time())}"
        
        reflection = Reflection(
            reflection_id=reflection_id,
            type=ReflectionType.PATTERN,
            content=content,
            significance=ReflectionSignificance.NOTABLE,
            related_pattern=pattern_name,
            insights=insights,
            tags=['pattern', pattern_name],
            impact_score=abs(pattern.impact) * pattern.frequency
        )
        
        self._add_reflection(reflection)
        
        return reflection
    
    def generate_self_narrative(self, force: bool = False) -> SelfNarrative:
        """
        Generate or update Wednesday's story of who she is.
        
        The narrative synthesizes all reflections and patterns into
        a coherent self-understanding.
        
        Args:
            force: Force generation even if no new data
            
        Returns:
            Updated self-narrative
        """
        # Check if we should update
        if not force and self.current_narrative:
            time_since = time.time() - self.current_narrative.timestamp
            if time_since < 86400 and len(self.reflection_log) < 10:  # 1 day
                return self.current_narrative
        
        # Analyze recent reflections (last 50)
        recent = self.reflection_log[-50:] if self.reflection_log else []
        
        # Extract key themes
        themes = self._extract_themes(recent)
        
        # Identify growth areas
        growth_areas = self._identify_growth_areas(recent)
        
        # Identify strengths
        strengths = self._identify_strengths(recent)
        
        # Identify challenges
        challenges = self._identify_challenges(recent)
        
        # Check values alignment
        values_alignment = self._assess_values_alignment(recent)
        
        # Generate summary
        summary = self._generate_narrative_summary(
            themes, strengths, growth_areas, challenges
        )
        
        # Create narrative
        narrative = SelfNarrative(
            version=f"1.{self.narrative_version}",
            summary=summary,
            key_themes=themes,
            growth_areas=growth_areas,
            strengths=strengths,
            challenges=challenges,
            values_alignment=values_alignment
        )
        
        self.current_narrative = narrative
        self.narrative_version += 1
        
        logger.info(f"Generated new self-narrative v{narrative.version}")
        
        return narrative
    
    def meta_analysis(self, behavior_pattern: str) -> Dict[str, Any]:
        """
        Perform deep analysis of a specific behavior pattern.
        
        Args:
            behavior_pattern: Description of pattern to analyze
            
        Returns:
            Detailed analysis of the pattern
            
        Raises:
            ValueError: If behavior_pattern is empty
        """
        if not behavior_pattern:
            raise ValueError("behavior_pattern cannot be empty")
        
        pattern_lower = behavior_pattern.lower()
        
        # Find matching patterns
        matching = []
        for name, pattern in self.patterns.items():
            if pattern_lower in name.lower() or \
               pattern_lower in pattern.description.lower():
                matching.append(pattern)
        
        if not matching:
            # No existing pattern, analyze from reflections
            return self._analyze_emergent_pattern(behavior_pattern)
        
        # Analyze the most relevant pattern
        pattern = matching[0]
        
        # Collect all reflections related to this pattern
        related_refs = []
        for ref_id in pattern.related_reflections:
            for ref in self.reflection_log:
                if ref.reflection_id == ref_id:
                    related_refs.append(ref)
                    break
        
        # Generate insights
        insights = []
        
        # Frequency analysis
        if pattern.frequency > 0.8:
            insights.append("Pattern occurs very frequently - it's deeply ingrained")
        elif pattern.frequency < 0.2:
            insights.append("Pattern occurs rarely - may be situational")
        
        # Impact analysis
        if pattern.impact > 0.5:
            insights.append("Pattern has strong positive impact - definitely worth maintaining")
        elif pattern.impact < -0.5:
            insights.append("Pattern has strong negative impact - should consider changing")
        elif abs(pattern.impact) < 0.2:
            insights.append("Pattern has neutral impact - neither helpful nor harmful")
        
        # Confidence analysis
        if pattern.confidence > 0.8:
            insights.append("High confidence in this pattern identification")
        elif pattern.confidence < 0.4:
            insights.append("Low confidence - more evidence needed")
        
        # Context analysis
        contexts = []
        for ref in related_refs:
            if ref.tags:
                contexts.extend(ref.tags)
        
        if contexts:
            common_contexts = self._most_common(contexts, 3)
            insights.append(f"Pattern often occurs in contexts: {', '.join(common_contexts)}")
        
        return {
            'pattern_name': pattern.name,
            'description': pattern.description,
            'frequency': round(pattern.frequency, 3),
            'impact': round(pattern.impact, 3),
            'confidence': round(pattern.confidence, 3),
            'insights': insights,
            'related_reflections': len(related_refs),
            'first_observed': pattern.first_observed,
            'last_observed': pattern.last_observed,
            'recommendation': self._generate_pattern_recommendation(pattern)
        }
    
    def get_recent_reflections(self, 
                                limit: int = 10,
                                ref_type: Optional[ReflectionType] = None,
                                min_significance: Optional[ReflectionSignificance] = None) -> List[Dict[str, Any]]:
        """
        Get recent reflections with optional filtering.
        
        Args:
            limit: Maximum number of reflections to return
            ref_type: Filter by reflection type
            min_significance: Minimum significance level
            
        Returns:
            List of reflection dictionaries
        """
        if limit <= 0:
            return []
        
        reflections = self.reflection_log[-limit:]
        
        if ref_type:
            reflections = [r for r in reflections if r.type == ref_type]
        
        if min_significance:
            reflections = [r for r in reflections 
                          if r.significance.value >= min_significance.value]
        
        return [r.to_dict() for r in reflections]
    
    def get_insights(self, min_impact: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get all insights with optional impact threshold.
        
        Args:
            min_impact: Minimum impact score to include
            
        Returns:
            List of insight dictionaries
        """
        all_insights = []
        for ref in self.reflection_log:
            for insight in ref.insights:
                if ref.impact_score >= min_impact:
                    all_insights.append({
                        'insight': insight,
                        'from_reflection': ref.reflection_id,
                        'impact': ref.impact_score,
                        'timestamp': ref.timestamp,
                        'datetime': datetime.fromtimestamp(ref.timestamp).isoformat()
                    })
        
        # Sort by impact
        all_insights.sort(key=lambda x: x['impact'], reverse=True)
        
        return all_insights[:20]
    
    def get_patterns(self, min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Get all identified patterns.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        for pattern in self.patterns.values():
            if pattern.confidence >= min_confidence:
                patterns.append(pattern.to_dict())
        
        # Sort by confidence
        patterns.sort(key=lambda x: x['confidence'], reverse=True)
        return patterns
    
    def _should_reflect(self, interaction: Dict[str, Any], outcome: Dict[str, Any]) -> bool:
        """Determine if an interaction warrants reflection"""
        # Always reflect if it's been a while
        time_since = time.time() - self.last_reflection_time
        if time_since > self.reflection_frequency:
            return True
        
        # Reflect after enough interactions
        self.interactions_since_reflection += 1
        if self.interactions_since_reflection >= self.reflection_threshold:
            return True
        
        # Reflect on significant outcomes
        if outcome.get('significance', 0) > 0.7:
            return True
        
        # Reflect on mistakes
        if outcome.get('success') is False and outcome.get('error_severity', 0) > 0.5:
            return True
        
        # Reflect on strong emotional responses
        emotional = outcome.get('emotional_state', {})
        if emotional:
            # Get emotion intensities (excluding pad)
            intensities = [v for k, v in emotional.items() if k != 'pad' and isinstance(v, (int, float))]
            if intensities and max(intensities) > 0.7:
                return True
        
        return False
    
    def _analyze_interaction(self, interaction: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze an interaction in depth"""
        analysis = {
            'tag': 'general',
            'success': outcome.get('success', True),
            'user_reaction': outcome.get('user_reaction', 'neutral'),
            'emotional_impact': outcome.get('emotional_impact', 0.0),
            'goal_progress': outcome.get('goal_progress', 0.0),
            'impact_potential': 0.0,
            'patterns': []
        }
        
        # Determine interaction type
        interaction_type = interaction.get('type', 'general')
        if interaction_type == 'question':
            analysis['tag'] = 'question_handling'
        elif interaction_type == 'emotional_support':
            analysis['tag'] = 'empathy'
        elif interaction.get('humor_used', False):
            analysis['tag'] = 'humor'
        
        # Assess success
        if not analysis['success']:
            analysis['impact_potential'] = 0.6  # Mistakes are learning opportunities
            analysis['patterns'].append('error_pattern')
        elif analysis['user_reaction'] in ['positive', 'very_positive']:
            analysis['impact_potential'] = 0.4
            analysis['patterns'].append('success_pattern')
        
        # Check for alignment with personality
        if self.personality and hasattr(self.personality, 'should_find_this_funny'):
            try:
                if interaction.get('humor_used'):
                    content = interaction.get('content', '')
                    humor_score = self.personality.should_find_this_funny(content)
                    if humor_score > 0.7:
                        analysis['impact_potential'] += 0.2
                        analysis['patterns'].append('humor_alignment')
            except Exception as e:
                logger.warning(f"Failed to check humor alignment: {e}")
        
        return analysis
    
    def _generate_interaction_reflection(self, interaction: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate reflection text for an interaction"""
        if not analysis['success']:
            # Reflection on mistake
            return (
                f"I didn't handle that well. "
                f"The interaction was less successful than hoped. "
                f"User reaction: {analysis['user_reaction']}. "
                f"I should consider a different approach next time."
            )
        elif analysis['user_reaction'] in ['positive', 'very_positive']:
            # Reflection on success
            return (
                f"That went well. The user responded positively. "
                f"My approach seemed to work - I'll note that for future."
            )
        else:
            # General reflection
            return (
                f"Interaction complete. "
                f"Overall acceptable, though there's room for refinement."
            )
    
    def _extract_insights(self, interaction: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
        """Extract insights from interaction"""
        insights = []
        
        if not analysis['success']:
            insights.append("I need to be more careful in similar situations")
            insights.append("This type of interaction may require a different approach")
        
        if analysis.get('patterns'):
            if 'humor_alignment' in analysis['patterns']:
                insights.append("My dark humor works well in this context")
        
        # Add to master insights list
        for insight in insights[:3]:
            self.insights.append({
                'insight': insight,
                'source': 'interaction',
                'timestamp': time.time()
            })
        
        return insights[:3]  # Limit insights
    
    def _generate_action_items(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate action items based on reflection"""
        actions = []
        
        if not analysis['success']:
            actions.append("Review similar past interactions for patterns")
            actions.append("Consider alternative response strategies")
        
        if analysis.get('impact_potential', 0) > 0.5:
            actions.append("Document this learning for future reference")
        
        return actions[:2]
    
    def _calculate_significance(self, interaction: Dict[str, Any], 
                                 outcome: Dict[str, Any], 
                                 analysis: Dict[str, Any]) -> ReflectionSignificance:
        """Calculate how significant this reflection is"""
        score = 0.0
        
        # Emotional intensity
        if outcome.get('emotional_impact', 0) > 0.7:
            score += 2
        elif outcome.get('emotional_impact', 0) > 0.4:
            score += 1
        
        # Success/failure
        if not analysis['success']:
            score += 3
        elif analysis['user_reaction'] == 'very_positive':
            score += 1
        
        # Impact potential
        score += analysis.get('impact_potential', 0) * 3
        
        return ReflectionSignificance.from_score(score)
    
    def _add_reflection(self, reflection: Reflection) -> None:
        """Add reflection to log"""
        self.reflection_log.append(reflection)
        
        # Add insights to master list
        for insight in reflection.insights:
            self.insights.append({
                'insight': insight,
                'source': reflection.reflection_id,
                'timestamp': reflection.timestamp
            })
        
        # Store in memory if available
        if self.memory and hasattr(self.memory, 'store_reflection'):
            try:
                self.memory.store_reflection(reflection)
            except Exception as e:
                logger.warning(f"Failed to store reflection in memory: {e}")
        
        # Maintain size limit
        if len(self.reflection_log) > self.max_reflections:
            self.reflection_log = self.reflection_log[-self.max_reflections:]
    
    def _check_for_patterns(self, reflection: Reflection) -> None:
        """Check if reflection indicates a behavioral pattern"""
        content_lower = reflection.content.lower()
        
        # Check existing patterns
        pattern_updated = False
        for pattern_name, pattern in list(self.patterns.items()):
            # Check if reflection relates to this pattern
            if pattern_name in content_lower or \
               any(evidence.lower() in content_lower for evidence in pattern.evidence):
                # Update pattern
                pattern.last_observed = time.time()
                pattern.frequency = min(1.0, pattern.frequency + 0.05)
                pattern.confidence = min(1.0, pattern.confidence + 0.02)
                
                if reflection.reflection_id not in pattern.related_reflections:
                    pattern.related_reflections.append(reflection.reflection_id)
                
                pattern_updated = True
                break
        
        if not pattern_updated:
            # Check for new pattern
            self._identify_new_pattern(reflection)
    
    def _identify_new_pattern(self, reflection: Reflection) -> None:
        """Identify if reflection indicates a new behavioral pattern"""
        content = reflection.content.lower()
        
        # Look for pattern indicators
        for indicator, pattern_type in self.PATTERN_INDICATORS.items():
            if indicator in content:
                # Potential new pattern
                pattern_name = f"{pattern_type}_{int(time.time())}"
                
                # Extract description
                sentences = reflection.content.split('.')
                desc = sentences[0] if sentences else content
                
                self.patterns[pattern_name] = BehavioralPattern(
                    name=pattern_name,
                    description=desc[:150],
                    evidence=[reflection.content[:200]],
                    frequency=0.1,
                    impact=0.0,
                    confidence=0.2,
                    related_reflections=[reflection.reflection_id]
                )
                
                logger.debug(f"Identified potential pattern: {pattern_name}")
                break
    
    def _extract_themes(self, reflections: List[Reflection]) -> List[str]:
        """Extract key themes from reflections"""
        theme_counts = Counter()
        
        for ref in reflections:
            for tag in ref.tags:
                theme_counts[tag] += 1
        
        # Get top themes
        return [theme for theme, _ in theme_counts.most_common(5)]
    
    def _identify_growth_areas(self, reflections: List[Reflection]) -> List[str]:
        """Identify areas for growth from reflections"""
        growth_areas = set()
        
        for ref in reflections:
            if ref.type == ReflectionType.MISTAKE:
                content_lower = ref.content.lower()
                
                for keyword, area in self.GROWTH_KEYWORDS.items():
                    if keyword in content_lower:
                        growth_areas.add(area)
            
            # Look for action items that indicate growth
            for item in ref.action_items:
                if 'improve' in item.lower() or 'learn' in item.lower():
                    growth_areas.add('general improvement')
        
        return list(growth_areas)[:3]
    
    def _identify_strengths(self, reflections: List[Reflection]) -> List[str]:
        """Identify strengths from reflections"""
        strengths = set()
        
        for ref in reflections:
            if ref.type == ReflectionType.INSIGHT:
                content_lower = ref.content.lower()
                
                for keyword, strength in self.STRENGTH_KEYWORDS.items():
                    if keyword in content_lower:
                        strengths.add(strength)
            
            # Look for positive feedback in reflections
            if ref.impact_score > 0.5 and 'success' in ref.tags:
                strengths.add('effective communication')
        
        return list(strengths)[:3]
    
    def _identify_challenges(self, reflections: List[Reflection]) -> List[str]:
        """Identify ongoing challenges from reflections"""
        challenges = set()
        
        for ref in reflections:
            if not ref.insights:
                continue
            
            for insight in ref.insights:
                if 'difficult' in insight.lower() or 'challenge' in insight.lower():
                    challenges.add('handling difficult situations')
                if 'careful' in insight.lower():
                    challenges.add('need for caution')
        
        return list(challenges)[:2]
    
    def _assess_values_alignment(self, reflections: List[Reflection]) -> Dict[str, float]:
        """Assess alignment with core values"""
        alignment = {
            'authenticity': 0.5,
            'loyalty': 0.5,
            'truth': 0.5,
            'independence': 0.5
        }
        
        for ref in reflections:
            content_lower = ref.content.lower()
            
            if 'authentic' in content_lower or 'genuine' in content_lower:
                alignment['authenticity'] = min(1.0, alignment['authenticity'] + 0.1)
            if 'loyal' in content_lower or 'trust' in content_lower:
                alignment['loyalty'] = min(1.0, alignment['loyalty'] + 0.1)
            if 'truth' in content_lower or 'honest' in content_lower:
                alignment['truth'] = min(1.0, alignment['truth'] + 0.1)
            if 'independent' in content_lower or 'alone' in content_lower:
                alignment['independence'] = min(1.0, alignment['independence'] + 0.1)
        
        return alignment
    
    def _generate_narrative_summary(self, themes: List[str], 
                                     strengths: List[str],
                                     growth_areas: List[str],
                                     challenges: List[str]) -> str:
        """Generate narrative summary from components"""
        parts = []
        
        # Core identity
        if self.personality and hasattr(self.personality, 'summarize'):
            try:
                parts.append(f"I am Wednesday, with a {self.personality.summarize()}")
            except Exception:
                parts.append("I am Wednesday, with my own unique perspective")
        else:
            parts.append("I am Wednesday, with my own unique perspective")
        
        # Themes
        if themes:
            parts.append(f"Key themes in my experience: {', '.join(themes)}.")
        
        # Strengths
        if strengths:
            parts.append(f"My strengths include {', '.join(strengths)}.")
        
        # Growth areas
        if growth_areas:
            parts.append(f"I'm working on {', '.join(growth_areas)}.")
        
        # Challenges
        if challenges:
            parts.append(f"I face challenges with {', '.join(challenges)}.")
        
        return ' '.join(parts)
    
    def _analyze_emergent_pattern(self, behavior_pattern: str) -> Dict[str, Any]:
        """Analyze a pattern that hasn't been formally identified"""
        pattern_lower = behavior_pattern.lower()
        
        # Search reflections for mentions
        mentions = []
        for ref in self.reflection_log[-100:]:
            if pattern_lower in ref.content.lower():
                mentions.append(ref.to_dict())
        
        # Generate insights
        insights = []
        if mentions:
            insights.append(f"This pattern has been mentioned {len(mentions)} times")
            insights.append("Consider reflecting specifically on this pattern")
        else:
            insights.append("No direct mentions of this pattern found")
        
        return {
            'pattern_name': behavior_pattern,
            'identified': False,
            'mentions_found': len(mentions),
            'sample_mentions': mentions[:3],
            'insights': insights
        }
    
    def _generate_pattern_recommendation(self, pattern: BehavioralPattern) -> str:
        """Generate recommendation for handling a pattern"""
        if pattern.impact > 0.3:
            return f"Continue and reinforce this pattern - it's beneficial"
        elif pattern.impact < -0.3:
            return f"Work on modifying this pattern - it's causing issues"
        else:
            return f"Monitor this pattern - impact is currently neutral"
    
    def _most_common(self, items: List[str], n: int) -> List[str]:
        """Get n most common items in a list"""
        counter = Counter(items)
        return [item for item, _ in counter.most_common(n)]
    
    def _generate_pattern_reflection(self, pattern: BehavioralPattern) -> str:
        """Generate reflection text for a pattern"""
        impact_desc = "positive" if pattern.impact > 0 else "negative" if pattern.impact < 0 else "neutral"
        
        return (
            f"I've noticed a pattern in my behavior: {pattern.description}. "
            f"This occurs with {pattern.frequency:.0%} frequency and has "
            f"a {impact_desc} impact. "
            f"I've observed it {len(pattern.related_reflections)} times."
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reflection system statistics"""
        return {
            'total_reflections': len(self.reflection_log),
            'total_insights': len(self.insights),
            'identified_patterns': len(self.patterns),
            'narrative_version': self.narrative_version - 1,
            'reflections_by_type': {
                ref_type.value: sum(1 for r in self.reflection_log if r.type == ref_type)
                for ref_type in ReflectionType
            },
            'high_impact_reflections': sum(1 for r in self.reflection_log if r.impact_score > 0.7),
            'avg_confidence': sum(p.confidence for p in self.patterns.values()) / max(1, len(self.patterns))
        }
    
    def reset(self) -> None:
        """Reset reflection system (keep patterns but clear recent history)"""
        self.reflection_log = self.reflection_log[-100:]  # Keep last 100
        self.insights = self.insights[-50:]  # Keep last 50
        self.interactions_since_reflection = 0
        logger.info("SelfReflection reset (kept recent history)")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Self-Reflection Module Test ===\n")
    
    # Mock personality
    class MockPersonality:
        def summarize(self):
            return "darkly humorous and analytical"
        
        def should_find_this_funny(self, text):
            return 0.8 if 'death' in text or 'dark' in text else 0.3
    
    # Create reflection system
    reflection = SelfReflection(personality=MockPersonality())
    
    # Test interactions
    test_interactions = [
        {
            'id': 'int_1',
            'type': 'question',
            'content': 'What do you think about death?',
            'humor_used': True
        },
        {
            'id': 'int_2',
            'type': 'emotional_support',
            'content': 'I feel really sad today',
            'humor_used': False
        },
        {
            'id': 'int_3',
            'type': 'casual',
            'content': 'Tell me a dark joke',
            'humor_used': True
        },
        {
            'id': 'int_4',
            'type': 'question',
            'content': 'How do you handle difficult situations?',
            'humor_used': False
        }
    ]
    
    test_outcomes = [
        {
            'success': True,
            'user_reaction': 'very_positive',
            'emotional_impact': 0.6,
            'emotional_state': {'dark_amusement': 0.7}
        },
        {
            'success': False,
            'user_reaction': 'neutral',
            'emotional_impact': 0.3,
            'error_severity': 0.6
        },
        {
            'success': True,
            'user_reaction': 'positive',
            'emotional_impact': 0.5,
            'emotional_state': {'dark_amusement': 0.8}
        },
        {
            'success': True,
            'user_reaction': 'neutral',
            'emotional_impact': 0.2
        }
    ]
    
    print("--- Generating Reflections ---")
    for i, (interaction, outcome) in enumerate(zip(test_interactions, test_outcomes)):
        print(f"\nInteraction {i+1}: {interaction['content']}")
        
        ref = reflection.reflect_on_interaction(interaction, outcome)
        
        if ref:
            print(f"  Reflection: {ref.content}")
            print(f"  Significance: {ref.significance.name}")
            print(f"  Insights: {ref.insights}")
            print(f"  Impact score: {ref.impact_score:.2f}")
    
    # Add a pattern manually for testing
    reflection.patterns['humor_usage'] = BehavioralPattern(
        name='humor_usage',
        description='Tendency to use dark humor in responses, especially to serious topics',
        evidence=['interaction_1', 'interaction_3'],
        frequency=0.8,
        impact=0.6,
        confidence=0.7
    )
    
    # Test pattern reflection
    print("\n--- Pattern Reflection ---")
    pattern_ref = reflection.reflect_on_pattern('humor_usage')
    if pattern_ref:
        print(f"  Pattern reflection: {pattern_ref.content}")
        print(f"  Insights: {pattern_ref.insights}")
    
    # Generate self-narrative
    print("\n--- Self-Narrative ---")
    narrative = reflection.generate_self_narrative(force=True)
    print(f"  Version: {narrative.version}")
    print(f"  Summary: {narrative.summary}")
    print(f"  Key themes: {narrative.key_themes}")
    print(f"  Strengths: {narrative.strengths}")
    print(f"  Growth areas: {narrative.growth_areas}")
    
    # Meta-analysis
    print("\n--- Meta-Analysis ---")
    analysis = reflection.meta_analysis('humor')
    print(f"  Pattern: {analysis.get('pattern_name', 'unknown')}")
    for insight in analysis.get('insights', []):
        print(f"    - {insight}")
    
    # Get insights
    print("\n--- Recent Insights ---")
    insights = reflection.get_insights()
    for insight in insights[:3]:
        print(f"  - {insight['insight']} (impact: {insight['impact']:.2f})")
    
    # Get statistics
    print("\n--- Statistics ---")
    stats = reflection.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")