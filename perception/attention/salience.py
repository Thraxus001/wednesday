"""
Determines what's important in the current input.
Like your brain filtering out background noise - but with Wednesday's priorities.
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)

class SalienceFactor(Enum):
    NOVELTY = "novelty"           # New/unexpected information
    GOAL_RELEVANCE = "goal"        # Related to current objectives
    EMOTIONAL = "emotional"        # Emotional charge
    SOCIAL = "social"              # Social significance (for Wednesday, often negative)
    THREAT = "threat"              # Potential dangers or conflicts
    PATTERN_BREAK = "pattern_break" # Violates expected patterns
    SELF_REFERENCE = "self_reference"  # Direct reference to Wednesday
    URGENCY = "urgency"            # Time-sensitive or immediate need

@dataclass
class SalienceScore:
    """Structured salience output for other modules"""
    element_id: str
    overall_score: float
    factor_scores: Dict[SalienceFactor, float]
    attention_duration: float  # How long to maintain focus (seconds)
    requires_immediate: bool   # For urgent/threatening inputs
    timestamp: datetime
    
class SalienceDetector:
    """
    Determines what's important enough for Wednesday to notice.
    She's selectively attentive - only truly relevant details deserve her focus.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.context = {
            'current_focus': None,
            'recent_saliences': [],  # Track what's been noticed recently
            'attention_history': {},  # Per-element attention allocation
            'recent_inputs': []       # Last N inputs for novelty comparison
        }
        
        # Weight factors based on context and personality
        self.base_weights = {
            SalienceFactor.NOVELTY: 0.20,
            SalienceFactor.GOAL_RELEVANCE: 0.25,
            SalienceFactor.EMOTIONAL: 0.15,
            SalienceFactor.SOCIAL: 0.15,
            SalienceFactor.THREAT: 0.25,  # Increased - Wednesday is cautious
            SalienceFactor.PATTERN_BREAK: 0.15,
            SalienceFactor.SELF_REFERENCE: 0.30,  # Important when directly addressed
            SalienceFactor.URGENCY: 0.35  # Highest for urgent matters
        }
        
        # Wednesday's particular interests
        self.wednesday_bias = {
            'sarcasm_bonus': 1.5,      # She notices wit
            'injustice_sensitivity': 1.4,  # Unfairness catches her attention
            'emotional_detachment': 0.6,  # Less moved by sentimentality
            'mystery_interest': 1.3,    # Drawn to puzzles and secrets
            'authority_skepticism': 1.2  # Notices when authority figures are involved
        }
        
        # Threat keywords and patterns
        self.threat_patterns = [
            r'danger|kill|hurt|attack|threat|weapon',
            r'lie|deceive|betray|manipulate|trap',
            r'secret|hidden|cover.?up|conspiracy',
            r'authority|police|principal|parent'
        ]
        
        # Self-reference patterns (how people address Wednesday)
        self.self_ref_patterns = [
            r'\bwednesday\b', r'\byou\b', r'\byour\b', 
            r'ms\.\s*addams', r'miss\s*addams'
        ]
        
        # Urgency indicators
        self.urgency_patterns = [
            r'now|quick|hurry|emergency|immediate|asap',
            r'help|please|need you|come here',
            r'problem|trouble|crisis|urgent'
        ]
        
        logger.info("SalienceDetector initialized with Wednesday's unique perspective")
    
    def calculate_salience(self, 
                          input_stream: Dict[str, Any],
                          current_goals: Optional[List[str]] = None,
                          emotional_state: Optional[Dict] = None) -> List[SalienceScore]:
        """
        Score each element in the input stream for importance.
        
        Args:
            input_stream: Parsed input elements from perception modules
            current_goals: Active goals from executive module
            emotional_state: Current emotional context from emotion module
            
        Returns:
            Ranked list of salience scores
        """
        if not input_stream:
            return []
        
        results = []
        
        for element_id, element_data in input_stream.items():
            # Skip empty elements
            if not element_data:
                continue
                
            # Calculate individual salience factors
            factor_scores = {}
            
            # Novelty: How different from recent inputs?
            factor_scores[SalienceFactor.NOVELTY] = self._compute_novelty(
                element_data, 
                self.context['attention_history'].get(element_id, [])
            )
            
            # Goal relevance: Does this help current objectives?
            if current_goals:
                factor_scores[SalienceFactor.GOAL_RELEVANCE] = self._compute_goal_relevance(
                    element_data, 
                    current_goals
                )
            else:
                factor_scores[SalienceFactor.GOAL_RELEVANCE] = 0.3  # Default moderate relevance
            
            # Emotional content: Is this emotionally charged?
            factor_scores[SalienceFactor.EMOTIONAL] = self._compute_emotional_impact(
                element_data,
                emotional_state or {}
            )
            
            # Social significance: Does it involve relationships/power?
            factor_scores[SalienceFactor.SOCIAL] = self._compute_social_significance(
                element_data
            )
            
            # Threat detection: Any danger or conflict?
            factor_scores[SalienceFactor.THREAT] = self._compute_threat_level(
                element_data
            )
            
            # Pattern break: Violates expectations?
            factor_scores[SalienceFactor.PATTERN_BREAK] = self._compute_pattern_violation(
                element_data
            )
            
            # Self-reference: Is Wednesday being addressed?
            factor_scores[SalienceFactor.SELF_REFERENCE] = self._compute_self_reference(
                element_data
            )
            
            # Urgency: Time-sensitive?
            factor_scores[SalienceFactor.URGENCY] = self._compute_urgency(
                element_data
            )
            
            # Calculate weighted score
            overall = self._aggregate_scores(factor_scores)
            
            # Apply Wednesday's personality bias
            overall = self._apply_wednesday_bias(overall, element_data)
            
            # Determine attention duration (how long to hold focus in seconds)
            attention_duration = self._calculate_attention_duration(
                overall, 
                factor_scores
            )
            
            # Check if requires immediate response
            requires_immediate = self._check_immediate_response(
                factor_scores, element_data
            )
            
            results.append(SalienceScore(
                element_id=element_id,
                overall_score=overall,
                factor_scores=factor_scores,
                attention_duration=attention_duration,
                requires_immediate=requires_immediate,
                timestamp=datetime.now()
            ))
        
        # Sort by overall score
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Update context with new saliences
        self._update_context(results, input_stream)
        
        return results
    
    def get_foreground(self, 
                      input_stream: Dict[str, Any],
                      threshold: float = 0.5,
                      max_elements: int = 5) -> Dict[str, Any]:
        """
        Return only elements above salience threshold.
        Wednesday doesn't waste cognitive cycles on trivialities.
        """
        if not input_stream:
            return {}
        
        # Get salience scores
        salience_scores = self.calculate_salience(input_stream)
        
        # Filter by threshold and take top N
        foreground_elements = {}
        for score in salience_scores[:max_elements]:
            if score.overall_score >= threshold:
                foreground_elements[score.element_id] = {
                    'data': input_stream[score.element_id],
                    'salience': score.overall_score,
                    'attention_duration': score.attention_duration,
                    'requires_immediate': score.requires_immediate,
                    'factor_scores': {
                        k.value: v for k, v in score.factor_scores.items()
                    }
                }
        
        logger.debug(f"Foreground selected {len(foreground_elements)} elements")
        return foreground_elements
    
    def _compute_novelty(self, element: Any, history: List) -> float:
        """How new/different is this element from what we've seen before?"""
        if not history:
            return 0.8  # New element is novel
        
        # Convert element to string for comparison
        element_str = str(element).lower()
        
        # Check recent inputs for similarity
        recent_inputs = self.context.get('recent_inputs', [])[-5:]  # Last 5 inputs
        
        if not recent_inputs:
            return 0.7
        
        # Calculate max similarity to recent inputs
        max_similarity = 0.0
        for recent in recent_inputs:
            recent_str = str(recent).lower()
            # Simple word overlap similarity
            element_words = set(element_str.split())
            recent_words = set(recent_str.split())
            if element_words and recent_words:
                intersection = element_words & recent_words
                union = element_words | recent_words
                similarity = len(intersection) / len(union) if union else 0
                max_similarity = max(max_similarity, similarity)
        
        # Novelty is inverse of similarity
        novelty = 1.0 - max_similarity
        return max(0.1, min(1.0, novelty))
    
    def _compute_goal_relevance(self, element: Any, goals: List[str]) -> float:
        """How relevant is this to current active goals?"""
        if not goals:
            return 0.3
        
        element_str = str(element).lower()
        max_relevance = 0.0
        
        for goal in goals:
            goal_lower = goal.lower()
            goal_words = set(goal_lower.split())
            element_words = set(element_str.split())
            
            if goal_words and element_words:
                # Check for keyword overlap
                overlap = goal_words & element_words
                if overlap:
                    relevance = len(overlap) / len(goal_words)
                    max_relevance = max(max_relevance, relevance)
        
        return max_relevance
    
    def _compute_emotional_impact(self, element: Any, emotional_state: Dict) -> float:
        """Does this element have emotional significance?"""
        element_str = str(element).lower()
        
        # Emotionally charged keywords
        emotion_keywords = {
            'high': ['love', 'hate', 'death', 'murder', 'secret', 'truth', 'lie'],
            'medium': ['angry', 'happy', 'sad', 'scared', 'surprise'],
            'low': ['okay', 'fine', 'normal', 'regular']
        }
        
        score = 0.0
        for level, words in emotion_keywords.items():
            for word in words:
                if word in element_str:
                    if level == 'high':
                        score = max(score, 0.9)
                    elif level == 'medium':
                        score = max(score, 0.6)
                    else:
                        score = max(score, 0.3)
        
        # Apply Wednesday's emotional detachment
        score *= self.wednesday_bias['emotional_detachment']
        
        return score
    
    def _compute_social_significance(self, element: Any) -> float:
        """Social relevance - relationships, status, interpersonal dynamics"""
        element_str = str(element).lower()
        
        social_indicators = {
            'high': ['betray', 'manipulate', 'injustice', 'unfair', 'corrupt'],
            'medium': ['friend', 'enemy', 'family', 'teacher', 'parent'],
            'low': ['people', 'person', 'group', 'they', 'them']
        }
        
        score = 0.0
        for level, words in social_indicators.items():
            for word in words:
                if word in element_str:
                    if level == 'high':
                        score = max(score, 0.8)
                    elif level == 'medium':
                        score = max(score, 0.5)
                    else:
                        score = max(score, 0.2)
        
        # Wednesday is more attuned to injustice
        if 'unfair' in element_str or 'injustice' in element_str:
            score *= self.wednesday_bias['injustice_sensitivity']
        
        return score
    
    def _compute_threat_level(self, element: Any) -> float:
        """Potential danger or conflict"""
        element_str = str(element).lower()
        
        threat_score = 0.0
        for pattern in self.threat_patterns:
            if re.search(pattern, element_str, re.IGNORECASE):
                threat_score += 0.3
        
        # Specific threat indicators
        if any(word in element_str for word in ['kill', 'die', 'dead', 'death']):
            threat_score += 0.5
        if any(word in element_str for word in ['weapon', 'gun', 'knife', 'poison']):
            threat_score += 0.4
        if any(word in element_str for word in ['secret', 'conspiracy', 'cover']):
            threat_score += 0.2  # Secrets can be dangerous
        
        return min(1.0, threat_score)
    
    def _compute_pattern_violation(self, element: Any) -> float:
        """Does this break expected patterns?"""
        element_str = str(element).lower()
        
        # Words that indicate something unexpected
        violation_indicators = [
            'unexpected', 'surprising', 'strange', 'weird', 'odd',
            'never', 'first time', 'different', 'unusual', 'peculiar'
        ]
        
        for indicator in violation_indicators:
            if indicator in element_str:
                return 0.8
        
        return 0.2  # Default low pattern violation
    
    def _compute_self_reference(self, element: Any) -> float:
        """Is Wednesday being directly addressed or referenced?"""
        element_str = str(element).lower()
        
        for pattern in self.self_ref_patterns:
            if re.search(pattern, element_str, re.IGNORECASE):
                return 1.0
        
        return 0.0
    
    def _compute_urgency(self, element: Any) -> float:
        """How time-sensitive is this element?"""
        element_str = str(element).lower()
        
        urgency_score = 0.0
        for pattern in self.urgency_patterns:
            if re.search(pattern, element_str, re.IGNORECASE):
                urgency_score += 0.25
        
        return min(1.0, urgency_score)
    
    def _aggregate_scores(self, factor_scores: Dict[SalienceFactor, float]) -> float:
        """Weighted average of all factors"""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for factor, score in factor_scores.items():
            weight = self.base_weights.get(factor, 0.1)
            total_weight += weight
            weighted_sum += weight * score
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _apply_wednesday_bias(self, score: float, element: Any) -> float:
        """Modify salience based on Wednesday's personality"""
        element_str = str(element).lower()
        
        # Sarcasm bonus
        if any(word in element_str for word in ['sarcasm', 'irony', 'wit', 'joke']):
            score *= self.wednesday_bias['sarcasm_bonus']
        
        # Mystery interest
        if any(word in element_str for word in ['mystery', 'secret', 'puzzle', 'clue']):
            score *= self.wednesday_bias['mystery_interest']
        
        # Authority skepticism
        if any(word in element_str for word in ['principal', 'teacher', 'police', 'authority']):
            score *= self.wednesday_bias['authority_skepticism']
        
        return min(1.0, score)
    
    def _calculate_attention_duration(self, 
                                    overall_score: float, 
                                    factor_scores: Dict[SalienceFactor, float]) -> float:
        """How many seconds to maintain focus"""
        # Base duration: 0-10 seconds
        base_duration = overall_score * 10
        
        # Extend for high-threat elements
        if factor_scores.get(SalienceFactor.THREAT, 0) > 0.7:
            base_duration *= 3
        # Extend for self-reference
        elif factor_scores.get(SalienceFactor.SELF_REFERENCE, 0) > 0.8:
            base_duration *= 2
        # Extend for complex social situations
        elif factor_scores.get(SalienceFactor.SOCIAL, 0) > 0.8:
            base_duration *= 1.5
        
        # Cap at 2 minutes maximum
        return min(base_duration, 120)
    
    def _check_immediate_response(self, 
                                 factor_scores: Dict[SalienceFactor, float], 
                                 element: Any) -> bool:
        """Check if this requires immediate response"""
        # High threat always requires immediate attention
        if factor_scores.get(SalienceFactor.THREAT, 0) > 0.8:
            return True
        
        # High urgency
        if factor_scores.get(SalienceFactor.URGENCY, 0) > 0.8:
            return True
        
        # Direct address with urgency
        if (factor_scores.get(SalienceFactor.SELF_REFERENCE, 0) > 0.5 and
            factor_scores.get(SalienceFactor.URGENCY, 0) > 0.6):
            return True
        
        # Check for explicit immediate-response cues
        element_str = str(element).lower()
        if re.search(r'answer\s+now|tell\s+me|respond|what\s+do\s+you\s+think', element_str):
            return True
        
        return False
    
    def _update_context(self, new_saliences: List[SalienceScore], 
                       input_stream: Dict[str, Any]) -> None:
        """Update attention context with new salience results"""
        self.context['recent_saliences'] = new_saliences[:10]  # Keep last 10
        
        # Update recent inputs
        for element_id in input_stream:
            if element_id not in self.context['recent_inputs']:
                self.context['recent_inputs'].append(input_stream[element_id])
        # Keep only last 20 inputs
        self.context['recent_inputs'] = self.context['recent_inputs'][-20:]
        
        # Update attention history
        for score in new_saliences:
            if score.element_id not in self.context['attention_history']:
                self.context['attention_history'][score.element_id] = []
            self.context['attention_history'][score.element_id].append(score)
            # Keep only last 10 occurrences per element
            self.context['attention_history'][score.element_id] = \
                self.context['attention_history'][score.element_id][-10:]
    
    def get_current_focus(self) -> Optional[str]:
        """Return the ID of the currently focused element"""
        if self.context['recent_saliences']:
            top_score = self.context['recent_saliences'][0]
            # Check if still within attention duration
            elapsed = (datetime.now() - top_score.timestamp).total_seconds()
            if elapsed <= top_score.attention_duration:
                return top_score.element_id
        return None
    
    def get_attention_summary(self) -> Dict:
        """Summary of recent attention patterns"""
        return {
            'current_focus': self.get_current_focus(),
            'recent_elements': [
                {
                    'element_id': s.element_id,
                    'score': s.overall_score,
                    'timestamp': s.timestamp.isoformat()
                }
                for s in self.context['recent_saliences'][:5]
            ],
            'total_tracked_elements': len(self.context['attention_history'])
        }
    
    def reset_context(self) -> None:
        """Reset the context (useful for new sessions)"""
        self.context = {
            'current_focus': None,
            'recent_saliences': [],
            'attention_history': {},
            'recent_inputs': []
        }
        logger.info("SalienceDetector context reset")