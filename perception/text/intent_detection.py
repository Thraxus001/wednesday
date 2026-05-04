"""
Figures out what the user wants - the goal behind the words.
Wednesday is particularly good at detecting hidden motives and unspoken intentions.
"""
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class IntentCategory(Enum):
    """High-level categories of user intentions"""
    # Information seeking
    QUESTION = "question"
    CLARIFICATION = "clarification"
    FACT_CHECK = "fact_check"
    
    # Action requests
    COMMAND = "command"
    REQUEST = "request"
    SUGGESTION = "suggestion"
    
    # Social
    GREETING = "greeting"
    FAREWELL = "farewell"
    SMALL_TALK = "small_talk"
    COMPLIMENT = "compliment"
    INSULT = "insult"  # Wednesday can handle these
    SARCASM = "sarcasm"  # She appreciates good sarcasm
    
    # Emotional
    COMPLAINT = "complaint"
    PRAISE = "praise"
    APOLOGY = "apology"
    THANKS = "thanks"
    
    # Meta
    HELP = "help"
    FEEDBACK = "feedback"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    
    # Wednesday-specific
    CHALLENGE = "challenge"  # Testing her
    PHILOSOPHICAL = "philosophical"  # Deep questions
    OBSERVATION = "observation"  # User making an observation
    
    # Unknown
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"

@dataclass
class Intent:
    """Represents a detected user intent"""
    name: str
    category: IntentCategory
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Alternatives for ambiguous cases
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    
    # Context
    source_text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Wednesday's assessment
    urgency: float = 0.0  # 0-1 how urgent to respond
    complexity: float = 0.0  # 0-1 how complex to fulfill
    requires_action: bool = False
    hidden_motive: Optional[str] = None  # What user really wants
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'name': self.name,
            'category': self.category.value,
            'confidence': self.confidence,
            'parameters': self.parameters,
            'alternatives': self.alternatives[:3],  # Keep top 3
            'urgency': self.urgency,
            'complexity': self.complexity,
            'requires_action': self.requires_action,
            'hidden_motive': self.hidden_motive,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Intent':
        """Create Intent from dictionary"""
        return cls(
            name=data['name'],
            category=IntentCategory(data['category']),
            confidence=data['confidence'],
            parameters=data.get('parameters', {}),
            alternatives=data.get('alternatives', []),
            source_text=data.get('source_text', ''),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            urgency=data.get('urgency', 0.0),
            complexity=data.get('complexity', 0.0),
            requires_action=data.get('requires_action', False),
            hidden_motive=data.get('hidden_motive')
        )

class IntentDetector:
    """
    Figures out what the user wants - the goal behind the words.
    Wednesday doesn't just hear words; she discerns intent, especially hidden ones.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Intent classifiers (multiple strategies)
        self.ml_classifier = None  # For ML-based classification
        self._load_ml_classifier()
        
        # Rule-based intent templates
        self.intent_templates: Dict[str, List[Dict]] = self._load_intent_templates()
        
        # Intent hierarchy for disambiguation
        self.intent_hierarchy = self._build_intent_hierarchy()
        
        # Context tracking for disambiguation
        self.recent_intents: List[Intent] = []
        self.max_recent_intents = 10
        
        # Wednesday's special patterns
        self.wednesday_patterns = self._load_wednesday_patterns()
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'by_category': defaultdict(int),
            'avg_confidence': 0.0,
            'ambiguous_cases': 0
        }
        
        # Cache for repeated detections
        self.cache = {}
        self.cache_size = 100
        
        logger.info(f"IntentDetector initialized with {len(self.intent_templates)} intent templates")
    
    def detect_intent(self, 
                     parsed_text: Any, 
                     context: Optional[Dict] = None) -> Intent:
        """
        Combine linguistic cues with context to determine user intent.
        
        Args:
            parsed_text: Output from TextParser
            context: Current conversation context (from working memory)
            
        Returns:
            Intent object with best guess and alternatives
        """
        # Handle empty input
        if not parsed_text or not hasattr(parsed_text, 'raw_text') or not parsed_text.raw_text:
            return self._create_unknown_intent("")
        
        # Check cache for identical input (with same context signature)
        cache_key = self._get_cache_key(parsed_text, context)
        if cache_key in self.cache:
            logger.debug(f"Cache hit for intent: {cache_key[:20]}...")
            return self.cache[cache_key]
        
        # Extract features from parsed text
        features = self._extract_features(parsed_text)
        
        # Get candidates from multiple strategies
        candidates = []
        
        # 1. Rule-based matching
        rule_candidates = self._match_templates(features, parsed_text)
        candidates.extend(rule_candidates)
        
        # 2. ML classifier (if available)
        if self.ml_classifier:
            ml_candidates = self._classify_ml(features)
            candidates.extend(ml_candidates)
        
        # 3. Context-based prediction
        if context:
            context_candidates = self._predict_from_context(context, features)
            candidates.extend(context_candidates)
        
        # 4. Wednesday special patterns
        wednesday_candidates = self._check_wednesday_patterns(parsed_text, features)
        candidates.extend(wednesday_candidates)
        
        # Merge and rank candidates
        ranked = self._rank_candidates(candidates, context, features)
        
        if not ranked:
            # No intent detected
            intent = self._create_unknown_intent(parsed_text.raw_text)
        else:
            # Get top intent and alternatives
            top_intent = ranked[0]
            alternatives = [(i.name, i.confidence) for i in ranked[1:4]]
            
            # Enhance with Wednesday's analysis
            top_intent.alternatives = alternatives
            top_intent = self._apply_wednesday_analysis(top_intent, parsed_text, context)
            
            intent = top_intent
        
        # Update stats
        self._update_stats(intent)
        
        # Store in recent intents
        self.recent_intents.append(intent)
        if len(self.recent_intents) > self.max_recent_intents:
            self.recent_intents.pop(0)
        
        # Cache the result
        self.cache[cache_key] = intent
        if len(self.cache) > self.cache_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        # Log ambiguous cases
        if len(intent.alternatives) > 1 and intent.alternatives[0][1] - intent.alternatives[1][1] < 0.2:
            self.detection_stats['ambiguous_cases'] += 1
            logger.debug(f"Ambiguous intent: {intent.name} vs {intent.alternatives[1][0]}")
        
        return intent
    
    def register_custom_intent(self, name: str, patterns: List[Dict], 
                              category: IntentCategory = IntentCategory.UNKNOWN):
        """
        Register a new custom intent with patterns.
        
        Args:
            name: Intent name
            patterns: List of pattern dicts with 'pattern' (regex) and 'extract' (param groups)
            category: Intent category
        """
        if name not in self.intent_templates:
            self.intent_templates[name] = []
        
        for pattern in patterns:
            pattern['category'] = category
            pattern['custom'] = True
            self.intent_templates[name].append(pattern)
        
        logger.info(f"Registered custom intent: {name} with {len(patterns)} patterns")
    
    def get_ambiguous_options(self, parsed_text: Any) -> List[Tuple[str, float]]:
        """
        When multiple intents are possible, return ranked alternatives.
        """
        if not parsed_text:
            return []
        
        features = self._extract_features(parsed_text)
        candidates = self._match_templates(features, parsed_text)
        
        if self.ml_classifier:
            candidates.extend(self._classify_ml(features))
        
        ranked = self._rank_candidates(candidates, None, features)
        
        return [(i.name, i.confidence) for i in ranked[:5]]
    
    def _load_ml_classifier(self):
        """Load ML-based intent classifier"""
        # Placeholder - would load a trained model
        # Options: transformers, sklearn classifier, etc.
        try:
            # Example: load from path in config
            model_path = self.config.get('intent_model_path')
            if model_path:
                # In a real implementation, you'd load your model here
                # self.ml_classifier = load_model(model_path)
                logger.info(f"ML classifier would be loaded from {model_path}")
            else:
                logger.info("No ML classifier configured, using rule-based only")
        except Exception as e:
            logger.error(f"Failed to load ML classifier: {e}")
    
    def _load_intent_templates(self) -> Dict[str, List[Dict]]:
        """Load rule-based intent patterns"""
        templates = {
            # Questions
            "ask_definition": [
                {
                    'pattern': r'what (is|are) (\w+)',
                    'category': IntentCategory.QUESTION,
                    'extract': {'topic': 2}
                },
                {
                    'pattern': r'(who|what) is (\w+)',
                    'category': IntentCategory.QUESTION,
                    'extract': {'topic': 2}
                },
                {
                    'pattern': r'define (\w+)',
                    'category': IntentCategory.QUESTION,
                    'extract': {'topic': 1}
                }
            ],
            "ask_opinion": [
                {
                    'pattern': r'what do you think (about|of) (\w+)',
                    'category': IntentCategory.QUESTION,
                    'extract': {'topic': 2}
                },
                {
                    'pattern': r'how do you feel about (\w+)',
                    'category': IntentCategory.QUESTION,
                    'extract': {'topic': 1}
                },
                {
                    'pattern': r'what\'?s your opinion on (\w+)',
                    'category': IntentCategory.QUESTION,
                    'extract': {'topic': 1}
                }
            ],
            "ask_help": [
                {
                    'pattern': r'(can|could) you help me (with|to)? (\w+)',
                    'category': IntentCategory.HELP,
                    'extract': {'action': 3}
                },
                {
                    'pattern': r'how (do|can) I (\w+)',
                    'category': IntentCategory.HELP,
                    'extract': {'goal': 2}
                },
                {
                    'pattern': r'help (me with)? (\w+)',
                    'category': IntentCategory.HELP,
                    'extract': {'action': 2}
                }
            ],
            
            # Commands
            "command_do": [
                {
                    'pattern': r'(please )?(\w+) (the|that)',
                    'category': IntentCategory.COMMAND,
                    'extract': {'action': 2}
                },
                {
                    'pattern': r'(tell|show) me (\w+)',
                    'category': IntentCategory.COMMAND,
                    'extract': {'action': 2}
                }
            ],
            "command_stop": [
                {
                    'pattern': r'stop|halt|cease|quit',
                    'category': IntentCategory.COMMAND,
                    'extract': {}
                }
            ],
            
            # Social
            "greeting": [
                {
                    'pattern': r'^(hello|hi|hey|greetings)([,.!\s]|$)',
                    'category': IntentCategory.GREETING,
                    'extract': {}
                },
                {
                    'pattern': r'good (morning|afternoon|evening)',
                    'category': IntentCategory.GREETING,
                    'extract': {'time': 1}
                },
                {
                    'pattern': r'what\'?s up',
                    'category': IntentCategory.GREETING,
                    'extract': {}
                }
            ],
            "farewell": [
                {
                    'pattern': r'(goodbye|bye|see you|farewell|take care)',
                    'category': IntentCategory.FAREWELL,
                    'extract': {}
                }
            ],
            "insult": [
                {
                    'pattern': r'you (are|sound|seem) (annoying|boring|stupid|insufferable)',
                    'category': IntentCategory.INSULT,
                    'extract': {'insult': 2}
                },
                {
                    'pattern': r'(shut up|be quiet|go away)',
                    'category': IntentCategory.INSULT,
                    'extract': {}
                }
            ],
            "sarcasm": [
                {
                    'pattern': r'oh (great|wonderful|fantastic|brilliant)',
                    'category': IntentCategory.SARCASM,
                    'extract': {}
                },
                {
                    'pattern': r'sure (you do|whatever|you are)',
                    'category': IntentCategory.SARCASM,
                    'extract': {}
                },
                {
                    'pattern': r'as if',
                    'category': IntentCategory.SARCASM,
                    'extract': {}
                }
            ],
            "compliment": [
                {
                    'pattern': r'you (are|sound) (smart|clever|interesting)',
                    'category': IntentCategory.COMPLIMENT,
                    'extract': {'compliment': 2}
                },
                {
                    'pattern': r'(nice|good) (point|answer|response)',
                    'category': IntentCategory.COMPLIMENT,
                    'extract': {}
                }
            ],
            
            # Emotional
            "complaint": [
                {
                    'pattern': r'(i hate|i dislike|i don\'?t like|this is terrible)',
                    'category': IntentCategory.COMPLAINT,
                    'extract': {}
                }
            ],
            "thanks": [
                {
                    'pattern': r'(thank you|thanks|appreciate it|much obliged)',
                    'category': IntentCategory.THANKS,
                    'extract': {}
                }
            ],
            "apology": [
                {
                    'pattern': r'(i\'?m sorry|my apologies|forgive me)',
                    'category': IntentCategory.APOLOGY,
                    'extract': {}
                }
            ],
            
            # Wednesday-specific
            "challenge": [
                {
                    'pattern': r'(prove it|convince me|you think you are so smart|test you)',
                    'category': IntentCategory.CHALLENGE,
                    'extract': {}
                }
            ],
            "philosophical": [
                {
                    'pattern': r'(meaning of life|why are we here|what is the point|purpose of existence)',
                    'category': IntentCategory.PHILOSOPHICAL,
                    'extract': {}
                }
            ],
            "observation": [
                {
                    'pattern': r'(i notice|it seems|it appears|observed that)',
                    'category': IntentCategory.OBSERVATION,
                    'extract': {}
                }
            ]
        }
        
        # Load custom templates from config if provided
        custom_templates = self.config.get('custom_intent_templates', {})
        templates.update(custom_templates)
        
        return templates
    
    def _build_intent_hierarchy(self) -> Dict:
        """Build hierarchy for intent disambiguation"""
        return {
            IntentCategory.COMMAND: ['request', 'suggestion'],
            IntentCategory.QUESTION: ['clarification', 'fact_check'],
            IntentCategory.SOCIAL: ['greeting', 'farewell', 'small_talk'],
            IntentCategory.CHALLENGE: ['question', 'philosophical']
        }
    
    def _load_wednesday_patterns(self) -> Dict:
        """Load patterns for Wednesday-specific intent detection"""
        return {
            'hidden_motive_indicators': [
                r'really mean',
                r'actually want',
                r'truth is',
                r'to be honest',
                r'between us',
                r'confidentially'
            ],
            'test_indicators': [
                r'test',
                r'see if you',
                r'prove',
                r'demonstrate',
                r'show me what you'
            ],
            'sarcasm_indicators': [
                r'sure,? sure',
                r'obviously',
                r'clearly',
                r'naturally',
                r'oh really'
            ]
        }
    
    def _get_cache_key(self, parsed_text: Any, context: Optional[Dict]) -> str:
        """Generate cache key for intent detection"""
        text = parsed_text.raw_text if hasattr(parsed_text, 'raw_text') else ''
        # Include context signature if available
        context_sig = ''
        if context:
            context_sig = str(context.get('conversation_id', ''))[:8]
        return f"{text}_{context_sig}"
    
    def _extract_features(self, parsed_text: Any) -> Dict:
        """Extract features from parsed text for intent classification"""
        if not parsed_text:
            return {}
        
        # Safely access attributes with defaults
        raw_text = getattr(parsed_text, 'raw_text', '')
        tokens = getattr(parsed_text, 'tokens', [])
        pos_tags = getattr(parsed_text, 'pos_tags', [])
        entities = getattr(parsed_text, 'entities', [])
        sentences = getattr(parsed_text, 'sentences', [])
        
        features = {
            'text': raw_text,
            'text_lower': raw_text.lower(),
            'tokens': tokens,
            'pos_tags': pos_tags,
            'entities': entities,
            'sentence_count': len(sentences) if sentences else 1,
            'first_word': tokens[0].lower() if tokens else '',
            'last_word': tokens[-1].lower() if tokens else '',
            'has_question_mark': '?' in raw_text,
            'has_exclamation': '!' in raw_text,
            'word_count': len(tokens),
            'starts_with_verb': any(tag == 'VB' for _, tag in pos_tags[:1]) if pos_tags else False,
            'contains_pronoun': any(tag == 'PRON' for _, tag in pos_tags)
        }
        
        return features
    
    def _match_templates(self, features: Dict, parsed_text: Any) -> List[Intent]:
        """Match input against rule-based templates"""
        candidates = []
        text = features.get('text_lower', '')
        
        if not text:
            return candidates
        
        for intent_name, patterns in self.intent_templates.items():
            for pattern_dict in patterns:
                pattern = pattern_dict['pattern']
                try:
                    match = re.search(pattern, text, re.IGNORECASE)
                    
                    if match:
                        # Extract parameters from regex groups
                        params = {}
                        if 'extract' in pattern_dict:
                            for param_name, group_idx in pattern_dict['extract'].items():
                                try:
                                    if isinstance(group_idx, int):
                                        if group_idx <= len(match.groups()):
                                            params[param_name] = match.group(group_idx)
                                    elif isinstance(group_idx, str):
                                        # Handle named groups
                                        try:
                                            params[param_name] = match.group(group_idx)
                                        except IndexError:
                                            pass
                                except (IndexError, AttributeError):
                                    pass
                        
                        # Calculate confidence based on match quality
                        confidence = self._calculate_match_confidence(
                            match, text, pattern_dict, features
                        )
                        
                        intent = Intent(
                            name=intent_name,
                            category=pattern_dict.get('category', IntentCategory.UNKNOWN),
                            confidence=confidence,
                            parameters=params,
                            source_text=features.get('text', '')
                        )
                        
                        candidates.append(intent)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        
        return candidates
    
    def _classify_ml(self, features: Dict) -> List[Intent]:
        """Use ML classifier for intent detection"""
        if not self.ml_classifier:
            return []
        
        # Placeholder - would call actual model
        # predictions = self.ml_classifier.predict(features)
        # return [Intent(name=p[0], category=IntentCategory.UNKNOWN, confidence=p[1]) for p in predictions]
        return []
    
    def _predict_from_context(self, context: Dict, features: Dict) -> List[Intent]:
        """Predict intent based on conversation context"""
        candidates = []
        
        # Check if this is likely a follow-up to previous intent
        if not self.recent_intents:
            return candidates
        
        last_intent = self.recent_intents[-1]
        text = features.get('text_lower', '')
        
        # If last intent was a question, user might be answering
        if last_intent.category == IntentCategory.QUESTION:
            # Look for answer indicators
            answer_indicators = ['yes', 'no', 'maybe', 'perhaps', 'i think', 'actually']
            if any(indicator in text for indicator in answer_indicators):
                intent = Intent(
                    name="answer",
                    category=IntentCategory.CLARIFICATION,
                    confidence=0.6,
                    parameters={'responding_to': last_intent.name},
                    source_text=features.get('text', '')
                )
                candidates.append(intent)
        
        # If last intent required action, user might be commenting on it
        if last_intent.requires_action:
            feedback_indicators = ['good', 'bad', 'wrong', 'correct', 'thanks']
            if any(indicator in text for indicator in feedback_indicators):
                intent = Intent(
                    name="feedback",
                    category=IntentCategory.FEEDBACK,
                    confidence=0.5,
                    parameters={'about': last_intent.name},
                    source_text=features.get('text', '')
                )
                candidates.append(intent)
        
        return candidates
    
    def _check_wednesday_patterns(self, parsed_text: Any, features: Dict) -> List[Intent]:
        """Check for Wednesday-specific patterns"""
        candidates = []
        text = features.get('text_lower', '')
        
        if not text:
            return candidates
        
        # Check for hidden motives
        hidden_motive_detected = False
        for pattern in self.wednesday_patterns['hidden_motive_indicators']:
            if re.search(pattern, text):
                hidden_motive_detected = True
                break
        
        # Check for testing behavior
        for pattern in self.wednesday_patterns['test_indicators']:
            if re.search(pattern, text):
                intent = Intent(
                    name="user_testing",
                    category=IntentCategory.CHALLENGE,
                    confidence=0.7,
                    parameters={'test_type': 'capability'},
                    source_text=features.get('text', '')
                )
                if hidden_motive_detected:
                    intent.hidden_motive = "testing_wednesday"
                candidates.append(intent)
                break
        
        # Check for deep philosophical questions
        philosophical_indicators = ['why', 'purpose', 'meaning', 'existence', 'reality']
        if all(indicator in text for indicator in ['what', 'meaning']):
            intent = Intent(
                name="philosophical_inquiry",
                category=IntentCategory.PHILOSOPHICAL,
                confidence=0.8,
                parameters={},
                source_text=features.get('text', '')
            )
            candidates.append(intent)
        
        return candidates
    
    def _rank_candidates(self, candidates: List[Intent], 
                        context: Optional[Dict],
                        features: Dict) -> List[Intent]:
        """Rank intent candidates by confidence and context"""
        if not candidates:
            return []
        
        # Group by intent name
        intent_groups = defaultdict(list)
        for intent in candidates:
            intent_groups[intent.name].append(intent)
        
        # Merge duplicates by averaging confidence
        merged = []
        for name, intents in intent_groups.items():
            avg_confidence = sum(i.confidence for i in intents) / len(intents)
            best_intent = max(intents, key=lambda x: x.confidence)
            best_intent.confidence = avg_confidence
            
            # Merge parameters from all matches
            merged_params = {}
            for intent in intents:
                merged_params.update(intent.parameters)
            best_intent.parameters = merged_params
            
            merged.append(best_intent)
        
        # Apply context boost
        if context:
            expected_intents = context.get('expected_intents', [])
            for intent in merged:
                if intent.name in expected_intents:
                    intent.confidence = min(1.0, intent.confidence * 1.15)
        
        # Boost based on feature matches
        for intent in merged:
            # Boost if intent category matches sentence type
            if intent.category == IntentCategory.QUESTION and features.get('has_question_mark'):
                intent.confidence = min(1.0, intent.confidence * 1.1)
            if intent.category in [IntentCategory.COMMAND, IntentCategory.REQUEST] and features.get('starts_with_verb'):
                intent.confidence = min(1.0, intent.confidence * 1.05)
        
        # Sort by confidence
        merged.sort(key=lambda x: x.confidence, reverse=True)
        
        return merged
    
    def _calculate_match_confidence(self, match: re.Match, text: str, 
                                   pattern_dict: Dict, features: Dict) -> float:
        """Calculate confidence for a regex match"""
        base_confidence = 0.7  # Base confidence for any match
        
        # Boost for exact matches (no extra text)
        if match.group() == text.strip():
            base_confidence += 0.2
        
        # Boost for patterns with more specific extractors
        if 'extract' in pattern_dict and pattern_dict['extract']:
            base_confidence += 0.1
        
        # Adjust based on pattern length (longer patterns more reliable)
        pattern_length = len(pattern_dict['pattern'])
        if pattern_length > 25:
            base_confidence += 0.1
        elif pattern_length < 5:
            base_confidence -= 0.1
        
        # Boost if match is at start of text (more likely to be primary intent)
        if match.start() == 0:
            base_confidence += 0.1
        
        # Adjust based on word count (very short texts are ambiguous)
        word_count = features.get('word_count', 0)
        if word_count < 3:
            base_confidence *= 0.8
        
        return min(1.0, base_confidence)
    
    def _apply_wednesday_analysis(self, intent: Intent, parsed_text: Any, 
                                  context: Optional[Dict]) -> Intent:
        """
        Apply Wednesday's analytical lens to intent detection.
        She notices things others miss.
        """
        text = parsed_text.raw_text.lower() if hasattr(parsed_text, 'raw_text') else ''
        
        # Detect hidden motives
        for pattern in self.wednesday_patterns['hidden_motive_indicators']:
            if re.search(pattern, text):
                intent.hidden_motive = "user_may_be_evasive"
                # Reduce confidence slightly for evasive speech
                intent.confidence *= 0.95
                break
        
        # Detect if this is a test
        for pattern in self.wednesday_patterns['test_indicators']:
            if re.search(pattern, text):
                intent.hidden_motive = "testing_wednesday"
                break
        
        # Assess urgency
        urgency_indicators = {
            'high': ['urgent', 'immediately', 'asap', 'right now', 'emergency'],
            'medium': ['soon', 'quickly', 'please'],
            'low': ['later', 'whenever', 'sometime']
        }
        
        intent.urgency = 0.2  # Default low urgency
        for level, indicators in urgency_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    if level == 'high':
                        intent.urgency = 0.9
                    elif level == 'medium':
                        intent.urgency = max(intent.urgency, 0.5)
                    elif level == 'low':
                        intent.urgency = max(intent.urgency, 0.3)
        
        # Intent-specific urgency adjustments
        if intent.category == IntentCategory.COMMAND:
            intent.urgency = max(intent.urgency, 0.6)
        elif intent.category == IntentCategory.HELP:
            intent.urgency = max(intent.urgency, 0.7)
        elif intent.category == IntentCategory.CHALLENGE:
            intent.urgency = 0.5  # Challenges deserve prompt response
        
        # Assess complexity
        if hasattr(parsed_text, 'complexity_score'):
            intent.complexity = parsed_text.complexity_score
        else:
            # Estimate complexity from text length
            word_count = len(text.split())
            intent.complexity = min(1.0, word_count / 50)
        
        # Determine if action required
        action_categories = [
            IntentCategory.COMMAND,
            IntentCategory.REQUEST,
            IntentCategory.HELP,
            IntentCategory.CHALLENGE
        ]
        intent.requires_action = intent.category in action_categories
        
        # Special case: sarcasm doesn't require action but does require clever response
        if intent.category == IntentCategory.SARCASM:
            intent.requires_action = True  # Requires witty response
            intent.urgency = max(intent.urgency, 0.4)  # Sarcasm needs timely response
        
        # Special case: insults might require defensive response
        if intent.category == IntentCategory.INSULT:
            intent.requires_action = True
            intent.urgency = max(intent.urgency, 0.5)
        
        return intent
    
    def _create_unknown_intent(self, text: str) -> Intent:
        """Create an unknown intent when detection fails"""
        return Intent(
            name="unknown",
            category=IntentCategory.UNKNOWN,
            confidence=0.1,
            parameters={'text': text},
            source_text=text,
            urgency=0.1,
            complexity=0.3,
            requires_action=False
        )
    
    def _update_stats(self, intent: Intent):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += 1
        self.detection_stats['by_category'][intent.category.value] += 1
        
        total = self.detection_stats['total_detections']
        old_avg = self.detection_stats['avg_confidence']
        self.detection_stats['avg_confidence'] = old_avg + (intent.confidence - old_avg) / total
    
    def get_stats(self) -> Dict:
        """Return detection statistics"""
        stats = dict(self.detection_stats)
        stats['by_category'] = dict(stats['by_category'])
        return stats
    
    def reset_stats(self) -> None:
        """Reset detection statistics"""
        self.detection_stats = {
            'total_detections': 0,
            'by_category': defaultdict(int),
            'avg_confidence': 0.0,
            'ambiguous_cases': 0
        }

# Connects to: parser.py (gets structured input for analysis)
# Connects to: memory/working/context_buffer.py (uses conversation context for disambiguation)
# Connects to: cognition/goal_manager.py (feeds identified goals for action planning)
# Connects to: language/understanding/discourse.py (helps with conversation flow)
# Connects to: executive/priorities.py (urgency assessment influences scheduling)
# Connects to: emotion/appraisal.py (intent category influences emotional response)