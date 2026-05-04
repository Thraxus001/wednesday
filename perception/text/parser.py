"""
Breaks down text input into structured form.
Tokenization, POS tagging, dependency parsing - the grammar of Wednesday's world.
She appreciates precise language, so this needs to be accurate.
"""
import spacy
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
from collections import defaultdict
import logging
import time

# Configure logging
logger = logging.getLogger(__name__)

class LinguisticFeature(Enum):
    """Features we extract during parsing"""
    TOKENS = "tokens"
    POS_TAGS = "pos_tags"
    DEPENDENCIES = "dependencies"
    ENTITIES = "entities"
    NOUN_CHUNKS = "noun_chunks"
    SENTIMENT_TOKENS = "sentiment_tokens"  # Tokens with emotional weight
    QUOTES = "quotes"  # Direct speech
    PARENTHETICALS = "parentheticals"  # Aside comments
    CONTRADICTIONS = "contradictions"  # Words indicating contradiction (but, however)

@dataclass
class ParsedText:
    """Complete linguistic analysis of input text"""
    raw_text: str
    tokens: List[str]
    pos_tags: List[Tuple[str, str]]  # (token, tag)
    dependencies: List[Tuple[str, str, str]]  # (token, dep_type, head)
    entities: List[Tuple[str, str, int, int]]  # (text, label, start, end)
    noun_chunks: List[str]
    sentences: List[str]
    language: str = "en"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Wednesday-specific annotations
    sarcasm_indicators: List[str] = field(default_factory=list)  # Quotes, exaggeration markers
    formality_level: float = 0.5  # 0 (casual) to 1 (formal)
    complexity_score: float = 0.5  # Linguistic complexity
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage in memory"""
        # Strip positions from entities for storage
        simplified_entities = [(e[0], e[1]) for e in self.entities]
        
        return {
            'raw': self.raw_text,
            'tokens': self.tokens,
            'pos_tags': self.pos_tags[:20],  # Limit for storage
            'entities': simplified_entities[:10],  # Limit for storage
            'noun_chunks': self.noun_chunks[:10],
            'sentences': self.sentences,
            'language': self.language,
            'formality': self.formality_level,
            'complexity': self.complexity_score,
            'sarcasm_indicators': self.sarcasm_indicators,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ParsedText':
        """Create ParsedText from dictionary"""
        return cls(
            raw_text=data.get('raw', ''),
            tokens=data.get('tokens', []),
            pos_tags=data.get('pos_tags', []),
            dependencies=[],  # Can't reconstruct from simplified dict
            entities=[(e[0], e[1], 0, 0) for e in data.get('entities', [])],
            noun_chunks=data.get('noun_chunks', []),
            sentences=data.get('sentences', []),
            language=data.get('language', 'en'),
            confidence=0.8,  # Reduced confidence for reconstructed parse
            metadata={},
            sarcasm_indicators=data.get('sarcasm_indicators', []),
            formality_level=data.get('formality', 0.5),
            complexity_score=data.get('complexity', 0.5)
        )

class TextParser:
    """
    Breaks down text input into structured form.
    Wednesday pays attention to linguistic details - they reveal more than words alone.
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        self.config = config or {}
        self.model_path = model_path
        
        # Load NLP model (spaCy by default)
        self.nlp = self._load_model(model_path)
        
        # Custom patterns for entity extraction
        self.custom_entity_patterns = self._load_custom_patterns()
        
        # Wednesday's linguistic preferences
        self.wednesday_preferences = {
            'formal_bias': 0.3,  # She appreciates formality but doesn't require it
            'sarcasm_sensitivity': 0.8,  # High sensitivity to sarcasm cues
            'complexity_preference': 0.6,  # Prefers moderately complex language
        }
        
        # Performance tracking
        self.parsing_stats = {
            'total_parsed': 0,
            'total_errors': 0,
            'avg_parse_time': 0.0,
            'errors': []
        }
        
        logger.info(f"TextParser initialized with model: {model_path or 'default'}")
    
    def parse(self, text: Union[str, Any], context: Optional[Dict] = None) -> ParsedText:
        """
        Convert raw text to complete linguistic structure.
        
        Args:
            text: Raw input text
            context: Optional context from previous conversation
            
        Returns:
            ParsedText object with full linguistic analysis
        """
        start_time = time.time()
        
        # Handle non-string input
        if not isinstance(text, str):
            logger.warning(f"Invalid input to parser: {type(text)}. Converting to string.")
            text = str(text) if text is not None else ""
        
        if not text.strip():
            logger.debug("Empty text input")
            return self._create_empty_parse(text)
        
        try:
            # Basic cleaning
            text = self._preprocess_text(text)
            
            # Run spaCy pipeline
            doc = self.nlp(text)
            
            # Extract core linguistic features
            tokens = [token.text for token in doc]
            pos_tags = [(token.text, token.pos_) for token in doc]
            dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
            
            # Extract entities
            entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) 
                       for ent in doc.ents]
            
            # Add custom entities
            custom_ents = self._extract_custom_entities(text)
            entities.extend(custom_ents)
            
            # Extract noun chunks
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            
            # Split into sentences
            sentences = [sent.text for sent in doc.sents]
            
            # Wednesday-specific analysis
            sarcasm_indicators = self._detect_sarcasm_indicators(doc, text)
            formality = self._calculate_formality(doc)
            complexity = self._calculate_complexity(doc)
            
            # Create parsed structure
            parsed = ParsedText(
                raw_text=text,
                tokens=tokens,
                pos_tags=pos_tags,
                dependencies=dependencies,
                entities=entities,
                noun_chunks=noun_chunks,
                sentences=sentences,
                language=doc.lang_,
                confidence=self._calculate_confidence(doc),
                metadata={
                    'parse_time': time.time() - start_time,
                    'token_count': len(tokens),
                    'sentence_count': len(sentences),
                    'has_context': context is not None
                },
                sarcasm_indicators=sarcasm_indicators,
                formality_level=formality,
                complexity_score=complexity
            )
            
            # Update stats
            self._update_stats(time.time() - start_time, success=True)
            
            # Log for debugging
            logger.debug(f"Parsed {len(tokens)} tokens in {parsed.metadata['parse_time']:.3f}s")
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing text: {e}", exc_info=True)
            self._update_stats(time.time() - start_time, success=False, error=str(e))
            return self._create_empty_parse(text)
    
    def extract_entities(self, parsed: ParsedText) -> Dict[str, List[str]]:
        """
        Extract and categorize named entities.
        
        Returns:
            Dictionary of entity types to lists of entities
        """
        entities_by_type = defaultdict(list)
        
        for entity_text, entity_type, _, _ in parsed.entities:
            if entity_text not in entities_by_type[entity_type]:
                entities_by_type[entity_type].append(entity_text)
        
        return dict(entities_by_type)
    
    def extract_key_phrases(self, parsed: ParsedText, top_n: int = 5) -> List[str]:
        """Extract the most significant phrases from the text"""
        if not parsed.noun_chunks and not parsed.entities:
            return []
        
        # Use noun chunks as base
        phrases = list(parsed.noun_chunks)
        
        # Add named entities
        entity_texts = [e[0] for e in parsed.entities]
        for entity in entity_texts:
            if entity not in phrases:
                phrases.append(entity)
        
        # Score and rank phrases
        scored_phrases = []
        for phrase in phrases:
            # Base score on length (prefer longer phrases)
            score = len(phrase.split()) * 0.5
            
            # Boost named entities
            if any(phrase == ent[0] for ent in parsed.entities):
                ent_type = next(ent[1] for ent in parsed.entities if ent[0] == phrase)
                # Different weights for different entity types
                type_weights = {
                    'PERSON': 2.5,
                    'ORG': 2.0,
                    'GPE': 2.0,
                    'DATE': 1.5,
                    'TIME': 1.5,
                    'MONEY': 1.8,
                    'PERCENT': 1.8
                }
                score += type_weights.get(ent_type, 2.0)
            
            # Boost phrases that appear in multiple sentences
            occurrence_count = sum(1 for sent in parsed.sentences if phrase.lower() in sent.lower())
            if occurrence_count > 1:
                score += occurrence_count * 0.3
            
            scored_phrases.append((phrase, score))
        
        # Sort by score and return top N
        scored_phrases.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in scored_phrases[:top_n]]
    
    def get_sentiment_tokens(self, parsed: ParsedText) -> List[str]:
        """Extract tokens that might carry emotional weight"""
        sentiment_weights = {
            'JJ': 0.7,  # Adjectives
            'JJR': 0.7,  # Comparative adjectives
            'JJS': 0.7,  # Superlative adjectives
            'RB': 0.5,   # Adverbs
            'RBR': 0.5,  # Comparative adverbs
            'RBS': 0.5,  # Superlative adverbs
            'VB': 0.3,   # Verbs
            'VBD': 0.3,  # Verbs (past tense)
            'VBG': 0.3,  # Verbs (gerund)
            'VBN': 0.3,  # Verbs (past participle)
            'VBP': 0.3,  # Verbs (non-3rd person)
            'VBZ': 0.3,  # Verbs (3rd person)
        }
        
        sentiment_tokens = []
        for token, pos in parsed.pos_tags:
            if pos in sentiment_weights:
                sentiment_tokens.append(token)
        
        return sentiment_tokens
    
    def _load_model(self, model_path: Optional[str]) -> Any:
        """Load the NLP model (spaCy)"""
        try:
            if model_path:
                return spacy.load(model_path)
            else:
                # Try to load default model, download if not available
                try:
                    return spacy.load("en_core_web_sm")
                except OSError:
                    logger.info("Downloading spaCy model...")
                    import subprocess
                    import sys
                    
                    subprocess.check_call([
                        sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                    ])
                    return spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to load NLP model: {e}")
            logger.warning("Using minimal tokenizer fallback")
            return self._create_minimal_tokenizer()
    
    def _load_custom_patterns(self) -> List[Dict]:
        """Load custom entity extraction patterns"""
        # Patterns for things spaCy might miss
        return [
            {'pattern': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 'label': 'PERSON', 'weight': 1.0},  # Full names
            {'pattern': r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b', 'label': 'PERSON', 'weight': 1.0},  # Names with initial
            {'pattern': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'label': 'IP_ADDRESS', 'weight': 0.9},
            {'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'label': 'EMAIL', 'weight': 0.9},
            {'pattern': r'\b\d{3}-\d{2}-\d{4}\b', 'label': 'SSN', 'weight': 1.0},  # US SSN
            {'pattern': r'\b\d{5}(?:-\d{4})?\b', 'label': 'ZIP_CODE', 'weight': 0.8},  # ZIP code
        ]
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text before parsing"""
        if not text:
            return text
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize quotes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r'['']', "'", text)
        
        # Normalize dashes
        text = re.sub(r'[–—]', '-', text)
        
        return text
    
    def _extract_custom_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Extract entities using custom patterns"""
        entities = []
        for pattern in self.custom_entity_patterns:
            try:
                for match in re.finditer(pattern['pattern'], text):
                    # Avoid duplicates
                    if not any(e[0] == match.group() and e[1] == pattern['label'] for e in entities):
                        entities.append((
                            match.group(),
                            pattern['label'],
                            match.start(),
                            match.end()
                        ))
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {pattern['pattern']} - {e}")
        
        return entities
    
    def _detect_sarcasm_indicators(self, doc: Any, text: str) -> List[str]:
        """
        Detect linguistic cues that might indicate sarcasm.
        Wednesday appreciates good sarcasm, so this is important.
        """
        indicators = []
        
        # Check for quotation marks around phrases
        quoted = re.findall(r'"([^"]*)"', text)
        if quoted:
            indicators.extend([f"quoted:{q[:50]}" for q in quoted])  # Limit length
        
        # Check for exaggeration markers
        exaggeration_patterns = [
            (r'\b(absolutely|completely|totally|literally)\s+(un)?[A-Za-z]+\b', 'absolute'),
            (r'\b(the most|the best|the worst|the greatest|the worst)\b', 'superlative'),
            (r'!{2,}', 'multiple_exclamation'),
            (r'\?{2,}', 'multiple_question'),
            (r'!+\?+|\?+!+', 'interrobang'),
            (r'\b(oh really|sure|right|of course)\b', 'dismissive'),
            (r'\b(as if|whatever|yeah right)\b', 'skeptical')
        ]
        
        for pattern, indicator_type in exaggeration_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                indicators.append(f"exaggeration:{indicator_type}")
        
        # Check for contrastive discourse markers
        contrast_markers = ['but', 'however', 'yet', 'although', 'though']
        for token in doc:
            if token.text.lower() in contrast_markers and token.dep_ in ['cc', 'advmod']:
                indicators.append(f"contrast:{token.text}")
        
        # Check for hyperbolic adjectives
        hyperbolic_adjectives = ['amazing', 'incredible', 'unbelievable', 'ridiculous', 
                                'hilarious', 'perfect', 'flawless', 'genius']
        for token in doc:
            if token.text.lower() in hyperbolic_adjectives and token.pos_ == 'ADJ':
                indicators.append(f"hyperbole:{token.text}")
        
        return indicators
    
    def _calculate_formality(self, doc: Any) -> float:
        """
        Calculate text formality on a scale from 0 (casual) to 1 (formal).
        Based on linguistic features: contractions, pronouns, passive voice, etc.
        """
        total_tokens = len(doc)
        if total_tokens == 0:
            return 0.5
        
        formality_score = 0.5  # Start at neutral
        
        # Penalize contractions (informal)
        contractions = 0
        contraction_list = [
            "i'm", "you're", "he's", "she's", "it's", "we're", "they're",
            "i've", "you've", "we've", "they've",
            "i'll", "you'll", "he'll", "she'll", "we'll", "they'll",
            "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
            "won't", "wouldn't", "couldn't", "shouldn't", "can't"
        ]
        
        for token in doc:
            if token.text.lower() in contraction_list:
                contractions += 1
        
        contraction_penalty = (contractions / total_tokens) * 0.4  # Increased penalty
        formality_score -= contraction_penalty
        
        # Boost for passive voice (more formal)
        passive_count = 0
        for token in doc:
            if token.dep_ in ['nsubjpass', 'csubjpass', 'auxpass']:
                passive_count += 1
        
        passive_boost = (passive_count / total_tokens) * 0.3
        formality_score += passive_boost
        
        # Penalize personal pronouns (less formal)
        personal_pronouns = 0
        personal_pronoun_list = ['i', 'you', 'we', 'they', 'me', 'us', 'them']
        for token in doc:
            if token.text.lower() in personal_pronoun_list and token.pos_ == 'PRON':
                personal_pronouns += 1
        
        pronoun_penalty = (personal_pronouns / total_tokens) * 0.25
        formality_score -= pronoun_penalty
        
        # Boost for longer words (more formal)
        long_words = sum(1 for token in doc if len(token.text) > 6)
        long_word_boost = (long_words / total_tokens) * 0.2
        formality_score += long_word_boost
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, formality_score))
    
    def _calculate_complexity(self, doc: Any) -> float:
        """Calculate linguistic complexity"""
        if len(doc) == 0:
            return 0.5
        
        # Sentence length complexity
        sentences = list(doc.sents)
        if sentences:
            avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences)
            length_score = min(1.0, avg_sentence_length / 30)  # 30 words = high complexity
        else:
            length_score = 0.3
        
        # Word length complexity
        avg_word_length = sum(len(token.text) for token in doc) / len(doc)
        word_score = min(1.0, avg_word_length / 8)  # 8 chars = high complexity
        
        # Dependency depth complexity (simplified)
        # Count unique dependency types as a proxy for syntactic complexity
        unique_deps = len(set(token.dep_ for token in doc))
        dep_score = min(1.0, unique_deps / 20)  # 20 unique deps = high complexity
        
        # Combine scores
        complexity = (length_score * 0.4 + word_score * 0.3 + dep_score * 0.3)
        
        return complexity
    
    def _calculate_confidence(self, doc: Any) -> float:
        """Calculate confidence in parse results"""
        if len(doc) == 0:
            return 0.0
        
        # Base confidence
        confidence = 0.95
        
        # Reduce confidence for very short texts
        if len(doc) < 3:
            confidence *= 0.8
        
        # Reduce confidence if many tokens have unknown POS
        unknown_pos = sum(1 for token in doc if token.pos_ == 'X' or token.pos_ == '')
        if unknown_pos > 0:
            confidence *= (1 - (unknown_pos / len(doc)) * 0.5)
        
        return confidence
    
    def _create_empty_parse(self, text: str) -> ParsedText:
        """Create empty parse for invalid input"""
        return ParsedText(
            raw_text=text,
            tokens=[],
            pos_tags=[],
            dependencies=[],
            entities=[],
            noun_chunks=[],
            sentences=[text] if text.strip() else [],
            confidence=0.0,
            metadata={'error': True, 'empty': True}
        )
    
    def _create_minimal_tokenizer(self) -> Any:
        """Fallback tokenizer when spaCy unavailable"""
        class MinimalTokenizer:
            def __call__(self, text):
                class Doc:
                    def __init__(self, text):
                        self.text = text
                        self.ents = []
                        self.noun_chunks = []
                        
                        # Simple tokenization
                        self.tokens = text.split()
                        
                    def __iter__(self):
                        class SimpleToken:
                            def __init__(self, text):
                                self.text = text
                                self.pos_ = 'NOUN'  # Default
                                self.dep_ = 'ROOT'
                                self.head = self
                                self.lang_ = 'en'
                        
                        return iter([SimpleToken(t) for t in self.tokens])
                    
                    def __len__(self):
                        return len(self.tokens)
                    
                    @property
                    def sents(self):
                        class Sent:
                            def __init__(self, text):
                                self.text = text
                        
                        return [Sent(self.text)]
                    
                    @property
                    def lang_(self):
                        return 'en'
                
                return Doc(text)
        
        return MinimalTokenizer()
    
    def _update_stats(self, parse_time: float, success: bool = True, error: str = "") -> None:
        """Update parsing statistics"""
        self.parsing_stats['total_parsed'] += 1
        if not success:
            self.parsing_stats['total_errors'] += 1
            if error:
                self.parsing_stats['errors'].append(error)
                # Keep only last 10 errors
                self.parsing_stats['errors'] = self.parsing_stats['errors'][-10:]
        
        # Update moving average
        total = self.parsing_stats['total_parsed']
        old_avg = self.parsing_stats['avg_parse_time']
        self.parsing_stats['avg_parse_time'] = old_avg + (parse_time - old_avg) / total
    
    def get_stats(self) -> Dict:
        """Return parsing statistics"""
        stats = self.parsing_stats.copy()
        # Add success rate
        if stats['total_parsed'] > 0:
            stats['success_rate'] = 1.0 - (stats['total_errors'] / stats['total_parsed'])
        else:
            stats['success_rate'] = 1.0
        return stats
    
    def reset_stats(self) -> None:
        """Reset parsing statistics"""
        self.parsing_stats = {
            'total_parsed': 0,
            'total_errors': 0,
            'avg_parse_time': 0.0,
            'errors': []
        }

# Connects to: intent_detection.py (provides parsed input for intent classification)
# Connects to: sentiment.py (provides tokens and linguistic features for sentiment analysis)
# Connects to: memory/working/ (stores parsed representation for context)
# Connects to: language/understanding/ (feeds into deeper comprehension)
# Connects to: emotion/appraisal.py (linguistic features inform emotional response)
# Connects to: self/theory_of_mind.py (parsing helps understand others' speech)