"""
Perception Package - Wednesday's Interface to the World
=======================================================
This package handles all sensory input processing including:
- Text understanding (parsing, intent, sentiment)
- Audio processing (speech-to-text, tone analysis)
- Vision processing (object recognition, face processing)
- Attention management (salience, focus)

The perception system acts as Wednesday's senses - taking raw input
from the world and converting it into structured, meaningful representations
that other modules can understand and act upon.
"""

import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime

# Import from submodules - these will be populated as we build each phase
from wednesday.perception.attention import (
    FocusManager,
    SalienceDetector
)

from wednesday.perception.text import (
    TextParser,
    IntentDetector,
    SentimentAnalyzer
)

from wednesday.perception.audio import (
    SpeechToText,
    ToneAnalyzer
)

from wednesday.perception.vision import (
    ObjectRecognizer,
    FaceProcessor
)

# Configure module logger
logger = logging.getLogger(__name__)


class PerceptionSystem:
    """
    Central perception coordinator - Wednesday's sensory integration hub.
    
    This class orchestrates all perception modules, providing a unified
    interface for processing multimodal input. It handles:
    - Routing input to appropriate processors
    - Fusing results from multiple modalities
    - Managing attention and focus
    - Providing rich, structured perception results
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the perception system with all its components.
        
        Args:
            config: Configuration dictionary for perception modules
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.PerceptionSystem")
        
        # Initialize all perception modules (lazy loading where appropriate)
        self._init_attention_modules()
        self._init_text_modules()
        self._init_audio_modules()
        self._init_vision_modules()
        
        # Track module readiness
        self.modules_ready = {
            'attention': True,
            'text': True,
            'audio': False,  # Not ready until configured/loaded
            'vision': False   # Not ready until configured/loaded
        }
        
        # Multi-modal fusion state
        self.current_focus = None
        self.last_perception = None
        self.perception_history = []
        
        self.logger.info("Perception system initialized")
    
    def _init_attention_modules(self):
        """Initialize attention and focus modules."""
        try:
            self.salience = SalienceDetector(
                self.config.get('attention', {})
            )
            self.focus = FocusManager(
                self.salience,
                self.config.get('focus', {})
            )
            self.logger.debug("Attention modules initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize attention modules: {e}")
            self.modules_ready['attention'] = False
    
    def _init_text_modules(self):
        """Initialize text processing modules."""
        try:
            self.text_parser = TextParser(
                self.config.get('text', {}).get('model_path')
            )
            self.intent_detector = IntentDetector(
                self.config.get('text', {}).get('intent_model')
            )
            self.sentiment = SentimentAnalyzer(
                self.config.get('text', {}).get('sentiment_config')
            )
            self.logger.debug("Text modules initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize text modules: {e}")
            self.modules_ready['text'] = False
    
    def _init_audio_modules(self):
        """Initialize audio processing modules (lazy loading)."""
        # Audio modules might require heavy models - initialize on demand
        self._speech_to_text = None
        self._tone_analyzer = None
    
    def _init_vision_modules(self):
        """Initialize vision processing modules (lazy loading)."""
        # Vision modules might require heavy models - initialize on demand
        self._object_recognizer = None
        self._face_processor = None
    
    @property
    def speech_to_text(self):
        """Lazy loader for speech-to-text module."""
        if self._speech_to_text is None:
            try:
                self._speech_to_text = SpeechToText(
                    self.config.get('audio', {}).get('stt_config')
                )
                self.modules_ready['audio'] = True
                self.logger.info("Speech-to-text module loaded")
            except Exception as e:
                self.logger.error(f"Failed to load speech-to-text: {e}")
                self.modules_ready['audio'] = False
        return self._speech_to_text
    
    @property
    def tone_analyzer(self):
        """Lazy loader for tone analysis module."""
        if self._tone_analyzer is None:
            try:
                self._tone_analyzer = ToneAnalyzer(
                    self.config.get('audio', {}).get('tone_config')
                )
                self.logger.info("Tone analyzer module loaded")
            except Exception as e:
                self.logger.error(f"Failed to load tone analyzer: {e}")
        return self._tone_analyzer
    
    @property
    def object_recognizer(self):
        """Lazy loader for object recognition module."""
        if self._object_recognizer is None:
            try:
                self._object_recognizer = ObjectRecognizer(
                    self.config.get('vision', {}).get('object_model')
                )
                self.modules_ready['vision'] = True
                self.logger.info("Object recognition module loaded")
            except Exception as e:
                self.logger.error(f"Failed to load object recognizer: {e}")
                self.modules_ready['vision'] = False
        return self._object_recognizer
    
    @property
    def face_processor(self):
        """Lazy loader for face processing module."""
        if self._face_processor is None:
            try:
                self._face_processor = FaceProcessor(
                    self.config.get('vision', {}).get('face_config')
                )
                self.logger.info("Face processor module loaded")
            except Exception as e:
                self.logger.error(f"Failed to load face processor: {e}")
        return self._face_processor
    
    def perceive(self, input_data: Any, modality: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point - process any type of input.
        
        This is the primary method for sending sensory input to Wednesday.
        It automatically detects the input type if not specified and routes
        to the appropriate processors.
        
        Args:
            input_data: Raw input (text, audio bytes, image, etc.)
            modality: Optional forced modality ('text', 'audio', 'vision')
            
        Returns:
            Rich perception result containing:
            - processed: Processed representation
            - intent: Detected intent (if applicable)
            - sentiment: Emotional content
            - attention: What to focus on
            - confidence: Processing confidence scores
            - timestamp: When perceived
        """
        start_time = datetime.now()
        
        # Auto-detect modality if not specified
        if modality is None:
            modality = self._detect_modality(input_data)
        
        self.logger.debug(f"Perceiving {modality} input")
        
        # Route to appropriate processor based on modality
        if modality == 'text':
            result = self._process_text(input_data)
        elif modality == 'audio':
            result = self._process_audio(input_data)
        elif modality == 'vision':
            result = self._process_vision(input_data)
        elif modality == 'multi':
            result = self._process_multimodal(input_data)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        # Apply attention to determine what's important
        result['attention'] = self._apply_attention(result)
        
        # Add metadata
        result['metadata'] = {
            'modality': modality,
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'timestamp': start_time.isoformat()
        }
        
        # Store in history
        self.last_perception = result
        self.perception_history.append({
            'timestamp': start_time,
            'modality': modality,
            'result_summary': self._summarize_result(result)
        })
        
        # Trim history if needed
        if len(self.perception_history) > 100:
            self.perception_history = self.perception_history[-100:]
        
        return result
    
    def _detect_modality(self, input_data: Any) -> str:
        """
        Automatically detect input modality.
        
        Args:
            input_data: Raw input
            
        Returns:
            Detected modality ('text', 'audio', 'vision', 'multi')
        """
        if isinstance(input_data, str):
            # Could be text or path to file
            if input_data.startswith(('http://', 'https://', '/', './')):
                # Probably a file path or URL - need to check extension
                if any(input_data.endswith(ext) for ext in ['.wav', '.mp3', '.m4a']):
                    return 'audio'
                elif any(input_data.endswith(ext) for ext in ['.jpg', '.png', '.jpeg', '.gif']):
                    return 'vision'
            return 'text'
        
        elif isinstance(input_data, bytes):
            # Could be audio or image bytes
            # Simple check based on first few bytes
            if input_data.startswith(b'\xff\xd8'):  # JPEG header
                return 'vision'
            elif input_data.startswith(b'RIFF'):  # WAV header
                return 'audio'
            else:
                return 'text'  # Assume text if not recognized
        
        elif hasattr(input_data, 'read'):  # File-like object
            # Defer to calling code to specify
            return 'multi'
        
        else:
            # Default to text for simple types
            return 'text'
    
    def _process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text input through all text modules.
        
        Args:
            text: Raw text string
            
        Returns:
            Text perception results
        """
        if not self.modules_ready['text']:
            self.logger.warning("Text modules not ready, attempting reinitialization")
            self._init_text_modules()
        
        # Parse the text
        parsed = self.text_parser.parse(text)
        
        # Extract entities
        entities = self.text_parser.extract_entities(parsed)
        
        # Detect intent
        intent = self.intent_detector.detect_intent(
            parsed,
            context={'entities': entities}
        )
        
        # Analyze sentiment
        sentiment = self.sentiment.analyze(text, context=parsed)
        
        # Check for sarcasm (Wednesday special)
        sarcasm_probability = self.sentiment.detect_sarcasm(text, parsed)
        
        return {
            'type': 'text',
            'raw': text,
            'processed': {
                'tokens': parsed.get('tokens'),
                'pos_tags': parsed.get('pos_tags'),
                'dependencies': parsed.get('dependencies')
            },
            'entities': entities,
            'intent': intent,
            'sentiment': sentiment,
            'sarcasm_probability': sarcasm_probability,
            'confidence': self._calculate_confidence({
                'parsed': parsed.get('confidence', 1.0),
                'intent': intent.get('confidence', 1.0),
                'sentiment': sentiment.get('confidence', 1.0)
            })
        }
    
    def _process_audio(self, audio_input: Union[bytes, str]) -> Dict[str, Any]:
        """
        Process audio input through speech and tone analysis.
        
        Args:
            audio_input: Audio bytes or file path
            
        Returns:
            Audio perception results
        """
        # Ensure audio modules are loaded
        stt = self.speech_to_text
        tone = self.tone_analyzer
        
        # Convert speech to text
        transcription = stt.transcribe(audio_input)
        
        # Analyze tone if available
        if tone:
            tone_analysis = tone.analyze(audio_input)
        else:
            tone_analysis = {'emotion': 'neutral', 'confidence': 0.5}
        
        # Process the transcribed text
        if transcription['text']:
            text_result = self._process_text(transcription['text'])
        else:
            text_result = {'intent': {'name': 'unknown'}, 'sentiment': {}}
        
        return {
            'type': 'audio',
            'transcription': transcription,
            'tone': tone_analysis,
            'text_analysis': text_result,
            'confidence': (transcription.get('confidence', 0) + 
                          tone_analysis.get('confidence', 0)) / 2
        }
    
    def _process_vision(self, image_input: Union[bytes, str]) -> Dict[str, Any]:
        """
        Process visual input through object and face recognition.
        
        Args:
            image_input: Image bytes, file path, or URL
            
        Returns:
            Vision perception results
        """
        # Ensure vision modules are loaded
        objects = self.object_recognizer
        faces = self.face_processor
        
        # Detect objects
        objects_result = objects.recognize(image_input)
        
        # Process faces if present
        if faces:
            faces_result = faces.process(image_input)
        else:
            faces_result = {'faces': [], 'count': 0}
        
        return {
            'type': 'vision',
            'objects': objects_result,
            'faces': faces_result,
            'scene': objects_result.get('scene', 'unknown'),
            'confidence': objects_result.get('confidence', 0.5)
        }
    
    def _process_multimodal(self, multi_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multimodal input (e.g., audio + video).
        
        Args:
            multi_input: Dictionary with keys for each modality
            
        Returns:
            Fused multimodal perception results
        """
        results = {}
        
        # Process each modality present
        for modality, data in multi_input.items():
            if modality == 'text':
                results['text'] = self._process_text(data)
            elif modality == 'audio':
                results['audio'] = self._process_audio(data)
            elif modality == 'vision':
                results['vision'] = self._process_vision(data)
        
        # Fuse results for better understanding
        fused = self._fuse_modalities(results)
        
        return {
            'type': 'multimodal',
            'modalities': list(results.keys()),
            'results': results,
            'fused': fused,
            'confidence': self._calculate_multimodal_confidence(results)
        }
    
    def _apply_attention(self, perception_result: Dict) -> Dict[str, Any]:
        """
        Determine what to focus on in the perception.
        
        Args:
            perception_result: Result from processing
            
        Returns:
            Attention directives (what's important)
        """
        # Update focus manager with new perception
        self.focus.update_context(perception_result)
        
        # Calculate salience
        salient_elements = self.salience.calculate_salience(perception_result)
        
        # Get current focus
        current_focus = self.focus.current_focus
        
        return {
            'current_focus': current_focus,
            'salient_elements': salient_elements,
            'attention_shift_needed': self.focus.should_shift_focus(salient_elements),
            'focus_history': self.focus.get_recent_focus(limit=5)
        }
    
    def _fuse_modalities(self, results: Dict) -> Dict[str, Any]:
        """
        Fuse results from multiple modalities for richer understanding.
        
        Args:
            results: Results from individual modality processors
            
        Returns:
            Fused understanding
        """
        fused = {
            'combined_intent': None,
            'emotional_coherence': 1.0,
            'cross_modal_consistency': True
        }
        
        # Example: Use vision to disambiguate audio
        if 'audio' in results and 'vision' in results:
            # Check if speaker's emotion matches tone
            audio_emotion = results['audio'].get('tone', {}).get('emotion')
            face_emotion = results['vision'].get('faces', {}).get('dominant_emotion')
            
            if audio_emotion and face_emotion:
                fused['emotional_coherence'] = (
                    1.0 if audio_emotion == face_emotion else 0.5
                )
        
        return fused
    
    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        """
        Calculate overall confidence from component scores.
        
        Args:
            scores: Dictionary of confidence scores from modules
            
        Returns:
            Aggregated confidence score
        """
        if not scores:
            return 0.0
        
        # Weighted average
        weights = {
            'parsed': 0.2,
            'intent': 0.4,
            'sentiment': 0.4
        }
        
        weighted_sum = sum(
            scores.get(key, 0) * weights.get(key, 0)
            for key in scores
        )
        
        return weighted_sum / sum(weights.values())
    
    def _calculate_multimodal_confidence(self, results: Dict) -> float:
        """
        Calculate confidence for multimodal perception.
        
        Args:
            results: Results from multiple modalities
            
        Returns:
            Combined confidence score
        """
        if not results:
            return 0.0
        
        # Average confidence across modalities
        confidences = [
            r.get('confidence', 0.5)
            for r in results.values()
            if isinstance(r, dict)
        ]
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _summarize_result(self, result: Dict) -> str:
        """
        Create a brief summary of perception result for logging.
        
        Args:
            result: Full perception result
            
        Returns:
            Brief summary string
        """
        result_type = result.get('type', 'unknown')
        
        if result_type == 'text':
            intent = result.get('intent', {}).get('name', 'unknown')
            return f"Text: intent={intent}"
        elif result_type == 'audio':
            text = result.get('transcription', {}).get('text', '')[:30]
            return f"Audio: '{text}...'"
        elif result_type == 'vision':
            objects = result.get('objects', {}).get('count', 0)
            faces = result.get('faces', {}).get('count', 0)
            return f"Vision: {objects} objects, {faces} faces"
        else:
            return f"{result_type} perception"
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of all perception modules.
        
        Returns:
            Status dictionary for monitoring/debugging
        """
        return {
            'modules_ready': self.modules_ready,
            'current_focus': self.current_focus,
            'history_length': len(self.perception_history),
            'last_perception_time': (
                self.perception_history[-1]['timestamp'].isoformat()
                if self.perception_history else None
            ),
            'config': self.config
        }
    
    def reload_module(self, module_name: str) -> bool:
        """
        Reload a specific perception module.
        
        Args:
            module_name: Name of module to reload
            
        Returns:
            Success status
        """
        self.logger.info(f"Reloading module: {module_name}")
        
        try:
            if module_name == 'text':
                self._init_text_modules()
            elif module_name == 'audio':
                self._speech_to_text = None
                self._tone_analyzer = None
                # Force reload on next use
            elif module_name == 'vision':
                self._object_recognizer = None
                self._face_processor = None
            elif module_name == 'attention':
                self._init_attention_modules()
            else:
                self.logger.error(f"Unknown module: {module_name}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to reload {module_name}: {e}")
            return False


# Convenience function for quick perception
def perceive(input_data: Any, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Quick one-off perception without initializing full system.
    
    Args:
        input_data: Raw input to perceive
        config: Optional configuration
        
    Returns:
        Perception result
    """
    system = PerceptionSystem(config)
    return system.perceive(input_data)


# What gets imported with "from wednesday.perception import *"
__all__ = [
    # Main class
    'PerceptionSystem',
    
    # Convenience function
    'perceive',
    
    # Submodule classes (explicitly expose)
    'FocusManager',
    'SalienceDetector',
    'TextParser',
    'IntentDetector',
    'SentimentAnalyzer',
    'SpeechToText',
    'ToneAnalyzer',
    'ObjectRecognizer',
    'FaceProcessor'
]

# Version info
__version__ = '0.1.0'

# Module initialization log
logger.info(f"Perception package v{__version__} loaded")