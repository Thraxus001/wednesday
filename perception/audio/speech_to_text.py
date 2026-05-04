"""
Converts speech to text; handles multiple languages; noise reduction.
Wednesday listens carefully, even when she seems uninterested.
She catches every word, every inflection.
"""
import wave
import io
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import threading
import queue
import time
import os
import speech_recognition as sr

# Audio processing libraries
try:
    import speech_recognition as sr
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logging.warning("Audio libraries not available. Install speech_recognition and pyaudio")

# Optional libraries for advanced processing
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

logger = logging.getLogger(__name__)

class AudioSource(Enum):
    """Types of audio input sources"""
    MICROPHONE = "microphone"
    FILE = "file"
    STREAM = "stream"
    SYSTEM = "system"  # System audio output

class LanguageCode(Enum):
    """Supported languages"""
    ENGLISH_US = "en-US"
    ENGLISH_UK = "en-GB"
    SPANISH = "es-ES"
    FRENCH = "fr-FR"
    GERMAN = "de-DE"
    ITALIAN = "it-IT"
    JAPANESE = "ja-JP"
    KOREAN = "ko-KR"
    CHINESE = "zh-CN"
    RUSSIAN = "ru-RU"
    # Add more as needed

class NoiseReductionLevel(Enum):
    """Levels of noise reduction"""
    OFF = 0
    LIGHT = 1
    MODERATE = 2
    AGGRESSIVE = 3

class RecognitionEngine(Enum):
    """Available speech recognition engines"""
    GOOGLE = "google"
    SPHINX = "sphinx"  # Offline
    WIT = "wit"
    AZURE = "azure"
    IBM = "ibm"

@dataclass
class TranscriptionResult:
    """Result of speech-to-text conversion"""
    text: str
    confidence: float
    language: LanguageCode
    raw_audio: Optional[np.ndarray] = None
    duration: float = 0.0  # seconds
    word_timings: List[Tuple[str, float, float]] = field(default_factory=list)  # (word, start, end)
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    
    # Audio characteristics
    sample_rate: int = 16000
    noise_level: float = 0.0  # Estimated background noise
    has_speech: bool = True
    is_partial: bool = False
    
    # Wednesday's observations
    speaker_gender: Optional[str] = None
    speaker_emotion: Optional[str] = None
    speaker_age_group: Optional[str] = None
    accent: Optional[str] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    source: AudioSource = AudioSource.MICROPHONE
    processing_time: float = 0.0
    engine_used: RecognitionEngine = RecognitionEngine.GOOGLE
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'language': self.language.value,
            'duration': self.duration,
            'has_speech': self.has_speech,
            'is_partial': self.is_partial,
            'timestamp': self.timestamp.isoformat(),
            'engine': self.engine_used.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TranscriptionResult':
        """Create TranscriptionResult from dictionary"""
        return cls(
            text=data.get('text', ''),
            confidence=data.get('confidence', 0.0),
            language=LanguageCode(data.get('language', 'en-US')),
            duration=data.get('duration', 0.0),
            has_speech=data.get('has_speech', True),
            is_partial=data.get('is_partial', False),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            engine_used=RecognitionEngine(data.get('engine', 'google'))
        )

@dataclass
class AudioStreamConfig:
    """Configuration for audio stream"""
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    format: Optional[int] = None  # Will be set in __post_init__
    input_device_index: Optional[int] = None
    stream_timeout: float = 5.0
    
    def __post_init__(self):
        if AUDIO_AVAILABLE and self.format is None:
            self.format = pyaudio.paInt16

class SpeechToText:
    """
    Converts speech to text; handles multiple languages; noise reduction.
    Wednesday may seem to be ignoring you, but she's processing every syllable.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self._configure_recognizer()
        
        # Audio processing
        self.audio_interface = None
        self.active_streams = {}
        
        # Language settings
        default_lang_str = self.config.get('default_language', 'en-US')
        try:
            self.default_language = LanguageCode(default_lang_str)
        except ValueError:
            logger.warning(f"Unsupported default language: {default_lang_str}, using en-US")
            self.default_language = LanguageCode.ENGLISH_US
            
        self.supported_languages = self._load_supported_languages()
        
        # Recognition engine
        engine_str = self.config.get('recognition_engine', 'google')
        try:
            self.recognition_engine = RecognitionEngine(engine_str)
        except ValueError:
            self.recognition_engine = RecognitionEngine.GOOGLE
        
        # API keys for different engines
        self.api_keys = self.config.get('api_keys', {})
        
        # Noise reduction settings
        noise_level_str = self.config.get('noise_reduction', 'moderate')
        try:
            self.noise_reduction_level = NoiseReductionLevel[noise_level_str.upper()]
        except KeyError:
            self.noise_reduction_level = NoiseReductionLevel.MODERATE
            
        self.ambient_noise_profile = None
        
        # Streaming settings
        self.stream_config = AudioStreamConfig(
            **{k: v for k, v in self.config.get('stream', {}).items() 
               if k in AudioStreamConfig.__dataclass_fields__}
        )
        
        # Callbacks for streaming
        self.partial_result_callbacks: List[Callable] = []
        self.final_result_callbacks: List[Callable] = []
        
        # Threading for async processing
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        
        # Performance tracking
        self.stats = {
            'total_transcriptions': 0,
            'total_duration': 0.0,
            'avg_confidence': 0.0,
            'errors': 0,
            'languages_detected': {}
        }
        
        logger.info(f"SpeechToText initialized. Default language: {self.default_language.value}")
        
        # Initialize audio if available
        if AUDIO_AVAILABLE:
            self._initialize_audio()
    
    def transcribe_file(self, 
                       file_path: str, 
                       language: Optional[LanguageCode] = None,
                       noise_reduction: Optional[NoiseReductionLevel] = None) -> TranscriptionResult:
        """
        Transcribe speech from an audio file.
        
        Args:
            file_path: Path to audio file
            language: Expected language (auto-detect if None)
            noise_reduction: Level of noise reduction to apply
            
        Returns:
            TranscriptionResult with text and metadata
        """
        start_time = time.time()
        
        if not AUDIO_AVAILABLE:
            logger.error("Audio libraries not available")
            return self._create_error_result("Audio libraries not installed")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return self._create_error_result(f"File not found: {file_path}")
        
        try:
            # Load audio file
            audio = self._load_audio_file(file_path)
            
            # Apply noise reduction if requested
            noise_level = noise_reduction or self.noise_reduction_level
            if noise_level != NoiseReductionLevel.OFF:
                audio = self._reduce_noise(audio, noise_level)
            
            # Perform transcription
            result = self._transcribe_audio(audio, language)
            
            # Add metadata
            result.source = AudioSource.FILE
            result.processing_time = time.time() - start_time
            
            # Update stats
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {e}")
            self.stats['errors'] += 1
            return self._create_error_result(str(e))
    
    def transcribe_microphone(self,
                            duration: Optional[float] = None,
                            language: Optional[LanguageCode] = None,
                            phrase_time_limit: Optional[float] = None) -> TranscriptionResult:
        """
        Transcribe speech from microphone.
        
        Args:
            duration: Maximum recording duration (None = until silence)
            language: Expected language
            phrase_time_limit: Maximum phrase length
            
        Returns:
            TranscriptionResult with text and metadata
        """
        start_time = time.time()
        
        if not AUDIO_AVAILABLE:
            logger.error("Audio libraries not available")
            return self._create_error_result("Audio libraries not installed")
        
        try:
            # Use microphone as source
            with sr.Microphone() as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                logger.info("Listening...")
                
                # Listen with or without duration
                try:
                    if duration:
                        audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=phrase_time_limit)
                    else:
                        audio = self.recognizer.listen(source, phrase_time_limit=phrase_time_limit)
                except sr.WaitTimeoutError:
                    logger.warning("No speech detected within timeout")
                    return self._create_error_result("No speech detected", is_partial=True)
                
                logger.info("Processing speech...")
                
                # Apply noise reduction
                if self.noise_reduction_level != NoiseReductionLevel.OFF:
                    audio = self._reduce_noise(audio, self.noise_reduction_level)
                
                # Perform transcription
                result = self._transcribe_audio(audio, language)
                
                # Add metadata
                result.source = AudioSource.MICROPHONE
                result.processing_time = time.time() - start_time
                
                # Update stats
                self._update_stats(result)
                
                return result
                
        except Exception as e:
            logger.error(f"Error transcribing from microphone: {e}")
            self.stats['errors'] += 1
            return self._create_error_result(str(e))
    
    def start_streaming(self, 
                       callback: Optional[Callable] = None,
                       language: Optional[LanguageCode] = None):
        """
        Start continuous streaming transcription.
        
        Args:
            callback: Function to call with partial results
            language: Expected language
        """
        if not AUDIO_AVAILABLE:
            logger.error("Audio libraries not available")
            return
        
        if self.running:
            logger.warning("Streaming already running")
            return
        
        self.running = True
        
        # Add callback if provided
        if callback:
            self.partial_result_callbacks.append(callback)
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._streaming_loop, 
            args=(language,),
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("Started streaming transcription")
    
    def stop_streaming(self):
        """Stop streaming transcription"""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        # Clean up audio interface
        if self.audio_interface:
            self.audio_interface.terminate()
            self.audio_interface = None
        
        logger.info("Stopped streaming transcription")
    
    def add_partial_callback(self, callback: Callable):
        """Add callback for partial transcription results"""
        self.partial_result_callbacks.append(callback)
    
    def add_final_callback(self, callback: Callable):
        """Add callback for final transcription results"""
        self.final_result_callbacks.append(callback)
    
    def set_noise_reduction(self, level: Union[NoiseReductionLevel, str]):
        """Set noise reduction level"""
        if isinstance(level, str):
            try:
                level = NoiseReductionLevel[level.upper()]
            except KeyError:
                logger.warning(f"Invalid noise reduction level: {level}")
                return
        
        self.noise_reduction_level = level
        logger.info(f"Noise reduction set to {level.name}")
    
    def calibrate_ambient_noise(self, duration: float = 3.0):
        """Calibrate ambient noise profile"""
        if not AUDIO_AVAILABLE:
            return
        
        try:
            with sr.Microphone() as source:
                logger.info(f"Calibrating ambient noise for {duration} seconds...")
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
                self.ambient_noise_profile = self.recognizer.energy_threshold
                logger.info(f"Ambient noise calibrated. Threshold: {self.ambient_noise_profile:.2f}")
        except Exception as e:
            logger.error(f"Error calibrating ambient noise: {e}")
    
    def detect_language(self, audio: sr.AudioData) -> LanguageCode:
        """
        Detect language from audio.
        
        Args:
            audio: Audio data
            
        Returns:
            Detected language code
        """
        # This would use a language detection model
        # Placeholder implementation using Google's language detection
        try:
            # Try to recognize with different language models
            test_languages = [
                LanguageCode.ENGLISH_US,
                LanguageCode.SPANISH, 
                LanguageCode.FRENCH,
                LanguageCode.GERMAN
            ]
            
            for lang in test_languages:
                try:
                    # Quick test with short timeout
                    text = self.recognizer.recognize_google(
                        audio, 
                        language=lang.value, 
                        show_all=False
                    )
                    if text and len(text) > 3:  # Got meaningful text
                        return lang
                except (sr.UnknownValueError, sr.RequestError):
                    continue
        except Exception as e:
            logger.debug(f"Language detection error: {e}")
        
        return self.default_language
    
    def _configure_recognizer(self):
        """Configure speech recognizer settings"""
        # Set energy threshold (lower = more sensitive)
        self.recognizer.energy_threshold = self.config.get('energy_threshold', 300)
        
        # Set pause threshold (seconds of silence to end phrase)
        self.recognizer.pause_threshold = self.config.get('pause_threshold', 0.8)
        
        # Set phrase threshold (minimum seconds to consider a phrase)
        self.recognizer.phrase_threshold = self.config.get('phrase_threshold', 0.3)
        
        # Set non-speaking duration (seconds of non-speaking before considering phrase ended)
        self.recognizer.non_speaking_duration = self.config.get('non_speaking_duration', 0.5)
        
        # Set dynamic energy threshold
        self.recognizer.dynamic_energy_threshold = self.config.get('dynamic_energy', True)
    
    def _initialize_audio(self):
        """Initialize PyAudio interface"""
        try:
            self.audio_interface = pyaudio.PyAudio()
            
            # Log available devices at debug level
            logger.debug("Available audio input devices:")
            for i in range(self.audio_interface.get_device_count()):
                device_info = self.audio_interface.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    logger.debug(f"  {i}: {device_info['name']}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio: {e}")
            self.audio_interface = None
    
    def _load_audio_file(self, file_path: str) -> sr.AudioData:
        """
        Load audio file and convert to SpeechRecognition format.
        
        Supports: WAV, MP3, FLAC, OGG, etc.
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # If it's a WAV file, use wave module directly
            if file_ext == '.wav':
                with wave.open(file_path, 'rb') as wav_file:
                    frames = wav_file.readframes(-1)
                    sample_width = wav_file.getsampwidth()
                    sample_rate = wav_file.getframerate()
                    
                return sr.AudioData(frames, sample_rate, sample_width)
            
            # For other formats, use pydub if available
            elif PYDUB_AVAILABLE:
                audio_segment = AudioSegment.from_file(file_path)
                
                # Convert to WAV in memory
                wav_io = io.BytesIO()
                audio_segment.export(wav_io, format="wav")
                wav_io.seek(0)
                
                # Read WAV data
                with wave.open(wav_io, 'rb') as wav_file:
                    frames = wav_file.readframes(-1)
                    sample_width = wav_file.getsampwidth()
                    sample_rate = wav_file.getframerate()
                
                return sr.AudioData(frames, sample_rate, sample_width)
            
            else:
                raise ImportError("pydub required for non-WAV files")
            
        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            raise
    
    def _reduce_noise(self, 
                     audio: sr.AudioData, 
                     level: NoiseReductionLevel) -> sr.AudioData:
        """
        Apply noise reduction to audio.
        """
        if level == NoiseReductionLevel.OFF:
            return audio
        
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio.frame_data, dtype=np.int16).astype(np.float32)
            
            # Apply different levels of noise reduction
            if level == NoiseReductionLevel.LIGHT:
                # Simple high-pass filter
                if LIBROSA_AVAILABLE:
                    audio_array = librosa.effects.preemphasis(audio_array, coef=0.97)
                else:
                    # Simple DC removal
                    audio_array = audio_array - np.mean(audio_array)
                    
            elif level == NoiseReductionLevel.MODERATE:
                if LIBROSA_AVAILABLE:
                    # More aggressive filtering
                    audio_array = librosa.effects.preemphasis(audio_array, coef=0.95)
                    # Apply harmonic separation
                    audio_array = librosa.effects.harmonic(audio_array)
                else:
                    # Simple bandpass approximation
                    audio_array = audio_array - np.mean(audio_array)
                    # Normalize
                    audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-6)
                    
            elif level == NoiseReductionLevel.AGGRESSIVE:
                # Use noise profile if available
                if self.ambient_noise_profile and LIBROSA_AVAILABLE:
                    # Advanced noise reduction would go here
                    # This is a placeholder
                    pass
            
            # Convert back to int16
            audio_array = np.clip(audio_array, -32768, 32767).astype(np.int16)
            audio_data = audio_array.tobytes()
            
            # Create new AudioData object
            return sr.AudioData(audio_data, audio.sample_rate, audio.sample_width)
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    def _transcribe_audio(self, 
                         audio: sr.AudioData, 
                         language: Optional[LanguageCode]) -> TranscriptionResult:
        """
        Perform actual transcription using selected recognition engine.
        """
        # Detect language if not specified
        if not language:
            language = self.detect_language(audio)
        
        try:
            result = None
            confidence = 0.0
            text = ""
            alternatives = []
            
            # Use selected recognition engine
            if self.recognition_engine == RecognitionEngine.GOOGLE:
                result = self.recognizer.recognize_google(
                    audio, 
                    language=language.value,
                    show_all=True
                )
                
            elif self.recognition_engine == RecognitionEngine.SPHINX:
                # Offline recognition (less accurate)
                text = self.recognizer.recognize_sphinx(audio)
                confidence = 0.6  # Sphinx doesn't provide confidence
                
            elif self.recognition_engine == RecognitionEngine.WIT:
                api_key = self.api_keys.get('wit')
                if api_key:
                    result = self.recognizer.recognize_wit(audio, key=api_key)
                else:
                    logger.error("Wit.ai API key not provided")
                    
            elif self.recognition_engine == RecognitionEngine.AZURE:
                api_key = self.api_keys.get('azure')
                if api_key:
                    result = self.recognizer.recognize_azure(audio, key=api_key)
                else:
                    logger.error("Azure API key not provided")
            
            # Parse result based on engine output format
            if result and isinstance(result, dict) and 'alternative' in result:
                # Google format with alternatives
                primary = result['alternative'][0]
                text = primary.get('transcript', '')
                confidence = primary.get('confidence', 0.8)
                
                # Extract alternatives
                for alt in result['alternative'][1:4]:  # Top 3 alternatives
                    alt_text = alt.get('transcript', '')
                    alt_conf = alt.get('confidence', 0.5)
                    if alt_text:
                        alternatives.append((alt_text, alt_conf))
                        
            elif result and isinstance(result, str):
                # Simple string result
                text = result
                confidence = 0.7  # Default confidence
                
            elif result and isinstance(result, list):
                # List of hypotheses
                if result:
                    text = result[0]
                    confidence = 0.7
            
            # Calculate duration (approximate from audio length)
            duration = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
            
            # Detect if speech present
            has_speech = bool(text.strip())
            
            # Estimate noise level (simplified)
            audio_array = np.frombuffer(audio.frame_data, dtype=np.int16)
            noise_level = float(np.std(audio_array) / 32768.0)  # Normalize to 0-1
            
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language=language,
                duration=duration,
                alternatives=alternatives,
                noise_level=min(1.0, noise_level),
                has_speech=has_speech,
                sample_rate=audio.sample_rate,
                engine_used=self.recognition_engine
            )
                
        except sr.UnknownValueError:
            return self._create_error_result("Could not understand audio", is_partial=True)
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return self._create_error_result(f"Service error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in transcription: {e}")
            return self._create_error_result(f"Transcription error: {e}")
    
    def _streaming_loop(self, language: Optional[LanguageCode]):
        """
        Main loop for streaming transcription.
        """
        if not self.audio_interface:
            logger.error("Audio interface not initialized")
            return
        
        stream = None
        try:
            # Open microphone stream
            stream = self.audio_interface.open(
                format=self.stream_config.format,
                channels=self.stream_config.channels,
                rate=self.stream_config.sample_rate,
                input=True,
                input_device_index=self.stream_config.input_device_index,
                frames_per_buffer=self.stream_config.chunk_size
            )
            
            logger.info("Streaming started. Listening...")
            
            buffer = []
            silence_duration = 0.0
            min_speech_duration = 0.5  # seconds
            silence_threshold = self.recognizer.energy_threshold
            
            while self.running:
                try:
                    # Read audio chunk
                    data = stream.read(self.stream_config.chunk_size, exception_on_overflow=False)
                    buffer.append(data)
                    
                    # Calculate audio energy
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    energy = float(np.abs(audio_array).mean())
                    
                    # Check if speech is present
                    if energy > silence_threshold:
                        silence_duration = 0.0
                    else:
                        silence_duration += len(data) / (self.stream_config.sample_rate * 2)
                    
                    # Process when we have enough speech followed by silence
                    buffer_duration = len(buffer) * self.stream_config.chunk_size / self.stream_config.sample_rate
                    
                    if buffer_duration >= min_speech_duration and silence_duration > self.recognizer.pause_threshold:
                        # Process this chunk
                        self._process_audio_chunk(buffer, language)
                        
                        # Clear buffer
                        buffer = []
                        silence_duration = 0.0
                    
                    # Small sleep to prevent CPU overload
                    time.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Error in streaming loop iteration: {e}")
                    time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            logger.info("Streaming loop ended")
    
    def _process_audio_chunk(self, buffer: List[bytes], language: Optional[LanguageCode]):
        """
        Process a chunk of audio from stream.
        """
        try:
            # Combine chunks
            audio_data = b''.join(buffer)
            
            # Create AudioData object
            audio = sr.AudioData(
                audio_data, 
                self.stream_config.sample_rate,
                2  # 16-bit = 2 bytes
            )
            
            # Apply noise reduction
            if self.noise_reduction_level != NoiseReductionLevel.OFF:
                audio = self._reduce_noise(audio, self.noise_reduction_level)
            
            # Transcribe
            result = self._transcribe_audio(audio, language)
            
            # Mark as partial (streaming result)
            result.is_partial = True
            
            # Call partial callbacks
            for callback in self.partial_result_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in partial callback: {e}")
            
            # If confidence is high enough, treat as final
            if result.confidence > 0.8 and result.text:
                result.is_partial = False
                for callback in self.final_result_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Error in final callback: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    def _load_supported_languages(self) -> Dict[str, LanguageCode]:
        """Load supported languages from config"""
        # Default supported languages
        languages = {
            'en-US': LanguageCode.ENGLISH_US,
            'en-GB': LanguageCode.ENGLISH_UK,
            'es-ES': LanguageCode.SPANISH,
            'fr-FR': LanguageCode.FRENCH,
            'de-DE': LanguageCode.GERMAN,
            'it-IT': LanguageCode.ITALIAN,
            'ja-JP': LanguageCode.JAPANESE,
            'ko-KR': LanguageCode.KOREAN,
            'zh-CN': LanguageCode.CHINESE,
            'ru-RU': LanguageCode.RUSSIAN,
        }
        
        # Add from config
        custom_langs = self.config.get('languages', {})
        for code, name in custom_langs.items():
            try:
                languages[code] = LanguageCode(code)
            except ValueError:
                logger.warning(f"Unsupported language code: {code}")
        
        return languages
    
    def _create_error_result(self, error_message: str, is_partial: bool = False) -> TranscriptionResult:
        """Create error result"""
        return TranscriptionResult(
            text=f"",
            confidence=0.0,
            language=self.default_language,
            has_speech=False,
            is_partial=is_partial,
            engine_used=self.recognition_engine
        )
    
    def _update_stats(self, result: TranscriptionResult):
        """Update transcription statistics"""
        self.stats['total_transcriptions'] += 1
        self.stats['total_duration'] += result.duration
        
        # Update average confidence
        total = self.stats['total_transcriptions']
        old_avg = self.stats['avg_confidence']
        self.stats['avg_confidence'] = old_avg + (result.confidence - old_avg) / total
        
        # Track language distribution
        lang = result.language.value
        self.stats['languages_detected'][lang] = self.stats['languages_detected'].get(lang, 0) + 1
    
    def get_stats(self) -> Dict:
        """Return transcription statistics"""
        return dict(self.stats)
    
    def get_available_languages(self) -> List[str]:
        """Return list of supported language codes"""
        return list(self.supported_languages.keys())
    
    def set_device(self, device_index: int):
        """Set input device by index"""
        self.stream_config.input_device_index = device_index
        logger.info(f"Input device set to index {device_index}")
    
    def set_recognition_engine(self, engine: Union[RecognitionEngine, str]):
        """Set the recognition engine to use"""
        if isinstance(engine, str):
            try:
                engine = RecognitionEngine(engine.lower())
            except ValueError:
                logger.warning(f"Invalid recognition engine: {engine}")
                return
        
        self.recognition_engine = engine
        logger.info(f"Recognition engine set to {engine.value}")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_streaming()

# Connects to: perception/attention/ (audio input triggers attention)
# Connects to: memory/working/ (stores transcriptions for context)
# Connects to: language/understanding/ (provides text for further processing)
# Connects to: self/theory_of_mind.py (speaker characteristics inform understanding)
# Connects to: emotion/appraisal.py (tone and emotion detected)