"""
Classifies environmental sounds; detects events; identifies sound sources.
Wednesday hears everything - the creak of a floorboard, the hum of electronics,
the distant thunder. Every sound tells a story.
"""
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import time
from collections import defaultdict, deque

# Audio processing libraries
try:
    import librosa
    import scipy.signal as signal
    from scipy.ndimage import label
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    logging.warning(f"Audio analysis libraries not available: {e}. Install librosa, scipy")

logger = logging.getLogger(__name__)

class SoundCategory(Enum):
    """Categories of environmental sounds"""
    # Human sounds
    HUMAN_SPEECH = "human_speech"
    HUMAN_FOOTSTEPS = "human_footsteps"
    HUMAN_LAUGHTER = "human_laughter"
    HUMAN_COUGH = "human_cough"
    HUMAN_CRYING = "human_crying"
    HUMAN_WHISPER = "human_whisper"
    HUMAN_BREATHING = "human_breathing"
    
    # Animal sounds
    ANIMAL_BARK = "animal_bark"
    ANIMAL_MEOW = "animal_meow"
    ANIMAL_BIRD = "animal_bird"
    ANIMAL_INSECT = "animal_insect"
    
    # Nature sounds
    NATURE_RAIN = "nature_rain"
    NATURE_THUNDER = "nature_thunder"
    NATURE_WIND = "nature_wind"
    NATURE_WATER = "nature_water"  # streams, waves
    NATURE_FIRE = "nature_fire"
    
    # Mechanical sounds
    MACHINE_ENGINE = "machine_engine"
    MACHINE_FAN = "machine_fan"
    MACHINE_HUM = "machine_hum"
    MACHINE_ALARM = "machine_alarm"
    MACHINE_BEEP = "machine_beep"
    MACHINE_CLICK = "machine_click"
    
    # Electronic sounds
    ELECTRONIC_KEYBOARD = "electronic_keyboard"
    ELECTRONIC_MOUSE = "electronic_mouse"
    ELECTRONIC_NOTIFICATION = "electronic_notification"
    ELECTRONIC_STATIC = "electronic_static"
    
    # Environmental
    ENV_DOOR = "env_door"  # opening/closing
    ENV_WINDOW = "env_window"
    ENV_GLASS_BREAK = "env_glass_break"
    ENV_IMPACT = "env_impact"  # objects hitting
    ENV_EXPLOSION = "env_explosion"
    ENV_SIREN = "env_siren"
    
    # Music
    MUSIC_INSTRUMENT = "music_instrument"
    MUSIC_VOICE = "music_voice"
    MUSIC_RHYTHM = "music_rhythm"
    
    # Silence/background
    SILENCE = "silence"
    AMBIENT = "ambient"
    UNKNOWN = "unknown"

class SoundSource(Enum):
    """Source/origin of sound"""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    NEARBY = "nearby"  # < 5 meters
    DISTANT = "distant"  # > 5 meters
    MOVING = "moving"
    STATIONARY = "stationary"
    DIGITAL = "digital"  # from device
    PHYSICAL = "physical"

@dataclass
class SoundEvent:
    """Represents a detected sound event"""
    category: SoundCategory
    confidence: float
    start_time: float  # seconds from start of audio
    end_time: float
    duration: float
    
    # Sound characteristics
    frequency_range: Tuple[float, float]  # Hz
    intensity: float  # dB
    pitch: float  # Hz
    rhythm: Optional[float] = None  # beats per minute for rhythmic sounds
    
    # Localization
    direction: Optional[float] = None  # angle in degrees (0 = front)
    distance: Optional[float] = None  # estimated meters
    source_type: Optional[SoundSource] = None
    
    # Multiple possibilities
    alternatives: List[Tuple[SoundCategory, float]] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'category': self.category.value,
            'confidence': float(self.confidence),
            'start_time': float(self.start_time),
            'end_time': float(self.end_time),
            'duration': float(self.duration),
            'intensity': float(self.intensity),
            'pitch': float(self.pitch),
            'direction': float(self.direction) if self.direction is not None else None,
            'distance': float(self.distance) if self.distance is not None else None,
            'source_type': self.source_type.value if self.source_type else None,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SoundEvent':
        """Create SoundEvent from dictionary"""
        return cls(
            category=SoundCategory(data['category']),
            confidence=data['confidence'],
            start_time=data['start_time'],
            end_time=data['end_time'],
            duration=data['duration'],
            frequency_range=(0, 0),  # Not stored in dict
            intensity=data['intensity'],
            pitch=data.get('pitch', 0),
            direction=data.get('direction'),
            distance=data.get('distance'),
            source_type=SoundSource(data['source_type']) if data.get('source_type') else None,
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now()
        )

@dataclass
class Soundscape:
    """Complete analysis of environmental sounds"""
    events: List[SoundEvent] = field(default_factory=list)
    background_sound: SoundCategory = SoundCategory.AMBIENT
    background_intensity: float = 0.0
    
    # Scene characteristics
    primary_sounds: List[Tuple[SoundCategory, float]] = field(default_factory=list)
    sound_density: float = 0.0  # events per minute
    sound_diversity: float = 0.0  # number of distinct sound types
    
    # Temporal patterns
    sound_sequence: List[SoundEvent] = field(default_factory=list)
    repeating_patterns: List[List[SoundCategory]] = field(default_factory=list)
    
    # Environmental context
    is_noisy: bool = False
    is_quiet: bool = False
    has_human_presence: bool = False
    has_animal_presence: bool = False
    has_machinery: bool = False
    
    # Wednesday's observations
    anomalies: List[SoundEvent] = field(default_factory=list)  # Unexpected sounds
    threats: List[SoundEvent] = field(default_factory=list)  # Potentially dangerous
    points_of_interest: List[SoundEvent] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'event_count': len(self.events),
            'primary_sounds': [(c.value, float(s)) for c, s in self.primary_sounds[:3]],
            'sound_density': float(self.sound_density),
            'sound_diversity': float(self.sound_diversity),
            'is_noisy': self.is_noisy,
            'has_human_presence': self.has_human_presence,
            'anomaly_count': len(self.anomalies),
            'threat_count': len(self.threats),
            'timestamp': self.timestamp.isoformat(),
            'duration': float(self.duration)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Soundscape':
        """Create Soundscape from dictionary"""
        return cls(
            events=[],  # Events would need separate loading
            primary_sounds=[(SoundCategory(c), s) for c, s in data.get('primary_sounds', [])],
            sound_density=data.get('sound_density', 0.0),
            sound_diversity=data.get('sound_diversity', 0.0),
            is_noisy=data.get('is_noisy', False),
            has_human_presence=data.get('has_human_presence', False),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            duration=data.get('duration', 0.0)
        )

class SoundClassifier:
    """
    Classifies environmental sounds; detects events; identifies sound sources.
    Wednesday's hearing is exceptional - she notices the mouse in the wall,
    the flicker in the fluorescent light, the footsteps approaching.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Sound models
        self.sound_templates = self._load_sound_templates()
        self.frequency_profiles = self._load_frequency_profiles()
        
        # ML classifier (if available)
        self.ml_classifier = None
        self._load_ml_classifier()
        
        # Feature extractors
        self.feature_extractors = self._init_feature_extractors()
        
        # Sound library for known sounds
        self.sound_library = defaultdict(dict)  # location -> name -> template
        
        # Event detection
        self.event_buffer = deque(maxlen=100)  # Last 100 events
        self.pattern_detector = PatternDetector()
        
        # Localization
        self.microphone_array = self.config.get('microphone_array', False)
        self.mic_positions = self.config.get('mic_positions', [])
        
        # Background modeling
        self.background_model = None
        self.background_update_rate = self.config.get('background_update_rate', 0.1)
        
        # Streaming settings
        self.running = False
        self.processing_thread = None
        
        # Performance tracking
        self.stats = {
            'total_events': 0,
            'classifications': 0,
            'avg_confidence': 0.0,
            'errors': 0,
            'categories_found': defaultdict(int)
        }
        
        logger.info(f"SoundClassifier initialized with {len(SoundCategory)} sound categories")
    
    def classify(self, 
                audio_data: Union[np.ndarray, bytes],
                sample_rate: int,
                context: Optional[Dict] = None) -> Soundscape:
        """
        Classify sounds in audio data.
        
        Args:
            audio_data: Raw audio samples
            sample_rate: Sample rate of audio
            context: Optional context (location, time, etc.)
            
        Returns:
            Soundscape with detected sound events
        """
        import time
        start_time = time.time()
        
        if not AUDIO_AVAILABLE:
            logger.error("Audio analysis libraries not available")
            return Soundscape()
        
        try:
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_data, np.ndarray) and audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            
            # Ensure audio is not empty
            if len(audio_data) == 0:
                logger.warning("Empty audio data")
                return Soundscape()
            
            # Extract features
            features = self._extract_features(audio_data, sample_rate)
            
            # Update background model
            self._update_background_model(features)
            
            # Detect sound events
            events = self._detect_events(audio_data, sample_rate, features, context)
            
            # Classify each event
            classified_events = []
            for event_segment, start_sample, end_sample in events:
                sound_event = self._classify_event(event_segment, sample_rate, context)
                if sound_event:
                    # Set absolute timing
                    sound_event.start_time = start_sample / sample_rate
                    sound_event.end_time = end_sample / sample_rate
                    classified_events.append(sound_event)
                    
                    # Update stats
                    self.stats['categories_found'][sound_event.category.value] += 1
            
            # Analyze soundscape
            soundscape = self._analyze_soundscape(classified_events, features, context)
            soundscape.duration = len(audio_data) / sample_rate
            soundscape.processing_time = time.time() - start_time
            
            # Update event buffer
            self.event_buffer.extend(classified_events)
            
            # Detect patterns
            if len(self.event_buffer) >= self.pattern_detector.pattern_length * 2:
                soundscape.repeating_patterns = self.pattern_detector.detect_patterns(
                    list(self.event_buffer)
                )
            
            # Wednesday's special analysis
            soundscape = self._apply_wednesday_analysis(soundscape, context)
            
            # Update stats
            self._update_stats(classified_events, soundscape.processing_time)
            
            return soundscape
            
        except Exception as e:
            logger.error(f"Error in sound classification: {e}", exc_info=True)
            self.stats['errors'] += 1
            return Soundscape()
    
    def classify_streaming(self,
                          audio_chunk: np.ndarray,
                          sample_rate: int,
                          previous_soundscape: Optional[Soundscape] = None) -> Soundscape:
        """
        Classify sounds in streaming audio.
        
        Args:
            audio_chunk: New chunk of audio
            sample_rate: Sample rate
            previous_soundscape: Previous analysis for continuity
            
        Returns:
            Updated soundscape
        """
        # For streaming, we'd maintain state and update incrementally
        # For now, just classify each chunk independently
        return self.classify(audio_chunk, sample_rate)
    
    def identify_sound_source(self, 
                            sound_event: SoundEvent,
                            context: Optional[Dict] = None) -> Optional[str]:
        """
        Attempt to identify specific sound source (e.g., "front door", "refrigerator").
        """
        if not context or 'location' not in context:
            return None
        
        location = context['location']
        
        # Match with known sound sources in this location
        if location in self.sound_library:
            for source_name, template in self.sound_library[location].items():
                if self._match_source_template(sound_event, template):
                    return source_name
        
        return None
    
    def learn_sound(self, 
                   name: str,
                   audio_data: np.ndarray,
                   sample_rate: int,
                   location: str):
        """
        Learn a new sound and associate it with a name/location.
        """
        # Classify the sound first to get features
        features = self._extract_features(audio_data, sample_rate)
        
        # Get basic characteristics
        duration = len(audio_data) / sample_rate
        
        # Frequency analysis
        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)
        magnitudes = np.abs(fft)
        
        # Find dominant frequency
        positive_freqs = freqs[:len(freqs)//2]
        positive_mags = magnitudes[:len(magnitudes)//2]
        if len(positive_mags) > 0:
            dominant_freq = positive_freqs[np.argmax(positive_mags)]
        else:
            dominant_freq = 0
        
        # Store sound template
        template = {
            'name': name,
            'features': features,
            'duration': duration,
            'dominant_freq': dominant_freq,
            'timestamp': datetime.now(),
            'location': location
        }
        
        self.sound_library[location][name] = template
        logger.info(f"Learned new sound: {name} at {location}")
    
    def locate_sound(self, 
                    audio_channels: List[np.ndarray],
                    sample_rate: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimate sound direction and distance using multiple microphones.
        
        Args:
            audio_channels: List of audio from each microphone
            sample_rate: Sample rate
            
        Returns:
            (direction_angle, distance) or (None, None) if not localized
        """
        if not self.microphone_array or len(audio_channels) < 2:
            return None, None
        
        try:
            # Calculate time difference of arrival
            # Ensure same length
            min_len = min(len(ch) for ch in audio_channels)
            ch1 = audio_channels[0][:min_len]
            ch2 = audio_channels[1][:min_len]
            
            # Cross-correlation
            correlation = np.correlate(ch1, ch2, mode='same')
            delay_samples = np.argmax(np.abs(correlation)) - len(correlation) // 2
            
            # Convert to time delay
            delay_seconds = delay_samples / sample_rate
            
            # Calculate angle (simplified - assumes linear array)
            if self.mic_positions and len(self.mic_positions) >= 2:
                mic_distance = abs(self.mic_positions[1] - self.mic_positions[0])
                speed_sound = 343  # m/s
                
                if mic_distance > 0 and abs(delay_seconds) <= mic_distance / speed_sound:
                    angle = np.arcsin((delay_seconds * speed_sound) / mic_distance)
                    angle_degrees = float(np.degrees(angle))
                    
                    # Estimate distance from intensity (simplified)
                    intensity_ratio = np.std(ch1) / (np.std(ch2) + 1e-6)
                    # This would need calibration
                    distance = None
                    
                    return angle_degrees, distance
            
        except Exception as e:
            logger.warning(f"Sound localization failed: {e}")
        
        return None, None
    
    def _load_sound_templates(self) -> Dict[SoundCategory, List[Dict]]:
        """
        Load templates for known sounds.
        In production, this would load from files.
        """
        templates = {}
        
        # Human sounds
        templates[SoundCategory.HUMAN_FOOTSTEPS] = [
            {'freq_range': (80, 200), 'duration_range': (0.2, 0.5), 'rhythm_range': (1, 2)}
        ]
        templates[SoundCategory.HUMAN_LAUGHTER] = [
            {'freq_range': (200, 2000), 'duration_range': (0.5, 3), 'rhythm_range': (3, 6)}
        ]
        templates[SoundCategory.HUMAN_COUGH] = [
            {'freq_range': (300, 3000), 'duration_range': (0.1, 0.5)}
        ]
        templates[SoundCategory.HUMAN_CRYING] = [
            {'freq_range': (300, 2000), 'duration_range': (1, 10)}
        ]
        templates[SoundCategory.HUMAN_WHISPER] = [
            {'freq_range': (1000, 4000), 'intensity_range': (-40, -20)}
        ]
        
        # Animal sounds
        templates[SoundCategory.ANIMAL_BARK] = [
            {'freq_range': (400, 2000), 'duration_range': (0.1, 0.5), 'rhythm_range': (2, 4)}
        ]
        templates[SoundCategory.ANIMAL_BIRD] = [
            {'freq_range': (2000, 8000), 'duration_range': (0.5, 3), 'rhythm_range': (2, 8)}
        ]
        
        # Nature sounds
        templates[SoundCategory.NATURE_RAIN] = [
            {'freq_range': (2000, 8000), 'continuous': True, 'intensity_variation': 0.3}
        ]
        templates[SoundCategory.NATURE_THUNDER] = [
            {'freq_range': (20, 200), 'duration_range': (1, 5), 'intensity_peak': True}
        ]
        templates[SoundCategory.NATURE_WIND] = [
            {'freq_range': (50, 500), 'continuous': True, 'intensity_variation': 0.5}
        ]
        
        # Mechanical sounds
        templates[SoundCategory.MACHINE_HUM] = [
            {'freq_range': (50, 120), 'continuous': True, 'stable_pitch': True}
        ]
        templates[SoundCategory.MACHINE_ALARM] = [
            {'freq_range': (1000, 4000), 'rhythm_range': (2, 5), 'repeating': True}
        ]
        templates[SoundCategory.MACHINE_BEEP] = [
            {'freq_range': (1000, 3000), 'duration_range': (0.05, 0.2), 'repeating': True}
        ]
        templates[SoundCategory.MACHINE_CLICK] = [
            {'freq_range': (2000, 5000), 'duration_range': (0.01, 0.05)}
        ]
        
        # Environmental
        templates[SoundCategory.ENV_DOOR] = [
            {'freq_range': (100, 500), 'duration_range': (0.3, 1), 'attack': 'sharp'}
        ]
        templates[SoundCategory.ENV_GLASS_BREAK] = [
            {'freq_range': (2000, 10000), 'duration_range': (0.5, 2), 'attack': 'very_sharp'}
        ]
        templates[SoundCategory.ENV_SIREN] = [
            {'freq_range': (500, 2000), 'rhythm_range': (0.5, 2), 'repeating': True}
        ]
        
        return templates
    
    def _load_frequency_profiles(self) -> Dict[SoundCategory, Tuple[float, float]]:
        """Load typical frequency ranges for sound categories"""
        return {
            SoundCategory.HUMAN_SPEECH: (300, 3400),
            SoundCategory.HUMAN_WHISPER: (1000, 4000),
            SoundCategory.HUMAN_FOOTSTEPS: (50, 200),
            SoundCategory.ANIMAL_BARK: (400, 2000),
            SoundCategory.ANIMAL_BIRD: (2000, 8000),
            SoundCategory.NATURE_THUNDER: (20, 200),
            SoundCategory.NATURE_RAIN: (2000, 8000),
            SoundCategory.NATURE_WIND: (50, 500),
            SoundCategory.MACHINE_HUM: (50, 120),
            SoundCategory.MACHINE_ENGINE: (50, 500),
            SoundCategory.MACHINE_ALARM: (1000, 4000),
            SoundCategory.ENV_DOOR: (100, 500),
            SoundCategory.ENV_GLASS_BREAK: (2000, 10000),
        }
    
    def _load_ml_classifier(self):
        """Load ML sound classifier if available"""
        try:
            # Would load model from config path
            model_path = self.config.get('sound_classifier_model')
            if model_path:
                # self.ml_classifier = load_model(model_path)
                logger.info(f"ML sound classifier would be loaded from {model_path}")
            else:
                logger.info("No ML sound classifier configured")
        except Exception as e:
            logger.error(f"Failed to load ML classifier: {e}")
    
    def _init_feature_extractors(self) -> Dict:
        """Initialize feature extraction functions"""
        return {
            'mfcc': lambda y, sr: librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13),
            'spectral_centroid': lambda y, sr: librosa.feature.spectral_centroid(y=y, sr=sr),
            'spectral_rolloff': lambda y, sr: librosa.feature.spectral_rolloff(y=y, sr=sr),
            'zero_crossing_rate': lambda y, sr: librosa.feature.zero_crossing_rate(y),
            'rms': lambda y, sr: librosa.feature.rms(y=y),
        }
    
    def _extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract all audio features"""
        features = {}
        
        try:
            for name, extractor in self.feature_extractors.items():
                try:
                    result = extractor(audio, sr)
                    # Ensure result is at least 1D
                    if result.ndim == 0:
                        result = np.array([result])
                    features[name] = result
                except Exception as e:
                    logger.debug(f"Feature extraction failed for {name}: {e}")
                    features[name] = np.array([0.0])
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
        
        return features
    
    def _detect_events(self, 
                      audio: np.ndarray, 
                      sr: int,
                      features: Dict,
                      context: Optional[Dict]) -> List[Tuple[np.ndarray, int, int]]:
        """
        Detect distinct sound events in audio.
        Returns list of (audio_segment, start_sample, end_sample) for each event.
        """
        events = []
        
        try:
            # Use RMS energy to detect events
            hop_length = 512
            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            
            # Simple threshold-based segmentation
            threshold = np.mean(rms) * 1.5
            time_per_frame = hop_length / sr
            
            # Find regions above threshold
            is_event = rms > threshold
            
            # Smooth with moving average
            window = 5
            if len(is_event) > window:
                kernel = np.ones(window) / window
                is_event_smooth = np.convolve(is_event, kernel, mode='same') > 0.3
            else:
                is_event_smooth = is_event
            
            # Find contiguous regions
            labeled, num_features = label(is_event_smooth)
            
            for i in range(1, num_features + 1):
                region = np.where(labeled == i)[0]
                if len(region) > 5:  # Minimum event length (5 frames)
                    start_frame = region[0]
                    end_frame = region[-1]
                    
                    start_sample = start_frame * hop_length
                    end_sample = min(end_frame * hop_length + hop_length, len(audio))
                    
                    # Extract event audio
                    event_audio = audio[start_sample:end_sample]
                    
                    # Only include if long enough (> 50ms) and has sufficient energy
                    if len(event_audio) / sr > 0.05 and np.max(np.abs(event_audio)) > 0.01:
                        events.append((event_audio, start_sample, end_sample))
                        
        except Exception as e:
            logger.warning(f"Event detection failed: {e}")
        
        return events
    
    def _classify_event(self,
                       event_audio: np.ndarray,
                       sr: int,
                       context: Optional[Dict]) -> Optional[SoundEvent]:
        """
        Classify a single sound event.
        """
        # Extract event features
        features = self._extract_features(event_audio, sr)
        
        # Calculate basic characteristics
        duration = len(event_audio) / sr
        
        # Frequency analysis
        fft = np.fft.fft(event_audio)
        freqs = np.fft.fftfreq(len(event_audio), 1/sr)
        magnitudes = np.abs(fft)
        
        # Find dominant frequency
        positive_freqs = freqs[:len(freqs)//2]
        positive_mags = magnitudes[:len(magnitudes)//2]
        if len(positive_mags) > 0 and np.max(positive_mags) > 0:
            dominant_freq = positive_freqs[np.argmax(positive_mags)]
            
            # Find frequency range where magnitude > 50% of max
            threshold = np.max(positive_mags) * 0.5
            above_threshold = np.where(positive_mags > threshold)[0]
            if len(above_threshold) > 1:
                freq_range = (positive_freqs[above_threshold[0]], 
                            positive_freqs[above_threshold[-1]])
            else:
                freq_range = (dominant_freq * 0.9, dominant_freq * 1.1)
        else:
            dominant_freq = 0
            freq_range = (0, 0)
        
        # Intensity (dB)
        rms = np.sqrt(np.mean(event_audio**2))
        intensity = 20 * np.log10(rms + 1e-10)
        
        # Rhythm detection
        rhythm = None
        if duration > 1:
            try:
                tempo = librosa.beat.tempo(y=event_audio, sr=sr)[0]
                rhythm = float(tempo)
            except:
                pass
        
        # Generate candidates
        candidates = []
        
        # Template matching
        for category, templates in self.sound_templates.items():
            for template in templates:
                score = self._match_template({
                    'duration': duration,
                    'freq_range': freq_range,
                    'dominant_freq': dominant_freq,
                    'rhythm': rhythm,
                    'intensity': intensity
                }, template)
                
                if score > 0.4:  # Lower threshold for candidate inclusion
                    candidates.append((category, score))
        
        # ML classification if available
        if self.ml_classifier:
            try:
                # ml_result = self.ml_classifier.predict(features)
                # candidates.extend(ml_result)
                pass
            except:
                pass
        
        # Rank candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            primary_category, confidence = candidates[0]
            alternatives = candidates[1:4]
            
            # Estimate source type
            source_type = self._estimate_source_type(primary_category, features)
            
            return SoundEvent(
                category=primary_category,
                confidence=float(confidence),
                start_time=0,  # Will be set by caller
                end_time=duration,
                duration=float(duration),
                frequency_range=(float(freq_range[0]), float(freq_range[1])),
                intensity=float(intensity),
                pitch=float(dominant_freq),
                rhythm=rhythm,
                source_type=source_type,
                alternatives=[(c, float(s)) for c, s in alternatives],
                timestamp=datetime.now()
            )
        
        return None
    
    def _match_template(self, event_features: Dict, template: Dict) -> float:
        """Match event features against a template"""
        score = 0.0
        matches = 0
        
        # Check frequency range
        if 'freq_range' in template:
            t_min, t_max = template['freq_range']
            e_min, e_max = event_features.get('freq_range', (0, 0))
            
            # Calculate overlap
            overlap = max(0, min(t_max, e_max) - max(t_min, e_min))
            total_range = max(t_max - t_min, e_max - e_min, 1)  # Avoid division by zero
            
            freq_score = overlap / total_range
            score += freq_score
            matches += 1
        
        # Check duration
        if 'duration_range' in template:
            d_min, d_max = template['duration_range']
            duration = event_features.get('duration', 0)
            
            if d_min <= duration <= d_max:
                duration_score = 1.0
            else:
                # Penalize based on distance from range
                if duration < d_min:
                    duration_score = max(0, 1 - (d_min - duration) / d_min)
                else:
                    duration_score = max(0, 1 - (duration - d_max) / d_max)
            
            score += duration_score
            matches += 1
        
        # Check rhythm
        if 'rhythm_range' in template and event_features.get('rhythm'):
            r_min, r_max = template['rhythm_range']
            rhythm = event_features['rhythm']
            
            if r_min <= rhythm <= r_max:
                rhythm_score = 1.0
            else:
                # Distance-based score
                mid = (r_min + r_max) / 2
                range_size = max(r_max - r_min, 1)
                rhythm_score = max(0, 1 - abs(rhythm - mid) / (range_size * 2))
            
            score += rhythm_score
            matches += 1
        
        # Check intensity range
        if 'intensity_range' in template:
            i_min, i_max = template['intensity_range']
            intensity = event_features.get('intensity', -100)
            
            if i_min <= intensity <= i_max:
                intensity_score = 1.0
            else:
                intensity_score = max(0, 1 - abs(intensity - (i_min + i_max)/2) / 20)
            
            score += intensity_score
            matches += 1
        
        # Check continuous property
        if 'continuous' in template:
            # Would need longer-term analysis - placeholder
            matches += 1
            score += 0.5  # Neutral score
        
        if matches > 0:
            return score / matches
        return 0.0
    
    def _match_source_template(self, sound_event: SoundEvent, template: Dict) -> bool:
        """Check if sound event matches a known source template"""
        # Simple matching based on dominant frequency and duration
        freq_diff = abs(sound_event.pitch - template.get('dominant_freq', 0))
        duration_diff = abs(sound_event.duration - template.get('duration', 0))
        
        return freq_diff < 20 and duration_diff < 0.2
    
    def _estimate_source_type(self, category: SoundCategory, features: Dict) -> Optional[SoundSource]:
        """Estimate source type based on category and features"""
        # Indoor/outdoor estimation based on reverb (would need more analysis)
        if category in [SoundCategory.NATURE_RAIN, SoundCategory.NATURE_THUNDER, 
                       SoundCategory.NATURE_WIND, SoundCategory.ANIMAL_BIRD]:
            return SoundSource.OUTDOOR
        elif category in [SoundCategory.HUMAN_SPEECH, SoundCategory.MACHINE_HUM,
                         SoundCategory.ELECTRONIC_NOTIFICATION]:
            return SoundSource.INDOOR
        
        # Distance estimation from intensity (simplified)
        intensity = np.mean(features.get('rms', np.array([0])))
        if intensity > 0.1:
            return SoundSource.NEARBY
        elif intensity > 0.01:
            return SoundSource.DISTANT
        
        return SoundSource.PHYSICAL
    
    def _update_background_model(self, features: Dict):
        """Update background noise model"""
        if self.background_model is None:
            self.background_model = features.copy()
        else:
            # Moving average update for numeric features
            for key in features:
                if key in self.background_model:
                    try:
                        self.background_model[key] = (
                            (1 - self.background_update_rate) * self.background_model[key] +
                            self.background_update_rate * features[key]
                        )
                    except:
                        pass
    
    def _analyze_soundscape(self, 
                           events: List[SoundEvent], 
                           features: Dict,
                           context: Optional[Dict]) -> Soundscape:
        """Analyze overall soundscape characteristics"""
        soundscape = Soundscape(events=events)
        
        if not events:
            soundscape.background_sound = SoundCategory.SILENCE
            soundscape.is_quiet = True
            return soundscape
        
        # Count categories
        category_counts = defaultdict(int)
        for event in events:
            category_counts[event.category] += 1
        
        # Primary sounds
        total_events = len(events)
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        soundscape.primary_sounds = [(cat, count/total_events) for cat, count in sorted_cats[:5]]
        
        # Diversity
        soundscape.sound_diversity = len(category_counts)
        
        # Density
        total_duration = sum(e.duration for e in events)
        if total_duration > 0:
            soundscape.sound_density = len(events) / total_duration * 60  # per minute
        
        # Presence detection
        human_categories = [c for c in SoundCategory if c.value.startswith('human_')]
        soundscape.has_human_presence = any(e.category in human_categories for e in events)
        
        animal_categories = [c for c in SoundCategory if c.value.startswith('animal_')]
        soundscape.has_animal_presence = any(e.category in animal_categories for e in events)
        
        machine_categories = [c for c in SoundCategory if c.value.startswith('machine_')]
        soundscape.has_machinery = any(e.category in machine_categories for e in events)
        
        # Noise level
        avg_intensity = np.mean([e.intensity for e in events])
        soundscape.is_noisy = avg_intensity > -20  # dB threshold
        soundscape.is_quiet = avg_intensity < -40
        
        return soundscape
    
    def _apply_wednesday_analysis(self, soundscape: Soundscape, context: Optional[Dict]) -> Soundscape:
        """
        Apply Wednesday's special analysis to soundscape.
        She notices anomalies and potential threats.
        """
        # Detect anomalies (unexpected sounds in context)
        if context and 'expected_sounds' in context:
            expected = set(context['expected_sounds'])
            for event in soundscape.events:
                if event.category.value not in expected:
                    soundscape.anomalies.append(event)
        
        # Detect potential threats
        threat_categories = [
            SoundCategory.ENV_GLASS_BREAK,
            SoundCategory.ENV_EXPLOSION,
            SoundCategory.ENV_SIREN,
            SoundCategory.HUMAN_FOOTSTEPS,  # Unexpected footsteps
            SoundCategory.MACHINE_ALARM,
        ]
        
        for event in soundscape.events:
            if event.category in threat_categories:
                # Check context - footsteps at night are more threatening
                if event.category == SoundCategory.HUMAN_FOOTSTEPS:
                    if context and context.get('time_of_night', False):
                        event.confidence *= 1.2  # Boost confidence for threat
                        soundscape.threats.append(event)
                else:
                    soundscape.threats.append(event)
            
            # High intensity sounds are always notable
            if event.intensity > -10:  # Very loud
                soundscape.points_of_interest.append(event)
        
        return soundscape
    
    def _update_stats(self, events: List[SoundEvent], processing_time: float):
        """Update classification statistics"""
        self.stats['total_events'] += len(events)
        self.stats['classifications'] += 1
        
        # Update average confidence
        if events:
            avg_conf = float(np.mean([e.confidence for e in events]))
            total = self.stats['classifications']
            old_avg = self.stats['avg_confidence']
            self.stats['avg_confidence'] = old_avg + (avg_conf - old_avg) / total
    
    def get_stats(self) -> Dict:
        """Return classification statistics"""
        stats = dict(self.stats)
        stats['categories_found'] = dict(stats['categories_found'])
        return stats
    
    def reset_stats(self) -> None:
        """Reset classification statistics"""
        self.stats = {
            'total_events': 0,
            'classifications': 0,
            'avg_confidence': 0.0,
            'errors': 0,
            'categories_found': defaultdict(int)
        }


class PatternDetector:
    """
    Detects repeating patterns in sound events.
    """
    
    def __init__(self, min_pattern_length: int = 3):
        self.patterns = []
        self.pattern_length = min_pattern_length
        self.max_pattern_length = 10
    
    def detect_patterns(self, events: List[SoundEvent]) -> List[List[SoundCategory]]:
        """
        Detect repeating sequences of sound events.
        """
        if len(events) < self.pattern_length * 2:
            return []
        
        # Extract category sequence
        sequence = [e.category for e in events]
        
        # Find repeating subsequences
        patterns = []
        max_len = min(self.max_pattern_length, len(sequence) // 2)
        
        for length in range(self.pattern_length, max_len + 1):
            # Use a sliding window to find repeated patterns
            pattern_counts = defaultdict(int)
            pattern_positions = defaultdict(list)
            
            for i in range(len(sequence) - length + 1):
                pattern = tuple(sequence[i:i+length])
                pattern_counts[pattern] += 1
                pattern_positions[pattern].append(i)
            
            # Check for patterns that appear at least twice with separation
            for pattern, count in pattern_counts.items():
                if count >= 2:
                    # Check if occurrences are separated (not overlapping)
                    positions = pattern_positions[pattern]
                    if max(positions) - min(positions) >= length:
                        if list(pattern) not in patterns:
                            patterns.append(list(pattern))
        
        return patterns

# Connects to: speech_to_text.py (shares audio input)
# Connects to: tone_analysis.py (comprehensive audio analysis)
# Connects to: perception/attention/salience.py (unusual sounds attract attention)
# Connects to: memory/episodic/ (stores sound events in episodic memory)
# Connects to: emotion/appraisal.py (threatening sounds trigger emotional response)
# Connects to: executive/priorities.py (alarms and threats get priority)