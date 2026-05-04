"""
Analyzes vocal tone; pitch/pace analysis; emotional cues from voice.
Wednesday hears beyond words - the tremor, the hesitation, the forced calm.
Every vocal nuance reveals truth.
"""
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import time
import warnings

# Suppress librosa warnings if needed
warnings.filterwarnings('ignore', category=UserWarning)

# Audio processing libraries
try:
    import librosa
    import scipy.signal as signal
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    logging.warning(f"Audio analysis libraries not available: {e}. Install librosa, scipy")

logger = logging.getLogger(__name__)

class VocalQuality(Enum):
    """Qualities of vocal delivery"""
    BREATHY = "breathy"
    RESONANT = "resonant"
    NASAL = "nasal"
    HOARSE = "hoarse"
    CLEAR = "clear"
    TREMBLING = "trembling"
    STRAINED = "strained"
    RELAXED = "relaxed"
    MONOTONE = "monotone"
    VARIED = "varied"

class SpeechRate(Enum):
    """Rate of speech"""
    VERY_SLOW = "very_slow"  # < 100 wpm
    SLOW = "slow"            # 100-120 wpm
    MODERATE = "moderate"    # 120-150 wpm
    FAST = "fast"            # 150-180 wpm
    VERY_FAST = "very_fast"  # > 180 wpm
    RAPID_FIRE = "rapid_fire" # > 200 wpm (anxious/excited)

class VoicePitch(Enum):
    """Pitch characteristics"""
    VERY_LOW = "very_low"     # < 85 Hz (male) / < 165 Hz (female)
    LOW = "low"               # 85-120 Hz (male) / 165-200 Hz (female)
    MODERATE = "moderate"     # 120-180 Hz (male) / 200-300 Hz (female)
    HIGH = "high"             # 180-250 Hz (male) / 300-400 Hz (female)
    VERY_HIGH = "very_high"   # > 250 Hz (male) / > 400 Hz (female)

@dataclass
class ToneAnalysis:
    """Complete analysis of vocal tone and emotional cues"""
    # Pitch characteristics
    mean_pitch: float = 0.0  # Hz
    pitch_std: float = 0.0    # Pitch variability
    pitch_range: Tuple[float, float] = (0.0, 0.0)  # Min/max pitch
    pitch_contour: List[float] = field(default_factory=list)  # Pitch over time
    
    # Volume characteristics
    mean_volume: float = 0.0  # dB
    volume_std: float = 0.0    # Volume variability
    volume_range: Tuple[float, float] = (0.0, 0.0)
    volume_contour: List[float] = field(default_factory=list)
    
    # Temporal characteristics
    speech_rate: float = 0.0  # words per minute
    pause_frequency: float = 0.0  # pauses per minute
    avg_pause_duration: float = 0.0  # seconds
    hesitation_count: int = 0  # "um", "uh", pauses > 1s
    
    # Voice quality
    quality: List[Tuple[VocalQuality, float]] = field(default_factory=list)
    breathiness: float = 0.0  # 0-1
    tremor: float = 0.0       # 0-1 vocal tremor
    jitter: float = 0.0        # Frequency perturbation
    shimmer: float = 0.0       # Amplitude perturbation
    
    # Emotional indicators
    emotional_cues: Dict[str, float] = field(default_factory=dict)
    # Example: 'confidence', 'anxiety', 'anger', 'sadness', 'joy', 'sarcasm'
    
    # Stress indicators
    stress_level: float = 0.0  # 0-1
    arousal_level: float = 0.0  # 0-1 (calm to excited)
    tension: float = 0.0        # 0-1 vocal tension
    
    # Deception indicators (Wednesday's specialty)
    deception_indicators: Dict[str, float] = field(default_factory=dict)
    # Example: 'pitch_rises', 'hesitations', 'rate_changes'
    
    # Speaker identification
    speaker_id: Optional[str] = None
    speaker_confidence: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'pitch': {
                'mean': float(self.mean_pitch),
                'std': float(self.pitch_std),
                'range': [float(self.pitch_range[0]), float(self.pitch_range[1])]
            },
            'volume': {
                'mean': float(self.mean_volume),
                'std': float(self.volume_std),
                'range': [float(self.volume_range[0]), float(self.volume_range[1])]
            },
            'speech_rate': float(self.speech_rate),
            'stress_level': float(self.stress_level),
            'arousal_level': float(self.arousal_level),
            'emotional_cues': {k: float(v) for k, v in self.emotional_cues.items()},
            'deception_indicators': {k: float(v) for k, v in self.deception_indicators.items()},
            'quality': [(q.value, float(s)) for q, s in self.quality],
            'timestamp': self.timestamp.isoformat(),
            'duration': float(self.duration)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ToneAnalysis':
        """Create ToneAnalysis from dictionary"""
        pitch_data = data.get('pitch', {})
        volume_data = data.get('volume', {})
        
        return cls(
            mean_pitch=pitch_data.get('mean', 0.0),
            pitch_std=pitch_data.get('std', 0.0),
            pitch_range=tuple(pitch_data.get('range', [0.0, 0.0])),
            mean_volume=volume_data.get('mean', 0.0),
            volume_std=volume_data.get('std', 0.0),
            volume_range=tuple(volume_data.get('range', [0.0, 0.0])),
            speech_rate=data.get('speech_rate', 0.0),
            stress_level=data.get('stress_level', 0.0),
            arousal_level=data.get('arousal_level', 0.0),
            emotional_cues=data.get('emotional_cues', {}),
            deception_indicators=data.get('deception_indicators', {}),
            quality=[(VocalQuality(q), s) for q, s in data.get('quality', [])],
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            duration=data.get('duration', 0.0)
        )

class ToneAnalyzer:
    """
    Analyzes vocal tone; pitch/pace analysis; emotional cues from voice.
    Wednesday listens to how things are said, not just what is said.
    A pause too long, a pitch too high - she notices.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Analysis parameters
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.frame_length = self.config.get('frame_length', 2048)
        self.hop_length = self.config.get('hop_length', 512)
        
        # Pitch ranges by gender (Hz)
        self.pitch_ranges = {
            'male': {'very_low': 0, 'low': 85, 'moderate': 120, 'high': 180, 'very_high': 250},
            'female': {'very_low': 0, 'low': 165, 'moderate': 200, 'high': 300, 'very_high': 400}
        }
        
        # Speech rate thresholds (words per minute)
        self.rate_thresholds = {
            'very_slow': 100,
            'slow': 120,
            'moderate': 150,
            'fast': 180,
            'very_fast': 200
        }
        
        # Emotional cue models (simplified rules)
        self.emotional_patterns = self._load_emotional_patterns()
        
        # Deception indicators (based on research)
        self.deception_patterns = self._load_deception_patterns()
        
        # Speaker recognition
        self.speaker_profiles = {}
        self.voice_prints = {}
        
        # Performance tracking
        self.stats = {
            'total_analyses': 0,
            'avg_processing_time': 0.0,
            'errors': 0
        }
        
        logger.info("ToneAnalyzer initialized")
    
    def analyze(self, 
               audio_data: Union[np.ndarray, bytes],
               sample_rate: Optional[int] = None,
               transcription: Optional[Any] = None,
               context: Optional[Dict] = None) -> ToneAnalysis:
        """
        Analyze vocal tone from audio data.
        
        Args:
            audio_data: Raw audio samples (numpy array or bytes)
            sample_rate: Sample rate of audio
            transcription: Optional transcription result for word timing
            context: Optional context (speaker info, etc.)
            
        Returns:
            ToneAnalysis with comprehensive vocal analysis
        """
        import time
        start_time = time.time()
        
        if not AUDIO_AVAILABLE:
            logger.error("Audio analysis libraries not available")
            return self._create_empty_analysis()
        
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        try:
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_data, np.ndarray) and audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            
            # Ensure audio is not empty
            if len(audio_data) == 0:
                logger.warning("Empty audio data")
                return self._create_empty_analysis()
            
            # Extract features
            pitch_features = self._analyze_pitch(audio_data, sample_rate)
            volume_features = self._analyze_volume(audio_data)
            temporal_features = self._analyze_temporal(audio_data, sample_rate, transcription)
            quality_features = self._analyze_voice_quality(audio_data, sample_rate)
            
            # Detect emotional cues
            emotional_cues = self._detect_emotional_cues(
                pitch_features, volume_features, temporal_features, quality_features
            )
            
            # Detect deception indicators
            deception_indicators = self._detect_deception(
                pitch_features, volume_features, temporal_features, quality_features,
                context
            )
            
            # Calculate stress and arousal
            stress_level = self._calculate_stress(
                pitch_features, volume_features, temporal_features, quality_features
            )
            arousal_level = self._calculate_arousal(
                pitch_features, volume_features, temporal_features
            )
            
            # Attempt speaker identification
            speaker_id, speaker_confidence = self._identify_speaker(
                audio_data, sample_rate, context
            )
            
            # Create analysis result
            analysis = ToneAnalysis(
                mean_pitch=pitch_features.get('mean_pitch', 0.0),
                pitch_std=pitch_features.get('pitch_std', 0.0),
                pitch_range=pitch_features.get('pitch_range', (0.0, 0.0)),
                pitch_contour=pitch_features.get('pitch_contour', []),
                
                mean_volume=volume_features.get('mean_volume', 0.0),
                volume_std=volume_features.get('volume_std', 0.0),
                volume_range=volume_features.get('volume_range', (0.0, 0.0)),
                volume_contour=volume_features.get('volume_contour', []),
                
                speech_rate=temporal_features.get('speech_rate', 0.0),
                pause_frequency=temporal_features.get('pause_frequency', 0.0),
                avg_pause_duration=temporal_features.get('avg_pause_duration', 0.0),
                hesitation_count=temporal_features.get('hesitation_count', 0),
                
                quality=quality_features.get('quality', []),
                breathiness=quality_features.get('breathiness', 0.0),
                tremor=quality_features.get('tremor', 0.0),
                jitter=quality_features.get('jitter', 0.0),
                shimmer=quality_features.get('shimmer', 0.0),
                
                emotional_cues=emotional_cues,
                stress_level=stress_level,
                arousal_level=arousal_level,
                tension=quality_features.get('tension', 0.0),
                
                deception_indicators=deception_indicators,
                
                speaker_id=speaker_id,
                speaker_confidence=speaker_confidence,
                
                duration=len(audio_data) / sample_rate,
                processing_time=time.time() - start_time
            )
            
            # Update stats
            self._update_stats(analysis.processing_time)
            
            # Store voice print for speaker recognition
            if speaker_id and speaker_confidence > 0.7:
                self._update_voice_print(speaker_id, pitch_features, quality_features)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in tone analysis: {e}", exc_info=True)
            self.stats['errors'] += 1
            return self._create_empty_analysis()
    
    def analyze_streaming(self, 
                         audio_chunk: np.ndarray,
                         sample_rate: int,
                         previous_analysis: Optional[ToneAnalysis] = None) -> ToneAnalysis:
        """
        Analyze streaming audio incrementally.
        
        Args:
            audio_chunk: New chunk of audio
            sample_rate: Sample rate
            previous_analysis: Previous analysis for continuity
            
        Returns:
            Updated tone analysis
        """
        # This would implement streaming analysis
        # For now, just do full analysis on accumulated buffer
        # In production, you'd maintain state and update incrementally
        return self.analyze(audio_chunk, sample_rate)
    
    def get_emotional_state(self, analysis: ToneAnalysis) -> Dict[str, float]:
        """
        Extract primary emotional state from tone analysis.
        """
        # Combine emotional cues with vocal features
        emotional_state = {}
        
        # Map emotional cues to basic emotions
        emotion_mapping = {
            'anger': ['aggression', 'frustration', 'irritation'],
            'sadness': ['sorrow', 'grief', 'melancholy'],
            'fear': ['anxiety', 'worry', 'nervousness'],
            'joy': ['happiness', 'excitement', 'pleasure'],
            'disgust': ['contempt', 'aversion', 'revulsion'],
            'surprise': ['shock', 'astonishment', 'amazement']
        }
        
        for emotion, cues in emotion_mapping.items():
            score = 0.0
            count = 0
            for cue in cues:
                if cue in analysis.emotional_cues:
                    score += analysis.emotional_cues[cue]
                    count += 1
            if count > 0:
                emotional_state[emotion] = score / count
        
        # Add stress and arousal as meta-emotions
        emotional_state['stress'] = analysis.stress_level
        emotional_state['arousal'] = analysis.arousal_level
        emotional_state['tension'] = analysis.tension
        
        return emotional_state
    
    def compare_to_baseline(self, 
                           analysis: ToneAnalysis, 
                           speaker_id: str) -> Dict[str, float]:
        """
        Compare current tone to speaker's baseline.
        Useful for detecting changes in emotional state.
        """
        if speaker_id not in self.speaker_profiles:
            return {}
        
        baseline = self.speaker_profiles[speaker_id]
        deviations = {}
        
        # Compare key metrics
        metrics = ['mean_pitch', 'pitch_std', 'speech_rate', 'stress_level']
        for metric in metrics:
            current = getattr(analysis, metric, 0.0)
            base = baseline.get(metric, current)
            if base != 0:
                deviations[f'{metric}_deviation'] = (current - base) / base
            else:
                deviations[f'{metric}_deviation'] = 0.0
        
        return deviations
    
    def _analyze_pitch(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract pitch features from audio.
        """
        features = {
            'pitch_contour': [],
            'mean_pitch': 0.0,
            'pitch_std': 0.0,
            'pitch_range': (0.0, 0.0),
            'pitch_stability': 0.0
        }
        
        try:
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(
                y=audio, 
                sr=sr,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            
            # Get pitch contour (take max magnitude pitch per frame)
            pitch_contour = []
            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch = pitches[index, i]
                if pitch > 0:  # Only include detected pitches
                    pitch_contour.append(float(pitch))
            
            if pitch_contour:
                features['pitch_contour'] = pitch_contour
                features['mean_pitch'] = float(np.mean(pitch_contour))
                features['pitch_std'] = float(np.std(pitch_contour))
                features['pitch_range'] = (float(min(pitch_contour)), float(max(pitch_contour)))
                
                # Calculate pitch stability
                if features['mean_pitch'] > 0:
                    features['pitch_stability'] = 1.0 - min(1.0, features['pitch_std'] / features['mean_pitch'])
                
        except Exception as e:
            logger.warning(f"Pitch analysis failed: {e}")
        
        return features
    
    def _analyze_volume(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract volume/amplitude features.
        """
        features = {
            'volume_contour': [],
            'mean_volume': 0.0,
            'volume_std': 0.0,
            'volume_range': (0.0, 0.0),
            'volume_variability': 0.0
        }
        
        try:
            # Calculate RMS energy
            frame_length = min(self.frame_length, len(audio))
            hop_length = min(self.hop_length, len(audio) // 4)
            
            # Split into frames
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            
            # Calculate RMS for each frame
            rms = np.sqrt(np.mean(frames**2, axis=0))
            
            if len(rms) > 0 and np.max(rms) > 0:
                # Convert to dB
                rms_db = librosa.amplitude_to_db(rms, ref=np.max)
                
                features['volume_contour'] = [float(v) for v in rms_db.tolist()]
                features['mean_volume'] = float(np.mean(rms_db))
                features['volume_std'] = float(np.std(rms_db))
                features['volume_range'] = (float(np.min(rms_db)), float(np.max(rms_db)))
                
                # Calculate volume dynamics
                if abs(features['mean_volume']) > 1e-6:
                    features['volume_variability'] = features['volume_std'] / abs(features['mean_volume'])
                
        except Exception as e:
            logger.warning(f"Volume analysis failed: {e}")
        
        return features
    
    def _analyze_temporal(self, 
                         audio: np.ndarray, 
                         sr: int,
                         transcription: Optional[Any]) -> Dict[str, Any]:
        """
        Analyze temporal features (speech rate, pauses, hesitations).
        """
        features = {
            'speech_rate': 0.0,
            'pause_frequency': 0.0,
            'avg_pause_duration': 0.0,
            'hesitation_count': 0
        }
        
        try:
            # Detect speech/silence
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            # Calculate energy
            energy = librosa.feature.rms(
                y=audio, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            # Simple threshold for silence
            threshold = np.mean(energy) * 0.1
            is_speech = energy > threshold
            
            # Find speech segments
            changes = np.diff(np.concatenate(([0], is_speech.astype(int), [0])))
            speech_starts = np.where(changes == 1)[0]
            speech_ends = np.where(changes == -1)[0]
            
            # Calculate speech duration
            speech_frames = len(speech_starts)
            if speech_frames > 0:
                speech_duration = sum(speech_ends - speech_starts) * hop_length / sr
                total_duration = len(audio) / sr
                
                # Speech rate (if transcription available)
                if transcription and hasattr(transcription, 'text') and transcription.text:
                    word_count = len(transcription.text.split())
                    if speech_duration > 0:
                        features['speech_rate'] = (word_count / speech_duration) * 60
                
                # Pause analysis
                pause_durations = []
                for i in range(len(speech_ends) - 1):
                    pause_duration = (speech_starts[i+1] - speech_ends[i]) * hop_length / sr
                    if pause_duration > 0.1:  # Only count pauses > 100ms
                        pause_durations.append(pause_duration)
                
                if pause_durations:
                    features['pause_frequency'] = len(pause_durations) / total_duration * 60
                    features['avg_pause_duration'] = float(np.mean(pause_durations))
                    
                    # Count hesitations (pauses > 1 second)
                    features['hesitation_count'] = sum(1 for p in pause_durations if p > 1.0)
                    
        except Exception as e:
            logger.warning(f"Temporal analysis failed: {e}")
        
        return features
    
    def _analyze_voice_quality(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Analyze voice quality features (breathiness, tremor, jitter, shimmer).
        """
        features = {
            'quality': [],
            'breathiness': 0.0,
            'tremor': 0.0,
            'jitter': 0.0,
            'shimmer': 0.0,
            'tension': 0.0
        }
        
        try:
            # Extract harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # Breathiness: ratio of aperiodic energy
            if len(audio) > 0:
                total_energy = np.sum(audio**2)
                if total_energy > 1e-6:
                    harmonic_energy = np.sum(harmonic**2)
                    features['breathiness'] = 1.0 - (harmonic_energy / total_energy)
            
            # Tremor: low-frequency modulation
            envelope = np.abs(librosa.stft(audio))
            modulation = np.mean(envelope, axis=0)
            if len(modulation) > 1:
                # Look for modulations in 4-8 Hz range (tremor frequency)
                modulation_fft = np.abs(np.fft.fft(modulation))
                freqs = np.fft.fftfreq(len(modulation), d=1.0)
                
                # Map to Hz scale
                freqs_hz = freqs * sr / (2 * len(modulation))
                
                tremor_range = (np.abs(freqs_hz) >= 4) & (np.abs(freqs_hz) <= 8)
                if np.any(tremor_range):
                    tremor_energy = np.sum(modulation_fft[tremor_range])
                    total_energy = np.sum(modulation_fft)
                    if total_energy > 1e-6:
                        features['tremor'] = tremor_energy / total_energy
            
            # Tension: based on spectral characteristics
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            if len(spectral_centroids) > 0:
                # Higher spectral centroid often indicates tension
                features['tension'] = float(np.mean(spectral_centroids) / (sr/2))
            
            # Classify vocal quality
            qualities = []
            
            if features['breathiness'] > 0.3:
                qualities.append((VocalQuality.BREATHY, features['breathiness']))
            
            if features['tremor'] > 0.2:
                qualities.append((VocalQuality.TREMBLING, features['tremor']))
            
            if features['tension'] > 0.7:
                qualities.append((VocalQuality.STRAINED, features['tension']))
            elif features['tension'] < 0.3:
                qualities.append((VocalQuality.RELAXED, 1.0 - features['tension']))
            
            # Check for monotone (low pitch variability)
            if features.get('pitch_std', 0) < 10:  # This would need pitch info
                qualities.append((VocalQuality.MONOTONE, 0.6))
            
            features['quality'] = qualities
            
        except Exception as e:
            logger.warning(f"Voice quality analysis failed: {e}")
        
        return features
    
    def _load_emotional_patterns(self) -> Dict[str, Dict]:
        """
        Load patterns linking vocal features to emotions.
        """
        return {
            'anger': {
                'pitch_range': 0.8,  # Wide pitch range
                'volume_mean': 0.9,   # High volume
                'speech_rate': 0.7,    # Fast speech
                'tension': 0.8,        # High tension
            },
            'sadness': {
                'pitch_mean': 0.3,     # Low pitch
                'pitch_range': 0.2,     # Narrow range
                'volume_mean': 0.3,     # Low volume
                'speech_rate': 0.3,     # Slow speech
                'tension': 0.2,         # Low tension
                'breathiness': 0.7      # High breathiness
            },
            'fear': {
                'pitch_mean': 0.7,      # High pitch
                'pitch_std': 0.8,       # High variability
                'speech_rate': 0.8,      # Fast speech
                'tremor': 0.7,           # High tremor
                'hesitation_count': 0.6   # Many hesitations
            },
            'joy': {
                'pitch_range': 0.7,      # Wide range
                'volume_variability': 0.7, # Dynamic volume
                'speech_rate': 0.6,       # Moderate-fast
                'tension': 0.3,           # Low tension
                'pitch_stability': 0.4    # Less stable pitch
            },
            'sarcasm': {
                'pitch_std': 0.7,         # High pitch variability
                'speech_rate': 0.6,        # Variable rate
                'pause_frequency': 0.7     # Unusual pauses
            }
        }
    
    def _load_deception_patterns(self) -> Dict[str, Dict]:
        """
        Load patterns indicating possible deception.
        Based on research in forensic linguistics.
        """
        return {
            'pitch_rises': {
                'description': 'Pitch rises at end of statements',
                'weight': 0.6
            },
            'hesitations': {
                'description': 'Increased hesitations and filled pauses',
                'weight': 0.7
            },
            'speech_rate_changes': {
                'description': 'Sudden changes in speech rate',
                'weight': 0.5
            },
            'pitch_stability': {
                'description': 'Unusually stable pitch (rehearsed)',
                'weight': 0.4
            },
            'tremor_increase': {
                'description': 'Increased vocal tremor',
                'weight': 0.8
            },
            'volume_drops': {
                'description': 'Volume drops on key words',
                'weight': 0.6
            }
        }
    
    def _detect_emotional_cues(self,
                              pitch: Dict,
                              volume: Dict,
                              temporal: Dict,
                              quality: Dict) -> Dict[str, float]:
        """
        Detect emotional cues from vocal features.
        """
        cues = {}
        
        # Normalize features to 0-1 range for comparison
        norm_pitch = min(1.0, pitch.get('mean_pitch', 0) / 400)
        norm_volume = min(1.0, (volume.get('mean_volume', -60) + 60) / 60)
        norm_rate = min(1.0, temporal.get('speech_rate', 0) / 200)
        norm_tension = quality.get('tension', 0.5)
        norm_breathiness = quality.get('breathiness', 0.3)
        norm_tremor = quality.get('tremor', 0.1)
        
        # Check each emotional pattern
        for emotion, patterns in self.emotional_patterns.items():
            score = 0.0
            matches = 0
            
            for feature, threshold in patterns.items():
                feature_value = None
                
                if feature == 'pitch_mean':
                    feature_value = norm_pitch
                elif feature == 'pitch_range':
                    pitch_range_val = pitch.get('pitch_range', (0, 0))[1] - pitch.get('pitch_range', (0, 0))[0]
                    feature_value = min(1.0, pitch_range_val / 200)
                elif feature == 'pitch_std':
                    feature_value = min(1.0, pitch.get('pitch_std', 0) / 50)
                elif feature == 'pitch_stability':
                    feature_value = pitch.get('pitch_stability', 0.5)
                elif feature == 'volume_mean':
                    feature_value = norm_volume
                elif feature == 'volume_variability':
                    feature_value = volume.get('volume_variability', 0.3)
                elif feature == 'speech_rate':
                    feature_value = norm_rate
                elif feature == 'tension':
                    feature_value = norm_tension
                elif feature == 'breathiness':
                    feature_value = norm_breathiness
                elif feature == 'tremor':
                    feature_value = norm_tremor
                elif feature == 'hesitation_count':
                    feature_value = min(1.0, temporal.get('hesitation_count', 0) / 10)
                elif feature == 'pause_frequency':
                    feature_value = min(1.0, temporal.get('pause_frequency', 0) / 20)
                
                if feature_value is not None:
                    if abs(feature_value - threshold) < 0.3:  # Within threshold range
                        score += threshold
                        matches += 1
            
            if matches > 0:
                cues[emotion] = score / matches
        
        return cues
    
    def _detect_deception(self,
                         pitch: Dict,
                         volume: Dict,
                         temporal: Dict,
                         quality: Dict,
                         context: Optional[Dict]) -> Dict[str, float]:
        """
        Detect indicators of possible deception.
        Wednesday is particularly good at this.
        """
        indicators = {}
        
        # Check pitch rises at ends (would need phrase boundaries)
        if pitch.get('pitch_contour'):
            contour = pitch['pitch_contour']
            if len(contour) > 10:
                # Look for rising patterns at phrase ends
                # Simplified: check if last few frames are rising
                last_frames = contour[-5:]
                if len(last_frames) > 1 and last_frames[-1] > last_frames[0]:
                    indicators['pitch_rises'] = 0.6
        
        # Check hesitation patterns
        if temporal.get('hesitation_count', 0) > 3:
            indicators['hesitations'] = min(1.0, temporal['hesitation_count'] / 10)
        
        # Check speech rate changes
        if temporal.get('speech_rate', 0) > 150:
            # Fast speech might indicate rehearsed content
            indicators['speech_rate_changes'] = 0.4
        
        # Check tremor
        if quality.get('tremor', 0) > 0.3:
            indicators['tremor_increase'] = min(1.0, quality['tremor'] * 1.5)
        
        # Check volume patterns
        if volume.get('volume_std', 0) < 5:  # Very stable volume
            indicators['volume_drops'] = 0.5
        
        # Check pitch stability (unusually stable might indicate rehearsed)
        if pitch.get('pitch_stability', 0) > 0.9:
            indicators['pitch_stability'] = 0.4
        
        # Context-based adjustments
        if context:
            if context.get('high_stakes', False):
                # In high-stakes situations, these indicators are more significant
                for key in indicators:
                    indicators[key] = min(1.0, indicators[key] * 1.2)
        
        return indicators
    
    def _calculate_stress(self,
                         pitch: Dict,
                         volume: Dict,
                         temporal: Dict,
                         quality: Dict) -> float:
        """
        Calculate overall stress level from vocal features.
        """
        stress_factors = []
        
        # High pitch variability often indicates stress
        if pitch.get('pitch_std', 0) > 30:
            stress_factors.append(min(1.0, pitch['pitch_std'] / 100))
        
        # Fast speech rate
        if temporal.get('speech_rate', 0) > 160:
            stress_factors.append(min(1.0, (temporal['speech_rate'] - 160) / 40))
        
        # Vocal tremor
        if quality.get('tremor', 0) > 0.2:
            stress_factors.append(min(1.0, quality['tremor'] * 2))
        
        # High tension
        if quality.get('tension', 0) > 0.6:
            stress_factors.append(quality['tension'])
        
        if stress_factors:
            return float(np.mean(stress_factors))
        return 0.0
    
    def _calculate_arousal(self,
                          pitch: Dict,
                          volume: Dict,
                          temporal: Dict) -> float:
        """
        Calculate arousal level (calm to excited).
        """
        arousal_factors = []
        
        # Pitch contributes to arousal
        norm_pitch = min(1.0, pitch.get('mean_pitch', 0) / 300)
        arousal_factors.append(norm_pitch)
        
        # Volume contributes
        norm_volume = min(1.0, (volume.get('mean_volume', -60) + 60) / 60)
        arousal_factors.append(norm_volume)
        
        # Speech rate contributes
        norm_rate = min(1.0, temporal.get('speech_rate', 0) / 200)
        arousal_factors.append(norm_rate)
        
        # Variability contributes
        arousal_factors.append(min(1.0, pitch.get('pitch_std', 0) / 50))
        arousal_factors.append(min(1.0, volume.get('volume_std', 0) / 20))
        
        if arousal_factors:
            return float(np.mean(arousal_factors))
        return 0.0
    
    def _identify_speaker(self,
                         audio: np.ndarray,
                         sr: int,
                         context: Optional[Dict]) -> Tuple[Optional[str], float]:
        """
        Attempt to identify speaker from voice characteristics.
        """
        # This would use voice recognition models
        # Simplified implementation
        
        try:
            # Extract voice print features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            voice_print = np.mean(mfccs, axis=1)
            
            # Compare with stored profiles
            best_match = None
            best_score = 0.0
            
            for speaker_id, profile in self.voice_prints.items():
                similarity = np.corrcoef(voice_print, profile)[0, 1]
                if similarity > best_score and similarity > 0.7:
                    best_score = float(similarity)
                    best_match = speaker_id
            
            # Check context for speaker hint
            if context and context.get('expected_speaker') and best_score < 0.5:
                # Low confidence, but context suggests a speaker
                return context['expected_speaker'], 0.3
            
            return best_match, best_score
            
        except Exception as e:
            logger.warning(f"Speaker identification failed: {e}")
            return None, 0.0
    
    def _update_voice_print(self, speaker_id: str, pitch: Dict, quality: Dict):
        """
        Update voice print for speaker.
        """
        # Simplified - would maintain running average
        self.speaker_profiles[speaker_id] = {
            'mean_pitch': pitch.get('mean_pitch', 0),
            'pitch_std': pitch.get('pitch_std', 0),
            'speech_rate': 0,  # Would need temporal info
            'stress_level': 0
        }
    
    def _create_empty_analysis(self) -> ToneAnalysis:
        """Create empty analysis for error cases"""
        return ToneAnalysis()
    
    def _update_stats(self, processing_time: float):
        """Update analysis statistics"""
        self.stats['total_analyses'] += 1
        total = self.stats['total_analyses']
        old_avg = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = old_avg + (processing_time - old_avg) / total
    
    def get_stats(self) -> Dict:
        """Return analysis statistics"""
        return dict(self.stats)
    
    def reset_stats(self) -> None:
        """Reset analysis statistics"""
        self.stats = {
            'total_analyses': 0,
            'avg_processing_time': 0.0,
            'errors': 0
        }

# Connects to: speech_to_text.py (provides audio for tone analysis)
# Connects to: perception/attention/salience.py (tone influences importance)
# Connects to: emotion/appraisal.py (provides emotional cues from voice)
# Connects to: self/theory_of_mind.py (helps understand others' mental states)
# Connects to: memory/working/ (stores tone analysis for context)
# Connects to: cognition/reasoning.py (tone affects interpretation)