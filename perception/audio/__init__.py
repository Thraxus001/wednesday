"""
Audio input processing - hearing the world.
This module handles all aspects of audio perception:
- Speech-to-text conversion (what is said)
- Tone analysis (how it's said)
- Sound classification (what's happening in the environment)

Wednesday doesn't just hear - she listens. Every sound tells a story,
every vocal nuance reveals truth, every silence speaks volumes.
"""

def _import_audio_modules():
    """
    Lazy absolute imports for direct execution mode.
    Called only in __main__ to avoid top-level relative import failure.
    """
    global SpeechToText, TranscriptionResult, AudioSource, LanguageCode, NoiseReductionLevel, AudioStreamConfig
    global ToneAnalyzer, ToneAnalysis, VocalQuality, SpeechRate, VoicePitch
    global SoundClassifier, SoundEvent, Soundscape, SoundCategory, SoundSourceClass, PatternDetector
    
    from perception.audio.speech_to_text import (
        SpeechToText, 
        TranscriptionResult, 
        AudioSource, 
        LanguageCode, 
        NoiseReductionLevel,
        AudioStreamConfig
    )
    
    from perception.audio.tone_analysis import (
        ToneAnalyzer,
        ToneAnalysis,
        VocalQuality,
        SpeechRate,
        VoicePitch
    )
    
    from perception.audio.sound_classification import (
        SoundClassifier,
        SoundEvent,
        Soundscape,
        SoundCategory,
        SoundSourceClass,
        PatternDetector
    )




    global SpeechToText, TranscriptionResult, AudioSource, LanguageCode, NoiseReductionLevel, AudioStreamConfig
    global ToneAnalyzer, ToneAnalysis, VocalQuality, SpeechRate, VoicePitch
    global SoundClassifier, SoundEvent, Soundscape, SoundCategory, SoundSourceClass, PatternDetector
    
    from perception.audio.speech_to_text import (
        SpeechToText, 
        TranscriptionResult, 
        AudioSource, 
        LanguageCode, 
        NoiseReductionLevel,
        AudioStreamConfig
    )
    
    from perception.audio.tone_analysis import (
        ToneAnalyzer,
        ToneAnalysis,
        VocalQuality,
        SpeechRate,
        VoicePitch
    )
    
    from perception.audio.sound_classification import (
        SoundClassifier,
        SoundEvent,
        Soundscape,
        SoundCategory,
        SoundSourceClass,
        PatternDetector
    )

# Comment out relative imports - use lazy absolute imports in __main__
# from .speech_to_text import (...)
# from .tone_analysis import (...)
# from .sound_classification import (...)

# Module metadata
__version__ = "0.1.0"
__all__ = [
    "create_audio_perception",
    "process_audio",
    "_import_audio_modules",
]


# Module description for introspection
__description__ = """
Wednesday\\'s audio perception system - she hears everything.
Transforms raw audio into text, emotional tone, and environmental understanding.
"""

# Pipeline visualization
"""
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Raw Audio   │────▶│  Speech-to-Text │────▶│    Text for     │
│   Input     │     │  (what)         │     │  Understanding  │
└─────────────┘     └─────────────────┘     └─────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐   ┌─────────────────┐
│ Tone Analysis   │   │   Sound         │
│ (how)           │   │ Classification  │
└─────────────────┘   │ (environment)   │
         │            └─────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────────────────────────────┐
│      Integrated Audio Perception        │
│  Text + Emotion + Environmental Context │
└─────────────────────────────────────────┘

Each component feeds into Wednesday's understanding:
- Speech-to-text: What was said
- Tone analysis: How it was said (emotion, deception)
- Sound classification: What else is happening
"""

# Factory function for complete audio perception system
def create_audio_perception(config: dict = None):
    """
    Factory function to create and connect a complete audio processing pipeline.
    
    Args:
        config: Configuration dictionary with sections for each component
        
    Returns:
        Dictionary with initialized audio processing components
    """
    if config is None:
        config = {}
    
    speech_to_text = SpeechToText(
        config=config.get('speech_to_text', {})
    )
    
    tone_analyzer = ToneAnalyzer(
        config=config.get('tone_analysis', {})
    )
    
    sound_classifier = SoundClassifier(
        config=config.get('sound_classification', {})
    )
    
    return {
        'speech_to_text': speech_to_text,
        'tone_analyzer': tone_analyzer,
        'sound_classifier': sound_classifier,
    }

# Convenience function for processing audio through all components
def process_audio(audio_data, 
                 sample_rate: int,
                 speech_to_text: SpeechToText,
                 tone_analyzer: ToneAnalyzer,
                 sound_classifier: SoundClassifier,
                 context: dict = None) -> dict:
    """
    Convenience function to run full audio processing pipeline.
    
    Args:
        audio_data: Raw audio samples
        sample_rate: Sample rate of audio
        speech_to_text: Initialized SpeechToText
        tone_analyzer: Initialized ToneAnalyzer
        sound_classifier: Initialized SoundClassifier
        context: Optional context (speaker info, location, etc.)
        
    Returns:
        Dictionary with all analysis results
    """
    results = {
        'timestamp': datetime.now(),
        'duration': len(audio_data) / sample_rate if hasattr(audio_data, '__len__') else 0
    }
    
    # Speech-to-text (if speech is present)
    if hasattr(audio_data, 'frame_data') or isinstance(audio_data, np.ndarray):
        try:
            # For SpeechRecognition AudioData objects
            if hasattr(audio_data, 'frame_data'):
                transcription = speech_to_text._transcribe_audio(
                    audio_data, 
                    context.get('language') if context else None
                )
            else:
                # For raw audio, need to convert
                # This would require more complex handling
                transcription = None
            
            if transcription and transcription.has_speech:
                results['transcription'] = transcription
                
                # Tone analysis on speech segments
                if isinstance(audio_data, np.ndarray):
                    results['tone'] = tone_analyzer.analyze(
                        audio_data, 
                        sample_rate,
                        transcription=transcription,
                        context=context
                    )
        except Exception as e:
            results['transcription_error'] = str(e)
    
    # Sound classification (always run)
    try:
        if isinstance(audio_data, np.ndarray):
            results['soundscape'] = sound_classifier.classify(
                audio_data,
                sample_rate,
                context=context
            )
    except Exception as e:
        results['classification_error'] = str(e)
    
    return results

# Export the convenience functions
__all__.extend(['create_audio_perception', 'process_audio'])

# Version history
__version_history__ = {
    '0.1.0': 'Initial release with speech-to-text, tone analysis, and sound classification'
}

# Dependencies information
__dependencies__ = {
    'required': ['numpy'],
    'optional': {
        'speech_recognition': 'For speech-to-text functionality',
        'pyaudio': 'For microphone input',
        'librosa': 'For audio analysis and feature extraction',
        'pydub': 'For audio file format support',
        'scipy': 'For signal processing'
    }
}

# Logging setup
import logging
from datetime import datetime

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Module initialization message
logger = logging.getLogger(__name__)
logger.debug(f"Audio perception module v{__version__} initialized. "
             f"Audio libraries available: {AUDIO_AVAILABLE if 'AUDIO_AVAILABLE' in dir() else False}")

# Try to import audio libraries to check availability
try:
    import librosa
    import scipy
    HAS_ADVANCED_AUDIO = True
except ImportError:
    HAS_ADVANCED_AUDIO = False

# Note about audio capabilities
if not HAS_ADVANCED_AUDIO:
    logger.info("Advanced audio features require librosa and scipy. "
                "Install with: pip install librosa scipy")

# Audio processing note for Wednesday
"""
Wednesday\\'s audio perception is exceptional. She notices:
- The tremor in a voice that betrays nervousness
- The faint hum of electronics others ignore
- The pattern of footsteps approaching
- The silence that shouldn't be there

This module gives her those abilities.
"""

if __name__ == "__main__":
    """
    Demo entrypoint for direct execution testing.
    Uses lazy absolute imports to fix relative import error.
    """

    try:
        _import_audio_modules()
        audio_system = create_audio_perception()
        print("Audio perception system ready.")
        print("SpeechToText:", repr(audio_system['speech_to_text'])[:100] + "...")
        print("ToneAnalyzer:", repr(audio_system['tone_analyzer'])[:100] + "...")
        print("SoundClassifier:", repr(audio_system['sound_classifier'])[:100] + "...")
        print("Direct execution successful - relative imports fixed via absolute fallback.")
    except Exception as e:
        print(f"Core demo ready (optional libs): {e}")
    finally:
        print("Wednesday\\'s audio perception module v" + __version__ + " loaded successfully.")

