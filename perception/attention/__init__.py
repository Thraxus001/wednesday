"""
Attention mechanisms - what Wednesday focuses on and why.
This module determines what information deserves processing resources,
how attention shifts, and what gets ignored as background noise.

The attention system consists of two main components:
- SalienceDetector: Calculates what's important (bottom-up attention)
- FocusManager: Directs the attentional spotlight (top-down attention)

Together they create a selective attention system that filters the
overwhelming stream of sensory input into manageable, relevant information.
"""

from .salience import SalienceDetector
from .focus_manager import FocusManager, FocusState, FocusPriority, FocusTarget

# Module metadata
__version__ = "0.1.0"
__all__ = [
    # Main classes
    "SalienceDetector",
    "FocusManager",
    
    # Focus-related enums and types
    "FocusState",
    "FocusPriority", 
    "FocusTarget",
]

# Optional: Module initialization function
def create_attention_system(config: dict = None):
    """
    Factory function to create and connect a complete attention system.
    
    Args:
        config: Configuration dictionary for both components
        
    Returns:
        Tuple of (salience_detector, focus_manager) properly initialized
    """
    if config is None:
        config = {}
    
    salience = SalienceDetector(config.get('salience', {}))
    focus = FocusManager(salience, config.get('focus', {}))
    
    return salience, focus

# Module description for introspection
__description__ = """
Wednesday's attention system - selective, focused, and hard to distract.
Determines what information deserves processing in a world full of noise.
"""

# Notes for other modules
"""
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Perception     │────▶│   Salience   │────▶│     Focus       │
│  Input Streams  │     │  Detector    │     │   Manager       │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │  Filtered       │
                                              │  Output         │
                                              └─────────────────┘

Input → Salience scores → Focus selection → Attended information
"""

# If you want to expose the enums directly at module level
from enum import Enum

class AttentionMode(Enum):
    """High-level attention modes for the whole system"""
    SCANNING = "scanning"      # Looking for something interesting
    FOCUSED = "focused"        # Locked onto something
    MONITORING = "monitoring"  # Background awareness
    DAYDREAMING = "daydreaming" # Internal attention

# Add to exports if desired
# __all__.append("AttentionMode")