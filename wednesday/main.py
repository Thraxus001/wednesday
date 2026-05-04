"""
/wednesday/main.py
Core Wednesday AI class - Integrated with Perception (Phase 2) and Emotion (Phase 3).
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Import our new modules
from wednesday.perception.text_engine import TextPerception
from wednesday.perception.vision_engine import VisionPerception
from wednesday.config import WednesdayConfig

# --- BLOCK 1: EMOTION ENGINE ---
class EmotionEngine:
    """Manages Wednesday's internal emotional state and sarcasm levels."""
    def __init__(self, base_sarcasm: float = 0.7):
        self.sarcasm = base_sarcasm
        self.boredom = 0.2
        self.empathy = 0.1
        self.current_mood = "neutral"

    def update_mood(self, user: str, sentiment_label: str):
        """Logic to shift Wednesday's mood based on context."""
        if user == "Stranger":
            self.sarcasm = min(1.0, self.sarcasm + 0.1) # Gets more sarcastic with strangers
            self.current_mood = "defensive"
        elif sentiment_label == "hostile":
            self.sarcasm = 1.0 # Maximum sass
            self.current_mood = "combative"
        elif sentiment_label == "enthusiastic":
            self.boredom += 0.2 # Excessive cheer bores her
            self.current_mood = "unimpressed"
        else:
            self.current_mood = "chilled_gloom"

# --- BLOCK 2: MAIN ORCHESTRATOR ---
class Wednesday:
    def __init__(self, config: Optional[WednesdayConfig] = None):
        self.config = config or WednesdayConfig()
        
        # Initialize Sub-Systems
        self.text_eye = TextPerception()
        self.vision_eye = VisionPerception()
        self.emotions = EmotionEngine(base_sarcasm=self.config.emotion.default_sarcasm)
        
        self.initialized = False
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("WednesdayCore")

    def initialize(self) -> bool:
        """Boot sequence for all modules."""
        # In a real CS project, you'd check if camera/models load here
        self.initialized = True
        return True

    def process_interaction(self, user_text: str) -> str:
        """The Multimodal Pipeline: Vision -> Text -> Emotion -> Response."""
        if not self.initialized:
            return "System not ready."

        # 1. PERCEPTION: Who and What
        user_name = self.vision_eye.identify_user()
        sentiment = self.text_eye.analyze_sentiment(user_text)

        # 2. EMOTION: Update internal state
        self.emotions.update_mood(user_name, sentiment.mood_label)

        # 3. RESPONSE GENERATION: Sarcasm-weighted output
        return self._generate_response(user_name, sentiment.mood_label)

    def _generate_response(self, user: str, mood: str) -> str:
        """Placeholder for Language Phase. Uses Emotion state to pick tone."""
        sarcasm_level = self.emotions.sarcasm
        
        if sarcasm_level > 0.9:
            return f"[{user}] spoke. How exhausting. Your '{mood}' energy is truly polluting the room."
        else:
            return f"Hello {user}. I've noted your {mood} state. Don't expect a hug."

# --- BLOCK 3: EXECUTION ---
if __name__ == "__main__":
    wed = Wednesday()
    if wed.initialize():
        print("--- Wednesday is watching ---")
        # Simulate a user interaction
        response = wed.process_interaction("I am so excited to be working with you!")
        print(f"AI: {response}")
        print(f"Current Sarcasm Level: {wed.emotions.sarcasm}")