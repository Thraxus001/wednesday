from textblob import TextBlob
from dataclasses import dataclass
import logging

@dataclass
class SentimentResult:
    """Structured result for perception analysis."""
    polarity: float  # -1.0 (very negative) to 1.0 (very positive)
    subjectivity: float # 0.0 (objective fact) to 1.0 (personal opinion)
    mood_label: str

class TextPerception:
    """
    Wednesday's 'Ear' for text. 
    Analyzes nuance and emotional undertones.
    """
    def __init__(self):
        self.logger = logging.getLogger("Wednesday.Perception")

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Uses TextBlob to calculate the emotional weight of a sentence.
        """
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        # Mapping polarity to Wednesday's internal labels
        if polarity > 0.5:
            label = "enthusiastic"
        elif polarity > 0.1:
            label = "pleasant"
        elif polarity < -0.5:
            label = "hostile"
        elif polarity < -0.1:
            label = "annoyed"
        else:
            label = "neutral"
            
        return SentimentResult(
            polarity=polarity,
            subjectivity=analysis.sentiment.subjectivity,
            mood_label=label
        )

# Quick verification
if __name__ == "__main__":
    parser = TextPerception()
    test_phrase = "I absolutely love how gloomy today is."
    result = parser.analyze_sentiment(test_phrase)
    print(f"Text: {test_phrase}")
    print(f"Detected Mood: {result.mood_label} (Polarity: {result.polarity})")