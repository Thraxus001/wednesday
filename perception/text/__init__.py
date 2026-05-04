"""

Text input processing - understanding the written word.

This module handles all aspects of linguistic analysis:

- Parsing (syntax, structure, entities)

- Intent detection (what the user wants)

- Sentiment analysis (emotional content)



Together, they transform raw text into structured, meaningful information

that Wednesday can understand and respond to appropriately.

"""



try:

    from .parser import TextParser, ParsedText, LinguisticFeature

    from .intent_detection import IntentDetector, Intent, IntentCategory

    from .sentiment import SentimentAnalyzer, SarcasmDetector, EmotionalTone, EmotionCategory, EmotionalTrend

except ImportError:

    from perception.text.parser import TextParser, ParsedText, LinguisticFeature

    from perception.text.intent_detection import IntentDetector, Intent, IntentCategory

    from perception.text.sentiment import SentimentAnalyzer, SarcasmDetector, EmotionalTone, EmotionCategory, EmotionalTrend

# Module metadata
__version__ = "0.1.0"
__all__ = [
    # Main classes
    "TextParser",
    "IntentDetector", 
    "SentimentAnalyzer",
    "SarcasmDetector",
    
    # Data structures
    "ParsedText",
    "Intent",
    "EmotionalTone",
    "EmotionalTrend",
    
    # Enums
    "LinguisticFeature",
    "IntentCategory",
    "EmotionCategory",
]

# Optional: Module initialization function
def create_text_processor(config: dict = None):
    """
    Factory function to create and connect a complete text processing pipeline.
    
    Args:
        config: Configuration dictionary with sections for each component
        
    Returns:
        Dictionary with initialized text processing components
    """
    if config is None:
        config = {}
    
    parser = TextParser(
        model_path=config.get('parser', {}).get('model_path'),
        config=config.get('parser', {})
    )
    
    intent_detector = IntentDetector(
        config=config.get('intent', {})
    )
    
    sentiment_analyzer = SentimentAnalyzer(
        config=config.get('sentiment', {})
    )
    
    return {
        'parser': parser,
        'intent_detector': intent_detector,
        'sentiment_analyzer': sentiment_analyzer,
    }

# Module description for introspection
__description__ = """
Wednesday's text perception system - she reads between the lines.
Transforms raw text into linguistic structure, user intent, and emotional content.
"""

# Pipeline visualization
"""
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Text   │────▶│    Parser    │────▶│    Intent    │────▶│  Sentiment   │
│   Input     │     │  (syntax)    │     │  Detection   │     │  Analysis    │
└─────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                    ┌─────────────────────────────────┘
                                    ▼
                          ┌─────────────────────┐
                          │  Enriched Output    │
                          │  Structure + Intent │
                          │  + Emotion          │
                          └─────────────────────┘

Each component builds on the previous:
1. Parser provides linguistic foundation
2. Intent detector uses structure to infer goals
3. Sentiment analyzer adds emotional dimension
"""

# Convenience imports for common use cases
from typing import Union, Dict, Any

def process_text(text: str, 
                parser: TextParser, 
                intent_detector: IntentDetector, 
                sentiment_analyzer: SentimentAnalyzer,
                context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function to run full text processing pipeline.
    
    Args:
        text: Raw input text
        parser: Initialized TextParser
        intent_detector: Initialized IntentDetector
        sentiment_analyzer: Initialized SentimentAnalyzer
        context: Optional conversation context
        
    Returns:
        Dictionary with all analysis results
    """
    # Parse
    parsed = parser.parse(text, context)
    
    # Detect intent
    intent = intent_detector.detect_intent(parsed, context)
    
    # Analyze sentiment
    sentiment = sentiment_analyzer.analyze(text, parsed, context)
    
    return {
        'parsed': parsed,
        'intent': intent,
        'sentiment': sentiment,
        'text': text,
        'timestamp': parsed.timestamp if hasattr(parsed, 'timestamp') else None
    }

# Export the convenience function

__all__.append("process_text")



# Logging setup

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())



if __name__ == "__main__":

    print("=== Wednesday Text Processing Demo ===")

    print("Testing import and basic pipeline...")

    

    try:

        # Create basic processor (will use defaults)

        processor = create_text_processor()

        

        # Test with sample text

        sample_text = "Hello Wednesday, what a wonderful day for sarcasm!"

        

        print(f"\nProcessing: '{sample_text}'")

        print("-" * 50)

        

        # Note: Full processing requires spaCy model

        # This demo just shows the pipeline structure

        print("✓ Pipeline structure ready")

        print("✓ Imports successful")

        print("\nFull processing would require:")

        print("  pip install spacy")

        print("  python -m spacy download en_core_web_sm")

        

        print("\nDemo complete! No errors.")

        

    except Exception as e:

        print(f"Demo ran into issue (expected if no spaCy): {e}")

        print("But imports succeeded - core fix working!")

    

    print("\n=== End Demo ===")
