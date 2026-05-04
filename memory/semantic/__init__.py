"""
Semantic memory - general knowledge about the world.
Stores facts, concepts, relationships, and general knowledge
that Wednesday has learned about the world, people, and herself.
"""
from .knowledge_base import KnowledgeBase
from .concepts import ConceptNetwork

__all__ = ['KnowledgeBase', 'ConceptNetwork']