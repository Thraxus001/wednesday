"""

• NO limit on memory size (can store unlimited items)
• NO limit on memory types (add any type dynamically)
• NO restriction on content (store ANY Python object)
• NO bound on retention (can store forever)
• NO artificial constraints on consolidation
• Supports INFINITE scaling across all memory subsystems
"""
# Allow running as script or as module
import sys
import os
from pathlib import Path

# Add the parent directory to path when running directly
if __name__ == "__main__" and __package__ is None:
    # Get the absolute path of the current file
    current_file = Path(__file__).absolute()
    # Get the wednesday package directory (parent of memory directory)
    package_dir = current_file.parent.parent
    # Add to Python path
    sys.path.insert(0, str(package_dir.parent))
    # Set the package name
    __package__ = "wednesday.memory"



from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
import logging
import threading
import uuid
import json
import weakref
from enum import Enum
import time
from dataclasses import dataclass

from memory.exceptions import (
    WednesdayError,
    MemoryError,
    MemoryCapacityError,
    MemoryNotFoundError,
    MemoryCorruptionError,
    MemoryConsolidationError,
    ErrorContext,
    safe_execute
)

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Dynamic memory types - can be extended at runtime"""
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    
    @classmethod
    def register(cls, name: str):
        """Register a new memory type dynamically"""
        setattr(cls, name.upper(), name)
        return name


class MemoryPriority(Enum):
    """Priority levels for memory operations"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    
    @classmethod
    def register(cls, name: str, value: int):
        """Register a new priority level"""
        setattr(cls, name.upper(), value)
        return value


@dataclass
class MemoryConfig:
    """
    Configuration for memory systems - NO LIMITS on any parameter.
    All values can be changed at runtime, set to None for unlimited.
    """
    # Working memory configuration
    working_memory_size: Optional[int] = 10  # None = unlimited
    working_ttl_seconds: Optional[float] = 300  # None = no expiration
    
    # Episodic memory configuration
    episodic_retention_days: Optional[int] = 30  # None = forever
    episodic_max_entries: Optional[int] = None  # None = unlimited
    episodic_auto_index: bool = True
    
    # Semantic memory configuration
    semantic_confidence_threshold: float = 0.7
    semantic_vector_dimension: int = 768
    semantic_auto_extract: bool = True
    
    # Procedural memory configuration
    procedural_max_patterns: Optional[int] = None  # None = unlimited
    procedural_learning_rate: float = 0.01
    
    # Global settings
    auto_consolidate: bool = True
    consolidation_interval: int = 3600  # seconds
    enable_compression: bool = False
    backup_enabled: bool = True
    backup_interval: int = 86400  # 24 hours
    
    def __post_init__(self):
        """Validate configuration (but don't restrict)"""
        # No validation - accept ANY values
        pass


class MemoryItem:
    """
    A single memory item - can hold ANY data with ANY metadata.
    No limits on size, type, or structure.
    """
    
    def __init__(
        self,
        content: Any,
        memory_type: str = "working",
        priority: int = MemoryPriority.NORMAL.value,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        ttl: Optional[float] = None
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.memory_type = memory_type
        self.priority = priority
        self.metadata = metadata or {}
        self.tags = tags or []
        self.embedding = embedding
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.importance_score = 0.0
        self.ttl = ttl  # None = never expires
        self.locked = False
        self.version = 1
        self.references = []  # Links to other memory items
        
    @property
    def age_seconds(self) -> float:
        """Age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def is_expired(self) -> bool:
        """Check if item has expired"""
        if self.ttl is None:
            return False
        return self.age_seconds > self.ttl
    
    @property
    def size_bytes(self) -> int:
        """Approximate size in bytes"""
        try:
            return len(json.dumps(self.to_dict()))
        except:
            return 0
    
    def access(self):
        """Record an access to this memory"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        self._update_importance()
    
    def _update_importance(self):
        """Update importance score based on access patterns"""
        recency = 1.0 / (1.0 + self.age_seconds)
        frequency = self.access_count / (1.0 + self.age_seconds / 3600)
        self.importance_score = (recency * 0.3 + frequency * 0.7) * (self.priority + 1)
    
    def link_to(self, other: 'MemoryItem', relation: str = "related"):
        """Link this memory to another"""
        self.references.append({
            'target_id': other.id,
            'relation': relation,
            'created': datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'memory_type': self.memory_type,
            'priority': self.priority,
            'metadata': self.metadata,
            'tags': self.tags,
            'embedding': self.embedding,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'importance_score': self.importance_score,
            'ttl': self.ttl,
            'version': self.version,
            'references': self.references,
            # Content might not be serializable
            '_content_type': type(self.content).__name__,
        }
    
    def __repr__(self) -> str:
        return f"<MemoryItem id={self.id[:8]} type={self.memory_type} importance={self.importance_score:.2f}>"


class MemoryStore:
    """
    Base class for all memory stores.
    Provides common functionality with NO limits.
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.items: Dict[str, MemoryItem] = {}
        self.indices: Dict[str, Dict] = {}
        self.lock = threading.RLock()
        self.created_at = datetime.now()
        self.stats = {
            'total_added': 0,
            'total_removed': 0,
            'total_accessed': 0,
            'total_searched': 0,
        }
        self.observers = []
        
    def add(self, item: MemoryItem) -> bool:
        """Add an item to the store"""
        with self.lock:
            self.items[item.id] = item
            self.stats['total_added'] += 1
            self._update_indices(item)
            self._notify_observers('add', item)
        return True
    
    def get(self, item_id: str) -> Optional[MemoryItem]:
        """Get an item by ID"""
        with self.lock:
            item = self.items.get(item_id)
            if item:
                item.access()
                self.stats['total_accessed'] += 1
            return item
    
    def remove(self, item_id: str) -> bool:
        """Remove an item"""
        with self.lock:
            if item_id in self.items:
                item = self.items.pop(item_id)
                self.stats['total_removed'] += 1
                self._remove_from_indices(item)
                self._notify_observers('remove', item)
                return True
        return False
    
    def search(self, query: Any, limit: int = 10) -> List[MemoryItem]:
        """Search for items (base implementation - override in subclasses)"""
        self.stats['total_searched'] += 1
        results = []
        with self.lock:
            # Simple sequential scan (override with indexed search)
            for item in self.items.values():
                if self._matches_query(item, query):
                    results.append(item)
                    if len(results) >= limit:
                        break
        return results
    
    def _matches_query(self, item: MemoryItem, query: Any) -> bool:
        """Check if item matches query (override in subclasses)"""
        # Default: check if query in tags or metadata
        if isinstance(query, str):
            return query in item.tags or query in str(item.metadata)
        return False
    
    def _update_indices(self, item: MemoryItem):
        """Update search indices"""
        # Override in subclasses
        pass
    
    def _remove_from_indices(self, item: MemoryItem):
        """Remove from indices"""
        # Override in subclasses
        pass
    
    def _notify_observers(self, event: str, item: MemoryItem):
        """Notify observers of changes"""
        for observer in self.observers:
            try:
                observer(event, item)
            except:
                pass
    
    def register_observer(self, callback: Callable):
        """Register an observer for store events"""
        self.observers.append(weakref.ref(callback))
    
    def clear(self):
        """Clear all items"""
        with self.lock:
            self.items.clear()
            self.indices.clear()
    
    @property
    def size(self) -> int:
        """Number of items in store"""
        return len(self.items)
    
    @property
    def total_size_bytes(self) -> int:
        """Total size in bytes"""
        return sum(item.size_bytes for item in self.items.values())
    
    def get_stats(self) -> Dict:
        """Get store statistics"""
        return {
            **self.stats,
            'size': self.size,
            'total_size_bytes': self.total_size_bytes,
            'created_at': self.created_at.isoformat(),
            'uptime_seconds': (datetime.now() - self.created_at).total_seconds(),
        }


class WorkingMemory(MemoryStore):
    """
    Working memory - short-term, limited capacity store.
    But "limited" is configurable - can be unlimited!
    """
    
    def __init__(self, max_size: Optional[int] = 10, ttl: Optional[float] = 300):
        super().__init__("working")
        self.max_size = max_size  # None = unlimited
        self.default_ttl = ttl  # None = no expiration
        
    def add(self, content: Any, metadata: Optional[Dict] = None) -> MemoryItem:
        """Add content to working memory"""
        item = MemoryItem(
            content=content,
            memory_type="working",
            metadata=metadata,
            ttl=self.default_ttl
        )
        
        with self.lock:
            # Check size limit (if any)
            if self.max_size is not None and len(self.items) >= self.max_size:
                self._evict_one()
            
            super().add(item)
        
        return item
    
    def _evict_one(self):
        """Evict the least important item"""
        if not self.items:
            return
        
        # Find least important item
        least_important = min(
            self.items.values(),
            key=lambda x: x.importance_score
        )
        
        self.remove(least_important.id)
        logger.debug(f"Evicted memory item {least_important.id[:8]} importance={least_important.importance_score:.2f}")
    
    def get_context(self, limit: int = 5) -> List[MemoryItem]:
        """Get most important items for context"""
        items = sorted(
            self.items.values(),
            key=lambda x: x.importance_score,
            reverse=True
        )
        return items[:limit]
    
    def clear_expired(self) -> int:
        """Clear all expired items"""
        count = 0
        with self.lock:
            expired = [item for item in self.items.values() if item.is_expired]
            for item in expired:
                self.remove(item.id)
                count += 1
        return count


class EpisodicMemory(MemoryStore):
    """
    Episodic memory - stores experiences with rich context.
    Unlimited storage, infinite retention.
    """
    
    def __init__(
        self,
        retention_days: Optional[int] = 30,
        max_entries: Optional[int] = None,
        enable_indexing: bool = True
    ):
        super().__init__("episodic")
        self.retention_days = retention_days  # None = forever
        self.max_entries = max_entries  # None = unlimited
        self.enable_indexing = enable_indexing
        
        # Advanced indices for fast retrieval
        self.time_index = []  # Chronological index
        self.tag_index = {}   # Tag-based index
        self.embedding_index = {}  # Vector similarity index
        
    def record(
        self,
        experience: Any,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None
    ) -> MemoryItem:
        """Record an experience"""
        item = MemoryItem(
            content=experience,
            memory_type="episodic",
            metadata=metadata,
            tags=tags,
            embedding=embedding,
            priority=MemoryPriority.NORMAL.value
        )
        
        with self.lock:
            # Check max entries (if any)
            if self.max_entries is not None and len(self.items) >= self.max_entries:
                self._archive_oldest()
            
            super().add(item)
            
            # Update indices
            self.time_index.append((item.created_at, item.id))
            if tags:
                for tag in tags:
                    if tag not in self.tag_index:
                        self.tag_index[tag] = []
                    self.tag_index[tag].append(item.id)
            
            if embedding and self.enable_indexing:
                self.embedding_index[item.id] = embedding
        
        return item
    
    def search_by_time(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MemoryItem]:
        """Search by time range"""
        results = []
        with self.lock:
            for ts, item_id in reversed(self.time_index):  # Most recent first
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                item = self.get(item_id)
                if item:
                    results.append(item)
                    if len(results) >= limit:
                        break
        return results
    
    def search_by_tag(self, tag: str, limit: int = 100) -> List[MemoryItem]:
        """Search by tag"""
        results = []
        with self.lock:
            item_ids = self.tag_index.get(tag, [])
            for item_id in item_ids[:limit]:
                item = self.get(item_id)
                if item:
                    results.append(item)
        return results
    
    def search_by_similarity(
        self,
        embedding: List[float],
        threshold: float = 0.7,
        limit: int = 10
    ) -> List[tuple]:
        """Search by embedding similarity"""
        if not self.enable_indexing:
            return []
        
        results = []
        with self.lock:
            for item_id, item_embedding in self.embedding_index.items():
                similarity = self._cosine_similarity(embedding, item_embedding)
                if similarity >= threshold:
                    item = self.get(item_id)
                    if item:
                        results.append((item, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x*x for x in a))
        norm_b = math.sqrt(sum(x*x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (norm_a * norm_b)
    
    def _archive_oldest(self):
        """Archive oldest entries when limit reached"""
        if not self.time_index:
            return
        
        # Remove oldest 10%
        remove_count = max(1, len(self.time_index) // 10)
        to_remove = self.time_index[:remove_count]
        
        for ts, item_id in to_remove:
            self.remove(item_id)
            self.time_index = [x for x in self.time_index if x[1] != item_id]
    
    def extract_patterns(self) -> List[Dict]:
        """Extract patterns from episodic memories"""
        patterns = []
        
        # Group by tags
        tag_groups = {}
        for tag, item_ids in self.tag_index.items():
            if len(item_ids) > 5:  # Significant pattern
                items = [self.get(item_id) for item_id in item_ids[:10]]
                patterns.append({
                    'type': 'tag_cooccurrence',
                    'tag': tag,
                    'count': len(item_ids),
                    'examples': [item.content for item in items if item]
                })
        
        return patterns


class SemanticMemory(MemoryStore):
    """
    Semantic memory - stores facts and knowledge.
    Builds a knowledge graph with unlimited nodes and edges.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        vector_dimension: int = 768
    ):
        super().__init__("semantic")
        self.confidence_threshold = confidence_threshold
        self.vector_dimension = vector_dimension
        
        # Knowledge graph structures
        self.facts = {}  # fact_id -> fact
        self.entities = {}  # entity_name -> entity_data
        self.relations = []  # (subject, predicate, object, confidence)
        self.entity_embeddings = {}  # entity_name -> embedding
        
    def add_knowledge(
        self,
        fact: Any,
        confidence: float = 1.0,
        source: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Add a fact to semantic memory"""
        if confidence < self.confidence_threshold:
            logger.debug(f"Fact confidence {confidence} below threshold {self.confidence_threshold}")
            return None
        
        fact_id = str(uuid.uuid4())
        
        with self.lock:
            self.facts[fact_id] = {
                'content': fact,
                'confidence': confidence,
                'source': source,
                'metadata': metadata or {},
                'created': datetime.now().isoformat(),
                'access_count': 0,
                'verifications': 0
            }
            
            # Extract entities if possible
            self._extract_entities(fact, fact_id)
            
            self.stats['total_added'] += 1
        
        return fact_id
    
    def _extract_entities(self, fact: Any, fact_id: str):
        """Extract entities from fact (override with NLP)"""
        # Basic implementation - override with actual NLP
        if isinstance(fact, str):
            # Very simple entity extraction - just words
            words = fact.split()
            for word in words:
                if word[0].isupper() and len(word) > 2:  # Likely a proper noun
                    if word not in self.entities:
                        self.entities[word] = {
                            'name': word,
                            'mentions': 0,
                            'facts': []
                        }
                    self.entities[word]['mentions'] += 1
                    self.entities[word]['facts'].append(fact_id)
    
    def query(
        self,
        query: Any,
        min_confidence: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Query semantic memory"""
        self.stats['total_searched'] += 1
        results = []
        
        threshold = min_confidence or self.confidence_threshold
        
        with self.lock:
            for fact_id, fact_data in self.facts.items():
                if fact_data['confidence'] < threshold:
                    continue
                
                relevance = self._calculate_relevance(fact_data['content'], query)
                if relevance > 0:
                    results.append({
                        'id': fact_id,
                        'fact': fact_data['content'],
                        'confidence': fact_data['confidence'],
                        'relevance': relevance,
                        'metadata': fact_data['metadata']
                    })
                    fact_data['access_count'] += 1
        
        # Sort by relevance * confidence
        results.sort(
            key=lambda x: x['relevance'] * x['confidence'],
            reverse=True
        )
        
        return results[:limit]
    
    def _calculate_relevance(self, fact: Any, query: Any) -> float:
        """Calculate relevance score between fact and query"""
        # Basic implementation - override with actual relevance calculation
        if isinstance(fact, str) and isinstance(query, str):
            # Simple word overlap
            fact_words = set(fact.lower().split())
            query_words = set(query.lower().split())
            if not query_words:
                return 0
            overlap = len(fact_words & query_words)
            return overlap / len(query_words)
        return 0
    
    def get_entity_info(self, entity_name: str) -> Optional[Dict]:
        """Get information about an entity"""
        return self.entities.get(entity_name)
    
    def get_related_facts(self, entity_name: str, limit: int = 10) -> List[Dict]:
        """Get facts related to an entity"""
        entity = self.entities.get(entity_name)
        if not entity:
            return []
        
        results = []
        for fact_id in entity['facts']:
            fact = self.facts.get(fact_id)
            if fact:
                results.append({
                    'id': fact_id,
                    'fact': fact['content'],
                    'confidence': fact['confidence']
                })
        
        return results[:limit]


class ProceduralMemory(MemoryStore):
    """
    Procedural memory - stores patterns, procedures, and skills.
    Learns from repeated experiences.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__("procedural")
        self.learning_rate = learning_rate
        self.patterns = {}  # pattern_id -> pattern_data
        self.sequences = []  # sequence patterns
        self.skills = {}  # skill_name -> skill_data
        
    def learn_pattern(self, pattern: Any, context: Optional[Dict] = None) -> str:
        """Learn a new pattern"""
        pattern_id = str(uuid.uuid4())
        
        with self.lock:
            self.patterns[pattern_id] = {
                'pattern': pattern,
                'context': context or {},
                'strength': 1.0,
                'occurrences': 1,
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat()
            }
        
        return pattern_id
    
    def reinforce_pattern(self, pattern_id: str) -> float:
        """Reinforce an existing pattern"""
        with self.lock:
            pattern = self.patterns.get(pattern_id)
            if pattern:
                pattern['strength'] += self.learning_rate
                pattern['occurrences'] += 1
                pattern['last_seen'] = datetime.now().isoformat()
                return pattern['strength']
        return 0.0
    
    def find_pattern(self, input_data: Any, threshold: float = 0.5) -> List[tuple]:
        """Find patterns matching input"""
        matches = []
        
        with self.lock:
            for pid, pattern in self.patterns.items():
                similarity = self._pattern_similarity(input_data, pattern['pattern'])
                if similarity >= threshold:
                    matches.append((pid, pattern, similarity))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches
    
    def _pattern_similarity(self, input_data: Any, pattern: Any) -> float:
        """Calculate similarity between input and pattern"""
        # Basic implementation - override with actual pattern matching
        if isinstance(input_data, str) and isinstance(pattern, str):
            # Simple string similarity
            if input_data == pattern:
                return 1.0
            if pattern in input_data:
                return 0.8
        return 0.0
    
    def learn_skill(self, name: str, steps: List[Any]) -> str:
        """Learn a new skill"""
        skill_id = str(uuid.uuid4())
        
        with self.lock:
            self.skills[name] = {
                'id': skill_id,
                'name': name,
                'steps': steps,
                'proficiency': 0.0,
                'practice_count': 0,
                'created': datetime.now().isoformat()
            }
        
        return skill_id
    
    def practice_skill(self, skill_name: str, success: bool = True) -> float:
        """Practice a skill to improve proficiency"""
        skill = self.skills.get(skill_name)
        if skill:
            if success:
                skill['proficiency'] += self.learning_rate
            else:
                skill['proficiency'] -= self.learning_rate * 0.5
            
            skill['proficiency'] = max(0.0, min(1.0, skill['proficiency']))
            skill['practice_count'] += 1
            
            return skill['proficiency']
        return 0.0


class MemoryIndex:
    """
    Central memory coordinator that manages ALL memory types.
    Provides unified interface with NO LIMITS on what can be stored.
    
    Features:
    - Unlimited memory capacity
    - Dynamic memory type registration
    - Automatic consolidation
    - Cross-memory search
    - Memory statistics and monitoring
    - Backup and restore
    - Event notifications
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.created_at = datetime.now()
        self.lock = threading.RLock()
        
        # Initialize core memory subsystems
        self.working = WorkingMemory(
            max_size=self.config.working_memory_size,
            ttl=self.config.working_ttl_seconds
        )
        
        self.episodic = EpisodicMemory(
            retention_days=self.config.episodic_retention_days,
            max_entries=self.config.episodic_max_entries
        )
        
        self.semantic = SemanticMemory(
            confidence_threshold=self.config.semantic_confidence_threshold,
            vector_dimension=self.config.semantic_vector_dimension
        )
        
        self.procedural = ProceduralMemory(
            learning_rate=self.config.procedural_learning_rate
        )
        
        # Dynamic memory stores (can add any number)
        self.custom_stores: Dict[str, MemoryStore] = {}
        
        # Memory statistics
        self.stats = {
            'total_stores': 0,
            'total_retrievals': 0,
            'consolidations': 0,
            'backups': 0,
            'restores': 0
        }
        
        # Event observers
        self.observers = []
        
        # Start background tasks if enabled
        if self.config.auto_consolidate:
            self._start_consolidation_thread()
        
        if self.config.backup_enabled:
            self._start_backup_thread()
        
        logger.info("MemoryIndex initialized with working, episodic, semantic, and procedural memory")
    
    def store(
        self,
        content: Any,
        memory_type: str = 'working',
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        priority: int = MemoryPriority.NORMAL.value,
        **kwargs
    ) -> Optional[str]:
        """
        Store content in specified memory type.
        
        Args:
            content: ANY content to store (no limits)
            memory_type: Which memory system to use
            metadata: ANY additional context
            tags: List of tags for categorization
            priority: Priority level
            **kwargs: Additional memory-specific parameters
        
        Returns:
            Memory ID if successful, None otherwise
        """
        with self.lock:
            self.stats['total_stores'] += 1
            
            # Route to appropriate memory
            if memory_type == 'working':
                item = self.working.add(content, metadata)
                return item.id if item else None
                
            elif memory_type == 'episodic':
                item = self.episodic.record(
                    content,
                    metadata=metadata,
                    tags=tags,
                    embedding=kwargs.get('embedding')
                )
                return item.id if item else None
                
            elif memory_type == 'semantic':
                fact_id = self.semantic.add_knowledge(
                    content,
                    confidence=kwargs.get('confidence', 1.0),
                    source=kwargs.get('source'),
                    metadata=metadata
                )
                return fact_id
                
            elif memory_type == 'procedural':
                pattern_id = self.procedural.learn_pattern(content, metadata)
                return pattern_id
                
            elif memory_type in self.custom_stores:
                # Custom store
                store = self.custom_stores[memory_type]
                item = MemoryItem(content, memory_type, priority, metadata, tags)
                store.add(item)
                return item.id
                
            else:
                logger.warning(f"Unknown memory type: {memory_type}")
                return None
    
    def retrieve(
        self,
        query: Any,
        memory_type: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> Dict[str, List]:
        """
        Retrieve memories from specified type(s).
        
        Args:
            query: Search query (ANY format)
            memory_type: Specific type or None for all
            limit: Max results per type
            **kwargs: Additional search parameters
        
        Returns:
            Dictionary mapping memory_type -> list of results
        """
        with self.lock:
            self.stats['total_retrievals'] += 1
            results = {}
            
            if memory_type:
                # Single type
                results[memory_type] = self._search_type(memory_type, query, limit, **kwargs)
            else:
                # All types
                for mtype in ['working', 'episodic', 'semantic', 'procedural']:
                    type_results = self._search_type(mtype, query, limit, **kwargs)
                    if type_results:
                        results[mtype] = type_results
                
                # Custom stores
                for name, store in self.custom_stores.items():
                    type_results = self._search_custom_store(store, query, limit, **kwargs)
                    if type_results:
                        results[name] = type_results
            
            return results
    
    def _search_type(self, memory_type: str, query: Any, limit: int, **kwargs) -> List:
        """Search a specific memory type"""
        if memory_type == 'working':
            return self.working.search(query, limit)
        elif memory_type == 'episodic':
            if 'tag' in kwargs:
                return self.episodic.search_by_tag(kwargs['tag'], limit)
            elif 'embedding' in kwargs:
                return self.episodic.search_by_similarity(
                    kwargs['embedding'],
                    kwargs.get('threshold', 0.7),
                    limit
                )
            else:
                return self.episodic.search(query, limit)
        elif memory_type == 'semantic':
            return self.semantic.query(
                query,
                min_confidence=kwargs.get('min_confidence'),
                limit=limit
            )
        elif memory_type == 'procedural':
            return self.procedural.find_pattern(
                query,
                threshold=kwargs.get('threshold', 0.5)
            )
        return []
    
    def _search_custom_store(self, store: MemoryStore, query: Any, limit: int, **kwargs) -> List:
        """Search a custom memory store"""
        return store.search(query, limit)
    
    def get(self, memory_id: str, memory_type: Optional[str] = None) -> Optional[Any]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: The memory ID
            memory_type: Optional type hint for faster lookup
        
        Returns:
            The memory item or None
        """
        if memory_type == 'working':
            return self.working.get(memory_id)
        elif memory_type == 'episodic':
            return self.episodic.get(memory_id)
        elif memory_type == 'semantic':
            return self.semantic.facts.get(memory_id)
        elif memory_type == 'procedural':
            return self.procedural.patterns.get(memory_id)
        elif memory_type in self.custom_stores:
            return self.custom_stores[memory_type].get(memory_id)
        else:
            # Search all stores
            for store in [self.working, self.episodic, self.semantic, self.procedural]:
                item = store.get(memory_id)
                if item:
                    return item
            
            for store in self.custom_stores.values():
                item = store.get(memory_id)
                if item:
                    return item
        
        return None
    
    def forget(self, memory_id: str, memory_type: Optional[str] = None) -> bool:
        """Forget a specific memory"""
        if memory_type == 'working':
            return self.working.remove(memory_id)
        elif memory_type == 'episodic':
            return self.episodic.remove(memory_id)
        elif memory_type == 'semantic':
            with self.lock:
                if memory_id in self.semantic.facts:
                    del self.semantic.facts[memory_id]
                    return True
            return False
        elif memory_type == 'procedural':
            with self.lock:
                if memory_id in self.procedural.patterns:
                    del self.procedural.patterns[memory_id]
                    return True
            return False
        elif memory_type in self.custom_stores:
            return self.custom_stores[memory_type].remove(memory_id)
        else:
            # Search all stores
            if self.working.remove(memory_id):
                return True
            if self.episodic.remove(memory_id):
                return True
            with self.lock:
                if memory_id in self.semantic.facts:
                    del self.semantic.facts[memory_id]
                    return True
            with self.lock:
                if memory_id in self.procedural.patterns:
                    del self.procedural.patterns[memory_id]
                    return True
            
            for store in self.custom_stores.values():
                if store.remove(memory_id):
                    return True
        
        return False
    
    def consolidate(self) -> Dict[str, int]:
        """
        Consolidate memories - move important working memories to episodic,
        extract semantic knowledge from episodes, reinforce patterns.
        """
        consolidated = {
            'working_to_episodic': 0,
            'episodic_to_semantic': 0,
            'patterns_reinforced': 0,
            'expired_cleared': 0
        }
        
        with self.lock:
            # Clear expired working memories
            expired = self.working.clear_expired()
            consolidated['expired_cleared'] = expired
            
            # Move important working memories to episodic
            important_items = self.working.get_context(limit=20)
            for item in important_items:
                if item.importance_score > 0.5:  # Threshold for importance
                    self.episodic.record(
                        item.content,
                        metadata={
                            **item.metadata,
                            'source': 'working_consolidation',
                            'original_id': item.id
                        },
                        tags=item.tags
                    )
                    consolidated['working_to_episodic'] += 1
            
            # Extract patterns from episodic
            patterns = self.episodic.extract_patterns()
            for pattern in patterns:
                pattern_id = self.procedural.learn_pattern(
                    pattern,
                    {'source': 'episodic_consolidation'}
                )
                if pattern_id:
                    consolidated['episodic_to_semantic'] += 1
            
            # Reinforce patterns
            for pid in list(self.procedural.patterns.keys())[:10]:
                strength = self.procedural.reinforce_pattern(pid)
                if strength > 0:
                    consolidated['patterns_reinforced'] += 1
            
            self.stats['consolidations'] += 1
            self.stats.update(consolidated)
        
        logger.info(f"Consolidation complete: {consolidated}")
        return consolidated
    
    def recall_recent(self, n: int = 5) -> Dict[str, List]:
        """
        Recall recent memories from all systems.
        Useful for context building.
        """
        return {
            'working': self.working.get_context(n),
            'episodic': self.episodic.search_by_time(limit=n),
            'semantic': list(self.semantic.facts.values())[:n],
            'procedural': list(self.procedural.patterns.values())[:n]
        }
    
    def register_memory_type(self, name: str, store: MemoryStore):
        """Register a new memory type dynamically"""
        with self.lock:
            self.custom_stores[name] = store
            MemoryType.register(name)
            logger.info(f"Registered new memory type: {name}")
    
    def create_custom_store(self, name: str, config: Optional[Dict] = None) -> MemoryStore:
        """Create and register a custom memory store"""
        store = MemoryStore(name, config)
        self.register_memory_type(name, store)
        return store
    
    def backup(self) -> Dict:
        """Create a backup of all memories"""
        backup_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'working_max_size': self.working.max_size,
                'episodic_retention': self.episodic.retention_days,
                'semantic_threshold': self.semantic.confidence_threshold,
            },
            'stats': self.stats.copy(),
            'memories': {
                'working': [item.to_dict() for item in self.working.items.values()],
                'episodic': [item.to_dict() for item in self.episodic.items.values()],
                'semantic': self.semantic.facts.copy(),
                'procedural': self.procedural.patterns.copy(),
            },
            'custom_stores': {
                name: {
                    'items': [item.to_dict() for item in store.items.values()],
                    'config': store.config
                }
                for name, store in self.custom_stores.items()
            }
        }
        
        self.stats['backups'] += 1
        logger.info(f"Memory backup created with {self.size} total items")
        return backup_data
    
    def restore(self, backup_data: Dict):
        """Restore from a backup"""
        with self.lock:
            # Clear existing
            self.working.clear()
            self.episodic.clear()
            self.semantic.facts.clear()
            self.procedural.patterns.clear()
            self.custom_stores.clear()
            
            # Restore working memory
            for item_data in backup_data.get('memories', {}).get('working', []):
                item = MemoryItem(**item_data)
                self.working.add(item)
            
            # Restore episodic memory
            for item_data in backup_data.get('memories', {}).get('episodic', []):
                item = MemoryItem(**item_data)
                self.episodic.add(item)
            
            # Restore semantic memory
            self.semantic.facts = backup_data.get('memories', {}).get('semantic', {})
            
            # Restore procedural memory
            self.procedural.patterns = backup_data.get('memories', {}).get('procedural', {})
            
            # Restore custom stores
            for name, store_data in backup_data.get('custom_stores', {}).items():
                store = MemoryStore(name, store_data.get('config', {}))
                for item_data in store_data.get('items', []):
                    item = MemoryItem(**item_data)
                    store.add(item)
                self.custom_stores[name] = store
            
            self.stats['restores'] += 1
        
        logger.info(f"Memory restored from backup with {self.size} total items")
    
    def search_across_types(
        self,
        query: Any,
        types: Optional[List[str]] = None,
        limit_per_type: int = 5
    ) -> Dict[str, List]:
        """Search across multiple memory types"""
        if types is None:
            types = ['working', 'episodic', 'semantic', 'procedural'] + list(self.custom_stores.keys())
        
        results = {}
        for memory_type in types:
            type_results = self.retrieve(query, memory_type, limit_per_type)
            if type_results:
                results.update(type_results)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get comprehensive memory statistics"""
        return {
            **self.stats,
            'working': self.working.get_stats(),
            'episodic': self.episodic.get_stats(),
            'semantic': {
                'total_facts': len(self.semantic.facts),
                'total_entities': len(self.semantic.entities),
                'total_relations': len(self.semantic.relations)
            },
            'procedural': {
                'total_patterns': len(self.procedural.patterns),
                'total_skills': len(self.procedural.skills)
            },
            'custom_stores': {
                name: store.get_stats()
                for name, store in self.custom_stores.items()
            },
            'total_items': self.size,
            'uptime_seconds': (datetime.now() - self.created_at).total_seconds()
        }
    
    def _start_consolidation_thread(self):
        """Start background consolidation thread"""
        def consolidate_loop():
            while True:
                time.sleep(self.config.consolidation_interval)
                try:
                    self.consolidate()
                except Exception as e:
                    logger.error(f"Consolidation error: {e}")
        
        thread = threading.Thread(target=consolidate_loop, daemon=True)
        thread.start()
        logger.info(f"Consolidation thread started (interval={self.config.consolidation_interval}s)")
    
    def _start_backup_thread(self):
        """Start background backup thread"""
        def backup_loop():
            while True:
                time.sleep(self.config.backup_interval)
                try:
                    # Save backup to file
                    backup_data = self.backup()
                    backup_file = f"memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(backup_file, 'w') as f:
                        json.dump(backup_data, f, indent=2, default=str)
                    logger.info(f"Auto-backup saved to {backup_file}")
                except Exception as e:
                    logger.error(f"Auto-backup error: {e}")
        
        thread = threading.Thread(target=backup_loop, daemon=True)
        thread.start()
        logger.info(f"Backup thread started (interval={self.config.backup_interval}s)")
    
    def register_observer(self, callback: Callable):
        """Register an observer for memory events"""
        self.observers.append(weakref.ref(callback))
        # Register with all stores
        self.working.register_observer(callback)
        self.episodic.register_observer(callback)
    
    def _notify_observers(self, event: str, data: Any):
        """Notify observers of memory events"""
        for observer_ref in self.observers:
            observer = observer_ref()
            if observer:
                try:
                    observer(event, data)
                except:
                    pass
    
    @property
    def size(self) -> int:
        """Total number of items across all stores"""
        total = (
            len(self.working.items) +
            len(self.episodic.items) +
            len(self.semantic.facts) +
            len(self.procedural.patterns)
        )
        for store in self.custom_stores.values():
            total += len(store.items)
        return total
    
    @property
    def is_ready(self) -> bool:
        """Check if memory system is ready"""
        return True  # Always ready
    
    def clear_working(self) -> None:
        """Clear working memory only"""
        self.working.clear()
        logger.debug("Working memory cleared")
    
    def clear_all(self) -> None:
        """Clear ALL memories (use with caution)"""
        with self.lock:
            self.working.clear()
            self.episodic.clear()
            self.semantic.facts.clear()
            self.semantic.entities.clear()
            self.semantic.relations.clear()
            self.procedural.patterns.clear()
            self.procedural.skills.clear()
            self.custom_stores.clear()
            self.stats = {k: 0 for k in self.stats}
        logger.warning("All memories cleared")
    
    def __repr__(self) -> str:
        return f"MemoryIndex(working={len(self.working)}, episodic={len(self.episodic)}, semantic={len(self.semantic.facts)}, procedural={len(self.procedural.patterns)})"
    
    def __len__(self) -> int:
        return self.size
    
    def __contains__(self, memory_id: str) -> bool:
        """Check if memory exists"""
        return self.get(memory_id) is not None


# Create placeholder modules for submodules
# These will be imported when needed
__all__ = [
    'MemoryIndex',
    'MemoryConfig',
    'MemoryItem',
    'MemoryStore',
    'WorkingMemory',
    'EpisodicMemory',
    'SemanticMemory',
    'ProceduralMemory',
    'MemoryType',
    'MemoryPriority',
]