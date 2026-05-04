"""
Experience Log - Long-term storage of personal experiences.
Like Wednesday's personal diary - each entry is a memory with context,
emotional valence, and importance. Used for learning from past interactions.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
from enum import Enum
from collections import defaultdict

# Configure logger
logger = logging.getLogger(__name__)


class EmotionalValence(Enum):
    """Emotional tone of a memory with Wednesday-appropriate values."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2
    DARK_HUMOR = 3  # Special for Wednesday - appreciates dark jokes
    SARCASTIC = 4   # For those perfectly timed sarcastic moments
    
    @classmethod
    def from_value(cls, value: int):
        """Get enum member from integer value."""
        for member in cls:
            if member.value == value:
                return member
        return cls.NEUTRAL


class MemorySignificance(Enum):
    """How important/significant a memory is (as float values)."""
    TRIVIAL = 0.2
    MUNDANE = 0.4
    NOTABLE = 0.6
    IMPORTANT = 0.8
    LIFE_CHANGING = 1.0
    
    @classmethod
    def from_float(cls, value: float):
        """Get nearest significance level from float value."""
        if value >= 0.9:
            return cls.LIFE_CHANGING
        elif value >= 0.7:
            return cls.IMPORTANT
        elif value >= 0.5:
            return cls.NOTABLE
        elif value >= 0.3:
            return cls.MUNDANE
        else:
            return cls.TRIVIAL


# Type alias for memory dictionary
MemoryEntry = Dict[str, Any]


class ExperienceLog:
    """
    Long-term storage of personal experiences - like a diary.
    Each entry is a timestamped memory with context, emotional valence,
    and importance. Used for building personality, learning patterns,
    and recalling past interactions.
    
    Features:
    - Persistent storage with JSON files
    - In-memory caching for fast access
    - Multiple indices for efficient querying
    - Embedding support for semantic similarity
    - Automatic cache pruning based on value
    """
    
    # Class constants for configuration
    DEFAULT_CACHE_SIZE = 1000
    RECENT_DAYS_DEFAULT = 7
    MAX_RECENT_LIMIT = 100
    SIMILARITY_LIMIT = 5
    CONSOLIDATION_DAYS = 30
    IMPORTANCE_ROUNDING = 1  # Decimal places for importance index
    
    def __init__(self, storage_path: Union[str, Path], auto_save: bool = True):
        """
        Initialize the experience log.
        
        Args:
            storage_path: Directory path for persistent storage
            auto_save: Whether to automatically save memories to disk
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save
        
        # In-memory cache (most recent + frequently accessed)
        self.memories: List[MemoryEntry] = []
        self.cache_size = self.DEFAULT_CACHE_SIZE
        self.access_counts: Dict[str, int] = {}  # Track recall frequency
        
        # Indices for fast lookup
        self.date_index: Dict[str, List[str]] = defaultdict(list)  # YYYY-MM-DD -> [memory_ids]
        self.importance_index: Dict[float, List[str]] = defaultdict(list)  # importance -> [memory_ids]
        self.emotion_index: Dict[str, List[str]] = defaultdict(list)  # emotion -> [memory_ids]
        self.tag_index: Dict[str, List[str]] = defaultdict(list)  # tag -> [memory_ids]
        self.user_index: Dict[str, List[str]] = defaultdict(list)  # user_id -> [memory_ids]
        
        # Search enhancements (will be populated by consolidation)
        self.embeddings: Dict[str, np.ndarray] = {}  # memory_id -> embedding
        
        # Load existing memories
        self._load_from_disk()
        logger.info(f"ExperienceLog initialized at {self.storage_path} with {len(self.memories)} memories")
    
    def add_memory(self, 
                   memory_data: Any, 
                   importance: float = 0.5,
                   emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL,
                   context: Optional[Dict] = None,
                   tags: Optional[List[str]] = None,
                   user_id: str = "anonymous") -> str:
        """
        Create a timestamped memory entry.
        
        Args:
            memory_data: The actual experience content
            importance: 0-1 how important this memory is
            emotional_valence: Emotional tone from EmotionalValence enum
            context: Additional context (location, conversation state, etc.)
            tags: Categorization tags
            user_id: Who was this experience with
        
        Returns:
            memory_id for future reference
        
        Raises:
            ValueError: If importance is outside valid range
        """
        # Validate importance
        if not 0 <= importance <= 1:
            raise ValueError(f"Importance must be between 0 and 1, got {importance}")
        
        # Generate unique ID
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        
        # Normalize tags
        normalized_tags = [tag.lower().strip() for tag in (tags or []) if tag.strip()]
        
        # Create memory entry
        memory = {
            'id': memory_id,
            'timestamp': timestamp.isoformat(),
            'datetime_obj': timestamp,  # For internal use only (not serialized)
            'data': memory_data,
            'importance': importance,
            'emotional_valence': emotional_valence.value,
            'emotional_label': emotional_valence.name,
            'context': context or {},
            'tags': normalized_tags,
            'user_id': user_id,
            'access_count': 0,
            'last_accessed': None,
            'consolidated': False,
            'has_embedding': False,  # Will be True when embedding is set
            'related_memories': []  # IDs of related memories
        }
        
        # Add to cache
        self.memories.append(memory)
        
        # Update indices
        self.date_index[date_str].append(memory_id)
        
        # Round importance for indexing (to group similar values)
        imp_key = round(importance, self.IMPORTANCE_ROUNDING)
        self.importance_index[imp_key].append(memory_id)
        
        self.emotion_index[emotional_valence.name].append(memory_id)
        
        for tag in normalized_tags:
            self.tag_index[tag].append(memory_id)
        
        self.user_index[user_id].append(memory_id)
        
        # Manage cache size
        if len(self.memories) > self.cache_size:
            self._prune_cache()
        
        # Persist if auto-save enabled
        if self.auto_save:
            self._save_memory(memory)
        
        logger.debug(f"Added memory {memory_id[:8]}: {str(memory_data)[:50]}...")
        
        return memory_id
    
    def recall(self, 
               query: Any, 
               limit: int = 5, 
               min_importance: float = 0.0,
               emotional_filter: Optional[List[EmotionalValence]] = None,
               timeframe_days: Optional[int] = None,
               user_filter: Optional[str] = None,
               tags: Optional[List[str]] = None) -> List[MemoryEntry]:
        """
        Find similar past experiences using semantic search.
        
        Args:
            query: What to search for (text, concept, or memory_id)
            limit: Maximum number of memories to return
            min_importance: Only return memories above this importance
            emotional_filter: Only return memories with these emotions
            timeframe_days: Only return memories from last N days
            user_filter: Only return memories with this user_id
            tags: Only return memories with these tags (any match)
        
        Returns:
            List of matching memories, sorted by relevance
        """
        # Convert query to string for initial filtering
        query_str = str(query).lower()
        
        # Determine timeframe cutoff if specified
        cutoff_date = None
        if timeframe_days is not None and timeframe_days > 0:
            cutoff_date = datetime.now() - timedelta(days=timeframe_days)
        
        # Prepare emotion filter set for O(1) lookup
        emotion_filter_set = None
        if emotional_filter:
            emotion_filter_set = {e.name for e in emotional_filter}
        
        # Prepare tag filter set
        tag_filter_set = {tag.lower().strip() for tag in (tags or []) if tag.strip()}
        
        # Collect candidate memories (using indices for efficiency)
        candidates = []
        
        # If we have embeddings and query is a memory_id, use semantic search
        if isinstance(query, str) and query in self.embeddings:
            return self.get_similar_memories(query, limit)
        
        # Otherwise, do filtered scan through memories
        for memory in self.memories:
            # Apply filters efficiently
            if not self._passes_filters(memory, min_importance, emotion_filter_set, 
                                       cutoff_date, user_filter, tag_filter_set):
                continue
            
            # Simple text matching (will be enhanced with embeddings)
            memory_str = self._get_memory_text(memory)
            if query_str in memory_str.lower():
                # Calculate relevance score
                relevance = self._calculate_relevance(query_str, memory_str)
                candidates.append((relevance, memory))
        
        # Sort by relevance
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Update access counts for retrieved memories
        retrieved = []
        for relevance, memory in candidates[:limit]:
            self._record_access(memory)
            # Add relevance score to returned memory
            memory_copy = memory.copy()
            memory_copy['relevance_score'] = relevance
            retrieved.append(memory_copy)
        
        logger.debug(f"Recalled {len(retrieved)} memories for query: {query_str[:50]}")
        return retrieved
    
    def _passes_filters(self, 
                        memory: MemoryEntry,
                        min_importance: float,
                        emotion_filter_set: Optional[set],
                        cutoff_date: Optional[datetime],
                        user_filter: Optional[str],
                        tag_filter_set: set) -> bool:
        """Check if a memory passes all filters."""
        # Importance filter
        if min_importance > 0 and memory['importance'] < min_importance:
            return False
        
        # Emotion filter
        if emotion_filter_set and memory['emotional_label'] not in emotion_filter_set:
            return False
        
        # Time filter
        if cutoff_date:
            mem_time = datetime.fromisoformat(memory['timestamp'])
            if mem_time < cutoff_date:
                return False
        
        # User filter
        if user_filter and memory['user_id'] != user_filter:
            return False
        
        # Tags filter (any match)
        if tag_filter_set and not (set(memory['tags']) & tag_filter_set):
            return False
        
        return True
    
    def _get_memory_text(self, memory: MemoryEntry) -> str:
        """Extract searchable text from memory."""
        texts = []
        
        # Add main data
        if memory.get('data'):
            texts.append(str(memory['data']))
        
        # Add context if available
        context = memory.get('context', {})
        for key, value in context.items():
            if isinstance(value, str):
                texts.append(value)
        
        # Add tags
        texts.extend(memory.get('tags', []))
        
        return ' '.join(texts)
    
    def recall_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Recall a specific memory by ID.
        
        Args:
            memory_id: ID of memory to retrieve
        
        Returns:
            Memory entry or None if not found
        """
        for memory in self.memories:
            if memory['id'] == memory_id:
                self._record_access(memory)
                return memory.copy()  # Return copy to prevent modification
        
        logger.warning(f"Memory {memory_id[:8]} not found")
        return None
    
    def _record_access(self, memory: MemoryEntry) -> None:
        """Record that a memory was accessed."""
        memory['access_count'] += 1
        memory['last_accessed'] = datetime.now().isoformat()
        self.access_counts[memory['id']] = memory['access_count']
    
    def recent(self, days: int = RECENT_DAYS_DEFAULT, limit: int = MAX_RECENT_LIMIT) -> List[MemoryEntry]:
        """
        Get memories from the last N days.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of memories to return
        
        Returns:
            List of recent memories, newest first
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        recent_memories = []
        for memory in self.memories:
            mem_time = datetime.fromisoformat(memory['timestamp'])
            if mem_time > cutoff:
                recent_memories.append(memory)
        
        # Sort by timestamp, newest first
        recent_memories.sort(key=lambda m: m['timestamp'], reverse=True)
        
        return recent_memories[:limit]
    
    def get_by_emotion(self, emotion: EmotionalValence, limit: int = 50) -> List[MemoryEntry]:
        """
        Get memories with specific emotional valence.
        
        Args:
            emotion: Emotional valence to filter by
            limit: Maximum number to return
        
        Returns:
            List of matching memories
        """
        memory_ids = self.emotion_index.get(emotion.name, [])
        
        memories = []
        for mid in memory_ids[:limit]:
            memory = self.recall_by_id(mid)
            if memory:
                memories.append(memory)
        
        return memories
    
    def get_by_importance(self, min_importance: float = 0.7, limit: int = 50) -> List[MemoryEntry]:
        """
        Get important memories above threshold.
        
        Args:
            min_importance: Minimum importance threshold
            limit: Maximum number to return
        
        Returns:
            List of important memories
        """
        important = []
        for memory in self.memories:
            if memory['importance'] >= min_importance:
                important.append(memory)
                if len(important) >= limit:
                    break
        
        return important
    
    def get_by_tag(self, tag: str, limit: int = 50) -> List[MemoryEntry]:
        """
        Get memories with a specific tag.
        
        Args:
            tag: Tag to search for
            limit: Maximum number to return
        
        Returns:
            List of matching memories
        """
        tag_lower = tag.lower().strip()
        memory_ids = self.tag_index.get(tag_lower, [])
        
        memories = []
        for mid in memory_ids[:limit]:
            memory = self.recall_by_id(mid)
            if memory:
                memories.append(memory)
        
        return memories
    
    def get_by_user(self, user_id: str, limit: int = 50) -> List[MemoryEntry]:
        """
        Get memories associated with a specific user.
        
        Args:
            user_id: User identifier
            limit: Maximum number to return
        
        Returns:
            List of matching memories
        """
        memory_ids = self.user_index.get(user_id, [])
        
        memories = []
        for mid in memory_ids[:limit]:
            memory = self.recall_by_id(mid)
            if memory:
                memories.append(memory)
        
        return memories
    
    def get_memories_for_date(self, date: str) -> List[MemoryEntry]:
        """
        Get all memories from a specific date.
        
        Args:
            date: Date string in YYYY-MM-DD format
        
        Returns:
            List of memories from that date
        """
        memory_ids = self.date_index.get(date, [])
        memories = []
        
        for mid in memory_ids:
            memory = self.recall_by_id(mid)
            if memory:
                memories.append(memory)
        
        # Sort by time within the day
        memories.sort(key=lambda m: m['timestamp'])
        
        return memories
    
    def add_related_memory(self, memory_id: str, related_id: str) -> bool:
        """
        Link two memories as related.
        
        Args:
            memory_id: ID of first memory
            related_id: ID of related memory
        
        Returns:
            True if successfully linked
        """
        # Find both memories
        memory = None
        related = None
        
        for m in self.memories:
            if m['id'] == memory_id:
                memory = m
            elif m['id'] == related_id:
                related = m
        
        if not memory or not related:
            logger.warning(f"Cannot link memories - one or both not found: {memory_id[:8]}, {related_id[:8]}")
            return False
        
        # Add links if not already present
        if related_id not in memory['related_memories']:
            memory['related_memories'].append(related_id)
        
        if memory_id not in related['related_memories']:
            related['related_memories'].append(memory_id)
        
        logger.debug(f"Linked memories: {memory_id[:8]} <-> {related_id[:8]}")
        return True
    
    def update_embedding(self, memory_id: str, embedding: np.ndarray) -> bool:
        """
        Update or set the embedding for a memory.
        
        Args:
            memory_id: ID of memory
            embedding: Numpy array embedding vector
        
        Returns:
            True if successful
        """
        memory = self.recall_by_id(memory_id)
        if memory:
            # Store embedding separately
            self.embeddings[memory_id] = embedding
            memory['has_embedding'] = True
            logger.debug(f"Updated embedding for memory {memory_id[:8]}")
            return True
        
        logger.warning(f"Cannot update embedding - memory not found: {memory_id[:8]}")
        return False
    
    def get_similar_memories(self, memory_id: str, limit: int = SIMILARITY_LIMIT) -> List[MemoryEntry]:
        """
        Find memories similar to a given memory using embeddings.
        
        Args:
            memory_id: ID of reference memory
            limit: Maximum number to return
        
        Returns:
            List of similar memories with similarity scores
        """
        if memory_id not in self.embeddings:
            logger.debug(f"No embedding for memory {memory_id[:8]}")
            return []
        
        query_embedding = self.embeddings[memory_id]
        similarities = []
        
        for mid, emb in self.embeddings.items():
            if mid != memory_id:
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, emb)
                similarities.append((similarity, mid))
        
        # Sort by similarity (highest first)
        similarities.sort(reverse=True)
        
        # Retrieve top memories
        similar_memories = []
        for sim, mid in similarities[:limit]:
            memory = self.recall_by_id(mid)
            if memory:
                memory['similarity_score'] = float(sim)
                similar_memories.append(memory)
        
        return similar_memories
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory (use with caution - Wednesday doesn't forget easily).
        
        Args:
            memory_id: ID of memory to delete
        
        Returns:
            True if successfully deleted
        """
        for i, memory in enumerate(self.memories):
            if memory['id'] == memory_id:
                # Remove from cache
                del self.memories[i]
                
                # Remove from indices
                self._remove_from_indices(memory)
                
                # Remove embedding if exists
                self.embeddings.pop(memory_id, None)
                
                # Remove file
                memory_file = self.storage_path / f"{memory_id}.json"
                if memory_file.exists():
                    memory_file.unlink()
                
                logger.info(f"Deleted memory {memory_id[:8]}")
                return True
        
        logger.warning(f"Attempted to delete non-existent memory: {memory_id[:8]}")
        return False
    
    def _remove_from_indices(self, memory: MemoryEntry) -> None:
        """Remove a memory from all indices."""
        memory_id = memory['id']
        
        # Date index
        date_str = memory['timestamp'][:10]
        if date_str in self.date_index and memory_id in self.date_index[date_str]:
            self.date_index[date_str].remove(memory_id)
        
        # Importance index
        imp_key = round(memory['importance'], self.IMPORTANCE_ROUNDING)
        if imp_key in self.importance_index and memory_id in self.importance_index[imp_key]:
            self.importance_index[imp_key].remove(memory_id)
        
        # Emotion index
        emotion_key = memory['emotional_label']
        if emotion_key in self.emotion_index and memory_id in self.emotion_index[emotion_key]:
            self.emotion_index[emotion_key].remove(memory_id)
        
        # Tag index
        for tag in memory['tags']:
            if tag in self.tag_index and memory_id in self.tag_index[tag]:
                self.tag_index[tag].remove(memory_id)
        
        # User index
        user_id = memory['user_id']
        if user_id in self.user_index and memory_id in self.user_index[user_id]:
            self.user_index[user_id].remove(memory_id)
    
    def consolidate_memories(self) -> Dict[str, int]:
        """
        Called by memory_consolidation.py to process memories.
        Marks old memories as consolidated for long-term storage.
        
        Returns:
            Statistics about consolidation
        """
        stats = {'processed': 0, 'marked_consolidated': 0}
        
        cutoff_date = datetime.now() - timedelta(days=self.CONSOLIDATION_DAYS)
        
        for memory in self.memories:
            mem_time = datetime.fromisoformat(memory['timestamp'])
            if mem_time < cutoff_date and not memory.get('consolidated', False):
                memory['consolidated'] = True
                stats['marked_consolidated'] += 1
            stats['processed'] += 1
        
        logger.info(f"Consolidation stats: {stats}")
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the experience log."""
        if not self.memories:
            return {'total_memories': 0}
        
        # Calculate emotion distribution
        emotions = defaultdict(int)
        for memory in self.memories:
            emotions[memory['emotional_label']] += 1
        
        # Calculate importance distribution
        importance_levels = defaultdict(int)
        for memory in self.memories:
            level = MemorySignificance.from_float(memory['importance']).name
            importance_levels[level] += 1
        
        # Get date range
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in self.memories]
        
        return {
            'total_memories': len(self.memories),
            'date_range': {
                'oldest': min(timestamps).isoformat(),
                'newest': max(timestamps).isoformat()
            },
            'avg_importance': float(np.mean([m['importance'] for m in self.memories])),
            'emotion_distribution': dict(emotions),
            'importance_distribution': dict(importance_levels),
            'total_accesses': sum(self.access_counts.values()),
            'memories_with_embeddings': len(self.embeddings),
            'unique_tags': len(self.tag_index),
            'unique_users': len(self.user_index)
        }
    
    def _calculate_relevance(self, query: str, memory_text: str) -> float:
        """
        Calculate simple relevance score for text matching.
        
        Args:
            query: Search query
            memory_text: Memory text to score
        
        Returns:
            Relevance score between 0 and 1
        """
        # Tokenize
        query_words = query.split()
        if not query_words:
            return 0.0
        
        memory_words = memory_text.lower().split()
        if not memory_words:
            return 0.0
        
        # Count query term occurrences
        occurrences = 0
        for word in query_words:
            occurrences += memory_text.lower().count(word)
        
        # Normalize by memory length
        base_score = occurrences / len(memory_words)
        
        # Bonus for exact phrase match
        if query in memory_text.lower():
            base_score *= 1.5
        
        # Bonus for recency (simulated - would use timestamp in production)
        # This is a placeholder for more sophisticated relevance scoring
        
        return min(1.0, base_score)
    
    def _save_memory(self, memory: MemoryEntry) -> None:
        """
        Save a single memory to disk.
        
        Args:
            memory: Memory entry to save
        """
        memory_file = self.storage_path / f"{memory['id']}.json"
        
        # Create a serializable copy (remove datetime_obj)
        save_copy = {}
        for key, value in memory.items():
            if key != 'datetime_obj':  # Skip internal datetime object
                if isinstance(value, (datetime, np.ndarray)):
                    # Convert non-serializable types
                    save_copy[key] = str(value)
                else:
                    save_copy[key] = value
        
        try:
            with open(memory_file, 'w') as f:
                json.dump(save_copy, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save memory {memory['id'][:8]}: {e}")
    
    def _load_from_disk(self) -> None:
        """Load memories from disk into cache."""
        memory_files = sorted(
            self.storage_path.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        loaded_count = 0
        for file in memory_files[:self.cache_size]:
            try:
                with open(file, 'r') as f:
                    memory = json.load(f)
                    
                    # Add datetime_obj for internal use
                    memory['datetime_obj'] = datetime.fromisoformat(memory['timestamp'])
                    
                    # Ensure all expected fields exist
                    memory.setdefault('tags', [])
                    memory.setdefault('context', {})
                    memory.setdefault('related_memories', [])
                    memory.setdefault('access_count', 0)
                    memory.setdefault('consolidated', False)
                    memory.setdefault('has_embedding', False)
                    
                    self.memories.append(memory)
                    
                    # Rebuild indices
                    self._add_to_indices(memory)
                    
                    loaded_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to load memory {file}: {e}")
        
        logger.info(f"Loaded {loaded_count} memories from disk")
    
    def _add_to_indices(self, memory: MemoryEntry) -> None:
        """Add a memory to all indices."""
        memory_id = memory['id']
        
        # Date index
        date_str = memory['timestamp'][:10]
        self.date_index[date_str].append(memory_id)
        
        # Importance index
        imp_key = round(memory['importance'], self.IMPORTANCE_ROUNDING)
        self.importance_index[imp_key].append(memory_id)
        
        # Emotion index
        self.emotion_index[memory['emotional_label']].append(memory_id)
        
        # Tag index
        for tag in memory['tags']:
            self.tag_index[tag].append(memory_id)
        
        # User index
        self.user_index[memory['user_id']].append(memory_id)
    
    def _prune_cache(self) -> None:
        """Prune least valuable memories from cache."""
        if len(self.memories) <= self.cache_size:
            return
        
        # Calculate value for each memory
        def memory_value(memory: MemoryEntry) -> float:
            """Calculate cache value based on recency, access, and importance."""
            age_days = (datetime.now() - datetime.fromisoformat(memory['timestamp'])).days
            access = memory['access_count']
            importance = memory['importance']
            
            # Value formula: recency factor * (access + 1) * importance
            recency_factor = 1.0 / (age_days + 1)  # +1 to avoid division by zero
            return recency_factor * (access + 1) * (importance + 0.5)
        
        # Sort by value and keep top ones
        self.memories.sort(key=memory_value, reverse=True)
        self.memories = self.memories[:self.cache_size]
        
        logger.debug(f"Pruned cache to {len(self.memories)} memories")
    
    def search_by_text(self, text: str, limit: int = 10) -> List[MemoryEntry]:
        """
        Simple text search through memories.
        
        Args:
            text: Text to search for
            limit: Maximum results
        
        Returns:
            List of matching memories
        """
        text_lower = text.lower()
        results = []
        
        for memory in self.memories:
            memory_text = self._get_memory_text(memory).lower()
            if text_lower in memory_text:
                results.append(memory)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_all_tags(self) -> List[str]:
        """Get list of all unique tags."""
        return list(self.tag_index.keys())
    
    def __len__(self) -> int:
        return len(self.memories)
    
    def __repr__(self) -> str:
        return f"ExperienceLog(memories={len(self.memories)}, path={self.storage_path})"