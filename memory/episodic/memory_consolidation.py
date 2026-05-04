"""
Memory Consolidation - Transforms short-term memories into long-term storage.
Processes experiences during "sleep" or idle periods to extract patterns,
build semantic knowledge, and strengthen important memories.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import logging
import numpy as np
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

# Type aliases for better code clarity
MemoryEpisode = Dict[str, Any]
Pattern = Dict[str, Any]
ConsolidationStats = Dict[str, Union[int, float, str]]

# Configure logger
logger = logging.getLogger(__name__)


class MemoryConsolidation:
    """
    Memory consolidation system that processes episodic memories during idle periods.
    Mimics human sleep consolidation - strengthens important memories,
    extracts patterns, and builds semantic knowledge.
    
    The consolidation process:
    1. Processes recent memories, updating their strength based on importance and age
    2. Extracts patterns across multiple memories (topics, sequences)
    3. Converts patterns to semantic knowledge in the knowledge base
    4. Performs memory replay to strengthen randomly selected memories
    """
    
    # Class constants for configuration defaults
    DEFAULT_CONFIG = {
        'strength_threshold': 0.7,  # Minimum strength for long-term storage
        'pattern_extraction_threshold': 3,  # Occurrences needed for pattern
        'consolidation_interval': 3600,  # Seconds between cycles (1 hour)
        'max_memories_per_cycle': 100,
        'replay_batch_size': 10,  # Memories to replay each cycle
        'forgetting_curve_factor': 0.5,  # How fast memories decay
        'emotional_boost': 0.3,  # Extra strength for emotional memories
        'sleep_cycle_threshold': 0.8,  # Strength for immediate consolidation
        'min_strength': 0.0,  # Minimum possible strength
        'max_strength': 1.0,  # Maximum possible strength
        'base_strength': 0.5,  # Default starting strength
        'pruning_threshold': 0.2,  # Below this, mark for pruning
        'strengthen_delta': 0.1,  # Minimum change to log as strengthened/weakened
        'replay_strengthen_amount': 0.05,  # How much replay strengthens
        'access_boost_max': 0.2,  # Maximum boost from access frequency
        'access_boost_per_access': 0.05,  # Boost per access
        'similarity_threshold': 0.5,  # Threshold for memory similarity
        'pattern_confidence_min': 0.6  # Minimum confidence for pattern extraction
    }
    
    def __init__(self, experience_log, knowledge_base,
                 config_path: Optional[Path] = None):
        """
        Initialize the memory consolidation system.
        
        Args:
            experience_log: Component that stores episodic memories
            knowledge_base: Component that stores semantic knowledge
            config_path: Optional path to JSON configuration file
        """
        # Store dependencies
        self.experience_log = experience_log
        self.knowledge_base = knowledge_base
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Consolidation statistics
        self.stats: Dict[str, Union[int, float, str, None]] = {
            'consolidation_cycles': 0,
            'memories_consolidated': 0,
            'patterns_extracted': 0,
            'semantic_facts_created': 0,
            'last_consolidation': None  # Will store ISO format string
        }
        
        # Memory strength tracking
        self.memory_strength: Dict[str, float] = {}  # episode_id -> current strength
        self.access_counts: Dict[str, int] = {}  # episode_id -> times accessed
        self.pruning_queue: List[str] = []  # episode_ids marked for pruning
        
        logger.info(f"MemoryConsolidation initialized with config: {self.config}")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """
        Load consolidation configuration from file or use defaults.
        
        Args:
            config_path: Path to JSON config file (optional)
        
        Returns:
            Configuration dictionary
        """
        config = self.DEFAULT_CONFIG.copy()
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded = json.load(f)
                    # Update only keys that exist in defaults
                    for key in loaded:
                        if key in config:
                            config[key] = loaded[key]
                        else:
                            logger.warning(f"Ignoring unknown config key: {key}")
                logger.info(f"Loaded consolidation config from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load consolidation config: {e}")
        
        return config
    
    def consolidate(self, force: bool = False) -> ConsolidationStats:
        """
        Run memory consolidation cycle.
        Should be called during idle periods or "sleep".
        
        Args:
            force: Force consolidation regardless of timing
        
        Returns:
            Dict with consolidation statistics from this cycle
        """
        # Check if it's time to consolidate
        if not force and self.stats['last_consolidation']:
            last = datetime.fromisoformat(str(self.stats['last_consolidation']))
            time_since_last = datetime.now() - last
            if time_since_last < timedelta(seconds=self.config['consolidation_interval']):
                logger.debug(f"Skipping consolidation - only {time_since_last.total_seconds():.0f}s since last cycle")
                return {'skipped': True, 'reason': 'too_soon'}
        
        # Initialize cycle statistics
        cycle_stats: ConsolidationStats = {
            'memories_processed': 0,
            'strengthened': 0,
            'weakened': 0,
            'patterns_found': 0,
            'semantic_added': 0,
            'memories_replayed': 0,
            'errors': []
        }
        
        try:
            # Step 1: Get recent memories for processing
            recent_memories = self.experience_log.recent(self.config['max_memories_per_cycle'])
            
            if not recent_memories:
                logger.debug("No memories to consolidate")
                cycle_stats['skipped'] = True
                cycle_stats['reason'] = 'no_memories'
                return cycle_stats
            
            logger.info(f"Starting consolidation cycle with {len(recent_memories)} memories")
            
            # Step 2: Process each memory (update strength)
            for memory in recent_memories:
                result = self._process_memory(memory)
                cycle_stats['memories_processed'] += 1
                if result.get('strengthened'):
                    cycle_stats['strengthened'] += 1
                if result.get('weakened'):
                    cycle_stats['weakened'] += 1
            
            # Step 3: Extract patterns across memories
            patterns = self._extract_patterns(recent_memories)
            cycle_stats['patterns_found'] = len(patterns)
            
            # Step 4: Convert patterns to semantic knowledge
            for pattern in patterns:
                if self._pattern_to_semantic(pattern):
                    cycle_stats['semantic_added'] += 1
            
            # Step 5: Memory replay - randomly access old memories to strengthen them
            replay_stats = self._memory_replay()
            cycle_stats['memories_replayed'] = replay_stats.get('replayed', 0)
            
            # Update permanent stats
            self.stats['consolidation_cycles'] = int(self.stats['consolidation_cycles']) + 1
            self.stats['memories_consolidated'] = int(self.stats['memories_consolidated']) + cycle_stats['memories_processed']
            self.stats['patterns_extracted'] = int(self.stats['patterns_extracted']) + cycle_stats['patterns_found']
            self.stats['semantic_facts_created'] = int(self.stats['semantic_facts_created']) + cycle_stats['semantic_added']
            self.stats['last_consolidation'] = datetime.now().isoformat()
            
            logger.info(f"Consolidation cycle complete: {cycle_stats}")
            
        except Exception as e:
            logger.error(f"Consolidation failed: {e}", exc_info=True)
            cycle_stats['errors'].append(str(e))
            cycle_stats['success'] = False
        
        cycle_stats['success'] = True
        return cycle_stats
    
    def _process_memory(self, memory: MemoryEpisode) -> Dict[str, bool]:
        """
        Process a single memory - update its strength based on factors.
        
        Args:
            memory: The memory episode to process
        
        Returns:
            Dict indicating what happened to the memory
        """
        # Ensure memory has required fields
        if 'id' not in memory:
            logger.warning("Memory missing 'id' field, skipping")
            return {'strengthened': False, 'weakened': False}
        
        episode_id = memory['id']
        result = {'strengthened': False, 'weakened': False}
        
        # Get or initialize memory strength
        old_strength = self.memory_strength.get(episode_id, self.config['base_strength'])
        
        # Calculate new strength based on multiple factors
        new_strength = self._calculate_memory_strength(memory, episode_id)
        
        # Store updated strength
        self.memory_strength[episode_id] = new_strength
        
        # Determine if significantly strengthened or weakened
        strength_change = new_strength - old_strength
        if strength_change > self.config['strengthen_delta']:
            result['strengthened'] = True
            logger.debug(f"Memory {episode_id[:8]} strengthened: {old_strength:.2f} -> {new_strength:.2f}")
        elif strength_change < -self.config['strengthen_delta']:
            result['weakened'] = True
            logger.debug(f"Memory {episode_id[:8]} weakened: {old_strength:.2f} -> {new_strength:.2f}")
        
        # If memory falls below threshold, mark for potential pruning
        if new_strength < self.config['pruning_threshold']:
            self._mark_for_pruning(episode_id)
        
        return result
    
    def _calculate_memory_strength(self, memory: MemoryEpisode, episode_id: str) -> float:
        """
        Calculate memory strength based on multiple factors:
        - Initial importance from metadata
        - Time decay (forgetting curve)
        - Emotional content boost
        - Access frequency boost
        
        Args:
            memory: The memory episode
            episode_id: ID of the memory
        
        Returns:
            Calculated strength between 0 and 1
        """
        # Factor 1: Base importance from metadata
        base_strength = self._calculate_initial_strength(memory)
        
        # Factor 2: Time decay (forgetting curve)
        time_decay = self._apply_forgetting_curve(memory)
        
        # Factor 3: Emotional boost
        emotional_boost = 0.0
        if memory.get('metadata', {}).get('emotional', False):
            emotional_boost = self.config['emotional_boost']
        
        # Factor 4: Access frequency boost
        access_count = self.access_counts.get(episode_id, 0)
        access_boost = min(
            self.config['access_boost_max'],
            access_count * self.config['access_boost_per_access']
        )
        
        # Combine factors
        # Base strength decays over time, then we add boosts
        new_strength = (base_strength * time_decay) + emotional_boost + access_boost
        
        # Clamp to valid range [min_strength, max_strength]
        return max(
            self.config['min_strength'],
            min(self.config['max_strength'], new_strength)
        )
    
    def _calculate_initial_strength(self, memory: MemoryEpisode) -> float:
        """
        Calculate initial memory strength based on metadata.
        
        Args:
            memory: The memory episode
        
        Returns:
            Base strength between 0 and 1
        """
        strength = self.config['base_strength']
        metadata = memory.get('metadata', {})
        
        # Boost based on explicit importance if provided
        if 'importance' in metadata:
            try:
                importance = float(metadata['importance'])
                # Importance expected to be between 0 and 1
                strength += importance * 0.3
            except (ValueError, TypeError):
                logger.debug(f"Invalid importance value in memory {memory.get('id', 'unknown')[:8]}")
        
        # Boost for user-initiated interactions
        if metadata.get('user_said', False):
            strength += 0.1
        
        # Boost for novelty
        if metadata.get('novel', False):
            strength += 0.2
        
        return min(self.config['max_strength'], strength)
    
    def _apply_forgetting_curve(self, memory: MemoryEpisode) -> float:
        """
        Apply Ebbinghaus forgetting curve.
        Memories decay over time unless reinforced.
        
        Args:
            memory: The memory episode
        
        Returns:
            Decay factor between min_strength and 1.0
        """
        # Parse timestamp
        try:
            timestamp_str = memory.get('timestamp')
            if not timestamp_str:
                return 1.0  # No timestamp = no decay
            
            timestamp = datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            logger.warning(f"Invalid timestamp in memory {memory.get('id', 'unknown')[:8]}")
            return 1.0
        
        # Calculate age in hours
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        
        # Forgetting curve: R = e^(-t/S) where S is strength factor
        # S = 1/forgetting_curve_factor gives us control over decay rate
        decay = np.exp(-age_hours * self.config['forgetting_curve_factor'])
        
        # Don't decay below base level (memories aren't completely forgotten)
        return max(self.config['min_strength'], decay)
    
    def _extract_patterns(self, memories: List[MemoryEpisode]) -> List[Pattern]:
        """
        Extract patterns across multiple memories.
        Looks for recurring themes, topics, or sequences.
        
        Args:
            memories: List of memory episodes
        
        Returns:
            List of extracted patterns
        """
        patterns = []
        
        if len(memories) < self.config['pattern_extraction_threshold']:
            return patterns
        
        # Pattern 1: Topic-based patterns
        topic_patterns = self._extract_topic_patterns(memories)
        patterns.extend(topic_patterns)
        
        # Pattern 2: Sequential patterns
        sequential_patterns = self._find_sequential_patterns(memories)
        patterns.extend(sequential_patterns)
        
        # Pattern 3: Content-based similarity patterns
        similarity_patterns = self._find_similarity_patterns(memories)
        patterns.extend(similarity_patterns)
        
        return patterns
    
    def _extract_topic_patterns(self, memories: List[MemoryEpisode]) -> List[Pattern]:
        """Extract patterns grouped by topic."""
        patterns = []
        
        # Group memories by topic
        topics = defaultdict(list)
        for memory in memories:
            topic = memory.get('metadata', {}).get('topic', 'general')
            topics[topic].append(memory)
        
        # Find patterns in each topic group
        for topic, topic_memories in topics.items():
            if len(topic_memories) >= self.config['pattern_extraction_threshold']:
                pattern = self._analyze_topic_pattern(topic, topic_memories)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_topic_pattern(self, topic: str, memories: List[MemoryEpisode]) -> Optional[Pattern]:
        """
        Analyze memories in a topic for recurring patterns.
        
        Args:
            topic: Topic name
            memories: Memories with this topic
        
        Returns:
            Pattern dict or None if no pattern found
        """
        if len(memories) < 2:
            return None
        
        # Extract content from memories
        contents = []
        for memory in memories:
            # Try different possible content fields
            content = memory.get('experience') or memory.get('content') or ''
            contents.append(str(content))
        
        # Find common words (simplified - would use NLP in production)
        word_counts = Counter()
        for content in contents:
            # Simple word tokenization
            words = content.lower().split()
            # Remove common stop words (very simplified)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            words = [w for w in words if w not in stop_words and len(w) > 3]
            word_counts.update(words[:20])  # First 20 words
        
        # Find words that appear in most memories
        threshold = len(memories) * 0.7
        common_words = [
            word for word, count in word_counts.items() 
            if count >= threshold
        ]
        
        if common_words:
            return {
                'type': 'topic_pattern',
                'topic': topic,
                'common_elements': common_words,
                'frequency': len(memories),
                'confidence': min(1.0, len(memories) / 10),
                'sample_memories': [m['id'] for m in memories[:3] if 'id' in m]
            }
        
        return None
    
    def _find_sequential_patterns(self, memories: List[MemoryEpisode]) -> List[Pattern]:
        """
        Find patterns in sequences of memories.
        Looks for repeating sequences of events.
        
        Args:
            memories: List of memory episodes
        
        Returns:
            List of sequential patterns
        """
        patterns = []
        
        # Sort by timestamp
        try:
            sorted_memories = sorted(
                memories,
                key=lambda m: datetime.fromisoformat(m.get('timestamp', '2000-01-01'))
            )
        except (ValueError, TypeError):
            logger.warning("Could not sort memories by timestamp for sequential pattern detection")
            return patterns
        
        # Look for repeated sequences (simplified bigram detection)
        # In production, would use more sophisticated sequence mining
        for i in range(len(sorted_memories) - 1):
            current = sorted_memories[i]
            next_mem = sorted_memories[i + 1]
            
            # Skip if missing IDs
            if 'id' not in current or 'id' not in next_mem:
                continue
            
            # Check if this pair appears elsewhere
            occurrences = 1
            for j in range(i + 2, len(sorted_memories) - 1):
                if j + 1 >= len(sorted_memories):
                    break
                    
                if (self._memories_similar(current, sorted_memories[j]) and
                    self._memories_similar(next_mem, sorted_memories[j + 1])):
                    occurrences += 1
            
            if occurrences >= 2:
                patterns.append({
                    'type': 'sequential',
                    'pattern': [current['id'], next_mem['id']],
                    'confidence': min(0.9, occurrences * 0.3),
                    'occurrences': occurrences
                })
        
        return patterns
    
    def _find_similarity_patterns(self, memories: List[MemoryEpisode]) -> List[Pattern]:
        """Find patterns based on content similarity."""
        patterns = []
        
        # Group similar memories
        similar_groups = defaultdict(list)
        
        for i, memory in enumerate(memories):
            if 'id' not in memory:
                continue
                
            # Find a group for this memory
            placed = False
            for group_id in similar_groups:
                # Check similarity with first memory in group
                if self._memories_similar(memory, similar_groups[group_id][0]):
                    similar_groups[group_id].append(memory)
                    placed = True
                    break
            
            if not placed:
                # Create new group with memory ID as key
                similar_groups[memory['id']].append(memory)
        
        # Create patterns from groups with multiple memories
        for group_id, group_memories in similar_groups.items():
            if len(group_memories) >= self.config['pattern_extraction_threshold']:
                patterns.append({
                    'type': 'similarity_cluster',
                    'cluster_id': group_id,
                    'memory_count': len(group_memories),
                    'confidence': min(0.8, len(group_memories) * 0.2),
                    'sample_memories': [m['id'] for m in group_memories[:3] if 'id' in m]
                })
        
        return patterns
    
    def _memories_similar(self, m1: MemoryEpisode, m2: MemoryEpisode) -> bool:
        """
        Check if two memories are similar.
        In production, would use embeddings or more sophisticated NLP.
        
        Args:
            m1: First memory
            m2: Second memory
        
        Returns:
            True if memories are similar
        """
        # Extract content from both memories
        content1 = str(m1.get('experience') or m1.get('content') or '')
        content2 = str(m2.get('experience') or m2.get('content') or '')
        
        if not content1 or not content2:
            return False
        
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'is', 'was', 'were', 'have', 'has', 'had'}
        words1 = {w for w in words1 if w not in stop_words}
        words2 = {w for w in words2 if w not in stop_words}
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0
        return similarity > self.config['similarity_threshold']
    
    def _pattern_to_semantic(self, pattern: Pattern) -> bool:
        """
        Convert an extracted pattern to semantic knowledge.
        
        Args:
            pattern: Extracted pattern
        
        Returns:
            True if successfully added to knowledge base
        """
        # Check confidence threshold
        if pattern.get('confidence', 0) < self.config['pattern_confidence_min']:
            logger.debug(f"Pattern confidence too low: {pattern.get('confidence', 0)}")
            return False
        
        # Create semantic fact from pattern based on type
        if pattern['type'] == 'topic_pattern':
            fact = f"Common topics in {pattern['topic']}: {', '.join(pattern['common_elements'])}"
        elif pattern['type'] == 'sequential':
            fact = f"Sequence pattern detected with {pattern['occurrences']} occurrences"
        elif pattern['type'] == 'similarity_cluster':
            fact = f"Cluster of {pattern['memory_count']} similar memories found"
        else:
            fact = f"Pattern: {pattern}"
        
        # Prepare metadata
        metadata = {
            'source': 'consolidation',
            'pattern_type': pattern['type'],
            'confidence': pattern['confidence'],
            'frequency': pattern.get('frequency', pattern.get('occurrences', 1)),
            'consolidated_at': datetime.now().isoformat()
        }
        
        # Add to knowledge base
        try:
            result = self.knowledge_base.add_knowledge(fact, metadata)
            if result:
                logger.debug(f"Added semantic fact from {pattern['type']} pattern")
            return result
        except Exception as e:
            logger.error(f"Failed to add pattern to knowledge base: {e}")
            return False
    
    def _memory_replay(self) -> Dict[str, int]:
        """
        Randomly access old memories to strengthen them.
        Mimics memory replay during sleep.
        
        Returns:
            Statistics about replay operation
        """
        stats = {'replayed': 0, 'strengthened_from_replay': 0}
        
        # Get all memory IDs with current strengths
        all_memories = list(self.memory_strength.items())
        if not all_memories:
            return stats
        
        # Randomly select memories for replay
        replay_count = min(self.config['replay_batch_size'], len(all_memories))
        selected = random.sample(all_memories, replay_count)
        
        for episode_id, strength in selected:
            # Simulate accessing the memory
            self.access_counts[episode_id] = self.access_counts.get(episode_id, 0) + 1
            stats['replayed'] += 1
            
            # Strengthen replayed memories
            new_strength = min(
                self.config['max_strength'],
                strength + self.config['replay_strengthen_amount']
            )
            self.memory_strength[episode_id] = new_strength
            stats['strengthened_from_replay'] += 1
        
        if stats['replayed'] > 0:
            logger.debug(f"Memory replay: strengthened {stats['strengthened_from_replay']} memories")
        
        return stats
    
    def _mark_for_pruning(self, episode_id: str) -> None:
        """
        Mark a weak memory for potential pruning.
        
        Args:
            episode_id: ID of memory to mark
        """
        if episode_id not in self.pruning_queue:
            self.pruning_queue.append(episode_id)
            logger.debug(f"Memory {episode_id[:8]} marked for pruning (strength below {self.config['pruning_threshold']})")
    
    def prune_weak_memories(self, threshold: Optional[float] = None) -> int:
        """
        Remove or archive memories below strength threshold.
        
        Args:
            threshold: Strength threshold (uses config.pruning_threshold if None)
        
        Returns:
            Number of memories pruned
        """
        prune_threshold = threshold or self.config['pruning_threshold']
        
        # Find memories below threshold
        to_prune = [
            ep_id for ep_id, strength in self.memory_strength.items()
            if strength < prune_threshold
        ]
        
        # In production, would archive or delete these memories
        for ep_id in to_prune:
            logger.info(f"Pruning weak memory: {ep_id[:8]} (strength: {self.memory_strength[ep_id]:.2f})")
            # Remove from tracking
            self.memory_strength.pop(ep_id, None)
            self.access_counts.pop(ep_id, None)
        
        # Clear from pruning queue
        self.pruning_queue = [ep_id for ep_id in self.pruning_queue if ep_id not in to_prune]
        
        return len(to_prune)
    
    def get_memory_strength(self, episode_id: str) -> float:
        """
        Get current strength of a memory.
        
        Args:
            episode_id: ID of the memory
        
        Returns:
            Current strength (0-1) or base_strength if not tracked
        """
        return self.memory_strength.get(episode_id, self.config['base_strength'])
    
    def record_access(self, episode_id: str) -> None:
        """
        Record that a memory was accessed.
        
        Args:
            episode_id: ID of accessed memory
        """
        self.access_counts[episode_id] = self.access_counts.get(episode_id, 0) + 1
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """
        Get consolidation system statistics.
        
        Returns:
            Dictionary with current stats
        """
        # Calculate average strength safely
        avg_strength = 0.0
        if self.memory_strength:
            avg_strength = float(np.mean(list(self.memory_strength.values())))
        
        return {
            **self.stats,
            'tracked_memories': len(self.memory_strength),
            'pruning_queue_size': len(self.pruning_queue),
            'average_strength': round(avg_strength, 3),
            'total_accesses': sum(self.access_counts.values()),
            'config': self.config
        }
    
    def reset_stats(self) -> None:
        """Reset consolidation statistics (for testing/debugging)."""
        self.stats = {
            'consolidation_cycles': 0,
            'memories_consolidated': 0,
            'patterns_extracted': 0,
            'semantic_facts_created': 0,
            'last_consolidation': None
        }
        logger.info("Consolidation statistics reset")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"MemoryConsolidation(cycles={self.stats['consolidation_cycles']}, "
                f"tracked={len(self.memory_strength)}, "
                f"pruning={len(self.pruning_queue)})")