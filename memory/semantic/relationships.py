"""
Relationships - Manages connections between entities and concepts in semantic memory.
Tracks how things relate to each other with types, strength, and temporal aspects.
Enables Wednesday to understand complex relationship networks and navigate them.
"""
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
import uuid
import json
from pathlib import Path
import logging
from enum import Enum
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

class RelationCategory(Enum):
    """High-level categories of relationships"""
    HIERARCHICAL = "hierarchical"  # is_a, part_of, has_a
    SOCIAL = "social"  # friend_of, works_with, knows
    SPATIAL = "spatial"  # located_in, near, far_from
    TEMPORAL = "temporal"  # before, after, during
    CAUSAL = "causal"  # causes, leads_to, prevents
    EMOTIONAL = "emotional"  # loves, hates, fears
    FUNCTIONAL = "functional"  # used_for, capable_of
    COMPARATIVE = "comparative"  # similar_to, opposite_of
    OWNERSHIP = "ownership"  # owns, belongs_to
    INTERACTION = "interaction"  # talked_to, met, influenced

class RelationDirection(Enum):
    """Directionality of the relationship"""
    UNIDIRECTIONAL = "unidirectional"  # A -> B only
    BIDIRECTIONAL = "bidirectional"    # A <-> B
    SYMMETRIC = "symmetric"            # A and B are interchangeable

class Relationship:
    """
    Represents a single relationship between two entities/concepts.
    Contains metadata about type, strength, and temporal aspects.
    """
    
    def __init__(self, 
                 source_id: str,
                 target_id: str,
                 relation_type: str,
                 category: RelationCategory,
                 strength: float = 0.5,
                 direction: RelationDirection = RelationDirection.UNIDIRECTIONAL,
                 confidence: float = 1.0,
                 metadata: Optional[Dict] = None):
        
        self.id = str(uuid.uuid4())
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.category = category
        self.strength = min(1.0, max(0.0, strength))
        self.direction = direction
        self.confidence = confidence
        self.metadata = metadata or {}
        
        # Temporal tracking
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.last_accessed = self.created_at
        self.expires_at = None
        
        # Usage statistics
        self.access_count = 0
        self.reinforcement_count = 0  # How many times this relation has been reinforced
        
        # Context
        self.context_tags = []  # Situations where this relation applies
        self.exceptions = []    # Cases where this relation doesn't hold
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type,
            'category': self.category.value,
            'strength': self.strength,
            'direction': self.direction.value,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'access_count': self.access_count,
            'reinforcement_count': self.reinforcement_count,
            'context_tags': self.context_tags,
            'exceptions': self.exceptions
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Relationship':
        """Create a Relationship from dictionary data"""
        rel = cls(
            source_id=data['source_id'],
            target_id=data['target_id'],
            relation_type=data['relation_type'],
            category=RelationCategory(data['category']),
            strength=data['strength'],
            direction=RelationDirection(data['direction']),
            confidence=data['confidence'],
            metadata=data.get('metadata', {})
        )
        rel.id = data['id']
        rel.created_at = datetime.fromisoformat(data['created_at'])
        rel.updated_at = datetime.fromisoformat(data['updated_at'])
        rel.last_accessed = datetime.fromisoformat(data['last_accessed'])
        rel.expires_at = datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None
        rel.access_count = data.get('access_count', 0)
        rel.reinforcement_count = data.get('reinforcement_count', 0)
        rel.context_tags = data.get('context_tags', [])
        rel.exceptions = data.get('exceptions', [])
        return rel
    
    def reinforce(self, amount: float = 0.1) -> None:
        """Strengthen the relationship through reinforcement"""
        self.strength = min(1.0, self.strength + amount)
        self.reinforcement_count += 1
        self.updated_at = datetime.now()
        logger.debug(f"Reinforced relation {self.id[:8]}: strength now {self.strength:.2f}")
    
    def weaken(self, amount: float = 0.1) -> None:
        """Weaken the relationship through disuse or contradiction"""
        self.strength = max(0.0, self.strength - amount)
        self.updated_at = datetime.now()
        logger.debug(f"Weakened relation {self.id[:8]}: strength now {self.strength:.2f}")
    
    def access(self) -> None:
        """Record an access to this relationship"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def is_active(self) -> bool:
        """Check if the relationship is still active (not expired)"""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return self.strength > 0.2  # Minimum strength threshold
    
    def applies_in_context(self, context: str) -> bool:
        """Check if this relation applies in a given context"""
        if not self.context_tags:
            return True
        return context in self.context_tags
    
    def __repr__(self) -> str:
        return (f"Relationship({self.source_id[:8]} -> {self.target_id[:8]}, "
                f"type={self.relation_type}, strength={self.strength:.2f})")

class RelationshipManager:
    """
    Manages all relationships in semantic memory.
    Provides querying, inference, and maintenance of relationship networks.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data/semantic/relationships")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core storage
        self.relationships: Dict[str, Relationship] = {}  # rel_id -> Relationship
        
        # Indexes for fast lookup
        self.source_index: Dict[str, Set[str]] = defaultdict(set)  # source_id -> rel_ids
        self.target_index: Dict[str, Set[str]] = defaultdict(set)  # target_id -> rel_ids
        self.type_index: Dict[str, Set[str]] = defaultdict(set)  # relation_type -> rel_ids
        self.category_index: Dict[str, Set[str]] = defaultdict(set)  # category -> rel_ids
        
        # Graph representation for path finding
        # Format: source -> {target: [(rel_id, strength)]}
        self.graph: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Statistics
        self.stats = {
            'total_relationships': 0,
            'by_category': defaultdict(int),
            'by_type': defaultdict(int),
            'total_accesses': 0
        }
        
        self._load_from_disk()
        logger.info(f"RelationshipManager initialized with {len(self.relationships)} relationships")
    
    def add_relationship(self,
                        source_id: str,
                        target_id: str,
                        relation_type: str,
                        category: RelationCategory,
                        strength: float = 0.5,
                        direction: RelationDirection = RelationDirection.UNIDIRECTIONAL,
                        confidence: float = 1.0,
                        metadata: Optional[Dict] = None,
                        bidirectional: bool = False) -> Tuple[str, Optional[str]]:
        """
        Add a new relationship between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relationship (e.g., "friend_of", "part_of")
            category: High-level category
            strength: How strong the relationship is (0-1)
            direction: Directionality of the relationship
            confidence: Confidence in this relationship
            metadata: Additional metadata
            bidirectional: Also add reverse relationship
        
        Returns:
            Tuple of (primary_relation_id, reverse_relation_id or None)
        """
        # Check if relationship already exists
        existing = self.find_relationship(source_id, target_id, relation_type)
        if existing:
            # Strengthen existing relationship
            existing.reinforce(0.05)  # Smaller increment for reinforcement
            self._save_relationship(existing.id)
            logger.debug(f"Strengthened existing relationship: {relation_type}")
            return existing.id, None
        
        # Create new relationship
        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            category=category,
            strength=strength,
            direction=direction,
            confidence=confidence,
            metadata=metadata
        )
        
        self.relationships[rel.id] = rel
        
        # Update indexes
        self.source_index[source_id].add(rel.id)
        self.target_index[target_id].add(rel.id)
        self.type_index[relation_type].add(rel.id)
        self.category_index[category.value].add(rel.id)
        
        # Update graph
        self.graph[source_id][target_id].append((rel.id, strength))
        
        # Update statistics
        self.stats['total_relationships'] = len(self.relationships)
        self.stats['by_category'][category.value] += 1
        self.stats['by_type'][relation_type] += 1
        
        # Save
        self._save_relationship(rel.id)
        
        reverse_id = None
        if bidirectional:
            reverse_type = self._get_reverse_type(relation_type)
            # Only add reverse if it doesn't exist
            reverse_existing = self.find_relationship(target_id, source_id, reverse_type)
            if not reverse_existing:
                reverse_id = self.add_relationship(
                    source_id=target_id,
                    target_id=source_id,
                    relation_type=reverse_type,
                    category=category,
                    strength=strength,
                    direction=RelationDirection.BIDIRECTIONAL,
                    confidence=confidence,
                    metadata={'reverse_of': rel.id}
                )[0]  # Get the primary ID from the tuple
        
        logger.info(f"Added relationship: {source_id[:8]} {relation_type} {target_id[:8]}")
        return rel.id, reverse_id
    
    def find_relationship(self, source_id: str, target_id: str, 
                         relation_type: Optional[str] = None) -> Optional[Relationship]:
        """Find a specific relationship between two entities"""
        # Get all relationships from source
        source_rels = self.source_index.get(source_id, set())
        
        for rel_id in source_rels:
            rel = self.relationships[rel_id]
            if rel.target_id == target_id:
                if relation_type is None or rel.relation_type == relation_type:
                    rel.access()
                    # Update stats
                    self.stats['total_accesses'] += 1
                    return rel
        
        return None
    
    def get_relationships(self, 
                         entity_id: Optional[str] = None,
                         relation_type: Optional[str] = None,
                         category: Optional[RelationCategory] = None,
                         direction: str = "outgoing",  # "outgoing", "incoming", "both"
                         min_strength: float = 0.0,
                         active_only: bool = True) -> List[Relationship]:
        """
        Get all relationships matching criteria.
        
        Args:
            entity_id: Filter by entity (as source or target)
            relation_type: Filter by relationship type
            category: Filter by category
            direction: Which direction relative to entity
            min_strength: Minimum strength threshold
            active_only: Only return active (non-expired) relationships
        """
        results = set()
        
        if entity_id:
            if direction in ["outgoing", "both"]:
                results.update(self.source_index.get(entity_id, set()))
            if direction in ["incoming", "both"]:
                results.update(self.target_index.get(entity_id, set()))
        else:
            results = set(self.relationships.keys())
        
        # Apply type filter
        if relation_type:
            results &= self.type_index.get(relation_type, set())
        
        # Apply category filter
        if category:
            results &= self.category_index.get(category.value, set())
        
        # Convert to Relationship objects and apply remaining filters
        relationships = []
        for rel_id in results:
            rel = self.relationships[rel_id]
            
            if rel.strength < min_strength:
                continue
            
            if active_only and not rel.is_active():
                continue
            
            rel.access()
            self.stats['total_accesses'] += 1
            relationships.append(rel)
        
        # Sort by strength (strongest first)
        relationships.sort(key=lambda r: r.strength, reverse=True)
        
        return relationships
    
    def find_path(self, start_id: str, end_id: str, 
                 max_depth: int = 5,
                 relation_types: Optional[List[str]] = None) -> Optional[List[Dict]]:
        """
        Find a path between two entities through the relationship graph.
        Uses BFS to find shortest path.
        
        Args:
            start_id: Starting entity
            end_id: Target entity
            max_depth: Maximum path length
            relation_types: Allowed relationship types
        
        Returns:
            List of steps in the path, or None if no path found
        """
        if start_id not in self.graph:
            return None
        
        # BFS queue: (entity_id, path, visited)
        queue = [(start_id, [], {start_id})]
        
        while queue:
            current_id, path, visited = queue.pop(0)
            
            if len(path) >= max_depth:
                continue
            
            # Explore outgoing relationships
            for target_id, rels in self.graph.get(current_id, {}).items():
                if target_id in visited:
                    continue
                
                # Check relationship type filter
                valid_rels = []
                if relation_types:
                    for rel_id, strength in rels:
                        rel = self.relationships[rel_id]
                        if rel.relation_type in relation_types:
                            valid_rels.append((rel_id, strength, rel))
                else:
                    # Use all relationships, get the strongest
                    strongest = max(rels, key=lambda x: x[1])
                    rel_id, strength = strongest
                    rel = self.relationships[rel_id]
                    valid_rels = [(rel_id, strength, rel)]
                
                if not valid_rels:
                    continue
                
                # For path finding, use the strongest relationship
                strongest_rel = max(valid_rels, key=lambda x: x[1])
                rel_id, strength, rel = strongest_rel
                
                new_path = path + [{
                    'from': current_id,
                    'to': target_id,
                    'relation_type': rel.relation_type,
                    'strength': strength,
                    'rel_id': rel_id
                }]
                
                if target_id == end_id:
                    return new_path
                
                new_visited = visited | {target_id}
                queue.append((target_id, new_path, new_visited))
        
        return None  # No path found
    
    def get_connected_component(self, entity_id: str, 
                               max_depth: int = 3,
                               min_strength: float = 0.3) -> Dict[int, Set[str]]:
        """
        Get all entities connected to the given entity within max_depth.
        Returns a dictionary mapping depth to sets of entity IDs.
        """
        component = defaultdict(set)
        visited = {entity_id}
        queue = [(entity_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get all neighbors (both outgoing and incoming)
            neighbors = set()
            
            # Outgoing
            for target_id, rels in self.graph.get(current_id, {}).items():
                # Check strength
                for rel_id, strength in rels:
                    if strength >= min_strength:
                        neighbors.add(target_id)
                        break
            
            # Incoming (need to search graph for edges pointing to current)
            for source_id, targets in self.graph.items():
                if current_id in targets:
                    for rel_id, strength in targets[current_id]:
                        if strength >= min_strength:
                            neighbors.add(source_id)
                            break
            
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    component[depth + 1].add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
        
        return dict(component)
    
    def reinforce_path(self, path: List[Dict], amount: float = 0.05) -> None:
        """Reinforce all relationships along a path"""
        for step in path:
            rel_id = step.get('rel_id')
            if rel_id and rel_id in self.relationships:
                self.relationships[rel_id].reinforce(amount)
                self._save_relationship(rel_id)
    
    def merge_entities(self, keep_id: str, merge_id: str) -> bool:
        """Merge relationships when two entities are merged"""
        if keep_id == merge_id:
            return False
        
        # Get all relationships involving merge_id
        outgoing = self.source_index.get(merge_id, set()).copy()
        incoming = self.target_index.get(merge_id, set()).copy()
        
        # Transfer outgoing relationships
        for rel_id in outgoing:
            rel = self.relationships[rel_id]
            # Check if similar relationship already exists from keep_id
            existing = self.find_relationship(keep_id, rel.target_id, rel.relation_type)
            if existing:
                # Merge strengths
                existing.reinforce(rel.strength * 0.3)
                self._delete_relationship(rel_id)
            else:
                rel.source_id = keep_id
                self.source_index[keep_id].add(rel_id)
                self._save_relationship(rel_id)
        
        # Transfer incoming relationships
        for rel_id in incoming:
            rel = self.relationships[rel_id]
            # Check if similar relationship already exists to keep_id
            existing = self.find_relationship(rel.source_id, keep_id, rel.relation_type)
            if existing:
                # Merge strengths
                existing.reinforce(rel.strength * 0.3)
                self._delete_relationship(rel_id)
            else:
                rel.target_id = keep_id
                self.target_index[keep_id].add(rel_id)
                self._save_relationship(rel_id)
        
        # Remove old indexes
        if merge_id in self.source_index:
            del self.source_index[merge_id]
        if merge_id in self.target_index:
            del self.target_index[merge_id]
        
        # Clean up graph
        if merge_id in self.graph:
            del self.graph[merge_id]
        for source in list(self.graph.keys()):
            if merge_id in self.graph[source]:
                del self.graph[source][merge_id]
        
        # Update stats
        self.stats['total_relationships'] = len(self.relationships)
        
        logger.info(f"Merged relationships from {merge_id[:8]} into {keep_id[:8]}")
        return True
    
    def prune_expired(self, days_threshold: int = 30) -> int:
        """Remove expired and very weak relationships"""
        pruned = 0
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        
        rel_ids = list(self.relationships.keys())
        for rel_id in rel_ids:
            rel = self.relationships[rel_id]
            
            # Check expiration
            if rel.expires_at and datetime.now() > rel.expires_at:
                self._delete_relationship(rel_id)
                pruned += 1
                continue
            
            # Check last accessed
            if rel.last_accessed < threshold_date and rel.strength < 0.3:
                self._delete_relationship(rel_id)
                pruned += 1
                continue
        
        if pruned > 0:
            self.stats['total_relationships'] = len(self.relationships)
            logger.info(f"Pruned {pruned} expired/weak relationships")
        
        return pruned
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the relationship network"""
        if not self.relationships:
            return {
                'total_relationships': 0,
                'unique_sources': 0,
                'unique_targets': 0,
                'unique_types': 0,
                'avg_strength': 0.0,
                'total_accesses': 0,
                'by_category': {},
                'by_type': {}
            }
        
        strengths = [r.strength for r in self.relationships.values()]
        
        # Calculate standard deviation safely
        try:
            std_strength = statistics.stdev(strengths) if len(strengths) > 1 else 0.0
        except statistics.StatisticsError:
            std_strength = 0.0
        
        return {
            'total_relationships': len(self.relationships),
            'unique_sources': len(self.source_index),
            'unique_targets': len(self.target_index),
            'unique_types': len(self.type_index),
            'avg_strength': statistics.mean(strengths),
            'std_strength': std_strength,
            'max_strength': max(strengths),
            'min_strength': min(strengths),
            'total_accesses': self.stats['total_accesses'],
            'by_category': dict(self.stats['by_category']),
            'by_type': dict(self.stats['by_type'])
        }
    
    def _get_reverse_type(self, relation_type: str) -> str:
        """Get the reverse relationship type"""
        reverse_map = {
            'parent_of': 'child_of',
            'child_of': 'parent_of',
            'part_of': 'contains',
            'contains': 'part_of',
            'located_in': 'location_of',
            'location_of': 'located_in',
            'owns': 'owned_by',
            'owned_by': 'owns',
            'friend_of': 'friend_of',  # Symmetric
            'similar_to': 'similar_to',  # Symmetric
            'opposite_of': 'opposite_of',  # Symmetric
            'causes': 'caused_by',
            'caused_by': 'causes',
            'knows': 'known_by',
            'known_by': 'knows'
        }
        return reverse_map.get(relation_type, f"inverse_{relation_type}")
    
    def _get_all_entities(self) -> Set[str]:
        """Get all entity IDs that appear in relationships"""
        entities = set(self.source_index.keys()) | set(self.target_index.keys())
        return entities
    
    def _save_relationship(self, rel_id: str) -> None:
        """Save a relationship to disk"""
        if rel_id not in self.relationships:
            return
        
        rel_file = self.storage_path / f"{rel_id}.json"
        try:
            with open(rel_file, 'w') as f:
                json.dump(self.relationships[rel_id].to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save relationship {rel_id}: {e}")
    
    def _delete_relationship(self, rel_id: str) -> None:
        """Delete a relationship"""
        if rel_id not in self.relationships:
            return
        
        rel = self.relationships[rel_id]
        
        # Remove from indexes
        self.source_index[rel.source_id].discard(rel_id)
        self.target_index[rel.target_id].discard(rel_id)
        self.type_index[rel.relation_type].discard(rel_id)
        self.category_index[rel.category.value].discard(rel_id)
        
        # Remove from graph
        if rel.source_id in self.graph and rel.target_id in self.graph[rel.source_id]:
            # Filter out this relationship
            self.graph[rel.source_id][rel.target_id] = [
                (rid, s) for rid, s in self.graph[rel.source_id][rel.target_id] if rid != rel_id
            ]
            # Clean up empty entries
            if not self.graph[rel.source_id][rel.target_id]:
                del self.graph[rel.source_id][rel.target_id]
            if not self.graph[rel.source_id]:
                del self.graph[rel.source_id]
        
        # Update stats
        self.stats['by_category'][rel.category.value] -= 1
        self.stats['by_type'][rel.relation_type] -= 1
        
        # Delete file
        rel_file = self.storage_path / f"{rel_id}.json"
        if rel_file.exists():
            rel_file.unlink()
        
        # Remove from relationships
        del self.relationships[rel_id]
        
        logger.debug(f"Deleted relationship {rel_id[:8]}")
    
    def _load_from_disk(self) -> None:
        """Load relationships from disk"""
        for file in self.storage_path.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    rel = Relationship.from_dict(data)
                    
                    self.relationships[rel.id] = rel
                    
                    # Rebuild indexes
                    self.source_index[rel.source_id].add(rel.id)
                    self.target_index[rel.target_id].add(rel.id)
                    self.type_index[rel.relation_type].add(rel.id)
                    self.category_index[rel.category.value].add(rel.id)
                    
                    # Rebuild graph
                    self.graph[rel.source_id][rel.target_id].append((rel.id, rel.strength))
                    
                    # Update stats
                    self.stats['by_category'][rel.category.value] += 1
                    self.stats['by_type'][rel.relation_type] += 1
                    self.stats['total_accesses'] += rel.access_count
                    
            except Exception as e:
                logger.error(f"Failed to load relationship from {file}: {e}")
        
        self.stats['total_relationships'] = len(self.relationships)
    
    def __repr__(self) -> str:
        return f"RelationshipManager(relationships={len(self.relationships)})"