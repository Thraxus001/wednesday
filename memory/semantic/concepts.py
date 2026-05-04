"""
Concept Network - Represents relationships between concepts and ideas.
Like a knowledge graph that shows how different concepts connect,
enabling analogical thinking and deeper understanding.
"""

from typing import Dict, Any, Optional, List, Set, Tuple, Union
from datetime import datetime
import uuid
import json
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict
from enum import Enum

# Configure logger
logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of relationships between concepts in the knowledge graph."""
    # Hierarchical relations
    IS_A = "is_a"  # Hierarchy (e.g., dog is_a animal)
    HAS_A = "has_a"  # Composition (e.g., person has_a name)
    PART_OF = "part_of"  # Meronymy (e.g., wheel part_of car)
    
    # Similarity relations
    RELATED_TO = "related_to"  # General association
    OPPOSITE_OF = "opposite_of"  # Antonyms
    SIMILAR_TO = "similar_to"  # Synonyms
    
    # Causal relations
    CAUSES = "causes"  # Causality
    CAUSED_BY = "caused_by"  # Reverse causality
    PRECEDES = "precedes"  # Temporal order
    FOLLOWS = "follows"  # Temporal after
    
    # Spatial relations
    LOCATED_IN = "located_in"  # Spatial
    
    # Functional relations
    USED_FOR = "used_for"  # Function
    ATTRIBUTE_OF = "attribute_of"  # Property
    EXAMPLE_OF = "example_of"  # Instance
    CREATED_BY = "created_by"  # Agency
    
    # Psychological relations (for understanding users)
    BELIEVES = "believes"  # Belief/opinion
    WANTS = "wants"  # Desire
    
    @classmethod
    def from_string(cls, relation_str: str):
        """Get enum member from string value."""
        for member in cls:
            if member.value == relation_str:
                return member
        return cls.RELATED_TO


# Type aliases
ConceptDict = Dict[str, Any]
RelationDict = Dict[str, Any]
PathType = List[Dict[str, Any]]


class ConceptNetwork:
    """
    A network of interconnected concepts representing Wednesday's understanding.
    Enables analogical reasoning, inference, and deeper comprehension.
    
    Features:
    - Graph-based concept storage with typed relations
    - Bidirectional traversal and path finding
    - Embedding-based similarity search
    - Analogical inference and pattern discovery
    - Persistent storage with JSON
    """
    
    # Class constants
    DEFAULT_STORAGE_PATH = Path("./data/semantic/concepts")
    DEFAULT_SIMILARITY_THRESHOLD = 0.6
    DEFAULT_PATH_MAX_DEPTH = 5
    DEFAULT_RELATION_STRENGTH = 1.0
    MIN_SIMILARITY_FOR_INFERENCE = 0.6
    MAX_INFERENCE_RESULTS = 10
    MAX_PATTERNS_PER_TYPE = 5
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the concept network.
        
        Args:
            storage_path: Directory for persistent storage
        """
        self.storage_path = storage_path or self.DEFAULT_STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core data structures
        self.concepts: Dict[str, ConceptDict] = {}  # concept_id -> concept data
        self.relations: Dict[str, List[RelationDict]] = defaultdict(list)  # concept_id -> outgoing relations
        
        # Indexes for fast lookup
        self.name_to_id: Dict[str, str] = {}  # concept name -> concept_id
        self.type_index: Dict[str, List[str]] = defaultdict(list)  # concept_type -> concept_ids
        
        # Embeddings for similarity
        self.embeddings: Dict[str, np.ndarray] = {}  # concept_id -> embedding vector
        
        # Statistics
        self.stats: Dict[str, Any] = {
            'total_concepts': 0,
            'total_relations': 0,
            'last_updated': None
        }
        
        # Load existing data
        self._load_from_disk()
        logger.info(f"ConceptNetwork initialized with {len(self.concepts)} concepts")
    
    def add_concept(self, 
                   name: str, 
                   concept_type: str = "general",
                   properties: Optional[Dict] = None,
                   definition: Optional[str] = None,
                   embedding: Optional[np.ndarray] = None,
                   confidence: float = 1.0,
                   source: str = "explicit") -> str:
        """
        Add a new concept to the network.
        
        Args:
            name: The concept name (e.g., "dark_humor", "loyalty")
            concept_type: Category (e.g., "emotion", "trait", "object")
            properties: Additional properties about the concept
            definition: Textual definition/description
            embedding: Vector representation for similarity
            confidence: How confident we are about this concept (0-1)
            source: Where this concept came from (explicit, inferred, learned)
        
        Returns:
            concept_id for the new concept
            
        Raises:
            ValueError: If confidence is out of range or concept name already exists
        """
        # Validate inputs
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        
        # Check for duplicate concept
        normalized_name = name.lower().strip()
        if normalized_name in self.name_to_id:
            raise ValueError(f"Concept '{name}' already exists")
        
        # Generate ID and timestamp
        concept_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create concept entry
        concept: ConceptDict = {
            'id': concept_id,
            'name': name,
            'normalized_name': normalized_name,
            'type': concept_type,
            'properties': properties or {},
            'definition': definition,
            'confidence': confidence,
            'created_at': timestamp.isoformat(),
            'updated_at': timestamp.isoformat(),
            'access_count': 0,
            'source': source,
            'related_concepts': []
        }
        
        # Store concept
        self.concepts[concept_id] = concept
        self.name_to_id[normalized_name] = concept_id
        self.type_index[concept_type].append(concept_id)
        
        # Store embedding if provided
        if embedding is not None:
            self.embeddings[concept_id] = embedding
        
        # Update statistics
        self.stats['total_concepts'] = len(self.concepts)
        self.stats['last_updated'] = timestamp.isoformat()
        
        # Persist to disk
        self._save_concept(concept_id)
        
        logger.debug(f"Added concept '{name}' ({concept_id[:8]}) of type '{concept_type}'")
        return concept_id
    
    def add_relation(self, 
                    from_concept: Union[str, ConceptDict], 
                    to_concept: Union[str, ConceptDict],
                    relation_type: RelationType,
                    strength: float = DEFAULT_RELATION_STRENGTH,
                    bidirectional: bool = False,
                    metadata: Optional[Dict] = None) -> bool:
        """
        Add a relationship between two concepts.
        
        Args:
            from_concept: Source concept name or ID or concept dict
            to_concept: Target concept name or ID or concept dict
            relation_type: Type of relationship
            strength: 0-1 how strong/confident the relation is
            bidirectional: Also add the reverse relation
            metadata: Additional relation metadata
        
        Returns:
            True if relation was added successfully
        
        Raises:
            ValueError: If strength is out of range
        """
        # Validate strength
        if not 0 <= strength <= 1:
            raise ValueError(f"Strength must be between 0 and 1, got {strength}")
        
        # Resolve concept IDs
        from_id = self._resolve_concept(from_concept)
        to_id = self._resolve_concept(to_concept)
        
        if not from_id or not to_id:
            logger.warning(f"Could not resolve concepts: {from_concept} -> {to_concept}")
            return False
        
        if from_id == to_id:
            logger.warning(f"Cannot add relation from concept to itself: {from_concept}")
            return False
        
        # Check if relation already exists and update it
        existing = self._find_relation(from_id, to_id, relation_type)
        if existing:
            return self._update_existing_relation(existing, from_id, strength)
        
        # Create new relation
        relation = self._create_relation(from_id, to_id, relation_type, strength, metadata)
        
        # Add to relations
        self.relations[from_id].append(relation)
        self.stats['total_relations'] += 1
        
        # Update concept's related list
        if to_id not in self.concepts[from_id]['related_concepts']:
            self.concepts[from_id]['related_concepts'].append(to_id)
        
        # Add reverse relation if bidirectional
        if bidirectional:
            self._add_reverse_relation(to_id, from_id, relation_type, strength, metadata)
        
        # Persist changes
        self._save_concept(from_id)
        if bidirectional:
            self._save_concept(to_id)
        
        logger.debug(f"Added relation: {self._get_concept_name(from_id)} {relation_type.value} {self._get_concept_name(to_id)}")
        return True
    
    def _find_relation(self, from_id: str, to_id: str, relation_type: RelationType) -> Optional[RelationDict]:
        """Find an existing relation between concepts."""
        for rel in self.relations.get(from_id, []):
            if rel['target_id'] == to_id and rel['type'] == relation_type.value:
                return rel
        return None
    
    def _update_existing_relation(self, relation: RelationDict, from_id: str, strength: float) -> bool:
        """Update an existing relation."""
        relation['strength'] = strength
        relation['updated_at'] = datetime.now().isoformat()
        relation['access_count'] = relation.get('access_count', 0) + 1
        self._save_concept(from_id)
        logger.debug(f"Updated existing relation strength to {strength}")
        return True
    
    def _create_relation(self, 
                        from_id: str, 
                        to_id: str, 
                        relation_type: RelationType,
                        strength: float,
                        metadata: Optional[Dict]) -> RelationDict:
        """Create a new relation dictionary."""
        return {
            'source_id': from_id,
            'target_id': to_id,
            'type': relation_type.value,
            'strength': strength,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'access_count': 0,
            'metadata': metadata or {}
        }
    
    def _add_reverse_relation(self, 
                             from_id: str, 
                             to_id: str, 
                             relation_type: RelationType,
                             strength: float,
                             metadata: Optional[Dict]) -> None:
        """Add the reverse of a bidirectional relation."""
        reverse_map = {
            RelationType.IS_A: RelationType.IS_A,  # Note: IS_A is hierarchical, reverse is also IS_A? Needs thought
            RelationType.HAS_A: RelationType.PART_OF,
            RelationType.PART_OF: RelationType.HAS_A,
            RelationType.OPPOSITE_OF: RelationType.OPPOSITE_OF,  # Symmetric
            RelationType.SIMILAR_TO: RelationType.SIMILAR_TO,  # Symmetric
            RelationType.CAUSES: RelationType.CAUSED_BY,
            RelationType.CAUSED_BY: RelationType.CAUSES,
            RelationType.PRECEDES: RelationType.FOLLOWS,
            RelationType.FOLLOWS: RelationType.PRECEDES,
            RelationType.LOCATED_IN: RelationType.CONTAINS,  # Would need CONTAINS enum
            RelationType.USED_FOR: RelationType.USED_BY,  # Would need USED_BY
            RelationType.CREATED_BY: RelationType.CREATES,  # Would need CREATES
        }
        
        # Get reverse type or default to RELATED_TO
        reverse_type = reverse_map.get(relation_type, RelationType.RELATED_TO)
        
        # Only add if different from original to avoid infinite recursion
        if reverse_type != relation_type:
            self.add_relation(
                from_id, 
                to_id, 
                reverse_type, 
                strength, 
                bidirectional=False, 
                metadata=metadata
            )
    
    def get_related(self, 
                   concept: Union[str, ConceptDict], 
                   relation_types: Optional[List[RelationType]] = None,
                   min_strength: float = 0.5,
                   max_depth: int = 1) -> List[Dict[str, Any]]:
        """
        Get concepts related to the given concept.
        
        Args:
            concept: Concept name or ID or concept dict
            relation_types: Filter by relation types
            min_strength: Minimum relation strength
            max_depth: How many hops to traverse
        
        Returns:
            List of related concepts with relation info
        """
        concept_id = self._resolve_concept(concept)
        if not concept_id or concept_id not in self.concepts:
            return []
        
        # Prepare relation type filter set
        relation_type_values = None
        if relation_types:
            relation_type_values = {rt.value for rt in relation_types}
        
        results = []
        visited = {concept_id}
        
        def traverse(current_id: str, depth: int, path: List[str]):
            """Recursive traversal of the concept graph."""
            if depth >= max_depth:
                return
            
            for relation in self.relations.get(current_id, []):
                # Apply filters
                if relation['strength'] < min_strength:
                    continue
                
                if relation_type_values and relation['type'] not in relation_type_values:
                    continue
                
                target_id = relation['target_id']
                if target_id not in visited and target_id in self.concepts:
                    visited.add(target_id)
                    
                    target_concept = self.concepts[target_id]
                    
                    # Build result entry
                    result_entry = {
                        'concept': target_concept['name'],
                        'concept_id': target_id,
                        'relation_type': relation['type'],
                        'strength': relation['strength'],
                        'depth': depth,
                        'path': path + [target_concept['name']]
                    }
                    
                    # Add relation metadata if present
                    if relation.get('metadata'):
                        result_entry['metadata'] = relation['metadata']
                    
                    results.append(result_entry)
                    
                    # Continue traversal
                    traverse(target_id, depth + 1, path + [target_concept['name']])
        
        # Start traversal
        traverse(concept_id, 1, [self.concepts[concept_id]['name']])
        
        # Sort by strength
        results.sort(key=lambda x: x['strength'], reverse=True)
        
        # Update access count
        self.concepts[concept_id]['access_count'] += 1
        
        return results
    
    def find_path(self, 
                 from_concept: Union[str, ConceptDict], 
                 to_concept: Union[str, ConceptDict],
                 max_depth: int = DEFAULT_PATH_MAX_DEPTH) -> Optional[PathType]:
        """
        Find a path between two concepts through the network.
        Enables analogical reasoning.
        
        Args:
            from_concept: Start concept
            to_concept: Target concept
            max_depth: Maximum search depth
        
        Returns:
            List of relations forming the path, or None if no path found
        """
        from_id = self._resolve_concept(from_concept)
        to_id = self._resolve_concept(to_concept)
        
        if not from_id or not to_id or from_id not in self.concepts or to_id not in self.concepts:
            logger.warning(f"Cannot find path - concepts not found")
            return None
        
        if from_id == to_id:
            return []  # Same concept, empty path
        
        # BFS to find shortest path
        queue = [(from_id, [])]
        visited = {from_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            for relation in self.relations.get(current_id, []):
                next_id = relation['target_id']
                
                if next_id == to_id:
                    # Found target - construct complete path
                    complete_path = path + [{
                        'from': self.concepts[current_id]['name'],
                        'to': self.concepts[next_id]['name'],
                        'relation': relation['type'],
                        'strength': relation['strength'],
                        'relation_type': relation['type']
                    }]
                    return complete_path
                
                if next_id not in visited and next_id in self.concepts:
                    visited.add(next_id)
                    new_path = path + [{
                        'from': self.concepts[current_id]['name'],
                        'to': self.concepts[next_id]['name'],
                        'relation': relation['type'],
                        'strength': relation['strength'],
                        'relation_type': relation['type']
                    }]
                    queue.append((next_id, new_path))
        
        return None  # No path found
    
    def get_similar(self, 
                   concept: Union[str, ConceptDict], 
                   threshold: float = DEFAULT_SIMILARITY_THRESHOLD, 
                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find concepts similar to the given concept using embeddings.
        
        Args:
            concept: Concept name or ID or concept dict
            threshold: Minimum similarity score (0-1)
            limit: Maximum number of results
        
        Returns:
            List of similar concepts with similarity scores
        """
        concept_id = self._resolve_concept(concept)
        if not concept_id or concept_id not in self.embeddings:
            return []
        
        query_emb = self.embeddings[concept_id]
        similarities = []
        
        for cid, emb in self.embeddings.items():
            if cid != concept_id:
                # Cosine similarity
                similarity = self._cosine_similarity(query_emb, emb)
                if similarity >= threshold:
                    similarities.append((similarity, cid))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Build results
        results = []
        for sim, cid in similarities[:limit]:
            concept_data = self.concepts.get(cid, {})
            if concept_data:
                results.append({
                    'concept': concept_data['name'],
                    'concept_id': cid,
                    'similarity': float(sim),
                    'type': concept_data['type']
                })
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def infer_relation(self, 
                      concept_a: Union[str, ConceptDict], 
                      concept_b: Union[str, ConceptDict]) -> List[Dict[str, Any]]:
        """
        Infer possible relations between two concepts based on common patterns.
        Uses analogical reasoning: if A relates to B like C relates to D...
        
        Args:
            concept_a: First concept
            concept_b: Second concept
        
        Returns:
            List of inferred relations with confidence scores
        """
        a_id = self._resolve_concept(concept_a)
        b_id = self._resolve_concept(concept_b)
        
        if not a_id or not b_id:
            return []
        
        # Find concepts similar to A
        similar_to_a = self.get_similar(concept_a, threshold=0.5, limit=5)
        
        inferences = []
        
        # For each concept similar to A, see how it relates to things
        for sim_a in similar_to_a:
            sim_a_id = sim_a['concept_id']
            
            # Get relations of similar concept
            for relation in self.relations.get(sim_a_id, []):
                target_id = relation['target_id']
                target_concept = self.concepts.get(target_id)
                
                if not target_concept:
                    continue
                
                # Check if target is similar to B
                similarity = self._concept_similarity(target_id, b_id)
                
                if similarity > self.MIN_SIMILARITY_FOR_INFERENCE:
                    # Calculate confidence as product of strengths
                    confidence = relation['strength'] * similarity * sim_a['similarity']
                    
                    inferences.append({
                        'inferred_relation': relation['type'],
                        'confidence': float(confidence),
                        'analogy': f"{sim_a['concept']} : {target_concept['name']}",
                        'explanation': f"Like how {sim_a['concept']} {relation['type']} {target_concept['name']}",
                        'source_similarity': float(sim_a['similarity']),
                        'target_similarity': float(similarity)
                    })
        
        # Sort by confidence and remove duplicates
        inferences.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Deduplicate by relation type
        seen_types = set()
        unique_inferences = []
        for inf in inferences:
            if inf['inferred_relation'] not in seen_types:
                seen_types.add(inf['inferred_relation'])
                unique_inferences.append(inf)
                if len(unique_inferences) >= self.MAX_INFERENCE_RESULTS:
                    break
        
        return unique_inferences
    
    def merge_concepts(self, 
                      concept1: Union[str, ConceptDict], 
                      concept2: Union[str, ConceptDict]) -> Optional[str]:
        """
        Merge two concepts that represent the same thing.
        Keeps the stronger/more confident one.
        
        Args:
            concept1: First concept
            concept2: Second concept
        
        Returns:
            ID of the kept concept, or None if merge failed
        """
        id1 = self._resolve_concept(concept1)
        id2 = self._resolve_concept(concept2)
        
        if not id1 or not id2 or id1 == id2:
            logger.warning(f"Cannot merge concepts - invalid IDs")
            return None
        
        # Decide which to keep (higher confidence)
        conf1 = self.concepts[id1]['confidence']
        conf2 = self.concepts[id2]['confidence']
        
        keep_id = id1 if conf1 >= conf2 else id2
        merge_id = id2 if keep_id == id1 else id1
        
        logger.info(f"Merging concept '{self.concepts[merge_id]['name']}' into '{self.concepts[keep_id]['name']}'")
        
        # Transfer all outgoing relations from merge to keep
        for relation in self.relations.get(merge_id, []):
            # Update relation to point from keep_id
            relation['source_id'] = keep_id
            self.relations[keep_id].append(relation)
        
        # Transfer all incoming relations to merge
        for source_id, relations in self.relations.items():
            for relation in relations:
                if relation['target_id'] == merge_id:
                    relation['target_id'] = keep_id
        
        # Update name to be more comprehensive (keep longer name)
        if len(self.concepts[keep_id]['name']) < len(self.concepts[merge_id]['name']):
            self.concepts[keep_id]['name'] = self.concepts[merge_id]['name']
            self.concepts[keep_id]['normalized_name'] = self.concepts[merge_id]['normalized_name']
        
        # Merge properties
        self.concepts[keep_id]['properties'].update(self.concepts[merge_id]['properties'])
        
        # Merge definitions if one is missing
        if not self.concepts[keep_id].get('definition') and self.concepts[merge_id].get('definition'):
            self.concepts[keep_id]['definition'] = self.concepts[merge_id]['definition']
        
        # Update confidence (take max)
        self.concepts[keep_id]['confidence'] = max(conf1, conf2)
        
        # Update related concepts list
        self.concepts[keep_id]['related_concepts'].extend(
            self.concepts[merge_id]['related_concepts']
        )
        
        # Remove merged concept
        self._delete_concept(merge_id)
        
        # Save the kept concept
        self._save_concept(keep_id)
        
        logger.info(f"Successfully merged concepts, kept: {self.concepts[keep_id]['name']}")
        
        return keep_id
    
    def get_concept(self, concept: Union[str, ConceptDict]) -> Optional[ConceptDict]:
        """
        Get full concept data.
        
        Args:
            concept: Concept name or ID or concept dict
        
        Returns:
            Concept data dictionary or None if not found
        """
        concept_id = self._resolve_concept(concept)
        if concept_id and concept_id in self.concepts:
            self.concepts[concept_id]['access_count'] += 1
            return self.concepts[concept_id].copy()  # Return copy to prevent modification
        
        return None
    
    def get_concept_by_type(self, concept_type: str) -> List[ConceptDict]:
        """
        Get all concepts of a specific type.
        
        Args:
            concept_type: Type to filter by
        
        Returns:
            List of concepts of that type
        """
        concept_ids = self.type_index.get(concept_type, [])
        return [self.concepts[cid].copy() for cid in concept_ids if cid in self.concepts]
    
    def get_common_paths(self, 
                        concept: Union[str, ConceptDict], 
                        depth: int = 2) -> Dict[str, List[Dict]]:
        """
        Get common paths/patterns involving a concept.
        
        Args:
            concept: Concept to analyze
            depth: Maximum depth to traverse
        
        Returns:
            Dictionary mapping pattern types to lists of occurrences
        """
        concept_id = self._resolve_concept(concept)
        if not concept_id:
            return {}
        
        patterns = defaultdict(list)
        
        def collect_patterns(current_id: str, path: List[str], current_depth: int):
            """Recursively collect patterns."""
            if current_depth >= depth:
                return
            
            for relation in self.relations.get(current_id, []):
                target_id = relation['target_id']
                if target_id in self.concepts:
                    target_concept = self.concepts[target_id]
                    
                    # Create pattern key from relation type and target concept type
                    pattern_key = f"{relation['type']}_{target_concept['type']}"
                    
                    pattern_entry = {
                        'concept': target_concept['name'],
                        'concept_id': target_id,
                        'relation': relation['type'],
                        'strength': relation['strength']
                    }
                    
                    patterns[pattern_key].append(pattern_entry)
                    
                    # Continue traversal
                    collect_patterns(target_id, path + [target_id], current_depth + 1)
        
        collect_patterns(concept_id, [], 0)
        
        # Deduplicate and sort results
        result_patterns = {}
        for key, entries in patterns.items():
            # Deduplicate by concept_id
            seen = set()
            unique = []
            for entry in entries:
                if entry['concept_id'] not in seen:
                    seen.add(entry['concept_id'])
                    unique.append(entry)
            
            # Sort by strength and limit
            unique.sort(key=lambda x: x['strength'], reverse=True)
            result_patterns[key] = unique[:self.MAX_PATTERNS_PER_TYPE]
        
        return result_patterns
    
    def delete_concept(self, concept: Union[str, ConceptDict]) -> bool:
        """
        Delete a concept from the network.
        
        Args:
            concept: Concept to delete
        
        Returns:
            True if successfully deleted
        """
        concept_id = self._resolve_concept(concept)
        if not concept_id:
            return False
        
        self._delete_concept(concept_id)
        logger.info(f"Deleted concept {concept_id[:8]}")
        return True
    
    def _resolve_concept(self, concept: Union[str, ConceptDict]) -> Optional[str]:
        """
        Resolve a concept name, ID, or dict to its ID.
        
        Args:
            concept: Concept identifier
        
        Returns:
            Concept ID or None if not found
        """
        if isinstance(concept, dict):
            # If it's a concept dict, extract ID
            return concept.get('id')
        
        if isinstance(concept, str):
            # Check if it's already an ID
            if concept in self.concepts:
                return concept
            
            # Check if it's a name
            normalized = concept.lower().strip()
            return self.name_to_id.get(normalized)
        
        return None
    
    def _get_concept_name(self, concept_id: str) -> str:
        """Get concept name from ID."""
        return self.concepts.get(concept_id, {}).get('name', 'unknown')
    
    def _concept_similarity(self, id1: str, id2: str) -> float:
        """
        Calculate similarity between two concepts.
        Uses embeddings if available, falls back to relation overlap.
        
        Args:
            id1: First concept ID
            id2: Second concept ID
        
        Returns:
            Similarity score between 0 and 1
        """
        # Use embeddings if available
        if id1 in self.embeddings and id2 in self.embeddings:
            return self._cosine_similarity(self.embeddings[id1], self.embeddings[id2])
        
        # Fallback to relation overlap
        rels1 = {r['target_id'] for r in self.relations.get(id1, [])}
        rels2 = {r['target_id'] for r in self.relations.get(id2, [])}
        
        if not rels1 or not rels2:
            return 0.0
        
        intersection = len(rels1 & rels2)
        union = len(rels1 | rels2)
        
        return intersection / union if union > 0 else 0.0
    
    def _save_concept(self, concept_id: str) -> None:
        """
        Save a concept to disk.
        
        Args:
            concept_id: ID of concept to save
        """
        if concept_id not in self.concepts:
            return
        
        concept_file = self.storage_path / f"{concept_id}.json"
        
        # Create a copy of concept with relations included
        concept_copy = self.concepts[concept_id].copy()
        
        # Add relations to the saved concept
        concept_copy['relations'] = self.relations.get(concept_id, [])
        
        try:
            with open(concept_file, 'w') as f:
                json.dump(concept_copy, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save concept {concept_id[:8]}: {e}")
    
    def _delete_concept(self, concept_id: str) -> None:
        """
        Delete a concept from memory and disk.
        
        Args:
            concept_id: ID of concept to delete
        """
        if concept_id not in self.concepts:
            return
        
        # Get concept data for index cleanup
        concept = self.concepts[concept_id]
        normalized_name = concept['normalized_name']
        concept_type = concept['type']
        
        # Remove from indices
        if normalized_name in self.name_to_id:
            del self.name_to_id[normalized_name]
        
        if concept_id in self.type_index[concept_type]:
            self.type_index[concept_type].remove(concept_id)
        
        # Remove from relations (outgoing)
        if concept_id in self.relations:
            del self.relations[concept_id]
        
        # Remove from relations (incoming) - scan all relations
        for source_id in list(self.relations.keys()):
            self.relations[source_id] = [
                r for r in self.relations[source_id] 
                if r['target_id'] != concept_id
            ]
        
        # Remove embedding
        if concept_id in self.embeddings:
            del self.embeddings[concept_id]
        
        # Remove concept
        del self.concepts[concept_id]
        
        # Remove file
        concept_file = self.storage_path / f"{concept_id}.json"
        if concept_file.exists():
            concept_file.unlink()
        
        # Update stats
        self.stats['total_concepts'] = len(self.concepts)
        self.stats['total_relations'] = sum(len(r) for r in self.relations.values())
    
    def _load_from_disk(self) -> None:
        """Load concepts from disk into memory."""
        concept_files = self.storage_path.glob("*.json")
        
        loaded_count = 0
        for file in concept_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                concept_id = data['id']
                
                # Load concept (exclude relations from main concept dict)
                concept = {k: v for k, v in data.items() if k != 'relations'}
                self.concepts[concept_id] = concept
                
                # Load relations
                if 'relations' in data:
                    self.relations[concept_id] = data['relations']
                
                # Update indices
                self.name_to_id[concept['normalized_name']] = concept_id
                self.type_index[concept['type']].append(concept_id)
                
                loaded_count += 1
                
            except Exception as e:
                logger.error(f"Failed to load concept from {file}: {e}")
        
        # Update stats
        self.stats['total_concepts'] = len(self.concepts)
        self.stats['total_relations'] = sum(len(r) for r in self.relations.values())
        
        logger.info(f"Loaded {loaded_count} concepts from disk")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get network statistics.
        
        Returns:
            Dictionary with network statistics
        """
        # Calculate relation type distribution
        relation_counts = defaultdict(int)
        for relations in self.relations.values():
            for rel in relations:
                relation_counts[rel['type']] += 1
        
        # Calculate concept type distribution
        concept_types = {t: len(ids) for t, ids in self.type_index.items()}
        
        # Calculate average relations per concept
        total_relations = sum(len(r) for r in self.relations.values())
        avg_relations = total_relations / max(len(self.concepts), 1)
        
        return {
            'total_concepts': len(self.concepts),
            'total_relations': total_relations,
            'concept_types': concept_types,
            'relation_distribution': dict(relation_counts),
            'avg_relations_per_concept': round(avg_relations, 2),
            'concepts_with_embeddings': len(self.embeddings),
            'last_updated': self.stats['last_updated']
        }
    
    def clear(self) -> None:
        """Clear all concepts and relations (for testing/debugging)."""
        self.concepts.clear()
        self.relations.clear()
        self.name_to_id.clear()
        self.type_index.clear()
        self.embeddings.clear()
        self.stats = {
            'total_concepts': 0,
            'total_relations': 0,
            'last_updated': None
        }
        logger.warning("Concept network cleared")
    
    def __len__(self) -> int:
        return len(self.concepts)
    
    def __contains__(self, concept: Union[str, ConceptDict]) -> bool:
        """Check if a concept exists in the network."""
        return self._resolve_concept(concept) is not None
    
    def __repr__(self) -> str:
        return (f"ConceptNetwork(concepts={len(self.concepts)}, "
                f"relations={sum(len(r) for r in self.relations.values())})")