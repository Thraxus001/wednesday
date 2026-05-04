"""
Knowledge Base - Facts and general knowledge about the world.
Like an internal Wikipedia that Wednesday can query, update,
and use for reasoning. Stores structured information about
entities, their properties, and relationships.
"""
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import uuid
import json
from pathlib import Path
import logging
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

class FactType(Enum):
    """Types of facts that can be stored"""
    ATTRIBUTE = "attribute"  # Entity has property
    RELATIONSHIP = "relationship"  # Entity connects to entity
    EVENT = "event"  # Something that happened
    RULE = "rule"  # If-then relationship
    DEFINITION = "definition"  # What something is
    PREFERENCE = "preference"  # Likes/dislikes (for users/Wednesday)
    CAPABILITY = "capability"  # What something can do
    HISTORY = "history"  # Historical fact
    SCIENCE = "science"  # Scientific fact
    CULTURAL = "cultural"  # Cultural knowledge
    PERSONAL = "personal"  # About Wednesday or users

class FactSource(Enum):
    """Where a fact came from"""
    EXPLICIT = "explicit"  # Directly taught
    INFERRED = "inferred"  # Deduced from other facts
    EXPERIENCE = "experience"  # Learned from interaction
    EXTERNAL = "external"  # From external knowledge source
    USER_PROVIDED = "user_provided"  # User told Wednesday
    CONSOLIDATED = "consolidated"  # From memory consolidation

class KnowledgeBase:
    """
    Stores facts and general knowledge about the world.
    Provides structured querying and relationship traversal.
    """
    
    def __init__(self, storage_path: Optional[Path] = None, 
                 confidence_threshold: float = 0.5):
        self.storage_path = storage_path or Path("./data/semantic/facts")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        
        # Core storage
        self.facts: Dict[str, Dict] = {}  # fact_id -> fact
        self.entities: Dict[str, Dict] = {}  # entity_id -> entity info
        
        # Indexes for fast lookup
        self.subject_index: Dict[str, List[str]] = defaultdict(list)  # subject -> fact_ids
        self.predicate_index: Dict[str, List[str]] = defaultdict(list)  # predicate -> fact_ids
        self.object_index: Dict[str, List[str]] = defaultdict(list)  # object -> fact_ids
        self.entity_index: Dict[str, str] = {}  # entity name -> entity_id
        self.type_index: Dict[str, List[str]] = defaultdict(list)  # entity_type -> entity_ids
        
        # Relationships graph
        self.entity_relations: Dict[str, Dict[str, List[Tuple[str, str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )  # entity_id -> {relation_type: [(target_id, fact_id, confidence)]}
        
        # Statistics
        self.stats = {
            'total_facts': 0,
            'total_entities': 0,
            'facts_by_type': defaultdict(int),
            'facts_by_source': defaultdict(int),
            'last_updated': None
        }
        
        self._load_from_disk()
        logger.info(f"KnowledgeBase initialized with {len(self.facts)} facts and {len(self.entities)} entities")
    
    def add_entity(self, name: str, entity_type: str = "unknown",
                   properties: Optional[Dict] = None,
                   aliases: Optional[List[str]] = None) -> str:
        """
        Add a new entity to the knowledge base.
        
        Args:
            name: Primary name of the entity
            entity_type: Category (person, place, concept, etc.)
            properties: Additional properties
            aliases: Alternative names for the entity
        
        Returns:
            entity_id
        """
        # Check if entity already exists
        norm_name = name.lower().strip()
        if norm_name in self.entity_index:
            return self.entity_index[norm_name]
        
        entity_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        entity = {
            'id': entity_id,
            'name': name,
            'normalized_name': norm_name,
            'type': entity_type,
            'properties': properties or {},
            'aliases': [a.lower().strip() for a in (aliases or [])],
            'created_at': timestamp.isoformat(),
            'updated_at': timestamp.isoformat(),
            'fact_count': 0,
            'confidence': 1.0  # Base confidence for entity existence
        }
        
        self.entities[entity_id] = entity
        self.entity_index[norm_name] = entity_id
        
        # Add aliases to index
        for alias in entity['aliases']:
            self.entity_index[alias] = entity_id
        
        self.type_index[entity_type].append(entity_id)
        
        self.stats['total_entities'] = len(self.entities)
        self.stats['last_updated'] = timestamp.isoformat()
        
        self._save_entity(entity_id)
        
        logger.debug(f"Added entity: {name} ({entity_type})")
        return entity_id
    
    def add_fact(self, subject: Union[str, Dict], predicate: str, 
                object_val: Union[str, Dict], 
                fact_type: FactType = FactType.ATTRIBUTE,
                confidence: float = 1.0,
                source: FactSource = FactSource.EXPLICIT,
                metadata: Optional[Dict] = None,
                expiration: Optional[datetime] = None) -> Optional[str]:
        """
        Store a fact in the knowledge base.
        
        Args:
            subject: Entity or value that is the subject
            predicate: Relationship or property name
            object_val: Entity or value that is the object
            fact_type: Type of fact
            confidence: How confident we are (0-1)
            source: Where this fact came from
            metadata: Additional context
            expiration: When this fact becomes invalid
        
        Returns:
            fact_id if added, None if below confidence threshold
        """
        if confidence < self.confidence_threshold:
            logger.debug(f"Fact below confidence threshold: {confidence} < {self.confidence_threshold}")
            return None
        
        fact_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Resolve or create entities
        subj_id = self._resolve_entity(subject)
        obj_id = self._resolve_entity(object_val)
        
        # If object is not an entity, treat as literal value
        obj_is_literal = obj_id is None
        
        # Get names for display
        subj_name = self._get_entity_name(subject)
        obj_name = self._get_entity_name(object_val) if not obj_is_literal else str(object_val)
        
        fact = {
            'id': fact_id,
            'subject_id': subj_id,
            'subject_name': subj_name,
            'predicate': predicate,
            'object_id': obj_id if not obj_is_literal else None,
            'object_literal': object_val if obj_is_literal else None,
            'object_name': obj_name,
            'fact_type': fact_type.value,
            'confidence': confidence,
            'source': source.value,
            'created_at': timestamp.isoformat(),
            'updated_at': timestamp.isoformat(),
            'access_count': 0,
            'expiration': expiration.isoformat() if expiration else None,
            'metadata': metadata or {},
            'contradicted_by': [],  # fact_ids that contradict this
            'supports': []  # fact_ids that support this
        }
        
        self.facts[fact_id] = fact
        
        # Update indexes
        if subj_id:
            self.subject_index[subj_id].append(fact_id)
            self.entities[subj_id]['fact_count'] += 1
        
        self.predicate_index[predicate].append(fact_id)
        
        if obj_id:
            self.object_index[obj_id].append(fact_id)
            self.entities[obj_id]['fact_count'] += 1
            
            # Update relationship graph
            self.entity_relations[subj_id][predicate].append((obj_id, fact_id, confidence))
        
        # Update statistics
        self.stats['total_facts'] = len(self.facts)
        self.stats['facts_by_type'][fact_type.value] += 1
        self.stats['facts_by_source'][source.value] += 1
        self.stats['last_updated'] = timestamp.isoformat()
        
        self._save_fact(fact_id)
        
        logger.debug(f"Added fact: {subj_name} {predicate} {obj_name}")
        return fact_id
    
    def query(self, subject: Optional[Union[str, Dict]] = None,
             predicate: Optional[str] = None,
             object_val: Optional[Union[str, Dict]] = None,
             fact_type: Optional[FactType] = None,
             min_confidence: float = 0.0,
             include_expired: bool = False) -> List[Dict]:
        """
        Find facts matching the given pattern.
        
        Args:
            subject: Subject to match (None = any)
            predicate: Predicate to match (None = any)
            object_val: Object to match (None = any)
            fact_type: Filter by fact type
            min_confidence: Minimum confidence threshold
            include_expired: Include expired facts
        
        Returns:
            List of matching facts
        """
        # Resolve entities
        subj_id = self._resolve_entity(subject) if subject else None
        obj_id = self._resolve_entity(object_val) if object_val else None
        
        # Start with candidate facts
        candidates = set(self.facts.keys())
        
        # Apply indexes
        if subj_id:
            candidates &= set(self.subject_index.get(subj_id, []))
        
        if predicate:
            candidates &= set(self.predicate_index.get(predicate, []))
        
        if obj_id:
            candidates &= set(self.object_index.get(obj_id, []))
        
        # Collect and filter results
        results = []
        now = datetime.now()
        
        for fact_id in candidates:
            fact = self.facts[fact_id]
            
            # Apply filters
            if fact['confidence'] < min_confidence:
                continue
            
            if fact_type and fact['fact_type'] != fact_type.value:
                continue
            
            if not include_expired and fact.get('expiration'):
                exp_date = datetime.fromisoformat(fact['expiration'])
                if exp_date < now:
                    continue
            
            # Update access count
            fact['access_count'] += 1
            
            # Format result
            result = {
                'id': fact_id,
                'subject': fact['subject_name'],
                'predicate': fact['predicate'],
                'object': fact['object_name'] if fact['object_name'] else fact['object_literal'],
                'fact_type': fact['fact_type'],
                'confidence': fact['confidence'],
                'source': fact['source'],
                'created_at': fact['created_at']
            }
            results.append(result)
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results
    
    def get_entity_info(self, entity: Union[str, Dict]) -> Optional[Dict]:
        """
        Get all information about an entity.
        
        Returns:
            Dictionary with entity details and all related facts
        """
        entity_id = self._resolve_entity(entity)
        if not entity_id or entity_id not in self.entities:
            return None
        
        entity_data = self.entities[entity_id]
        
        # Get all facts where this entity is subject
        subject_facts = []
        for fact_id in self.subject_index.get(entity_id, []):
            fact = self.facts[fact_id]
            if fact['confidence'] >= self.confidence_threshold:
                subject_facts.append({
                    'predicate': fact['predicate'],
                    'object': fact['object_name'] if fact['object_name'] else fact['object_literal'],
                    'confidence': fact['confidence'],
                    'fact_type': fact['fact_type']
                })
        
        # Get all facts where this entity is object
        object_facts = []
        for fact_id in self.object_index.get(entity_id, []):
            fact = self.facts[fact_id]
            if fact['confidence'] >= self.confidence_threshold:
                object_facts.append({
                    'subject': fact['subject_name'],
                    'predicate': fact['predicate'],
                    'confidence': fact['confidence'],
                    'fact_type': fact['fact_type']
                })
        
        return {
            'id': entity_id,
            'name': entity_data['name'],
            'type': entity_data['type'],
            'properties': entity_data['properties'],
            'facts_about': subject_facts,
            'facts_involving': object_facts,
            'total_facts': len(subject_facts) + len(object_facts)
        }
    
    def get_related(self, entity: Union[str, Dict], 
                   relationship_type: Optional[str] = None,
                   max_depth: int = 1) -> List[Dict]:
        """
        Find everything connected to an entity through relationships.
        
        Args:
            entity: Starting entity
            relationship_type: Filter by relationship type
            max_depth: How many hops to traverse
        
        Returns:
            List of related entities with relationship info
        """
        entity_id = self._resolve_entity(entity)
        if not entity_id:
            return []
        
        results = []
        visited = {entity_id}
        
        def traverse(current_id: str, depth: int, path: List):
            if depth >= max_depth:
                return
            
            relations = self.entity_relations.get(current_id, {})
            
            for rel_type, targets in relations.items():
                if relationship_type and rel_type != relationship_type:
                    continue
                
                for target_id, fact_id, confidence in targets:
                    if target_id not in visited and target_id in self.entities:
                        visited.add(target_id)
                        
                        results.append({
                            'entity': self.entities[target_id]['name'],
                            'entity_id': target_id,
                            'relationship': rel_type,
                            'confidence': confidence,
                            'depth': depth + 1,
                            'path': path + [rel_type]
                        })
                        
                        traverse(target_id, depth + 1, path + [rel_type])
        
        traverse(entity_id, 0, [])
        
        return results
    
    def update_confidence(self, fact_id: str, new_confidence: float,
                         reason: Optional[str] = None) -> bool:
        """
        Adjust confidence based on new information or contradictions.
        
        Args:
            fact_id: Fact to update
            new_confidence: New confidence value (0-1)
            reason: Why the confidence changed
        
        Returns:
            bool: Success status
        """
        if fact_id not in self.facts:
            return False
        
        old_confidence = self.facts[fact_id]['confidence']
        self.facts[fact_id]['confidence'] = max(0.0, min(1.0, new_confidence))
        self.facts[fact_id]['updated_at'] = datetime.now().isoformat()
        
        if reason:
            if 'confidence_history' not in self.facts[fact_id]:
                self.facts[fact_id]['confidence_history'] = []
            self.facts[fact_id]['confidence_history'].append({
                'old': old_confidence,
                'new': new_confidence,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            })
        
        self._save_fact(fact_id)
        
        logger.info(f"Updated confidence for fact {fact_id[:8]}: {old_confidence:.2f} -> {new_confidence:.2f}")
        return True
    
    def add_contradiction(self, fact_id1: str, fact_id2: str) -> bool:
        """
        Record that two facts contradict each other.
        This helps in resolving conflicts and updating confidence.
        """
        if fact_id1 not in self.facts or fact_id2 not in self.facts:
            return False
        
        if fact_id2 not in self.facts[fact_id1]['contradicted_by']:
            self.facts[fact_id1]['contradicted_by'].append(fact_id2)
        
        if fact_id1 not in self.facts[fact_id2]['contradicted_by']:
            self.facts[fact_id2]['contradicted_by'].append(fact_id1)
        
        # Reduce confidence in both facts
        self.update_confidence(fact_id1, self.facts[fact_id1]['confidence'] * 0.8,
                              reason="Contradiction with another fact")
        self.update_confidence(fact_id2, self.facts[fact_id2]['confidence'] * 0.8,
                              reason="Contradiction with another fact")
        
        self._save_fact(fact_id1)
        self._save_fact(fact_id2)
        
        return True
    
    def get_fact_network(self, central_entity: Union[str, Dict], 
                        depth: int = 2) -> Dict:
        """
        Get a subgraph of facts around a central entity.
        Useful for visualization and deep understanding.
        """
        entity_id = self._resolve_entity(central_entity)
        if not entity_id:
            return {'nodes': [], 'edges': []}
        
        nodes = {entity_id: self.entities[entity_id]['name']}
        edges = []
        visited = {entity_id}
        
        def traverse(current_id: str, current_depth: int):
            if current_depth >= depth:
                return
            
            relations = self.entity_relations.get(current_id, {})
            for rel_type, targets in relations.items():
                for target_id, fact_id, confidence in targets:
                    if target_id not in visited and target_id in self.entities:
                        visited.add(target_id)
                        nodes[target_id] = self.entities[target_id]['name']
                        edges.append({
                            'from': current_id,
                            'to': target_id,
                            'relation': rel_type,
                            'confidence': confidence,
                            'fact_id': fact_id
                        })
                        traverse(target_id, current_depth + 1)
        
        traverse(entity_id, 0)
        
        return {
            'nodes': [{'id': nid, 'name': name} for nid, name in nodes.items()],
            'edges': edges
        }
    
    def _resolve_entity(self, entity: Union[str, Dict]) -> Optional[str]:
        """Resolve an entity reference to its ID"""
        if isinstance(entity, dict):
            # If it's a dict with entity info, create or find
            if 'name' in entity:
                name = entity['name']
                norm_name = name.lower().strip()
                if norm_name in self.entity_index:
                    return self.entity_index[norm_name]
                else:
                    # Create new entity
                    return self.add_entity(
                        name=name,
                        entity_type=entity.get('type', 'unknown'),
                        properties=entity.get('properties', {})
                    )
            return None
        elif isinstance(entity, str):
            # Check if it's already an ID
            if entity in self.entities:
                return entity
            # Check if it's a name
            norm_name = entity.lower().strip()
            return self.entity_index.get(norm_name)
        return None
    
    def _get_entity_name(self, entity: Union[str, Dict]) -> str:
        """Get the name of an entity for display"""
        if isinstance(entity, dict):
            return entity.get('name', str(entity))
        elif isinstance(entity, str):
            if entity in self.entities:
                return self.entities[entity]['name']
            return entity
        return str(entity)
    
    def _save_fact(self, fact_id: str) -> None:
        """Save a fact to disk"""
        fact_file = self.storage_path / f"{fact_id}.json"
        try:
            with open(fact_file, 'w') as f:
                json.dump(self.facts[fact_id], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save fact {fact_id}: {e}")
    
    def _save_entity(self, entity_id: str) -> None:
        """Save an entity to disk"""
        entity_file = self.storage_path / f"entity_{entity_id}.json"
        try:
            with open(entity_file, 'w') as f:
                json.dump(self.entities[entity_id], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save entity {entity_id}: {e}")
    
    def _load_from_disk(self) -> None:
        """Load facts and entities from disk"""
        # Load facts
        for file in self.storage_path.glob("*.json"):
            if file.name.startswith("entity_"):
                continue
            try:
                with open(file, 'r') as f:
                    fact = json.load(f)
                    self.facts[fact['id']] = fact
                    
                    # Rebuild indexes
                    if fact.get('subject_id'):
                        self.subject_index[fact['subject_id']].append(fact['id'])
                    self.predicate_index[fact['predicate']].append(fact['id'])
                    if fact.get('object_id'):
                        self.object_index[fact['object_id']].append(fact['id'])
                    
            except Exception as e:
                logger.error(f"Failed to load fact from {file}: {e}")
        
        # Load entities
        for file in self.storage_path.glob("entity_*.json"):
            try:
                with open(file, 'r') as f:
                    entity = json.load(f)
                    self.entities[entity['id']] = entity
                    self.entity_index[entity['normalized_name']] = entity['id']
                    for alias in entity.get('aliases', []):
                        self.entity_index[alias] = entity['id']
                    self.type_index[entity['type']].append(entity['id'])
            except Exception as e:
                logger.error(f"Failed to load entity from {file}: {e}")
        
        # Rebuild relationship graph
        for fact_id, fact in self.facts.items():
            if fact.get('subject_id') and fact.get('object_id'):
                self.entity_relations[fact['subject_id']][fact['predicate']].append(
                    (fact['object_id'], fact_id, fact['confidence'])
                )
        
        # Update statistics
        self.stats['total_facts'] = len(self.facts)
        self.stats['total_entities'] = len(self.entities)
        for fact in self.facts.values():
            self.stats['facts_by_type'][fact['fact_type']] += 1
            self.stats['facts_by_source'][fact['source']] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        # Convert defaultdicts to regular dicts for return
        facts_by_type = dict(self.stats['facts_by_type'])
        facts_by_source = dict(self.stats['facts_by_source'])
        entities_by_type = {t: len(ids) for t, ids in self.type_index.items()}
        
        # Calculate average confidence safely
        if self.facts:
            avg_confidence = sum(f['confidence'] for f in self.facts.values()) / len(self.facts)
        else:
            avg_confidence = 0.0
        
        # Calculate total relations
        total_relations = sum(
            len(rels) for rels_dict in self.entity_relations.values() 
            for rels in rels_dict.values()
        )
        
        return {
            'total_facts': len(self.facts),
            'total_entities': len(self.entities),
            'facts_by_type': facts_by_type,
            'facts_by_source': facts_by_source,
            'entities_by_type': entities_by_type,
            'avg_confidence': avg_confidence,
            'total_relations': total_relations,
            'last_updated': self.stats['last_updated']
        }
    
    def __repr__(self) -> str:
        return f"KnowledgeBase(facts={len(self.facts)}, entities={len(self.entities)})"