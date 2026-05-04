"""
Scene understanding - holistic interpretation of visual scenes.
Wednesday doesn't just see objects and faces - she understands the story they tell.
The arrangement of furniture, the tension in a room, the dynamics between people,
the narrative implied by every detail.
"""
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict, Counter
import math

# Try to import networkx
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logging.warning("networkx not available. Install networkx for spatial graph features")

logger = logging.getLogger(__name__)

# Try to import vision components
try:
    from .object_recognition import ObjectRecognizer, DetectedObject, ObjectCategory
    from .face_processing import FaceProcessor, FaceAnalysis, FacialExpression, GazeDirection
    HAS_VISION = True
except ImportError:
    HAS_VISION = False
    # Create placeholder enums if imports fail
    class ObjectCategory(Enum):
        UNKNOWN = "unknown"
    class FacialExpression(Enum):
        NEUTRAL = "neutral"
    class GazeDirection(Enum):
        TOWARDS_CAMERA = "towards_camera"
    logger.warning("Vision components not available for scene understanding")

class SocialDynamics(Enum):
    """Types of social interactions"""
    CONVERSATION = "conversation"
    ARGUMENT = "argument"
    FLIRTING = "flirting"
    IGNORING = "ignoring"
    WATCHING = "watching"
    AVOIDING = "avoiding"
    COMFORTING = "comforting"
    THREATENING = "threatening"
    COOPERATING = "cooperating"
    COMPETING = "competing"

class SceneMood(Enum):
    """Overall mood of a scene"""
    PEACEFUL = "peaceful"
    TENSE = "tense"
    HOSTILE = "hostile"
    JOYFUL = "joyful"
    MELANCHOLIC = "melancholic"
    MYSTERIOUS = "mysterious"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"
    BORING = "boring"
    CHAOTIC = "chaotic"
    ORDERLY = "orderly"

class ActivityType(Enum):
    """Types of activities"""
    WORKING = "working"
    EATING = "eating"
    SLEEPING = "sleeping"
    READING = "reading"
    WATCHING_TV = "watching_tv"
    EXERCISING = "exercising"
    SOCIALIZING = "socializing"
    ARGUING = "arguing"
    IGNORING = "ignoring"
    WAITING = "waiting"
    SEARCHING = "searching"
    HIDING = "hiding"
    FOLLOWING = "following"
    FLEEING = "fleeing"
    SITTING = "sitting"
    STANDING = "standing"
    WALKING = "walking"

class SpatialRelation(Enum):
    """Spatial relationships between objects/people"""
    NEAR = "near"
    FAR = "far"
    TOUCHING = "touching"
    INSIDE = "inside"
    ON_TOP = "on_top"
    UNDERNEATH = "underneath"
    BEHIND = "behind"
    IN_FRONT = "in_front"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    FACING = "facing"
    AWAY_FROM = "away_from"

@dataclass
class PersonState:
    """State of a person in the scene"""
    face_analysis: FaceAnalysis
    position: Tuple[int, int]  # Center position
    bounding_box: Tuple[int, int, int, int]
    
    # Behavioral state
    activity: ActivityType = ActivityType.WAITING
    attention_focus: Optional[str] = None  # What they're looking at
    interacting_with: List[str] = field(default_factory=list)  # People they're interacting with
    
    # Dynamics
    movement_direction: Optional[str] = None
    movement_speed: float = 0.0  # pixels per second
    personal_space: float = 50.0  # pixels radius
    
    # Wednesday's observations
    seems_comfortable: bool = True
    seems_suspicious: bool = False
    seems_threatened: bool = False
    seems_threatening: bool = False
    notes: List[str] = field(default_factory=list)
    
    # Identification
    person_id: str = ""
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'person_id': self.person_id,
            'position': self.position,
            'activity': self.activity.value,
            'attention_focus': self.attention_focus,
            'interacting_with': self.interacting_with,
            'seems_suspicious': self.seems_suspicious,
            'seems_threatening': self.seems_threatening
        }

@dataclass
class ObjectState:
    """State of an object in the scene"""
    detection: DetectedObject
    position: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]
    
    # Context
    is_in_use: bool = False
    used_by: Optional[str] = None  # Person ID using it
    is_out_of_place: bool = False
    expected_location: Optional[str] = None
    
    # Relationships
    part_of_larger_object: Optional[str] = None
    contains_objects: List[str] = field(default_factory=list)
    
    # Identification
    object_id: str = ""
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'object_id': self.object_id,
            'category': self.detection.category.value,
            'position': self.position,
            'is_in_use': self.is_in_use,
            'is_out_of_place': self.is_out_of_place
        }

@dataclass
class SpatialGraph:
    """Graph representation of spatial relationships"""
    graph: Any = None  # nx.Graph
    
    def __post_init__(self):
        if HAS_NETWORKX and self.graph is None:
            self.graph = nx.Graph()
    
    def add_object(self, obj_id: str, obj_type: str, position: Tuple[int, int]):
        if HAS_NETWORKX and self.graph is not None:
            self.graph.add_node(obj_id, type='object', object_type=obj_type, position=position)
    
    def add_person(self, person_id: str, position: Tuple[int, int]):
        if HAS_NETWORKX and self.graph is not None:
            self.graph.add_node(person_id, type='person', position=position)
    
    def add_relation(self, from_id: str, to_id: str, relation: SpatialRelation, distance: float):
        if HAS_NETWORKX and self.graph is not None:
            self.graph.add_edge(from_id, to_id, relation=relation.value, distance=distance)
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        if not HAS_NETWORKX or self.graph is None:
            return {'nodes': [], 'edges': []}
        
        nodes = []
        for node, data in self.graph.nodes(data=True):
            nodes.append({'id': node, **data})
        
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({'from': u, 'to': v, **data})
        
        return {'nodes': nodes, 'edges': edges}

@dataclass
class SceneNarrative:
    """The story unfolding in the scene"""
    primary_activity: Optional[ActivityType] = None
    secondary_activities: List[ActivityType] = field(default_factory=list)
    
    # Social dynamics
    social_dynamics: List[Tuple[SocialDynamics, float]] = field(default_factory=list)
    group_structures: List[List[str]] = field(default_factory=list)  # Groups of people
    
    # Temporal aspects
    is_changing: bool = False
    change_rate: float = 0.0  # How fast scene is changing
    predicted_next: Optional[str] = None  # Predicted next event
    
    # Story elements
    tension_level: float = 0.0  # 0-1
    drama_level: float = 0.0
    mystery_level: float = 0.0
    humor_level: float = 0.0
    
    # Wednesday's interpretation
    what_is_happening: str = ""  # Text description
    what_is_unusual: List[str] = field(default_factory=list)
    what_is_hidden: List[str] = field(default_factory=list)  # Things being concealed
    what_might_happen_next: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'primary_activity': self.primary_activity.value if self.primary_activity else None,
            'tension_level': self.tension_level,
            'drama_level': self.drama_level,
            'mystery_level': self.mystery_level,
            'what_is_happening': self.what_is_happening,
            'what_is_unusual': self.what_is_unusual[:3],
            'what_might_happen_next': self.what_might_happen_next[:3]
        }

@dataclass
class SceneUnderstanding:
    """
    Holistic understanding of a visual scene.
    Wednesday sees the complete picture - objects, people, their relationships,
    the story unfolding, and what's hidden beneath the surface.
    """
    # Basic scene info
    scene_type: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Components
    objects: List[ObjectState] = field(default_factory=list)
    people: List[PersonState] = field(default_factory=list)
    
    # Spatial understanding
    spatial_graph: SpatialGraph = field(default_factory=SpatialGraph)
    layout_description: str = ""
    
    # Scene properties
    mood: SceneMood = SceneMood.PEACEFUL
    mood_confidence: float = 0.0
    activities: List[ActivityType] = field(default_factory=list)
    
    # Social understanding
    social_dynamics: List[Tuple[SocialDynamics, float]] = field(default_factory=list)
    groups: List[List[str]] = field(default_factory=list)  # Groups by person ID
    power_dynamics: Dict[str, float] = field(default_factory=dict)  # Person ID -> dominance (0-1)
    
    # Narrative
    narrative: SceneNarrative = field(default_factory=SceneNarrative)
    
    # Attention
    focus_points: List[Tuple[int, int]] = field(default_factory=list)  # Where to look
    salient_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    
    # Anomalies
    anomalies: List[str] = field(default_factory=list)
    points_of_interest: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    
    # Metadata
    frame_number: int = 0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'scene_type': self.scene_type,
            'confidence': self.confidence,
            'mood': self.mood.value,
            'mood_confidence': self.mood_confidence,
            'people_count': len(self.people),
            'object_count': len(self.objects),
            'primary_activity': self.narrative.primary_activity.value if self.narrative.primary_activity else None,
            'tension_level': self.narrative.tension_level,
            'drama_level': self.narrative.drama_level,
            'anomaly_count': len(self.anomalies),
            'threat_count': len(self.threats),
            'frame_number': self.frame_number,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SceneUnderstanding':
        """Create SceneUnderstanding from dictionary"""
        narrative = SceneNarrative(
            primary_activity=ActivityType(data['primary_activity']) if data.get('primary_activity') else None,
            tension_level=data.get('tension_level', 0.0),
            drama_level=data.get('drama_level', 0.0),
            what_is_happening=data.get('what_is_happening', '')
        )
        
        return cls(
            scene_type=data.get('scene_type', 'unknown'),
            confidence=data.get('confidence', 0.0),
            mood=SceneMood(data.get('mood', 'peaceful')),
            mood_confidence=data.get('mood_confidence', 0.0),
            people=[],  # Would need separate loading
            objects=[],  # Would need separate loading
            narrative=narrative,
            frame_number=data.get('frame_number', 0),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now()
        )

class SceneUnderstandingSystem:
    """
    Holistic scene understanding - integrates object recognition, face processing,
    and spatial analysis to comprehend complete visual scenes.
    
    Wednesday sees not just what is there, but what it means.
    The tension in a room, the dynamics between people, the story unfolding.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Component modules (injected or created)
        self.object_recognizer: Optional[ObjectRecognizer] = None
        self.face_processor: Optional[FaceProcessor] = None
        
        # Scene memory for temporal analysis
        self.scene_memory = []
        self.max_memory_size = 30
        self.previous_scene: Optional[SceneUnderstanding] = None
        
        # Knowledge bases
        self.spatial_knowledge = self._load_spatial_knowledge()
        self.social_knowledge = self._load_social_knowledge()
        self.activity_models = self._load_activity_models()
        self.scene_scripts = self._load_scene_scripts()  # Common scene patterns
        
        # Groups tracking
        self.groups: List[Set[str]] = []
        
        # Performance tracking
        self.stats = {
            'total_scenes': 0,
            'avg_processing_time': 0.0,
            'errors': 0,
            'moods_detected': defaultdict(int)
        }
        
        logger.info("SceneUnderstandingSystem initialized")
    
    def set_components(self, 
                      object_recognizer: ObjectRecognizer,
                      face_processor: FaceProcessor):
        """Set vision components"""
        self.object_recognizer = object_recognizer
        self.face_processor = face_processor
        logger.info("Vision components connected")
    
    def understand(self,
                  image: np.ndarray,
                  frame_number: int = 0,
                  context: Optional[Dict] = None) -> SceneUnderstanding:
        """
        Develop comprehensive understanding of a scene.
        
        Args:
            image: Input image
            frame_number: Frame number for video
            context: Optional context (location, time, expectations)
            
        Returns:
            SceneUnderstanding with holistic interpretation
        """
        import time
        start_time = time.time()
        
        if not HAS_VISION or self.object_recognizer is None or self.face_processor is None:
            logger.error("Vision components not available")
            return SceneUnderstanding(scene_type="unknown", confidence=0.0)
        
        if image is None or image.size == 0:
            logger.warning("Empty image provided")
            return SceneUnderstanding(scene_type="unknown", confidence=0.0)
        
        try:
            # Run object recognition
            object_scene = self.object_recognizer.recognize(image, frame_number, context)
            
            # Run face processing
            faces = self.face_processor.process_frame(image, frame_number, context)
            
            # Convert to our data structures
            objects = self._process_objects(object_scene.objects if hasattr(object_scene, 'objects') else [])
            people = self._process_people(faces)
            
            # Build spatial graph
            spatial_graph = self._build_spatial_graph(objects, people)
            
            # Analyze spatial relationships
            spatial_relations = self._analyze_spatial_relations(spatial_graph)
            
            # Detect activities
            activities = self._detect_activities(people, objects, spatial_relations)
            
            # Analyze social dynamics
            social_dynamics, groups, power = self._analyze_social_dynamics(
                people, spatial_relations, context
            )
            self.groups = [set(g) for g in groups]
            
            # Determine scene mood
            mood, mood_conf = self._determine_mood(
                people, objects, activities, social_dynamics, context
            )
            
            # Build narrative
            narrative = self._build_narrative(
                people, objects, activities, social_dynamics, mood,
                spatial_relations, context
            )
            
            # Identify anomalies and points of interest
            anomalies = self._identify_anomalies(
                people, objects, spatial_relations, context
            )
            points_of_interest = self._identify_points_of_interest(
                people, objects, activities, narrative
            )
            threats = self._identify_threats(
                people, objects, activities, social_dynamics
            )
            
            # Determine focus points (where Wednesday should look)
            focus_points = self._determine_focus_points(
                people, objects, anomalies, threats, context
            )
            
            # Create scene understanding
            scene = SceneUnderstanding(
                scene_type=object_scene.scene_type.value if hasattr(object_scene, 'scene_type') and object_scene.scene_type else "unknown",
                confidence=object_scene.confidence if hasattr(object_scene, 'confidence') else 0.0,
                objects=objects,
                people=people,
                spatial_graph=spatial_graph,
                layout_description=self._describe_layout(objects, spatial_relations),
                mood=mood,
                mood_confidence=mood_conf,
                activities=activities,
                social_dynamics=social_dynamics,
                groups=groups,
                power_dynamics=power,
                narrative=narrative,
                focus_points=focus_points,
                salient_regions=self._find_salient_regions(people, objects, anomalies),
                anomalies=anomalies,
                points_of_interest=points_of_interest,
                threats=threats,
                frame_number=frame_number,
                processing_time=time.time() - start_time
            )
            
            # Add temporal context
            if self.previous_scene is not None:
                scene = self._add_temporal_context(scene, self.previous_scene)
            
            # Store in memory
            self.scene_memory.append(scene)
            if len(self.scene_memory) > self.max_memory_size:
                self.scene_memory.pop(0)
            self.previous_scene = scene
            
            # Update stats
            self._update_stats(scene.processing_time, mood)
            
            return scene
            
        except Exception as e:
            logger.error(f"Error in scene understanding: {e}", exc_info=True)
            self.stats['errors'] += 1
            return SceneUnderstanding(scene_type="unknown", confidence=0.0)
    
    def _process_objects(self, detected_objects: List[DetectedObject]) -> List[ObjectState]:
        """Convert detected objects to object states"""
        objects = []
        
        for i, obj in enumerate(detected_objects):
            x1, y1, x2, y2 = obj.bounding_box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            state = ObjectState(
                detection=obj,
                position=center,
                bounding_box=obj.bounding_box,
                is_in_use=False,  # Will be determined later
                is_out_of_place=self._check_if_out_of_place(obj),
                object_id=f"obj_{i}_{id(obj)}"
            )
            
            objects.append(state)
        
        return objects
    
    def _process_people(self, face_analyses: List[FaceAnalysis]) -> List[PersonState]:
        """Convert face analyses to person states"""
        people = []
        
        for i, face in enumerate(face_analyses):
            x1, y1, x2, y2 = face.bounding_box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            person = PersonState(
                face_analysis=face,
                position=center,
                bounding_box=face.bounding_box,
                attention_focus=self._estimate_attention_focus(face, center),
                person_id=face.face_id if hasattr(face, 'face_id') else f"person_{i}"
            )
            
            people.append(person)
        
        return people
    
    def _build_spatial_graph(self, 
                            objects: List[ObjectState], 
                            people: List[PersonState]) -> SpatialGraph:
        """Build graph of spatial relationships"""
        graph = SpatialGraph()
        
        if not HAS_NETWORKX:
            return graph
        
        # Add all nodes
        for obj in objects:
            graph.add_object(obj.object_id, obj.detection.category.value, obj.position)
        
        for person in people:
            graph.add_person(person.person_id, person.position)
        
        # Add spatial relations
        all_nodes = []
        if graph.graph is not None:
            all_nodes = list(graph.graph.nodes)
        
        for i, node1 in enumerate(all_nodes):
            for node2 in all_nodes[i+1:]:
                pos1 = graph.graph.nodes[node1]['position']
                pos2 = graph.graph.nodes[node2]['position']
                
                distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                # Determine relation based on position and distance
                relation = self._determine_spatial_relation(pos1, pos2, distance)
                
                graph.add_relation(node1, node2, relation, distance)
        
        return graph
    
    def _determine_spatial_relation(self, 
                                   pos1: Tuple[int, int], 
                                   pos2: Tuple[int, int],
                                   distance: float) -> SpatialRelation:
        """Determine spatial relation between two positions"""
        if distance < 20:
            return SpatialRelation.TOUCHING
        elif distance < 50:
            return SpatialRelation.NEAR
        elif distance > 300:
            return SpatialRelation.FAR
        
        # Determine relative direction
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        if abs(dx) > abs(dy):
            return SpatialRelation.RIGHT_OF if dx > 0 else SpatialRelation.LEFT_OF
        else:
            return SpatialRelation.IN_FRONT if dy < 0 else SpatialRelation.BEHIND
    
    def _analyze_spatial_relations(self, graph: SpatialGraph) -> Dict:
        """Analyze spatial relationships in graph"""
        analysis = {
            'clusters': [],  # Clusters of objects/people
            'isolated': [],  # Isolated entities
            'facing_pairs': [],  # Pairs facing each other
        }
        
        if not HAS_NETWORKX or graph.graph is None or len(graph.graph.nodes) <= 1:
            return analysis
        
        # Find clusters using simple distance-based clustering
        nodes = list(graph.graph.nodes)
        clusters = []
        used = set()
        
        for node in nodes:
            if node in used:
                continue
            
            cluster = [node]
            used.add(node)
            
            # Find nearby nodes
            for other in nodes:
                if other in used:
                    continue
                
                if graph.graph.has_edge(node, other):
                    edge_data = graph.graph.get_edge_data(node, other)
                    if edge_data['distance'] < 100:  # Threshold for cluster
                        cluster.append(other)
                        used.add(other)
            
            if len(cluster) > 1:
                clusters.append(cluster)
            else:
                analysis['isolated'].append(node)
        
        analysis['clusters'] = clusters
        
        return analysis
    
    def _detect_activities(self,
                          people: List[PersonState],
                          objects: List[ObjectState],
                          spatial_relations: Dict) -> List[ActivityType]:
        """Detect activities of people in scene"""
        activities = []
        
        for person in people:
            # Check for common activities
            activity = self._classify_person_activity(person, objects, spatial_relations)
            person.activity = activity
            activities.append(activity)
        
        return activities
    
    def _classify_person_activity(self,
                                 person: PersonState,
                                 objects: List[ObjectState],
                                 spatial_relations: Dict) -> ActivityType:
        """Classify a person's activity"""
        # Check if interacting with objects
        for obj in objects:
            dist = math.sqrt(
                (person.position[0] - obj.position[0])**2 + 
                (person.position[1] - obj.position[1])**2
            )
            
            if dist < 50:  # Close to object
                obj_type = obj.detection.category.value
                
                # Object-based activities
                if obj_type in ['chair', 'sofa']:
                    return ActivityType.SITTING
                elif obj_type in ['bed']:
                    return ActivityType.SLEEPING
                elif obj_type in ['book', 'notebook', 'magazine']:
                    return ActivityType.READING
                elif obj_type in ['tv', 'monitor']:
                    return ActivityType.WATCHING_TV
                elif obj_type in ['cup', 'bottle', 'plate', 'bowl']:
                    return ActivityType.EATING
                elif obj_type in ['laptop', 'keyboard', 'desk']:
                    return ActivityType.WORKING
        
        # Check facial expression for clues
        expr = person.face_analysis.primary_expression if hasattr(person.face_analysis, 'primary_expression') else FacialExpression.NEUTRAL
        
        if expr in [FacialExpression.ANGER, FacialExpression.SUSPICION]:
            return ActivityType.ARGUING
        elif expr == FacialExpression.HAPPINESS:
            return ActivityType.SOCIALIZING
        elif expr == FacialExpression.FEAR:
            return ActivityType.FLEEING
        elif expr == FacialExpression.SADNESS:
            return ActivityType.IGNORING
        
        # Default based on posture
        return ActivityType.STANDING
    
    def _analyze_social_dynamics(self,
                                people: List[PersonState],
                                spatial_relations: Dict,
                                context: Optional[Dict]) -> Tuple[List, List, Dict]:
        """Analyze social dynamics between people"""
        dynamics = []
        groups = []
        power = {}
        
        if len(people) < 2:
            return dynamics, groups, power
        
        # Look for pairs in close proximity
        for i, p1 in enumerate(people):
            for p2 in people[i+1:]:
                dist = math.sqrt(
                    (p1.position[0] - p2.position[0])**2 + 
                    (p1.position[1] - p2.position[1])**2
                )
                
                if dist < 100:  # Close enough for interaction
                    # Determine dynamics based on expressions and orientation
                    expr1 = p1.face_analysis.primary_expression if hasattr(p1.face_analysis, 'primary_expression') else FacialExpression.NEUTRAL
                    expr2 = p2.face_analysis.primary_expression if hasattr(p2.face_analysis, 'primary_expression') else FacialExpression.NEUTRAL
                    
                    # Check if they're facing each other (simplified)
                    facing = self._are_facing_each_other(p1, p2)
                    
                    if not facing:
                        dynamics.append((SocialDynamics.IGNORING, 0.7))
                    elif expr1 == FacialExpression.ANGER or expr2 == FacialExpression.ANGER:
                        dynamics.append((SocialDynamics.ARGUMENT, 0.8))
                    elif expr1 == FacialExpression.HAPPINESS and expr2 == FacialExpression.HAPPINESS:
                        dynamics.append((SocialDynamics.CONVERSATION, 0.7))
                    elif expr1 == FacialExpression.FEAR or expr2 == FacialExpression.FEAR:
                        dynamics.append((SocialDynamics.THREATENING, 0.8))
                    elif expr1 == FacialExpression.SUSPICION or expr2 == FacialExpression.SUSPICION:
                        dynamics.append((SocialDynamics.WATCHING, 0.6))
                    
                    # Add to group
                    p1_id = p1.person_id
                    p2_id = p2.person_id
                    
                    group_found = False
                    for group in groups:
                        group_set = set(group)
                        if p1_id in group_set or p2_id in group_set:
                            if p1_id not in group_set:
                                group.append(p1_id)
                            if p2_id not in group_set:
                                group.append(p2_id)
                            group_found = True
                            break
                    
                    if not group_found:
                        groups.append([p1_id, p2_id])
        
        # Analyze power dynamics
        for person in people:
            # Power indicators: expression, posture, who they're looking at
            power_score = 0.5  # Baseline
            
            expr = person.face_analysis.primary_expression if hasattr(person.face_analysis, 'primary_expression') else FacialExpression.NEUTRAL
            
            if expr == FacialExpression.ANGER:
                power_score += 0.2
            elif expr == FacialExpression.FEAR:
                power_score -= 0.2
            elif expr == FacialExpression.HAPPINESS:
                power_score += 0.1  # Confident happiness
            
            # Looking down can indicate submission
            gaze = person.face_analysis.eyes.gaze_direction if hasattr(person.face_analysis, 'eyes') else GazeDirection.TOWARDS_CAMERA
            if gaze == GazeDirection.DOWN:
                power_score -= 0.1
            
            power[person.person_id] = max(0.0, min(1.0, power_score))
        
        return dynamics, groups, power
    
    def _are_facing_each_other(self, p1: PersonState, p2: PersonState) -> bool:
        """Check if two people are facing each other"""
        # Simplified - would use head pose
        # For now, check if they're looking towards each other's direction
        gaze1 = p1.face_analysis.eyes.gaze_direction if hasattr(p1.face_analysis, 'eyes') else GazeDirection.TOWARDS_CAMERA
        gaze2 = p2.face_analysis.eyes.gaze_direction if hasattr(p2.face_analysis, 'eyes') else GazeDirection.TOWARDS_CAMERA
        
        # Very simplified - assumes they're facing each other if both look towards camera
        # In reality, would need head pose and relative positions
        return gaze1 == GazeDirection.TOWARDS_CAMERA and gaze2 == GazeDirection.TOWARDS_CAMERA
    
    def _determine_mood(self,
                       people: List[PersonState],
                       objects: List[ObjectState],
                       activities: List[ActivityType],
                       social_dynamics: List[Tuple[SocialDynamics, float]],
                       context: Optional[Dict]) -> Tuple[SceneMood, float]:
        """Determine overall mood of the scene"""
        mood_scores = {mood: 0.0 for mood in SceneMood}
        
        # Mood from people's expressions
        for person in people:
            expr = person.face_analysis.primary_expression if hasattr(person.face_analysis, 'primary_expression') else FacialExpression.NEUTRAL
            
            if expr == FacialExpression.HAPPINESS:
                mood_scores[SceneMood.JOYFUL] += 0.3
            elif expr == FacialExpression.ANGER:
                mood_scores[SceneMood.HOSTILE] += 0.3
                mood_scores[SceneMood.TENSE] += 0.2
            elif expr == FacialExpression.FEAR:
                mood_scores[SceneMood.DANGEROUS] += 0.3
                mood_scores[SceneMood.TENSE] += 0.2
            elif expr == FacialExpression.SADNESS:
                mood_scores[SceneMood.MELANCHOLIC] += 0.3
            elif expr == FacialExpression.SUSPICION:
                mood_scores[SceneMood.SUSPICIOUS] += 0.3
                mood_scores[SceneMood.MYSTERIOUS] += 0.2
            elif expr == FacialExpression.SURPRISE:
                mood_scores[SceneMood.MYSTERIOUS] += 0.2
        
        # Mood from social dynamics
        for dyn, conf in social_dynamics:
            if dyn == SocialDynamics.ARGUMENT:
                mood_scores[SceneMood.TENSE] += 0.3
                mood_scores[SceneMood.HOSTILE] += 0.2
            elif dyn == SocialDynamics.THREATENING:
                mood_scores[SceneMood.DANGEROUS] += 0.4
                mood_scores[SceneMood.TENSE] += 0.3
            elif dyn == SocialDynamics.CONVERSATION:
                mood_scores[SceneMood.PEACEFUL] += 0.2
                mood_scores[SceneMood.JOYFUL] += 0.1
        
        # Mood from number of people/objects
        if len(people) > 5:
            mood_scores[SceneMood.CHAOTIC] += 0.2
        elif len(people) == 0:
            mood_scores[SceneMood.BORING] += 0.2
            mood_scores[SceneMood.MYSTERIOUS] += 0.1  # Empty rooms can be mysterious
        
        # Get top mood
        if mood_scores:
            top_mood = max(mood_scores.items(), key=lambda x: x[1])
            confidence = min(1.0, top_mood[1])
            return top_mood[0], confidence
        
        return SceneMood.PEACEFUL, 0.5
    
    def _build_narrative(self,
                        people: List[PersonState],
                        objects: List[ObjectState],
                        activities: List[ActivityType],
                        social_dynamics: List[Tuple[SocialDynamics, float]],
                        mood: SceneMood,
                        spatial_relations: Dict,
                        context: Optional[Dict]) -> SceneNarrative:
        """Build a narrative understanding of the scene"""
        narrative = SceneNarrative()
        
        # Determine primary activity (most common)
        if activities:
            activity_counts = Counter(activities)
            narrative.primary_activity = activity_counts.most_common(1)[0][0]
            narrative.secondary_activities = [
                a for a, c in activity_counts.most_common(3)[1:]
            ]
        
        # Set social dynamics
        narrative.social_dynamics = social_dynamics
        
        # Set group structures
        narrative.group_structures = self.groups
        
        # Calculate tension level
        tension = 0.0
        
        # Tension from expressions
        for person in people:
            expr = person.face_analysis.primary_expression if hasattr(person.face_analysis, 'primary_expression') else FacialExpression.NEUTRAL
            if expr in [FacialExpression.ANGER, FacialExpression.FEAR, FacialExpression.SUSPICION]:
                tension += 0.1
        
        # Tension from dynamics
        for dyn, conf in social_dynamics:
            if dyn in [SocialDynamics.ARGUMENT, SocialDynamics.THREATENING]:
                tension += conf * 0.3
        
        narrative.tension_level = min(1.0, tension)
        
        # Drama from number of people and intensity
        narrative.drama_level = min(1.0, (len(people) * 0.1 + narrative.tension_level * 0.5))
        
        # Mystery from unusual objects or suspicious expressions
        mystery = 0.0
        for person in people:
            expr = person.face_analysis.primary_expression if hasattr(person.face_analysis, 'primary_expression') else FacialExpression.NEUTRAL
            if expr == FacialExpression.SUSPICION:
                mystery += 0.2
        
        # Check for unusual objects
        unusual_objects = [obj for obj in objects if obj.is_out_of_place]
        mystery += len(unusual_objects) * 0.1
        
        narrative.mystery_level = min(1.0, mystery)
        
        # Generate narrative descriptions
        narrative.what_is_happening = self._generate_scene_description(
            people, objects, activities, social_dynamics, mood
        )
        
        # Identify unusual elements
        if unusual_objects:
            for obj in unusual_objects[:3]:
                narrative.what_is_unusual.append(
                    f"{obj.detection.category.value} is out of place"
                )
        
        # Predict what might happen next
        narrative.what_might_happen_next = self._predict_next_events(
            people, activities, social_dynamics, mood
        )
        
        return narrative
    
    def _generate_scene_description(self,
                                   people: List[PersonState],
                                   objects: List[ObjectState],
                                   activities: List[ActivityType],
                                   social_dynamics: List[Tuple[SocialDynamics, float]],
                                   mood: SceneMood) -> str:
        """Generate a text description of the scene"""
        parts = []
        
        # People count
        if len(people) == 0:
            parts.append("An empty room")
        elif len(people) == 1:
            person = people[0]
            activity = person.activity.value if person.activity else "present"
            parts.append(f"A person {activity}")
        else:
            parts.append(f"{len(people)} people")
        
        # Mood
        parts.append(f"The atmosphere is {mood.value}")
        
        # Activities
        if activities:
            activity_counts = Counter(activities)
            primary = activity_counts.most_common(1)[0][0].value
            parts.append(f"They appear to be {primary}")
        
        # Social dynamics
        if social_dynamics:
            top_dyn = max(social_dynamics, key=lambda x: x[1])[0].value
            parts.append(f"The dynamic seems {top_dyn}")
        
        return ". ".join(parts) + "."
    
    def _predict_next_events(self,
                            people: List[PersonState],
                            activities: List[ActivityType],
                            social_dynamics: List[Tuple[SocialDynamics, float]],
                            mood: SceneMood) -> List[str]:
        """Predict what might happen next in the scene"""
        predictions = []
        
        # If people are arguing, might escalate
        if any(dyn[0] == SocialDynamics.ARGUMENT for dyn in social_dynamics):
            predictions.append("Argument might escalate")
            predictions.append("Someone might leave in anger")
        
        # If someone looks suspicious, they might do something
        for person in people:
            expr = person.face_analysis.primary_expression if hasattr(person.face_analysis, 'primary_expression') else FacialExpression.NEUTRAL
            if expr == FacialExpression.SUSPICION:
                predictions.append("Suspicious person might act")
                break
        
        # If scene is tense, something might happen
        if mood in [SceneMood.TENSE, SceneMood.DANGEROUS]:
            predictions.append("Tension might break")
        
        # If people are watching TV, they'll keep watching
        if ActivityType.WATCHING_TV in activities:
            predictions.append("Continue watching")
        
        # If someone is eating, they might finish soon
        if ActivityType.EATING in activities:
            predictions.append("Meal might end")
        
        return predictions[:3]  # Top 3 predictions
    
    def _identify_anomalies(self,
                           people: List[PersonState],
                           objects: List[ObjectState],
                           spatial_relations: Dict,
                           context: Optional[Dict]) -> List[str]:
        """Identify anomalies in the scene"""
        anomalies = []
        
        # Check for out-of-place objects
        for obj in objects:
            if obj.is_out_of_place:
                anomalies.append(f"Out of place: {obj.detection.category.value}")
        
        # Check for unusual behavior
        for person in people:
            # Person alone but looking around nervously
            if len(people) == 1:
                expr = person.face_analysis.primary_expression if hasattr(person.face_analysis, 'primary_expression') else FacialExpression.NEUTRAL
                if expr == FacialExpression.FEAR:
                    anomalies.append("Person seems fearful while alone")
            
            # Person with suspicious expression
            if expr == FacialExpression.SUSPICION:
                anomalies.append("Person appears suspicious")
        
        # Check context-based anomalies
        if context and 'expected_objects' in context:
            expected = set(context['expected_objects'])
            present = {obj.detection.category.value for obj in objects}
            
            missing = expected - present
            for m in missing:
                anomalies.append(f"Missing expected: {m}")
        
        return anomalies
    
    def _identify_points_of_interest(self,
                                    people: List[PersonState],
                                    objects: List[ObjectState],
                                    activities: List[ActivityType],
                                    narrative: SceneNarrative) -> List[str]:
        """Identify points of interest in the scene"""
        points = []
        
        # People with strong expressions
        for person in people:
            expr_intensity = person.face_analysis.expression_intensity if hasattr(person.face_analysis, 'expression_intensity') else 0.0
            if expr_intensity > 0.7:
                expr = person.face_analysis.primary_expression.value if hasattr(person.face_analysis, 'primary_expression') else "unknown"
                points.append(f"Person showing strong {expr}")
        
        # Unusual objects
        for obj in objects:
            if obj.is_out_of_place:
                points.append(f"Unusual object: {obj.detection.category.value}")
        
        # Where people are looking (simplified)
        gaze_targets = defaultdict(int)
        for person in people:
            if person.attention_focus:
                gaze_targets[person.attention_focus] += 1
        
        for target, count in gaze_targets.items():
            if count > 1:
                points.append(f"Multiple people looking at {target}")
        
        return points
    
    def _identify_threats(self,
                         people: List[PersonState],
                         objects: List[ObjectState],
                         activities: List[ActivityType],
                         social_dynamics: List[Tuple[SocialDynamics, float]]) -> List[str]:
        """Identify potential threats in the scene"""
        threats = []
        
        # Check for threatening objects
        threat_objects = ['knife', 'scissors', 'gun']  # Would be more comprehensive
        for obj in objects:
            if obj.detection.category.value in threat_objects:
                threats.append(f"Potential weapon: {obj.detection.category.value}")
        
        # Check for threatening behavior
        for dyn, conf in social_dynamics:
            if dyn == SocialDynamics.THREATENING and conf > 0.7:
                threats.append("Threatening interaction between people")
        
        # Check for angry people
        for person in people:
            expr = person.face_analysis.primary_expression if hasattr(person.face_analysis, 'primary_expression') else FacialExpression.NEUTRAL
            expr_intensity = person.face_analysis.expression_intensity if hasattr(person.face_analysis, 'expression_intensity') else 0.0
            
            if expr == FacialExpression.ANGER and expr_intensity > 0.8:
                threats.append("Very angry person")
        
        return threats
    
    def _determine_focus_points(self,
                               people: List[PersonState],
                               objects: List[ObjectState],
                               anomalies: List[str],
                               threats: List[str],
                               context: Optional[Dict]) -> List[Tuple[int, int]]:
        """Determine where Wednesday should focus her attention"""
        focus_points = []
        
        # Priority 1: Threats
        if threats and people:
            # Look at threatening people
            for person in people:
                expr = person.face_analysis.primary_expression if hasattr(person.face_analysis, 'primary_expression') else FacialExpression.NEUTRAL
                if expr == FacialExpression.ANGER:
                    focus_points.append(person.position)
                    break
        
        # Priority 2: Anomalies
        if anomalies and objects:
            # Look at anomalous objects
            for obj in objects:
                if obj.is_out_of_place:
                    focus_points.append(obj.position)
                    break
        
        # Priority 3: People with strong expressions
        for person in people:
            expr_intensity = person.face_analysis.expression_intensity if hasattr(person.face_analysis, 'expression_intensity') else 0.0
            if expr_intensity > 0.7:
                focus_points.append(person.position)
                break
        
        # Priority 4: Where people are looking (would need mapping)
        
        return focus_points[:3]  # Top 3 focus points
    
    def _find_salient_regions(self,
                             people: List[PersonState],
                             objects: List[ObjectState],
                             anomalies: List[str]) -> List[Tuple[int, int, int, int]]:
        """Find salient regions in the image"""
        regions = []
        
        # People are always salient
        for person in people:
            regions.append(person.bounding_box)
        
        # Anomalous objects are salient
        for obj in objects:
            if obj.is_out_of_place:
                regions.append(obj.bounding_box)
        
        return regions[:5]  # Top 5 regions
    
    def _check_if_out_of_place(self, obj: DetectedObject) -> bool:
        """Check if an object seems out of place"""
        # Would use spatial knowledge and context
        # Simplified implementation - always returns False for now
        return False
    
    def _estimate_attention_focus(self, 
                                 face: FaceAnalysis,
                                 person_position: Tuple[int, int]) -> Optional[str]:
        """Estimate what a person is looking at"""
        if not hasattr(face, 'eyes') or not hasattr(face.eyes, 'gaze_direction'):
            return None
        
        gaze = face.eyes.gaze_direction
        
        if gaze == GazeDirection.TOWARDS_CAMERA:
            return "camera"
        elif gaze == GazeDirection.LEFT:
            return "left_side"
        elif gaze == GazeDirection.RIGHT:
            return "right_side"
        elif gaze == GazeDirection.UP:
            return "above"
        elif gaze == GazeDirection.DOWN:
            return "below"
        elif gaze == GazeDirection.AWAY_CAMERA:
            return "away"
        
        return None
    
    def _describe_layout(self, 
                        objects: List[ObjectState], 
                        spatial_relations: Dict) -> str:
        """Generate a description of the spatial layout"""
        if not objects:
            return "Empty space"
        
        # Count objects by type
        obj_types = Counter([obj.detection.category.value for obj in objects])
        
        # Generate description
        parts = []
        
        # Major furniture
        furniture = ['chair', 'table', 'sofa', 'bed', 'desk', 'cabinet']
        present_furniture = [f for f in furniture if f in obj_types]
        if present_furniture:
            parts.append(f"Contains {', '.join(present_furniture)}")
        
        # Clutter assessment
        if len(objects) > 20:
            parts.append("Somewhat cluttered")
        elif len(objects) > 10:
            parts.append("Moderately furnished")
        else:
            parts.append("Sparsely furnished")
        
        return ". ".join(parts)
    
    def _add_temporal_context(self,
                             current: SceneUnderstanding,
                             previous: SceneUnderstanding) -> SceneUnderstanding:
        """Add temporal context from previous scene"""
        # Detect if scene is changing
        if len(current.people) != len(previous.people):
            current.narrative.is_changing = True
            current.narrative.change_rate = 0.5
        
        # Check for new people
        current_ids = {p.person_id for p in current.people}
        prev_ids = {p.person_id for p in previous.people}
        
        new_people = current_ids - prev_ids
        if new_people:
            current.narrative.what_might_happen_next.append(
                f"New person arrived"
            )
        
        # Check for people leaving
        left_people = prev_ids - current_ids
        if left_people:
            current.narrative.what_might_happen_next.append(
                f"Someone left"
            )
        
        # Check for mood changes
        if current.mood != previous.mood:
            current.narrative.what_might_happen_next.append(
                f"Mood shifted from {previous.mood.value} to {current.mood.value}"
            )
        
        return current
    
    def _load_spatial_knowledge(self) -> Dict:
        """Load knowledge about spatial arrangements"""
        return {
            'typical_room_layouts': {
                'living_room': ['sofa', 'tv', 'coffee_table', 'chair'],
                'bedroom': ['bed', 'nightstand', 'dresser', 'lamp'],
                'kitchen': ['refrigerator', 'oven', 'sink', 'cabinet'],
                'office': ['desk', 'chair', 'computer', 'bookshelf'],
                'bathroom': ['toilet', 'sink', 'shower', 'mirror'],
            },
            'object_placement': {
                'book': ['shelf', 'table', 'desk'],
                'cup': ['table', 'desk', 'counter'],
                'laptop': ['desk', 'table', 'lap'],
                'phone': ['table', 'bed', 'hand'],
                'remote': ['table', 'sofa', 'hand'],
            }
        }
    
    def _load_social_knowledge(self) -> Dict:
        """Load knowledge about social interactions"""
        return {
            'personal_space': 50,  # pixels
            'intimate_space': 20,  # pixels
            'social_space': 100,   # pixels
            'public_space': 300,    # pixels
            
            'conversation_distance': (50, 150),
            'argument_distance': (30, 80),
            'avoidance_distance': (200, 500),
        }
    
    def _load_activity_models(self) -> Dict:
        """Load models for activity recognition"""
        return {
            'sitting': {
                'posture': 'seated',
                'typical_objects': ['chair', 'sofa'],
                'duration': 'variable'
            },
            'reading': {
                'posture': 'seated_or_standing',
                'typical_objects': ['book', 'notebook', 'kindle'],
                'head_position': 'down'
            },
            'watching_tv': {
                'posture': 'seated_or_lying',
                'typical_objects': ['tv', 'monitor'],
                'gaze': 'towards_tv'
            },
            'eating': {
                'posture': 'seated',
                'typical_objects': ['plate', 'cup', 'fork', 'spoon', 'bowl'],
                'hand_movement': 'to_mouth'
            },
            'working': {
                'posture': 'seated',
                'typical_objects': ['laptop', 'keyboard', 'mouse', 'desk'],
                'hand_movement': 'typing'
            }
        }
    
    def _load_scene_scripts(self) -> Dict:
        """Load common scene scripts (sequences of events)"""
        return {
            'arrival': {
                'sequence': ['door_open', 'person_enter', 'greeting'],
                'typical_mood': 'joyful',
                'duration': 'short'
            },
            'departure': {
                'sequence': ['stand_up', 'gather_items', 'wave', 'door_close'],
                'typical_mood': 'bittersweet',
                'duration': 'short'
            },
            'argument': {
                'sequence': ['tense_atmosphere', 'raised_voices', 'angry_expressions', 'storm_out'],
                'typical_mood': 'hostile',
                'duration': 'medium'
            },
            'relaxing': {
                'sequence': ['sit_down', 'turn_on_tv', 'get_comfortable'],
                'typical_mood': 'peaceful',
                'duration': 'long'
            },
            'dining': {
                'sequence': ['sit_at_table', 'food_served', 'eating', 'clean_up'],
                'typical_mood': 'joyful',
                'duration': 'medium'
            }
        }
    
    def _update_stats(self, processing_time: float, mood: SceneMood):
        """Update processing statistics"""
        self.stats['total_scenes'] += 1
        
        # Update average processing time
        total = self.stats['total_scenes']
        old_avg = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = old_avg + (processing_time - old_avg) / total
        
        # Track moods
        self.stats['moods_detected'][mood.value] += 1
    
    def get_stats(self) -> Dict:
        """Return processing statistics"""
        stats = dict(self.stats)
        stats['moods_detected'] = dict(stats['moods_detected'])
        return stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics"""
        self.stats = {
            'total_scenes': 0,
            'avg_processing_time': 0.0,
            'errors': 0,
            'moods_detected': defaultdict(int)
        }

# Connects to: object_recognition.py (object detection)
# Connects to: face_processing.py (face analysis)
# Connects to: perception/attention/salience.py (focus points)
# Connects to: memory/episodic/ (stores scene memories)
# Connects to: memory/semantic/ (scene knowledge)
# Connects to: cognition/reasoning.py (scene reasoning)
# Connects to: emotion/appraisal.py (scene mood affects emotion)
# Connects to: self/theory_of_mind.py (understanding social dynamics)