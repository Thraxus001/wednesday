"""
Interaction History - Tracks the sequence of exchanges between Wednesday and users.
Like a chat log but with metadata, intent tracking, and response evaluation.
Sits between working memory (current session) and episodic memory (long-term experiences).
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import uuid
import json
from pathlib import Path
import logging
from enum import Enum
from collections import defaultdict

# Configure logger
logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of interactions in the conversation."""
    USER_INPUT = "user_input"
    WEDNESDAY_RESPONSE = "wednesday_response"
    SYSTEM_EVENT = "system_event"
    ERROR = "error"
    LEARNING_MOMENT = "learning_moment"


class InteractionStatus(Enum):
    """How well the interaction went."""
    SUCCESS = "success"  # User got what they wanted
    PARTIAL = "partial"  # Partially successful
    FAILURE = "failure"  # Didn't work
    CLARIFICATION = "clarification"  # Needed to ask for clarification
    CONFUSION = "confusion"  # Wednesday was confused
    
    @classmethod
    def from_string(cls, status_str: str):
        """Get enum member from string value."""
        for member in cls:
            if member.value == status_str:
                return member
        return cls.SUCCESS


class FeedbackType(Enum):
    """User feedback on interactions."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    
    @classmethod
    def from_string(cls, feedback_str: str):
        """Get enum member from string value."""
        for member in cls:
            if member.value == feedback_str:
                return member
        return cls.NEUTRAL


# Type aliases for better code clarity
InteractionDict = Dict[str, Any]
SessionMetadata = Dict[str, Any]


class InteractionHistory:
    """
    Tracks the complete history of interactions in a conversation.
    Maintains turn-by-turn exchange records with metadata about
    intent, success, emotional tone, and learning opportunities.
    
    Features:
    - Session-based organization of conversations
    - Persistent storage of interactions
    - Multiple indices for efficient querying
    - Automatic archiving of old interactions
    - Statistics tracking and feedback collection
    """
    
    # Class constants
    DEFAULT_STORAGE_PATH = Path("./data/interactions")
    MAX_HISTORY_PER_SESSION = 1000
    MAX_RECENT_SESSIONS_PER_USER = 10
    DEFAULT_CONTEXT_TURNS = 5
    DEFAULT_SEARCH_LIMIT = 50
    SENTIMENT_RANGE = (-1.0, 1.0)  # Min and max sentiment values
    
    def __init__(self, 
                 storage_path: Optional[Path] = None, 
                 max_history_per_session: int = MAX_HISTORY_PER_SESSION):
        """
        Initialize the interaction history system.
        
        Args:
            storage_path: Directory for persistent storage
            max_history_per_session: Maximum interactions to keep in memory per session
        """
        self.storage_path = storage_path or self.DEFAULT_STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_history_per_session = max_history_per_session
        
        # In-memory cache organized by session
        self.sessions: Dict[str, List[InteractionDict]] = defaultdict(list)
        self.current_session_id: Optional[str] = None
        
        # Session metadata cache
        self.session_metadata: Dict[str, SessionMetadata] = {}
        
        # Indices for quick lookup
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)  # user_id -> session_ids
        self.interaction_count = 0
        
        # Load existing sessions metadata
        self._load_session_metadata()
        
        logger.info(f"InteractionHistory initialized at {self.storage_path}")
    
    def start_session(self, 
                     user_id: str = "anonymous", 
                     metadata: Optional[Dict] = None) -> str:
        """
        Start a new interaction session.
        
        Args:
            user_id: Who is interacting
            metadata: Session metadata (platform, location, etc.)
        
        Returns:
            session_id for the new session
        """
        session_id = str(uuid.uuid4())
        
        # Create session metadata
        session_start: SessionMetadata = {
            'session_id': session_id,
            'user_id': user_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'metadata': metadata or {},
            'interaction_count': 0,
            'topics': [],
            'success_rate': 1.0,  # Starts optimistic
            'average_sentiment': 0.0,
            'status': 'active'
        }
        
        # Initialize session storage
        self.sessions[session_id] = []
        self.session_metadata[session_id] = session_start
        self.current_session_id = session_id
        
        # Update user index
        self.user_sessions[user_id].append(session_id)
        
        # Save session start
        self._save_session_metadata(session_id, session_start)
        
        logger.info(f"Started session {session_id[:8]} for user {user_id}")
        return session_id
    
    def end_session(self, session_id: Optional[str] = None) -> bool:
        """
        End an active session and calculate final statistics.
        
        Args:
            session_id: Session to end (uses current if None)
        
        Returns:
            True if session was ended successfully
        """
        session_id = session_id or self.current_session_id
        
        if not session_id or session_id not in self.sessions:
            logger.warning(f"Cannot end session - session not found: {session_id}")
            return False
        
        # Get current metadata
        metadata = self.session_metadata.get(session_id, {})
        
        # Update with end time and final stats
        metadata['end_time'] = datetime.now().isoformat()
        metadata['status'] = 'ended'
        metadata['interaction_count'] = len(self.sessions[session_id])
        
        # Calculate session statistics
        if self.sessions[session_id]:
            self._calculate_session_stats(metadata, self.sessions[session_id])
        
        # Save updated metadata
        self._save_session_metadata(session_id, metadata)
        self.session_metadata[session_id] = metadata
        
        # Clear current session if it was the active one
        if session_id == self.current_session_id:
            self.current_session_id = None
        
        logger.info(f"Ended session {session_id[:8]} with {metadata['interaction_count']} interactions")
        return True
    
    def _calculate_session_stats(self, metadata: SessionMetadata, 
                                 interactions: List[InteractionDict]) -> None:
        """
        Calculate statistics for a completed session.
        
        Args:
            metadata: Session metadata to update
            interactions: List of interactions in the session
        """
        # Calculate average sentiment
        sentiments = [
            i.get('sentiment', 0) for i in interactions 
            if i.get('sentiment') is not None
        ]
        if sentiments:
            metadata['average_sentiment'] = sum(sentiments) / len(sentiments)
        
        # Calculate success rate
        successes = [
            i for i in interactions 
            if i.get('status') == InteractionStatus.SUCCESS.value
        ]
        if successes:
            metadata['success_rate'] = len(successes) / len(interactions)
        
        # Collect all topics
        all_topics = set()
        for interaction in interactions:
            all_topics.update(interaction.get('tags', []))
        metadata['topics'] = list(all_topics)
    
    def add_interaction(self,
                       user_input: str,
                       wednesday_response: str,
                       session_id: Optional[str] = None,
                       intent: Optional[str] = None,
                       status: InteractionStatus = InteractionStatus.SUCCESS,
                       sentiment: float = 0.0,
                       confidence: float = 1.0,
                       context: Optional[Dict] = None,
                       metadata: Optional[Dict] = None) -> str:
        """
        Add an interaction turn to the history.
        
        Args:
            user_input: What the user said
            wednesday_response: How Wednesday responded
            session_id: Which session (uses current if not specified)
            intent: Detected user intent
            status: How well the interaction went
            sentiment: Emotional tone (-1 to 1)
            confidence: How confident Wednesday was (0-1)
            context: Additional context about the interaction
            metadata: Any other metadata
        
        Returns:
            interaction_id for the new interaction
        
        Raises:
            ValueError: If sentiment or confidence are out of range
        """
        # Validate inputs
        if not self._validate_sentiment(sentiment):
            raise ValueError(f"Sentiment must be between {self.SENTIMENT_RANGE[0]} and {self.SENTIMENT_RANGE[1]}")
        
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        
        # Get or create session
        session_id = session_id or self.current_session_id
        if not session_id:
            session_id = self.start_session()
        
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        # Create interaction record
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        interaction: InteractionDict = {
            'id': interaction_id,
            'session_id': session_id,
            'timestamp': timestamp.isoformat(),
            'datetime_obj': timestamp,  # For internal use only
            'user_input': user_input,
            'wednesday_response': wednesday_response,
            'intent': intent,
            'status': status.value,
            'sentiment': sentiment,
            'confidence': confidence,
            'context': context or {},
            'metadata': metadata or {},
            'learning_moment': False,
            'recalled_memories': [],  # IDs of episodic memories used
            'feedback': None,  # User feedback (thumbs up/down)
            'tags': []
        }
        
        # Extract topics (simplified - would use NLP in production)
        topics = self._extract_topics(f"{user_input} {wednesday_response}")
        interaction['tags'].extend(topics)
        
        # Add to session
        self.sessions[session_id].append(interaction)
        self.interaction_count += 1
        
        # Manage session size - archive oldest if needed
        if len(self.sessions[session_id]) > self.max_history_per_session:
            self._archive_oldest_interaction(session_id)
        
        # Save interaction to disk
        self._save_interaction(interaction, session_id)
        
        # Update session metadata
        self._update_session_stats(session_id, interaction)
        
        logger.debug(f"Added interaction {interaction_id[:8]} to session {session_id[:8]}")
        
        return interaction_id
    
    def _validate_sentiment(self, sentiment: float) -> bool:
        """Validate that sentiment is within allowed range."""
        return self.SENTIMENT_RANGE[0] <= sentiment <= self.SENTIMENT_RANGE[1]
    
    def get_session_history(self, 
                           session_id: Optional[str] = None,
                           limit: Optional[int] = None) -> List[InteractionDict]:
        """
        Get all interactions in a session.
        
        Args:
            session_id: Session to retrieve (uses current if None)
            limit: Maximum number to return (returns most recent)
        
        Returns:
            List of interactions in the session
        """
        session_id = session_id or self.current_session_id
        
        if not session_id or session_id not in self.sessions:
            return []
        
        history = self.sessions[session_id]
        if limit:
            history = history[-limit:]
        
        # Return copies to prevent modification
        return [h.copy() for h in history]
    
    def get_conversation_context(self, 
                                session_id: Optional[str] = None,
                                turns: int = DEFAULT_CONTEXT_TURNS) -> List[Dict[str, Any]]:
        """
        Get recent conversation turns for context building.
        Used by working memory and language generation.
        
        Args:
            session_id: Session to get context from
            turns: Number of recent turns to include
        
        Returns:
            List of alternating user/Wednesday messages with metadata
        """
        history = self.get_session_history(session_id)
        if not history:
            return []
        
        # Get last N turns
        recent = history[-turns:]
        
        # Format as alternating user/Wednesday turns
        context = []
        for interaction in recent:
            # User turn
            context.append({
                'role': 'user',
                'content': interaction['user_input'],
                'timestamp': interaction['timestamp'],
                'sentiment': interaction.get('sentiment', 0)
            })
            
            # Wednesday turn
            context.append({
                'role': 'wednesday',
                'content': interaction['wednesday_response'],
                'timestamp': interaction['timestamp'],
                'confidence': interaction.get('confidence', 1.0),
                'status': interaction.get('status', 'success'),
                'intent': interaction.get('intent')
            })
        
        return context
    
    def get_user_history(self, 
                        user_id: str, 
                        limit: int = DEFAULT_SEARCH_LIMIT) -> List[InteractionDict]:
        """
        Get all interactions for a specific user across sessions.
        
        Args:
            user_id: User to get history for
            limit: Maximum number of interactions to return
        
        Returns:
            List of interactions, newest first
        """
        if user_id not in self.user_sessions:
            return []
        
        all_interactions = []
        
        # Get last N sessions for this user
        recent_sessions = self.user_sessions[user_id][-self.MAX_RECENT_SESSIONS_PER_USER:]
        
        for session_id in recent_sessions:
            session_history = self.get_session_history(session_id)
            all_interactions.extend(session_history)
        
        # Sort by timestamp, newest first
        all_interactions.sort(key=lambda i: i['timestamp'], reverse=True)
        
        return all_interactions[:limit]
    
    def mark_learning_moment(self, 
                            interaction_id: str, 
                            learning_type: str,
                            description: str) -> bool:
        """
        Mark an interaction as a learning moment for Wednesday.
        These get special treatment in consolidation.
        
        Args:
            interaction_id: ID of the interaction
            learning_type: Category of learning (e.g., 'user_feedback', 'pattern')
            description: Description of what was learned
        
        Returns:
            True if successfully marked
        """
        for session_id, interactions in self.sessions.items():
            for interaction in interactions:
                if interaction['id'] == interaction_id:
                    # Update the interaction
                    interaction['learning_moment'] = True
                    interaction['learning'] = {
                        'type': learning_type,
                        'description': description,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Save to disk
                    self._save_interaction(interaction, session_id)
                    
                    logger.info(f"Marked interaction {interaction_id[:8]} as learning moment: {learning_type}")
                    return True
        
        logger.warning(f"Interaction {interaction_id[:8]} not found for marking as learning moment")
        return False
    
    def add_feedback(self, interaction_id: str, feedback: str) -> bool:
        """
        Add user feedback to an interaction.
        
        Args:
            interaction_id: ID of the interaction
            feedback: 'positive', 'negative', or 'neutral'
        
        Returns:
            True if feedback was added successfully
        """
        # Validate feedback
        try:
            feedback_enum = FeedbackType.from_string(feedback)
        except ValueError:
            logger.warning(f"Invalid feedback value: {feedback}")
            return False
        
        for session_id, interactions in self.sessions.items():
            for interaction in interactions:
                if interaction['id'] == interaction_id:
                    # Add feedback
                    interaction['feedback'] = feedback_enum.value
                    self._save_interaction(interaction, session_id)
                    
                    # If negative feedback, automatically mark as learning moment
                    if feedback_enum == FeedbackType.NEGATIVE:
                        self.mark_learning_moment(
                            interaction_id,
                            'user_feedback',
                            'User provided negative feedback'
                        )
                    
                    logger.debug(f"Added {feedback} feedback to interaction {interaction_id[:8]}")
                    return True
        
        logger.warning(f"Interaction {interaction_id[:8]} not found for feedback")
        return False
    
    def get_interactions_by_status(self, 
                                  status: InteractionStatus,
                                  limit: int = DEFAULT_SEARCH_LIMIT) -> List[InteractionDict]:
        """
        Get interactions with a specific status.
        
        Args:
            status: Status to filter by
            limit: Maximum number to return
        
        Returns:
            List of matching interactions
        """
        results = []
        status_value = status.value
        
        for interactions in self.sessions.values():
            for interaction in interactions:
                if interaction.get('status') == status_value:
                    results.append(interaction)
                    if len(results) >= limit:
                        return results
        
        return results
    
    def get_failed_interactions(self, limit: int = DEFAULT_SEARCH_LIMIT) -> List[InteractionDict]:
        """
        Get interactions where Wednesday failed or was confused.
        
        Args:
            limit: Maximum number to return
        
        Returns:
            List of failed/confused interactions
        """
        failed_statuses = {
            InteractionStatus.FAILURE.value,
            InteractionStatus.CONFUSION.value
        }
        
        results = []
        for interactions in self.sessions.values():
            for interaction in interactions:
                if interaction.get('status') in failed_statuses:
                    results.append(interaction)
                    if len(results) >= limit:
                        return results
        
        return results
    
    def search_interactions(self, 
                           query: str, 
                           limit: int = DEFAULT_SEARCH_LIMIT) -> List[InteractionDict]:
        """
        Search across all interactions for text matches.
        
        Args:
            query: Text to search for
            limit: Maximum number to return
        
        Returns:
            List of matching interactions
        """
        query_lower = query.lower()
        results = []
        
        for interactions in self.sessions.values():
            for interaction in interactions:
                # Search in user input and Wednesday response
                if (query_lower in interaction['user_input'].lower() or 
                    query_lower in interaction['wednesday_response'].lower()):
                    results.append(interaction)
                    if len(results) >= limit:
                        return results
        
        return results
    
    def get_interaction_stats(self, 
                             session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about interactions.
        
        Args:
            session_id: Specific session (None for all sessions)
        
        Returns:
            Dictionary with interaction statistics
        """
        if session_id:
            interactions = self.get_session_history(session_id)
        else:
            # Aggregate all sessions
            interactions = []
            for sess_interactions in self.sessions.values():
                interactions.extend(sess_interactions)
        
        if not interactions:
            return {}
        
        # Calculate status distribution
        status_counts = defaultdict(int)
        sentiments = []
        confidences = []
        learning_count = 0
        feedback_count = 0
        
        for interaction in interactions:
            status_counts[interaction.get('status', 'unknown')] += 1
            
            if interaction.get('sentiment') is not None:
                sentiments.append(interaction['sentiment'])
            
            confidences.append(interaction.get('confidence', 1.0))
            
            if interaction.get('learning_moment'):
                learning_count += 1
            
            if interaction.get('feedback'):
                feedback_count += 1
        
        return {
            'total_interactions': len(interactions),
            'status_distribution': dict(status_counts),
            'average_sentiment': sum(sentiments) / len(sentiments) if sentiments else 0,
            'average_confidence': sum(confidences) / len(confidences),
            'learning_moments': learning_count,
            'feedback_received': feedback_count,
            'unique_sessions': len(set(i['session_id'] for i in interactions))
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract potential topics from text (simplified).
        In production, would use NLP or embeddings.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of detected topics
        """
        # Very simple topic extraction based on keyword matching
        common_topics = {
            'weather': ['weather', 'rain', 'sun', 'cloud', 'temperature'],
            'time': ['time', 'clock', 'hour', 'minute', 'today', 'tomorrow'],
            'help': ['help', 'assist', 'support', 'guide'],
            'question': ['what', 'why', 'when', 'where', 'how', 'who'],
            'story': ['story', 'tell', 'narrate', 'tale'],
            'joke': ['joke', 'funny', 'humor', 'laugh'],
            'fact': ['fact', 'did you know', 'actually'],
            'opinion': ['think', 'believe', 'opinion', 'feel'],
            'memory': ['remember', 'recall', 'forget', 'memory'],
            'learning': ['learn', 'understand', 'comprehend', 'explain']
        }
        
        topics = []
        text_lower = text.lower()
        
        for topic, keywords in common_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _update_session_stats(self, session_id: str, interaction: InteractionDict) -> None:
        """
        Update session statistics based on new interaction.
        
        Args:
            session_id: Session to update
            interaction: New interaction added
        """
        metadata = self.session_metadata.get(session_id, {})
        
        if not metadata:
            return
        
        # Update interaction count
        metadata['interaction_count'] = len(self.sessions[session_id])
        
        # Update topics
        new_topics = interaction.get('tags', [])
        current_topics = set(metadata.get('topics', []))
        metadata['topics'] = list(current_topics.union(new_topics))
        
        # Update running sentiment average
        current_avg = metadata.get('average_sentiment', 0)
        count = metadata['interaction_count']
        
        if count > 1:
            # Weighted average
            metadata['average_sentiment'] = (
                (current_avg * (count - 1) + interaction['sentiment']) / count
            )
        else:
            metadata['average_sentiment'] = interaction['sentiment']
        
        # Save updated metadata
        self._save_session_metadata(session_id, metadata)
        self.session_metadata[session_id] = metadata
    
    def _save_interaction(self, interaction: InteractionDict, session_id: str) -> None:
        """
        Save an interaction to disk.
        
        Args:
            interaction: Interaction to save
            session_id: Session it belongs to
        """
        session_dir = self.storage_path / session_id
        session_dir.mkdir(exist_ok=True)
        
        interaction_file = session_dir / f"{interaction['id']}.json"
        
        # Create a serializable copy (remove datetime_obj)
        save_copy = {}
        for key, value in interaction.items():
            if key != 'datetime_obj':  # Skip internal datetime object
                if isinstance(value, (datetime, Path)):
                    save_copy[key] = str(value)
                else:
                    save_copy[key] = value
        
        try:
            with open(interaction_file, 'w') as f:
                json.dump(save_copy, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save interaction {interaction['id'][:8]}: {e}")
    
    def _save_session_metadata(self, session_id: str, metadata: SessionMetadata) -> None:
        """
        Save session metadata to disk.
        
        Args:
            session_id: Session identifier
            metadata: Metadata to save
        """
        session_dir = self.storage_path / session_id
        session_dir.mkdir(exist_ok=True)
        
        metadata_file = session_dir / "metadata.json"
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save session metadata for {session_id[:8]}: {e}")
    
    def _load_session_metadata(self) -> None:
        """Load existing session metadata from disk."""
        if not self.storage_path.exists():
            return
        
        for session_dir in self.storage_path.iterdir():
            if not session_dir.is_dir():
                continue
            
            metadata_file = session_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    session_id = metadata.get('session_id', session_dir.name)
                    self.session_metadata[session_id] = metadata
                    
                    # Add to user index if not already there
                    user_id = metadata.get('user_id', 'anonymous')
                    if user_id not in self.user_sessions:
                        self.user_sessions[user_id] = []
                    if session_id not in self.user_sessions[user_id]:
                        self.user_sessions[user_id].append(session_id)
                    
                except Exception as e:
                    logger.error(f"Failed to load session metadata from {metadata_file}: {e}")
        
        logger.info(f"Loaded metadata for {len(self.session_metadata)} sessions")
    
    def _archive_oldest_interaction(self, session_id: str) -> None:
        """
        Archive the oldest interaction from a session to free memory.
        Interaction is already saved to disk, just remove from memory cache.
        
        Args:
            session_id: Session to archive from
        """
        if session_id in self.sessions and self.sessions[session_id]:
            oldest = self.sessions[session_id].pop(0)
            logger.debug(f"Archived oldest interaction {oldest['id'][:8]} from session {session_id[:8]}")
    
    def get_session_list(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of all sessions with basic metadata.
        
        Args:
            user_id: Filter by user (None for all users)
        
        Returns:
            List of session metadata dictionaries
        """
        if user_id:
            session_ids = self.user_sessions.get(user_id, [])
        else:
            session_ids = list(self.session_metadata.keys())
        
        sessions = []
        for session_id in session_ids:
            metadata = self.session_metadata.get(session_id, {})
            if metadata:
                sessions.append({
                    'session_id': session_id,
                    'user_id': metadata.get('user_id', 'unknown'),
                    'start_time': metadata.get('start_time'),
                    'end_time': metadata.get('end_time'),
                    'interaction_count': metadata.get('interaction_count', 0),
                    'status': metadata.get('status', 'unknown')
                })
        
        # Sort by start time, newest first
        sessions.sort(key=lambda s: s.get('start_time', ''), reverse=True)
        
        return sessions
    
    def clear_session_cache(self, session_id: Optional[str] = None) -> None:
        """
        Clear in-memory cache for a session to free memory.
        Interactions remain on disk.
        
        Args:
            session_id: Session to clear (None for all sessions)
        """
        if session_id:
            if session_id in self.sessions:
                self.sessions[session_id].clear()
                logger.debug(f"Cleared cache for session {session_id[:8]}")
        else:
            self.sessions.clear()
            logger.debug("Cleared all session caches")
    
    def __len__(self) -> int:
        return self.interaction_count
    
    def __repr__(self) -> str:
        current = f", current={self.current_session_id[:8]}" if self.current_session_id else ""
        return (f"InteractionHistory(sessions={len(self.sessions)}, "
                f"interactions={self.interaction_count}{current})")