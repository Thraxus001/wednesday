"""
Active Session - Tracks the current interaction session.
Like a coffee chat with a friend - maintains context for the duration of conversation.
Persists for the duration of a conversation and links to user identity.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import uuid
import logging

# Configure logger
logger = logging.getLogger(__name__)


class ActiveSession:
    """
    Tracks current interaction session metadata.
    Maintains conversation state, goals, and temporal context.
    Like remembering what you were talking about with a friend over coffee.
    
    A session represents a single continuous conversation with a user,
    tracking goals, emotional tone, important moments, and activity status.
    """
    
    # Class constants for configuration
    DEFAULT_TIMEOUT_MINUTES = 5
    MAX_IMPORTANT_MOMENTS = 10
    PRIORITY_HIGHEST = 1
    PRIORITY_LOWEST = 5
    
    def __init__(self, session_id: Optional[str] = None, user_id: str = "anonymous"):
        """
        Initialize a new active session.
        
        Args:
            session_id: Unique identifier (auto-generated if None)
            user_id: Identifier for the user (defaults to "anonymous")
        """
        # Core identification
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        
        # Temporal tracking
        self.start_time = datetime.now()
        self.last_active = self.start_time
        
        # Session metrics
        self.interaction_count = 0
        
        # Session context tracking
        self.session_goals: List[Dict[str, Any]] = []  # What user wants in this session
        self.context_tags: List[str] = []  # Topics discussed
        self.emotional_tone: str = "neutral"  # Overall session tone
        self.important_moments: List[Dict[str, Any]] = []  # Key points to remember
        
        logger.info(f"New session started: {self.session_id} for user {user_id}")
    
    def touch(self) -> None:
        """Update last_active timestamp to current time."""
        self.last_active = datetime.now()
        logger.debug(f"Session {self.session_id} touched at {self.last_active}")
    
    def add_interaction(self) -> None:
        """Record a new interaction in this session."""
        self.interaction_count += 1
        self.touch()
        logger.debug(f"Session {self.session_id} interaction #{self.interaction_count}")
    
    def add_goal(self, goal: str, priority: int = 1, context: Optional[Dict] = None) -> None:
        """
        Track what user is trying to accomplish in this session.
        
        Args:
            goal: Description of user's goal
            priority: 1-5 (1 highest, 5 lowest)
            context: Additional context about the goal
        
        Raises:
            ValueError: If priority is outside valid range
        """
        # Validate priority range
        if priority < self.PRIORITY_HIGHEST or priority > self.PRIORITY_LOWEST:
            raise ValueError(f"Priority must be between {self.PRIORITY_HIGHEST} and {self.PRIORITY_LOWEST}")
        
        goal_entry = {
            'id': str(uuid.uuid4()),
            'description': goal,
            'priority': priority,
            'created_at': datetime.now(),
            'last_referenced': datetime.now(),
            'context': context or {},
            'completed': False,
            'completed_at': None
        }
        
        self.session_goals.append(goal_entry)
        logger.info(f"Goal added to session {self.session_id}: {goal[:50]}...")
    
    def update_goal(self, goal_id: str, completed: bool = False, 
                   new_priority: Optional[int] = None) -> bool:
        """
        Update goal status or priority.
        
        Args:
            goal_id: ID of goal to update
            completed: Whether goal is completed
            new_priority: New priority value (optional)
        
        Returns:
            bool: True if goal was found and updated
        
        Raises:
            ValueError: If new_priority is outside valid range
        """
        # Validate new_priority if provided
        if new_priority is not None:
            if new_priority < self.PRIORITY_HIGHEST or new_priority > self.PRIORITY_LOWEST:
                raise ValueError(f"Priority must be between {self.PRIORITY_HIGHEST} and {self.PRIORITY_LOWEST}")
        
        # Find and update goal
        for goal in self.session_goals:
            if goal['id'] == goal_id:
                # Update completion status
                if completed and not goal['completed']:
                    goal['completed'] = True
                    goal['completed_at'] = datetime.now()
                    logger.info(f"Goal completed: {goal['description'][:50]}...")
                
                # Update priority if provided
                if new_priority is not None:
                    goal['priority'] = new_priority
                    logger.debug(f"Goal priority updated to {new_priority}")
                
                # Update last referenced timestamp
                goal['last_referenced'] = datetime.now()
                return True
        
        # Goal not found
        logger.warning(f"Goal {goal_id} not found in session {self.session_id}")
        return False
    
    def get_active_goals(self) -> List[Dict[str, Any]]:
        """
        Get all incomplete goals, sorted by priority (highest first).
        
        Returns:
            List of active goals sorted by priority
        """
        # Filter incomplete goals
        active = [g for g in self.session_goals if not g['completed']]
        
        # Sort by priority (lower number = higher priority)
        return sorted(active, key=lambda g: g['priority'])
    
    def add_context_tag(self, tag: str) -> None:
        """
        Add a topic tag to session context.
        
        Args:
            tag: Topic tag to add
        """
        # Normalize tag (lowercase, strip whitespace)
        normalized_tag = tag.lower().strip()
        
        if normalized_tag and normalized_tag not in self.context_tags:
            self.context_tags.append(normalized_tag)
            logger.debug(f"Context tag added to session {self.session_id}: {normalized_tag}")
    
    def set_emotional_tone(self, tone: str) -> None:
        """
        Set overall emotional tone of session.
        
        Args:
            tone: Emotional tone description (e.g., "happy", "frustrated", "neutral")
        """
        # Normalize tone
        normalized_tone = tone.lower().strip()
        
        if normalized_tone:
            self.emotional_tone = normalized_tone
            logger.debug(f"Session {self.session_id} tone set to: {normalized_tone}")
    
    def add_important_moment(self, moment: str, significance: float = 0.8,
                            context: Optional[Dict] = None) -> None:
        """
        Flag an important moment in the conversation.
        
        Args:
            moment: Description of the important moment
            significance: 0-1 how significant (0 = trivial, 1 = crucial)
            context: Additional context about the moment
        
        Raises:
            ValueError: If significance is outside valid range
        """
        # Validate significance
        if not 0 <= significance <= 1:
            raise ValueError("Significance must be between 0 and 1")
        
        moment_entry = {
            'id': str(uuid.uuid4()),
            'moment': moment,
            'significance': significance,
            'timestamp': datetime.now(),
            'context': context or {}
        }
        
        # Add new moment
        self.important_moments.append(moment_entry)
        
        # Sort by significance (highest first)
        self.important_moments.sort(key=lambda m: m['significance'], reverse=True)
        
        # Keep only most important moments
        if len(self.important_moments) > self.MAX_IMPORTANT_MOMENTS:
            removed = self.important_moments[self.MAX_IMPORTANT_MOMENTS:]
            self.important_moments = self.important_moments[:self.MAX_IMPORTANT_MOMENTS]
            logger.debug(f"Removed {len(removed)} less significant moments")
        
        logger.info(f"Important moment recorded in session {self.session_id}: {moment[:50]}...")
    
    def is_stale(self, timeout_minutes: Optional[int] = None) -> bool:
        """
        Check if session has timed out due to inactivity.
        
        Args:
            timeout_minutes: Minutes of inactivity before session is stale.
                           Uses DEFAULT_TIMEOUT_MINUTES if None.
        
        Returns:
            bool: True if session is stale/timed out
        """
        # Use default timeout if not specified
        timeout = timeout_minutes or self.DEFAULT_TIMEOUT_MINUTES
        
        # Check if last_active is set (should always be)
        if self.last_active is None:
            return True
        
        # Calculate staleness
        stale_time = datetime.now() - timedelta(minutes=timeout)
        is_stale = self.last_active < stale_time
        
        if is_stale:
            logger.debug(f"Session {self.session_id} is stale (inactive > {timeout} min)")
        
        return is_stale
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the session for context.
        
        Returns:
            Dictionary with session metrics and status
        """
        # Calculate session duration
        duration = datetime.now() - self.start_time
        duration_minutes = duration.total_seconds() / 60
        
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'duration_minutes': round(duration_minutes, 2),
            'interaction_count': self.interaction_count,
            'active_goals': len(self.get_active_goals()),
            'total_goals': len(self.session_goals),
            'topics': self.context_tags[-5:],  # Last 5 topics
            'emotional_tone': self.emotional_tone,
            'important_moments': len(self.important_moments),
            'is_stale': self.is_stale()
        }
    
    def end_session(self) -> Dict[str, Any]:
        """
        End the session and return final stats for storage/archival.
        
        Returns:
            Dictionary with complete session statistics
        """
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Calculate goal completion statistics
        goals_completed = sum(1 for g in self.session_goals if g['completed'])
        
        summary = {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': round(duration.total_seconds(), 2),
            'interaction_count': self.interaction_count,
            'goals_completed': goals_completed,
            'goals_attempted': len(self.session_goals),
            'topics_discussed': self.context_tags.copy(),  # Return copy to prevent modification
            'emotional_tone': self.emotional_tone,
            'important_moments': self.important_moments.copy()  # Return copy
        }
        
        logger.info(f"Session {self.session_id} ended. {summary['interaction_count']} interactions, "
                   f"{goals_completed}/{len(self.session_goals)} goals completed.")
        
        return summary
    
    def get_goal_by_id(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific goal by its ID.
        
        Args:
            goal_id: ID of goal to retrieve
        
        Returns:
            Goal dictionary or None if not found
        """
        for goal in self.session_goals:
            if goal['id'] == goal_id:
                return goal.copy()  # Return copy to prevent modification
        return None
    
    def get_recent_topics(self, count: int = 5) -> List[str]:
        """
        Get most recent topics discussed.
        
        Args:
            count: Number of recent topics to return
        
        Returns:
            List of most recent topic tags
        """
        return self.context_tags[-count:] if self.context_tags else []
    
    def __len__(self) -> int:
        """Return number of interactions in session."""
        return self.interaction_count
    
    def __repr__(self) -> str:
        """
        String representation for debugging.
        
        Returns:
            Summary string with key session attributes
        """
        return (f"ActiveSession(id={self.session_id[:8]}, user={self.user_id}, "
                f"interactions={self.interaction_count}, stale={self.is_stale()})")