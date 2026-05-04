"""
Context Buffer - Working memory for active context.
Maintains recent interactions and current focus with limited capacity FIFO behavior.
"""

from typing import Dict, Any, Optional, List
from collections import deque
from datetime import datetime
import uuid
import logging

# Configure logger
logger = logging.getLogger(__name__)

class ContextBuffer:
    """
    Working memory buffer that maintains recent context.
    Like human working memory - limited capacity, active focus.
    Uses FIFO (First-In-First-Out) behavior when buffer reaches maximum size.
    """
    
    def __init__(self, size: int = 10):
        """
        Initialize context buffer with specified capacity.
        
        Args:
            size: Maximum number of items to store in buffer (default: 10)
        """
        self.size = size
        # Use deque with maxlen for automatic FIFO behavior when buffer fills up
        self.buffer = deque(maxlen=size)
        # Track currently focused item ID (None if no focus set)
        self.attention_focus: Optional[str] = None
        # Record buffer creation timestamp
        self.created_at = datetime.now()
        
    def add(self, content: Any, metadata: Optional[Dict] = None) -> bool:
        """
        Add item to working memory.
        
        Args:
            content: The main content/data to store
            metadata: Optional dictionary with additional context
        
        Returns:
            bool: True if item was successfully added
        """
        # Create structured item with metadata and tracking fields
        item = {
            'id': str(uuid.uuid4()),  # Unique identifier
            'content': content,  # Main content
            'metadata': metadata or {},  # Additional context (default empty dict)
            'timestamp': datetime.now(),  # When item was added
            'importance': self._calculate_importance(content, metadata)  # Priority score
        }
        
        # Add to buffer (deque automatically handles maxlen)
        self.buffer.append(item)
        logger.debug(f"Added to working memory: {item['id']}")
        return True
    
    def get(self, query: Any, limit: int = 5) -> List[Any]:
        """
        Get items from working memory matching query.
        
        Note: This is a simplified implementation that returns recent items.
        For production use, implement actual matching logic based on query.
        
        Args:
            query: Search criteria (currently unused)
            limit: Maximum number of items to return
        
        Returns:
            List of matching items (most recent first)
        """
        # Convert deque to list and return last 'limit' items
        # Items are returned in chronological order (oldest to newest)
        items = list(self.buffer)
        return items[-limit:] if items else []
    
    def recent(self, n: int) -> List[Dict]:
        """
        Get n most recent items.
        
        Args:
            n: Number of recent items to retrieve
        
        Returns:
            List of most recent items
        """
        # Return last n items (or all if n > buffer size)
        items = list(self.buffer)
        return items[-n:] if items else []
    
    def get_important(self, threshold: float = 0.5) -> List[Dict]:
        """
        Get items above importance threshold (for consolidation).
        
        Args:
            threshold: Minimum importance score (0.0 to 1.0)
        
        Returns:
            List of items with importance > threshold
        """
        # Filter items by importance score
        return [item for item in self.buffer if item.get('importance', 0) > threshold]
    
    def set_focus(self, item_id: str) -> bool:
        """
        Set attention focus to specific item.
        
        Args:
            item_id: ID of item to focus on
        
        Returns:
            bool: True if focus was set successfully
        """
        # Check if item exists before setting focus
        for item in self.buffer:
            if item['id'] == item_id:
                self.attention_focus = item_id
                logger.debug(f"Focus set to item: {item_id}")
                return True
        
        # Item not found in buffer
        logger.warning(f"Attempted to set focus on non-existent item: {item_id}")
        return False
    
    def get_focus(self) -> Optional[Dict]:
        """
        Get current focus item.
        
        Returns:
            Focus item dict if focus is set, None otherwise
        """
        if self.attention_focus:
            for item in self.buffer:
                if item['id'] == self.attention_focus:
                    return item
        
        # Focus item may have been removed from buffer
        if self.attention_focus:
            logger.debug(f"Focus item {self.attention_focus} no longer in buffer")
            self.attention_focus = None
        
        return None
    
    def remove(self, item_id: str) -> bool:
        """
        Remove specific item from buffer.
        
        Args:
            item_id: ID of item to remove
        
        Returns:
            bool: True if item was removed
        """
        # Find and remove item by ID
        for i, item in enumerate(self.buffer):
            if item['id'] == item_id:
                # Remove item and reconstruct deque
                items = list(self.buffer)
                del items[i]
                self.buffer = deque(items, maxlen=self.size)
                
                # Clear focus if focused item was removed
                if self.attention_focus == item_id:
                    self.attention_focus = None
                    logger.debug(f"Focus cleared - removed item: {item_id}")
                
                logger.debug(f"Removed item: {item_id}")
                return True
        
        logger.warning(f"Attempted to remove non-existent item: {item_id}")
        return False
    
    def clear(self) -> None:
        """Clear working memory and reset focus."""
        self.buffer.clear()
        self.attention_focus = None
        logger.debug("Context buffer cleared")
    
    def _calculate_importance(self, content: Any, metadata: Optional[Dict]) -> float:
        """
        Calculate importance score for consolidation decisions.
        
        Args:
            content: Item content (may influence importance)
            metadata: Item metadata with importance indicators
        
        Returns:
            float: Importance score between 0.0 and 1.0
        """
        importance = 0.5  # Base importance (neutral)
        
        # Boost importance based on metadata flags
        if metadata:
            # User-initiated content is more important
            if metadata.get('user_said'):
                importance += 0.2
            # Emotionally charged content gets priority
            if metadata.get('emotional'):
                importance += 0.2
            # Repeated/reinforced information is more important
            if metadata.get('repeated'):
                importance += 0.1
            # Add more importance factors as needed
        
        # Ensure importance stays within valid range
        return min(max(importance, 0.0), 1.0)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def __repr__(self) -> str:
        """
        String representation for debugging.
        
        Returns:
            str: Buffer status summary
        """
        return f"ContextBuffer(items={len(self.buffer)}, focus={'set' if self.attention_focus else 'None'})"