"""
Memory System Exceptions
Custom exceptions for the memory subsystem to handle error cases gracefully.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

class WednesdayError(Exception):
    """Base exception for all Wednesday AI errors"""
    pass

class MemoryError(WednesdayError):
    """Base exception for all memory-related errors"""
    pass

class MemoryStorageError(MemoryError):
    """Raised when there's an error storing data in memory"""
    pass

class MemoryRetrievalError(MemoryError):
    """Raised when there's an error retrieving data from memory"""
    pass

class MemoryNotFoundError(MemoryError):
    """Raised when a requested memory doesn't exist"""
    pass

class MemoryIndexError(MemoryError):
    """Raised when there's an error with memory indexing"""
    pass

class MemoryConsolidationError(MemoryError):
    """Raised when memory consolidation fails"""
    pass

class MemoryCorruptionError(MemoryError):
    """Raised when a memory appears corrupted"""
    pass

class MemoryFullError(MemoryError):
    """Raised when a memory store reaches capacity"""
    pass

class MemoryCapacityError(MemoryError):
    """Raised when memory capacity limits are reached"""
    pass

class MemoryPermissionError(MemoryError):
    """Raised when trying to access protected memories"""
    pass

class MemoryTimeoutError(MemoryError):
    """Raised when a memory operation times out"""
    pass

class WorkingMemoryError(MemoryError):
    """Raised for working memory specific errors"""
    pass

class EpisodicMemoryError(MemoryError):
    """Raised for episodic memory specific errors"""
    pass

class SemanticMemoryError(MemoryError):
    """Raised for semantic memory specific errors"""
    pass

class ProceduralMemoryError(MemoryError):
    """Raised for procedural memory specific errors"""
    pass

class MemorySerializationError(MemoryError):
    """Raised when failing to serialize/deserialize memories"""
    pass

class MemoryVersionError(MemoryError):
    """Raised when memory format version mismatch occurs"""
    pass

class MemoryQuotaExceededError(MemoryError):
    """Raised when memory usage exceeds configured quota"""
    pass

class MemoryLockError(MemoryError):
    """Raised when a memory is locked by another process"""
    pass

class MemoryTransactionError(MemoryError):
    """Raised when a memory transaction fails"""
    pass

def handle_memory_error(func):
    """
    Decorator for handling memory errors gracefully
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MemoryError as e:
            # Log the error and re-raise
            logger = logging.getLogger(__name__)
            logger.error(f"Memory operation failed in {func.__name__}: {e}")
            raise
        except Exception as e:
            # Wrap unexpected errors
            logger = logging.getLogger(__name__)
            logger.exception(f"Unexpected error in {func.__name__}")
            raise MemoryError(f"Unexpected error: {e}") from e
    return wrapper

ERROR_CODES = {
    MemoryStorageError: "MEMORY_STORAGE_ERROR",
    MemoryRetrievalError: "MEMORY_RETRIEVAL_ERROR",
    MemoryNotFoundError: "MEMORY_NOT_FOUND",
    MemoryIndexError: "MEMORY_INDEX_ERROR",
    MemoryConsolidationError: "MEMORY_CONSOLIDATION_ERROR",
    MemoryCorruptionError: "MEMORY_CORRUPTION",
    MemoryFullError: "MEMORY_FULL",
    MemoryCapacityError: "MEMORY_CAPACITY_ERROR",
    MemoryPermissionError: "MEMORY_PERMISSION_DENIED",
    MemoryTimeoutError: "MEMORY_TIMEOUT",
    WorkingMemoryError: "WORKING_MEMORY_ERROR",
    EpisodicMemoryError: "EPISODIC_MEMORY_ERROR",
    SemanticMemoryError: "SEMANTIC_MEMORY_ERROR",
    ProceduralMemoryError: "PROCEDURAL_MEMORY_ERROR",
    MemorySerializationError: "MEMORY_SERIALIZATION_ERROR",
    MemoryVersionError: "MEMORY_VERSION_MISMATCH",
    MemoryQuotaExceededError: "MEMORY_QUOTA_EXCEEDED",
    MemoryLockError: "MEMORY_LOCKED",
    MemoryTransactionError: "MEMORY_TRANSACTION_FAILED"
}

def get_error_code(exception):
    """Get the error code for a given exception"""
    for exc_type, code in ERROR_CODES.items():
        if isinstance(exception, exc_type):
            return code
    return "UNKNOWN_MEMORY_ERROR"

@dataclass
class ErrorContext:
    """Context information for memory errors"""
    context: str
    details: Optional[Dict] = None

def safe_execute(func):
    """
    Decorator for safe memory operations with error context
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MemoryError:
            raise
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.exception(f"Safe execute failed in {func.__name__}: {e}")
            raise MemoryError(f"Operation failed: {e}") from e
    return wrapper

class MemoryErrorResponse:
    """Standard error response structure for memory operations"""
    
    def __init__(self, error: Exception, operation: str, details: dict = None):
        self.error = error
        self.operation = operation
        self.details = details or {}
        self.error_code = get_error_code(error)
        self.timestamp = None  # Will be set when used
        
    def to_dict(self):
        """Convert to dictionary for API responses"""
        from datetime import datetime
        self.timestamp = datetime.now().isoformat()
        
        return {
            'success': False,
            'error': {
                'code': self.error_code,
                'message': str(self.error),
                'type': self.error.__class__.__name__,
                'operation': self.operation,
                'timestamp': self.timestamp,
                'details': self.details
            }
        }
    
    def __str__(self):
        return f"[{self.error_code}] {self.error} (during {self.operation})"

