"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                 ∞ INFINITE EXCEPTIONS SYSTEM - NO LIMITS ∞                    ║
║              Maximum Efficiency - Zero Restrictions - Absolute Power          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import traceback
from typing import Any, Dict, Optional, List, Set, Union, Callable
from datetime import datetime
import inspect
import threading
import json
import weakref
from enum import Enum
import uuid
import time


class ErrorSeverity:
    """
    Dynamic error severity levels - can create ANY severity at runtime.
    No limits on number of severity levels or their values.
    """
    _levels = {}
    _numeric_values = {}
    _lock = threading.RLock()
    
    # Initialize with common levels (can add infinitely more)
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"
    
    def __init__(self):
        """Initialize the class with default levels."""
        with self._lock:
            if not self._levels:
                self._levels = {
                    "DEBUG": "DEBUG",
                    "INFO": "INFO",
                    "WARNING": "WARNING",
                    "ERROR": "ERROR",
                    "CRITICAL": "CRITICAL",
                    "FATAL": "FATAL"
                }
                self._numeric_values = {
                    "DEBUG": 10,
                    "INFO": 20,
                    "WARNING": 30,
                    "ERROR": 40,
                    "CRITICAL": 50,
                    "FATAL": 60
                }
    
    @classmethod
    def register(cls, name: str, numeric_value: float = None):
        """Register a new severity level with ANY numeric value."""
        with cls._lock:
            if numeric_value is None:
                numeric_value = len(cls._levels) * 10
            cls._levels[name] = name
            cls._numeric_values[name] = numeric_value
            setattr(cls, name, name)
        return name
    
    @classmethod
    def get_numeric(cls, severity: str) -> float:
        """Get numeric value for comparison."""
        return cls._numeric_values.get(severity, 0)
    
    @classmethod
    def compare(cls, sev1: str, sev2: str) -> int:
        """Compare severity levels."""
        return cls.get_numeric(sev1) - cls.get_numeric(sev2)
    
    @classmethod
    def all(cls) -> Dict:
        """Get all registered severity levels."""
        return cls._levels.copy()


# Initialize ErrorSeverity
ErrorSeverity()


# ============================================================================
# INFINITE EXCEPTION BASE - CAN BE EXTENDED UNLIMITEDLY
# ============================================================================

class WednesdayError(Exception):
    """
    Base exception for all Wednesday AI errors.
    This class can be extended INFINITELY with ANY attributes.
    """
    
    # Registry of all exception types (for dynamic lookup)
    _exception_registry = {}
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """Allow dynamic creation of exception instances with ANY arguments."""
        instance = super().__new__(cls)
        return instance
    
    def __init__(
        self,
        message: Any = None,
        severity: str = "ERROR",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Any = None,
        **kwargs
    ):
        """
        Initialize with ANY parameters - no restrictions.
        
        Args:
            message: ANY message (string, object, None) - no length limit
            severity: ANY severity level (will be created if doesn't exist)
            details: ANY dictionary with ANY values (no size/type limits)
            cause: ANY exception (can be ANY exception type)
            error_code: ANY error code (int, str, object, etc.)
            **kwargs: ANY additional attributes (stored in details)
        """
        # Store EVERYTHING - no filtering, no limits
        self.message = message
        self.severity = self._ensure_severity(severity)
        self.details = details or {}
        self.cause = cause
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()
        self.exception_id = str(uuid.uuid4())
        
        # Add all kwargs to details (unlimited storage)
        self.details.update(kwargs)
        
        # Capture stack trace (full trace, no truncation)
        self.stack_trace = traceback.format_exc() if sys.exc_info()[0] else None
        
        # Build message (lazy evaluation - only when needed)
        # Initialize message cache before capturing context to avoid
        # __str__ being called on partially-initialized objects.
        self._full_message = None

        # Capture context (can be ANY data)
        self.context = self._capture_context()
        
        # Register this exception type if not already registered
        self._register_exception_type()
        
        # Call parent constructor with message
        super().__init__(str(message) if message is not None else "")
    
    def _ensure_severity(self, severity: Any) -> str:
        """Ensure severity exists, create if it doesn't."""
        if isinstance(severity, str):
            if not hasattr(ErrorSeverity, severity):
                ErrorSeverity.register(severity)
            return severity
        return "ERROR"
    
    def _register_exception_type(self):
        """Register this exception type for dynamic lookup."""
        with self._lock:
            cls_name = self.__class__.__name__
            if cls_name not in self._exception_registry:
                self._exception_registry[cls_name] = self.__class__
    
    def _capture_context(self) -> Dict[str, Any]:
        """Capture execution context (unlimited)."""
        context = {}
        
        # Capture frame info (no limit on depth)
        frame = inspect.currentframe()
        if frame:
            try:
                # Go back 2 frames to skip this method and __init__
                frame = frame.f_back.f_back
                if frame:
                    # Capture local variables (ALL of them, no filtering)
                    context['locals'] = {
                        k: str(v) for k, v in frame.f_locals.items()
                        if not k.startswith('__')
                    }
                    
                    # Capture function/module info
                    context['function'] = frame.f_code.co_name
                    context['module'] = frame.f_globals.get('__name__', 'unknown')
                    context['line'] = frame.f_lineno
                    
                    # Capture call stack (unlimited depth)
                    stack = []
                    f = frame
                    while f:
                        stack.append({
                            'function': f.f_code.co_name,
                            'file': f.f_code.co_filename,
                            'line': f.f_lineno
                        })
                        f = f.f_back
                    context['stack'] = stack
            finally:
                del frame  # Avoid reference cycles
        
        return context
    
    def __str__(self) -> str:
        """String representation (built lazily)."""
        if self._full_message is None:
            self._full_message = self._build_message()
        return self._full_message
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<{self.__class__.__name__} id={self.exception_id[:8]} severity={self.severity}>"
    
    def _build_message(self) -> str:
        """Build complete error message (unlimited length)."""
        parts = []
        
        # Add severity
        parts.append(f"[{self.severity}]")
        
        # Add error code if present
        if self.error_code is not None:
            parts.append(f"({self.error_code})")
        
        # Add message
        if self.message is not None:
            parts.append(str(self.message))
        
        # Add exception type if not already clear
        if self.__class__.__name__ != 'WednesdayError':
            parts.append(f"[{self.__class__.__name__}]")
        
        # Add ID for tracking
        parts.append(f"[ID:{self.exception_id[:8]}]")
        
        # Join with spaces
        return " ".join(parts)
    
    def to_dict(self, max_depth: int = None) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Args:
            max_depth: Optional limit for recursion (None = unlimited)
        """
        result = {
            'exception_id': self.exception_id,
            'type': self.__class__.__name__,
            'severity': self.severity,
            'message': str(self.message) if self.message is not None else None,
            'error_code': self._serialize_value(self.error_code, max_depth),
            'timestamp': self.timestamp,
            'details': self._serialize_value(self.details, max_depth),
            'has_cause': self.cause is not None,
        }
        
        # Add stack trace (full, no truncation)
        if self.stack_trace:
            result['stack_trace'] = self.stack_trace
        
        # Add context (full, no truncation)
        if self.context:
            result['context'] = self._serialize_value(self.context, max_depth)
        
        # Add cause recursively (unlimited depth)
        if self.cause:
            if isinstance(self.cause, WednesdayError):
                result['cause'] = self.cause.to_dict(max_depth)
            else:
                result['cause'] = {
                    'type': type(self.cause).__name__,
                    'message': str(self.cause),
                    'traceback': traceback.format_exception(
                        type(self.cause), self.cause, self.cause.__traceback__
                    )
                }
        
        return result
    
    def _serialize_value(self, value: Any, max_depth: int = None, current_depth: int = 0) -> Any:
        """Serialize ANY value for JSON output (no limits)."""
        if max_depth is not None and current_depth >= max_depth:
            return "..."
        
        # Handle None
        if value is None:
            return None
        
        # Handle basic types
        if isinstance(value, (str, int, float, bool)):
            return value
        
        # Handle datetime
        if isinstance(value, datetime):
            return value.isoformat()
        
        # Handle enum
        if isinstance(value, Enum):
            return value.value
        
        # Handle exceptions
        if isinstance(value, Exception):
            return {
                'type': type(value).__name__,
                'message': str(value)
            }
        
        # Handle sequences
        if isinstance(value, (list, tuple, set)):
            return [
                self._serialize_value(item, max_depth, current_depth + 1)
                for item in value
            ]
        
        # Handle dictionaries
        if isinstance(value, dict):
            return {
                str(k): self._serialize_value(v, max_depth, current_depth + 1)
                for k, v in value.items()
            }
        
        # Handle objects with __dict__
        if hasattr(value, '__dict__'):
            return self._serialize_value(value.__dict__, max_depth, current_depth + 1)
        
        # Default to string representation
        try:
            return str(value)
        except:
            return "<unserializable>"
    
    def to_json(self, **kwargs) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)
    
    def log(self, logger=None):
        """Log this exception."""
        if logger is None:
            import logging
            logger = logging.getLogger(__name__)
        
        log_method = getattr(logger, self.severity.lower(), logger.error)
        log_method(str(self), extra={'error_data': self.to_dict()})
    
    def with_details(self, **kwargs) -> 'WednesdayError':
        """Add more details to this exception (fluent interface)."""
        self.details.update(kwargs)
        self._full_message = None  # Reset message cache
        return self
    
    def with_cause(self, cause: Exception) -> 'WednesdayError':
        """Add cause to this exception (fluent interface)."""
        self.cause = cause
        return self
    
    @classmethod
    def create(cls, name: str, base: Optional[type] = None, **attributes):
        """
        Dynamically create a new exception type at runtime.
        
        This allows INFINITE extension - create ANY exception type ANY time.
        
        Args:
            name: Name of the new exception class
            base: Base class (defaults to WednesdayError)
            **attributes: Additional class attributes
        
        Returns:
            A new exception class
        """
        if base is None:
            base = cls
        
        # Create new exception type
        new_exception = type(name, (base,), attributes)
        
        # Register it
        with cls._lock:
            cls._exception_registry[name] = new_exception
        
        return new_exception
    
    @classmethod
    def get_exception_class(cls, name: str) -> Optional[type]:
        """Get an exception class by name."""
        return cls._exception_registry.get(name)
    
    @classmethod
    def all_exception_types(cls) -> Dict[str, type]:
        """Get all registered exception types."""
        return cls._exception_registry.copy()


# ============================================================================
# DYNAMIC EXCEPTION FACTORY - CREATE ANY EXCEPTION AT RUNTIME
# ============================================================================

class ExceptionFactory:
    """
    Factory for creating ANY exception type at runtime.
    No limits on what can be created.
    """
    
    @staticmethod
    def create_exception_type(
        name: str,
        base: type = WednesdayError,
        docstring: str = None,
        **attributes
    ) -> type:
        """
        Create a new exception type dynamically.
        
        Args:
            name: Name of the exception class
            base: Base class (defaults to WednesdayError)
            docstring: Optional docstring
            **attributes: Additional class attributes
        
        Returns:
            A new exception class
        """
        # Create docstring if provided
        if docstring:
            attributes['__doc__'] = docstring
        
        # Create the class
        exception_class = type(name, (base,), attributes)
        
        return exception_class
    
    @staticmethod
    def create_exception(
        name: str,
        message: Any = None,
        severity: str = "ERROR",
        **kwargs
    ) -> WednesdayError:
        """
        Create an exception instance of ANY type.
        If the type doesn't exist, it's created automatically.
        
        Args:
            name: Exception type name
            message: Error message
            severity: Error severity
            **kwargs: Additional arguments
        
        Returns:
            An exception instance
        """
        # Get or create exception class
        exception_class = WednesdayError.get_exception_class(name)
        if exception_class is None:
            exception_class = ExceptionFactory.create_exception_type(name)
        
        # Create instance
        return exception_class(message, severity=severity, **kwargs)
    
    @staticmethod
    def register_exception_module(module_name: str):
        """
        Register an entire module of exceptions.
        
        Args:
            module_name: Name of module containing exception classes
        """
        try:
            module = __import__(module_name, fromlist=['*'])
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, Exception):
                    WednesdayError._exception_registry[attr_name] = attr
        except ImportError:
            pass


# ============================================================================
# CONTEXT MANAGER FOR ERROR HANDLING
# ============================================================================

class ErrorContext:
    """
    Context manager for capturing and handling errors with full context.
    No limits on what can be captured or handled.
    """
    
    def __init__(self, *error_types, handler: Callable = None, reraise: bool = False):
        """
        Initialize error context.
        
        Args:
            *error_types: Error types to catch (empty = catch all)
            handler: Custom handler function
            reraise: Whether to reraise after handling
        """
        self.error_types = error_types or (Exception,)
        self.handler = handler
        self.reraise = reraise
        self.caught_exceptions = []
        self.context_data = {}
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context, handling any exception."""
        if exc_val is None:
            return True
        
        # Check if we should handle this exception
        if not isinstance(exc_val, self.error_types):
            return False
        
        # Add context
        if hasattr(exc_val, 'with_details'):
            exc_val.with_details(context=self.context_data)
        
        # Store caught exception
        self.caught_exceptions.append(exc_val)
        
        # Call handler if provided
        if self.handler:
            result = self.handler(exc_val)
            if result is False:  # Handler says don't continue
                return False
        
        # Log the exception
        if hasattr(exc_val, 'log'):
            exc_val.log()
        
        # Reraise if requested
        if self.reraise:
            return False
        
        # Exception handled
        return True
    
    def add_context(self, **kwargs):
        """Add context data for errors."""
        self.context_data.update(kwargs)
        return self


# ============================================================================
# ERROR AGGREGATOR - COLLECT MULTIPLE ERRORS
# ============================================================================

class ErrorAggregator(WednesdayError):
    """
    Aggregates multiple errors into one.
    No limit on number of errors that can be aggregated.
    """
    
    def __init__(self, message: str = "Multiple errors occurred", **kwargs):
        super().__init__(message, **kwargs)
        self.errors = []
        self._lock = threading.RLock()
    
    def add_error(self, error: Exception):
        """Add an error to the aggregate."""
        with self._lock:
            self.errors.append(error)
        return self
    
    def add_errors(self, errors: List[Exception]):
        """Add multiple errors."""
        with self._lock:
            self.errors.extend(errors)
        return self
    
    @property
    def count(self) -> int:
        """Number of aggregated errors."""
        return len(self.errors)
    
    @property
    def has_errors(self) -> bool:
        """Whether there are any errors."""
        return len(self.errors) > 0
    
    def __iter__(self):
        """Iterate over aggregated errors."""
        return iter(self.errors)
    
    def __getitem__(self, index):
        """Get error by index."""
        return self.errors[index]
    
    def to_dict(self, max_depth: int = None) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = super().to_dict(max_depth)
        result['error_count'] = self.count
        result['errors'] = [
            e.to_dict(max_depth) if isinstance(e, WednesdayError)
            else {'type': type(e).__name__, 'message': str(e)}
            for e in self.errors
        ]
        return result


# ============================================================================
# ERROR RETRY HANDLER (FIXED VERSION - NO NONLOCAL ISSUE)
# ============================================================================

class RetryHandler:
    """
    Handle errors with retry logic.
    Unlimited retries, unlimited backoff strategies.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        max_backoff: float = 60.0,
        retry_on: tuple = (Exception,),
        on_retry: Callable = None
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.retry_on = retry_on
        self.on_retry = on_retry
        self.retry_count = 0
        self.last_error = None
    
    def __call__(self, func: Callable, *args, **kwargs):
        """Execute function with retries."""
        self.retry_count = 0
        self.last_error = None
        
        while self.retry_count <= self.max_retries:
            try:
                return func(*args, **kwargs)
            except self.retry_on as e:
                self.last_error = e
                self.retry_count += 1
                
                if self.retry_count > self.max_retries:
                    # Create an aggregator error with all attempts
                    if hasattr(self, 'errors'):
                        agg_error = ErrorAggregator(f"Failed after {self.max_retries} retries")
                        agg_error.add_errors(self.errors)
                        raise agg_error from e
                    else:
                        raise
                
                # Store error for aggregation
                if not hasattr(self, 'errors'):
                    self.errors = []
                self.errors.append(e)
                
                if self.on_retry:
                    self.on_retry(self.retry_count, e)
                
                # Calculate backoff
                backoff = min(
                    self.backoff_factor * (2 ** (self.retry_count - 1)),
                    self.max_backoff
                )
                
                time.sleep(backoff)
        
        return None
    
    def reset(self):
        """Reset retry counter."""
        self.retry_count = 0
        self.last_error = None
        if hasattr(self, 'errors'):
            self.errors = []


# ============================================================================
# ERROR OBSERVER - MONITOR ERRORS GLOBALLY
# ============================================================================

class ErrorObserver:
    """
    Global error observer - watch ALL errors in the system.
    No limits on number of observers or errors observed.
    """
    
    _observers = []
    _lock = threading.RLock()
    
    @classmethod
    def register(cls, callback: Callable):
        """Register an observer callback."""
        with cls._lock:
            cls._observers.append(weakref.ref(callback))
    
    @classmethod
    def unregister(cls, callback: Callable):
        """Unregister an observer."""
        with cls._lock:
            cls._observers = [
                ref for ref in cls._observers
                if ref() is not None and ref() != callback
            ]
    
    @classmethod
    def notify(cls, error: Exception):
        """Notify all observers of an error."""
        dead_refs = []
        
        with cls._lock:
            for ref in cls._observers:
                callback = ref()
                if callback is None:
                    dead_refs.append(ref)
                else:
                    try:
                        callback(error)
                    except:
                        pass
            
            # Clean up dead references
            for ref in dead_refs:
                cls._observers.remove(ref)


# ============================================================================
# DECORATORS FOR ERROR HANDLING
# ============================================================================

def catch_errors(
    *error_types,
    handler: Callable = None,
    reraise: bool = False,
    default_return: Any = None
):
    """
    Decorator to catch and handle errors.
    
    Args:
        *error_types: Error types to catch
        handler: Custom handler function
        reraise: Whether to reraise after handling
        default_return: Default return value on error
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types or Exception as e:
                # Notify observers
                ErrorObserver.notify(e)
                
                # Call handler if provided
                if handler:
                    result = handler(e, *args, **kwargs)
                    if result is not None:
                        return result
                
                # Log the error
                if hasattr(e, 'log'):
                    e.log()
                
                # Reraise if requested
                if reraise:
                    raise
                
                # Return default
                return default_return
        return wrapper
    return decorator


def retry(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    max_backoff: float = 60.0,
    retry_on: tuple = (Exception,)
):
    """
    Decorator to retry functions on error.
    
    Args:
        max_retries: Maximum number of retries
        backoff_factor: Backoff multiplier
        max_backoff: Maximum backoff time
        retry_on: Exception types to retry on
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            handler = RetryHandler(max_retries, backoff_factor, max_backoff, retry_on)
            return handler(func, *args, **kwargs)
        return wrapper
    return decorator


def error_boundary(default_return: Any = None):
    """
    Decorator that creates an error boundary - prevents errors from propagating.
    
    Args:
        default_return: Default return value on error
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ErrorObserver.notify(e)
                if hasattr(e, 'log'):
                    e.log()
                return default_return
        return wrapper
    return decorator


# ============================================================================
# UNLIMITED EXCEPTION REGISTRY - CREATE ANY EXCEPTION TYPE
# ============================================================================

# Pre-create common exception types (can add infinitely more)
EXCEPTION_TYPES = {}

# Configuration errors
EXCEPTION_TYPES['ConfigurationError'] = ExceptionFactory.create_exception_type(
    'ConfigurationError',
    docstring="Base class for configuration errors."
)

EXCEPTION_TYPES['ConfigFileNotFoundError'] = ExceptionFactory.create_exception_type(
    'ConfigFileNotFoundError',
    base=EXCEPTION_TYPES['ConfigurationError'],
    docstring="Configuration file not found."
)

EXCEPTION_TYPES['ConfigValidationError'] = ExceptionFactory.create_exception_type(
    'ConfigValidationError',
    base=EXCEPTION_TYPES['ConfigurationError'],
    docstring="Configuration validation failed."
)

EXCEPTION_TYPES['ConfigTypeError'] = ExceptionFactory.create_exception_type(
    'ConfigTypeError',
    base=EXCEPTION_TYPES['ConfigurationError'],
    docstring="Wrong type in configuration."
)

EXCEPTION_TYPES['ConfigMissingError'] = ExceptionFactory.create_exception_type(
    'ConfigMissingError',
    base=EXCEPTION_TYPES['ConfigurationError'],
    docstring="Required configuration missing."
)

# Module errors
EXCEPTION_TYPES['ModuleError'] = ExceptionFactory.create_exception_type(
    'ModuleError',
    docstring="Base class for module errors."
)

EXCEPTION_TYPES['ModuleNotFoundError'] = ExceptionFactory.create_exception_type(
    'ModuleNotFoundError',
    base=EXCEPTION_TYPES['ModuleError'],
    docstring="Module not found."
)

EXCEPTION_TYPES['ModuleLoadError'] = ExceptionFactory.create_exception_type(
    'ModuleLoadError',
    base=EXCEPTION_TYPES['ModuleError'],
    docstring="Failed to load module."
)

# Memory errors
EXCEPTION_TYPES['MemoryError'] = ExceptionFactory.create_exception_type(
    'MemoryError',
    docstring="Base class for memory errors."
)

EXCEPTION_TYPES['MemoryCapacityError'] = ExceptionFactory.create_exception_type(
    'MemoryCapacityError',
    base=EXCEPTION_TYPES['MemoryError'],
    docstring="Memory capacity exceeded."
)

EXCEPTION_TYPES['MemoryNotFoundError'] = ExceptionFactory.create_exception_type(
    'MemoryNotFoundError',
    base=EXCEPTION_TYPES['MemoryError'],
    docstring="Memory not found."
)

EXCEPTION_TYPES['MemoryCorruptionError'] = ExceptionFactory.create_exception_type(
    'MemoryCorruptionError',
    base=EXCEPTION_TYPES['MemoryError'],
    docstring="Memory corruption detected."
)

# Perception errors
EXCEPTION_TYPES['PerceptionError'] = ExceptionFactory.create_exception_type(
    'PerceptionError',
    docstring="Base class for perception errors."
)

EXCEPTION_TYPES['InputTooLongError'] = ExceptionFactory.create_exception_type(
    'InputTooLongError',
    base=EXCEPTION_TYPES['PerceptionError'],
    docstring="Input exceeds maximum length."
)

EXCEPTION_TYPES['UnsupportedFormatError'] = ExceptionFactory.create_exception_type(
    'UnsupportedFormatError',
    base=EXCEPTION_TYPES['PerceptionError'],
    docstring="Unsupported input format."
)

# Emotion errors
EXCEPTION_TYPES['EmotionError'] = ExceptionFactory.create_exception_type(
    'EmotionError',
    docstring="Base class for emotion errors."
)

EXCEPTION_TYPES['InvalidMoodError'] = ExceptionFactory.create_exception_type(
    'InvalidMoodError',
    base=EXCEPTION_TYPES['EmotionError'],
    docstring="Invalid mood specified."
)

# Cognition errors
EXCEPTION_TYPES['CognitionError'] = ExceptionFactory.create_exception_type(
    'CognitionError',
    docstring="Base class for cognition errors."
)

EXCEPTION_TYPES['ReasoningTimeoutError'] = ExceptionFactory.create_exception_type(
    'ReasoningTimeoutError',
    base=EXCEPTION_TYPES['CognitionError'],
    docstring="Reasoning timeout."
)

# Learning errors
EXCEPTION_TYPES['LearningError'] = ExceptionFactory.create_exception_type(
    'LearningError',
    docstring="Base class for learning errors."
)

# Language errors
EXCEPTION_TYPES['LanguageError'] = ExceptionFactory.create_exception_type(
    'LanguageError',
    docstring="Base class for language errors."
)

EXCEPTION_TYPES['GenerationError'] = ExceptionFactory.create_exception_type(
    'GenerationError',
    base=EXCEPTION_TYPES['LanguageError'],
    docstring="Failed to generate response."
)

# Executive errors
EXCEPTION_TYPES['ExecutiveError'] = ExceptionFactory.create_exception_type(
    'ExecutiveError',
    docstring="Base class for executive errors."
)

# Interface errors
EXCEPTION_TYPES['InterfaceError'] = ExceptionFactory.create_exception_type(
    'InterfaceError',
    docstring="Base class for interface errors."
)

EXCEPTION_TYPES['APIError'] = ExceptionFactory.create_exception_type(
    'APIError',
    base=EXCEPTION_TYPES['InterfaceError'],
    docstring="API error."
)

# Data errors
EXCEPTION_TYPES['DataError'] = ExceptionFactory.create_exception_type(
    'DataError',
    docstring="Base class for data errors."
)

EXCEPTION_TYPES['DatabaseConnectionError'] = ExceptionFactory.create_exception_type(
    'DatabaseConnectionError',
    base=EXCEPTION_TYPES['DataError'],
    docstring="Database connection error."
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_exception_for_log(e: Exception) -> Dict[str, Any]:
    """
    Format ANY exception for logging.
    No limits on what can be formatted.
    """
    if isinstance(e, WednesdayError):
        return e.to_dict()
    
    return {
        'exception_id': str(uuid.uuid4()),
        'type': type(e).__name__,
        'severity': 'ERROR',
        'message': str(e),
        'timestamp': datetime.now().isoformat(),
        'traceback': traceback.format_exc(),
    }


def should_shutdown(e: Exception) -> bool:
    """
    Determine if an exception requires shutdown.
    Customizable for ANY error type.
    """
    if isinstance(e, WednesdayError):
        # CRITICAL and FATAL require shutdown
        return e.severity in ('CRITICAL', 'FATAL')
    
    # By default, only SystemExit and KeyboardInterrupt cause shutdown
    return isinstance(e, (SystemExit, KeyboardInterrupt))


def safe_execute(func: Callable, *args, default: Any = None, **kwargs):
    """
    Safely execute a function, catching ANY errors.
    No limits on what functions can be executed.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        ErrorObserver.notify(e)
        if hasattr(e, 'log'):
            e.log()
        return default


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("INFINITE EXCEPTIONS SYSTEM - DEMONSTRATION")
    print("=" * 60)
    
    # Create a custom exception type dynamically
    MySpecialError = ExceptionFactory.create_exception_type(
        'MySpecialError',
        docstring="A completely custom error type"
    )
    
    # Create and raise a custom error
    try:
        error = MySpecialError(
            "Something went wrong",
            severity="CRITICAL",
            custom_data={"key": "value", "number": 42},
            error_code="ERR-1234"
        )
        raise error
    except WednesdayError as e:
        print(f"\n📌 Caught: {e}")
        print(f"   Severity: {e.severity}")
        print(f"   Error ID: {e.exception_id}")
        print(f"   Details: {e.details}")
    
    # Create an error with the factory
    print(f"\n🔧 Creating error dynamically...")
    error = ExceptionFactory.create_exception(
        'RuntimeError',
        "Created at runtime!",
        severity="WARNING",
        dynamic_field="This field didn't exist before"
    )
    print(f"   {error}")
    
    # Use error context
    print(f"\n🛡️ Using error context...")
    with ErrorContext(Exception, handler=lambda e: print(f"   Handled: {e}")):
        x = 1 / 0  # This will be caught
    
    # Use retry handler
    print(f"\n🔄 Testing retry handler...")
    
    @retry(max_retries=3)
    def failing_function():
        if not hasattr(failing_function, 'count'):
            failing_function.count = 0
        failing_function.count += 1
        print(f"   Attempt {failing_function.count}...")
        if failing_function.count < 3:
            raise ValueError("Not ready yet")
        return "Success!"
    
    result = failing_function()
    print(f"   Result: {result}")
    
    # Aggregate multiple errors
    print(f"\n📚 Testing error aggregation...")
    aggregator = ErrorAggregator("Multiple validation errors")
    aggregator.add_error(ValueError("Field 'name' is required"))
    aggregator.add_error(TypeError("Field 'age' must be integer"))
    aggregator.add_error(KeyError("Missing key 'data'"))
    
    print(f"   Aggregated {aggregator.count} errors:")
    for i, err in enumerate(aggregator):
        print(f"     {i+1}. {type(err).__name__}: {err}")
    
    # Show all registered exception types
    print(f"\n📋 Registered exception types: {len(WednesdayError.all_exception_types())}")
    sample_types = list(WednesdayError.all_exception_types().keys())[:5]
    print(f"   Sample: {sample_types}")
    
    print(f"\n✅ System ready - INFINITE exception handling!")
