"""
Wednesday AI - A human-like cognitive AI with personality.
Version 0.1.0 - Core Foundation Phase

Wednesday is an AI with memory, emotion, self-awareness, cognition,
learning, and language capabilities - all wrapped in Wednesday Addams' 
distinctive personality: deadpan humor, dark wit, and analytical curiosity.

This package provides the main Wednesday class and core utilities.
"""

__version__ = "0.1.0"
__version_name__ = "Addams Family Reunion"
__phases_completed__ = ["core"]  # Track completed build phases
__author__ = "Wednesday AI Team"
__description__ = "Cognitive AI with human-like memory, emotion, and self-awareness"

import logging
from typing import Optional, Dict, Any

# ============================================================================
# Core exports - what users get with "from wednesday import *"
# ============================================================================

__all__ = [
    # Main class
    'Wednesday',
    
    # Core utilities
    'Config',
    'WednesdayError',
    
    # Version info
    '__version__',
    '__version_name__',
    
    # Helper functions
    'create_default_config',
    'quick_start',
]

# ============================================================================
# Import core components
# ============================================================================

# Import main class
from .main import Wednesday

# Import configuration
from .config import Config

# Import exceptions (base class)
from .exceptions import WednesdayError

# Import constants (selectively)
from .constants import VERSION as _VERSION, VERSION_NAME as _VERSION_NAME

# Ensure version consistency
assert __version__ == _VERSION, "Version mismatch between __init__ and constants"
assert __version_name__ == _VERSION_NAME, "Version name mismatch between __init__ and constants"

# ============================================================================
# Package-level logger
# ============================================================================

# Create a null handler to avoid "No handler found" warnings
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


def get_logger() -> logging.Logger:
    """
    Get the package-level logger.
    
    Returns:
        Logger instance for wednesday package
    """
    return _logger


# ============================================================================
# Helper functions
# ============================================================================

def create_default_config(config_path: str = "./config.yaml") -> None:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path where to save the config file
    
    Example:
        >>> from wednesday import create_default_config
        >>> create_default_config("my_wednesday_config.yaml")
    """
    from .config import create_sample_config
    create_sample_config(config_path)
    print(f"✨ Default configuration created at {config_path}")


def quick_start(config_path: Optional[str] = None) -> 'Wednesday':
    """
    Quick start Wednesday with default or custom configuration.
    
    This is the easiest way to get Wednesday running:
    
    Example:
        >>> from wednesday import quick_start
        >>> wed = quick_start()
        >>> response = wed.process_input("Hello, Wednesday!")
    
    Args:
        config_path: Optional path to custom config file
    
    Returns:
        Initialized Wednesday instance
    """
    wed = Wednesday(config_path)
    success = wed.initialize()
    
    if success:
        print(f"🎭 Wednesday v{__version__} is ready. Personality: Wednesday Addams")
        return wed
    else:
        error_msg = f"Failed to initialize Wednesday: {wed.last_error}"
        raise WednesdayError(error_msg, details={"config_path": config_path})


# ============================================================================
# Package metadata (for packaging tools)
# ============================================================================

def get_package_metadata() -> Dict[str, Any]:
    """
    Get package metadata for documentation or debugging.
    
    Returns:
        Dictionary with package information
    """
    return {
        "name": "wednesday",
        "version": __version__,
        "version_name": __version_name__,
        "description": __description__,
        "author": __author__,
        "phases_completed": __phases_completed__,
        "python_requires": ">=3.9",
        "modules": [
            "core",
            "memory",
            "perception",
            "emotion",
            "self",
            "cognition",
            "learning",
            "language",
            "executive",
            "interface",
        ],
    }


# ============================================================================
# Context manager support (for advanced usage)
# ============================================================================

class WednesdayContext:
    """
    Context manager for Wednesday AI.
    
    Ensures proper initialization and shutdown.
    
    Example:
        >>> with WednesdayContext() as wed:
        ...     response = wed.process_input("Hello")
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.wednesday = Wednesday(config_path)
        self.config_path = config_path
    
    def __enter__(self):
        success = self.wednesday.initialize()
        if not success:
            raise WednesdayError(f"Failed to initialize: {self.wednesday.last_error}")
        return self.wednesday
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wednesday.shutdown()
        if exc_type:
            # Log the exception but don't suppress it
            _logger.error(f"Exception in Wednesday context: {exc_val}")
            return False  # Re-raise the exception
        return True


# ============================================================================
# Command-line interface (if run directly)
# ============================================================================

def main_cli():
    """Simple command-line interface for testing."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Wednesday AI - Command Line Interface")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--create-config", help="Create a default config file at specified path")
    parser.add_argument("--version", "-v", action="store_true", help="Show version information")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--query", "-q", help="Single query to process")
    
    args = parser.parse_args()
    
    if args.version:
        print(f"Wednesday AI v{__version__} - {__version_name__}")
        print(f"Phases completed: {', '.join(__phases_completed__)}")
        sys.exit(0)
    
    if args.create_config:
        create_default_config(args.create_config)
        sys.exit(0)
    
    if args.interactive:
        print(f"🎭 Wednesday AI v{__version__} - Interactive Mode")
        print("Type 'exit' to quit, 'help' for commands\n")
        
        wed = quick_start(args.config)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ('exit', 'quit'):
                    print("Wednesday: Goodbye. It was... tolerable.")
                    break
                
                if user_input.lower() == 'help':
                    print("Commands:")
                    print("  exit/quit - Exit interactive mode")
                    print("  status    - Show Wednesday's current status")
                    print("  config    - Show current configuration")
                    continue
                
                if user_input.lower() == 'status':
                    status = wed.get_status()
                    print(f"\nWednesday Status:")
                    for key, value in status.items():
                        print(f"  {key}: {value}")
                    continue
                
                if user_input.lower() == 'config':
                    print(f"\nEnvironment: {wed.config.get('environment')}")
                    print(f"Debug mode: {wed.config.get('debug')}")
                    print(f"Modules enabled: {wed.config.modules_enabled}")
                    continue
                
                if not user_input:
                    continue
                
                response = wed.process_input(user_input)
                print(f"Wednesday: {response}")
                
            except KeyboardInterrupt:
                print("\n\nWednesday: Interrupted. How... rude.")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        wed.shutdown()
        
    elif args.query:
        wed = quick_start(args.config)
        response = wed.process_input(args.query)
        print(response)
        wed.shutdown()
    
    else:
        parser.print_help()


# ============================================================================
# Module initialization
# ============================================================================

# Log package import
_logger.debug(f"Wednesday AI v{__version__} package imported")


# Run CLI if executed directly
if __name__ == "__main__":
    main_cli()