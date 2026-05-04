"""
/wednesday/config.py
Configuration management for Wednesday AI - handles settings from multiple sources
with validation and runtime updates.

This module serves as the central configuration hub for all Wednesday AI subsystems.
It provides:
- Structured configuration using dataclasses for type safety
- Multiple configuration sources (defaults, files, environment variables)
- Validation and error checking
- Runtime configuration updates
- Serialization/deserialization support

Connects to: main.py (provides config to all modules)
All modules import this for their settings
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()


class Environment(Enum):
    """Deployment environments enumeration for environment-specific behavior."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class MemoryConfig:
    """
    Memory subsystem configuration.
    
    Controls how Wednesday stores and retrieves information across different
    memory types (working, episodic, semantic).
    """
    working_memory_size: int = 10  # Number of items maintained in active memory
    episodic_index_path: str = "./data/memory/episodic"  # Storage for personal experiences
    semantic_store: str = "sqlite"  # Database type for factual knowledge
    vector_dimension: int = 768  # Embedding dimension (matches many transformer models)
    similarity_threshold: float = 0.75  # Minimum similarity for memory retrieval
    enable_compression: bool = True  # Compress old memories to save space
    retention_days: int = 30  # How long to keep memories before archiving


@dataclass
class PerceptionConfig:
    """
    Perception subsystem configuration.
    
    Controls how Wednesday processes sensory input from various modalities.
    """
    # Text processing settings
    text_encoding_model: str = "all-MiniLM-L6-v2"  # Model for text embeddings
    max_text_length: int = 512  # Maximum characters to process
    
    # Audio processing settings
    enable_audio: bool = False  # Audio input disabled by default
    audio_sample_rate: int = 16000  # Sample rate for audio processing
    speech_recognition_model: str = "base"  # Whisper model size
    
    # Vision processing settings
    enable_vision: bool = False  # Vision input disabled by default
    image_size: tuple = (224, 224)  # Input image dimensions
    face_recognition_tolerance: float = 0.6  # Face matching threshold
    
    # Attention mechanism settings
    attention_span: float = 30.0  # Seconds before attention decays
    distraction_threshold: float = 0.3  # Sensitivity to interruptions


@dataclass
class EmotionConfig:
    """
    Emotion subsystem configuration.
    
    Controls Wednesday's emotional responses and personality traits.
    Includes Wednesday's signature dark humor and sarcasm settings.
    """
    base_mood: str = "neutral"  # Starting emotional state
    mood_update_rate: float = 0.1  # How quickly mood changes (0-1)
    empathy_enabled: bool = True  # Ability to understand others' emotions
    emotional_memory: bool = True  # Remember emotional contexts
    expression_intensity: float = 1.0  # 0.0 (stoic) to 1.0 (expressive)
    
    # Wednesday's distinctive personality traits
    default_sarcasm: float = 0.7  # Sarcasm level (0-1 scale)
    dark_humor_tolerance: float = 0.9  # Appreciation for dark humor (0-1)


@dataclass
class SelfConfig:
    """
    Self-awareness subsystem configuration.
    
    Controls Wednesday's self-reflection and metacognitive abilities.
    """
    personality_file: str = "./data/self/personality.yaml"  # Personality traits storage
    values_file: str = "./data/self/values.yaml"  # Core values and principles
    enable_metacognition: bool = True  # Thinking about thinking
    introspection_frequency: int = 100  # Iterations between self-checks
    theory_of_mind_depth: int = 2  # Levels of recursive thinking about others' thoughts


@dataclass
class CognitionConfig:
    """
    Cognition subsystem configuration.
    
    Controls Wednesday's reasoning and problem-solving capabilities.
    """
    reasoning_depth: int = 3  # Levels of logical deduction
    enable_analogy: bool = True  # Use analogical reasoning
    causal_inference: bool = True  # Understand cause and effect
    problem_solving_timeout: float = 5.0  # Seconds max for problem solving
    max_solutions_to_consider: int = 5  # Number of alternative solutions


@dataclass
class LearningConfig:
    """
    Learning subsystem configuration.
    
    Controls how Wednesday learns from experiences and feedback.
    """
    learning_rate: float = 0.01  # Speed of adaptation (0-1)
    reinforcement_learning: bool = True  # Learn from rewards/punishments
    social_learning: bool = True  # Learn from observing others
    consolidation_interval: int = 3600  # Seconds between memory consolidation
    feedback_weight: float = 0.3  # Importance of user feedback
    exploration_rate: float = 0.1  # Try new things vs exploit known solutions


@dataclass
class LanguageConfig:
    """
    Language subsystem configuration.
    
    Controls Wednesday's language generation and comprehension.
    Includes settings for matching Wednesday's distinctive voice.
    """
    generation_model: str = "gpt2"  # Base model for text generation
    max_response_length: int = 150  # Maximum words in response
    temperature: float = 0.8  # Creativity in responses (0-1)
    top_p: float = 0.9  # Nucleus sampling probability
    repetition_penalty: float = 1.2  # Discourage repetitive responses
    
    # Wednesday's unique voice configuration
    style_file: str = "./data/language/wednesday_style.yaml"  # Style guide
    sarcasm_detection: bool = True  # Detect sarcasm in input


@dataclass
class ExecutiveConfig:
    """
    Executive controller configuration.
    
    Manages task scheduling and module coordination.
    """
    orchestration_mode: str = "sequential"  # Task execution mode (sequential/parallel/priority)
    scheduler_interval: float = 0.1  # Seconds between scheduler ticks
    priority_levels: int = 5  # Number of priority tiers
    max_concurrent_tasks: int = 3  # Maximum parallel tasks
    heartbeat_interval: float = 1.0  # Seconds between health checks


@dataclass
class InterfaceConfig:
    """
    Interface subsystem configuration.
    
    Controls external communication channels (API, WebSocket, etc.).
    """
    api_host: str = "127.0.0.1"  # Bind address for API server
    api_port: int = 8000  # Port for API server
    enable_websocket: bool = True  # Enable real-time communication
    enable_voice_output: bool = False  # Text-to-speech disabled by default
    max_request_size: int = 10 * 1024 * 1024  # Maximum request size (10MB)
    request_timeout: float = 30.0  # Seconds before request timeout


@dataclass
class LoggingConfig:
    """
    Logging configuration.
    
    Controls how Wednesday logs events, errors, and interactions.
    """
    log_level: str = "INFO"  # Minimum log level (DEBUG/INFO/WARNING/ERROR)
    log_to_file: bool = True  # Write logs to files
    log_to_console: bool = True  # Output logs to console
    log_dir: str = "./logs"  # Directory for log files
    log_retention_days: int = 7  # Days to keep log files
    log_structured: bool = False  # Use JSON format for machine parsing
    log_interactions: bool = True  # Log user interactions


@dataclass
class DatabaseConfig:
    """
    Database connections configuration.
    
    Manages connections to various database systems used by Wednesday.
    Supports multiple database types for different storage needs.
    """
    # SQL databases for structured data
    sqlite_path: str = "./data/wednesday.db"  # SQLite database path
    postgres_dsn: Optional[str] = None  # PostgreSQL connection string
    
    # Vector database for embeddings and similarity search
    vector_db_type: str = "faiss"  # Vector DB type (faiss/pinecone/weaviate)
    vector_db_path: str = "./data/vectors"  # Local vector storage path
    
    # Graph database for relationship storage
    graph_db_type: str = "neo4j"  # Graph DB type (neo4j/networkx)
    neo4j_uri: Optional[str] = None  # Neo4j connection URI
    neo4j_user: Optional[str] = None  # Neo4j username
    neo4j_password: Optional[str] = None  # Neo4j password
    
    # Document store for unstructured data
    mongodb_uri: Optional[str] = None  # MongoDB connection URI
    mongodb_db: str = "wednesday"  # MongoDB database name


@dataclass
class WednesdayConfig:
    """
    Main configuration container for Wednesday AI.
    
    Aggregates all subsystem configurations and provides global settings.
    This is the root configuration object used throughout the application.
    """
    
    # Basic system information
    name: str = "Wednesday"  # AI name
    version: str = "0.1.0"  # System version
    environment: Environment = Environment.DEVELOPMENT  # Deployment environment
    debug_mode: bool = False  # Enable debug features
    
    # File system paths
    project_root: Path = Path(__file__).parent.parent  # Project root directory
    data_dir: Path = field(default_factory=lambda: Path("./data"))  # Data storage
    models_dir: Path = field(default_factory=lambda: Path("./models"))  # Model storage
    
    # Subsystem configurations
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    self_model: SelfConfig = field(default_factory=SelfConfig)
    cognition: CognitionConfig = field(default_factory=CognitionConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    executive: ExecutiveConfig = field(default_factory=ExecutiveConfig)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Module enable/disable flags (used by main.py for system composition)
    modules_enabled: Dict[str, bool] = field(default_factory=lambda: {
        'memory': True,       # Memory system always enabled
        'perception': False,  # Start with text only (audio/vision disabled)
        'emotion': True,      # Emotional responses enabled
        'self': True,         # Self-awareness enabled
        'cognition': True,    # Reasoning enabled
        'learning': True,     # Learning enabled
        'language': True,     # Language processing enabled
        'executive': True,    # Executive control enabled
        'interface': False,   # API disabled initially for safety
    })
    
    def __post_init__(self):
        """
        Post-initialization processing.
        
        Converts string paths to Path objects and ensures required directories exist.
        Called automatically after dataclass initialization.
        """
        # Convert string paths to Path objects
        self.data_dir = Path(self.data_dir)
        self.models_dir = Path(self.models_dir)
        
        # Create required directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging directory
        self.logging.log_dir = str(self.data_dir / "logs")
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up database paths relative to data directory
        if self.database.sqlite_path == "./data/wednesday.db":
            self.database.sqlite_path = str(self.data_dir / "wednesday.db")
        
        if self.database.vector_db_path == "./data/vectors":
            self.database.vector_db_path = str(self.data_dir / "vectors")
            Path(self.database.vector_db_path).mkdir(parents=True, exist_ok=True)


class Config:
    """
    Configuration manager for Wednesday AI.
    
    Provides a unified interface for accessing and modifying configuration
    from multiple sources with clear priority:
    1. Environment variables (highest priority - for deployment)
    2. Runtime updates (for dynamic configuration changes)
    3. Config file (for persistent settings)
    4. Defaults (lowest priority - built into dataclasses)
    
    This class handles:
    - Loading configuration from YAML/JSON files
    - Reading environment variables with WEDNESDAY__ prefix
    - Runtime configuration updates
    - Configuration validation
    - Saving configuration to files
    
    Example:
        >>> config = Config("config.yaml")
        >>> memory_size = config.get("memory.working_memory_size")
        >>> config.update_runtime("learning.learning_rate", 0.001)
        >>> module_config = config.get_module_config("emotion")
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to YAML/JSON configuration file.
                        If not provided, only defaults and environment
                        variables will be used.
        """
        self._config = WednesdayConfig()
        self._config_path = Path(config_path) if config_path else None
        self._logger = logging.getLogger(__name__)
        
        # Load configuration in priority order
        self._load_defaults()    # Base configuration from dataclasses
        self._load_from_file()   # Override with file-based config
        self._load_from_env()    # Override with environment variables
        
        # Validate the final configuration
        self._validate()
        
        self._logger.info(f"Configuration loaded successfully. Environment: {self._config.environment.value}")
    
    def _load_defaults(self):
        """
        Load default configuration values.
        
        Defaults are already set in the WednesdayConfig dataclass,
        so this method is primarily for logging and future extension.
        """
        self._logger.debug("Loaded default configuration values")
    
    def _load_from_file(self):
        """
        Load configuration from YAML or JSON file.
        
        Supports both .yaml/.yml and .json file formats.
        Merges loaded values with existing configuration.
        Silently returns if file doesn't exist (using defaults instead).
        """
        if not self._config_path or not self._config_path.exists():
            self._logger.debug(f"No config file found at {self._config_path}")
            return
        
        try:
            with open(self._config_path, 'r') as f:
                # Determine file format by extension
                if self._config_path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                    self._logger.debug(f"Loaded YAML config from {self._config_path}")
                elif self._config_path.suffix == '.json':
                    data = json.load(f)
                    self._logger.debug(f"Loaded JSON config from {self._config_path}")
                else:
                    raise ValueError(f"Unsupported config file format: {self._config_path.suffix}")
            
            # Update configuration with loaded values
            if data and isinstance(data, dict):
                self._update_from_dict(data)
                self._logger.info(f"Successfully loaded configuration from {self._config_path}")
            else:
                self._logger.warning(f"Config file {self._config_path} is empty or invalid")
                
        except yaml.YAMLError as e:
            self._logger.error(f"Error parsing YAML config file: {e}")
        except json.JSONDecodeError as e:
            self._logger.error(f"Error parsing JSON config file: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error loading config file: {e}")
    
    def _load_from_env(self):
        """
        Load configuration from environment variables.
        
        Environment variables must follow the pattern:
        WEDNESDAY__MODULE__KEY = value
        
        Examples:
            WEDNESDAY__MEMORY__WORKING_MEMORY_SIZE=20
            WEDNESDAY__EMOTION__DEFAULT_SARCASM=0.8
            WEDNESDAY__ENVIRONMENT=production
        
        Type conversion is attempted based on the target configuration field.
        """
        prefix = "WEDNESDAY__"
        
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            
            # Parse the environment variable key
            # Example: "WEDNESDAY__MEMORY__WORKING_MEMORY_SIZE" -> ["memory", "working_memory_size"]
            parts = env_key[len(prefix):].lower().split("__")
            
            # Navigate through the configuration hierarchy
            current = self._config
            for part in parts[:-1]:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    self._logger.debug(f"Skipping {env_key}: Unknown section '{part}'")
                    break
            else:
                # Successfully navigated to the parent object
                last_part = parts[-1]
                if hasattr(current, last_part):
                    # Get the current value to determine expected type
                    current_value = getattr(current, last_part)
                    current_type = type(current_value)
                    
                    try:
                        # Convert string to appropriate type
                        if current_type == bool:
                            # Handle various boolean string representations
                            converted = env_value.lower() in ('true', '1', 'yes', 'on')
                        elif current_type == int:
                            converted = int(env_value)
                        elif current_type == float:
                            converted = float(env_value)
                        elif current_type == list:
                            # Split comma-separated values
                            converted = [item.strip() for item in env_value.split(',')]
                        elif current_type == tuple:
                            # Convert to tuple
                            converted = tuple(item.strip() for item in env_value.split(','))
                        elif current_type == Path:
                            converted = Path(env_value)
                        elif current_type == Environment:
                            # Handle enum conversion
                            try:
                                converted = Environment(env_value.lower())
                            except ValueError:
                                self._logger.warning(f"Invalid environment value: {env_value}")
                                continue
                        else:
                            # Default to string
                            converted = env_value
                        
                        # Set the value
                        setattr(current, last_part, converted)
                        self._logger.debug(f"Set {env_key} = {converted}")
                        
                    except ValueError as e:
                        self._logger.warning(f"Could not convert {env_key}={env_value} to {current_type}: {e}")
                    except Exception as e:
                        self._logger.warning(f"Error processing {env_key}: {e}")
                else:
                    self._logger.debug(f"Skipping {env_key}: Unknown key '{last_part}'")
    
    def _update_from_dict(self, data: Dict, prefix: str = ""):
        """
        Recursively update configuration from a dictionary.
        
        Args:
            data: Dictionary containing configuration values
            prefix: Current key prefix for nested structures (used in recursion)
        """
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively handle nested dictionaries
                self._update_from_dict(value, f"{prefix}{key}.")
            else:
                # Set the value using dot notation
                full_key = f"{prefix}{key}"
                try:
                    self.set(full_key, value)
                except (AttributeError, ValueError) as e:
                    self._logger.warning(f"Could not set {full_key}: {e}")
    
    def _validate(self):
        """
        Validate the configuration for correctness and consistency.
        
        Performs various checks:
        - Environment-specific validation (stricter for production)
        - Module dependency validation
        - Critical path validation
        - Value range validation
        
        Raises:
            AssertionError: If validation fails in production
            Logs warnings: If validation fails in development
        """
        # Validate environment-specific settings
        if self._config.environment == Environment.PRODUCTION:
            # Strict validation for production
            self._validate_production()
        else:
            # Lenient validation for development
            self._validate_development()
        
        # Validate module dependencies
        self._validate_module_dependencies()
        
        # Validate value ranges
        self._validate_value_ranges()
    
    def _validate_production(self):
        """
        Strict validation for production environment.
        Raises AssertionError on validation failures.
        """
        # Check log level
        assert self._config.logging.log_level in ["INFO", "WARNING", "ERROR"], \
            f"Invalid log level for production: {self._config.logging.log_level}"
        
        # Check database configuration
        if self._config.modules_enabled.get('memory', False):
            assert self._config.database.postgres_dsn or \
                   self._config.database.sqlite_path != "./data/wednesday.db", \
                "Production requires explicit database configuration (postgres_dsn or custom sqlite_path)"
        
        # Check API security
        if self._config.modules_enabled.get('interface', False):
            assert self._config.interface.api_host != "0.0.0.0" or \
                   self._config.environment != Environment.PRODUCTION, \
                "Warning: API bound to all interfaces in production"
    
    def _validate_development(self):
        """
        Lenient validation for development environment.
        Logs warnings but doesn't raise exceptions.
        """
        # Check log level
        if self._config.logging.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            self._logger.warning(f"Unusual log level: {self._config.logging.log_level}")
        
        # Check database configuration
        if self._config.modules_enabled.get('memory', False) and \
           self._config.database.sqlite_path == "./data/wednesday.db":
            self._logger.info("Using default SQLite database path")
    
    def _validate_module_dependencies(self):
        """
        Validate dependencies between modules.
        
        Some modules require others to function properly.
        This ensures the module enable flags are consistent.
        """
        # Learning requires memory
        if self._config.modules_enabled.get('learning', False) and \
           not self._config.modules_enabled.get('memory', False):
            self._logger.warning("Learning module enabled but memory is disabled - this may cause issues")
        
        # Language requires cognition
        if self._config.modules_enabled.get('language', False) and \
           not self._config.modules_enabled.get('cognition', False):
            self._logger.warning("Language module enabled but cognition is disabled - this may cause issues")
        
        # Emotion works better with memory
        if self._config.modules_enabled.get('emotion', False) and \
           not self._config.modules_enabled.get('memory', False) and \
           self._config.emotion.emotional_memory:
            self._logger.warning("Emotional memory enabled but memory module is disabled")
    
    def _validate_value_ranges(self):
        """
        Validate that configuration values are within acceptable ranges.
        Logs warnings for values outside expected ranges.
        """
        # Check learning rate range
        if not 0 <= self._config.learning.learning_rate <= 1:
            self._logger.warning(f"Learning rate {self._config.learning.learning_rate} outside [0,1]")
        
        # Check emotion intensity range
        if not 0 <= self._config.emotion.expression_intensity <= 1:
            self._logger.warning(f"Expression intensity {self._config.emotion.expression_intensity} outside [0,1]")
        
        # Check sarcasm level range
        if not 0 <= self._config.emotion.default_sarcasm <= 1:
            self._logger.warning(f"Sarcasm level {self._config.emotion.default_sarcasm} outside [0,1]")
        
        # Check temperature range for language generation
        if not 0 < self._config.language.temperature <= 2:
            self._logger.warning(f"Temperature {self._config.language.temperature} outside typical range (0,2]")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Dot notation path (e.g., "memory.working_memory_size")
            default: Value to return if the key doesn't exist
        
        Returns:
            The configuration value, or default if not found
        
        Example:
            >>> memory_size = config.get("memory.working_memory_size", 10)
            >>> sarcasm = config.get("emotion.default_sarcasm", 0.5)
        """
        try:
            value = self._config
            for part in key.split('.'):
                value = getattr(value, part)
            return value
        except (AttributeError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Dot notation path (e.g., "memory.working_memory_size")
            value: Value to set
        
        Raises:
            AttributeError: If the path doesn't exist in the configuration
        
        Example:
            >>> config.set("learning.learning_rate", 0.005)
        """
        parts = key.split('.')
        target = self._config
        
        # Navigate to the parent object
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise AttributeError(f"Configuration has no section '{part}'")
            target = getattr(target, part)
        
        # Set the value on the parent
        last_part = parts[-1]
        if hasattr(target, last_part):
            # Optional: Type checking could be added here
            setattr(target, last_part, value)
            self._logger.debug(f"Set {key} = {value}")
        else:
            raise AttributeError(f"Configuration has no key '{last_part}' in {target}")
    
    def update_runtime(self, key: str, value: Any):
        """
        Update a configuration value at runtime with logging.
        
        This method is specifically for runtime configuration changes
        and logs both old and new values for debugging.
        
        Args:
            key: Dot notation path
            value: New value to set
        
        Example:
            >>> config.update_runtime("learning.learning_rate", 0.001)
        """
        old_value = self.get(key)
        self.set(key, value)
        self._logger.info(f"Runtime config update: {key} = {value} (was {old_value})")
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """
        Get all configuration for a specific module as a dictionary.
        
        Useful for passing module-specific configuration to subsystems.
        
        Args:
            module_name: Name of the module (memory, perception, emotion, etc.)
        
        Returns:
            Dictionary containing all configuration values for the module,
            or empty dict if module doesn't exist
        
        Example:
            >>> memory_config = config.get_module_config("memory")
            >>> print(memory_config["working_memory_size"])
        """
        module_config = getattr(self._config, module_name, None)
        if module_config:
            # Convert dataclass to dictionary
            return asdict(module_config)
        return {}
    
    def save(self, path: Optional[Union[str, Path]] = None, format: str = "yaml"):
        """
        Save current configuration to a file.
        
        Args:
            path: Path to save to (uses config_path from initialization if not provided)
            format: Output format - either "yaml" or "json"
        
        Example:
            >>> config.save("backup_config.yaml")
            >>> config.save(format="json")  # Save to original path as JSON
        """
        save_path = Path(path) if path else self._config_path
        if not save_path:
            save_path = Path("./config.yaml")
            self._logger.info(f"No save path specified, using {save_path}")
        
        # Convert entire configuration to dictionary
        config_dict = asdict(self._config)
        
        # Clean dictionary for serialization
        self._clean_dict_for_serialization(config_dict)
        
        try:
            # Ensure parent directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                if format.lower() == "yaml":
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
                    self._logger.info(f"Configuration saved to {save_path} (YAML)")
                elif format.lower() == "json":
                    json.dump(config_dict, f, indent=2, sort_keys=False)
                    self._logger.info(f"Configuration saved to {save_path} (JSON)")
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
        except Exception as e:
            self._logger.error(f"Failed to save configuration to {save_path}: {e}")
            raise
    
    def _clean_dict_for_serialization(self, d: Dict):
        """
        Recursively clean a dictionary for JSON/YAML serialization.
        
        Converts non-serializable objects (Path, Enum) to strings.
        Modifies the dictionary in place.
        
        Args:
            d: Dictionary to clean
        """
        for key, value in list(d.items()):
            if isinstance(value, dict):
                # Recursively clean nested dictionaries
                self._clean_dict_for_serialization(value)
            elif isinstance(value, Path):
                # Convert Path objects to strings
                d[key] = str(value)
            elif isinstance(value, Enum):
                # Convert Enums to their values
                d[key] = value.value
            elif isinstance(value, (list, tuple)):
                # Clean items in sequences
                d[key] = [
                    str(item) if isinstance(item, (Path, Enum)) else item
                    for item in value
                ]
            elif value is None:
                # Keep None as is
                pass
            elif not isinstance(value, (str, int, float, bool)):
                # Convert any other non-serializable types to string
                d[key] = str(value)
    
    @property
    def raw(self) -> WednesdayConfig:
        """Access the raw WednesdayConfig object."""
        return self._config
    
    def __repr__(self) -> str:
        """String representation of the Config object."""
        enabled_count = sum(1 for v in self._config.modules_enabled.values() if v)
        return f"Config(environment={self._config.environment.value}, modules_enabled={enabled_count}/{len(self._config.modules_enabled)})"
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"Wednesday AI Configuration (v{self._config.version}) - Environment: {self._config.environment.value}"


def create_sample_config(path: Union[str, Path] = "./config.sample.yaml"):
    """
    Create a sample configuration file with all available options.
    
    This is useful for users to see all possible configuration options
    and their default values.
    
    Args:
        path: Where to save the sample configuration file
    
    Example:
        >>> create_sample_config("my_config.yaml")
    """
    config = WednesdayConfig()
    config_dict = asdict(config)
    
    # Clean for serialization using a temporary Config instance
    temp_config = Config()
    temp_config._clean_dict_for_serialization(config_dict)
    
    # Add helpful comments to the YAML file
    with open(path, 'w') as f:
        f.write("# Wednesday AI Configuration File\n")
        f.write("# This file contains all available configuration options.\n")
        f.write("# Uncomment and modify values as needed.\n\n")
        
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Sample configuration created at {path}")
    print(f"   Edit this file to customize Wednesday's behavior.")


# Quick test and demonstration
if __name__ == "__main__":
    # Set up basic logging for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=" * 50)
    print("Wednesday AI Configuration System Test")
    print("=" * 50)
    
    # Create a sample configuration file
    print("\n📝 Creating sample configuration...")
    create_sample_config()
    
    # Test configuration loading
    print("\n🔧 Testing configuration loading...")
    config = Config()
    
    # Display some key configuration values
    print(f"\n📊 Current Configuration:")
    print(f"   Environment: {config.get('environment')}")
    print(f"   Version: {config.get('version')}")
    print(f"   Memory size: {config.get('memory.working_memory_size')}")
    print(f"   Wednesday's sarcasm level: {config.get('emotion.default_sarcasm')}")
    print(f"   Dark humor tolerance: {config.get('emotion.dark_humor_tolerance')}")
    
    # Test runtime update
    print(f"\n🔄 Testing runtime update...")
    old_rate = config.get('learning.learning_rate')
    config.update_runtime("learning.learning_rate", 0.005)
    print(f"   Learning rate: {old_rate} → {config.get('learning.learning_rate')}")
    
    # Test module configuration retrieval
    print(f"\n📦 Getting emotion module config...")
    emotion_config = config.get_module_config("emotion")
    print(f"   Emotion config keys: {list(emotion_config.keys())}")
    
    # Test configuration saving
    print(f"\n💾 Testing configuration save...")
    config.save("./test_config.yaml")
    print(f"   Configuration saved to ./test_config.yaml")
    
    # Test environment variable override (simulated)
    print(f"\n🌍 Testing environment variable handling...")
    os.environ["WEDNESDAY__MEMORY__WORKING_MEMORY_SIZE"] = "42"
    os.environ["WEDNESDAY__EMOTION__DEFAULT_SARCASM"] = "0.9"
    
    config_with_env = Config()
    print(f"   Memory size from env: {config_with_env.get('memory.working_memory_size')} (should be 42)")
    print(f"   Sarcasm from env: {config_with_env.get('emotion.default_sarcasm')} (should be 0.9)")
    
    # Clean up test files
    print(f"\n🧹 Cleaning up test files...")
    Path("./test_config.yaml").unlink(missing_ok=True)
    Path("./config.sample.yaml").unlink(missing_ok=True)
    
    print("\n✅ All tests completed successfully!")
