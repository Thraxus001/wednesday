"""
Manages what Wednesday pays attention to moment to moment.
Like shifting your gaze or mental spotlight - but with her characteristic focus.
Once she's focused on something, it takes a lot to distract her.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Set
from datetime import datetime, timedelta
import threading
import time
import logging
import uuid

logger = logging.getLogger(__name__)

class FocusState(Enum):
    """Different levels/qualities of attention"""
    UNFOCUSED = "unfocused"           # Default state, scanning
    LIGHT = "light"                    # Casual attention
    ENGAGED = "engaged"                 # Active focus
    HYPERFOCUS = "hyperfocus"           # Intense concentration (Wednesday specialty)
    DIVIDED = "divided"                 # Splitting attention
    INTERRUPTED = "interrupted"         # Focus was broken

class FocusPriority(Enum):
    """Why something has attention"""
    URGENT = "urgent"                   # Immediate response needed
    GOAL_DRIVEN = "goal_driven"         # Part of current objective
    STIMULUS_DRIVEN = "stimulus_driven" # Something caught attention
    SOCIAL = "social"                    # Interpersonal demand
    INTERNAL = "internal"                # Self-generated thought
    BACKGROUND = "background"            # Monitoring, not focused

@dataclass
class FocusTarget:
    """What Wednesday is attending to"""
    id: str
    type: str  # 'conversation', 'thought', 'environmental', 'task', etc.
    content: Any
    priority: FocusPriority
    salience_score: float
    start_time: datetime
    expected_duration: float  # seconds
    context: Dict[str, Any] = field(default_factory=dict)
    interrupted_by: Optional[str] = None
    resumed_from: Optional[str] = None
    
    @property
    def elapsed(self) -> float:
        """How long we've been focused"""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def is_expired(self) -> bool:
        """Should focus naturally shift away?"""
        return self.elapsed > self.expected_duration
    
    @property
    def remaining(self) -> float:
        """Seconds remaining in expected focus duration"""
        return max(0, self.expected_duration - self.elapsed)

class FocusManager:
    """
    Directs Wednesday's attentional spotlight.
    She doesn't flit about like others - her focus, once fixed, is formidable.
    """
    
    def __init__(self, salience_detector):
        self.salience = salience_detector
        
        # Current attention state
        self.current_focus: Optional[FocusTarget] = None
        self.focus_state: FocusState = FocusState.UNFOCUSED
        self.focus_lock = threading.RLock()  # For thread safety
        
        # Attention history
        self.attention_history: List[FocusTarget] = []
        self.max_history = 100
        
        # Background monitoring buffer
        self.background_buffer: Dict[str, Dict] = {}
        self.buffer_max_size = 50
        
        # Focus parameters (tuned for Wednesday's personality)
        self.params = {
            'focus_stability': 0.85,        # How resistant to distraction (higher = more stable)
            'switch_cost': 0.3,              # Mental cost of switching focus (0-1)
            'hyperfocus_threshold': 0.9,      # Salience needed for hyperfocus
            'interruption_threshold': 0.95,    # Salience needed to interrupt hyperfocus
            'background_monitoring': True,     # Can we monitor while focused?
            'max_focus_duration': 300,         # 5 minutes before forced break
            'min_focus_duration': 2,           # Minimum focus time (2 seconds)
        }
        
        # Current attention threads/timers
        self._focus_monitor_thread = None
        self.running = False
        
        # Register with executive controller
        self.executive = None  # Will be set by controller
        
        # Track recent interruptions for pattern detection
        self.recent_interruptions: List[Dict] = []
        
    def initialize(self, executive_controller) -> None:
        """Connect to executive and start focus monitoring"""
        self.executive = executive_controller
        self.running = True
        self._start_focus_monitor()
        logger.info("FocusManager initialized and monitoring started")
        
    def process_input(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point from perception pipeline.
        Filters raw input through current focus state.
        """
        if not raw_input:
            return {
                'focused': {},
                'background': {},
                'attention_state': self.focus_state.value,
                'focus_target': None
            }
        
        # Calculate salience for all input elements
        current_goals = self._get_current_goals()
        emotional_state = self._get_emotional_state()
        
        salience_scores = self.salience.calculate_salience(
            raw_input,
            current_goals=current_goals,
            emotional_state=emotional_state
        )
        
        # Check for interruptions
        interruption = self._check_for_interruption(salience_scores, raw_input)
        if interruption:
            self.handle_interruption(interruption)
        
        # Filter input based on current focus
        focused_input = self._apply_attention_filter(raw_input, salience_scores)
        
        # Update attention history with what we processed
        self._log_attention(salience_scores[:3] if salience_scores else [])
        
        return focused_input
    
    def shift_focus(self, 
                   new_target_id: str,
                   target_type: str,
                   content: Any,
                   priority: FocusPriority,
                   salience_score: float,
                   expected_duration: Optional[float] = None,
                   context: Optional[Dict] = None) -> bool:
        """
        Move attention to a new target.
        Returns True if focus shift succeeded, False if resisted.
        """
        with self.focus_lock:
            # Validate inputs
            if not new_target_id or not target_type:
                logger.warning("Invalid focus shift parameters")
                return False
            
            # Calculate if we should shift focus
            should_shift, reason = self._evaluate_focus_shift(
                new_target_id, salience_score, priority
            )
            
            if not should_shift:
                # Wednesday resists unnecessary distractions
                self._log_resisted_shift(new_target_id, reason)
                return False
            
            # Calculate expected duration if not provided
            if expected_duration is None:
                expected_duration = self._calculate_expected_duration(
                    target_type, salience_score
                )
            
            # Enforce minimum focus duration
            expected_duration = max(expected_duration, self.params['min_focus_duration'])
            
            # Archive current focus if exists
            if self.current_focus:
                # Don't log if it's the same target
                if self.current_focus.id != new_target_id:
                    self.current_focus.interrupted_by = new_target_id
                    self.attention_history.append(self.current_focus)
            
            # Create new focus target
            new_focus = FocusTarget(
                id=new_target_id,
                type=target_type,
                content=content,
                priority=priority,
                salience_score=salience_score,
                start_time=datetime.now(),
                expected_duration=expected_duration,
                context=context or {},
                resumed_from=self.current_focus.id if self.current_focus else None
            )
            
            # Update state
            self.current_focus = new_focus
            self.focus_state = self._determine_focus_state(new_focus)
            
            # Notify executive of focus change
            if self.executive:
                self.executive.on_focus_changed(
                    self._focus_to_dict(self.current_focus), 
                    self.focus_state.value
                )
            
            # Trim history if needed
            if len(self.attention_history) > self.max_history:
                self.attention_history = self.attention_history[-self.max_history:]
            
            logger.info(f"Focus shifted to {target_type}: {new_target_id[:8]} (priority: {priority.value})")
            return True
    
    def get_focused_input(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter input based on current focus.
        Only returns elements related to current focus (unless background monitoring).
        """
        if not raw_input:
            return {
                'focused': {},
                'background': {},
                'attention_state': self.focus_state.value,
                'focus_target': None
            }
        
        if not self.current_focus:
            # No active focus, return everything (but with attention markers)
            return {
                'focused': {k: {'data': v, 'attention': 'scanning'} for k, v in raw_input.items()},
                'background': {},
                'attention_state': self.focus_state.value,
                'focus_target': None
            }
        
        # Filter based on focus
        focused_elements = {}
        background_elements = {}
        
        for elem_id, elem_data in raw_input.items():
            # Check if element relates to current focus
            if self._is_related_to_focus(elem_id, elem_data):
                focused_elements[elem_id] = {
                    'data': elem_data,
                    'attention': 'focused'
                }
            elif self.params['background_monitoring']:
                # Keep a compressed version for background
                compressed = self._compress_for_background(elem_data)
                if compressed is not None:  # Only add if compression succeeded
                    background_elements[elem_id] = {
                        'data': compressed,
                        'attention': 'background'
                    }
                    # Store in buffer for potential later retrieval
                    self._update_background_buffer(elem_id, compressed)
        
        return {
            'focused': focused_elements,
            'background': background_elements,
            'attention_state': self.focus_state.value,
            'focus_target': self._focus_to_dict(self.current_focus)
        }
    
    def handle_interruption(self, interrupt_signal: Dict[str, Any]) -> bool:
        """
        Decide if focus should be redirected to interruption.
        Wednesday is hard to distract, but not impossible.
        """
        with self.focus_lock:
            if not interrupt_signal:
                return False
            
            # Log interruption attempt
            self.recent_interruptions.append({
                'timestamp': datetime.now(),
                'source': interrupt_signal.get('id', 'unknown'),
                'salience': interrupt_signal.get('salience', 0)
            })
            # Keep only recent interruptions
            self.recent_interruptions = self.recent_interruptions[-10:]
            
            if not self.current_focus:
                # Nothing to interrupt, shift focus to interruption
                return self.shift_focus(
                    interrupt_signal['id'],
                    interrupt_signal.get('type', 'interruption'),
                    interrupt_signal.get('content', {}),
                    FocusPriority.URGENT,
                    interrupt_signal['salience'],
                    expected_duration=interrupt_signal.get('expected_duration')
                )
            
            # Calculate interruption resistance based on current state
            resistance = self._calculate_interruption_resistance()
            
            # Compare salience of interruption vs current focus
            interruption_salience = interrupt_signal['salience']
            current_salience = self.current_focus.salience_score
            
            # Wednesday's hyperfocus is extremely resistant
            if self.focus_state == FocusState.HYPERFOCUS:
                required_salience = current_salience * self.params['interruption_threshold']
                if interruption_salience < required_salience:
                    self._log_interruption_attempt(interrupt_signal, resisted=True)
                    logger.debug(f"Resisted interruption during hyperfocus (required: {required_salience:.2f})")
                    return False
            
            # Regular focus - use resistance factor
            elif interruption_salience < (current_salience * resistance):
                self._log_interruption_attempt(interrupt_signal, resisted=True)
                logger.debug(f"Resisted interruption (required: {current_salience * resistance:.2f})")
                return False
            
            # Interruption succeeds
            self._log_interruption_attempt(interrupt_signal, resisted=False)
            logger.info(f"Allowing interruption with salience {interruption_salience:.2f}")
            
            # Shift focus to interruption
            return self.shift_focus(
                interrupt_signal['id'],
                interrupt_signal.get('type', 'interruption'),
                interrupt_signal.get('content', {}),
                FocusPriority.URGENT,
                interruption_salience
            )
    
    def get_current_focus(self) -> Optional[Dict]:
        """Return current focus state for other modules"""
        if not self.current_focus:
            return None
        
        return self._focus_to_dict(self.current_focus)
    
    def get_background_element(self, element_id: str) -> Optional[Dict]:
        """Retrieve a background element from buffer if needed"""
        return self.background_buffer.get(element_id)
    
    def _evaluate_focus_shift(self, 
                             target_id: str, 
                             salience: float, 
                             priority: FocusPriority) -> Tuple[bool, str]:
        """Determine if we should shift focus to new target"""
        if not self.current_focus:
            return True, "no_current_focus"
        
        # Never shift to same target
        if self.current_focus.id == target_id:
            return False, "already_focused"
        
        # Urgent priorities always shift focus
        if priority == FocusPriority.URGENT:
            return True, "urgent"
        
        # Check if current focus expired
        if self.current_focus.is_expired:
            return True, "current_expired"
        
        # Check if current focus has been active long enough
        if self.current_focus.elapsed < self.params['min_focus_duration']:
            return False, "focus_too_recent"
        
        # Compare salience with current focus
        salience_threshold = self.current_focus.salience_score * self.params['focus_stability']
        
        if salience > salience_threshold:
            return True, "higher_salience"
        
        return False, "insufficient_salience"
    
    def _determine_focus_state(self, target: FocusTarget) -> FocusState:
        """Determine what kind of focus we're in"""
        if target.priority == FocusPriority.URGENT:
            return FocusState.ENGAGED
        
        if target.salience_score > self.params['hyperfocus_threshold']:
            return FocusState.HYPERFOCUS
        
        if target.priority == FocusPriority.GOAL_DRIVEN:
            return FocusState.ENGAGED
        
        if target.priority == FocusPriority.SOCIAL:
            return FocusState.LIGHT
        
        return FocusState.LIGHT
    
    def _calculate_interruption_resistance(self) -> float:
        """How resistant to interruption are we right now?"""
        if not self.current_focus:
            return 0.5
        
        base_resistance = self.params['focus_stability']
        
        # Modify based on focus state
        if self.focus_state == FocusState.HYPERFOCUS:
            base_resistance *= 1.5
        elif self.focus_state == FocusState.ENGAGED:
            base_resistance *= 1.2
        elif self.focus_state == FocusState.LIGHT:
            base_resistance *= 0.8
        
        # Modify based on time (fatigue reduces resistance)
        fatigue = min(0.3, self.current_focus.elapsed / 600)  # Max 30% reduction after 10 min
        base_resistance *= (1 - fatigue)
        
        # Recent interruptions make us more resistant (annoyance factor)
        recent_count = len([i for i in self.recent_interruptions 
                          if (datetime.now() - i['timestamp']).total_seconds() < 30])
        if recent_count > 3:
            base_resistance *= 1.2  # More resistant to frequent interruptions
        
        return min(1.0, max(0.2, base_resistance))
    
    def _apply_attention_filter(self, 
                              raw_input: Dict[str, Any], 
                              salience_scores: List) -> Dict[str, Any]:
        """Filter input based on current attention"""
        # Get top salience elements
        top_elements = [s.element_id for s in salience_scores[:3]]
        
        # Get focused input using get_focused_input
        focused = self.get_focused_input(raw_input)
        
        return {
            'raw': raw_input,
            'salient': {k: raw_input[k] for k in top_elements if k in raw_input},
            'attention_focus': self.get_current_focus(),
            'focused_elements': focused['focused'],
            'background_elements': focused['background']
        }
    
    def _check_for_interruption(self, salience_scores: List, raw_input: Dict) -> Optional[Dict]:
        """Check if any input element deserves interruption"""
        if not salience_scores:
            return None
        
        for score in salience_scores:
            if score.requires_immediate or score.overall_score > 0.9:
                # Get the actual content from raw_input
                content = raw_input.get(score.element_id, {})
                return {
                    'id': score.element_id,
                    'type': 'external_stimulus',
                    'content': content,
                    'salience': score.overall_score,
                    'expected_duration': score.attention_duration
                }
        return None
    
    def _is_related_to_focus(self, element_id: str, element_data: Any) -> bool:
        """Check if input element relates to current focus"""
        if not self.current_focus:
            return False
        
        # Convert to string for comparison
        element_str = str(element_data).lower()
        focus_id = self.current_focus.id.lower()
        focus_type = self.current_focus.type.lower()
        
        # Simple matching - could be enhanced with embeddings
        return (focus_id in element_str or 
                focus_type in element_str or
                (isinstance(self.current_focus.content, str) and 
                 self.current_focus.content.lower() in element_str))
    
    def _compress_for_background(self, data: Any) -> Any:
        """Create compressed version for background monitoring"""
        try:
            if data is None:
                return None
            
            if isinstance(data, dict):
                # Keep only first 3 keys, compress values
                compressed = {}
                for i, (k, v) in enumerate(list(data.items())[:3]):
                    compressed[k] = self._compress_for_background(v)
                return compressed
            
            if isinstance(data, (list, tuple)):
                # Keep first 2 items, compress each
                return [self._compress_for_background(item) for item in data[:2]]
            
            if isinstance(data, str):
                # Truncate long strings
                return data[:100] + "..." if len(data) > 100 else data
            
            # For primitive types, return as is
            return data
        except Exception as e:
            logger.debug(f"Error compressing data for background: {e}")
            return str(data)[:50]  # Fallback
    
    def _update_background_buffer(self, element_id: str, compressed_data: Any) -> None:
        """Update background buffer with compressed element"""
        self.background_buffer[element_id] = {
            'data': compressed_data,
            'timestamp': datetime.now()
        }
        
        # Trim buffer if too large
        if len(self.background_buffer) > self.buffer_max_size:
            # Remove oldest entries
            sorted_items = sorted(
                self.background_buffer.items(),
                key=lambda x: x[1]['timestamp']
            )
            for old_id, _ in sorted_items[:10]:  # Remove 10 oldest
                del self.background_buffer[old_id]
    
    def _calculate_expected_duration(self, target_type: str, salience: float) -> float:
        """How long should we expect to focus on this?"""
        base_durations = {
            'conversation': 60,
            'thought': 30,
            'environmental': 10,
            'task': 120,
            'social_interaction': 90,
            'interruption': 15,
            'external_stimulus': 5
        }
        
        base = base_durations.get(target_type, 30)
        # More salient things deserve more time, but cap at max
        duration = base * (0.5 + salience)
        return min(duration, self.params['max_focus_duration'])
    
    def _get_current_goals(self) -> List:
        """Get current goals from executive"""
        if self.executive and hasattr(self.executive, 'get_active_goals'):
            return self.executive.get_active_goals()
        return []
    
    def _get_emotional_state(self) -> Dict:
        """Get current emotional state"""
        # Would query emotion module if available
        if self.executive and hasattr(self.executive, 'get_emotional_state'):
            return self.executive.get_emotional_state()
        return {}
    
    def _focus_to_dict(self, focus: FocusTarget) -> Dict:
        """Convert FocusTarget to dictionary for external use"""
        return {
            'id': focus.id,
            'type': focus.type,
            'priority': focus.priority.value,
            'salience': focus.salience_score,
            'duration': focus.elapsed,
            'remaining': focus.remaining,
            'state': self.focus_state.value
        }
    
    def _log_attention(self, salience_scores: List) -> None:
        """Log what we attended to"""
        if salience_scores and logger.isEnabledFor(logging.DEBUG):
            top = salience_scores[0] if salience_scores else None
            if top:
                logger.debug(f"Attending to {top.element_id[:8]} (score: {top.overall_score:.2f})")
    
    def _log_resisted_shift(self, target_id: str, reason: str) -> None:
        """Log when we resist shifting focus"""
        logger.debug(f"Resisted focus shift to {target_id[:8]}: {reason}")
    
    def _log_interruption_attempt(self, interrupt: Dict, resisted: bool) -> None:
        """Log interruption attempts"""
        status = "resisted" if resisted else "allowed"
        logger.debug(f"Interruption {status}: {interrupt.get('id', 'unknown')[:8]}")
    
    def _start_focus_monitor(self) -> None:
        """Background thread to monitor focus state"""
        def monitor():
            logger.info("Focus monitor thread started")
            while self.running:
                try:
                    time.sleep(1)  # Check every second
                    with self.focus_lock:
                        if self.current_focus and self.current_focus.is_expired:
                            # Focus expired, archive it
                            logger.info(f"Focus expired after {self.current_focus.elapsed:.1f}s")
                            self.attention_history.append(self.current_focus)
                            self.current_focus = None
                            self.focus_state = FocusState.UNFOCUSED
                            
                            if self.executive and hasattr(self.executive, 'on_focus_lost'):
                                self.executive.on_focus_lost()
                except Exception as e:
                    logger.error(f"Error in focus monitor: {e}")
            
            logger.info("Focus monitor thread stopped")
        
        self._focus_monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._focus_monitor_thread.start()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get focus manager statistics"""
        return {
            'current_focus': self._focus_to_dict(self.current_focus) if self.current_focus else None,
            'focus_state': self.focus_state.value,
            'total_focus_events': len(self.attention_history),
            'recent_interruptions': len(self.recent_interruptions),
            'background_buffer_size': len(self.background_buffer),
            'params': self.params
        }
    
    def shutdown(self) -> None:
        """Clean shutdown of focus manager"""
        logger.info("Shutting down FocusManager")
        self.running = False
        if self._focus_monitor_thread and self._focus_monitor_thread.is_alive():
            self._focus_monitor_thread.join(timeout=2)
        
        # Archive current focus if exists
        with self.focus_lock:
            if self.current_focus:
                self.attention_history.append(self.current_focus)
                self.current_focus = None

# Connects to: salience.py (uses for importance scores)
# Connects to: perception modules (filters their output based on focus)
# Connects to: executive/controller.py (reports attention state, gets goals)
# Connects to: memory/working/ (stores attention history for context)
# Connects to: emotion/appraisal.py (emotional state affects interruption sensitivity)