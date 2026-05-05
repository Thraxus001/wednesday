"""
Symbolic reasoning engine with forward/backward chaining, consistency checking,
and explanation generation. Provides logical deduction capabilities for cognitive architectures.

Features:
- Forward chaining (data-driven) with agenda-based optimization
- Backward chaining (goal-driven) with proof discovery
- Variable unification and binding
- Confidence propagation with fuzzy logic
- Contradiction detection
- Explanation generation with history management
- Rule priority and conflict resolution
- Rete-inspired indexing for efficiency
- Thread-safe operations with atomic updates
- Memory management with configurable limits
- Comprehensive input validation
"""

from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
from itertools import product
import hashlib
from time import time
from threading import RLock

logger = logging.getLogger(__name__)


class Operator(Enum):
    """Logical operators for rule conditions."""
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"
    IMPLIES = "implies"
    EQUIV = "equiv"


class TruthValue(Enum):
    """Truth values with uncertainty."""
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
    CONTRADICTION = "contradiction"


@dataclass
class Variable:
    """Represents a logical variable."""
    name: str
    
    def __hash__(self):
        return hash(("var", self.name))
    
    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        return self.name == other.name
    
    def __str__(self):
        return f"?{self.name}"


@dataclass
class Term:
    """First-order logic term."""
    predicate: str
    args: List[Union[str, int, float, Variable]]
    
    def __post_init__(self):
        """Validate term structure."""
        if not isinstance(self.predicate, str) or not self.predicate:
            raise ValueError("Predicate must be a non-empty string")
        if not isinstance(self.args, list):
            raise TypeError("Args must be a list")
    
    def __hash__(self):
        return hash((self.predicate, tuple(str(arg) for arg in self.args)))
    
    def __eq__(self, other):
        if not isinstance(other, Term):
            return False
        return self.predicate == other.predicate and self.args == other.args
    
    def __str__(self):
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.predicate}({args_str})"


@dataclass
class Rule:
    """Logical rule with antecedents and consequent."""
    antecedents: List[Term]  # List of conditions
    consequent: Term          # Conclusion
    confidence: float = 1.0
    priority: int = 0         # Higher priority rules fire first
    name: Optional[str] = None
    explanation: Optional[str] = None
    is_defeasible: bool = False  # Can be overridden
    enabled: bool = True      # NEW: Rule can be disabled
    
    def __post_init__(self):
        if not self.antecedents:
            raise ValueError("Rules must have at least one antecedent")
        if not all(isinstance(t, Term) for t in self.antecedents):
            raise TypeError("All antecedents must be Terms")
        if not isinstance(self.consequent, Term):
            raise TypeError("Consequent must be a Term")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if self.name is None:
            self.name = f"rule_{hash(self) % 1000000}"
    
    def __hash__(self):
        return hash((self.name, tuple(self.antecedents), self.consequent))


@dataclass
class Fact:
    """Fact in knowledge base with metadata."""
    term: Term
    confidence: float = 1.0
    source: str = 'explicit'
    timestamp: float = field(default_factory=time)
    derivation_rule: Optional[str] = None
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def __hash__(self):
        return hash(("fact", self.term, self.source))
    
    def __eq__(self, other):
        if not isinstance(other, Fact):
            return False
        return self.term == other.term
    
    @property
    def key(self) -> str:
        """Unique key for indexing."""
        return str(self.term)


class UnificationError(Exception):
    """Raised when term unification fails."""
    pass


class Unifier:
    """Handles variable unification between terms."""
    
    @staticmethod
    def unify(term1: Term, term2: Term, bindings: Dict[Variable, Any] = None) -> Optional[Dict[Variable, Any]]:
        """
        Unify two terms with variable binding.
        
        Args:
            term1: First term
            term2: Second term
            bindings: Existing variable bindings
        
        Returns:
            Dictionary of variable bindings if unification succeeds, None otherwise
        """
        if bindings is None:
            bindings = {}
        
        # Same predicate required
        if term1.predicate != term2.predicate:
            return None
        
        if len(term1.args) != len(term2.args):
            return None
        
        new_bindings = bindings.copy()
        
        for arg1, arg2 in zip(term1.args, term2.args):
            if isinstance(arg1, Variable):
                # Bind variable to term
                if arg1 in new_bindings:
                    # Already bound - must match
                    if not Unifier._unify_values(new_bindings[arg1], arg2, new_bindings):
                        return None
                else:
                    # Occurs check for infinite recursion
                    if Unifier._occurs_check(arg1, arg2, new_bindings):
                        return None
                    new_bindings[arg1] = arg2
            
            elif isinstance(arg2, Variable):
                if arg2 in new_bindings:
                    if not Unifier._unify_values(arg1, new_bindings[arg2], new_bindings):
                        return None
                else:
                    if Unifier._occurs_check(arg2, arg1, new_bindings):
                        return None
                    new_bindings[arg2] = arg1
            
            elif isinstance(arg1, Term) and isinstance(arg2, Term):
                # Recursively unify sub-terms
                result = Unifier.unify(arg1, arg2, new_bindings)
                if result is None:
                    return None
                new_bindings = result
            
            elif arg1 != arg2:
                return None
        
        return new_bindings
    
    @staticmethod
    def _unify_values(val1: Any, val2: Any, bindings: Dict) -> bool:
        """Unify two concrete values."""
        if isinstance(val2, Variable):
            # Swap to handle variable on left
            return Unifier._unify_values(val2, val1, bindings) is not None
        return val1 == val2
    
    @staticmethod
    def _occurs_check(var: Variable, term: Any, bindings: Dict) -> bool:
        """Check if variable occurs in term (prevents infinite recursion)."""
        if var == term:
            return True
        
        if isinstance(term, Variable):
            if term in bindings:
                return Unifier._occurs_check(var, bindings[term], bindings)
            return False
        
        if isinstance(term, Term):
            for arg in term.args:
                if Unifier._occurs_check(var, arg, bindings):
                    return True
        
        return False
    
    @staticmethod
    def apply_bindings(term: Union[Term, Variable, Any], bindings: Dict) -> Union[Term, Any]:
        """Apply variable bindings to a term."""
        if isinstance(term, Variable):
            if term in bindings:
                return Unifier.apply_bindings(bindings[term], bindings)
            return term
        
        if isinstance(term, Term):
            new_args = [Unifier.apply_bindings(arg, bindings) for arg in term.args]
            return Term(term.predicate, new_args)
        
        return term


class SymbolicReasoner:
    """
    Symbolic reasoning engine with rule-based inference.
    Thread-safe with efficient indexing for large knowledge bases.
    """
    
    def __init__(self, enable_explanations: bool = True, 
                 max_history_size: int = 10000,
                 confidence_threshold: float = 0.0,
                 enable_debug: bool = False):
        """
        Initialize the symbolic reasoner.
        
        Args:
            enable_explanations: Track inference history
            max_history_size: Maximum inference history entries
            confidence_threshold: Minimum confidence to consider facts
            enable_debug: Enable debug trace mode
        """
        self._lock = RLock()
        
        # Configuration
        self.enable_explanations = enable_explanations
        self.max_history_size = max_history_size
        self.confidence_threshold = confidence_threshold
        self.enable_debug = enable_debug
        
        # Core data structures
        self.facts: Set[Fact] = set()
        self.rules: List[Rule] = []
        
        # Indexes for efficiency
        self._fact_by_predicate: Dict[str, Set[Fact]] = {}      # predicate -> facts
        self._rule_by_consequent: Dict[str, List[Rule]] = {}    # predicate -> rules
        self._rule_by_antecedent: Dict[str, List[Rule]] = {}    # predicate -> rules that need it
        self._antecedent_predicates: Set[str] = set()           # NEW: Cache of all antecedent predicates
        
        # State tracking
        self.inference_history: List[Dict] = []
        self.contradictions: List[Tuple[Fact, Fact]] = []
        
        # Agendas for forward chaining
        self._agenda: deque = deque()  # Facts ready for propagation
        self._processed_facts: Set[str] = set()
        
        # Rule management
        self._disabled_rules: Set[str] = set()  # NEW: Disabled rule names
        
        logger.info("SymbolicReasoner initialized")
    
    # ========== Fact Management ==========
    
    def add_fact(self, term: Term, confidence: float = 1.0, 
                 source: str = 'explicit', derivation_rule: Optional[str] = None) -> bool:
        """
        Add a fact to the knowledge base.
        
        Args:
            term: The term to add
            confidence: Confidence value (0.0-1.0)
            source: Source of the fact
            derivation_rule: Rule that derived this fact
        
        Returns:
            True if added successfully, False if contradiction
        """
        if not isinstance(term, Term):
            raise TypeError("term must be a Term instance")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        
        with self._lock:
            fact = Fact(term=term, confidence=confidence, 
                       source=source, derivation_rule=derivation_rule)
            
            # Check for contradictions
            if not self._check_fact_consistency(fact):
                logger.warning(f"Contradiction detected: {term}")
                return False
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                logger.debug(f"Fact below confidence threshold: {term} ({confidence:.2f})")
                return False
            
            # Add if new
            if fact not in self.facts:
                self.facts.add(fact)
                self._index_fact(fact)
                
                # Add to agenda for forward chaining
                if source != 'explicit' or confidence < 1.0:
                    self._agenda.append(fact)
                
                logger.debug(f"Added fact: {term} (conf: {confidence:.2f})")
                return True
            
            # Update existing fact with higher confidence (atomic operation)
            existing = next((f for f in self.facts if f == fact), None)
            if existing and confidence > existing.confidence:
                existing.confidence = max(existing.confidence, confidence)
                existing.source = source
                logger.debug(f"Updated fact confidence: {term} -> {existing.confidence:.2f}")
            
            return True
    
    def _index_fact(self, fact: Fact):
        """Add fact to indexes."""
        predicate = fact.term.predicate
        if predicate not in self._fact_by_predicate:
            self._fact_by_predicate[predicate] = set()
        self._fact_by_predicate[predicate].add(fact)
    
    def _remove_fact_from_index(self, fact: Fact):
        """Remove fact from indexes."""
        predicate = fact.term.predicate
        if predicate in self._fact_by_predicate:
            self._fact_by_predicate[predicate].discard(fact)
    
    def _check_fact_consistency(self, new_fact: Fact) -> bool:
        """Check if new fact is consistent with existing facts."""
        for fact in self.facts:
            if self._are_contradictory(new_fact.term, fact.term):
                self.contradictions.append((new_fact, fact))
                return False
        return True
    
    def _are_contradictory(self, term1: Term, term2: Term) -> bool:
        """Check if two terms are contradictory."""
        # Direct equality
        if term1 == term2:
            return False
        
        # Check for explicit negation (is_not vs is)
        if term1.predicate == "is_not" and term2.predicate == "is":
            return term1.args == term2.args
        if term2.predicate == "is_not" and term1.predicate == "is":
            return term1.args == term2.args
        
        # Check for opposite predicates
        opposites = {("alive", "dead"), ("open", "closed"), ("true", "false")}
        if (term1.predicate, term2.predicate) in opposites:
            return term1.args == term2.args
        
        return False
    
    # ========== Rule Management ==========
    
    def add_rule(self, antecedents: List[Term], consequent: Term,
                confidence: float = 1.0, priority: int = 0,
                name: Optional[str] = None, explanation: Optional[str] = None,
                is_defeasible: bool = False) -> Rule:
        """
        Add a logical rule with validation.
        
        Args:
            antecedents: List of condition terms
            consequent: Conclusion term
            confidence: Rule confidence (0.0-1.0)
            priority: Rule priority (higher fires first)
            name: Optional rule name
            explanation: Optional explanation
            is_defeasible: Whether rule can be overridden
        
        Returns:
            The created Rule object
        
        Raises:
            ValueError: If antecedents is empty
            TypeError: If terms are invalid
        """
        with self._lock:
            rule = Rule(
                antecedents=antecedents,
                consequent=consequent,
                confidence=confidence,
                priority=priority,
                name=name,
                explanation=explanation,
                is_defeasible=is_defeasible
            )
            
            self.rules.append(rule)
            
            # Index by consequent
            pred_key = consequent.predicate
            if pred_key not in self._rule_by_consequent:
                self._rule_by_consequent[pred_key] = []
            self._rule_by_consequent[pred_key].append(rule)
            
            # Index by antecedents and cache predicates
            for ant in antecedents:
                ant_key = ant.predicate
                self._antecedent_predicates.add(ant_key)  # NEW: Cache for quick lookup
                if ant_key not in self._rule_by_antecedent:
                    self._rule_by_antecedent[ant_key] = []
                self._rule_by_antecedent[ant_key].append(rule)
            
            # Sort rules by priority
            self._rule_by_consequent[pred_key].sort(key=lambda r: -r.priority)
            
            logger.debug(f"Added rule: {rule.name}")
            return rule
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable a disabled rule."""
        with self._lock:
            if rule_name in self._disabled_rules:
                self._disabled_rules.discard(rule_name)
                logger.info(f"Enabled rule: {rule_name}")
                return True
            return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable a rule from firing."""
        with self._lock:
            for rule in self.rules:
                if rule.name == rule_name:
                    self._disabled_rules.add(rule_name)
                    logger.info(f"Disabled rule: {rule_name}")
                    return True
            return False
    
    # ========== Forward Chaining ==========
    
    def forward_chain(self, max_iterations: int = 1000) -> List[Fact]:
        """
        Forward chaining with agenda-based propagation and conflict resolution.
        
        Args:
            max_iterations: Maximum inference iterations
        
        Returns:
            List of newly derived facts
        """
        with self._lock:
            derived_facts = []
            iterations = 0
            
            while self._agenda and iterations < max_iterations:
                iterations += 1
                fact = self._agenda.popleft()
                
                # Skip if already processed
                fact_key = fact.key
                if fact_key in self._processed_facts:
                    continue
                self._processed_facts.add(fact_key)
                
                if self.enable_debug:
                    logger.debug(f"[FC] Processing fact: {fact.term}")
                
                # Find rules that can fire
                rules_to_fire = self._get_rules_matching_fact(fact)
                
                for rule in rules_to_fire:
                    # Skip disabled rules
                    if rule.name in self._disabled_rules:
                        continue
                    
                    # Try to unify all antecedents
                    bindings, all_satisfied = self._try_unify_rule(rule, fact)
                    
                    if all_satisfied and bindings is not None:
                        # Apply bindings to consequent
                        bound_consequent = Unifier.apply_bindings(rule.consequent, bindings)
                        
                        # Calculate derived confidence
                        derived_confidence = self._compute_derived_confidence(rule, bindings)
                        
                        # Add derived fact
                        new_fact = Fact(
                            term=bound_consequent,
                            confidence=derived_confidence,
                            source=f"forward_chain({rule.name})",
                            derivation_rule=rule.name
                        )
                        
                        if new_fact not in self.facts:
                            self.facts.add(new_fact)
                            self._index_fact(new_fact)
                            derived_facts.append(new_fact)
                            self._agenda.append(new_fact)
                            
                            if self.enable_explanations:
                                self._add_to_history({
                                    'rule': rule.name,
                                    'trigger_fact': fact.term,
                                    'bindings': {str(k): str(v) for k, v in bindings.items()},
                                    'consequent': bound_consequent,
                                    'confidence': derived_confidence
                                })
                            
                            if self.enable_debug:
                                logger.debug(f"[FC] Derived: {bound_consequent} from {rule.name}")
            
            logger.info(f"Forward chaining done: {len(derived_facts)} facts in {iterations} iterations")
            return derived_facts
    
    def _get_rules_matching_fact(self, fact: Fact) -> List[Rule]:
        """Get rules that can be triggered by this fact (optimized)."""
        matching_rules = set()
        predicate = fact.term.predicate
        
        # Rules that have this predicate as an antecedent
        if predicate in self._rule_by_antecedent:
            matching_rules.update(self._rule_by_antecedent[predicate])
        
        # Rules that have this predicate as consequent (backward)
        if predicate in self._rule_by_consequent:
            matching_rules.update(self._rule_by_consequent[predicate])
        
        return sorted(list(matching_rules), key=lambda r: -r.priority)
    
    def _try_unify_rule(self, rule: Rule, trigger_fact: Fact) -> Tuple[Optional[Dict], bool]:
        """
        Try to unify rule antecedents with known facts.
        
        Returns:
            Tuple of (bindings, all_antecedents_satisfied)
        """
        # For each antecedent that matches the trigger predicate
        for ant_idx, antecedent in enumerate(rule.antecedents):
            if antecedent.predicate == trigger_fact.term.predicate:
                # Try to unify with trigger fact
                bindings = Unifier.unify(antecedent, trigger_fact.term)
                if bindings is not None:
                    # Check remaining antecedents
                    all_bound, final_bindings = self._check_all_antecedents(
                        rule, bindings, trigger_fact, ant_idx
                    )
                    if all_bound:
                        return final_bindings, True
        
        return None, False
    
    def _check_all_antecedents(self, rule: Rule, initial_bindings: Dict,
                               trigger_fact: Fact, skip_idx: int) -> Tuple[bool, Optional[Dict]]:
        """Check if all rule antecedents can be satisfied."""
        bindings = initial_bindings.copy()
        
        for i, antecedent in enumerate(rule.antecedents):
            if i == skip_idx:
                continue
            
            # Find matching fact
            found_match = False
            if antecedent.predicate in self._fact_by_predicate:
                for fact in self._fact_by_predicate[antecedent.predicate]:
                    if fact.confidence < self.confidence_threshold:
                        continue
                    
                    new_bindings = Unifier.unify(antecedent, fact.term, bindings)
                    if new_bindings is not None:
                        bindings = new_bindings
                        found_match = True
                        break
            
            if not found_match:
                return False, None
        
        return True, bindings
    
    def _compute_derived_confidence(self, rule: Rule, bindings: Dict) -> float:
        """Compute confidence for derived fact using fuzzy logic."""
        # Base confidence from rule
        confidence = rule.confidence
        
        # Find the confidence of all matched facts
        fact_confidences = []
        for antecedent in rule.antecedents:
            bound_ant = Unifier.apply_bindings(antecedent, bindings)
            if bound_ant.predicate in self._fact_by_predicate:
                for fact in self._fact_by_predicate[bound_ant.predicate]:
                    if Unifier.unify(antecedent, fact.term, bindings) is not None:
                        fact_confidences.append(fact.confidence)
                        break  # Only use first match per antecedent
        
        # Combine confidences (using minimum for conservative fuzzy AND)
        if fact_confidences:
            confidence *= min(fact_confidences)
        
        return min(1.0, max(0.0, confidence))
    
    # ========== Backward Chaining ==========
    
    def backward_chain(self, goal: Term, max_depth: int = 10,
                      bindings: Dict = None, depth: int = 0,
                      return_all_proofs: bool = False) -> Union[Tuple[bool, Optional[Dict], List], List]:
        """
        Backward chaining to prove a goal with variable binding.
        
        Args:
            goal: The goal term to prove
            max_depth: Maximum recursion depth
            bindings: Existing variable bindings
            depth: Current recursion depth
            return_all_proofs: Return all possible proofs
        
        Returns:
            If return_all_proofs: List of (success, bindings, proof) tuples
            Else: Tuple of (success, final_bindings, proof_tree)
        """
        if bindings is None:
            bindings = {}
        
        bound_goal = Unifier.apply_bindings(goal, bindings)
        
        # Check if already known
        if self._fact_exists(bound_goal):
            result = (True, bindings, [{'type': 'fact', 'term': bound_goal}])
            if return_all_proofs:
                return [result]
            return result
        
        if depth >= max_depth:
            result = (False, None, [])
            if return_all_proofs:
                return []
            return result
        
        all_proofs = []
        
        # Find rules that can prove this goal
        if bound_goal.predicate in self._rule_by_consequent:
            for rule in self._rule_by_consequent[bound_goal.predicate]:
                # Skip disabled rules
                if rule.name in self._disabled_rules:
                    continue
                
                # Try to unify rule consequent with goal
                unification = Unifier.unify(rule.consequent, bound_goal, bindings)
                if unification is not None:
                    # Prove all antecedents
                    all_proved = True
                    proofs = []
                    current_bindings = unification
                    
                    for antecedent in rule.antecedents:
                        success, new_bindings, sub_proof = self.backward_chain(
                            antecedent, max_depth, current_bindings, depth + 1,
                            return_all_proofs=False
                        )
                        
                        if success:
                            proofs.append({'antecedent': antecedent, 'proof': sub_proof})
                            current_bindings = new_bindings
                        else:
                            all_proved = False
                            break
                    
                    if all_proved:
                        proof_tree = [{
                            'type': 'rule',
                            'rule': rule.name,
                            'goal': bound_goal,
                            'proofs': proofs
                        }]
                        result = (True, current_bindings, proof_tree)
                        all_proofs.append(result)
                        if not return_all_proofs:
                            return result
        
        if return_all_proofs:
            return all_proofs
        return (False, None, [])
    
    def _fact_exists(self, term: Term) -> bool:
        """Check if a fact exists (with unification and confidence threshold)."""
        if term.predicate not in self._fact_by_predicate:
            return False
        
        for fact in self._fact_by_predicate[term.predicate]:
            if fact.confidence >= self.confidence_threshold:
                if Unifier.unify(term, fact.term) is not None:
                    return True
        
        return False
    
    # ========== Query and Explanation ==========
    
    def query(self, goal: Term) -> Tuple[TruthValue, float, Optional[List]]:
        """
        Query the knowledge base for a goal.
        
        Args:
            goal: The goal term to query
        
        Returns:
            Tuple of (truth_value, confidence, explanation)
        """
        # Check if directly known
        if self._fact_exists(goal):
            # Find best matching fact
            facts = self._fact_by_predicate.get(goal.predicate, [])
            best_confidence = 0.0
            best_fact = None
            
            for fact in facts:
                if fact.confidence >= self.confidence_threshold:
                    if Unifier.unify(goal, fact.term) is not None:
                        if fact.confidence > best_confidence:
                            best_confidence = fact.confidence
                            best_fact = fact
            
            if best_fact:
                return TruthValue.TRUE, best_confidence, [{'type': 'fact', 'fact': best_fact}]
        
        # Try backward chaining
        success, bindings, proof = self.backward_chain(goal)
        
        if success and bindings:
            confidence = self._compute_bc_confidence(proof)
            return TruthValue.TRUE, confidence, proof
        
        # Check for explicit false
        if self._fact_exists(Term("is_not", goal.args)):
            return TruthValue.FALSE, 0.0, None
        
        return TruthValue.UNKNOWN, 0.0, None
    
    def _compute_bc_confidence(self, proof: List) -> float:
        """Compute confidence from backward chaining proof."""
        if not proof:
            return 0.0
        
        confidences = []
        
        def extract_confidence(node):
            if isinstance(node, dict):
                if node.get('type') == 'fact':
                    if 'fact' in node:
                        confidences.append(node['fact'].confidence)
                elif node.get('type') == 'rule':
                    # Find matching rule
                    for rule in self.rules:
                        if rule.name == node.get('rule'):
                            confidences.append(rule.confidence)
                            break
                for value in node.values():
                    extract_confidence(value)
            elif isinstance(node, list):
                for item in node:
                    extract_confidence(item)
        
        extract_confidence(proof)
        return min(confidences) if confidences else 1.0
    
    def explain(self, fact: Term) -> Optional[Dict]:
        """Generate detailed explanation for a fact."""
        # Find the fact
        matching_fact = None
        if fact.predicate in self._fact_by_predicate:
            for f in self._fact_by_predicate[fact.predicate]:
                if Unifier.unify(fact, f.term) is not None:
                    matching_fact = f
                    break
        
        if not matching_fact:
            return None
        
        explanation = {
            'fact': str(matching_fact.term),
            'confidence': matching_fact.confidence,
            'source': matching_fact.source,
            'timestamp': matching_fact.timestamp
        }
        
        # Find derivation in history
        if matching_fact.derivation_rule:
            for inference in reversed(self.inference_history):
                if inference.get('rule') == matching_fact.derivation_rule:
                    explanation['derivation'] = inference
                    break
        
        return explanation
    
    # ========== Utility Methods ==========
    
    def _add_to_history(self, entry: Dict) -> None:
        """Add entry to inference history with memory management."""
        self.inference_history.append(entry)
        # Maintain max size with FIFO eviction
        if len(self.inference_history) > self.max_history_size:
            self.inference_history = self.inference_history[-self.max_history_size:]
    
    def get_stats(self) -> Dict:
        """Get reasoner statistics."""
        with self._lock:
            return {
                'facts': len(self.facts),
                'rules': len(self.rules),
                'enabled_rules': len(self.rules) - len(self._disabled_rules),
                'inferences': len(self.inference_history),
                'contradictions': len(self.contradictions),
                'agenda_size': len(self._agenda),
                'fact_index_size': sum(len(s) for s in self._fact_by_predicate.values()),
                'rule_index_size': sum(len(l) for l in self._rule_by_consequent.values()),
                'memory_usage_bytes': self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        import sys
        size = 0
        size += sys.getsizeof(self.facts)
        size += sys.getsizeof(self.rules)
        size += sys.getsizeof(self.inference_history)
        for fact in self.facts:
            size += sys.getsizeof(fact)
        for rule in self.rules:
            size += sys.getsizeof(rule)
        return size
    
    def clear(self):
        """Clear knowledge base and reset state."""
        with self._lock:
            self.facts.clear()
            self.rules.clear()
            self._fact_by_predicate.clear()
            self._rule_by_consequent.clear()
            self._rule_by_antecedent.clear()
            self._antecedent_predicates.clear()
            self.inference_history.clear()
            self.contradictions.clear()
            self._agenda.clear()
            self._processed_facts.clear()
            self._disabled_rules.clear()
            logger.info("Reasoner cleared")
    
    def serialize_facts(self) -> List[Dict]:
        """Serialize facts for persistence."""
        with self._lock:
            return [
                {
                    'term': str(f.term),
                    'confidence': f.confidence,
                    'source': f.source,
                    'timestamp': f.timestamp,
                    'derivation_rule': f.derivation_rule
                }
                for f in self.facts
            ]
    
    def serialize_rules(self) -> List[Dict]:
        """Serialize rules for persistence."""
        with self._lock:
            return [
                {
                    'name': r.name,
                    'confidence': r.confidence,
                    'priority': r.priority,
                    'enabled': r.name not in self._disabled_rules,
                    'is_defeasible': r.is_defeasible
                }
                for r in self.rules
            ]


# ========== Helper Functions ==========

def fact(predicate: str, *args) -> Term:
    """Helper to create facts/terms."""
    return Term(predicate, list(args))


def var(name: str) -> Variable:
    """Helper to create variables."""
    return Variable(name)


# ========== Example Usage ==========

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    reasoner = SymbolicReasoner(enable_explanations=True, enable_debug=True)
    
    # Add facts
    reasoner.add_fact(fact("is", "Socrates", "human"), confidence=1.0)
    
    # Add rule: All humans are mortal
    reasoner.add_rule(
        antecedents=[fact("is", var("X"), "human")],
        consequent=fact("is", var("X"), "mortal"),
        confidence=0.95,
        name="human_mortal"
    )
    
    # Forward chain
    derived = reasoner.forward_chain()
    
    # Query
    truth, conf, proof = reasoner.query(fact("is", "Socrates", "mortal"))
    print(f"Socrates mortal? {truth.value} (conf: {conf:.2f})")
    
    # Explanation
    explanation = reasoner.explain(fact("is", "Socrates", "mortal"))
    print(f"Explanation: {explanation}")
    
    # Stats
    print(f"Stats: {reasoner.get_stats()}")
