"""
Symbolic Engine - Neural-Symbolic Verification System
Implements logical consistency checks and selective deep traversal.
Prevents hallucinations through fact verification against knowledge graph.
"""
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class VerificationStatus(Enum):
    """Status of fact verification."""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"

class SymbolicEngine:
    def __init__(self, knowledge_graph):
        """
        Initialize Symbolic Engine.
        
        Args:
            knowledge_graph: Neo4jKnowledgeGraph instance
        """
        self.kg = knowledge_graph
        self.verification_cache = {}
        self.contradiction_rules = self._load_contradiction_rules()
        self.consistency_cache = {}
        self.reasoning_depth = 0
        self.max_reasoning_depth = 5
    
    def _load_contradiction_rules(self) -> List[Dict]:
        """
        Load rules for detecting contradictions.
        This is a foundational set; can be expanded dynamically.
        """
        return [
            {
                "name": "inverse_properties",
                "rule": "If A--INHIBITS-->B, then B--NOT_REQUIRED_BY-->A",
                "severity": "high"
            },
            {
                "name": "transitive_closure",
                "rule": "If A--ENABLES-->B and B--ENABLES-->C, then A--ENABLES-->C",
                "severity": "medium"
            },
            {
                "name": "temporal_consistency",
                "rule": "If event A happens before B, then ~(B happens before A)",
                "severity": "high"
            },
            {
                "name": "mutual_exclusivity",
                "rule": "If state A is exclusive, then ~(state B) when A is true",
                "severity": "high"
            }
        ]
    
    def verify_neural_proposal(
        self,
        proposed_fact: Dict,
        query_context: str = "",
        minimum_confidence: float = 0.6
    ) -> Tuple[VerificationStatus, float, Dict]:
        """
        Verify a neural network proposal against knowledge graph.
        This is the critical "Double-Lock" step.
        
        Args:
            proposed_fact: Dict with keys 'source', 'relation', 'target'
            query_context: Context of the user query
            minimum_confidence: Confidence threshold for acceptance
        
        Returns:
            Tuple of (status, confidence_score, details)
        """
        source = proposed_fact.get("source", "").lower()
        relation = proposed_fact.get("relation", "").upper()
        target = proposed_fact.get("target", "").lower()
        
        # Create caching key
        cache_key = f"{source}|{relation}|{target}"
        
        if cache_key in self.verification_cache:
            return self.verification_cache[cache_key]
        
        # Step 2: Symbolic Verification - Check against knowledge graph
        exists, kg_confidence = self.kg.validate_fact(source, relation, target)
        
        details = {
            "source": source,
            "relation": relation,
            "target": target,
            "exists_in_kg": exists,
            "kg_confidence": kg_confidence,
            "checked_for_contradictions": False,
            "violations": []
        }
        
        if exists:
            # Fact is in knowledge graph
            status = VerificationStatus.VERIFIED
            confidence = kg_confidence
        else:
            # Check for contradictions
            contradictions = self._check_contradictions(proposed_fact)
            details["checked_for_contradictions"] = True
            details["violations"] = contradictions
            
            if contradictions:
                status = VerificationStatus.CONTRADICTED
                confidence = 0.0
            else:
                status = VerificationStatus.UNCERTAIN
                confidence = 0.3  # Unknown facts get low confidence
        
        # Apply minimum confidence threshold
        if confidence < minimum_confidence and status != VerificationStatus.VERIFIED:
            status = VerificationStatus.UNVERIFIED
        
        result = (status, confidence, details)
        self.verification_cache[cache_key] = result
        
        return result
    
    def _check_contradictions(self, fact: Dict) -> List[Dict]:
        """
        Check if a fact contradicts existing knowledge.
        
        Args:
            fact: Fact to check
        
        Returns:
            List of contradiction objects
        """
        violations = []
        source = fact.get("source", "").lower()
        relation = fact.get("relation", "").upper()
        target = fact.get("target", "").lower()
        
        # Rule 1: Check inverse properties
        inverse_relation = self._get_inverse_relation(relation)
        if inverse_relation:
            exists, confidence = self.kg.validate_fact(target, inverse_relation, source)
            if exists and confidence > 0.7:
                violations.append({
                    "rule": "inverse_properties",
                    "description": f"Contradicts {target}--{inverse_relation}--{source}",
                    "confidence": confidence,
                    "severity": "high"
                })
        
        # Rule 2: Check transitive violations
        transitive_violations = self._check_transitive_consistency(source, relation, target)
        violations.extend(transitive_violations)
        
        return violations
    
    def _get_inverse_relation(self, relation: str) -> Optional[str]:
        """Get inverse of a relation if it exists."""
        inverse_map = {
            "ENABLES": "BLOCKED_BY",
            "REQUIRES": "NOT_REQUIRED_BY",
            "INHIBITS": "ACTIVATED_BY",
            "CAUSES": "PREVENTED_BY"
        }
        return inverse_map.get(relation)
    
    def _check_transitive_consistency(
        self, 
        source: str, 
        relation: str, 
        target: str
    ) -> List[Dict]:
        """
        Check transitive property consistency.
        
        E.g., if A-->ENABLES-->B and B-->REQUIRES-->C, 
        can we derive A-->?-->C?
        """
        violations = []
        
        try:
            # Find intermediate nodes that source connects to
            source_path = self.kg.find_reasoning_path(source, target, max_depth=3)
            
            if source_path and len(source_path) > 2:
                # Path exists; check consistency of intermediate steps
                for i in range(len(source_path) - 1):
                    intermediate_source = source_path[i]
                    intermediate_target = source_path[i + 1]
                    
                    # For now, just log; can be expanded with detailed checks
                    logger.debug(f"Transitive path segment: {intermediate_source}--{relation}--{intermediate_target}")
        except Exception as e:
            logger.debug(f"Transitive check error: {e}")
        
        return violations
    
    def validate_reasoning_path(
        self,
        path: List[str],
        path_relations: List[str]
    ) -> Tuple[bool, float, List[Dict]]:
        """
        Validate entire reasoning path for consistency.
        
        Args:
            path: List of entity names
            path_relations: List of relation types between entities
        
        Returns:
            Tuple of (is_valid, confidence_score, issues)
        """
        issues = []
        total_confidence = 0.0
        
        if len(path) < 2:
            return False, 0.0, [{"msg": "Path too short"}]
        
        # Check each edge in the path
        for i, (source, target) in enumerate(zip(path[:-1], path[1:])):
            relation = path_relations[i] if i < len(path_relations) else "RELATES"
            
            exists, confidence = self.kg.validate_fact(source, relation, target)
            total_confidence += confidence
            
            if not exists:
                issues.append({
                    "edge": f"{source}--{relation}--{target}",
                    "exists": False,
                    "confidence": confidence
                })
        
        avg_confidence = total_confidence / (len(path) - 1) if len(path) > 1 else 0.0
        is_valid = len(issues) == 0 and avg_confidence > 0.5
        
        return is_valid, avg_confidence, issues
    
    def selective_deep_traversal(
        self,
        start_entity: str,
        query: str,
        depth_limit: int = 3,
        expansion_threshold: float = 0.6
    ) -> Dict:
        """
        Selectively traverse graph based on query relevance.
        Only expands paths where confidence/relevance exceeds threshold.
        
        Args:
            start_entity: Starting entity
            query: User query for relevance evaluation
            depth_limit: Maximum traversal depth
            expansion_threshold: Confidence threshold for path expansion
        
        Returns:
            Dict with traversal results
        """
        self.reasoning_depth = 0
        explored_nodes = set()
        explored_edges = set()
        
        def traverse(node: str, depth: int, parent_confidence: float):
            if depth > depth_limit or self.reasoning_depth > self.max_reasoning_depth:
                return
            if node in explored_nodes:
                return
            
            self.reasoning_depth += 1
            explored_nodes.add(node)
            
            # Get neighbors
            neighbors_data = self.kg.get_entity_neighbors(node, depth=1)
            
            if not neighbors_data:
                return
            
            neighbors = neighbors_data.get("neighbors", [])
            for neighbor in neighbors:
                neighbor_name = neighbor.get("name", "")
                if neighbor_name in explored_nodes:
                    continue
                
                # Check expansion threshold
                if parent_confidence >= expansion_threshold:
                    edge_key = f"{node}|{neighbor_name}"
                    if edge_key not in explored_edges:
                        explored_edges.add(edge_key)
                        traverse(neighbor_name, depth + 1, parent_confidence)
        
        traverse(start_entity, 0, 1.0)
        
        return {
            "explored_nodes": list(explored_nodes),
            "explored_edges": list(explored_edges),
            "reasoning_depth": self.reasoning_depth,
            "traversal_config": {
                "depth_limit": depth_limit,
                "expansion_threshold": expansion_threshold
            }
        }
    
    def get_evidence_chain(
        self,
        source: str,
        target: str,
        min_confidence: float = 0.5
    ) -> Optional[List[Dict]]:
        """
        Retrieve the evidence chain (supporting papers/sources) for a reasoning path.
        
        Args:
            source: Source entity
            target: Target entity
            min_confidence: Minimum confidence for edges in chain
        
        Returns:
            List of evidence steps, or None if no path
        """
        path = self.kg.find_reasoning_path(source, target)
        
        if not path:
            return None
        
        evidence_chain = []
        for i in range(len(path) - 1):
            edge_exists, confidence = self.kg.validate_fact(path[i], "RELATES", path[i + 1])
            
            if confidence >= min_confidence:
                evidence_chain.append({
                    "from": path[i],
                    "to": path[i + 1],
                    "confidence": confidence,
                    "step": i + 1
                })
        
        return evidence_chain if evidence_chain else None
    
    def consistency_report(self, fact_list: List[Dict]) -> Dict:
        """
        Generate comprehensive consistency report for a set of facts.
        
        Args:
            fact_list: List of facts to check
        
        Returns:
            Consistency report
        """
        report = {
            "total_facts": len(fact_list),
            "verified": 0,
            "unverified": 0,
            "contradicted": 0,
            "uncertain": 0,
            "avg_confidence": 0.0,
            "details": []
        }
        
        total_confidence = 0.0
        
        for fact in fact_list:
            status, confidence, details = self.verify_neural_proposal(fact)
            
            if status == VerificationStatus.VERIFIED:
                report["verified"] += 1
            elif status == VerificationStatus.UNVERIFIED:
                report["unverified"] += 1
            elif status == VerificationStatus.CONTRADICTED:
                report["contradicted"] += 1
            else:
                report["uncertain"] += 1
            
            total_confidence += confidence
            report["details"].append({
                "fact": fact,
                "status": status.value,
                "confidence": confidence
            })
        
        report["avg_confidence"] = total_confidence / len(fact_list) if fact_list else 0.0
        
        return report
