"""
Cognitive Execution Bridge - Connects Cognitive Orchestrator to Goal Execution

This bridges the cognitive layer (beliefs, identity, dissonance) with 
the existing AI-OS execution layer (goals, skills, planning).
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


class CognitiveExecutionBridge:
    """
    Bridges cognitive state with execution pipeline.
    
    Before: Goal → Decompose → Execute → Done
    After:  Cognitive State → Filter Goals → Execute → Update State
    """
    
    def __init__(self, cognitive_orchestrator):
        self._orchestrator = cognitive_orchestrator
        self._execution_history: List[Dict] = []
    
    def prepare_execution(
        self,
        incoming_goal: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process goal through cognitive pipeline before execution.
        
        Returns execution plan with cognitive metadata.
        """
        # Step 1: Process through cognitive orchestrator
        result = self._orchestrator.process_goal_request(
            {'need': incoming_goal},
            execution_context
        )
        
        if result.get('status') == 'deferred':
            return {
                'execute': False,
                'reason': result['reason'],
                'suggestion': result.get('suggestion'),
                'cognitive_metadata': None
            }
        
        # Step 2: Add cognitive metadata to execution plan
        return {
            'execute': True,
            'strategy': result['strategy'],
            'goals': result['goals'],
            'cognitive_metadata': {
                'identity_continuity': result['identity_state']['continuity_score'],
                'dissonance_pressure': result['dissonance']['current_pressure'],
                'rupture_risk': result['dissonance']['rupture_risk'],
                'strategy_name': result['strategy']
            }
        }
    
    def record_outcome(
        self,
        goal: Dict[str, Any],
        execution_result: Dict[str, Any]
    ):
        """
        Record execution outcome back to cognitive system.
        """
        # Determine outcome
        status = execution_result.get('status', 'unknown')
        
        if status == 'completed':
            outcome = 'success'
        elif status in ['failed', 'error']:
            outcome = 'failure'
        else:
            outcome = 'partial'
        
        # Extract artifacts
        artifacts = execution_result.get('artifacts', [])
        
        # Report to orchestrator
        self._orchestrator.report_execution_result(goal, outcome, artifacts)
        
        # Store in history
        self._execution_history.append({
            'goal_id': goal.get('id'),
            'outcome': outcome,
            'timestamp': execution_result.get('timestamp'),
            'cognitive_impact': self._calculate_cognitive_impact(outcome, artifacts)
        })
    
    def _calculate_cognitive_impact(
        self,
        outcome: str,
        artifacts: List[Dict]
    ) -> Dict[str, float]:
        """Calculate how this execution affected cognitive state"""
        
        impact = {
            'belief_formation': 0.0,
            'identity_strengthening': 0.0,
            'skill_improvement': 0.0,
            'dissonance_reduction': 0.0
        }
        
        if outcome == 'success':
            impact['identity_strengthening'] = 0.3
            impact['dissonance_reduction'] = 0.2
        
        # Analyze artifacts for belief formation
        for artifact in artifacts:
            if artifact.get('type') == 'KNOWLEDGE':
                impact['belief_formation'] += 0.1
        
        return impact
    
    def get_execution_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations based on cognitive state + execution history.
        """
        identity = self._orchestrator.get_identity_state()
        dissonance = self._orchestrator.get_dissonance_status()
        
        recommendations = {
            'risk_level': 'low',
            'recommended_strategy': 'balanced',
            'goals_to_avoid': [],
            'preferred_domains': []
        }
        
        # High dissonance = be more conservative
        if dissonance.get('current_pressure', 0) > 0.6:
            recommendations['risk_level'] = 'high'
            recommendations['recommended_strategy'] = 'cautious'
        
        # Low continuity = avoid ambitious goals
        if identity.get('continuity_score', 1.0) < 0.5:
            recommendations['risk_level'] = 'high'
            recommendations['goals_to_avoid'].append('exploratory')
            recommendations['goals_to_avoid'].append('meta')
        
        # High continuity = can attempt challenging goals
        if identity.get('continuity_score', 0) > 0.8:
            recommendations['preferred_domains'].extend(['learning', 'creation'])
        
        return recommendations


# Integration point with existing execution system

class ExecutionSystemIntegration:
    """
    Integration with existing AI-OS execution layer.
    
    This would wrap the existing goal execution flow.
    """
    
    def __init__(self, bridge: CognitiveExecutionBridge):
        self._bridge = bridge
    
    def execute_goal_with_cognition(
        self,
        goal: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a goal with cognitive oversight.
        """
        # 1. Prepare through cognitive pipeline
        prep_result = self._bridge.prepare_execution(goal, context)
        
        if not prep_result.get('execute'):
            return {
                'status': 'blocked',
                'reason': prep_result['reason'],
                'cognitive_metadata': None
            }
        
        # 2. Execute using existing AI-OS system
        # This would call the actual goal execution system
        execution_result = self._execute_with_traditional_system(
            prep_result['goals'][0],  # Use prioritized goal
            context
        )
        
        # 3. Record outcome back to cognitive system
        self._bridge.record_outcome(goal, execution_result)
        
        return {
            'status': execution_result.get('status', 'completed'),
            'cognitive_metadata': prep_result.get('cognitive_metadata'),
            'execution_result': execution_result
        }
    
    def _execute_with_traditional_system(
        self,
        goal: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Placeholder for actual execution system integration.
        
        This would call:
        - goal_decomposer
        - skill selection
        - agent execution
        - evaluation
        """
        # For now, return mock result
        return {
            'status': 'completed',
            'artifacts': [],
            'timestamp': '2026-01-01T00:00:00'
        }


# Usage example:
"""
from cognitive_orchestrator import get_cognitive_orchestrator

orchestrator = get_cognitive_orchestrator()
bridge = CognitiveExecutionBridge(orchestrator)
integration = ExecutionSystemIntegration(bridge)

# Process a goal
result = integration.execute_goal_with_cognition(
    goal={'title': 'Learn Python', 'goal_type': 'achievable'},
    context={'available_resources': 0.8}
)

print(f"Execution strategy: {result['cognitive_metadata']['strategy_name']}")
print(f"Identity continuity: {result['cognitive_metadata']['identity_continuity']}")
"""