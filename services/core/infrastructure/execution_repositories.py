"""
Execution Repositories - Full implementation after subrepo restructuring
"""
from typing import Optional, List, Any
from datetime import datetime
from uuid import UUID


class GoalExecutionRepository:
    """Repository for goal execution records"""
    
    def __init__(self):
        self._storage = []
    
    async def add(self, session, execution_record):
        """Add execution record"""
        self._storage.append(execution_record)
        return execution_record
    
    async def get_by_goal(self, session, goal_id: UUID) -> List[Any]:
        """Get executions by goal"""
        return [e for e in self._storage if str(e.goal_id) == str(goal_id)]
    
    async def get(self, session, execution_id: UUID) -> Optional[Any]:
        """Get by ID"""
        for e in self._storage:
            if str(e.execution_id) == str(execution_id):
                return e
        return None
    
    async def update(self, session, execution_record):
        """Update execution record"""
        for i, e in enumerate(self._storage):
            if str(e.execution_id) == str(execution_record.execution_id):
                self._storage[i] = execution_record
                return
        self._storage.append(execution_record)


class SkillStatsRepository:
    """Repository for skill performance statistics"""
    
    def __init__(self):
        self._stats = {}
    
    async def update_from_execution(self, session, skill_id: str, execution_record):
        """Update skill stats from execution"""
        if skill_id not in self._stats:
            self._stats[skill_id] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_latency_ms": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0
            }
        
        stats = self._stats[skill_id]
        stats["total_executions"] += 1
        stats["total_latency_ms"] += execution_record.duration_ms or 0
        
        if execution_record.success:
            stats["successful_executions"] += 1
        
        if stats["total_executions"] > 0:
            stats["success_rate"] = stats["successful_executions"] / stats["total_executions"]
        if stats["total_latency_ms"] > 0 and stats["total_executions"] > 0:
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_executions"]
    
    async def get_stats(self, session, skill_id: str) -> dict:
        """Get skill stats"""
        return self._stats.get(skill_id, {
            "total_executions": 0,
            "success_rate": 0.0,
            "avg_latency_ms": 0.0
        })
    
    async def get(self, session, skill_id: str) -> Optional[Any]:
        """Get skill stats (for compatibility)"""
        return await self.get_stats(session, skill_id)
    
    async def get_all(self, session) -> List[dict]:
        """Get all skill stats"""
        return [
            {"skill_id": k, **v} 
            for k, v in self._stats.items()
        ]