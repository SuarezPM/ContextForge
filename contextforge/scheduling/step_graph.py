"""AgentStepGraph — workflow dependency graph for KV cache eviction priority.

Based on KVFlow (NeurIPS 2025, arXiv:2507.07400):
- Workflow-aware eviction: evict caches of agents with high steps-to-execution
  (agents far from being invoked) before agents about to run.
- Overlapped KV prefetching: proactively prefetch KV tensors for agents
  scheduled in the next N steps.

Result from paper: 1.83x speedup over SGLang, 2.19x for concurrent workflows.

V4.0 CHANGES: New module for workflow-aware eviction.
"""
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentStep:
    """A single step in a workflow graph."""
    agent_id: str
    depends_on: list[str] = field(default_factory=list)
    step_index: int = 0
    estimated_tokens: int = 0
    is_optional: bool = False  # True for dynamic conditional agents


class AgentStepGraph:
    """
    Workflow dependency graph for KV cache eviction priority.
    
    Usage:
        graph = AgentStepGraph()
        graph.add_step(AgentStep(agent_id="retriever", depends_on=[], step_index=0))
        graph.add_step(AgentStep(agent_id="summarizer", depends_on=["retriever"], step_index=1))
        order = graph.get_eviction_priority_order()  # agents far from execution first
    """
    
    def __init__(self):
        self._steps: dict[str, AgentStep] = {}
        self._step_list: list[AgentStep] = []  # topological order
    
    def add_step(self, step: AgentStep) -> "AgentStepGraph":
        """Add a step to the graph. Returns self for chaining."""
        self._steps[step.agent_id] = step
        self._step_list.append(step)
        return self
    
    def compute_steps_to_execution(self, agent_id: str, current_step: int = 0) -> int:
        """
        Returns how many steps must complete before agent_id is invoked.
        
        Returns:
            0 if agent is the current step.
            sys.maxsize if agent_id not in graph.
            Raises ValueError if graph has cycles.
        """
        self.validate_dag()  # Will raise if cycles
        
        if agent_id not in self._steps:
            return sys.maxsize
        
        step = self._steps[agent_id]
        
        # Compute longest path from any root to this step
        if step.step_index <= current_step:
            return 0
        
        # BFS/DFS to compute depth
        visited = set()
        
        def compute_depth(s: AgentStep, visited: set) -> int:
            if s.agent_id in visited:
                return 0
            visited.add(s.agent_id)
            
            if not s.depends_on:
                return s.step_index
            
            max_parent_depth = 0
            for dep_id in s.depends_on:
                if dep_id in self._steps:
                    max_parent_depth = max(max_parent_depth, compute_depth(self._steps[dep_id], visited))
            
            return max_parent_depth + 1
        
        return compute_depth(step, set())
    
    def get_prefetch_candidates(
        self,
        current_step: int,
        lookahead: int = 2,
    ) -> list[str]:
        """Return agent_ids to prefetch within `lookahead` steps."""
        candidates = []
        for step in self._step_list:
            if step.step_index <= current_step:
                continue
            if step.step_index <= current_step + lookahead:
                candidates.append(step.agent_id)
        return candidates
    
    def get_eviction_priority_order(self) -> list[str]:
        """
        Return agent_ids ordered from lowest to highest eviction priority
        (first in list = evict first = highest steps_to_execution).
        """
        # Sort by steps_to_execution descending (agents far from execution evict first)
        priorities = []
        for step in self._step_list:
            steps = self.compute_steps_to_execution(step.agent_id, current_step=0)
            priorities.append((step.agent_id, steps))
        
        # Sort descending by steps (highest first = evict first)
        priorities.sort(key=lambda x: x[1], reverse=True)
        return [agent_id for agent_id, _ in priorities]
    
    def validate_dag(self) -> None:
        """Raise ValueError if graph contains cycles."""
        # DFS-based cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {sid: WHITE for sid in self._steps}
        
        def dfs(node_id: str) -> None:
            color[node_id] = GRAY
            if node_id in self._steps:
                for dep in self._steps[node_id].depends_on:
                    if dep not in color:
                        color[dep] = WHITE
                    if color.get(dep, WHITE) == GRAY:
                        raise ValueError(f"Cycle detected involving agent '{node_id}'")
                    if color.get(dep, WHITE) == WHITE:
                        dfs(dep)
            color[node_id] = BLACK
        
        for sid in self._steps:
            if color[sid] == WHITE:
                dfs(sid)
    
    @property
    def size(self) -> int:
        """Number of steps in the graph."""
        return len(self._steps)
    
    def get_step(self, agent_id: str) -> Optional[AgentStep]:
        """Get step by agent_id."""
        return self._steps.get(agent_id)
    
    def get_all_agents(self) -> list[str]:
        """Get all agent IDs in the graph."""
        return list(self._steps.keys())