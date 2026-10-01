from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from src.domain.models.planning.cognition import (
    BlockedResolution,
    DialogueResolution,
    GoalDecision,
    GoalEvaluation,
)
from abc import ABC, abstractmethod
from src.domain.models.planning.planning import AgentCognitiveContext, PlanDecomposition

class ICognitionProvider(ABC):
    @abstractmethod
    async def decide_next_goal(self, context: dict[str, Any]) -> GoalDecision:
        """Ermittelt ein neues Ziel, wenn der Agent im Leerlauf ist."""
        pass

    @abstractmethod
    async def resolve_blockage(self, context: dict[str, Any]) -> BlockedResolution:
        """Trifft eine Handlungsentscheidung bei einer räumlichen Blockade."""
        pass

    @abstractmethod
    async def evaluate_goal_status(self, context: dict[str, Any]) -> GoalEvaluation:
        """Evaluiert kognitiv, ob ein aktives Teilziel abgeschlossen ist."""
        pass

    @abstractmethod
    async def respond_to_dialogue(self, context: dict[str, Any]) -> DialogueResolution:
        """Erzeugt eine Antwort oder beendet den Dialog auf Basis empfangener Nachrichten."""
        pass

    async def decompose_plan(
            self, context: AgentCognitiveContext
    ) -> PlanDecomposition:
        """Zerlegt den Kognitionskontext eines Agenten in ein Primärziel und Teilziele."""
        pass

