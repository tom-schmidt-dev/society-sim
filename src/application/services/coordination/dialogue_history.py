from __future__ import annotations

from typing import Any, Optional
from src.domain.models.communication.dialogue import DialogueRecord


class DialogueHistory:
    def __init__(self, max_capacity: Optional[int] = None) -> None:
        self._records: list[DialogueRecord] = []
        self._max_capacity: Optional[int] = max_capacity

    @property
    def records(self) -> list[DialogueRecord]:
        return list(self._records)

    def add_record(self, record: DialogueRecord) -> None:
        self._records.append(record)
        if self._max_capacity and len(self._records) > self._max_capacity:
            self._records.pop(0)

    def record_dialogue(
        self,
        tick: int,
        sender_id: str,
        sender_name: str,
        recipient_id: Optional[str],
        recipient_name: Optional[str],
        message: str,
        intent: Optional[str] = None,
        is_inspection: bool = False,
        is_empty_response: bool = False,
    ) -> DialogueRecord:
        record = DialogueRecord(
            tick=tick,
            sender_id=sender_id,
            sender_name=sender_name,
            recipient_id=recipient_id,
            recipient_name=recipient_name,
            message=message,
            intent=intent,
            is_inspection=is_inspection,
            is_empty_response=is_empty_response,
        )
        self.add_record(record)
        return record

    def get_recent_structured(self, limit: int = 8) -> list[dict[str, Any]]:
        recent = self.get_recent_records(limit=limit)
        return [record.to_dict() for record in recent]

    def get_recent_records(self, limit: int = 8) -> list[DialogueRecord]:
        return self._records[-limit:] if limit > 0 else []

    def get_recent_formatted(self, limit: int = 8) -> list[str]:
        recent = self.get_recent_records(limit=limit)
        formatted: list[str] = []
        for record in recent:
            line = record.format_for_display()
            # Abgewiesene Verabschiedungen für LLM-Kontext explizit kennzeichnen
            if record.intent == "reject":
                line = f"[ABGELEHNT] {line}"
            formatted.append(line)
        return formatted

    def get_unresolved_rejections(self, agent_a_id: str, agent_b_id: str) -> list[DialogueRecord]:
        """Ermittelt abgewiesene Verhandlungen zwischen zwei Entitäten zur Kontextinjektion."""
        pair = {agent_a_id, agent_b_id}
        return [
            r for r in self._records
            if {r.sender_id, r.recipient_id} == pair and r.intent == "reject"
        ]

    def clear(self) -> None:
        self._records.clear()