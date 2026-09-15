from __future__ import annotations

from typing import Optional
from src.domain.models.dialogue import DialogueRecord


class DialogueHistory:
    def __init__(self, max_capacity: Optional[int] = None) -> None:
        self._records: list[DialogueRecord] = []
        self._max_capacity: Optional[int] = max_capacity

    @property
    def records(self) -> list[DialogueRecord]:
        """Gibt die Liste aller unveränderten Ereignisdatensätze zurück."""
        return list(self._records)

    def add_record(self, record: DialogueRecord) -> None:
        """Fügt einen neuen Datensatz hinzu und begrenzt optional die Kapazität."""
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
        is_inspection: bool = False,
        is_empty_response: bool = False,
    ) -> DialogueRecord:
        """Erstellt und speichert einen DialogueRecord in einem Aufruf."""
        record = DialogueRecord(
            tick=tick,
            sender_id=sender_id,
            sender_name=sender_name,
            recipient_id=recipient_id,
            recipient_name=recipient_name,
            message=message,
            is_inspection=is_inspection,
            is_empty_response=is_empty_response,
        )
        self.add_record(record)
        return record

    def get_recent_records(self, limit: int = 8) -> list[DialogueRecord]:
        """Liefert die letzten n Datensätze."""
        return self._records[-limit:] if limit > 0 else []

    def get_recent_formatted(self, limit: int = 8) -> list[str]:
        """Liefert die letzten n Interaktionen als formatierte Textzeilen für UI und Prompts."""
        recent = self.get_recent_records(limit=limit)
        return [record.format_for_display() for record in recent]

    def clear(self) -> None:
        """Leert den gesamten Historienpuffer."""
        self._records.clear()