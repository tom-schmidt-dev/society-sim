from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.application.services.rendering.frame_buffer_service import FrameBufferService
from src.domain.models.agent.agent_cognition import AgentCognitiveSnapshot
from src.domain.models.rendering.render_frame import RenderFrame


class AgentTabWidget(QWidget):
    """Stellt Vitalwerte, Zielhierarchie und Gedankenstrom eines einzelnen Agenten dar."""

    def __init__(self, agent_id: str, agent_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.agent_id = agent_id
        self.agent_name = agent_name
        self._last_logged_thought: str = ""
        self._last_strategy: str = ""

        layout = QVBoxLayout(self)

        # Statusleiste (Vitalwerte & Ziele)
        status_box = QGroupBox("Aktueller Zustand")
        status_layout = QVBoxLayout(status_box)

        self.need_label = QLabel("Bedürfnis: -")
        self.goal_label = QLabel("Ziel: -")
        self.strategy_label = QLabel("Strategie: -")

        status_layout.addWidget(self.need_label)
        status_layout.addWidget(self.goal_label)
        status_layout.addWidget(self.strategy_label)
        layout.addWidget(status_box)

        # Chronologischer Gedanken-Stream
        stream_box = QGroupBox("Gedankenverlauf (Live)")
        stream_layout = QVBoxLayout(stream_box)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        stream_layout.addWidget(self.log_view)
        layout.addWidget(stream_box)

    def update_snapshot(self, snapshot: AgentCognitiveSnapshot) -> None:
        """Aktualisiert Anzeigen und protokolliert Gedanken- und Strategiewechsel."""
        dom_need = snapshot.dominant_need or "Ausgeglichen"
        need_str = f"Bedürfnis: {dom_need} ({snapshot.dominant_need_level:.2f})"
        goal_str = (
            f"Primärziel: {snapshot.primary_goal or 'Keines'} | "
            f"Subgoal: {snapshot.active_subgoal or 'Warten'}"
        )
        target_str = (
            f"({snapshot.target_position.x}, {snapshot.target_position.y})"
            if snapshot.target_position
            else "Kein Ziel"
        )
        pos_str = f"({snapshot.current_position.x}, {snapshot.current_position.y})"
        strat_str = f"Pos: {pos_str} -> Ziel: {target_str} | Strategie: {snapshot.intended_strategy}"

        self.need_label.setText(need_str)
        self.goal_label.setText(goal_str)
        self.strategy_label.setText(strat_str)

        # Logge bei neuem Takt oder veränderten Gedanken/Strategien
        thought_changed = snapshot.formatted_thought != self._last_logged_thought
        strategy_changed = snapshot.intended_strategy != self._last_strategy

        if thought_changed or strategy_changed:
            entry = f"[Takt {snapshot.tick:03d}] ({snapshot.intended_strategy}) {snapshot.formatted_thought}"
            self.log_view.append(entry)
            self.log_view.moveCursor(QTextCursor.MoveOperation.End)
            self._last_logged_thought = snapshot.formatted_thought
            self._last_strategy = snapshot.intended_strategy


class ThoughtStreamWindow(QMainWindow):
    """Hauptfenster zur entkoppelten Darstellung von Kognitionsprozessen und Dialogen."""

    def __init__(
        self,
        frame_buffer: Optional[FrameBufferService] = None,
        timer_interval_ms: int = 50,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._frame_buffer = frame_buffer
        self._current_tick: int = 0
        self._seen_dialogues: set[str] = set()
        self.agent_tabs: dict[str, AgentTabWidget] = {}

        self.setWindowTitle("Society Sim – Kognitions-Stream")
        self.resize(850, 650)

        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Globale Kopfzeile
        header_layout = QHBoxLayout()
        self.tick_label = QLabel("Takt: 000")
        self.buffer_label = QLabel("Puffer-Vorlauf: 0 Frames")
        header_layout.addWidget(self.tick_label)
        header_layout.addStretch()
        header_layout.addWidget(self.buffer_label)
        main_layout.addLayout(header_layout)

        # Tab-Container
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Dialog-Protokoll-Tab
        self.dialogue_view = QTextEdit()
        self.dialogue_view.setReadOnly(True)
        self.tab_widget.addTab(self.dialogue_view, "💬 Dialogprotokoll")

        # Consumer-Timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.consume_next_frame)
        if self._frame_buffer is not None:
            self._timer.start(timer_interval_ms)

    @property
    def current_tick(self) -> int:
        return self._current_tick

    def consume_next_frame(self) -> None:
        """Liest den nächsten Frame aus dem Puffer und aktualisiert die Ansicht."""
        if self._frame_buffer is None:
            return

        frame = self._frame_buffer.pop_frame()
        if frame is not None:
            self.process_frame(frame)

        self.buffer_label.setText(f"Puffer-Vorlauf: {self._frame_buffer.size} Frames")

    def process_frame(self, frame: RenderFrame) -> None:
        """Aktualisiert alle Komponenten mit den Daten des übergebenen Frames."""
        self._current_tick = frame.tick
        self.tick_label.setText(f"Takt: {frame.tick:03d}")
        self.setWindowTitle(f"Society Sim – Kognitions-Stream (Takt {frame.tick})")

        # 1. Dialoge aktualisieren
        if frame.dialogues:
            for dialogue in frame.dialogues:
                if dialogue not in self._seen_dialogues:
                    self.dialogue_view.append(dialogue)
                    self._seen_dialogues.add(dialogue)
            self.dialogue_view.moveCursor(QTextCursor.MoveOperation.End)

        # 2. Agenten aus Entitäten absichern (damit Tabs immer existieren)
        for ent in frame.entities:
            if ent.get("is_agent"):
                agent_id = ent["id"]
                agent_name = ent["name"]
                if agent_id not in self.agent_tabs:
                    tab = AgentTabWidget(agent_id, agent_name)
                    self.agent_tabs[agent_id] = tab
                    self.tab_widget.addTab(tab, f"🧠 {agent_name} [{agent_id}]")

        # 3. Kognitions-Snapshots einspielen
        for snapshot in frame.snapshots:
            if snapshot.agent_id not in self.agent_tabs:
                tab = AgentTabWidget(snapshot.agent_id, snapshot.agent_name)
                self.agent_tabs[snapshot.agent_id] = tab
                self.tab_widget.addTab(tab, f"🧠 {snapshot.agent_name} [{snapshot.agent_id}]")

            self.agent_tabs[snapshot.agent_id].update_snapshot(snapshot)