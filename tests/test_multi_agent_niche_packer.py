from __future__ import annotations

import pytest
from src.domain.models.agent import Agent
from src.domain.models.mental_map import AgentMentalMap
from src.domain.models.position import Position
from src.application.services.convoy_coordinator import Convoy
from src.application.services.multi_agent_niche_packer import (
    MultiAgentNichePacker,
    NicheType,
    NicheConfiguration,
)


class TestMultiAgentNichePacker:
    @pytest.fixture
    def packer(self) -> MultiAgentNichePacker:
        return MultiAgentNichePacker()

    def test_pocket_niche_packing_lifo(self, packer: MultiAgentNichePacker) -> None:
        """
        Korridor verläuft horizontal auf y=5 von x=1 bis x=10.
        Pocket-Nische erstreckt sich vertikal von (5, 5) nach (5, 6) und (5, 7).
        Konvoi von 2 Agenten (Leader an (3, 5), Follower an (2, 5)).
        """
        mmap = AgentMentalMap(width=20, height=20)
        # Korridor
        trajectory = [Position(x, 5) for x in range(1, 11)]
        for p in trajectory:
            mmap.update_tile(p, is_walkable=True, tick=1)

        # Nische
        niche_pocket = [Position(5, 6), Position(5, 7)]
        for p in niche_pocket:
            mmap.update_tile(p, is_walkable=True, tick=1)

        a1 = Agent(id="lead", name="Leader", position=Position(3, 5))
        a2 = Agent(id="foll", name="Follower", position=Position(2, 5))
        convoy = Convoy(id="c1", members=[a1, a2], direction_vector=(1, 0))

        config = packer.find_niche_configuration(
            convoy=convoy,
            mental_map=mmap,
            passing_convoy_trajectory=trajectory,
        )

        assert config is not None
        assert config.niche_type == NicheType.POCKET
        assert config.total_capacity >= 2
        assert config.junctions == [Position(5, 5)]

        # LIFO-Invariante: Leader erhält die tiefste Kachel (5, 7), Follower (5, 6)
        lead_slot = next(s for s in config.slots if s.agent_id == "lead")
        foll_slot = next(s for s in config.slots if s.agent_id == "foll")
        assert lead_slot.slot_position == Position(5, 7)
        assert foll_slot.slot_position == Position(5, 6)

        # Keine Nischenkachel darf auf der Trajektorie liegen
        assert all(s.slot_position not in trajectory for s in config.slots)

        # Pfad zum Slot muss zusammenhängend sein
        assert lead_slot.path_to_slot == [Position(5, 5), Position(5, 6), Position(5, 7)]
        assert foll_slot.path_to_slot == [Position(5, 5), Position(5, 6)]

    def test_bypass_niche_configuration(self, packer: MultiAgentNichePacker) -> None:
        """
        Korridor auf y=5 von x=1 bis x=10.
        Bypass-Parallelweg auf y=6 von x=3 bis x=6, verbunden an (3, 5) und (6, 5).
        Konvoi von 2 Agenten weicht aus.
        """
        mmap = AgentMentalMap(width=20, height=20)
        trajectory = [Position(x, 5) for x in range(1, 11)]
        for p in trajectory:
            mmap.update_tile(p, is_walkable=True, tick=1)

        bypass_cells = [Position(x, 6) for x in range(3, 7)]
        for p in bypass_cells:
            mmap.update_tile(p, is_walkable=True, tick=1)

        a1 = Agent(id="a1", name="A1", position=Position(2, 5))
        a2 = Agent(id="a2", name="A2", position=Position(1, 5))
        convoy = Convoy(id="c1", members=[a1, a2], direction_vector=(1, 0))

        config = packer.find_niche_configuration(
            convoy=convoy,
            mental_map=mmap,
            passing_convoy_trajectory=trajectory,
        )

        assert config is not None
        assert config.niche_type == NicheType.BYPASS
        assert len(config.junctions) >= 2
        assert Position(3, 5) in config.junctions
        assert Position(6, 5) in config.junctions
        assert all(s.slot_position not in trajectory for s in config.slots)

    def test_distributed_niches(self, packer: MultiAgentNichePacker) -> None:
        """
        Korridor auf y=5 von x=1 bis x=10.
        Getrennte Einzellücken bei (3, 6) und (7, 6) ohne Verbindung untereinander.
        """
        mmap = AgentMentalMap(width=20, height=20)
        trajectory = [Position(x, 5) for x in range(1, 11)]
        for p in trajectory:
            mmap.update_tile(p, is_walkable=True, tick=1)

        mmap.update_tile(Position(3, 6), is_walkable=True, tick=1)
        mmap.update_tile(Position(7, 6), is_walkable=True, tick=1)

        a1 = Agent(id="a1", name="A1", position=Position(3, 5))
        a2 = Agent(id="a2", name="A2", position=Position(7, 5))
        convoy = Convoy(id="c1", members=[a1, a2], direction_vector=(1, 0))

        config = packer.find_niche_configuration(
            convoy=convoy,
            mental_map=mmap,
            passing_convoy_trajectory=trajectory,
        )

        assert config is not None
        assert config.niche_type == NicheType.DISTRIBUTED
        assert len(config.slots) == 2
        assigned_positions = {s.slot_position for s in config.slots}
        assert assigned_positions == {Position(3, 6), Position(7, 6)}
        assert all(s.slot_position not in trajectory for s in config.slots)

    def test_insufficient_capacity_returns_none(self, packer: MultiAgentNichePacker) -> None:
        """
        Konvoi benötigt 3 Plätze, es ist jedoch nur eine Nische mit 2 Plätzen vorhanden.
        """
        mmap = AgentMentalMap(width=20, height=20)
        trajectory = [Position(x, 5) for x in range(1, 10)]
        for p in trajectory:
            mmap.update_tile(p, is_walkable=True, tick=1)

        # Nur 2 Nischenzellen
        mmap.update_tile(Position(5, 6), is_walkable=True, tick=1)
        mmap.update_tile(Position(5, 7), is_walkable=True, tick=1)

        a1 = Agent(id="1", name="A1", position=Position(3, 5))
        a2 = Agent(id="2", name="A2", position=Position(2, 5))
        a3 = Agent(id="3", name="A3", position=Position(1, 5))
        convoy = Convoy(id="c1", members=[a1, a2, a3], direction_vector=(1, 0))

        config = packer.find_niche_configuration(
            convoy=convoy,
            mental_map=mmap,
            passing_convoy_trajectory=trajectory,
        )
        assert config is None
