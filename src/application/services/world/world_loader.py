from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.infrastructure.container import ApplicationContainer

from src.domain.models.agent.agent import Agent
from src.domain.models.world.world_definition import WorldDefinition
from src.domain.models.world.world_entity import WorldEntity


class WorldLoader:
    """Instanziiert eine WorldDefinition deterministisch in einer Simulationsumgebung."""

    @staticmethod
    def apply_to_container(world: WorldDefinition, container: ApplicationContainer | Any) -> None:
        # Statische Hindernisse setzen
        for obs in world.obstacles:
            container.grid.set_obstacle(obs)

        # Entitäten registrieren
        for ent_data in world.entities:
            entity = WorldEntity(
                id=ent_data.id,
                name=ent_data.name,
                position=ent_data.position,
                entity_type=ent_data.entity_type,
                is_conversational=ent_data.is_conversational,
            )
            container.engine.register_entity(entity)

        # Agenten und deren Ziele initialisieren
        for ag_data in world.agents:
            agent = Agent(
                id=ag_data.id,
                name=ag_data.name,
                position=ag_data.position,
            )
            container.engine.register_agent(agent)

            if ag_data.target_position:
                container.engine.set_agent_target(
                    agent_id=agent.id,
                    target=ag_data.target_position,
                    destination_name=ag_data.destination_name,
                )
