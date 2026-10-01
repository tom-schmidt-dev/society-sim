from __future__ import annotations

import json
from typing import Any
import urllib.error
import urllib.request


class OllamaHealthChecker:
    """Prüft deterministisch die Erreichbarkeit des Ollama-Dienstes und das Vorhandensein geforderter Modelle."""

    @classmethod
    def normalize_model_name(cls, model_name: str) -> str:
        """Entfernt Framework-Präfixe wie 'ollama/' zur Bereinigung des Tag-Vergleichs."""
        name = model_name.strip()
        if name.startswith("ollama/"):
            name = name[len("ollama/") :]
        return name

    @classmethod
    def check_status(
        cls,
        api_base: str,
        model_name: str,
        timeout: float = 3.0,
    ) -> None:
        """
        Validiert die HTTP-Erreichbarkeit von /api/tags und prüft den Modellbestand.
        Wirft einen RuntimeError bei Verbindungsfehler, ungültigem HTTP-Status oder fehlendem Modell.
        """
        base_url = api_base.rstrip("/")
        endpoint = f"{base_url}/api/tags"
        target_model = cls.normalize_model_name(model_name)

        req = urllib.request.Request(endpoint, headers={"User-Agent": "society-sim-health-check"})

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Ollama-Dienst unter {base_url} antwortete mit unerwartetem HTTP-Fehlercode {response.status}."
                    )
                raw_data = response.read().decode("utf-8")
                payload: dict[str, Any] = json.loads(raw_data)
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Ollama-Dienst unter {base_url} meldet Fehlercode {e.code}: {e.reason}."
            ) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError(
                f"Ollama-Dienst nicht erreichbar unter {base_url}: {e}."
            ) from e

        available_models = [m.get("name", "") for m in payload.get("models", [])]

        is_present = any(
            target_model == model_tag
            or f"{target_model}:latest" == model_tag
            or model_tag.startswith(f"{target_model}:")
            for model_tag in available_models
        )

        if not is_present:
            raise RuntimeError(
                f"Modell '{model_name}' (gesucht als '{target_model}') ist im Ollama-Dienst unter {base_url} nicht geladen. "
                f"Verfügbare Modelle: {', '.join(available_models) if available_models else 'Keine'}."
            )