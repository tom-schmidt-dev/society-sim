from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from src.infrastructure.cognition.health.ollama_health_checker import OllamaHealthChecker


def _create_mock_response(status: int, payload: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


def test_check_status_success() -> None:
    payload = {
        "models": [
            {"name": "llama3-8b-q8:latest"},
            {"name": "nomic-embed-text:latest"},
        ]
    }
    mock_resp = _create_mock_response(200, payload)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        # Darf keine Exception werfen
        OllamaHealthChecker.check_status(
            api_base="http://localhost:11434",
            model_name="ollama/llama3-8b-q8",
            timeout=2.0,
        )


def test_check_status_raises_on_connection_error() -> None:
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            OllamaHealthChecker.check_status(
                api_base="http://localhost:11434",
                model_name="ollama/llama3-8b-q8",
            )
        assert "nicht erreichbar" in str(exc_info.value)


def test_check_status_raises_on_http_error() -> None:
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            url="http://localhost:11434/api/tags",
            code=500,
            msg="Internal Server Error",
            hdrs=MagicMock(),
            fp=None,
        ),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            OllamaHealthChecker.check_status(
                api_base="http://localhost:11434",
                model_name="ollama/llama3-8b-q8",
            )
        assert "Fehlercode 500" in str(exc_info.value)


def test_check_status_raises_when_model_missing() -> None:
    payload = {
        "models": [
            {"name": "mistral:latest"},
        ]
    }
    mock_resp = _create_mock_response(200, payload)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(RuntimeError) as exc_info:
            OllamaHealthChecker.check_status(
                api_base="http://localhost:11434",
                model_name="ollama/llama3-8b-q8",
            )
        assert "nicht geladen" in str(exc_info.value)