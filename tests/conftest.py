import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    """Tests never call a real AI API, even if a developer has a .env with a key."""
    for name in ("LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
