"""The optional AI layer, with the LLM mocked (no real API calls)."""
import json
import socket
import urllib.error

import llm_refiner
from extractor import extract

TEXT = "Lord Shiva and Goddess Parvati live on Mount Kailash."


def with_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")


def answer(obj):
    return lambda system, user, cfg: json.dumps(obj)


def test_no_key_keeps_nltk():
    r = extract(TEXT, smart=True)
    assert r["mode"] == "nltk" and r["ai"]["used"] is False
    assert r["entities"]["PERSON"] == ["Lord Shiva", "Goddess Parvati"]


def test_default_provider_is_xai(monkeypatch):
    with_key(monkeypatch)
    cfg = llm_refiner.config()
    assert cfg["provider"] == "xai" and cfg["base_url"] == "https://api.x.ai/v1"
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("LLM_MODEL")
    assert llm_refiner.config()["model"] == "gemini-3.1-flash-lite"
    assert llm_refiner.config()["timeout"] == 25


def test_valid_answer_applies_corrections(monkeypatch):
    with_key(monkeypatch)
    result = extract(TEXT)
    result["items"] = [
        {"text": "Lord Shiva", "type": "PERSON", "label": "PERSON", "source": "nltk"},
        {"text": "Goddess Parvati", "type": "ORGANIZATION", "label": "ORGANIZATION", "source": "nltk"},
        {"text": "Mount Kailash", "type": "PERSON", "label": "PERSON", "source": "nltk"}]
    llm_refiner.refine(TEXT, result, call=answer({
        "items": [{"text": "Lord Shiva", "type": "PERSON"},
                  {"text": "Goddess Parvati", "type": "PERSON"},
                  {"text": "Mount Kailash", "type": "PLACE"}],
        "facts": [{"who": ["Lord Shiva", "Goddess Parvati"], "action": "live",
                   "where": ["Mount Kailash"]}]}))
    assert result["mode"] == "hybrid" and result["ai"]["used"] is True
    assert {"text": "Mount Kailash", "nltk_type": "PERSON", "final_type": "PLACE"} in result["corrections"]
    assert {"text": "Goddess Parvati", "nltk_type": "ORGANIZATION",
            "final_type": "PERSON"} in result["corrections"]
    assert result["entities"]["PERSON"] == ["Lord Shiva", "Goddess Parvati"]
    assert result["entities"]["GPE"] == ["Mount Kailash"]
    assert result["facts"][0]["where"] == ["Mount Kailash"]


def test_invented_text_is_dropped(monkeypatch):
    with_key(monkeypatch)
    result = extract(TEXT)
    llm_refiner.refine(TEXT, result, call=answer({
        "items": [{"text": "Lord Shiva", "type": "PERSON"},
                  {"text": "Lord Vishnu", "type": "PERSON"},          # not in the text
                  {"text": "Mount Kailash", "type": "MOUNTAIN"}],      # not an allowed type
        "facts": [{"who": ["Nandi"], "action": "guards", "where": []}]}))
    texts = [i["text"] for i in result["items"]]
    assert "Lord Vishnu" not in texts
    assert all(f["who"] != ["Nandi"] for f in result["facts"])


def test_bad_json_falls_back(monkeypatch):
    with_key(monkeypatch)
    before = extract(TEXT)
    r = extract(TEXT)
    llm_refiner.refine(TEXT, r, call=lambda s, u, c: "sorry, no JSON here")
    assert r["mode"] == "nltk" and "rejected" in r["ai"]["error"]
    assert r["entities"] == before["entities"] and r["facts"] == before["facts"]


def test_timeout_falls_back(monkeypatch):
    with_key(monkeypatch)

    def slow(system, user, cfg):
        raise socket.timeout("timed out")

    r = extract(TEXT)
    llm_refiner.refine(TEXT, r, call=slow)
    assert r["mode"] == "nltk" and "timed out" in r["ai"]["error"]


def http_error(code):
    return urllib.error.HTTPError("https://ai.example", code, "error", {}, None)


def test_busy_provider_is_tried_once_more(monkeypatch):
    with_key(monkeypatch)
    monkeypatch.setattr(llm_refiner.time, "sleep", lambda s: None)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    calls = []

    def busy_then_ok(system, user, cfg):
        calls.append((cfg["model"], cfg["timeout"]))
        if len(calls) == 1:
            raise http_error(503)
        return json.dumps({"items": [{"text": "Mount Kailash", "type": "PLACE"}]})

    r = extract(TEXT)
    llm_refiner.refine(TEXT, r, call=busy_then_ok)
    assert len(calls) == 2 and r["mode"] == "hybrid"
    assert calls[1][0] == llm_refiner.GEMINI_BACKUP_MODEL == r["ai"]["model"]
    assert calls[0][1] == 15                      # first try: 60% of the 25 s limit
    assert calls[1][1] <= 24                      # second try: what is left (after a 1 s pause)


def test_hanging_provider_is_tried_once_more(monkeypatch):
    with_key(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    clock = [0.0]
    monkeypatch.setattr(llm_refiner.time, "perf_counter", lambda: clock[0])
    calls = []

    def hang_then_ok(system, user, cfg):
        calls.append((cfg["model"], cfg["timeout"]))
        if len(calls) == 1:
            clock[0] += cfg["timeout"]               # the first try used all its 15 s
            raise socket.timeout("timed out")
        return json.dumps({"items": [{"text": "Mount Kailash", "type": "PLACE"}]})

    r = extract(TEXT)
    llm_refiner.refine(TEXT, r, call=hang_then_ok)
    assert r["mode"] == "hybrid" and r["ai"]["model"] == llm_refiner.GEMINI_BACKUP_MODEL
    assert calls[0][1] == 15 and calls[1][1] == 10   # 15 + 10 = the 25 s limit


def test_hanging_twice_falls_back_within_the_limit(monkeypatch):
    with_key(monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(llm_refiner.time, "perf_counter", lambda: clock[0])
    calls = []

    def always_hang(system, user, cfg):
        calls.append(cfg["timeout"])
        clock[0] += cfg["timeout"]
        raise urllib.error.URLError(socket.timeout("timed out"))

    r = extract(TEXT)
    llm_refiner.refine(TEXT, r, call=always_hang)
    assert len(calls) == 2 and sum(calls) == 25
    assert r["mode"] == "nltk" and r["ai"]["seconds"] == 25


def test_busy_twice_falls_back_with_a_clear_message(monkeypatch):
    with_key(monkeypatch)
    monkeypatch.setattr(llm_refiner.time, "sleep", lambda s: None)

    def always_busy(system, user, cfg):
        raise http_error(503)

    r = extract(TEXT)
    llm_refiner.refine(TEXT, r, call=always_busy)
    assert r["mode"] == "nltk" and "busy" in r["ai"]["error"]


def test_wrong_key_is_not_retried(monkeypatch):
    with_key(monkeypatch)
    calls = []

    def forbidden(system, user, cfg):
        calls.append(1)
        raise http_error(403)

    r = extract(TEXT)
    llm_refiner.refine(TEXT, r, call=forbidden)
    assert len(calls) == 1 and "LLM_API_KEY" in r["ai"]["error"]


def test_code_fences_are_accepted(monkeypatch):
    with_key(monkeypatch)
    r = extract(TEXT)
    fenced = '```json\n{"items": [{"text": "Lord Shiva", "type": "PERSON"}]}\n```'
    llm_refiner.refine(TEXT, r, call=lambda s, u, c: fenced)
    assert r["mode"] == "hybrid"


def test_extract_smart_uses_refiner(monkeypatch):
    with_key(monkeypatch)
    monkeypatch.setattr(llm_refiner, "_call_llm",
                        answer({"items": [{"text": "Mount Kailash", "type": "PLACE"}]}))
    r = extract(TEXT, smart=True)
    assert r["mode"] == "hybrid" and r["ai"]["model"] == "test-model"


def test_model_is_picked_when_not_set(monkeypatch):
    llm_refiner._pick_model.cache_clear()

    class FakeResp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return json.dumps(self.body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(llm_refiner.urllib.request, "urlopen", lambda req, timeout: FakeResp(
        {"data": [{"id": "grok-imagine-image"}, {"id": "grok-9"}, {"id": "grok-9-fast"}]}))
    assert llm_refiner._pick_model("https://api.x.ai/v1", "k", 5) == "grok-9-fast"
