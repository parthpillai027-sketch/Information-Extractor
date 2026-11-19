"""Checks the Flask app (app/app.py): pages, JSON API, errors and limits."""
from app import MAX_INPUT_CHARS, app

client = app.test_client()


def test_api_extract_returns_contract_json():
    resp = client.post("/api/extract", json={"text": "Mail rahul.s@infosys.com before 15 Nov 2026."})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["emails"] == ["rahul.s@infosys.com"]
    assert data["dates"] == ["15 Nov 2026"]
    assert data["mode"] == "nltk"
    assert "elapsed_ms" in data["stats"]


def test_empty_text_is_allowed():
    resp = client.post("/api/extract", json={"text": ""})
    assert resp.status_code == 200 and resp.get_json()["facts"] == []


def test_bad_requests_get_400_json():
    cases = [{}, {"json": {}}, {"data": "not json"}, {"json": {"text": None}},
             {"json": {"text": 5}}, {"json": ["a list"]}, {"json": {"text": "hi", "smart": "yes"}}]
    for kwargs in cases:
        resp = client.post("/api/extract", **kwargs)
        assert resp.status_code == 400, kwargs
        body = resp.get_json()
        assert body["status"] == 400 and body["error"]


def test_too_long_text_is_rejected():
    resp = client.post("/api/extract", json={"text": "a" * (MAX_INPUT_CHARS + 1)})
    assert resp.status_code == 400
    assert "too long" in resp.get_json()["error"]


def test_smart_without_key_falls_back_to_nltk():
    resp = client.post("/api/extract", json={"text": "Lord Shiva lives on Mount Kailash.", "smart": True})
    data = resp.get_json()
    assert resp.status_code == 200
    assert data["mode"] == "nltk"
    assert data["ai"]["requested"] is True and data["ai"]["used"] is False
    assert "LLM_API_KEY" in data["ai"]["error"]


def test_health():
    data = client.get("/api/health").get_json()
    assert data["status"] == "ok"
    assert data["smart_available"] is False
    assert data["max_input_chars"] == MAX_INPUT_CHARS


def test_unknown_api_route_is_json_404():
    resp = client.get("/api/nope")
    assert resp.status_code == 404 and resp.get_json()["status"] == 404


def test_wrong_method_is_json_405():
    assert client.get("/api/extract").status_code == 405


def test_home_page_renders():
    assert client.get("/").status_code == 200
    resp = client.post("/", data={"text": "Visit Bengaluru 560001."})
    assert resp.status_code == 200
