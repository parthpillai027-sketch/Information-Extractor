"""Checks that the Flask app (app/app.py) serves the extractor correctly."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from app import app  # noqa: E402

client = app.test_client()


def test_api_extract_returns_contract_json():
    resp = client.post("/api/extract", json={"text": "Mail rahul.s@infosys.com before 15 Nov 2026."})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["emails"] == ["rahul.s@infosys.com"]
    assert data["dates"] == ["15 Nov 2026"]


def test_api_handles_missing_or_bad_body():
    for kwargs in ({}, {"json": {}}, {"data": "not json"}, {"json": {"text": None}}):
        resp = client.post("/api/extract", **kwargs)
        assert resp.status_code == 200
        assert resp.get_json()["nlp_steps"] == []


def test_home_page_renders():
    assert client.get("/").status_code == 200
    resp = client.post("/", data={"text": "Visit Bengaluru 560001."})
    assert resp.status_code == 200
    assert b"560001" in resp.data
