import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from extractor import extract  # noqa: E402

KEYS = {"entities", "emails", "phones", "urls", "dates", "money",
        "noun_phrases", "keywords", "nlp_steps"}

SAMPLE = ("Rahul Sharma from Infosys will visit Bengaluru on 15/11/2026. "
          "Call +91 9876543210 or mail rahul.s@infosys.com. Fee is Rs. 2,500.")


def test_all_keys_present():
    assert set(extract(SAMPLE)) == KEYS


def test_regex_fields():
    r = extract(SAMPLE)
    assert r["emails"] == ["rahul.s@infosys.com"]
    assert r["phones"] == ["+91 9876543210"]
    assert r["dates"] == ["15/11/2026"]
    assert r["money"] == ["Rs. 2,500"]


def test_nlp_steps_present():
    r = extract(SAMPLE)
    assert len(r["nlp_steps"]) == 3
    assert r["nlp_steps"][0]["tokens"][0] == "Rahul"


def test_empty_input():
    r = extract("")
    assert set(r) == KEYS and r["nlp_steps"] == []


def test_single_word_and_character():
    assert extract("Bengaluru")["nlp_steps"][0]["tokens"] == ["Bengaluru"]
    assert extract("a")["nlp_steps"][0]["tokens"] == ["a"]
