import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from extractor import MAX_CHARS, extract  # noqa: E402

KEYS = {"entities", "emails", "phones", "urls", "dates", "times", "money", "percentages",
        "pin_codes", "noun_phrases", "keywords", "nlp_steps", "stats"}

SAMPLE = ("Rahul Sharma from Infosys will visit Bengaluru on 15/11/2026. "
          "Call +91 9876543210 or mail rahul.s@infosys.com. Fee is Rs. 2,500.")


# ----------------------------------------------------------------- contract
def test_all_keys_present():
    assert set(extract(SAMPLE)) == KEYS


@pytest.mark.parametrize("text", ["", "   \n\t ", None, "a", ".", "9876543210", "12345"])
def test_every_input_returns_every_key(text):
    r = extract(text)
    assert set(r) == KEYS
    assert isinstance(r["entities"], dict)


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
    assert r["nlp_steps"][2]["sentence"] == "Fee is Rs. 2,500."   # "Rs." does not end a sentence


def test_entities():
    e = extract(SAMPLE)["entities"]
    assert e["PERSON"] == ["Rahul Sharma"]
    assert e["ORGANIZATION"] == ["Infosys"]
    assert e["GPE"] == ["Bengaluru"]


def test_entity_rules():
    e = extract("Dr. Anjali Rao of the University of Mysore met officials from Canara Bank "
                "in New Delhi.")["entities"]
    assert "Anjali Rao" in e["PERSON"]                       # title "Dr." removed
    assert "University of Mysore" in e["ORGANIZATION"]       # "X of Y" joined
    assert "Canara Bank" in e["ORGANIZATION"]                # org keyword "Bank"
    assert "New Delhi" in e["GPE"]


def test_sentence_initial_common_word_is_not_an_entity():
    e = extract("Fee is Rs. 500 for each student.")["entities"]
    assert not any("Fee" in names for names in e.values())


def test_imperative_verb_is_dropped_from_name():
    assert extract("Meet Priya at 5 pm.")["entities"] == {"PERSON": ["Priya"]}
    assert "Smriti Mandhana" in extract("Smriti Mandhana scored 117 runs.")["entities"]["PERSON"]


def test_mistagged_city_is_still_found():
    # The POS tagger tags "Guwahati" as a verb here; the capitalisation rule still finds it.
    assert extract("Fly from Delhi to Guwahati today.")["entities"]["GPE"] == ["Delhi", "Guwahati"]


def test_place_head_words():
    e = extract("The office is on Jubilee Hills Road near Rajaji Nagar.")["entities"]
    assert e["LOCATION"] == ["Jubilee Hills Road", "Rajaji Nagar"]


# ----------------------------------------------------------- new fields
def test_pin_codes():
    r = extract("Office: MG Road, Bengaluru 560001. Old address: Delhi - 110 001.")
    assert r["pin_codes"] == ["560001", "110 001"]


def test_pin_code_not_taken_from_phone_or_money():
    r = extract("Call 9876543210 and pay Rs. 250000 or 150000 rupees.")
    assert r["pin_codes"] == []
    assert r["phones"] == ["9876543210"]
    assert r["money"] == ["Rs. 250000", "150000 rupees"]


def test_percentages():
    r = extract("Attendance is 85%, results rose 12.5 % and fees fell 10 percent (or 5 per cent).")
    assert r["percentages"] == ["85%", "12.5 %", "10 percent", "5 per cent"]


def test_times():
    r = extract("Doors open at 9 am, talk at 10:30 AM, lunch 1.15 p.m., exit 18:45 hrs, bus 2030 hrs.")
    assert r["times"] == ["9 am", "10:30 AM", "1.15 p.m.", "18:45 hrs", "2030 hrs"]


def test_decimal_is_not_a_time():
    assert extract("The score was 9.30 out of 10.")["times"] == []


@pytest.mark.parametrize("text, expected", [
    ("Due on 15 Nov 2026.", "15 Nov 2026"),
    ("Due on 15th November, 2026.", "15th November, 2026"),
    ("Due on the 1st of March 2027.", "1st of March 2027"),
    ("Due on November 15, 2026.", "November 15, 2026"),
    ("Due on Nov. 15 2026.", "Nov. 15 2026"),
    ("Due in December 2026.", "December 2026"),
    ("Due on 2026-11-15.", "2026-11-15"),
    ("Due on 15.11.2026.", "15.11.2026"),
    ("Due on 15-11-26.", "15-11-26"),
])
def test_dates(text, expected):
    assert extract(text)["dates"] == [expected]


def test_month_words_alone_are_not_dates():
    assert extract("You may join in March or later.")["dates"] == []


@pytest.mark.parametrize("text, expected", [
    ("+919876543210", "+919876543210"),
    ("+91-98765-43210", "+91-98765-43210"),
    ("98765 43210", "98765 43210"),
    ("09876543210", "09876543210"),
    ("080-2345 6789", "080-2345 6789"),
    ("1800 202 6161", "1800 202 6161"),
])
def test_phone_formats(text, expected):
    assert extract(f"Call {text} today.")["phones"] == [expected]


def test_money_formats():
    r = extract("Pay ₹1,50,000/- or INR 3 lakh or $20 or 500 rupees.")
    assert r["money"] == ["₹1,50,000/-", "INR 3 lakh", "$20", "500 rupees"]


def test_urls_without_trailing_punctuation():
    r = extract("See https://dsu.edu.in/admissions, or www.dsu.edu.in.")
    assert r["urls"] == ["https://dsu.edu.in/admissions", "www.dsu.edu.in"]


def test_bare_domain_url_but_not_email_domain():
    r = extract("Apply at scholarships.gov.in or mail help@nsp.gov.in today.")
    assert r["urls"] == ["scholarships.gov.in"]
    assert r["emails"] == ["help@nsp.gov.in"]


def test_duplicates_removed_in_order():
    r = extract("Mail a@x.com, then b@y.org, then a@x.com again.")
    assert r["emails"] == ["a@x.com", "b@y.org"]


# ----------------------------------------------------------- edge cases
def test_empty_input():
    r = extract("")
    assert set(r) == KEYS and r["nlp_steps"] == []
    assert r["stats"] == {"characters": 0, "sentences": 0, "tokens": 0, "truncated": False}


def test_whitespace_only_and_none():
    assert extract("  \n ")["nlp_steps"] == []
    assert extract(None)["nlp_steps"] == []


def test_single_word_and_character():
    assert extract("Bengaluru")["nlp_steps"][0]["tokens"] == ["Bengaluru"]
    assert extract("a")["nlp_steps"][0]["tokens"] == ["a"]
    assert extract("?")["nlp_steps"][0]["tokens"] == ["?"]


def test_numbers_only():
    assert extract("9876543210")["phones"] == ["9876543210"]
    assert extract("560001")["pin_codes"] == ["560001"]
    r = extract("12345 2026 7")
    assert r["keywords"] == [] and r["entities"] == {}
    assert r["nlp_steps"][0]["tokens"] == ["12345", "2026", "7"]


def test_non_string_input():
    assert extract(9876543210)["phones"] == ["9876543210"]


def test_very_long_input_is_truncated_and_fast():
    text = "Priya Nair met the Infosys team in Mysuru on 15 Nov 2026. " * 2000
    start = time.time()
    r = extract(text)
    assert time.time() - start < 30
    assert r["stats"]["truncated"] is True
    assert r["stats"]["characters"] <= MAX_CHARS
    assert r["dates"] == ["15 Nov 2026"]          # duplicates removed
    assert "Priya Nair" in r["entities"]["PERSON"]


def test_long_single_word_does_not_hang():
    start = time.time()
    extract("a" * 100_000)
    extract("x." * 50_000)
    assert time.time() - start < 10
