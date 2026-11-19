"""Information extraction: typed items, facts (who/action/where/when...) and relations."""
from extractor import extract


def types(result):
    return {(i["text"], i["type"]) for i in result["items"]}


def facts(result):
    return [{k: v for k, v in f.items() if v and k != "sentence"} for f in result["facts"]]


def test_shiva_parvati_kailash():
    r = extract("Lord Shiva and Goddess Parvati live on Mount Kailash.")
    assert {("Lord Shiva", "PERSON"), ("Goddess Parvati", "PERSON"),
            ("Mount Kailash", "PLACE")} <= types(r)
    assert facts(r) == [{"who": ["Lord Shiva", "Goddess Parvati"], "action": "live",
                         "where": ["Mount Kailash"]}]
    rel = {(x["subject"], x["relation"], x["object"]) for x in r["relations"]}
    assert ("Lord Shiva", "live on", "Mount Kailash") in rel


def test_college_notice():
    r = extract("The Department of Computer Science at Dayananda Sagar University will conduct a "
                "workshop on 20 Nov 2026 at 10 am. Fee is Rs. 200 per student.")
    f = r["facts"][0]
    assert f["action"] == "will conduct"
    assert "Dayananda Sagar University" in f["who"]
    assert f["when"] == ["20 Nov 2026", "10 am"]
    assert f["what"] == ["workshop"]
    assert r["facts"][1]["how_much"] == ["Rs. 200"]


def test_order_sms():
    r = extract("Flipkart will deliver your order of Rs. 1,299 to Imphal 795001 by 22 Nov 2026. "
                "Call 98765 43210 for help.")
    f = r["facts"][0]
    assert f["who"] == ["Flipkart"] and f["action"] == "will deliver"
    assert f["where"] == ["Imphal", "795001"]
    assert f["when"] == ["22 Nov 2026"] and f["how_much"] == ["Rs. 1,299"]
    assert r["facts"][1]["contact"] == ["98765 43210"]
    assert ("795001", "PLACE") in types(r)


def test_news_line():
    r = extract("Mumbai, 3 March 2026: Tata Motors reported a 14.5% rise in profit.")
    f = r["facts"][0]
    assert f["who"] == ["Tata Motors"] and f["action"] == "reported"
    assert f["where"] == ["Mumbai"] and f["when"] == ["3 March 2026"]
    assert f["how_much"] == ["14.5%"]


def test_shared_subject_and_pronoun():
    r = extract("Sri Krishna was born in Mathura and grew up in Vrindavan. He later moved to Dwarka.")
    assert [(f["who"], f["action"], f["where"]) for f in r["facts"]] == [
        (["Sri Krishna"], "was born", ["Mathura"]),
        (["Sri Krishna"], "grew up", ["Vrindavan"]),
        (["Sri Krishna"], "moved", ["Dwarka"]),
    ]


def test_passive_agent_becomes_who():
    f = extract("The study was funded by the Indian Council of Medical Research.")["facts"][0]
    assert f["who"] == ["Indian Council of Medical Research"]
    assert f["action"] == "was funded"


def test_affiliation_is_not_a_second_doer():
    r = extract("Rahul Sharma from Infosys will visit Bengaluru.")
    f = r["facts"][0]
    assert f["who"] == ["Rahul Sharma"] and f["affiliation"] == ["Infosys"]
    rel = {(x["subject"], x["relation"], x["object"], x["method"]) for x in r["relations"]}
    assert ("Rahul Sharma", "from", "Infosys", "nltk.extract_rels") in rel


def test_greeting_name_is_not_the_doer():
    f = extract("Dear Arjun, Zoho Corporation has selected you.")["facts"][0]
    assert f["who"] == ["Zoho Corporation"]


def test_place_shapes_and_events():
    e = extract("The Ganga river flows past Kedarnath Temple during the Kumbh Mela.")["entities"]
    places = e.get("LOCATION", []) + e.get("GPE", [])
    assert "Ganga" in places and "Kedarnath Temple" in places
    assert not any("Mela" in v for vals in e.values() for v in vals)


def test_verbless_notice_line():
    f = extract("Venue: MG Road, Bengaluru 560001.")["facts"][0]
    assert f["what"] == ["Venue"] and "Bengaluru" in f["where"] and "560001" in f["where"]


def test_every_item_has_a_known_type():
    r = extract("Priya paid Rs. 500 (10%) on 1 Dec 2026 at 9 am to Zoho in Chennai via a@b.com.")
    allowed = {"PERSON", "PLACE", "ORGANIZATION", "DATE", "TIME", "MONEY", "PERCENT", "CONTACT", "THING"}
    assert r["items"] and all(i["type"] in allowed for i in r["items"])


def test_history_years():
    assert extract("Gandhi was born on 2 October 1869.")["dates"] == ["2 October 1869"]
