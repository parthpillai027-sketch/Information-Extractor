# JSON contract (v3)

`extract(text, smart=False)` in `app/extractor.py` returns this shape, and `POST /api/extract` returns the same JSON.
Both members must keep to it. **v3 only adds keys; every v1/v2 key is unchanged.**

## Request
```
POST /api/extract
Content-Type: application/json

{"text": "Lord Shiva and Goddess Parvati live on Mount Kailash.", "smart": false}
```
- `text` (string, required, at most 100,000 characters; the NLP reads the first 20,000).
- `smart` (true/false, optional, default false): ask the optional AI layer to check the result.
- Errors are JSON with a status code: `{"error": "\"text\" is required and must be a string.", "status": 400}`.
- `GET /api/health` returns `{"status": "ok", "nltk_version": "...", "smart_available": false, "ai_provider": "xai", "max_input_chars": 100000, "nlp_chars": 20000}`.

## Response
```json
{
  "entities":     {"PERSON": ["Lord Shiva", "Goddess Parvati"], "LOCATION": ["Mount Kailash"]},
  "emails":       [], "phones": [], "urls": [],
  "dates":        [], "times": [], "money": [], "percentages": [], "pin_codes": [],
  "noun_phrases": ["Lord Shiva", "Goddess Parvati", "Mount Kailash"],
  "keywords":     ["lord", "shiva", "goddess", "parvati", "live", "mount", "kailash"],
  "nlp_steps":    [{"sentence": "Lord Shiva and ...", "tokens": ["Lord", "Shiva"], "pos": [["Lord", "NNP"], ["Shiva", "NNP"]]}],
  "stats":        {"characters": 52, "sentences": 1, "tokens": 10, "truncated": false, "elapsed_ms": 12.3},

  "items": [
    {"text": "Lord Shiva",      "type": "PERSON", "label": "PERSON",   "source": "nltk"},
    {"text": "Goddess Parvati", "type": "PERSON", "label": "PERSON",   "source": "nltk"},
    {"text": "Mount Kailash",   "type": "PLACE",  "label": "LOCATION", "source": "nltk"}
  ],
  "facts": [
    {"sentence": "Lord Shiva and Goddess Parvati live on Mount Kailash.",
     "who": ["Lord Shiva", "Goddess Parvati"], "action": "live", "what": [],
     "where": ["Mount Kailash"], "when": [], "how_much": [], "contact": [], "affiliation": []}
  ],
  "relations": [
    {"subject": "Lord Shiva", "relation": "live on", "object": "Mount Kailash",
     "object_type": "PLACE", "method": "verb pattern"},
    {"subject": "Goddess Parvati", "relation": "live on", "object": "Mount Kailash",
     "object_type": "PLACE", "method": "verb pattern"}
  ],
  "mode": "nltk",
  "corrections": [],
  "ai": {"requested": false, "used": false, "error": null}
}
```
(`stats.elapsed_ms` is added by the API, not by `extract()`.)

## Keys
| Key | Meaning |
|---|---|
| `items` | **Every extracted item with a category** `type`: `PERSON`, `PLACE`, `ORGANIZATION`, `DATE`, `TIME`, `MONEY`, `PERCENT`, `CONTACT`, `THING`. `label` is the finer kind (`GPE`, `LOCATION`, `email`, `phone`, `url`, `pin_code`, `noun_phrase`, …). `source`: `nltk` (NER + rules), `regex`, `chunker`, or `ai`. |
| `facts` | **One record per action (verb) in each sentence**: `who` did `action` to `what`, `where`, `when`, `how_much`, `contact`, and `affiliation` (the organisation a person belongs to, e.g. "Rahul Sharma *from Infosys*"). Lists can be empty; `action` is `""` for a line with no verb ("Venue: MG Road"). |
| `relations` | `(subject, relation, object)` triples. `method` = `verb pattern` (built from the facts) or `nltk.extract_rels` (NLTK's relation extractor). |
| `mode` | `"nltk"`, or `"hybrid"` when the AI layer checked and corrected the result. |
| `corrections` | What the AI changed: `{"text", "nltk_type", "final_type"}`. `nltk_type: null` = AI added it; `final_type: null` = AI removed it. |
| `ai` | `requested`, `used`, `provider`, `model`, `seconds`, and `error` (why the AI was not used, e.g. no key or timeout). |
| `entities` | Same as v1: NLTK labels → names. Rebuilt from `items` when the AI corrects types. |
| `keywords`, `noun_phrases` | Extra information only (most frequent words, noun chunks). |

Rules:
- Every key is always present, even when empty.
- Entity labels come from NLTK: PERSON, ORGANIZATION, GPE, LOCATION, FACILITY, GSP.
- Gods, goddesses and epic characters are PERSON; honorifics (Lord, Goddess, Sri, Mahatma) are kept in the name, Mr./Dr./Prof. and job titles are removed.
- Lists keep the order the items appear in the text, with duplicates removed.
- Input longer than 20,000 characters is cut at a word boundary and `stats.truncated` is `true`.

Formats recognised:

| Key | Examples |
|---|---|
| `emails` | `rahul.s@infosys.com` |
| `phones` | `+91 9876543210`, `+919876543210`, `98765 43210`, `080-2345 6789`, `1800 202 6161` |
| `urls` | `https://dsu.edu.in/admissions`, `www.dsu.edu.in`, `scholarships.gov.in` |
| `dates` | `15/11/2026`, `15-11-26`, `15.11.2026`, `2026-11-15`, `15 Nov 2026`, `15th November, 2026`, `November 15, 2026`, `Nov 2026`, `2 October 1869` |
| `times` | `10:30`, `10:30 am`, `9 PM`, `9.30 p.m.`, `18:45 hrs`, `1830 hrs` |
| `money` | `Rs. 2,500`, `₹1,50,000/-`, `INR 3 lakh`, `$20`, `500 rupees` |
| `percentages` | `18%`, `12.5 %`, `40 percent`, `40 per cent` |
| `pin_codes` | `560001`, `560 001` (6 digits, first digit 1-9) |

Changelog:
- v3 (feature/info-extraction): added `items`, `facts`, `relations`, `mode`, `corrections`, `ai`; request field `smart`; `/api/health`; 400 JSON errors; 100,000-character limit; years 1000-2099 in word dates.
- v2: added `times`, `percentages`, `pin_codes` and `stats`. `dates` also includes dates written in words.
