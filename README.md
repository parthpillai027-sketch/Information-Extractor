# Information-Extractor

An **information extraction** website built with **NLTK**. Paste any English text and it finds **who did what, where, when and how much**, not just keywords.

> Live demo: _add link after deployment_

## What it extracts
For *"Lord Shiva and Goddess Parvati live on Mount Kailash. Rahul Sharma from Infosys will visit Bengaluru on 15 Nov 2026."*

| Who | Action | Where | When | Affiliation |
|---|---|---|---|---|
| Lord Shiva, Goddess Parvati | live | Mount Kailash | | |
| Rahul Sharma | will visit | Bengaluru | 15 Nov 2026 | Infosys |

- **Items with categories:** PERSON, PLACE, ORGANIZATION, DATE, TIME, MONEY, PERCENT, CONTACT (email/phone/link), THING (other important noun phrases).
- **Facts:** one record per action: `who · action · what · where · when · how_much · contact`.
- **Relations:** triples such as *(Lord Shiva, live on, Mount Kailash)* and *(Rahul Sharma, from, Infosys)*.
- **Extra:** keywords, noun phrases, and every NLP step (tokens + POS tags) for each sentence.
- **Optional smart mode:** an AI model (Google Gemini on the live site, or xAI Grok) checks NLTK's result and lists its corrections. NLTK always runs first; without an API key the site works fully with NLTK only.

## How it works
```
text ─► sentences (sent_tokenize) ─► words (word_tokenize) ─► POS tags (pos_tag)
          │
          ├─► NER: ne_chunk  +  rule layer ─────────────► PERSON / ORGANIZATION / PLACE
          ├─► regular expressions ───────────────────────► DATE / TIME / MONEY / PERCENT / CONTACT / PIN
          ├─► RegexpParser noun phrases ─────────────────► THING
          │
          └─► facts.py: verbs from POS tags + typed items
                 ├─► facts      who | action | what | where | when | how_much | contact
                 └─► relations  (subject, relation, object), also nltk.sem.extract_rels
                                     │
                    optional ──► llm_refiner.py: AI checks types + facts (strict JSON, 25 s, fallback)
```

1. **NER rule layer** (`app/extractor.py`, `app/gazetteer.py`). NLTK's `ne_chunk` was trained on American news, so on Indian text it calls *Mount Kailash* a PERSON and *Goddess Parvati* an ORGANIZATION. The rules fix this with:
   - word lists of Indian places, organisations, names, and gods and goddesses;
   - name shapes: *Mount/Lake/River X* → place; *Lord/Goddess/Sri X* → person; *X University/Bank* → organisation; *X Temple/Road/Nagar* → place; *X Games/Festival* → event, not a person;
   - removing titles (*Dr., Chairman*), and dropping capitalised ordinary words at the start of a sentence.
2. **Facts** (`app/facts.py`). The POS tags give the verb groups (*will visit*, *was born*, *grew up*). The sentence is cut into clauses at *and / , / that / which*. In each clause:
   - the person or organisation **before** the verb is **who**;
   - typed items **after** the verb fill **what / where / when / how_much / contact**.

   Extra rules:
   - **passive voice:** *funded **by** ICMR* → who = ICMR;
   - **pronouns:** *He* → the last person mentioned;
   - **shared subjects:** *born in A **and** grew up in B*;
   - **greetings:** in *Dear Arjun,*, Arjun is the reader, not the doer;
   - **affiliation:** *Rahul Sharma **from** Infosys*.
3. **Relations.** One triple per (who, object) pair of each fact, with the preposition added (*live **on***). On top of that, NLTK's own relation extractor `nltk.sem.extract_rels` runs on the entity-chunked sentence with patterns such as *PER from/of/at ORG* and *PER in/at/on PLACE*.
4. **Smart mode** (`app/llm_refiner.py`). The text, NLTK's items and the facts are sent to the AI model, which returns corrected JSON. Every type must be one of the 9 categories, and every piece of text must appear word for word in the input, so the AI cannot invent facts. If the AI service is busy (HTTP 5xx) or does not answer within 15 s, the call is tried once more with a backup model. On a timeout (25 s in total), an error or bad JSON, the NLTK result is kept and `ai.error` says why. The response shows `mode` (`nltk` / `hybrid`) and `corrections` such as *Mount Kailash: PERSON → PLACE*.
5. **Speed.** `nltk.ne_chunk()` reloads its model on every call (about 1 s per sentence), so the chunker is loaded once and reused.

## NLP techniques used
| Technique | Tool | Role |
|---|---|---|
| Sentence segmentation | `nltk.sent_tokenize` (Punkt) | Process each sentence; re-join splits after "Rs.", "Dr." |
| Tokenization | `nltk.word_tokenize` | Split sentences into words |
| POS tagging | `nltk.pos_tag` (averaged perceptron) | Find verbs (actions), nouns, numbers |
| Named-entity recognition | `ne_chunker` (maxent) + rule layer | People, organisations, places |
| Gazetteer lookup | `app/gazetteer.py`, `nltk.corpus.words` | Indian names/places/deities; ordinary words vs names |
| Noun-phrase chunking | `nltk.RegexpParser` `{<DT\|PRP$>?<JJ.*>*<NN.*>+}` | THINGs and the *what* of a fact |
| Fact extraction | POS verb groups + clause rules (`facts.py`) | who / action / what / where / when / how much |
| Relation extraction | `nltk.sem.extract_rels` + verb patterns | (subject, relation, object) triples |
| Pattern matching | `re` | Emails, phones, URLs, dates, times, money, %, PIN codes |
| Keyword ranking | `stopwords` + `nltk.FreqDist` | Extra: most frequent content words |
| Optional AI check | Google Gemini / xAI Grok (`llm_refiner.py`) | Correct types and facts, with validation and fallback |

## API
```bash
curl -X POST http://127.0.0.1:5000/api/extract -H "Content-Type: application/json" \
     -d '{"text": "Lord Shiva and Goddess Parvati live on Mount Kailash.", "smart": false}'
curl http://127.0.0.1:5000/api/health
```
- Text can be up to 100,000 characters; the NLP reads the first 20,000.
- Bad requests get a JSON error with status 400, e.g. `{"error": "...", "status": 400}`.
- The full response format is in [docs/json_contract.md](docs/json_contract.md).

## Setup
```bash
git clone https://github.com/parthpillai027-sketch/Information-Extractor.git
cd Information-Extractor
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python setup_nltk.py           # downloads NLTK models into ./nltk_data
python app/app.py              # open http://127.0.0.1:5000
```
**Smart mode (optional).** Copy `.env.example` to `.env` and put your key in `LLM_API_KEY` (a free key from aistudio.google.com with `LLM_PROVIDER=gemini`, or a console.x.ai key with `LLM_PROVIDER=xai`). `.env` is git-ignored, so never commit it. If `LLM_MODEL` is empty, Gemini uses `gemini-3.1-flash-lite`, and for xAI the app asks which Grok models your key can use.

Try the engine on its own: `python app/extractor.py "Sri Krishna was born in Mathura and grew up in Vrindavan."`

## Tests
```bash
pytest -q                      # 78 tests: categories, facts, relations, API errors, AI layer (mocked)
python scripts/evaluate.py     # accuracy tables below
```
The tests never call a real AI API. The AI layer is tested with fake answers: valid, invented text, bad JSON, and timeout.

## Accuracy
Two sets of hand-labelled texts:
- **Tuning set** (`samples/`, 15 texts). Used while writing the entity rules, so its scores are optimistic.
- **Unseen set** (`samples/unseen/`, 10 texts). Mythology, a college notice, an order SMS, news, history, sports and an offer letter. The expected answers, including the facts, were written and committed (commit `Add 10 unseen hand-labelled samples`) **before** the fact rules were written.

Facts are scored as *(action, slot, value)* triples, e.g. *("was born", where, Mathura)*. A value only counts when it is attached to the right action.

**F1 scores**

| Set | System | Entities (NER) | Regex fields | Overall items | Facts |
|---|---|--:|--:|--:|--:|
| Tuning (15) | NLTK only | 0.36 | 1.00 | 0.66 | - |
| Tuning (15) | NLTK + rules | **0.93** | 1.00 | **0.96** | - |
| **Unseen (10), first run** | NLTK only | 0.33 | 0.91 | 0.55 | 0.67 |
| **Unseen (10), first run** | NLTK + rules | **0.93** | 0.91 | **0.92** | **0.90** |
| Unseen (10), after fixes | NLTK + rules | 0.99 | 0.96 | 0.97 | 0.93 |

- **The honest test score is the first unseen run:** entities 0.33 → **0.93** and facts 0.67 → **0.90** compared with plain NLTK.
- After that run, three general bugs it revealed were fixed:
  - years before 1900 ("2 October 1869");
  - job titles before names ("Chairman V. Narayanan");
  - event names taken as people ("Asian Games").

  The last row is therefore no longer a fully unseen score.
- Some names in the unseen texts (Rama, Sita, Ravana, Ayodhya) are also in the word lists. Names that are in no list rely on NLTK and score lower.
- Remaining unseen errors:
  - a toll-free number in 4-4 format;
  - a bare year ("in 1930");
  - "Data Analyst" taken as a person;
  - verbs the tagger reads as nouns ("cost", "closes").
- **NLTK + AI** can be scored with `python scripts/evaluate.py --smart` when a key is set. It is not in the table because CI never calls a paid API.

Per-type tables: `python scripts/evaluate.py --markdown`.

## Deployment (Render, free plan)
`render.yaml` is a Render Blueprint:
- **Build:** `pip install -r requirements.txt && python setup_nltk.py`. This downloads the NLTK models into `./nltk_data`, and `NLTK_DATA` points there.
- **Start:** `gunicorn --chdir app app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`.
- **Health check:** `/api/health`.
- **Smart mode:** Render asks for `LLM_API_KEY` (and optionally `LLM_MODEL`) in its dashboard (`sync: false`), so the key never goes into git. Leave it empty for NLTK-only.

The free plan sleeps after 15 minutes idle, so the first request after that takes 30–60 s.

## Project structure
```
app/extractor.py      NLP pipeline: regex, NER + rule layer, orchestration (Member 1)
app/facts.py          Facts (who/action/where/when...) and relations (Member 1)
app/llm_refiner.py    Optional AI check with validation and fallback (Member 1)
app/gazetteer.py      Word lists: Indian places, organisations, names, deities (Member 1)
app/app.py            Flask routes + JSON API (Member 1 backend)
app/templates/        HTML (Member 2)
app/static/           CSS + JS (Member 2)
docs/json_contract.md JSON format shared by both halves
samples/              15 tuning texts + expected JSON;  samples/unseen/ 10 unseen texts
scripts/evaluate.py   Precision / recall / F1: NLTK vs NLTK + rules (vs AI with --smart)
tests/                78 unit tests (AI calls mocked)
setup_nltk.py         Downloads NLTK data into ./nltk_data
render.yaml           Render deployment blueprint;  .env.example: AI settings template
```

## Known limitations
- Facts come from rules on POS tags, not a full grammar parser. Long sentences with many clauses can attach a value to the wrong action.
- Verbs the tagger mistakes for nouns ("the mission **cost**…") produce no action.
- Names that are in no word list and that NLTK gets wrong stay wrong, unless smart mode is on.
- A 6-digit number such as "100000 people" is reported as a PIN code. A bare year ("in 1930") is not a date.
- English only.

## Team
- S Krithik Yadav — NLP engine and backend
- _Member 2 name_ — Website & deployment

## License
MIT
