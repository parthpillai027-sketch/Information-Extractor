# Information-Extractor

A website that extracts the important information from any English text — people, organizations, places, dates, times, phone numbers, emails, money, percentages, PIN codes and keywords — using **NLP with NLTK**.

> Live demo: _add link after deployment_

## Features
- Named-entity recognition: PERSON, ORGANIZATION, GPE / LOCATION, using NLTK plus a rule layer for Indian names and places
- Emails, phone numbers (+91 mobiles, landlines, 1800 toll-free), URLs (including bare domains like `dsu.edu.in`)
- Dates in numbers (`15/11/2026`, `2026-11-15`) and in words (`15 Nov 2026`, `November 15, 2026`)
- Times (`10:30 am`, `9.30 p.m.`, `18:45 hrs`), money (`Rs. 2,500`, `₹1,50,000/-`, `INR 3 lakh`), percentages, PIN codes
- Key noun phrases and top keywords
- "Show NLP steps" panel: tokens and POS tags for every sentence
- Handles empty input, a single character, numbers only, and very long text (cut at 20,000 characters)
- JSON API: `POST /api/extract` (format in [docs/json_contract.md](docs/json_contract.md))

## How it works
```
text ──► clean (None, odd spaces, cut very long input)
     ├─► regex fields, in priority order: emails → urls → phones → money → percentages → dates → times → PIN codes
     └─► for each sentence:  sent_tokenize → word_tokenize → pos_tag → ne_chunk → rule layer → entities
                                                                    └─► RegexpParser → noun phrases
         all sentences  ──►  drop stopwords → FreqDist → top 10 keywords
```

**Regex priority.** Every regex marks the characters it used, and a later regex cannot reuse them. So the digits of a phone number never also become a PIN code, and the domain inside an email is not also a URL.

**Why a rule layer on top of NLTK's NER?** NLTK's `ne_chunk` is a maximum-entropy classifier trained on American news (the ACE corpus). On Indian text it splits "Rahul Sharma" into PERSON + ORGANIZATION and tags Infosys and Bengaluru as PERSON. The rule layer (`find_entities` in `app/extractor.py`, word lists in `app/gazetteer.py`) does five things:
1. Merges neighbouring name tokens into one name ("Rahul" + "Sharma" → "Rahul Sharma").
2. Checks small word lists (gazetteers): Indian states and cities, well-known organizations, common Indian first names and surnames.
3. Uses head words: a name containing *University, Bank, Ltd, Institute…* is an ORGANIZATION. A name ending in *Road, Nagar, Layout, Hills, Park…* is a LOCATION. A title (*Dr., Mr., Prof.*) marks a PERSON and is removed.
4. Drops ordinary English words that are capitalised only because they start a sentence ("Fee is…", "Meet Priya"). It checks them against NLTK's `words` corpus.
5. Otherwise keeps NLTK's own label.

**Speed.** In NLTK 3.9+, `nltk.ne_chunk()` reloads its model on every call, which costs about 1 second per sentence. The extractor loads the chunker once and reuses it. A 20,000-character text takes about 1.5 s.

## NLP techniques used
| Technique | NLTK function | Role |
|---|---|---|
| Sentence segmentation | `sent_tokenize` (Punkt) | Process each sentence separately; re-join splits after "Rs.", "Dr." |
| Tokenization | `word_tokenize` | Split sentences into words |
| Stopword removal | `corpus.stopwords` | Drop common words before keyword ranking |
| POS tagging | `pos_tag` (averaged perceptron) | Label nouns, verbs, etc. |
| Noun-phrase chunking | `RegexpParser` | Find key phrases: `{<DT\|PRP$>?<JJ.*>*<NN.*>+}` |
| Named-entity recognition | `ne_chunker` (maxent) + rules | Find people, organizations, places |
| Word list lookup | `corpus.words` | Tell ordinary words from names |
| Frequency distribution | `FreqDist` | Rank keywords |
| Regular expressions | `re` | Emails, phones, URLs, dates, times, money, %, PIN codes |

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
`setup_nltk.py` downloads 5 NLTK packages into `./nltk_data` (or into `$NLTK_DATA` if set). It then runs the pipeline once and exits with an error if anything is missing. The folder is git-ignored.

Try the engine on its own: `python app/extractor.py "Meet Priya at 5 pm on 15 Nov 2026."`

## Run tests
```bash
pytest -q                      # 51 tests: every field, edge cases, the Flask API
```

## Accuracy
`python scripts/evaluate.py` runs the extractor on the 15 hand-labelled texts in `samples/`. Each `sample_XX.txt` has a matching `sample_XX.json` with the expected answer. Add `--errors` to list every miss.

| Type | TP | FP | FN | Precision | Recall | F1 |
|---|--:|--:|--:|--:|--:|--:|
| PERSON | 13 | 1 | 1 | 0.93 | 0.93 | 0.93 |
| ORGANIZATION | 15 | 2 | 2 | 0.88 | 0.88 | 0.88 |
| GPE | 30 | 3 | 0 | 0.91 | 1.00 | 0.95 |
| emails | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| phones | 6 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| urls | 5 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| dates | 14 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| times | 10 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| money | 16 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| percentages | 7 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| pin_codes | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| **Entities (NER)** | 58 | 6 | 3 | 0.91 | 0.95 | 0.93 |
| **Regex fields** | 65 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| **Overall (micro)** | 123 | 6 | 3 | 0.95 | 0.98 | 0.96 |

- **Precision** = TP / (TP + FP): of what we found, how much was right. **Recall** = TP / (TP + FN): of what was in the text, how much we found. **F1** is their harmonic mean. Counts are summed over all samples (micro-average).
- An item counts as correct only if its text matches exactly (ignoring case and extra spaces). The GPE row also counts places that NLTK labels LOCATION.
- With only NLTK's `ne_chunk` and no rule layer, entity F1 is **0.35** (precision 0.30, recall 0.43). The first version of the rules reached 0.78; the current rules reach **0.93**. The regex fields scored 0.98 before the toll-free and bare-domain patterns were added.
- **Be honest in the viva:** these same 15 samples were used to find errors while improving the rules. So these numbers are optimistic. A new, unseen set of texts would score lower, especially for NER.
- Remaining errors: "IndiGo" labelled a place, "Microsoft Research India" split in two, "Thangjam Bembem" labelled a place, and "Customer" in "Dear Customer" taken as a person.

## Deployment (Render, free plan)
`render.yaml` is a Render Blueprint:
- **Build:** `pip install -r requirements.txt && python setup_nltk.py`. This downloads the NLTK models into `./nltk_data` at build time. `NLTK_DATA` points to the same folder.
- **Start:** `gunicorn --chdir app app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`. One worker keeps a single copy of the models within the free plan's 512 MB.

Steps: sign in at render.com with GitHub → **New → Blueprint** → choose this repo → wait for the build. The free plan sleeps after 15 minutes idle, so the first request after that takes 30–60 s.

## Project structure
```
app/extractor.py      NLP engine (Member 1)
app/gazetteer.py      Word lists used by the NER rules (Member 1)
app/app.py            Flask routes (Member 2)
app/templates/        HTML
app/static/           CSS + JS
docs/json_contract.md JSON format shared by both halves
samples/              15 example texts + expected JSON answers
scripts/evaluate.py   Precision / recall / F1 per type
tests/                Unit tests (extractor + Flask API)
setup_nltk.py         Downloads NLTK data into ./nltk_data
render.yaml           Render deployment blueprint
```

## Known limitations
- NER is rule-assisted, not deep learning. Names that are in no list and that NLTK gets wrong stay wrong.
- A 6-digit number such as "100000 people" is reported as a PIN code (amounts with ₹ / Rs / rupees are not).
- Dates like 11/12/2026 are not interpreted, so DD/MM and MM/DD are not told apart; the text is returned as written.
- English only.

## Team
- S Krithik Yadav — NLP engine
- _Member 2 name_ — Website & deployment

## License
MIT
