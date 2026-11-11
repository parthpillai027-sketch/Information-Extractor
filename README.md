# Information-Extractor

A website that extracts the important information from any English text — people, organizations, places, dates, phone numbers, emails, money amounts and keywords — using **NLP with NLTK**.

> Live demo: _add link after deployment_

## Features
- Named-entity recognition: PERSON, ORGANIZATION, GPE / LOCATION
- Emails, Indian phone numbers (+91), URLs, DD/MM/YYYY dates, ₹ / Rs / INR amounts
- Key noun phrases and top keywords
- "Show NLP steps" panel: tokens and POS tags for every sentence
- Works on a paragraph, a sentence, a single word or a character
- JSON API: `POST /api/extract`

## NLP techniques used
| Technique | NLTK function | Role |
|---|---|---|
| Sentence segmentation | `sent_tokenize` | Process each sentence separately |
| Tokenization | `word_tokenize` | Split sentences into words |
| Stopword removal | `corpus.stopwords` | Drop common words before keyword ranking |
| POS tagging | `pos_tag` | Label nouns, verbs, etc. |
| Noun-phrase chunking | `RegexpParser` | Find key phrases |
| Named-entity recognition | `ne_chunk` | Find people, organizations, places |
| Frequency distribution | `FreqDist` | Rank keywords |

## Setup
```bash
git clone https://github.com/parthpillai027-sketch/Information-Extractor.git
cd Information-Extractor
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python setup_nltk.py
python app/app.py              # open http://127.0.0.1:5000
```

## Run tests
```bash
pytest -q
```

## Project structure
```
app/extractor.py      NLP engine (Member 1)
app/app.py            Flask routes (Member 2)
app/templates/        HTML
app/static/           CSS + JS
docs/json_contract.md JSON format shared by both halves
samples/              Example input texts
tests/                Unit tests
```

## Accuracy
_Precision / recall / F1 per entity type — add after evaluation._

## Team
- S Krithik Yadav — NLP engine
- _Member 2 name_ — Website & deployment

## License
MIT
