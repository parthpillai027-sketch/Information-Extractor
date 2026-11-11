# JSON contract

`extract(text)` in `app/extractor.py` returns this shape, and `POST /api/extract` returns the same JSON.
Both members must keep to it.

```json
{
  "entities":     {"PERSON": ["Rahul Sharma"], "ORGANIZATION": ["Infosys"], "GPE": ["Bengaluru"]},
  "emails":       ["rahul.s@infosys.com"],
  "phones":       ["+91 9876543210"],
  "urls":         ["www.dsu.edu.in"],
  "dates":        ["15/11/2026"],
  "money":        ["Rs. 2,500"],
  "noun_phrases": ["the annual tech fest"],
  "keywords":     ["rahul", "infosys", "fest"],
  "nlp_steps":    [{"sentence": "Rahul Sharma from Infosys ...",
                    "tokens": ["Rahul", "Sharma", "from"],
                    "pos": [["Rahul", "NNP"], ["Sharma", "NNP"], ["from", "IN"]]}]
}
```

Rules:
- Every key is always present, even when empty.
- Entity labels come from NLTK: PERSON, ORGANIZATION, GPE, LOCATION, FACILITY, GSP.
- Empty input returns all keys with empty lists/objects.
