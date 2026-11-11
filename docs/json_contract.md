# JSON contract

`extract(text)` in `app/extractor.py` returns this shape, and `POST /api/extract` returns the same JSON.
Both members must keep to it.

```json
{
  "entities":     {"PERSON": ["Rahul Sharma"], "ORGANIZATION": ["Infosys"], "GPE": ["Bengaluru"]},
  "emails":       ["rahul.s@infosys.com"],
  "phones":       ["+91 9876543210"],
  "urls":         ["www.dsu.edu.in"],
  "dates":        ["15/11/2026", "15 Nov 2026"],
  "times":        ["10:30 am"],
  "money":        ["Rs. 2,500"],
  "percentages":  ["18%"],
  "pin_codes":    ["560001"],
  "noun_phrases": ["the annual tech fest"],
  "keywords":     ["rahul", "infosys", "fest"],
  "nlp_steps":    [{"sentence": "Rahul Sharma from Infosys ...",
                    "tokens": ["Rahul", "Sharma", "from"],
                    "pos": [["Rahul", "NNP"], ["Sharma", "NNP"], ["from", "IN"]]}],
  "stats":        {"characters": 161, "sentences": 3, "tokens": 34, "truncated": false}
}
```

Rules:
- Every key is always present, even when empty.
- Entity labels come from NLTK: PERSON, ORGANIZATION, GPE, LOCATION, FACILITY, GSP.
- Empty input returns all keys with empty lists/objects (and zeros in `stats`).
- Lists keep the order the items appear in the text, with duplicates removed.
- Input longer than 20,000 characters is cut at a word boundary and `stats.truncated` is `true`.

Formats recognised:

| Key | Examples |
|---|---|
| `emails` | `rahul.s@infosys.com` |
| `phones` | `+91 9876543210`, `+919876543210`, `98765 43210`, `080-2345 6789` |
| `urls` | `https://dsu.edu.in/admissions`, `www.dsu.edu.in` |
| `dates` | `15/11/2026`, `15-11-26`, `15.11.2026`, `2026-11-15`, `15 Nov 2026`, `15th November, 2026`, `November 15, 2026`, `Nov 2026` |
| `times` | `10:30`, `10:30 am`, `9 PM`, `9.30 p.m.`, `18:45 hrs`, `1830 hrs` |
| `money` | `Rs. 2,500`, `₹1,50,000/-`, `INR 3 lakh`, `$20`, `500 rupees` |
| `percentages` | `18%`, `12.5 %`, `40 percent`, `40 per cent` |
| `pin_codes` | `560001`, `560 001` (6 digits, first digit 1-9) |

Changelog:
- v2 (feature/nlp-engine): added `times`, `percentages`, `pin_codes` and `stats`. `dates` now also
  includes dates written in words. All v1 keys are unchanged, so v1 front-end code keeps working.
