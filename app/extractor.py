"""NLP engine for Information-Extractor (Member 1).

NLTK pipeline, run on every sentence:
  1. Sentence segmentation    sent_tokenize (Punkt) + repair after abbreviations like "Rs."
  2. Tokenization             word_tokenize (Treebank rules)
  3. POS tagging              pos_tag (averaged perceptron tagger)
  4. Named-entity recognition ne_chunk (maximum-entropy classifier)
  5. Noun-phrase chunking     RegexpParser with a POS-tag grammar
  6. Keyword ranking          stopword removal + FreqDist

Fixed-format items (emails, phones, URLs, dates, times, money, percentages,
PIN codes) are found with regular expressions instead.

extract(text) returns a dict that follows docs/json_contract.md.
"""
import functools
import re
from collections import Counter
from pathlib import Path

import nltk
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
from nltk.corpus import words as english_words

from gazetteer import (FIRST_NAMES, KNOWN_ORGS, NOT_ENTITIES, ORG_WORDS, PLACES, SURNAMES,
                       TITLES)

# setup_nltk.py downloads the models into ./nltk_data. Look there first, so the app
# works locally without NLTK_DATA set. On Render, NLTK_DATA points to the same folder.
LOCAL_NLTK_DATA = Path(__file__).resolve().parent.parent / "nltk_data"
if LOCAL_NLTK_DATA.is_dir() and str(LOCAL_NLTK_DATA) not in nltk.data.path:
    nltk.data.path.insert(0, str(LOCAL_NLTK_DATA))

# Longer input is cut at a word boundary. This keeps one request under a few seconds.
MAX_CHARS = 20_000
TOP_KEYWORDS = 10

# --------------------------------------------------------------------------- regex
# Building blocks for dates written in words.
_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
          r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b(?:\.(?=\s*\d))?")
_DAY = r"(?:3[01]|[12]\d|0?[1-9])(?:st|nd|rd|th)?"
_YEAR = r"(?:19|20)\d{2}"
# "am", "AM", "a.m.", "p. m."
_AMPM = r"[AaPp](?:\.\s?[Mm]\.|\.?[Mm]\b)"
# "2,500", "1,00,000", "99.50"
_AMOUNT = r"\d+(?:,\d+)*(?:\.\d+)?"
_SCALE = r"(?:\s?(?:lakhs?|lacs?|crores?|cr|thousand|million|billion|mn|bn|k)\b)?"

# Ordered by priority. A match is skipped if a field earlier in the list already used
# those characters, e.g. the digits of a phone number never become a PIN code.
FIXED_FORMATS = [
    ("urls", re.compile(
        r"\b(?:https?://|www\.)[^\s<>\"']{0,2000}[^\s<>\"'.,;:!?)\]]")),
    # Length limits ({1,64}) stop slow backtracking on very long words.
    ("emails", re.compile(
        r"\b[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,8}\b")),
    ("phones", re.compile(r"""
        (?<![\w+])(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)                    # mobile: +91 98765 43210
      | (?<![\w+])(?:\+91[\s-]?\(?|\(?0)[1-9]\d{1,3}\)?[\s-]\d{3,4}[\s-]?\d{4}(?!\d)  # landline: 080-2345 6789
    """, re.VERBOSE)),
    ("money", re.compile(rf"""
        (?:₹|\bRs\b\.?|\bINR\b|\$|\bUSD\b|€|£)\s?{_AMOUNT}{_SCALE}(?:/-)?   # Rs. 2,500 | ₹500 | INR 3 lakh
      | \b{_AMOUNT}{_SCALE}\s?(?:rupees|rs\b\.?|inr\b|dollars|usd\b)       # 500 rupees | 2 lakh rupees
    """, re.VERBOSE | re.IGNORECASE)),
    ("percentages", re.compile(
        r"(?<![\w.])[-+]?\d+(?:\.\d+)?\s?(?:%|percent\b|per\s?cent\b)", re.IGNORECASE)),
    ("dates", re.compile(rf"""
        \b\d{{1,2}}([/-])\d{{1,2}}\1(?:\d{{4}}|\d{{2}})\b      # 15/11/2026, 15-11-26
      | \b\d{{1,2}}\.\d{{1,2}}\.\d{{4}}\b                    # 15.11.2026
      | \b\d{{4}}-\d{{2}}-\d{{2}}\b                          # 2026-11-15
      | \b{_DAY}(?:\s+of)?\s+{_MONTH}(?:,?\s+{_YEAR}\b)?     # 15 Nov 2026, 15th of November
      | \b{_MONTH}\s+{_DAY}\b(?:,?\s+{_YEAR}\b)?             # November 15, 2026
      | \b{_MONTH},?\s+{_YEAR}\b                             # November 2026
    """, re.VERBOSE)),
    ("times", re.compile(rf"""
        \b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?!\d)(?:\s?{_AMPM})?(?:\s?(?:IST|hrs)\b)?  # 10:30, 18:45 hrs, 10:30 am
      | \b(?:1[0-2]|0?[1-9])(?:\.[0-5]\d)?\s?{_AMPM}(?:\s?IST\b)?                          # 9 am, 9.30 p.m.
      | \b(?:[01]\d|2[0-3])[0-5]\d\s?hrs\b                                                 # 1830 hrs
    """, re.VERBOSE)),
    # Indian PIN code: 6 digits, first digit 1-9, sometimes written "560 001".
    ("pin_codes", re.compile(r"""
        (?<![\w.,/])(?<!\d-)[1-9]\d{2}\s?\d{3}(?!\d|[.,]\d)
        (?!\s?(?:%|percent|rupees|rs\b|inr\b|lakhs?|crores?|dollars))
    """, re.VERBOSE | re.IGNORECASE)),
]

# Order of keys in the returned dict (docs/json_contract.md).
REGEX_KEYS = ["emails", "phones", "urls", "dates", "times", "money", "percentages", "pin_codes"]

# ------------------------------------------------------------------------ NLTK setup
# NP = optional determiner/possessive, any adjectives, then one or more nouns.
CHUNKER = nltk.RegexpParser(r"NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}")
STOP = set(stopwords.words("english"))

# Punkt does not know Indian abbreviations such as "Rs." and would end a sentence there.
ABBREVIATIONS = {"rs", "dr", "mr", "mrs", "ms", "prof", "no", "st", "ltd", "pvt", "co",
                 "inc", "jr", "sr", "govt", "dept", "vs", "approx", "smt", "shri"}


@functools.lru_cache(maxsize=1)
def _ne_chunker():
    """Load the NE classifier once. nltk.ne_chunk() reloads it on every call (~1 s each)."""
    try:
        from nltk.chunk import ne_chunker  # NLTK 3.9+
    except ImportError:
        return None
    return ne_chunker()


def named_entities(tagged):
    chunker = _ne_chunker()
    return chunker.parse(tagged) if chunker else nltk.ne_chunk(tagged)


# ---------------------------------------------------------- NER correction rules
PLACE_KEYS = {p.lower() for p in PLACES}
ORG_KEYS = {o.lower() for o in KNOWN_ORGS}


@functools.lru_cache(maxsize=1)
def common_words():
    """Lower-case English words from NLTK's `words` corpus ("fee", "meet", "notice")."""
    return {w for w in english_words.words() if w.islower()}


def _key(word):
    return word.lower().rstrip(".")


def _in_name(row):
    """A token belongs to a name if the NE chunker tagged it, or it is a capitalised NNP."""
    word, pos, iob = row
    if _key(word) in NOT_ENTITIES or not any(ch.isalpha() for ch in word):
        return False
    return iob != "O" or (pos.startswith("NNP") and word[:1].isupper())


def _label(words, chunker_labels, sentence_start):
    """Decide PERSON / ORGANIZATION / GPE using the gazetteer first, then NLTK's label."""
    has_title = _key(words[0]) in TITLES
    words = words[1:] if has_title else words
    if not words:
        return None
    name = " ".join(words)
    keys = [_key(w) for w in words]

    if name.lower() in PLACE_KEYS:
        return "GPE", name
    if name.lower() in ORG_KEYS or any(k in ORG_WORDS for k in keys):
        return "ORGANIZATION", name
    if has_title or keys[0] in FIRST_NAMES or keys[-1] in SURNAMES:
        return "PERSON", name
    if not chunker_labels:
        return None                      # plain capitalised word, no evidence it is a name
    if sentence_start and len(keys) == 1 and keys[0] in common_words():
        return None                      # "Fee is Rs. 500" - capital only because it starts the sentence
    if "PERSON" in chunker_labels:       # "Rahul"(PERSON) + "Sharma"(ORGANIZATION) -> PERSON
        return "PERSON", name
    return Counter(chunker_labels).most_common(1)[0][0], name


def find_entities(tagged):
    """NE chunking with NLTK, then merge split names and fix labels with simple rules."""
    rows = tree2conlltags(named_entities(tagged))    # [(word, POS, "B-PERSON" | "I-PERSON" | "O")]
    found, i = [], 0
    while i < len(rows):
        if not _in_name(rows[i]):
            i += 1
            continue
        j = i
        while j < len(rows) and _in_name(rows[j]):
            j += 1
        # Join "University of Mysore", "Bank of Baroda".
        while (j + 1 < len(rows) and rows[j][0] == "of" and _in_name(rows[j + 1])
               and any(_key(w) in ORG_WORDS for w, _, _ in rows[i:j])):
            j += 2
            while j < len(rows) and _in_name(rows[j]):
                j += 1
        span = rows[i:j]
        # Split a trailing place off a longer name: "Infosys Bengaluru" -> "Infosys" + "Bengaluru".
        parts = [span]
        for size in (2, 1):
            tail = " ".join(w for w, _, _ in span[-size:]).lower()
            if len(span) > size and tail in PLACE_KEYS and " ".join(
                    w for w, _, _ in span).lower() not in PLACE_KEYS | ORG_KEYS and not any(
                    _key(w) in ORG_WORDS for w, _, _ in span):
                parts = [span[:-size], span[-size:]]
                break
        for part in parts:
            entity = _label([w for w, _, _ in part],
                            [iob[2:] for _, _, iob in part if iob != "O"],
                            sentence_start=(i == 0 and part is parts[0]))
            if entity:
                found.append(entity)
        i = j
    return found


def split_sentences(text: str) -> list:
    """Punkt sentence split, then re-join any split made right after a known abbreviation."""
    sentences = []
    for sent in nltk.sent_tokenize(text):
        prev = sentences[-1] if sentences else ""
        last_word = prev.rsplit(None, 1)[-1].rstrip(".").lower() if prev else ""
        if prev.endswith(".") and last_word in ABBREVIATIONS:
            sentences[-1] = prev + " " + sent
        else:
            sentences.append(sent)
    return sentences


def find_fixed_formats(text: str) -> dict:
    """Run the regexes in priority order; each character can belong to only one match."""
    taken = bytearray(len(text))
    found = {}
    for key, pattern in FIXED_FORMATS:
        items = []
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(taken[start:end]):
                continue
            taken[start:end] = b"\x01" * (end - start)
            items.append(match.group().strip())
        found[key] = unique(items)
    return found


def unique(items):
    """Remove duplicates but keep the first-seen order."""
    return list(dict.fromkeys(items))


def clean_input(text):
    """Accept None/non-strings, normalise odd spaces, cut very long input at a word."""
    text = "" if text is None else str(text)
    text = text.replace(" ", " ").replace("​", "").strip()
    if len(text) <= MAX_CHARS:
        return text, False
    cut = text[:MAX_CHARS]
    head = cut.rpartition(" ")[0]
    return (head if len(head) > MAX_CHARS - 200 else cut), True


def empty_result() -> dict:
    result = {"entities": {}}
    result.update({key: [] for key in REGEX_KEYS})
    result.update(noun_phrases=[], keywords=[], nlp_steps=[],
                  stats={"characters": 0, "sentences": 0, "tokens": 0, "truncated": False})
    return result


def extract(text) -> dict:
    result = empty_result()
    text, truncated = clean_input(text)
    result["stats"].update(characters=len(text), truncated=truncated)
    if not text:
        return result

    found = find_fixed_formats(text)
    for key in REGEX_KEYS:
        result[key] = found[key]

    words, phrases, token_count = [], [], 0
    for sent in split_sentences(text):                          # 1. sentence segmentation
        tokens = nltk.word_tokenize(sent, preserve_line=True)   # 2. tokenization
        if not tokens:
            continue
        tagged = nltk.pos_tag(tokens)                           # 3. POS tagging
        token_count += len(tokens)
        result["nlp_steps"].append({"sentence": sent, "tokens": tokens,
                                    "pos": [list(pair) for pair in tagged]})

        for label, name in find_entities(tagged):              # 4. named-entity recognition
            result["entities"].setdefault(label, []).append(name)

        for np in CHUNKER.parse(tagged).subtrees(lambda t: t.label() == "NP"):  # 5. NP chunking
            phrases.append(" ".join(word for word, _ in np.leaves()))

        words += [w.lower() for w in tokens
                  if w.isalpha() and len(w) > 1 and w.lower() not in STOP]

    result["entities"] = {label: unique(names) for label, names in result["entities"].items()}
    result["noun_phrases"] = unique(phrases)
    result["keywords"] = [w for w, _ in nltk.FreqDist(words).most_common(TOP_KEYWORDS)]  # 6.
    result["stats"].update(sentences=len(result["nlp_steps"]), tokens=token_count)
    return result


if __name__ == "__main__":
    import json
    import sys

    sample = " ".join(sys.argv[1:]) or (
        "Rahul Sharma from Infosys will visit Bengaluru 560001 on 15 Nov 2026 at 10:30 am. "
        "Call +91 9876543210 or mail rahul.s@infosys.com. Fee is Rs. 2,500 with 18% GST.")
    print(json.dumps(extract(sample), indent=2, ensure_ascii=False))
