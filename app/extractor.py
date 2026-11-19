"""NLP engine for Information-Extractor: a real information extractor built on NLTK.

Pipeline, run on every sentence:
  1. Sentence segmentation    sent_tokenize (Punkt) + repair after abbreviations like "Rs."
  2. Tokenization             word_tokenize (Treebank rules)
  3. POS tagging              pos_tag (averaged perceptron tagger)
  4. Named-entity recognition ne_chunk (maximum-entropy classifier) + rule layer
                              (gazetteers of Indian names, places, deities; name shapes such as
                              "Mount X", "Lord X", "X University", "X Temple")
  5. Noun-phrase chunking     RegexpParser with a POS-tag grammar
  6. Facts and relations      facts.py: verbs from POS tags + typed items ->
                              who / action / what / where / when / how_much / contact,
                              and (subject, relation, object) triples incl. nltk.sem.extract_rels
  7. Keyword ranking          stopword removal + FreqDist (extra information only)
Fixed-format items (emails, phones, URLs, dates, times, money, percentages, PIN codes) are
found with regular expressions. Optional: llm_refiner.py lets an LLM correct the result.

extract(text, smart=False) returns a dict that follows docs/json_contract.md.
"""
import functools
import re
from collections import Counter
from pathlib import Path

import nltk
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
from nltk.corpus import words as english_words

import facts as facts_mod
import llm_refiner
from gazetteer import (DEITIES, EVENT_WORDS, FIRST_NAMES, HONORIFICS, JOB_TITLES, KNOWN_ORGS,
                       NOT_ENTITIES, ORG_WORDS, PLACE_NOUNS, PLACE_PREFIXES, PLACE_WORDS,
                       PLACES, SURNAMES, TITLES)

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
_YEAR = r"(?:1[0-9]|20)\d{2}"   # years 1000-2099, so history dates work
# "am", "AM", "a.m.", "p. m."
_AMPM = r"[AaPp](?:\.\s?[Mm]\.|\.?[Mm]\b)"
# "2,500", "1,00,000", "99.50"
_AMOUNT = r"\d+(?:,\d+)*(?:\.\d+)?"
_SCALE = r"(?:\s?(?:lakhs?|lacs?|crores?|cr|thousand|million|billion|mn|bn|k)\b)?"

# Ordered by priority. A match is skipped if a field earlier in the list already used
# those characters, e.g. the digits of a phone number never become a PIN code.
FIXED_FORMATS = [
    # Length limits ({1,64}) stop slow backtracking on very long words.
    ("emails", re.compile(
        r"\b[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,8}\b")),
    ("urls", re.compile(r"""
        \b(?:https?://|www\.)[^\s<>"']{0,2000}[^\s<>"'.,;:!?)\]]                  # https://..., www....
      | (?<![\w@./-])(?:[a-z0-9-]{1,63}\.){1,6}(?:com|org|net|edu|gov|in|io|ai|co|ac|info)\b
        (?:/[^\s<>"']{0,2000}[^\s<>"'.,;:!?)\]])?                                # dsu.edu.in/path
    """, re.VERBOSE | re.IGNORECASE)),
    ("phones", re.compile(r"""
        (?<![\w+])(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)                    # mobile: +91 98765 43210
      | (?<![\w+])(?:\+91[\s-]?\(?|\(?0)[1-9]\d{1,3}\)?[\s-]\d{3,4}[\s-]?\d{4}(?!\d)  # landline: 080-2345 6789
      | (?<![\w+])1800[\s-]?\d{3}[\s-]?\d{4}(?!\d)                               # toll-free: 1800 202 6161
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
                 "inc", "jr", "sr", "govt", "dept", "vs", "approx", "smt", "shri", "mt"}


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


def is_common_word(word):
    """True for ordinary English words, including simple plurals ("shares" -> "share")."""
    w, words = word.lower(), common_words()
    return w in words or (w.endswith("s") and w[:-1] in words) or (w.endswith("es") and w[:-2] in words)


def _in_name(row):
    """A token can be part of a name if the NE chunker tagged it, or it is capitalised and
    either tagged NNP or not an ordinary English word (the tagger calls "Guwahati" a verb)."""
    word, pos, iob = row
    if _key(word) in NOT_ENTITIES or not any(ch.isalpha() for ch in word):
        return False
    if iob != "O":
        return True
    return word[:1].isupper() and (pos.startswith("NNP") or (
        pos[:1] in "NVJ" and not is_common_word(word)))


def _known_label(words):
    """Label from the word lists and name-shape rules alone, or None when they say nothing."""
    name = " ".join(words).lower()
    keys = [_key(w) for w in words]
    if name in PLACE_KEYS:
        return "GPE"
    if name in ORG_KEYS or any(k in ORG_WORDS for k in keys):
        return "ORGANIZATION"
    if len(keys) > 1 and (keys[0] in PLACE_PREFIXES or keys[-1] in PLACE_WORDS):
        return "LOCATION"                               # "Mount Kailash", "Kedarnath Temple"
    if keys[0] in HONORIFICS or any(k in DEITIES for k in keys) or name in DEITIES:
        return "PERSON"                                 # "Lord Shiva", "Goddess Parvati"
    if keys[0] in FIRST_NAMES or keys[-1] in SURNAMES:
        return "PERSON"
    return None


def _label(words, chunker_labels, sentence_start, next_word=""):
    """Decide PERSON / ORGANIZATION / GPE / LOCATION for one name.

    Returns (label, number of words dropped from the front) or None.
    Order: word lists and shape rules first, then NLTK's own label.
    """
    drop = 0
    while drop < len(words) - 1 and _key(words[drop]) in TITLES | JOB_TITLES:
        drop += 1                                          # "Dr. Anjali Rao", "Chairman V. Narayanan"
    has_title = drop > 0
    rest = words[drop:]
    # "Meet Priya", "Visit Shillong": drop a capitalised ordinary word at the start of a
    # sentence, but only when the rest is already a known name ("Smriti Mandhana" stays).
    if (sentence_start and not has_title and len(rest) > 1 and is_common_word(rest[0])
            and _key(rest[0]) not in HONORIFICS
            and not _known_label(rest) and _known_label(rest[1:])):
        drop += 1
        rest = words[drop:]
    if not rest:
        return None

    if len(rest) > 1 and _key(rest[-1]) in EVENT_WORDS and not _known_label(rest[-1:]):
        return None                                        # "Asian Games", "Ganesh Chaturthi"
    if next_word.lower() in PLACE_NOUNS and " ".join(rest).lower() not in ORG_KEYS:
        return "LOCATION", drop                           # "the Ganga river"
    known = _known_label(rest)
    if known:
        return known, drop
    if has_title:
        return "PERSON", drop
    if not chunker_labels:
        return None                      # plain capitalised word, no evidence it is a name
    if len(rest) == 1 and is_common_word(rest[0]) and (
            sentence_start or "PERSON" not in chunker_labels):
        return None                      # "Fee is...", "Python", "Team": ordinary words, not names
    if "PERSON" in chunker_labels:       # "Rahul"(PERSON) + "Sharma"(ORGANIZATION) -> PERSON
        return "PERSON", drop
    return Counter(chunker_labels).most_common(1)[0][0], drop


def plain_entities(tagged):
    """NLTK's ne_chunk output only, with no rules (the baseline used in evaluation)."""
    found, i = [], 0
    for row_start, node in _walk(named_entities(tagged)):
        if isinstance(node, nltk.Tree):
            words = [w for w, _ in node.leaves()]
            found.append({"label": node.label(), "text": " ".join(words),
                          "start": row_start, "end": row_start + len(words)})
    return found


def _walk(tree):
    pos = 0
    for node in tree:
        yield pos, node
        pos += len(node.leaves()) if isinstance(node, nltk.Tree) else 1


def find_entities(tagged):
    """NE chunking with NLTK, then merge split names and fix labels with simple rules.

    Returns [{"label", "text", "start", "end"}] with token positions in the sentence.
    """
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
        # Split a trailing place off a longer name: "Infosys Bengaluru" -> "Infosys" + "Bengaluru".
        cuts = [(i, j)]
        whole = " ".join(w for w, _, _ in rows[i:j]).lower()
        for size in (2, 1):
            tail = " ".join(w for w, _, _ in rows[j - size:j]).lower()
            if (j - i > size and tail in PLACE_KEYS and whole not in PLACE_KEYS | ORG_KEYS
                    and not any(_key(w) in ORG_WORDS | HONORIFICS for w, _, _ in rows[i:j])
                    and _key(rows[i][0]) not in PLACE_PREFIXES):
                cuts = [(i, j - size), (j - size, j)]
                break
        starts_sentence = i == 0 or rows[i - 1][0] in {":", ";", "(", "``", '"', "-", "--"}
        next_word = rows[j][0] if j < len(rows) else ""
        for n, (a, b) in enumerate(cuts):
            labelled = _label([w for w, _, _ in rows[a:b]],
                              [iob[2:] for _, _, iob in rows[a:b] if iob != "O"],
                              sentence_start=starts_sentence and n == 0,
                              next_word=next_word if b == j else "")
            if labelled:
                label, drop = labelled
                found.append({"label": label, "start": a + drop, "end": b,
                              "text": " ".join(w for w, _, _ in rows[a + drop:b])})
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


def fixed_format_matches(text: str) -> list:
    """Run the regexes in priority order; each character can belong to only one match.

    Returns [(key, value, start_char, end_char)] in priority order.
    """
    taken = bytearray(len(text))
    found = []
    for key, pattern in FIXED_FORMATS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(taken[start:end]):
                continue
            taken[start:end] = b"\x01" * (end - start)
            raw = match.group()
            value = raw.strip()
            start += len(raw) - len(raw.lstrip())
            found.append((key, value, start, start + len(value)))
    return found


def find_fixed_formats(text: str) -> dict:
    found = {key: [] for key, _ in FIXED_FORMATS}
    for key, value, _, _ in fixed_format_matches(text):
        found[key].append(value)
    return {key: unique(values) for key, values in found.items()}


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
                  stats={"characters": 0, "sentences": 0, "tokens": 0, "truncated": False},
                  items=[], facts=[], relations=[], mode="nltk", corrections=[],
                  ai={"requested": False, "used": False, "error": None})
    return result


def extract(text, smart=False, use_rules=True) -> dict:
    """Run the full pipeline and return the JSON described in docs/json_contract.md.

    smart=True asks the optional LLM layer (app/llm_refiner.py) to check the result;
    without an API key, or on any error, the NLTK result is returned unchanged.
    use_rules=False gives plain NLTK NER with no rule layer (used for the evaluation baseline).
    """
    result = empty_result()
    text, truncated = clean_input(text)
    result["stats"].update(characters=len(text), truncated=truncated)
    if not text:
        if smart:
            result["ai"].update(requested=True, error="empty text")
        return result

    for key, values in find_fixed_formats(text).items():
        result[key] = values

    words, phrases, token_count = [], [], 0
    memory = facts_mod.Memory()                                 # for "he"/"she"/"they"
    for sent in split_sentences(text):                          # 1. sentence segmentation
        tokens = nltk.word_tokenize(sent, preserve_line=True)   # 2. tokenization
        if not tokens:
            continue
        tagged = nltk.pos_tag(tokens)                           # 3. POS tagging
        token_count += len(tokens)
        result["nlp_steps"].append({"sentence": sent, "tokens": tokens,
                                    "pos": [list(pair) for pair in tagged]})

        ents = find_entities(tagged) if use_rules else plain_entities(tagged)  # 4. NER
        for ent in ents:
            result["entities"].setdefault(ent["label"], []).append(ent["text"])

        nps = [" ".join(w for w, _ in np.leaves())                              # 5. NP chunking
               for np in CHUNKER.parse(tagged).subtrees(lambda t: t.label() == "NP")]
        phrases += nps

        sent_matches = fixed_format_matches(sent)
        sentence_facts, sentence_rels, typed = facts_mod.analyse_sentence(   # 6. facts + relations
            sent, tokens, tagged, ents, sent_matches, memory, chunker=CHUNKER)
        result["facts"] += sentence_facts
        result["relations"] += sentence_rels
        result["items"] += typed

        words += [w.lower() for w in tokens if w.isalpha() and len(w) > 1
                  and w.lower() not in STOP and w.lower() not in NOT_ENTITIES]  # no "pm", "nov"

    result["entities"] = {label: unique(names) for label, names in result["entities"].items()}
    result["noun_phrases"] = unique(phrases)
    result["keywords"] = [w for w, _ in nltk.FreqDist(words).most_common(TOP_KEYWORDS)]  # 7.
    result["items"] = facts_mod.unique_items(result["items"])
    result["relations"] = facts_mod.unique_relations(result["relations"])
    result["stats"].update(sentences=len(result["nlp_steps"]), tokens=token_count)

    if smart:
        llm_refiner.refine(text, result)      # optional AI check; never raises
    return result


if __name__ == "__main__":
    import json
    import sys

    sample = " ".join(sys.argv[1:]) or (
        "Rahul Sharma from Infosys will visit Bengaluru 560001 on 15 Nov 2026 at 10:30 am. "
        "Call +91 9876543210 or mail rahul.s@infosys.com. Fee is Rs. 2,500 with 18% GST.")
    out = extract(sample)
    out.pop("nlp_steps")
    print(json.dumps(out, indent=2, ensure_ascii=False))
