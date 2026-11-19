"""Fact and relation extraction: turns tagged words + entities into structured information.

For each sentence:
  1. Every found item gets a token position and a TYPE:
       PERSON, ORGANIZATION, PLACE        <- named-entity recognition (extractor.find_entities)
       DATE, TIME, MONEY, PERCENT, CONTACT <- regular expressions
       QUANTITY ("117 runs"), DURATION ("14 days") <- POS pattern: number + noun
       THING                              <- other noun phrases (RegexpParser)
  2. Verb groups are found from POS tags: MD / VB* words, plus particles and adverbs
     inside them ("will visit", "was born", "grew up").
  3. The sentence is cut into clauses, one per verb group, at "and", ",", "that", "which"...
  4. Each clause becomes a fact:  who | action | what | where | when | how_much | contact
       who   = PERSON / ORGANIZATION before the verb (the subject)
       what  = PERSON / ORGANIZATION / noun phrase after the verb (the object)
       where / when / how_much / contact = items of that type in the clause
     Extra rules: passive voice ("funded by X" -> who = X), pronouns ("He" -> last person),
     shared subjects ("born in A and grew up in B"), greetings ("Dear Arjun," is not the doer),
     affiliations ("Rahul Sharma from Infosys" -> who = Rahul Sharma, affiliation = Infosys).
  5. Relations (subject, relation, object) come from the facts and from NLTK's
     nltk.sem.relextract.extract_rels on the entity-chunked sentence.
"""
import re

from nltk.sem.relextract import extract_rels
from nltk.tree import Tree

from gazetteer import GREETINGS

ENTITY_TYPE = {"PERSON": "PERSON", "ORGANIZATION": "ORGANIZATION", "GPE": "PLACE",
               "LOCATION": "PLACE", "FACILITY": "PLACE", "GSP": "PLACE"}
REGEX_TYPE = {"dates": ("DATE", "date"), "times": ("TIME", "time"), "money": ("MONEY", "money"),
              "percentages": ("PERCENT", "percent"), "emails": ("CONTACT", "email"),
              "phones": ("CONTACT", "phone"), "urls": ("CONTACT", "url"),
              "pin_codes": ("PLACE", "pin_code")}
SLOT_OF = {"PLACE": "where", "DATE": "when", "TIME": "when", "DURATION": "when",
           "MONEY": "how_much", "PERCENT": "how_much", "QUANTITY": "how_much",
           "CONTACT": "contact"}
DOERS = {"PERSON", "ORGANIZATION"}
FACT_SLOTS = ["who", "action", "what", "where", "when", "how_much", "contact", "affiliation"]
AFFILIATION_WORDS = {"from", "of", "at"}

BE_WORDS = {"is", "am", "are", "was", "were", "be", "been", "being", "get", "gets", "got"}
SPLIT_WORDS = {"and", "or", "but", ",", ";", "that", "which", "who", "whom", "while",
               "whereas", "because", "where", "when"}
HE_SHE = {"he", "she"}
THEY = {"they"}
DURATION_NOUNS = {"second", "seconds", "minute", "minutes", "hour", "hours", "day", "days",
                  "week", "weeks", "month", "months", "year", "years"}
QUOTES = {"``": ['"', "“"], "''": ['"', "”"]}

# extract_rels patterns: the words between two entities that signal a relation.
REL_RULES = [
    ("PER", "ORG", re.compile(r"(.*\b)?(from|of|at|with|for|joined|leads?|heads?)\b")),
    ("PER", "GPE", re.compile(r"(.*\b)?(in|at|from|near|on|to|visits?|visited|of)\b")),
    ("PER", "LOC", re.compile(r"(.*\b)?(in|at|from|near|on|to|visits?|visited|of)\b")),
    ("ORG", "GPE", re.compile(r"(.*\b)?(in|at|from|near|of|based)\b")),
    ("ORG", "LOC", re.compile(r"(.*\b)?(in|at|from|near|of|based)\b")),
]
ACE_LABEL = {"PERSON": "PERSON", "ORGANIZATION": "ORGANIZATION", "GPE": "GPE",
             "LOCATION": "LOCATION", "FACILITY": "LOCATION", "GSP": "GPE"}


class Memory:
    """What earlier sentences said, used to resolve "he", "she" and "they"."""

    def __init__(self):
        self.people = []      # last PERSON names mentioned as doers
        self.doers = []       # last non-empty "who" list


def align(sentence, tokens):
    """Character (start, end) of every token inside the sentence."""
    spans, pos = [], 0
    for tok in tokens:
        best = None
        for cand in [tok] + QUOTES.get(tok, []):
            at = sentence.find(cand, pos)
            if at != -1 and (best is None or at < best[0]):
                best = (at, at + len(cand))
        spans.append(best or (pos, pos))
        if best:
            pos = best[1]
    return spans


def _typed_spans(sentence, tagged, entities, matches):
    """All typed items of one sentence with token positions; regex matches win overlaps."""
    n = len(tagged)
    char_spans = align(sentence, [w for w, _ in tagged])
    owner = [None] * n
    spans = []

    def claim(span):
        if span["end"] > span["start"] and all(owner[i] is None for i in range(span["start"], span["end"])):
            for i in range(span["start"], span["end"]):
                owner[i] = span
            spans.append(span)
            return True
        return False

    for key, value, cs, ce in matches:
        idx = [i for i, (a, b) in enumerate(char_spans) if a < ce and b > cs and b > a]
        typ, label = REGEX_TYPE[key]
        span = {"text": value, "type": typ, "label": label, "source": "regex",
                "start": idx[0] if idx else 0, "end": idx[-1] + 1 if idx else 0}
        if not claim(span):
            spans.append(dict(span, start=-1, end=-1))       # keep as an item, no position
    for ent in entities:
        claim({"text": ent["text"], "type": ENTITY_TYPE.get(ent["label"], "THING"),
               "label": ent["label"], "source": "nltk", "start": ent["start"], "end": ent["end"]})

    # Quantities: a number followed by a noun ("117 runs", "88.88 metres", "14 days").
    i = 0
    while i < n:
        if owner[i] is None and tagged[i][1] == "CD":
            j = i
            while j < n and owner[j] is None and tagged[j][1] == "CD":
                j += 1
            k = j
            while k < n and owner[k] is None and tagged[k][1] == "JJ":
                k += 1
            m = k
            while m < n and owner[m] is None and tagged[m][1] in ("NN", "NNS"):
                m += 1
            if m > k:
                noun = tagged[m - 1][0].lower()
                text = sentence[char_spans[i][0]:char_spans[m - 1][1]]
                claim({"text": text, "type": "DURATION" if noun in DURATION_NOUNS else "QUANTITY",
                       "label": "quantity", "source": "pos", "start": i, "end": m})
                i = m
                continue
        i += 1
    spans.sort(key=lambda s: (s["start"], s["end"]))
    return spans, owner


def _verb_groups(tagged, owner):
    """[(start, end)] of verb groups such as "will visit", "was born", "grew up"."""
    def verbish(tag):
        return tag.startswith("VB") or tag == "MD"

    groups, i, n = [], 0, len(tagged)
    while i < n:
        word, tag = tagged[i]
        adjectival = tag in ("VBG", "VBN") and i > 0 and tagged[i - 1][1] in ("DT", "PRP$", "JJ")
        if owner[i] is None and verbish(tag) and not adjectival and word.lower() != "'s":
            j = i + 1
            while j < n and owner[j] is None and (verbish(tagged[j][1]) or tagged[j][1] in ("RB", "RP", "TO")):
                j += 1
            while j > i + 1 and tagged[j - 1][1] in ("RB", "TO"):
                j -= 1
            groups.append((i, j))
            i = j
        else:
            i += 1
    return groups


def _last_split(tagged, a, b):
    """Index of the last clause boundary word in tokens[a:b], or None."""
    for i in range(b - 1, a - 1, -1):
        word, tag = tagged[i]
        if word.lower() in SPLIT_WORDS or tag in ("WDT", "WP"):
            return i
    return None


def _noun_phrase(tagged, owner, a, b, chunker):
    """First noun phrase in tokens[a:b] that is not already a typed item."""
    piece = [(w, "X" if owner[i] is not None else t) for i, (w, t) in enumerate(tagged[a:b], start=a)]
    if not piece:
        return None
    for np in chunker.parse(piece).subtrees(lambda t: t.label() == "NP"):
        words = [(w, t) for w, t in np.leaves()]
        while words and words[0][1] in ("DT", "PRP$"):
            words = words[1:]
        if words and any(t.startswith("NN") for _, t in words):
            return " ".join(w for w, _ in words)
    return None


def _empty_fact(sentence):
    return {"sentence": sentence, "who": [], "action": "", "what": [], "where": [], "when": [],
            "how_much": [], "contact": [], "affiliation": []}


def _add(fact, slot, value):
    if value and value not in fact[slot]:
        fact[slot].append(value)


def analyse_sentence(sentence, tokens, tagged, entities, matches, memory, chunker=None):
    """Return (facts, relations, typed_items) for one sentence."""
    if chunker is None:
        from extractor import CHUNKER as chunker
    spans, owner = _typed_spans(sentence, tagged, entities, matches)
    placed = [s for s in spans if s["start"] >= 0]
    n = len(tagged)
    words = [w for w, _ in tagged]
    groups = _verb_groups(tagged, owner)

    def spans_in(a, b):
        return [s for s in placed if a <= s["start"] < b]

    facts, relations = [], []
    if not groups:                                            # "Venue: MG Road, Bengaluru 560001."
        fact = _empty_fact(sentence)
        for s in placed:
            _add(fact, "who" if s["type"] in DOERS else SLOT_OF.get(s["type"], "what"), s["text"])
        if ":" in words and words.index(":") <= 3 and not spans_in(0, words.index(":")):
            _add(fact, "what", " ".join(words[:words.index(":")]))
        if any(fact[k] for k in FACT_SLOTS if k != "action"):
            facts.append(fact)
    else:
        # Clause boundaries: subject region and argument region of every verb group.
        bounds = []
        for k, (gs, ge) in enumerate(groups):
            if k == 0:
                subj_start, has_own_subject = 0, True
            else:
                m = _last_split(tagged, groups[k - 1][1], gs)
                subj_start, has_own_subject = (m + 1, True) if m is not None else (gs, False)
            bounds.append([subj_start, gs, ge, n, has_own_subject])
        for k in range(len(groups) - 1):
            nxt = bounds[k + 1]
            m = _last_split(tagged, groups[k][1], groups[k + 1][0])
            bounds[k][3] = m if m is not None else nxt[1]

        prev_who = []
        for (ss, gs, ge, ae, own), (g0, g1) in zip(bounds, groups):
            fact = _empty_fact(sentence)
            fact["action"] = " ".join(words[g0:g1])
            # Subject region. Skip a greeting such as "Dear Arjun,".
            if ss < gs and words[ss].lower() in GREETINGS and "," in words[ss:gs]:
                ss = words.index(",", ss) + 1
            subject = spans_in(ss, gs)
            for s in subject:
                # "Rahul Sharma from Infosys", "director of AIIMS": the organisation is where the
                # person belongs, not a second doer.
                affiliated = (s["type"] == "ORGANIZATION" and s["start"] > 0
                              and words[s["start"] - 1].lower() in AFFILIATION_WORDS
                              and any(p["type"] == "PERSON" and p["start"] < s["start"] for p in subject))
                if affiliated:
                    _add(fact, "affiliation", s["text"])
                else:
                    _add(fact, "who" if s["type"] in DOERS else SLOT_OF.get(s["type"], "what"), s["text"])
            lowered = [w.lower() for w in words[ss:gs]]
            if not fact["who"]:
                if any(w in HE_SHE for w in lowered) and memory.people:
                    fact["who"] = [memory.people[-1]]
                elif any(w in THEY for w in lowered) and memory.doers:
                    fact["who"] = list(memory.doers)
                elif not any(t.startswith(("NN", "PRP")) for _, t in tagged[ss:gs]):
                    fact["who"] = list(prev_who)               # "...and grew up in Vrindavan"
                else:
                    np = _noun_phrase(tagged, owner, ss, gs, chunker)
                    _add(fact, "what", np)                     # "Students must register"

            # Argument region.
            args = spans_in(g1, ae)
            passive = (any(w.lower() in BE_WORDS for w in words[g0:g1])
                       and tagged[g1 - 1][1] == "VBN")
            agent = None
            if passive:
                for s in args:
                    before = s["start"] - 1
                    while before > g1 and tagged[before][1] in ("DT", "JJ"):
                        before -= 1                            # "by the Indian Council..."
                    if s["type"] in DOERS and before >= 0 and words[before].lower() == "by":
                        agent = s
                        break
            if agent:                                          # "funded by ICMR" -> who = ICMR
                for name in fact["who"]:
                    _add(fact, "what", name)
                fact["who"] = [agent["text"]]
            objects = []
            for s in args:
                if s is agent:
                    continue
                slot = "what" if s["type"] in DOERS else SLOT_OF.get(s["type"], "what")
                _add(fact, slot, s["text"])
                if s["type"] in DOERS or s["type"] == "PLACE" and s["label"] != "pin_code":
                    objects.append(s)
            if not any(s["type"] in DOERS for s in args if s is not agent):
                np = _noun_phrase(tagged, owner, g1, ae, chunker)
                if np:
                    _add(fact, "what", np)
                    objects.append({"text": np, "type": "THING", "start": None})
            facts.append(fact)

            # Relations from this fact: (who, action + preposition, object).
            for who in fact["who"][:3]:
                for obj in objects[:4]:
                    rel = fact["action"]
                    st = obj.get("start")
                    if st and st > g1 and tagged[st - 1][1] in ("IN", "TO"):
                        rel += " " + words[st - 1]
                    relations.append({"subject": who, "relation": rel, "object": obj["text"],
                                      "object_type": obj["type"], "method": "verb pattern"})

            if fact["who"]:
                prev_who = fact["who"]
                memory.doers = fact["who"]
                people = [s["text"] for s in placed if s["type"] == "PERSON" and s["text"] in fact["who"]]
                if people:
                    memory.people = people
            else:
                prev_who = []

    relations += _nltk_relations(words, entities)
    items = [{"text": s["text"], "type": s["type"], "label": s["label"], "source": s["source"]}
             for s in spans if s["type"] not in ("QUANTITY", "DURATION")]
    items += _things(tagged, owner, chunker)
    return facts, relations, items


def _things(tagged, owner, chunker):
    """Noun phrases that are not names, dates or numbers: the THINGs of the sentence."""
    piece = [(w, "X" if owner[i] is not None else t) for i, (w, t) in enumerate(tagged)]
    out = []
    for np in chunker.parse(piece).subtrees(lambda t: t.label() == "NP"):
        words = list(np.leaves())
        while words and words[0][1] in ("DT", "PRP$"):
            words = words[1:]
        if words and any(t.startswith("NN") for _, t in words) and not all(w.isupper() and len(w) <= 3 for w, _ in words):
            out.append({"text": " ".join(w for w, _ in words), "type": "THING", "label": "noun_phrase",
                        "source": "chunker"})
    return out


def _nltk_relations(words, entities):
    """Relations found by NLTK's extract_rels between neighbouring entities."""
    by_start = {e["start"]: e for e in entities if e["label"] in ACE_LABEL}
    children, i = [], 0
    while i < len(words):
        ent = by_start.get(i)
        if ent and ent["end"] > i:
            children.append(Tree(ACE_LABEL[ent["label"]], words[ent["start"]:ent["end"]]))
            i = ent["end"]
        else:
            children.append(words[i])
            i += 1
    # extract_rels only reads a pair when a third chunk follows it, so add an end marker.
    tree = Tree("S", children + [Tree("END", ["."])])
    out = []
    for subj, obj, pattern in REL_RULES:
        try:
            found = extract_rels(subj, obj, tree, corpus="ace", pattern=pattern, window=5)
        except Exception:                                    # never let relations break extraction
            continue
        for rel in found:
            filler = rel["filler"].strip()
            if filler:
                out.append({"subject": rel["subjtext"], "relation": filler, "object": rel["objtext"],
                            "object_type": "PLACE" if obj in ("GPE", "LOC") else "ORGANIZATION",
                            "method": "nltk.extract_rels"})
    return out


def unique_items(items, max_things=20):
    seen, out, things = set(), [], 0
    for it in items:
        key = (it["text"].lower(), it["type"])
        if key in seen:
            continue
        if it["type"] == "THING":
            if things >= max_things or any(it["text"].lower() == o["text"].lower() for o in out):
                continue
            things += 1
        seen.add(key)
        out.append(it)
    return out


def unique_relations(relations):
    seen, out = set(), []
    for r in relations:
        key = (r["subject"].lower(), r["relation"].lower(), r["object"].lower())
        if key not in seen and r["subject"].lower() != r["object"].lower():
            seen.add(key)
            out.append(r)
    return out
