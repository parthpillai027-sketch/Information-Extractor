"""NLP engine for Information-Extractor (Member 1).

Uses NLTK for sentence segmentation, tokenization, POS tagging, named-entity
recognition, noun-phrase chunking and keyword ranking, plus regex for
fixed-format items (emails, phones, URLs, dates, money).
Returns a dict that follows docs/json_contract.md.
"""
import re

import nltk
from nltk.corpus import stopwords

PATTERNS = {
    "emails": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    "phones": r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b",
    "urls": r"(?:https?://|www\.)[^\s]*[^\s.,;:)]",
    "dates": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    "money": r"(?:₹|Rs\.?|INR)\s?\d[\d,]*(?:\.\d+)?",
}

CHUNKER = nltk.RegexpParser("NP: {<DT>?<JJ>*<NN.*>+}")
STOP = set(stopwords.words("english"))


def extract(text: str) -> dict:
    text = (text or "").strip()
    result = {name: re.findall(p, text) for name, p in PATTERNS.items()}
    result.update(entities={}, noun_phrases=[], keywords=[], nlp_steps=[])
    if not text:
        return result

    words = []
    for sent in nltk.sent_tokenize(text):                  # 1. sentence segmentation
        tokens = nltk.word_tokenize(sent)                  # 2. tokenization
        tagged = nltk.pos_tag(tokens)                      # 3. POS tagging
        result["nlp_steps"].append({"sentence": sent, "tokens": tokens,
                                    "pos": [list(p) for p in tagged]})

        for node in nltk.ne_chunk(tagged):                 # 4. named-entity recognition
            if hasattr(node, "label"):
                name = " ".join(w for w, _ in node.leaves())
                result["entities"].setdefault(node.label(), []).append(name)

        for phrase in CHUNKER.parse(tagged).subtrees(lambda t: t.label() == "NP"):  # 5. NP chunking
            result["noun_phrases"].append(" ".join(w for w, _ in phrase.leaves()))

        words += [w.lower() for w in tokens if w.isalpha() and w.lower() not in STOP]

    result["keywords"] = [w for w, _ in nltk.FreqDist(words).most_common(8)]  # 6. keywords
    return result


if __name__ == "__main__":
    import json
    sample = ("Rahul Sharma from Infosys will visit Bengaluru on 15/11/2026. "
              "Call +91 9876543210 or mail rahul.s@infosys.com. Fee is Rs. 2,500.")
    print(json.dumps(extract(sample), indent=2, ensure_ascii=False))
