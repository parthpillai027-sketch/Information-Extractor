"""Download the NLTK data this project needs, then check that it loads.

    python setup_nltk.py

Data goes into ./nltk_data, or into $NLTK_DATA if that is set (Render sets it).
app/extractor.py looks in ./nltk_data automatically.
Exits with an error if anything is missing, so a broken deploy build fails early.
"""
import os
import sys
from pathlib import Path

import nltk

TARGET = Path(os.environ.get("NLTK_DATA") or Path(__file__).resolve().parent / "nltk_data")

PACKAGES = [
    "punkt_tab",                       # sentence tokenizer (Punkt)
    "averaged_perceptron_tagger_eng",  # POS tagger
    "maxent_ne_chunker_tab",           # named-entity chunker
    "words",                           # word list used by the NE chunker and our NER rules
    "stopwords",                       # stopword list for keyword ranking
]


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    if str(TARGET) not in nltk.data.path:
        nltk.data.path.insert(0, str(TARGET))

    for pkg in PACKAGES:
        if not nltk.download(pkg, download_dir=str(TARGET), quiet=True):
            print(f"ERROR: could not download NLTK package '{pkg}'", file=sys.stderr)
            return 1

    # Run the whole pipeline once to prove every model loads.
    try:
        tagged = nltk.pos_tag(nltk.word_tokenize("Rahul Sharma lives in Delhi."))
        nltk.ne_chunk(tagged)
        nltk.sent_tokenize("One. Two.")
        nltk.corpus.stopwords.words("english")
    except LookupError as err:
        print(f"ERROR: NLTK data missing after download:\n{err}", file=sys.stderr)
        return 1

    print(f"NLTK data ready in {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
