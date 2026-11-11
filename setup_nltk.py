"""Run once after `pip install -r requirements.txt` to download NLTK data."""
import nltk

# Older and newer NLTK versions use different package names; downloading both covers either.
PACKAGES = [
    "punkt", "punkt_tab",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker", "maxent_ne_chunker_tab",
    "words", "stopwords",
]

for pkg in PACKAGES:
    nltk.download(pkg, quiet=True)
print("NLTK data ready.")
