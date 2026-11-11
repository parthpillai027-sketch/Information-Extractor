"""Measure the extractor against the hand-labelled samples.

    python scripts/evaluate.py              # precision / recall / F1 per type
    python scripts/evaluate.py --errors     # also list every missed and extra item
    python scripts/evaluate.py --markdown   # print the table as Markdown (for the README)

Each samples/sample_XX.txt has a sample_XX.json with the expected ("gold") answer.
For every type we count, over all samples:
    TP = items found that are in the gold answer
    FP = items found that are not in the gold answer
    FN = gold items that were not found
    precision = TP / (TP + FP)   - of what we found, how much was right
    recall    = TP / (TP + FN)   - of what was there, how much we found
    F1        = 2PR / (P + R)    - harmonic mean of the two
Items match when they are the same text, ignoring case and extra spaces.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from extractor import extract  # noqa: E402

ENTITY_TYPES = ["PERSON", "ORGANIZATION", "GPE"]
REGEX_TYPES = ["emails", "phones", "urls", "dates", "times", "money", "percentages", "pin_codes"]
# NLTK may call a place GPE, LOCATION or GSP; all count as a place.
PLACE_LABELS = ["GPE", "LOCATION", "GSP"]


def norm(text):
    return " ".join(str(text).split()).lower()


def items(result, kind):
    if kind in ENTITY_TYPES:
        labels = PLACE_LABELS if kind == "GPE" else [kind]
        values = [v for label in labels for v in result.get("entities", {}).get(label, [])]
    else:
        values = result.get(kind, [])
    return {norm(v) for v in values}


def ratio(num, den):
    return num / den if den else None


def scores(tp, fp, fn):
    p, r = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r else (
        0.0 if p is not None and r is not None else None)
    return p, r, f1


def fmt(x):
    return "  -  " if x is None else f"{x:.2f}"


def evaluate(samples_dir):
    counts = {kind: [0, 0, 0] for kind in ENTITY_TYPES + REGEX_TYPES}
    errors = []
    files = sorted(samples_dir.glob("sample_*.txt"))
    for txt in files:
        gold = json.loads(txt.with_suffix(".json").read_text(encoding="utf-8"))
        pred = extract(txt.read_text(encoding="utf-8"))
        for kind, c in counts.items():
            g, p = items(gold, kind), items(pred, kind)
            c[0] += len(g & p)
            c[1] += len(p - g)
            c[2] += len(g - p)
            for v in sorted(g - p):
                errors.append(f"{txt.stem}  {kind:<12} missed  {v}")
            for v in sorted(p - g):
                errors.append(f"{txt.stem}  {kind:<12} extra   {v}")
    return len(files), counts, errors


def rows(counts):
    out = [(kind, *c, *scores(*c)) for kind, c in counts.items()]
    for name, kinds in (("Entities (NER)", ENTITY_TYPES), ("Regex fields", REGEX_TYPES),
                        ("Overall (micro)", ENTITY_TYPES + REGEX_TYPES)):
        tp, fp, fn = (sum(counts[k][i] for k in kinds) for i in range(3))
        out.append((name, tp, fp, fn, *scores(tp, fp, fn)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=Path, default=ROOT / "samples")
    parser.add_argument("--errors", action="store_true", help="list missed and extra items")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = parser.parse_args()

    n, counts, errors = evaluate(args.samples)
    table = rows(counts)
    if args.markdown:
        print(f"| Type | TP | FP | FN | Precision | Recall | F1 |\n|---|--:|--:|--:|--:|--:|--:|")
        for kind, tp, fp, fn, p, r, f1 in table:
            bold = "**" if kind[0].isupper() and kind not in ENTITY_TYPES else ""
            print(f"| {bold}{kind}{bold} | {tp} | {fp} | {fn} | {fmt(p)} | {fmt(r)} | {fmt(f1)} |")
    else:
        print(f"Evaluated {n} samples from {args.samples}\n")
        print(f"{'Type':<16}{'TP':>4}{'FP':>4}{'FN':>4}{'Precision':>11}{'Recall':>8}{'F1':>7}")
        for i, (kind, tp, fp, fn, p, r, f1) in enumerate(table):
            if i == len(counts):
                print("-" * 54)
            print(f"{kind:<16}{tp:>4}{fp:>4}{fn:>4}{fmt(p):>11}{fmt(r):>8}{fmt(f1):>7}")
    if args.errors:
        print("\nErrors:" if errors else "\nNo errors.")
        print("\n".join(errors))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # "₹" on Windows consoles
    main()
