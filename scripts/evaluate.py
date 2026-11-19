"""Measure the extractor against the hand-labelled samples.

    python scripts/evaluate.py                  # both sets, NLTK-only vs NLTK + rules
    python scripts/evaluate.py --set unseen     # only the 10 unseen samples
    python scripts/evaluate.py --errors         # also list every missed and extra item
    python scripts/evaluate.py --markdown       # Markdown tables (for the README)
    python scripts/evaluate.py --smart          # also score NLTK + AI (needs LLM_API_KEY)

Sample sets
    tuning  samples/sample_XX      15 texts used while writing the rules (optimistic scores)
    unseen  samples/unseen/...     10 texts labelled BEFORE the fact rules and never used to tune

Systems
    nltk     NLTK ne_chunk only, no rule layer          (baseline)
    rules    NLTK + our rule layer (gazetteers, shapes)  (what the website uses)
    ai       rules + optional LLM check                  (only with --smart and a key)

Scores. For every type, over all samples of a set:
    TP = found and correct, FP = found but wrong, FN = expected but missed
    precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R)
Items match when the text is the same, ignoring case and extra spaces.
Facts are scored as (action, slot, value) triples, e.g. ("was born", "where", "mathura"),
so a fact only counts when the value is attached to the right action.
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
PLACE_LABELS = ["GPE", "LOCATION", "GSP", "FACILITY"]   # every place label counts as a place
FACT_SLOTS = ["who", "where", "when", "how_much"]
SETS = {"tuning": ROOT / "samples", "unseen": ROOT / "samples" / "unseen"}


def norm(text):
    return " ".join(str(text).split()).lower()


def items(result, kind):
    if kind in ENTITY_TYPES:
        labels = PLACE_LABELS if kind == "GPE" else [kind]
        values = [v for label in labels for v in result.get("entities", {}).get(label, [])]
    else:
        values = result.get(kind, [])
    return {norm(v) for v in values}


def fact_triples(facts):
    out = set()
    for f in facts or []:
        action = norm(f.get("action", ""))
        out.add((action, "action", action))
        for slot in FACT_SLOTS:
            for v in f.get(slot, []):
                out.add((action, slot, norm(v)))
    return out


def scores(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = (2 * p * r / (p + r) if p + r else 0.0) if p is not None and r is not None else None
    return p, r, f1


def fmt(x):
    return "  -  " if x is None else f"{x:.2f}"


def evaluate(folder, system):
    kinds = ENTITY_TYPES + REGEX_TYPES + ["facts"]
    counts = {kind: [0, 0, 0] for kind in kinds}
    errors, has_facts, files = [], False, sorted(folder.glob("sample_*.txt"))
    for txt in files:
        gold = json.loads(txt.with_suffix(".json").read_text(encoding="utf-8"))
        text = txt.read_text(encoding="utf-8")
        pred = extract(text, use_rules=system != "nltk", smart=system == "ai")
        for kind in kinds:
            if kind == "facts":
                if "facts" not in gold:
                    continue
                has_facts = True
                g, p = fact_triples(gold["facts"]), fact_triples(pred["facts"])
            else:
                g, p = items(gold, kind), items(pred, kind)
            c = counts[kind]
            c[0] += len(g & p)
            c[1] += len(p - g)
            c[2] += len(g - p)
            for v in sorted(map(str, g - p)):
                errors.append(f"{txt.stem}  {kind:<12} missed  {v}")
            for v in sorted(map(str, p - g)):
                errors.append(f"{txt.stem}  {kind:<12} extra   {v}")
    if not has_facts:
        counts.pop("facts")
    return len(files), counts, errors


def summary_rows(counts):
    rows = [(kind, *c, *scores(*c)) for kind, c in counts.items()]
    groups = [("Entities (NER)", ENTITY_TYPES), ("Regex fields", REGEX_TYPES),
              ("Overall items", ENTITY_TYPES + REGEX_TYPES)]
    for name, kinds in groups:
        tp, fp, fn = (sum(counts[k][i] for k in kinds) for i in range(3))
        rows.append((name, tp, fp, fn, *scores(tp, fp, fn)))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Score the extractor on the labelled samples.")
    parser.add_argument("--set", choices=["tuning", "unseen", "both"], default="both")
    parser.add_argument("--errors", action="store_true", help="list missed and extra items")
    parser.add_argument("--markdown", action="store_true", help="print Markdown tables")
    parser.add_argument("--smart", action="store_true", help="also score NLTK + AI (needs a key)")
    args = parser.parse_args()

    systems = ["nltk", "rules"] + (["ai"] if args.smart else [])
    sets = ["tuning", "unseen"] if args.set == "both" else [args.set]
    compare = []
    for name in sets:
        for system in systems:
            n, counts, errors = evaluate(SETS[name], system)
            rows = summary_rows(counts)
            compare.append((name, system, {r[0]: r for r in rows}))
            title = f"{name} set ({n} samples), system = {system}"
            if args.markdown:
                print(f"\n**{title}**\n\n| Type | TP | FP | FN | Precision | Recall | F1 |\n|---|--:|--:|--:|--:|--:|--:|")
                for kind, tp, fp, fn, p, r, f1 in rows:
                    bold = "**" if kind[0].isupper() and kind not in ENTITY_TYPES else ""
                    print(f"| {bold}{kind}{bold} | {tp} | {fp} | {fn} | {fmt(p)} | {fmt(r)} | {fmt(f1)} |")
            else:
                print(f"\n== {title}")
                print(f"{'Type':<16}{'TP':>4}{'FP':>4}{'FN':>4}{'Precision':>11}{'Recall':>8}{'F1':>7}")
                for i, (kind, tp, fp, fn, p, r, f1) in enumerate(rows):
                    if i == len(counts):
                        print("-" * 54)
                    print(f"{kind:<16}{tp:>4}{fp:>4}{fn:>4}{fmt(p):>11}{fmt(r):>8}{fmt(f1):>7}")
            if args.errors and system != "nltk":
                print("\n".join(["Errors:"] + errors) if errors else "No errors.")

    print("\nF1 comparison" if not args.markdown else "\n**F1 comparison**\n")
    keys = ["Entities (NER)", "Regex fields", "Overall items", "facts"]
    header = "| Set | System | " + " | ".join(keys) + " |"
    print(header + "\n|" + "---|" * (len(keys) + 2))
    for name, system, rows in compare:
        cells = [fmt(rows[k][6]) if k in rows else "-" for k in keys]
        print(f"| {name} | {system} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # "₹" on Windows consoles
    main()
