"""Convert LeNER-Br CoNLL (token TAG, blank-line separated) to GLiNER2 JSONL.

Usage: python conll_to_gliner2.py [lener_dir] [out_dir]  (both default to this folder)
"""
import sys
from collections import Counter
from pathlib import Path

from gliner2.training.data import InputExample, TrainingDataset

HERE = Path(__file__).resolve().parent

LABEL_MAP = {
    "PESSOA": "pessoa",
    "ORGANIZACAO": "organização",
    "LOCAL": "local",
    "TEMPO": "tempo",
    "LEGISLACAO": "legislação",
    "JURISPRUDENCIA": "jurisprudência",
}


def read_conll(path):
    sent = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    yield sent
                sent = []
                continue
            tok, tag = line.rsplit(" ", 1)
            sent.append((tok, tag))
    if sent:
        yield sent


def bio_to_spans(tags):
    """Return (start, end_exclusive, type). Orphan I- starts a new span."""
    spans, start, etype = [], None, None
    for i, tag in enumerate(tags + ["O"]):
        prefix, _, t = tag.partition("-")
        if prefix == "B" or prefix == "O" or (prefix == "I" and t != etype):
            if start is not None:
                spans.append((start, i, etype))
                start, etype = None, None
            if prefix in ("B", "I"):
                start, etype = i, t
    return spans


def convert(path, stats):
    examples = []
    for sent in read_conll(path):
        tokens = [t for t, _ in sent]
        tags = [g for _, g in sent]
        stats["orphan_I"] += sum(
            1 for i, g in enumerate(tags)
            if g.startswith("I-") and (i == 0 or tags[i - 1][2:] != g[2:])
        )
        text = " ".join(tokens)
        entities = {v: [] for v in LABEL_MAP.values()}  # empty lists = negatives
        seen_type = {}
        for s, e, t in bio_to_spans(tags):
            mention = " ".join(tokens[s:e])
            label = LABEL_MAP[t]
            if mention.lower() in seen_type and seen_type[mention.lower()] != label:
                stats["cross_type_collision"] += 1
            seen_type[mention.lower()] = label
            if mention not in entities[label]:
                entities[label].append(mention)
            else:
                stats["dup_mentions"] += 1
        stats["sentences"] += 1
        stats["max_tokens"] = max(stats["max_tokens"], len(tokens))
        stats["over_384_tokens"] += len(tokens) > 384
        stats["no_entities"] += not any(entities.values())
        examples.append(InputExample(text=text, entities=entities))
    return TrainingDataset(examples)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE
    for split in ("train", "dev", "test"):
        stats = Counter()
        ds = convert(src / f"{split}.conll", stats)
        ds.save(out / f"{split}.jsonl")  # validates (strict) before writing
        print(split, dict(stats))
    ds.print_stats()
