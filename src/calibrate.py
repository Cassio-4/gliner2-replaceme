"""Layer-redundancy calibration for GLiNER2 encoders (ReplaceMe step 1).

Runs GLiNER2-formatted (text, schema) batches through the encoder, collects
every layer's hidden states on non-pad tokens, and accumulates the mean cosine
distance between each pair of layers. ``hidden_states[k]`` is the output of
layer k (0 = embeddings), so a block (i, n) means replacing layers i+1..i+n
with a linear map from ``hidden_states[i]`` to ``hidden_states[i+n]``.

Usage:
    python src/calibrate.py --num-samples 2000 --out results/calib
    python src/calibrate.py --data my_calib.jsonl   # {"text": ..., "schema": {...}} per line
"""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from gliner2 import AutoExtractor
from gliner2.inference.schema import Schema

# Generic label pools for synthetic schemas when no in-domain data is given.
ENTITY_LABELS = [
    "person", "organization", "location", "date", "event", "product",
    "country", "city", "company", "money", "time", "law", "language",
    "work of art", "facility", "nationality", "quantity", "disease",
]
RELATION_LABELS = ["works_for", "located_in", "founded_by", "born_in", "part_of"]
CLASSIFICATIONS = {
    "sentiment": ["positive", "negative", "neutral"],
    "topic": ["politics", "sports", "science", "business", "arts", "history"],
    "formality": ["formal", "informal"],
}


def random_schema(rng: random.Random) -> Schema:
    """Sample a mixed-task schema via the builder, so the dict layout matches inference."""
    schema = Schema().entities(rng.sample(ENTITY_LABELS, rng.randint(2, 8)))
    if rng.random() < 0.3:
        schema = schema.relations(rng.sample(RELATION_LABELS, rng.randint(1, 3)))
    if rng.random() < 0.4:
        task = rng.choice(list(CLASSIFICATIONS))
        schema = schema.classification(task, CLASSIFICATIONS[task])
    return schema


def load_records(args, rng: random.Random) -> list:
    if args.data:
        with open(args.data) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows = rows[: args.num_samples]
        return [(r["text"], r["schema"]) for r in rows]

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    records = []
    for row in ds:
        text = row["text"].strip()
        # Skip headings and fragments; keep paragraph-sized passages.
        if len(text.split()) < 30 or text.startswith("="):
            continue
        records.append((text, random_schema(rng)))
        if len(records) >= args.num_samples:
            break
    return records


@torch.no_grad()
def accumulate_distances(model, records, args, device) -> tuple:
    """Return (summed pairwise cosine similarity [L+1, L+1], token count)."""
    model_texts, model_schemas = zip(*records)
    schema_dicts, _ = model._build_schema_dicts_and_metadata(list(model_schemas))
    rows = list(zip(model_texts, schema_dicts))

    cos_sum = None
    n_tokens = 0
    for start in tqdm(range(0, len(rows), args.batch_size), desc="calibrating"):
        batch = model.processor.collate_fn_inference(
            rows[start:start + args.batch_size],
            max_len=args.max_len,
            error_policy="raise",
            architecture=model.architecture,
        ).to(device)
        out = model.encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
        )
        mask = batch.attention_mask.bool()
        # [L+1, T, H] over real tokens only, unit-normalized in fp32.
        states = torch.stack([h[mask] for h in out.hidden_states]).float()
        states = torch.nn.functional.normalize(states, dim=-1)
        sims = torch.einsum("itd,jtd->ij", states, states)
        cos_sum = sims if cos_sum is None else cos_sum + sims
        n_tokens += states.shape[1]
    return cos_sum, n_tokens


def best_blocks(dist: torch.Tensor, max_n: int, allow_last: bool) -> list:
    """For each block length n, the start i minimizing distance(h_i, h_{i+n})."""
    num_layers = dist.shape[0] - 1
    last_end = num_layers if allow_last else num_layers - 1
    blocks = []
    for n in range(1, max_n + 1):
        candidates = [(dist[i, i + n].item(), i) for i in range(0, last_end - n + 1)]
        if not candidates:
            break
        d, i = min(candidates)
        blocks.append({"n": n, "start": i, "removed_layers": list(range(i + 1, i + n + 1)), "distance": d})
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="fastino/gliner2-multi-v1")
    parser.add_argument("--data", help="JSONL with {'text', 'schema'} rows; defaults to wikitext + random schemas")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=384, help="max words per text")
    parser.add_argument("--max-block", type=int, default=6, help="largest block length n to report")
    parser.add_argument("--allow-last", action="store_true", help="allow blocks that remove the final layer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/calib")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoExtractor.from_pretrained(args.model).to(device).eval()
    num_layers = len(model.encoder.encoder.layer)
    print(f"{type(model.encoder).__name__}: {num_layers} layers, hidden={model.encoder.config.hidden_size}")

    records = load_records(args, rng)
    print(f"{len(records)} calibration records")

    cos_sum, n_tokens = accumulate_distances(model, records, args, device)
    dist = 1.0 - cos_sum / n_tokens
    blocks = best_blocks(dist, args.max_block, args.allow_last)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"distance": dist.cpu(), "n_tokens": n_tokens, "args": vars(args)}, out / "distances.pt")
    with open(out / "blocks.json", "w") as f:
        json.dump({"model": args.model, "num_layers": num_layers, "n_tokens": n_tokens, "blocks": blocks}, f, indent=2)

    print(f"\n{n_tokens} tokens. Adjacent-layer distance d(h_k, h_k+1):")
    for k in range(num_layers):
        print(f"  layer {k + 1:2d}: {dist[k, k + 1]:.4f}")
    print("\nBest block per length:")
    for b in blocks:
        print(f"  n={b['n']}  remove layers {b['removed_layers']}  d(h_{b['start']}, h_{b['start'] + b['n']})={b['distance']:.4f}")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
