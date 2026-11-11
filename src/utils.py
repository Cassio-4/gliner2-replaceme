"""Transformer Layer Analysis and Optimization Toolkit

This module provides utilities for analyzing, transforming, and evaluating transformer model layers.
Includes functionality for distance metrics, model truncation, activation analysis, and evaluation.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import datasets
import numpy as np
import torch
import torch.nn as nn
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

# --------------------------
# Seeding Function
# --------------------------
def seed_all(seed: int = 42):
    """Seed all major sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seeded with seed: {seed}")

# --------------------------
# Distance Metric Functions
# --------------------------
def angular_distance(x_l: torch.Tensor, x_l_plus_n: torch.Tensor) -> torch.Tensor:
    """Compute angular distance between normalized layer outputs."""
    x_l_norm = x_l / torch.norm(x_l, dim=-1, keepdim=True)
    x_l_plus_n_norm = x_l_plus_n / torch.norm(x_l_plus_n, dim=-1, keepdim=True)
    cosine_sim = (x_l_norm * x_l_plus_n_norm).sum(-1).clamp(min=-1, max=1)
    return torch.acos(cosine_sim) / torch.pi


# --------------------------
# Model Manipulation
# --------------------------

def truncate_model(model: nn.Module, start_layer: int, end_layer: int) -> nn.Module:
    """Truncate a DeBERTa-style encoder (e.g. GLiNER2's DebertaV2Model) by removing layers [start_layer, end_layer)."""
    model.config.num_hidden_layers -= (end_layer - start_layer)
    model.encoder.layer = nn.ModuleList([
        layer for idx, layer in enumerate(model.encoder.layer)
        if idx < start_layer or idx >= end_layer
    ])
    return model


# --------------------------
# Data Loading Utilities
# --------------------------

def get_calib_dataloader(
    dataset: str,
    dataset_subset: str,
    dataset_column: str,
    dataset_size: Optional[int],
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase
) -> DataLoader:
    """Load and prepare calibration dataset."""
    dataset_handlers = {
        'HuggingFaceFW/fineweb': lambda: datasets.load_dataset(dataset, name='sample-10BT', split=dataset_subset),
        'allenai/c4': lambda: datasets.load_dataset(dataset, 'en', split=dataset_subset),
        'arcee-ai/sec-data-mini': lambda: datasets.load_dataset(dataset, split=dataset_subset),
        'wikitext': lambda: datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split=dataset_subset),
        'Open-Orca/SlimOrca': lambda: _load_orca_dataset(dataset_size, tokenizer),
        'fineweb_and_orca': lambda: _load_mixed_dataset(dataset_size, dataset_subset, tokenizer)
    }

    if dataset not in dataset_handlers:
        raise ValueError(f"Dataset {dataset} not implemented")

    data = dataset_handlers[dataset]()
    if dataset_size:
        data = data.select(range(min(dataset_size, len(data))))
    
    return DataLoader(data[dataset_column], batch_size=batch_size, shuffle=False, drop_last=True)

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
# Dataset name -> (data dir, description module defining ENTITY_DESCRIPTIONS and ENTITY_NAMES)
LEGALNER_DATASETS = {
    'lener': (os.path.join(DATA_ROOT, 'lener'), 'lener_dataset.py'),
    'ulysses': (os.path.join(DATA_ROOT, 'ulyssesbr', 'pl_corpus_v2_categories'), 'ulysses_ner_br.py'),
}

def load_legalner_module(dataset: str):
    """Import the description module (DATA_DIR, SPLIT_FILES, ENTITY_DESCRIPTIONS, ENTITY_NAMES) of ``dataset``."""
    import importlib.util

    if dataset not in LEGALNER_DATASETS:
        raise ValueError(f"Dataset {dataset} not implemented; choose from {sorted(LEGALNER_DATASETS)}")
    data_dir, module_file = LEGALNER_DATASETS[dataset]
    spec = importlib.util.spec_from_file_location(f"{dataset}_description", os.path.join(data_dir, module_file))
    ds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds)
    return ds

def get_gliner2_calib_dataloader(
    extractor: nn.Module,
    dataset: str,
    dataset_subset: str,
    dataset_size: Optional[int],
    batch_size: int,
    max_len: Optional[int] = None,
) -> DataLoader:
    """Calibration batches built exactly as GLiNER2's ``batch_extract_entities`` builds them in production.

    Each text is paired with the dataset's entity schema (names + Portuguese descriptions) and
    collated by the model's own inference collator, so the encoder sees the same
    ``[P] ... [E] label ... [SEP_TEXT] text`` input it was fine-tuned on. Yields dicts holding the
    only two tensors GLiNER2 passes to its encoder, ``input_ids`` and ``attention_mask``, plus the
    routing indices/masks the head gathers from the encoder output (see
    ``get_text_and_query_indices_tokens``).
    """
    from gliner2.training.data import TrainingDataset
    from gliner2.training.trainer import ExtractorCollator

    data_dir = LEGALNER_DATASETS[dataset][0]
    ds = load_legalner_module(dataset)
    entity_types = {ds.ENTITY_NAMES[tag]: desc for tag, desc in ds.ENTITY_DESCRIPTIONS.items()}

    path = os.path.join(data_dir, f"{dataset_subset}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; dataset_subset must be train, dev or test")
    texts = [ex.text for ex in TrainingDataset.load(path)]
    if dataset_size:
        texts = texts[:dataset_size]

    # Same schema construction and collator as batch_extract_entities -> batch_extract
    schema = extractor.create_schema().entities(entity_types)
    schema_dicts, _ = extractor._build_schema_dicts_and_metadata([schema] * len(texts))
    extractor.processor.change_mode(is_training=False)
    collator = ExtractorCollator(
        extractor.processor, is_training=False, max_len=max_len, architecture=extractor.architecture
    )

    def collate(rows):
        batch = collator(rows)
        return {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "text_word_indices": batch.text_word_indices,
            "text_word_mask": batch.text_word_mask,
            "query_marker_indices": batch.query_marker_indices,
            "query_marker_mask": batch.query_marker_mask,
        }

    return DataLoader(
        list(zip(texts, schema_dicts)), batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate
    )

def _load_orca_dataset(size: int, tokenizer: PreTrainedTokenizerBase) -> datasets.Dataset:
    """Helper to load and format Orca dataset."""
    dd = datasets.load_dataset("Open-Orca/SlimOrca", split="train").select(range(size))
    processed = []
    
    for item in dd:
        idx = 0 if item['conversations'][0]["from"] == "human" else 1
        dialog = [
            {"role": "user", "content": item['conversations'][idx]['value']},
            {"role": "assistant", "content": item['conversations'][idx + 1]['value']},
        ]
        text = tokenizer.apply_chat_template(dialog, tokenize=False)
        processed.append({"text": text})
    
    return datasets.Dataset.from_list(processed)

def _load_mixed_dataset(size: int, subset: str, tokenizer: PreTrainedTokenizerBase) -> datasets.Dataset:
    """Helper to load mixed Orca and FineWeb dataset."""
    orca_data = _load_orca_dataset(size//2, tokenizer)
    fineweb = datasets.load_dataset(
        "HuggingFaceFW/fineweb", 
        name='sample-10BT', 
        split=subset
    ).select(range(size - size//2))
    return datasets.concatenate_datasets([orca_data, fineweb])


# --------------------------
# Activation Analysis
# --------------------------

def get_last_non_padded_tokens(
    hidden_states: List[torch.Tensor], 
    attention_mask: torch.Tensor
) -> List[torch.Tensor]:
    """Extract last non-padded tokens from each layer's hidden states."""
    return [
        torch.stack([
            layer[batch, mask.nonzero()[-1], :] 
            for batch, mask in enumerate(attention_mask)
        ])
        for layer in hidden_states
    ]


def get_text_and_query_indices_tokens(
    hidden_states: List[torch.Tensor],
    text_word_indices: torch.Tensor,
    text_word_mask: torch.Tensor,
    query_marker_indices: torch.Tensor,
    query_marker_mask: torch.Tensor,
) -> List[torch.Tensor]:
    """Extract, from each layer, the token states GLiNER2's head actually reads.

    Mirrors the ``token_pooling="first"`` routing in GLiNER2's BoundaryExtractor: the first
    sub-token of every text word and the schema's [E] marker tokens. Padded slots are
    dropped, so each layer yields a flat (num_valid_tokens, hidden) tensor across the batch.
    """
    positions = torch.cat([text_word_indices, query_marker_indices], dim=1)
    valid = torch.cat([text_word_mask, query_marker_mask], dim=1).bool()
    rows = torch.arange(positions.size(0), device=positions.device).unsqueeze(1).expand_as(positions)
    rows, positions = rows[valid], positions[valid]
    return [layer[rows, positions, :] for layer in hidden_states]


# --------------------------
# Block Analysis Utilities
# --------------------------

def compute_block_distances(
    hidden_states: List[torch.Tensor], 
    layers_to_skip: int,
    distance_fn=angular_distance
) -> List[float]:
    """Compute distances between layer blocks using specified metric."""
    return [
        distance_fn(hidden_states[l], hidden_states[l + layers_to_skip]).mean().item()
        for l in range(len(hidden_states) - layers_to_skip)
    ]


# --------------------------
# Optimization Methods
# --------------------------

class LowerTriangularLinear(nn.Module):
    """Linear layer with lower triangular weight matrix."""
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        identity = torch.eye(min(input_size, output_size))
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.weight.data[:identity.size(0), :identity.size(1)] = identity
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ torch.tril(self.weight).t()

def optimizing_method(
    a1: torch.Tensor, 
    a2: torch.Tensor,
    a3: torch.Tensor = None,
    solver: str = "cg"
) -> torch.Tensor:
    """Optimize transformation matrix using specified solver."""
    class ActivationDataset(Dataset):
        def __init__(self, a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor):
            self.a1, self.a2, self.a3 = a1, a2, a3            
        def __len__(self) -> int:
            return len(self.a1)
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            attn = [-1]
            if self.a3 is not None:
                attn = self.a3[idx]                
            return self.a1[idx], self.a2[idx], attn

    def cosine_loss(XA: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        XA_norm = XA / XA.norm(dim=1, keepdim=True)
        Y_norm = Y / Y.norm(dim=1, keepdim=True)
        return 1 - (XA_norm * Y_norm).sum(dim=1).mean()

    # Imported lazily: pytorch-minimize is only needed by the cosine_dist cg/l-bfgs/trust-ncg solvers
    from torchmin import minimize

    dataset = ActivationDataset(a1, a2, a3)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    A = torch.eye(a1.shape[1], requires_grad=True, device="cuda")

    with tqdm(range(10), desc="Estimating Transformation") as pbar:
        for _ in pbar:
            for X, Y, Z in loader:
                def objective(A_tensor):
                    XA = X.float().to("cuda") @ A_tensor
                    if len(Z) != 1:
                        XA += Z.float().to("cuda")
                    return cosine_loss(XA, Y.float().to("cuda"))
                
                result = minimize(objective, A, method=solver, max_iter=1)
                pbar.set_postfix({'Loss': colored(f'{result.fun:.4f}', 'green')})
                A = result.x
    
    return A.to(torch.float64)

def adam_method(
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor = None,
    loss: str = "cosine",
    diag: bool = False,
    two_vectors: bool = False,
    thri: bool = False
) -> torch.Tensor:
    """Optimize transformation using Adam optimizer."""
    class ActivationDataset(Dataset):
        def __init__(self, a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor):
            self.a1, self.a2, self.a3 = a1, a2, a3            
        def __len__(self) -> int:
            return len(self.a1)
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            attn = [-1]
            if self.a3 is not None:
                attn = self.a3[idx]
            return self.a1[idx], self.a2[idx], attn

    # Initialize model and optimizer
    if diag:
        transform = torch.ones(a1.shape[1], requires_grad=True, device="cuda")
        optimizer = torch.optim.Adam([transform], lr=1e-4)
    elif two_vectors:
        t1 = torch.ones((a1.shape[1], 1), requires_grad=True, device="cuda")
        t2 = torch.ones((a1.shape[1], 1), requires_grad=True, device="cuda")
        optimizer = torch.optim.Adam([t1, t2], lr=1e-4)
    else:
        model = LowerTriangularLinear(a1.shape[1], a1.shape[1]).to("cuda") if thri \
               else nn.Linear(a1.shape[1], a1.shape[1], bias=False).to("cuda")
        if not thri:
            model.weight.data.copy_(torch.eye(a1.shape[1]))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Define loss functions
    def cosine_loss(XA: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        XA_norm = XA / XA.norm(dim=1, keepdim=True)
        Y_norm = Y / Y.norm(dim=1, keepdim=True)
        return 1 - (XA_norm * Y_norm).sum(dim=1).mean()

    loss_fn = {
        "cosine": cosine_loss,
        "mse": nn.MSELoss(reduction='mean'),
        "elasticnet": lambda XA, Y: nn.MSELoss(reduction='mean')(XA, Y) + \
                                   0.09 * torch.norm(XA, p=1) + \
                                   0.045 * torch.norm(XA, p=2)**2
    }[loss]

    # Training loop
    dataset = ActivationDataset(a1, a2, a3)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    with tqdm(range(10), desc="Optimizing Transformation") as pbar:
        for _ in pbar:
            for X, Y, Z in loader:
                optimizer.zero_grad()
                
                if diag:
                    XA = X.float().to("cuda") @ torch.diag(transform)
                elif two_vectors:
                    XA = X.float().to("cuda") @ (t1 @ t2.T)
                else:
                    XA = model(X.float().to("cuda"))
                if len(Z) != 1:
                    XA += Z.float().to("cuda")
                loss_val = loss_fn(XA, Y.float().to("cuda"))
                loss_val.backward()
                optimizer.step()
                
                pbar.set_postfix({f'{loss} Loss': colored(f'{loss_val.item():.4f}', 'green')})

    # Return appropriate transformation
    if diag:
        return torch.diag(transform).to(torch.float64)
    elif two_vectors:
        return (t1 @ t2.T).to(torch.float64)
    return model.triangular_weight.T.to(torch.float64) if thri else model.weight.T.to(torch.float64)


# --------------------------
# Evaluation Utilities
# --------------------------

def parse_conll(path: str) -> List[Tuple[List[str], List[str]]]:
    """Parse a whitespace-separated CoNLL file into (tokens, tags) sentences."""
    sentences, tokens, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            tokens.append(parts[0])
            tags.append(parts[-1])
    if tokens:
        sentences.append((tokens, tags))
    return sentences

def bio_to_token_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    """Convert BIO tags to half-open (start, end, label) token spans; a stray I- tag starts a new span."""
    spans, start, label = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag.startswith("B-") or (tag.startswith("I-") and (start is None or tag[2:] != label)):
            if start is not None:
                spans.append((start, i, label))
            start, label = i, tag[2:]
        elif not tag.startswith("I-"):
            if start is not None:
                spans.append((start, i, label))
            start, label = None, None
    return spans

def load_ner_split(ds, split: str, limit: Optional[int] = None) -> Tuple[List[str], List[List[Tuple[int, int, str]]]]:
    """Space-join each CoNLL sentence and map its gold BIO spans to character offsets."""
    texts, golds = [], []
    for tokens, tags in parse_conll(os.path.join(ds.DATA_DIR, ds.SPLIT_FILES[split]))[:limit]:
        offsets, cursor = [], 0
        for tok in tokens:
            offsets.append((cursor, cursor + len(tok)))
            cursor += len(tok) + 1
        texts.append(" ".join(tokens))
        golds.append([(offsets[s][0], offsets[e - 1][1], label) for s, e, label in bio_to_token_spans(tags)])
    return texts, golds

def eval_model(
    model_path: str,
    dataset: str,
    eval_subset: str = "test",
    threshold: float = 0.5,
    batch_size: int = 16,
    eval_size: Optional[int] = None,
) -> Dict:
    """Exact-span precision/recall/F1 of a GLiNER2 model on a LeNER-Br / UlyssesNER-Br CoNLL split.

    Port of legalner-gliner2's ``eval_legalner_br.evaluate``: the model is queried through
    ``batch_extract_entities`` with the dataset's entity names and descriptions.
    """
    from gliner2 import AutoExtractor
    from gliner2.training.metrics import f1_from_counts

    if os.path.isdir(model_path):
        result_path = os.path.join(model_path, f'benchmark_{dataset}_{eval_subset}.json')
    else:
        # A Hub model id has no local experiment directory to save into
        os.makedirs('benchmark_results', exist_ok=True)
        result_path = f'benchmark_results/{os.path.basename(os.path.normpath(model_path))}_{dataset}_{eval_subset}.json'
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    ds = load_legalner_module(dataset)
    labels = list(ds.ENTITY_DESCRIPTIONS)
    entity_types = {ds.ENTITY_NAMES[label]: desc for label, desc in ds.ENTITY_DESCRIPTIONS.items()}
    texts, golds = load_ner_split(ds, eval_subset, limit=eval_size)

    extractor = AutoExtractor.from_pretrained(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    extractor.eval()
    preds = extractor.batch_extract_entities(
        texts, entity_types, batch_size=batch_size, threshold=threshold, include_spans=True
    )

    tp, fp, fn = ({label: 0 for label in labels} for _ in range(3))
    for gold, pred in zip(golds, preds):
        entities = pred.get("entities", {})
        for label in labels:
            g = {(s, e) for s, e, l in gold if l == label}
            p = {(item["start"], item["end"]) for item in entities.get(ds.ENTITY_NAMES[label], []) if isinstance(item, dict)}
            tp[label] += len(g & p)
            fp[label] += len(p - g)
            fn[label] += len(g - p)

    per_entity = {}
    for label in labels:
        p, r, f1 = f1_from_counts(tp[label], fp[label], fn[label])
        per_entity[label] = {"precision": p, "recall": r, "f1": f1, "support": tp[label] + fn[label]}
    micro_p, micro_r, micro_f1 = f1_from_counts(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    results = {
        "model": model_path,
        "dataset": dataset,
        "split": eval_subset,
        "threshold": threshold,
        "n_sentences": len(texts),
        "per_entity": per_entity,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {k: sum(m[k] for m in per_entity.values()) / len(labels) for k in ("precision", "recall", "f1")},
    }

    for label, m in per_entity.items():
        print(colored(f"{label:<16}P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}", "cyan"))
    print(colored(f"{'MICRO AVG':<16}P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f}", "green"))
    print(colored(f"{'MACRO AVG':<16}F1={results['macro']['f1']:.3f}", "green"))

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    del extractor
    torch.cuda.empty_cache()
    return results


# --------------------------
# Block Selection Utilities
# --------------------------

def select_non_overlapping_blocks(
    average_distances: List[float],
    layers_to_skip: int,
    num_blocks: int = 4,
    merge_consecutive: bool = False
) -> List[Tuple[int, int]]:
    """Select optimal non-overlapping layer blocks based on distances."""
    blocks = [
        (i + 1, i + layers_to_skip + 1, avg)
        for i, avg in enumerate(average_distances)
    ]
    
    # Sort by distance and select non-overlapping
    selected = []
    used_layers = set()
    
    for block in sorted(blocks, key=lambda x: x[2]):
        start, end, _ = block
        block_layers = set(range(start, end))
        
        if not block_layers & used_layers:
            selected.append(block)
            used_layers.update(block_layers)
            if len(selected) >= num_blocks:
                break
    
    # Merge consecutive blocks if requested
    if merge_consecutive and selected:
        selected.sort()
        merged = []
        current_start, current_end, _ = selected[0]
        
        for start, end, _ in selected[1:]:
            if start == current_end:
                current_end = end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        print(f"List of layers to prune {merged}")
        return merged
    selected = [(start, end) for start, end, _ in selected]
    print(f"List of layers to prune {selected}")
    return selected