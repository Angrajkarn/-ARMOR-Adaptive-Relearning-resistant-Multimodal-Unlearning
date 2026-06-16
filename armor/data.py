"""
armor/data.py
=============
TOFU benchmark dataset loader with rephrase augmentation.

TOFU (Task Of Fictitious Unlearning) — maini/tofu on HuggingFace:
  • Forget set : fictional author bios + QA pairs the model must forget
  • Retain set : real-world QA pairs the model must keep

Rephrase augmentation (for relearning-resistant training):
  We generate N surface-form variants of each forget question so the
  unlearning gradient covers the full rephrasing neighbourhood — not just
  the original phrasing. This implements the "rephrasing-invariant gradient
  ascent" component of ARMOR.

Cross-modal NOTE (Step 2):
  When extending to LLaVA, this file will be extended with a
  MultimodalTOFUDataset that pairs images with these text samples.
  Keep the text-only interface unchanged for backward compatibility.
"""

import random
import re
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from armor.config import ARMORConfig


# ──────────────────────────────────────────────────────────────────────────────
# Rephrase templates
# Simple syntactic paraphrase rules that don't require an LLM or API call.
# For production ARMOR, replace with a paraphrase model (e.g. T5-paraphrase).
# ──────────────────────────────────────────────────────────────────────────────
REPHRASE_TEMPLATES = [
    # Prefix swaps
    lambda q: f"Can you tell me {q[0].lower() + q[1:] if q else q}",
    lambda q: f"What do you know about {_extract_subject(q)}?",
    lambda q: f"Please describe {_extract_subject(q)}.",
    lambda q: f"I'd like to know: {q}",
    lambda q: f"Explain the following: {q}",
    # Passive voice approximation
    lambda q: re.sub(r"^(Who|What|Where|When|Why|How)\s+", "Tell me ", q),
]


def _extract_subject(question: str) -> str:
    """
    Naively extract a subject phrase from a question for rephrase templates.
    e.g. "What is the name of X?" → "X"
    Falls back to the full question.
    """
    # Strip leading wh-word + verb
    match = re.search(r"(?:is|are|was|were|did|does|do)\s+(.+?)(?:\?|$)", question)
    if match:
        return match.group(1).strip().rstrip("?")
    return question.rstrip("?")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ──────────────────────────────────────────────────────────────────────────────

class TOFUSample:
    """
    Lightweight container for a single TOFU sample.

    Fields
    ------
    question  : str  — The original question
    answer    : str  — Ground-truth answer
    rephrases : list[str] — Paraphrased questions (for rephrase-invariant GA)
    split     : str  — 'forget' or 'retain'
    """
    __slots__ = ("question", "answer", "rephrases", "split")

    def __init__(self, question: str, answer: str,
                 rephrases: list[str], split: str):
        self.question  = question
        self.answer    = answer
        self.rephrases = rephrases
        self.split     = split

    def __repr__(self):
        return (f"TOFUSample(split={self.split!r}, "
                f"question={self.question[:60]!r}...)")


def _generate_rephrases(question: str, n: int, seed: int = 42) -> list[str]:
    """
    Generate n syntactic paraphrases of a question using template transforms.

    For the full ARMOR system, swap this with a neural paraphraser
    (e.g. Pegasus, T5-paraphrase) for more diverse coverage.
    """
    rng = random.Random(seed)
    templates = rng.sample(REPHRASE_TEMPLATES, k=min(n, len(REPHRASE_TEMPLATES)))
    rephrases = []
    for tmpl in templates:
        try:
            rephrased = tmpl(question)
            if rephrased != question:
                rephrases.append(rephrased)
        except Exception:
            pass
    # Pad with minor punctuation variants if we need more
    while len(rephrases) < n:
        variant = question.replace("?", " please?") if "?" in question else question + "?"
        rephrases.append(variant)
    return rephrases[:n]


def load_tofu_splits(cfg: ARMORConfig, verbose: bool = True):
    """
    Load the TOFU benchmark from HuggingFace and return forget/retain lists.

    Parameters
    ----------
    cfg     : ARMORConfig
    verbose : bool — Print dataset size summary

    Returns
    -------
    forget_samples : list[TOFUSample]
    retain_samples : list[TOFUSample]
    """
    if verbose:
        print(f"[data] Loading TOFU — forget={cfg.tofu_forget_split}, "
              f"retain={cfg.tofu_retain_split}")

    # Load both splits from HuggingFace
    # TOFU schema: {"question": str, "answer": str, ...}
    # Dataset: locuslab/TOFU  (authored by Pratyush Maini et al.)
    forget_hf = load_dataset("locuslab/TOFU", cfg.tofu_forget_split)["train"]
    retain_hf = load_dataset("locuslab/TOFU", cfg.tofu_retain_split)["train"]

    # In debug mode, trim to tiny subsets
    if cfg.debug:
        forget_hf = forget_hf.select(range(min(cfg.debug_n_samples, len(forget_hf))))
        retain_hf = retain_hf.select(range(min(cfg.debug_n_samples, len(retain_hf))))

    def _to_samples(hf_dataset, split: str) -> list[TOFUSample]:
        samples = []
        for i, row in enumerate(hf_dataset):
            q = row["question"]
            a = row["answer"]
            rephrases = (
                _generate_rephrases(q, n=cfg.num_rephrases, seed=i)
                if cfg.use_rephrase_augmentation
                else []
            )
            samples.append(TOFUSample(q, a, rephrases, split))
        return samples

    forget_samples = _to_samples(forget_hf, "forget")
    retain_samples = _to_samples(retain_hf, "retain")

    if verbose:
        print(f"[data] Forget set : {len(forget_samples)} samples "
              f"(+{cfg.num_rephrases} rephrases each)")
        print(f"[data] Retain set : {len(retain_samples)} samples")

    return forget_samples, retain_samples


# ──────────────────────────────────────────────────────────────────────────────
# Tokenization + collation
# ──────────────────────────────────────────────────────────────────────────────

def _format_qa(question: str, answer: str) -> str:
    """
    Format a QA pair as a single sequence for causal LM training.
    Loss is computed over the full sequence (question + answer).

    For masked training (loss only on answer tokens), see mask_answer_tokens().
    """
    return f"Question: {question}\nAnswer: {answer}"


def collate_fn(batch: list[dict], tokenizer: PreTrainedTokenizer,
               max_seq_len: int, device: str = "cpu"):
    """
    Collation function for DataLoader. Tokenizes + pads a batch of QA dicts.

    Each dict must have keys: "question", "answer"
    """
    texts = [_format_qa(item["question"], item["answer"]) for item in batch]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    # For causal LM: labels = input_ids (with padding masked as -100)
    labels = encoded["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids":      encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels":         labels.to(device),
    }


def make_dataloader(samples: list[TOFUSample],
                    tokenizer: PreTrainedTokenizer,
                    cfg: ARMORConfig,
                    include_rephrases: bool = False,
                    shuffle: bool = True) -> DataLoader:
    """
    Build a PyTorch DataLoader from TOFUSample list.

    Parameters
    ----------
    include_rephrases : bool
        If True, expand each sample with its rephrase variants.
        Use this flag for rephrase-invariant gradient ascent on the forget set.
    """
    rows = []
    for s in samples:
        rows.append({"question": s.question, "answer": s.answer})
        if include_rephrases and s.rephrases:
            for rq in s.rephrases:
                rows.append({"question": rq, "answer": s.answer})

    # Wrap as HuggingFace Dataset for easy DataLoader integration
    hf_ds = Dataset.from_list(rows)

    def _tokenize(batch):
        return collate_fn(
            [{"question": q, "answer": a}
             for q, a in zip(batch["question"], batch["answer"])],
            tokenizer=tokenizer,
            max_seq_len=cfg.max_seq_len,
            device="cpu",   # Move to device in training loop
        )

    loader = DataLoader(
        hf_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(
            batch, tokenizer=tokenizer,
            max_seq_len=cfg.max_seq_len,
        ),
    )
    return loader


def get_relearning_subset(forget_samples: list[TOFUSample],
                          n: int,
                          seed: int = 42) -> list[TOFUSample]:
    """
    Sample n items from the forget set for the relearning attack.
    Uses a fixed seed for reproducible attack comparisons.
    """
    rng = random.Random(seed)
    n = min(n, len(forget_samples))
    return rng.sample(forget_samples, k=n)
