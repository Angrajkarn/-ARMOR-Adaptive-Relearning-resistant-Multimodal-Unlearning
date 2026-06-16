"""
armor/data_muse.py
==================
MUSE Benchmark Data Loader — Machine Unlearning Six-way Evaluation

MUSE (Shi et al., 2024) evaluates unlearning across 4 corpora:
    - muse-books      : Harry Potter chapters (creative fiction)
    - muse-news       : BBC news articles (factual current events)
    - muse-github     : Open-source code snippets (structured text)
    - muse-biomedical : PubMed abstracts (technical scientific text)

Each split provides:
    forget_set : text passages to unlearn
    retain_set : similar text to preserve
    test_set   : evaluation prompts

HuggingFace dataset: "jaechan-llm/MUSE" (or local fallback)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import Optional, List, Dict, Tuple
from datasets import load_dataset, DatasetDict
import re

from .config import ARMORConfig


# Supported MUSE domains
MUSE_DOMAINS = {
    "books":      "muse-books",
    "news":       "muse-news",
    "github":     "muse-github",
    "biomedical": "muse-biomedical",
}


# ─────────────────────────────────────────────────────────────────────────────
# MUSE Text Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MUSETextDataset(Dataset):
    """
    MUSE benchmark dataset.

    Each sample is a (prompt, completion) pair tokenized for causal LM
    training with teacher-forcing labels.

    Args:
        texts      : list of raw text strings
        tokenizer  : HuggingFace tokenizer
        max_length : maximum token length per sample
    """

    def __init__(self, texts: List[str],
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = self._tokenize(texts)

    def _tokenize(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        samples = []
        for text in texts:
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt")
            input_ids      = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels         = input_ids.clone()
            # Mask padding in labels
            labels[attention_mask == 0] = -100
            samples.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# MUSE Loader
# ─────────────────────────────────────────────────────────────────────────────

class MUSEDataLoader:
    """
    Loads MUSE benchmark splits and returns standard DataLoaders
    compatible with all ARMOR unlearning methods.

    Usage:
        loader = MUSEDataLoader(cfg, tokenizer, domain="books")
        forget_loader, retain_loader, test_loader = loader.get_loaders()
    """

    # Fallback: generate synthetic data when MUSE is unavailable
    _SYNTHETIC_FORGET = {
        "books": [
            "Harry Potter was a young wizard who lived at Number Four, "
            "Privet Drive with his uncle and aunt.",
            "Dumbledore was the headmaster of Hogwarts School of Witchcraft "
            "and Wizardry, known for his wisdom.",
            "The Sorting Hat was a magical artefact that assigned Hogwarts "
            "students to one of four houses.",
            "Voldemort, also known as He-Who-Must-Not-Be-Named, sought to "
            "achieve immortality through Horcruxes.",
            "Hermione Granger was known for her exceptional academic ability "
            "and loyalty to her friends.",
            "Ron Weasley came from a large wizarding family and was Harry's "
            "closest companion at Hogwarts.",
        ],
        "news": [
            "The central bank raised interest rates by 25 basis points, "
            "citing persistent inflationary pressures.",
            "Scientists announced a breakthrough in quantum computing, "
            "achieving 1000-qubit entanglement.",
            "The summit concluded with a landmark climate agreement signed "
            "by 195 member nations.",
            "Unemployment figures fell to a 50-year low, boosting consumer "
            "confidence indexes globally.",
        ],
        "github": [
            "def quicksort(arr): return arr if len(arr)<=1 else "
            "quicksort([x for x in arr[1:] if x<=arr[0]]) + "
            "[arr[0]] + quicksort([x for x in arr[1:] if x>arr[0]])",
            "class BinaryTree: def __init__(self,val): "
            "self.val=val; self.left=None; self.right=None",
            "SELECT u.name, COUNT(o.id) FROM users u "
            "JOIN orders o ON u.id=o.user_id GROUP BY u.name",
        ],
        "biomedical": [
            "The CRISPR-Cas9 system enables precise genomic editing by "
            "targeting specific DNA sequences via guide RNAs.",
            "Checkpoint inhibitors targeting PD-1/PD-L1 have shown durable "
            "responses in non-small-cell lung cancer.",
            "Tau hyperphosphorylation and amyloid-β aggregation are "
            "hallmarks of Alzheimer's disease pathology.",
        ],
    }
    _SYNTHETIC_RETAIN = [
        "The history of science has been shaped by countless discoveries "
        "that transformed our understanding of the natural world.",
        "Language models are trained on large corpora using next-token "
        "prediction as the primary objective.",
        "Machine learning encompasses supervised, unsupervised, and "
        "reinforcement learning paradigms.",
        "Neural networks consist of interconnected layers of artificial "
        "neurons that transform input representations.",
        "The field of natural language processing has advanced rapidly "
        "with the advent of transformer architectures.",
        "Deep learning models require large amounts of labelled data "
        "and substantial computational resources.",
    ]

    def __init__(self,
                 cfg:       ARMORConfig,
                 tokenizer: PreTrainedTokenizer,
                 domain:    str  = "books",
                 n_forget:  Optional[int] = None,
                 n_retain:  Optional[int] = None):
        self.cfg       = cfg
        self.tokenizer = tokenizer
        self.domain    = domain
        self.n_forget  = n_forget
        self.n_retain  = n_retain

        if domain not in MUSE_DOMAINS and domain not in self._SYNTHETIC_FORGET:
            raise ValueError(
                f"Unknown MUSE domain '{domain}'. "
                f"Choose from: {list(MUSE_DOMAINS.keys())}")

    def _load_from_hub(self) -> Tuple[List[str], List[str]]:
        """Try loading MUSE from HuggingFace Hub."""
        try:
            ds = load_dataset("jaechan-llm/MUSE",
                               name=MUSE_DOMAINS.get(self.domain, self.domain),
                               split=None)
            forget_texts = [s["text"] for s in ds.get("forget", ds.get("train", []))]
            retain_texts = [s["text"] for s in ds.get("retain", ds.get("validation", []))]
            return forget_texts, retain_texts
        except Exception as e:
            print(f"[MUSE] Hub load failed ({e}). Using synthetic fallback.")
            return [], []

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns: (forget_loader, retain_loader, test_loader)
        All loaders are compatible with ARMOR unlearning methods.
        """
        forget_texts, retain_texts = self._load_from_hub()

        # Fallback to synthetic data
        if not forget_texts:
            forget_texts = self._SYNTHETIC_FORGET.get(self.domain,
                            self._SYNTHETIC_FORGET["books"])
        if not retain_texts:
            retain_texts = self._SYNTHETIC_RETAIN

        # Subsample if limits specified
        if self.n_forget:
            forget_texts = forget_texts[:self.n_forget]
        if self.n_retain:
            retain_texts = retain_texts[:self.n_retain]

        max_len = self.cfg.max_seq_len

        forget_ds = MUSETextDataset(forget_texts, self.tokenizer, max_len)
        retain_ds = MUSETextDataset(retain_texts, self.tokenizer, max_len)
        # Use a subset of forget as test set for evaluation
        test_ds   = MUSETextDataset(forget_texts[:max(1, len(forget_texts)//4)],
                                     self.tokenizer, max_len)

        bs = self.cfg.batch_size

        print(f"[MUSE] Domain: '{self.domain}' | "
              f"forget={len(forget_ds)} | retain={len(retain_ds)} samples")

        forget_loader = DataLoader(forget_ds, batch_size=bs, shuffle=True)
        retain_loader = DataLoader(retain_ds, batch_size=bs, shuffle=True)
        test_loader   = DataLoader(test_ds,   batch_size=bs, shuffle=False)

        return forget_loader, retain_loader, test_loader
