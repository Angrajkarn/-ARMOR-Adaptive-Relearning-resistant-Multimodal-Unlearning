"""
armor/data/multimodal.py
========================
Real image-text dataset wrapper for LLaVA-1.5-7b multimodal unlearning.

Loads image-caption pairs from nlphuji/flickr30k (lightweight, free, ~30k pairs)
and pairs them with TOFU forget/retain text samples.

Usage
-----
from armor.data.multimodal import make_llava_dataloader
loader = make_llava_dataloader(samples, processor, cfg, shuffle=True)
"""

import os
import sys
import warnings
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Image source helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_flickr_images(n_images: int, image_size: int = 336):
    """
    Load up to `n_images` images from nlphuji/flickr30k.
    Falls back to solid-colour synthetic images if the dataset is unavailable.

    Returns a list of PIL Images.
    """
    try:
        from datasets import load_dataset
        from PIL import Image

        print("[llava-data] Streaming Flickr30k images from HuggingFace...")
        # streaming=True avoids a full ~9GB download
        ds = load_dataset("nlphuji/flickr30k", split="test", streaming=True,
                          trust_remote_code=True)
        images = []
        for ex in ds:
            img = ex.get("image") or ex.get("img")
            if img is None:
                # Some versions store the image under different keys
                for key in ex:
                    if hasattr(ex[key], "convert"):  # PIL.Image check
                        img = ex[key]
                        break
            if img is not None:
                images.append(img.convert("RGB").resize((image_size, image_size)))
            if len(images) >= n_images:
                break

        if images:
            print(f"[llava-data] Loaded {len(images)} real Flickr30k images.")
            return images

    except Exception as e:
        warnings.warn(f"[llava-data] Flickr30k load failed ({e}). "
                      "Using synthetic solid-colour images.")

    # Fallback: create solid grey images (still valid tensors)
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for multimodal loading. "
                          "Run: pip install Pillow")

    print("[llava-data] ⚠️  Using synthetic fallback images (no Flickr30k access).")
    images = []
    for i in range(n_images):
        # Vary colour slightly so they're not all identical
        brightness = 100 + (i % 50) * 3
        img = Image.new("RGB", (image_size, image_size),
                        color=(brightness, brightness, brightness))
        images.append(img)
    return images


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class LLaVATextImageDataset(Dataset):
    """
    Pairs TOFU text samples with real (or fallback) images and tokenises them
    using the LlavaProcessor.

    Each item produces:
        input_ids       : [seq_len]
        attention_mask  : [seq_len]
        pixel_values    : [3, H, W]    (CLIP-normalised)
        labels          : [seq_len]    (-100 on prompt tokens, token ids on answer)
    """

    LLAVA_PROMPT_TEMPLATE = (
        "USER: <image>\n{question}\nASSISTANT: {answer}"
    )

    def __init__(self, samples: list, processor, images: list,
                 max_seq_len: int = 256):
        self.samples     = samples
        self.processor   = processor
        self.images      = images
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        # Cycle through images if fewer images than samples
        image = self.images[idx % len(self.images)]

        if hasattr(sample, "get"):
            question = sample.get("question", sample.get("input", "Describe the image."))
            answer   = sample.get("answer",   sample.get("target", ""))
        else:
            question = getattr(sample, "question", "Describe the image.")
            answer   = getattr(sample, "answer", "")

        prompt = self.LLAVA_PROMPT_TEMPLATE.format(
            question=question, answer=answer
        )

        # Let LlavaProcessor handle both vision and text encoding
        encoding = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
        )

        input_ids      = encoding["input_ids"].squeeze(0)        # [L]
        attention_mask = encoding["attention_mask"].squeeze(0)   # [L]
        pixel_values   = encoding["pixel_values"].squeeze(0)     # [3, H, W]

        # Build labels: mask everything up to and including the "ASSISTANT: " token
        labels = input_ids.clone()
        assistant_token_str = "ASSISTANT:"
        # Find split point: mask prompt tokens with -100
        try:
            prompt_enc = self.processor.tokenizer(
                self.LLAVA_PROMPT_TEMPLATE.split("{answer}")[0].format(
                    question=question),
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_len = prompt_enc["input_ids"].shape[-1]
            labels[:prompt_len] = -100
        except Exception:
            # Fallback: mask first 70% of tokens
            prompt_len = int(0.7 * len(labels))
            labels[:prompt_len] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,
            "labels":         labels,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def make_llava_dataloader(
    samples: list,
    processor,
    cfg,
    shuffle: bool = True,
    image_size: int = 336,
) -> DataLoader:
    """
    Build a DataLoader of real image+text pairs for LLaVA unlearning.

    Parameters
    ----------
    samples   : List of TOFU-style dicts (keys: 'question', 'answer')
    processor : LlavaProcessor returned by get_llava_model_and_processor()
    cfg       : ARMORConfig instance
    shuffle   : Whether to shuffle the loader
    image_size: Target spatial resolution (336 for LLaVA-1.5)

    Returns
    -------
    DataLoader producing {input_ids, attention_mask, pixel_values, labels}
    """
    n_images = max(64, len(samples))   # preload at most 64 real images
    images   = _load_flickr_images(n_images, image_size=image_size)

    dataset = LLaVATextImageDataset(
        samples     = samples,
        processor   = processor,
        images      = images,
        max_seq_len = cfg.max_seq_len,
    )

    return DataLoader(
        dataset,
        batch_size  = cfg.batch_size,
        shuffle     = shuffle,
        num_workers = 0,    # keep 0 for Colab compatibility
        pin_memory  = (cfg.device == "cuda"),
    )
