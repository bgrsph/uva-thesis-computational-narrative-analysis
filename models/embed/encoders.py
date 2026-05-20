"""E5 and SBERT encoder registry + encode() helper.

See docs/superpowers/specs/2026-05-19-uni-14-embeddings-design.md (UNI-14)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer


ENCODER_REGISTRY: dict[str, tuple[str, str | None]] = {
    "e5_mistral":  ("intfloat/e5-mistral-7b-instruct",
                    "Retrieve stories with a similar narrative to the given story."),
    "sbert_mpnet": ("sentence-transformers/all-mpnet-base-v2", None),
    "story_emb":   ("uhhlt/story-emb",
                    "Retrieve stories with a similar narrative to the given story."),
    # Add new encoder keys here — no other code changes needed.
}


# Per-encoder condition override. Default is "embed all six conditions"
# (CONDITION_KEYS in the driver). Entries here override that to a subset.
# StoryEmbed is the ceiling encoder, trained on raw narrative text — embedding
# it on the five linearized conditions would be out-of-domain and redundant
# with e5_mistral, which shares the same backbone weights (thesis §3.4).
ENCODER_CONDITIONS: dict[str, tuple[str, ...]] = {
    "story_emb": ("raw_text",),
}


def encode(model: "SentenceTransformer", texts: list[str], task: str | None,
           batch_size: int = 8) -> "np.ndarray":
    """Prepend E5 instruction template iff task is set; return L2-normalized numpy embeddings."""
    if task is not None:
        texts = [f"Instruct: {task}\nQuery: {t}" for t in texts]
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
