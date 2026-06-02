"""BERT+CRF event extraction over the TMA subset (UNI-23).

Run via the colocated venv:

    models/bert_crf/.venv-maven-train/bin/python models/bert_crf/infer_tma.py \\
        --input      data/intermediate/tma_subset.jsonl \\
        --output     data/intermediate/tma_subset_events.jsonl \\
        --checkpoint data/intermediate/models/bert_crf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple


def bio_to_spans(
    bio_tags: List[str],
    offsets: List[Tuple[int, int]],
) -> List[Tuple[int, int, str]]:
    """Decode BIO subword tags into (start_char, end_char, event_type) spans.

    Special-token positions (offsets == (0, 0)) are skipped without affecting
    open spans. Malformed BIO ('I-X' without a matching 'B-X', or a type switch)
    is treated as the start of a new span — standard relaxation.
    """
    if len(bio_tags) != len(offsets):
        raise ValueError(f"length mismatch: {len(bio_tags)} tags vs {len(offsets)} offsets")

    spans: List[Tuple[int, int, str]] = []
    open_start = open_end = open_type = None

    def close_open():
        if open_start is not None:
            spans.append((open_start, open_end, open_type))

    for tag, (s, e) in zip(bio_tags, offsets):
        if s == 0 and e == 0:
            continue  # special token; do not affect open span
        if tag == "O":
            close_open()
            open_start = open_end = open_type = None
            continue
        prefix, _, etype = tag.partition("-")
        if prefix == "B":
            close_open()
            open_start, open_end, open_type = s, e, etype
        elif prefix == "I":
            if open_type == etype:
                open_end = e
            else:
                close_open()
                open_start, open_end, open_type = s, e, etype
        else:
            raise ValueError(f"Unknown BIO tag: {tag!r}")

    close_open()
    return spans


def first_subword_mask(word_ids: List) -> List[bool]:
    """True at the first sub-word of each word; False at continuation sub-words
    and special tokens (``word_id is None``).

    Mirrors the training convention (utils_maven.py:125-128) where a word's label
    lives only on its first sub-word and continuations are masked (-100). Used to
    build the CRF label mask so inference decodes exactly the positions the model
    was trained on — one per word.
    """
    mask: List[bool] = []
    prev = object()  # sentinel: never equal to any word_id
    for wid in word_ids:
        mask.append(wid is not None and wid != prev)
        prev = wid
    return mask


def words_from_subwords(
    word_ids: List,
    offsets: List[Tuple[int, int]],
    bio_tags: List[str],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Collapse per-sub-word BIO tags to per-word ``(tag, char-span)``.

    Each word's tag is read from its FIRST sub-word (the only position the model
    was trained to label); the word's char span runs from the first sub-word's
    start to the last sub-word's end. Special tokens (``word_id is None``) are
    skipped. Returns ``(word_tags, word_offsets)`` ready for :func:`bio_to_spans`,
    so triggers come out as whole words instead of sub-word slices ("su", "cing").
    """
    if not (len(word_ids) == len(offsets) == len(bio_tags)):
        raise ValueError(
            f"length mismatch: {len(word_ids)} word_ids, {len(offsets)} offsets, "
            f"{len(bio_tags)} tags"
        )

    word_tags: List[str] = []
    word_offsets: List[Tuple[int, int]] = []
    i, n = 0, len(word_ids)
    while i < n:
        wid = word_ids[i]
        if wid is None:
            i += 1
            continue
        start, end = offsets[i][0], offsets[i][1]
        tag = bio_tags[i]                      # first sub-word carries the label
        j = i + 1
        while j < n and word_ids[j] == wid:    # extend span over continuations
            end = offsets[j][1]
            j += 1
        word_tags.append(tag)
        word_offsets.append((start, end))
        i = j
    return word_tags, word_offsets


def indexed_nonempty_sentences(sentences: List[str]) -> List[Tuple[int, str]]:
    """``[(sent_id, text), ...]`` for sentences with non-whitespace content.

    Blank sentences are skipped: they contain no events, and (since we mask the
    CRF to first-sub-words only) an empty sentence yields zero decodable
    positions, which crashes the CRF (``_score_sentence`` indexes ``length-1``
    → -1). Surviving sentences keep their original ``sent_id``.
    """
    return [(i, s) for i, s in enumerate(sentences) if str(s).strip()]


def _pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    # MPS is intentionally skipped: the THU-KEG CRF Viterbi decoder mixes CPU and
    # MPS tensors internally (predates MPS support) and crashes with
    # "torch.cat(): all input tensors must be on the same device". CPU is fine
    # for the TMA-scale workload (~540K sentence forward passes; hours on CPU).
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description="BERT+CRF inference over TMA subset.")
    ap.add_argument("--input",      required=True, type=Path)
    ap.add_argument("--output",     required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--sample-size", type=int, default=None,
        help="If set, process only the first N input rows (sanity-check runs).",
    )
    args = ap.parse_args()

    # Imports here so `--help` and the unit tests don't pay for them.
    import torch
    # BertTokenizerFast is required for return_offsets_mapping; the slow Python
    # tokenizer raises NotImplementedError. Fast tokenizer is built from vocab.txt.
    from transformers import BertConfig, BertTokenizerFast

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bert_crf import BertCRFForTokenClassification
    from utils_maven import get_labels

    device = _pick_device()
    print(f"[infer_tma] device={device}  checkpoint={args.checkpoint}", flush=True)

    labels = get_labels("")           # canonical 337-element list
    if len(labels) != 337:
        raise RuntimeError(f"expected 337 labels, got {len(labels)}")
    id2label = {i: lbl for i, lbl in enumerate(labels)}

    config = BertConfig.from_pretrained(str(args.checkpoint), num_labels=len(labels))
    tokenizer = BertTokenizerFast.from_pretrained(str(args.checkpoint), do_lower_case=True)
    model = BertCRFForTokenClassification.from_pretrained(str(args.checkpoint), config=config)
    model.to(device).eval()
    pad_id = -100  # CrossEntropyLoss().ignore_index, matches training-time setup

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Pre-count input rows for a real ETA. One pass over a 41k-line file is ~1s.
    with args.input.open("r", encoding="utf-8") as fin:
        total_rows = sum(1 for ln in fin if ln.strip())
    if args.sample_size is not None:
        total_rows = min(total_rows, args.sample_size)
        print(f"[infer_tma] sample-size={args.sample_size}  processing {total_rows} rows", flush=True)

    n_in = n_events = 0
    t0 = time.time()
    PROGRESS_EVERY = 500

    def fmt_secs(s: float) -> str:
        s = int(s)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

    with args.input.open("r", encoding="utf-8") as fin, \
         args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            if args.sample_size is not None and n_in >= args.sample_size:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sentences: List[str] = row["sentences"]
            events: List[dict] = []

            # Drop blank sentences up front (no events; an empty sentence would
            # give the CRF a zero-length sequence and crash). Surviving sentences
            # keep their original sent_id via the (sent_id, text) pairs.
            indexed = indexed_nonempty_sentences(sentences)
            for batch_start in range(0, len(indexed), args.batch_size):
                chunk    = indexed[batch_start : batch_start + args.batch_size]
                sent_ids = [sid for sid, _ in chunk]
                batch    = [txt for _, txt in chunk]
                enc = tokenizer(
                    batch,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    add_special_tokens=True,
                )
                if enc["input_ids"].shape[1] > 128:
                    raise RuntimeError(
                        f"sentence in {row['wikidata_id']}/{row['summary_id']} exceeds 128 subwords; "
                        "post-§2 filter should have prevented this"
                    )
                word_ids_list = [enc.word_ids(b) for b in range(len(batch))]
                offsets_list  = enc.pop("offset_mapping").tolist()
                attn_list     = enc["attention_mask"].tolist()
                enc           = {k: v.to(device) for k, v in enc.items()}

                # Dummy labels select which positions the CRF decodes. We pass
                # them so forward() takes the with-labels branch (the no-labels
                # branch mixes Float/Long tensors and crashes on newer torch).
                # Mirroring training (utils_maven.py:125-128), ONLY the first
                # sub-word of each word is decoded (label 0); special tokens and
                # continuation sub-words are pad_id (-100), so the CRF runs over
                # one position per word — exactly as it was trained/evaluated.
                # (The previous code marked every real token, making the CRF
                # decode continuation sub-words it never learned and the decoder
                # slice triggers into fragments like "su"/"cing".)
                first_mask = torch.tensor(
                    [first_subword_mask(w) for w in word_ids_list],
                    dtype=torch.bool, device=enc["input_ids"].device,
                )
                dummy_labels = torch.full_like(enc["input_ids"], pad_id)
                dummy_labels[first_mask] = 0
                with torch.no_grad():
                    out = model(pad_token_label_id=pad_id, labels=dummy_labels, **enc)
                best_path = out[-1].cpu().tolist()  # forward returns (..., best_path)

                for off_in_batch, (tag_ids, wids, off, mask) in enumerate(
                    zip(best_path, word_ids_list, offsets_list, attn_list)
                ):
                    n = sum(mask)
                    # best_path is -100 at non-decoded positions (continuations /
                    # specials); .get(..., "O") guards those. words_from_subwords
                    # reads only first-sub-word positions, which hold real tags,
                    # and emits whole-word char spans.
                    bio_tags = [id2label.get(t, "O") for t in tag_ids[:n]]
                    sent_id  = sent_ids[off_in_batch]
                    sent     = sentences[sent_id]
                    word_tags, word_offsets = words_from_subwords(wids[:n], off[:n], bio_tags)
                    for (s, e, etype) in bio_to_spans(word_tags, word_offsets):
                        events.append({
                            "event_id":   f"e{len(events) + 1}",
                            "sent_id":    sent_id,
                            "trigger":    sent[s:e],
                            "event_type": etype,
                            "start":      s,
                            "end":        e,
                        })

            row["events"] = events
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_in += 1
            n_events += len(events)
            if n_in % PROGRESS_EVERY == 0 or n_in == total_rows:
                now     = time.time()
                elapsed = now - t0
                rate    = n_in / elapsed
                pct     = 100.0 * n_in / total_rows
                eta     = (total_rows - n_in) / rate if rate > 0 else 0
                now_str    = time.strftime("%H:%M:%S", time.localtime(now))
                finish_str = time.strftime("%H:%M:%S", time.localtime(now + eta))
                print(
                    f"[infer_tma] {now_str}  {n_in}/{total_rows} ({pct:.1f}%)  "
                    f"events={n_events}  rate={rate:.2f}/s  "
                    f"elapsed={fmt_secs(elapsed)}  ETA={fmt_secs(eta)} (done ~{finish_str})",
                    flush=True,
                )

    print(f"[infer_tma] DONE  summaries={n_in}  events={n_events}  in {fmt_secs(time.time() - t0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
