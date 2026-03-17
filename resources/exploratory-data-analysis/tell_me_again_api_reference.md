## `tell_me_again` API Reference

### Top-level imports

```python
from tell_me_again import StoryDataset, SimilarityDataset
```

---

### `StoryDataset`

The main entry point for the corpus. Iterates over `Story` objects.

| Method / Attribute | Signature | Returns | Description |
|---|---|---|---|
| Constructor | `StoryDataset(data_path=None, only_include=[], stories=None)` | `StoryDataset` | Load the dataset from a `.zip` path. Pass `only_include=[wikidata_id, ...]` to load a subset. If `data_path=None` it auto-downloads. |
| `__iter__` | `for story in dataset:` | `Story` | Iterate over all stories. |
| `__getitem__` | `dataset["Q123"]` | `Story` | Fetch a story by its Wikidata ID string. |
| `__len__` | `len(dataset)` | `int` | Total number of stories. |
| `perform_splits` | `dataset.perform_splits()` | `dict[str, StoryDataset]` | Returns `{"train": ..., "dev": ..., "test": ...}` splits from CSV files bundled in the zip. |
| `chaturvedi_like_split` | `dataset.chaturvedi_like_split(use_anonymized=False, seed=1337)` | `(summaries, labels, included)` | Produces a Chaturvedi-style split: sampled target counts per cluster size (2→235, 3→20, 4→10, 5→1). |
| `stratified_split` | `dataset.stratified_split(label_dict, seed=2)` | `list` of splits | Stratified 2-fold split using scikit-learn. `label_dict` maps Wikidata ID → label. |
| `get_metadata_stats` | `dataset.get_metadata_stats()` | `dict` | Corpus-wide counts: books, movies, genre distribution (Wikidata P136), Gutenberg IDs, ISBNs. Reads from local `data/wikidata/` — not the zip. |
| `get_lang_stats` | `dataset.get_lang_stats()` | `dict` | Per-language summary counts, with and without near-duplicates removed. Keys: `"languages"`, `"languages_direct_translations_removed"`. |

---

### `Story` (dataclass)

One row of the corpus — a single narrative work with all its multilingual summaries.

| Attribute | Type | Description |
|---|---|---|
| `wikidata_id` | `str` | Wikidata entity ID, e.g. `"Q1000352"`. |
| `title` | `str` | Primary title. |
| `titles` | `Dict[str, str]` | Titles keyed by language code. |
| `description` | `str` | Short Wikidata description string. |
| `summaries_original` | `Dict[str, str]` | Raw Wikipedia plot summaries keyed by ISO 639-1 language code (e.g. `"en"`, `"de"`). |
| `summaries_translated` | `Dict[str, str]` | All non-English summaries machine-translated to English, keyed by source language. |
| `summaries_anonymized` | `Optional[Dict[str, str]]` | Named-entity-anonymized versions of the translated summaries (may be `None`). |
| `num_sentences` | `Dict[str, int]` | Number of sentences per language version (includes `"en"` if present). |
| `sentences` | `Dict[str, List[str]]` | Actual sentence lists per language version. |
| `similarities` | `Tensor / ndarray` | Pairwise cosine similarity matrix over the cluster's summaries. |
| `similarities_labels` | `List[str]` | Language-code labels corresponding to the similarity matrix rows/columns. |
| `genres` | `List[str]` | Wikidata entity IDs for genre tags (P136), e.g. `["Q130232"]`. No built-in label mapping — resolve via Wikidata API if needed. |

| Method | Signature | Returns | Description |
|---|---|---|---|
| `get_all_summaries_en` | `story.get_all_summaries_en(max_similarity=0.6, min_sentences=0)` | `(ids, summaries)` | Returns the English original (if present) + all translated summaries with near-duplicates removed (similarity threshold 0.6). Optionally filters by minimum sentence count. |
| `remove_duplicates` | `story.remove_duplicates(threshold=0.6)` | `Dict[str, str]` | Returns `summaries_translated` with near-duplicates removed, prioritised by translation quality score (en > fr > de > it > es). |
| `get_anonymized` | `story.get_anonymized(min_sentences=0)` | `Dict[str, str]` | Returns `summaries_anonymized` filtered by a minimum sentence count. |

---

### `SimilarityDataset`

Wraps `StoryDataset` to produce positive/negative summary pairs for training a similarity model.

| Argument | Default | Description |
|---|---|---|
| `data_path` | `None` | Path to the zip; auto-downloads if `None`. |
| `anonymized` | `True` | Use anonymized summaries if `True`, otherwise translated. |
| `min_sentences` | `0` | Filter out summaries shorter than this. |
| `negative_sample_scale` | `1.0` | Ratio of negative to positive pairs. |
| `seed` | `42` | Random seed for negative sampling. |
| `min_length` | `0` | Alias for `min_sentences` at the split level. |

| Method | Signature | Returns | Description |
|---|---|---|---|
| `__getitem__` | `dataset["train"]` | HuggingFace `Dataset` | Returns the split as a `{"text_a", "text_b", "label"}` dataset (`1` = same story, `-1` = different story). |

---

### On genres

There is no bundled genre lookup. The `genres` field on `Story` is a raw list of Wikidata Q-IDs. To resolve them to human-readable labels, query the Wikidata API:

```
https://www.wikidata.org/wiki/Special:EntityData/Q130232.json
```

Or use the `wikidataintegrator` Python library. `get_metadata_stats()` on `StoryDataset` returns a `genre_counter` with Q-IDs and their corpus frequencies.