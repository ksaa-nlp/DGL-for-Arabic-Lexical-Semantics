# Arabic Word Retrieval from Definitions — Baseline & Hybrid Experiments

This repository contains two experiments for the task of **retrieving an
Arabic word (lemma) given its definition**, using **Sentence
Transformers**:

| File | Description |
|---|---|
| [`baseline_embeddings_only.py`](./baseline_embeddings_only.py) | **Baseline:** fine-tuning + evaluation via pure semantic search (Embeddings only) with FAISS |
| [`hybrid_retrieval.py`](./hybrid_retrieval.py) | **Hybrid:** same fine-tuning, but evaluation combines **BM25 + Embeddings** through a weighted fusion of both signals |

Both scripts are fully self-contained — either one can be copied and run
on its own, without depending on other files in this repository.

---

## Key Methodological Contribution: Root-Disjoint Split

The core methodological distinction of this work, compared to many
similar studies, is how the data is split: **word roots (lemmas) present
in the training set never appear in the test set, and vice versa.**

This differs from a conventional random split, which may allow the same
root (or its derivatives) to appear in both sets, leading to misleadingly
optimistic evaluation (the model may simply "memorize" the root instead
of learning genuine semantic generalization). A fully root-disjoint split
makes evaluation reflect the model's real ability to generalize to
previously unseen words.

Both scripts include a `check_root_disjoint_split` function that runs
automatically at execution time, printing a report of unique roots per
set and any overlap (expected to be zero), and also adding this result to
the final results table for transparency and reproducibility.

---

## Requirements

### Baseline experiment:
```bash
pip install pandas numpy torch faiss-cpu scikit-learn datasets sentence-transformers
```

### Hybrid experiment (also needs rank_bm25):
```bash
pip install pandas numpy torch faiss-cpu scikit-learn datasets sentence-transformers rank_bm25
```

---

## Data Format

Both `plain_train.csv` and `plain_test.csv` must contain at least these
two columns:

| Column | Description |
|---|---|
| `definition_plain` | The word's definition |
| `lemma_plain` | The word itself (ground truth) |

By default, the scripts look for `plain_train.csv` and `plain_test.csv`
in the working directory. You can pass custom paths via environment
variables:

```bash
export TRAIN_FILE=/path/to/your_train.csv
export TEST_FILE=/path/to/your_test.csv
```

---

## Running the Experiments

### Baseline:
```bash
python baseline_embeddings_only.py
```

### Hybrid:
```bash
python hybrid_retrieval.py
```

Both scripts print:
1. The Root-Disjoint Split Check report.
2. Training progress.
3. A results summary table (Top@1/3/5/10, MRR, NDCG@10).
4. A sample of detailed results (first 20 queries).

---

## Key Hyperparameters

All hyperparameters are gathered in the `CONFIG` dictionary at the top of
each file, including:

| Parameter | Value |
|---|---|
| `base_model_name` | `intfloat/multilingual-e5-large` |
| `num_train_epochs` | 5 |
| `train_batch_size` | **256 (true batch size, not an accumulated/effective size)** |
| `cache_mini_batch_size` | 64 (memory control only during the forward pass; does **not** reduce the number of in-batch negatives) |
| In-batch negatives per anchor | **255** (true count, achieved via `CachedMultipleNegativesRankingLoss` with gradient caching) |
| `matryoshka_dims` | [768] |
| Loss function | `CachedMultipleNegativesRankingLoss` wrapped in `MatryoshkaLoss` |
| **`learning_rate`** | **Not set explicitly → defaults to HuggingFace Transformers' default, `5e-5`** |
| `seed` (for reproducibility) | 42 |

**Note on batch size implementation:** a batch size of 256 with naive gradient accumulation (e.g., `per_device_train_batch_size=64` + `gradient_accumulation_steps=4`) does **not** yield 255 in-batch negatives, since each accumulation micro-batch is processed independently through the loss function — only the optimizer's weight-update step is effectively 256-sized, while the number of negatives seen per loss computation remains bounded by the micro-batch size. To obtain a true 256-item batch for the contrastive loss itself, this codebase uses `CachedMultipleNegativesRankingLoss`, which applies gradient caching to compute the loss over the full batch while still controlling GPU memory via `cache_mini_batch_size` during the forward pass.

### Hybrid experiment only:
| Parameter | Value |
|---|---|
| `hybrid_alpha` | 0.7 (weight of embeddings vs. BM25; 1.0 = embeddings only, 0.0 = BM25 only) |
| `bm25_k1` | 1.5 |
| `bm25_b` | 0.75 |

Note: the query template (`query_template`) used to wrap definitions
before encoding is kept in Arabic, as it is part of the task/data itself
rather than code logic.

---

## Evaluation Metrics

- **Top@1 / Top@3 / Top@5 / Top@10**: proportion of queries where the
  correct word appears within the top-k retrieved results.
- **MRR** (Mean Reciprocal Rank).
- **NDCG@10** (Normalized Discounted Cumulative Gain).

---

## Notes on Reproducibility

- A global random seed (`SEED = 42`) is set for `random`, `numpy`, and
  `torch`.
- `bf16` support is checked automatically based on hardware availability,
  instead of being hardcoded, to avoid crashes on GPUs that don't support
  it.
- The retrieval corpus (candidate pool) used during evaluation = test
  words + training words combined, **with duplicates kept** (not
  deduplicated). This is entirely separate from the train/test split of
  (definition, word) pairs used for training and evaluation, which
  follows the root-disjoint principle described above.

---

## License

This work is licensed under the [MIT License](./LICENSE).

## Citation

If you use this code in your work, please cite this repository.
<!-- Update the following block with your paper details once accepted -->
<!--
```
@inproceedings{your_paper_2026,
  title     = {Paper Title},
  author    = {Author Name},
  booktitle = {Conference Name},
  year      = {2026}
}
```
-->
