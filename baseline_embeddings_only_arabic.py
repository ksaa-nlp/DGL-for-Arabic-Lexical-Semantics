"""
==========================================================================
Arabic Word Retrieval from Definitions using Sentence Embeddings (Matryoshka)
Baseline Experiment: Embeddings only, no BM25
==========================================================================

Overview:
    This script fine-tunes a Sentence Transformer model on the task of
    retrieving a word (lemma) given its definition, using
    MultipleNegativesRankingLoss wrapped in MatryoshkaLoss. Evaluation is
    performed via pure semantic similarity search (FAISS), without any
    lexical/statistical component (BM25). This script represents the
    baseline that is compared against the hybrid version (BM25 +
    Embeddings) found in a separate file.

--------------------------------------------------------------------------
Key Methodological Contribution: Root-Disjoint Split
--------------------------------------------------------------------------
    The train/test split used in this work is based on a "Root-Disjoint
    Split": word roots (lemmas) present in the training set never appear
    in the test set, and vice versa.

    This differs fundamentally from a conventional random split, which may
    allow the same root (or its derivatives) to appear in both training
    and test sets, leading to misleadingly optimistic evaluation results —
    the model may simply "memorize" the root instead of learning to
    generalize the semantic relationship between a definition and its
    word.

    By fully separating roots between the two sets, evaluation becomes
    more realistic and genuinely reflects the model's ability to
    generalize to roots/words it has never seen during training. This is
    the most important methodological distinction of this experiment and
    should be kept in mind when comparing these numbers against any work
    that relies on a conventional random split.

    Implementation note: root separation is performed beforehand, when
    plain_train.csv and plain_test.csv are created (outside this script),
    ensuring the data preparer guarantees no overlap of the lemma_plain
    column between the two files. The code below only verifies this
    condition and reports it statistically (see check_root_disjoint_split
    below) to ensure transparency and a reproducible sanity check.

Requirements (installable in a first Colab cell if needed):
    pip install pandas numpy torch faiss-cpu scikit-learn datasets \
                sentence-transformers

Note on Learning Rate:
    learning_rate is not explicitly set in
    SentenceTransformerTrainingArguments, so the default value used by
    HuggingFace Transformers is applied automatically, which is 5e-5.

Note on Batch Size and In-Batch Negatives:
    Training uses a TRUE effective batch size of 256, i.e., 255 real
    in-batch negatives per anchor. This is achieved via
    CachedMultipleNegativesRankingLoss (gradient caching), which computes
    the contrastive loss over the full 256-item batch while controlling
    GPU memory usage through mini_batch_size during the forward pass.
    Note that naive gradient accumulation does NOT increase the number of
    in-batch negatives, since each micro-batch would otherwise be
    processed independently through the loss function; gradient caching
    is required to obtain a true 256-item batch for the loss computation.
==========================================================================
"""

import os
import re
import random
import logging
import warnings
from datetime import datetime

import pandas as pd
import torch
import faiss
import numpy as np
from sklearn.metrics import ndcg_score

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MatryoshkaLoss, CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# ==========================================================================
# 0) Reproducibility
# ==========================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================================================
# 1) Global Configuration — all hyperparameters gathered here
# ==========================================================================
CONFIG = {
    # ---- Data paths — adjust to your environment ----
    "train_file": os.environ.get("TRAIN_FILE", "plain_train.csv"),
    "test_file":  os.environ.get("TEST_FILE",  "plain_test.csv"),

    # ---- Base model ----
    "base_model_name": "intfloat/multilingual-e5-large",

    # ---- Training settings ----
    # learning_rate: not set explicitly => defaults to Transformers' 5e-5
    "num_train_epochs": 5,
    # True effective batch size = 256 (255 real in-batch negatives per
    # anchor), achieved via CachedMultipleNegativesRankingLoss.
    "train_batch_size": 256,
    "cache_mini_batch_size": 64,   # memory control only, does NOT reduce negatives
    "per_device_eval_batch_size": 128,
    "warmup_ratio": 0.1,
    "matryoshka_dims": [768],
    "eval_steps": 100,
    "logging_steps": 100,
    "train_test_split_ratio": 0.1,  # internal train/eval split of the training data only

    # ---- Query template (kept in Arabic — part of the task/data itself) ----
    "query_template": "ماهي الكلمه التي تعني: {}",

    # ---- Evaluation settings ----
    "eval_batch_size": 256,
    "top_k_values": [1, 3, 5, 10],

    # ---- Experiment name ----
    "experiment_name": "baseline_embeddings_only_def_and_w_all",
}

# Automatically check bf16 support (instead of hardcoding it, to avoid
# crashes on GPUs that don't support bf16)
BF16_SUPPORTED = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# ==========================================================================
# 2) Arabic Preprocessing
# ==========================================================================
_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0610-\u061A\u06D6-\u06ED]")


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text: remove diacritics, unify alef/heh/taa marbuta forms."""
    if not isinstance(text, str):
        return ""
    text = re.sub(_DIACRITICS_RE, "", text)
    text = re.sub("[إأٱآا]", "ا", text)
    text = text.replace("ة", "ه")
    return text.strip()


def make_query(definition: str, template: str = CONFIG["query_template"]) -> str:
    """Wrap a definition into a query using the given template."""
    definition = normalize_arabic(definition)
    if definition == "":
        return ""
    return template.format(definition)


def safe_str(val) -> str:
    """
    Core safety helper: converts any value to str, stripping None/NaN/empty.
    Needed because HuggingFace Dataset sometimes converts "" to None.
    """
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    return s


# ==========================================================================
# 3) Root-Disjoint Split Check
# ==========================================================================
def check_root_disjoint_split(train_lemmas: list, test_lemmas: list) -> dict:
    """
    Verifies that word roots (lemmas) in the train and test sets are
    fully disjoint (no overlap) — the key methodological contribution of
    this work.

    Prints a statistical report showing:
        - Number of unique roots in each set.
        - Number of overlapping roots (expected to be zero).
        - Whether the split is fully disjoint.
    """
    train_set = set(normalize_arabic(w) for w in train_lemmas if safe_str(w) != "")
    test_set  = set(normalize_arabic(w) for w in test_lemmas  if safe_str(w) != "")

    overlap = train_set & test_set
    is_fully_disjoint = len(overlap) == 0

    report = {
        "train_unique_roots": len(train_set),
        "test_unique_roots": len(test_set),
        "overlapping_roots_count": len(overlap),
        "is_fully_root_disjoint": is_fully_disjoint,
    }

    print("\n" + "-" * 70)
    print("Root-Disjoint Split Check (train vs. test)")
    print("-" * 70)
    print(f"Unique roots in train: {report['train_unique_roots']}")
    print(f"Unique roots in test : {report['test_unique_roots']}")
    print(f"Overlapping roots    : {report['overlapping_roots_count']}")
    if is_fully_disjoint:
        print("OK: No overlap found — the split is 100% root-disjoint.")
    else:
        print(
            "WARNING: Overlap detected between train and test roots! "
            "This violates the core methodological assumption of this "
            "work (Root-Disjoint Split). Please review how "
            "plain_train.csv and plain_test.csv were constructed."
        )
    print("-" * 70 + "\n")

    return report


# ==========================================================================
# 4) Data Preparation (in-memory, no intermediate files saved)
# ==========================================================================
def prepare_train_df(input_path: str) -> pd.DataFrame:
    """Converts the raw training file into (anchor, positive) pairs ready for training."""
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df = df[["definition_plain", "lemma_plain"]].copy()

    df["anchor"]   = df["definition_plain"].map(safe_str).apply(make_query)
    df["positive"] = df["lemma_plain"].map(safe_str)

    df = df[(df["anchor"] != "") & (df["positive"] != "")].reset_index(drop=True)
    df = df[["anchor", "positive"]]

    print(f"Cleaned training data: {len(df)} rows")
    return df


def prepare_test_df(input_path: str) -> pd.DataFrame:
    """Converts the raw test file into (anchor, positive) pairs ready for evaluation."""
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    if "definition_plain" in df.columns and "lemma_plain" in df.columns:
        df["anchor"]   = df["definition_plain"].map(safe_str).apply(make_query)
        df["positive"] = df["lemma_plain"].map(safe_str)
    elif "definition" in df.columns and "text" in df.columns:
        df["anchor"]   = df["definition"].map(safe_str).apply(make_query)
        df["positive"] = df["text"].map(safe_str)
    else:
        raise ValueError(
            f"Test file is missing required columns. Columns found: {df.columns.tolist()}\n"
            "Expected: 'definition_plain'+'lemma_plain' or 'definition'+'text'."
        )

    df = df[(df["anchor"] != "") & (df["positive"] != "")].reset_index(drop=True)
    df = df[["anchor", "positive"]]

    print(f"Cleaned test data: {len(df)} rows")
    return df


def load_train_dataset(prepared_train_df: pd.DataFrame, test_size: float, seed: int):
    """Converts a DataFrame into a HuggingFace Dataset split into train/eval."""
    df = prepared_train_df.copy()

    for col in ["anchor", "positive"]:
        df[col] = df[col].map(safe_str)

    df = df[(df["anchor"] != "") & (df["positive"] != "")].reset_index(drop=True)

    for col in ["anchor", "positive"]:
        none_count = df[col].isnull().sum()
        empty_count = (df[col] == "").sum()
        if none_count > 0 or empty_count > 0:
            raise ValueError(
                f"Column '{col}' contains {none_count} None and {empty_count} empty values!"
            )

    print(f"All columns are clean. Row count: {len(df)}")

    dataset = Dataset.from_pandas(df[["anchor", "positive"]], preserve_index=False)

    for col in ["anchor", "positive"]:
        none_in_dataset = sum(1 for v in dataset[col] if v is None)
        if none_in_dataset > 0:
            raise ValueError(
                f"HuggingFace Dataset converted {none_in_dataset} values to None in column '{col}'!"
            )

    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"], len(df)


# ==========================================================================
# 5) Training
# ==========================================================================
def train_one_model(exp_name: str, prepared_train_df: pd.DataFrame, config: dict):
    print("\n" + "=" * 90)
    print(f"Training: {exp_name}")

    train_dataset, eval_dataset, train_rows = load_train_dataset(
        prepared_train_df,
        test_size=config["train_test_split_ratio"],
        seed=SEED,
    )
    print("Train size:", len(train_dataset))
    print("Eval size:",  len(eval_dataset))

    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"./tmp_trainer_{exp_name}_{run_time}"

    model = SentenceTransformer(config["base_model_name"])

    # True 256-item batch, 255 real in-batch negatives per anchor.
    # mini_batch_size only controls memory during the forward pass
    # (gradient caching) — it does NOT reduce the negative count, since
    # the loss is still computed over the full 256-item batch.
    inner_loss = CachedMultipleNegativesRankingLoss(
        model=model,
        mini_batch_size=config["cache_mini_batch_size"],
    )
    train_loss = MatryoshkaLoss(
        model=model,
        loss=inner_loss,
        matryoshka_dims=config["matryoshka_dims"],
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["train_batch_size"],  # true 256
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        warmup_ratio=config["warmup_ratio"],
        fp16=False,
        bf16=BF16_SUPPORTED,   # automatic check instead of hardcoding
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="no",
        logging_steps=config["logging_steps"],
        report_to="none",
        disable_tqdm=False,
        run_name=f"cmnrl256-{exp_name}",
        seed=SEED,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
    )

    trainer.train()
    print("Training finished.")
    return model, train_rows


# ==========================================================================
# 6) Evaluation Metrics
# ==========================================================================
def compute_mrr(retrieved: list, ground_truth: str) -> float:
    for rank, word in enumerate(retrieved, start=1):
        if word == ground_truth:
            return 1.0 / rank
    return 0.0


def compute_ndcg_sklearn(retrieved: list, ground_truth: str, k: int = 10) -> float:
    relevance = np.array([1 if word == ground_truth else 0 for word in retrieved[:k]])
    if relevance.sum() == 0:
        return 0.0
    mock_scores = np.arange(len(relevance), 0, -1)
    return ndcg_score([relevance], [mock_scores], k=k)


# ==========================================================================
# 7) Evaluation via Pure Semantic Similarity: FAISS (Embeddings)
# ==========================================================================
def evaluate_one_model(
    model,
    prepared_test_df: pd.DataFrame,
    prepared_train_df: pd.DataFrame = None,
    config: dict = CONFIG,
):
    """
    The corpus here = test words + training words combined (duplicates
    kept). This merging does not violate the root-disjoint principle:
    root separation only concerns the (anchor/query, positive) pairs
    actually used as training/test examples, whereas the corpus here is
    simply the "candidate answer bank" (retrieval pool) the model is
    evaluated against — an entirely separate concern from the train/test
    split itself.
    """
    test_df = prepared_test_df.copy()
    for col in ["anchor", "positive"]:
        test_df[col] = test_df[col].map(safe_str)
    test_df = test_df[
        (test_df["anchor"] != "") & (test_df["positive"] != "")
    ].reset_index(drop=True)

    queries   = test_df["anchor"].tolist()
    positives = test_df["positive"].tolist()

 # Corpus = test words + training words combined, then deduplicated
    # at the normalized-word level (surface-form deduplication, not
    # morphological root analysis). If no training data is given, the
    # test words are used as-is without deduplication.
    if prepared_train_df is not None:
        train_words = prepared_train_df["positive"].map(safe_str).tolist()
        train_words = [w for w in train_words if w != ""]
        combined = [w for w in (positives + train_words) if w != ""]

        seen_normalized = set()
        corpus = []
        for w in combined:
            norm_w = normalize_arabic(w)
            if norm_w not in seen_normalized:
                seen_normalized.add(norm_w)
                corpus.append(w)
    else:
        corpus = list(positives)

    print(
        f"Queries: {len(queries)} | "
        f"Corpus size (duplicates kept): {len(corpus)}"
    )

    query_embeddings = model.encode(
        queries,
        batch_size=config["eval_batch_size"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    corpus_embeddings = model.encode(
        corpus,
        batch_size=config["eval_batch_size"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embeddings)

    max_k = max(config["top_k_values"])
    scores, indices = index.search(query_embeddings, max_k)

    results = []
    for i in range(len(queries)):
        retrieved_words  = [corpus[idx] for idx in indices[i]]
        retrieved_scores = [float(scores[i][j]) for j in range(len(indices[i]))]
        gt = positives[i]

        results.append({
            "query":        queries[i],
            "ground_truth": gt,
            "top1":         gt == retrieved_words[0],
            "top3":         gt in retrieved_words[:3],
            "top5":         gt in retrieved_words[:5],
            "top10":        gt in retrieved_words[:10],
            "MRR":          compute_mrr(retrieved_words, gt),
            "NDCG@10":      compute_ndcg_sklearn(retrieved_words, gt, k=10),
            "predictions":  retrieved_words,
            "scores":       retrieved_scores,
        })

    df_results = pd.DataFrame(results)

    summary = {
        "num_queries": len(df_results),
        "corpus_size": len(corpus),
        "Top@1":       df_results["top1"].mean(),
        "Top@3":       df_results["top3"].mean(),
        "Top@5":       df_results["top5"].mean(),
        "Top@10":      df_results["top10"].mean(),
        "MRR":         df_results["MRR"].mean(),
        "NDCG@10":     df_results["NDCG@10"].mean(),
    }

    merged_df = pd.concat(
        [test_df.reset_index(drop=True), df_results.reset_index(drop=True)], axis=1
    )

    return summary, merged_df


# ==========================================================================
# 8) Main Entry Point
# ==========================================================================
def main(config: dict = CONFIG):
    prepared_test_df = prepare_test_df(config["test_file"])
    prepared_test_rows = len(prepared_test_df)
    print("Test rows:", prepared_test_rows)

    exp_name = config["experiment_name"]
    print("\n" + "#" * 100)
    print("Experiment:", exp_name)

    prepared_train_df = prepare_train_df(config["train_file"])
    prepared_train_rows = len(prepared_train_df)

    # Verify root-disjoint split between train and test — the key
    # methodological contribution of this work
    root_disjoint_report = check_root_disjoint_split(
        train_lemmas=prepared_train_df["positive"].tolist(),
        test_lemmas=prepared_test_df["positive"].tolist(),
    )

    model, train_rows = train_one_model(exp_name, prepared_train_df, config)

    eval_result, detailed_df = evaluate_one_model(
        model, prepared_test_df, prepared_train_df=prepared_train_df, config=config
    )

    row = {
        "name":                     exp_name,
        "base_model":               config["base_model_name"],
        "train_file":               config["train_file"],
        "test_file":                config["test_file"],
        "prepared_train_rows":      prepared_train_rows,
        "prepared_test_rows":       prepared_test_rows,
        "train_rows":               train_rows,
        "root_disjoint_split":      root_disjoint_report["is_fully_root_disjoint"],
        "overlapping_roots_count":  root_disjoint_report["overlapping_roots_count"],
        **eval_result,
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results_df = pd.DataFrame([row])

    print("\nTraining and evaluation finished.")
    print("Results summary:")
    print(results_df.to_string(index=False))

    print("\nDetailed results (first 20 rows):")
    print(detailed_df.head(20).to_string(index=False))

    return results_df, detailed_df


if __name__ == "__main__":
    results_df, detailed_df = main(CONFIG)
