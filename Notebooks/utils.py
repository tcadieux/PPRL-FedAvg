"""Shared helpers which are reused across multiple notebooks. This includes file paths, feature names, data loading, metrics, and histogram plotting"""

from pathlib import Path

import numpy as np
import pandas as pd
import jellyfish
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data" / "output"
RESULTS_DIR = PROJECT_DIR / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [
    "first_name",
    "last_name",
    "street_address",
    "date_of_birth",
    "email",
    "zip_code",
    "state",
    "phone",
]

# Three example pairs to track across tiers in the comparison notebook.
EXAMPLE_PAIRS = [
    ("Clear match", 11841, 11841),
    ("Missed match (false negative)", 10516, 10516),
    ("Non-match", 47135, 39460),
]

# String comparison function
def jw_text_similarity(series_a, series_b):
    """Jaro-Winkler similarity. Missing or blank values score 0."""
    return pd.Series(
        [
            (
                jellyfish.jaro_winkler_similarity(str(a), str(b))
                if pd.notna(a) and pd.notna(b) and a and b
                else 0.0
            )
            for a, b in zip(series_a, series_b)
        ],
        index=series_a.index,
    )


def exact_match(series_a, series_b):
    return (series_a == series_b).astype(float)


def load_candidates():
    """Merge Party A, Party B, and the candidate pairs into one row-per-pair frame."""
    party_a = pd.read_csv(DATA_DIR / "party_a.csv", dtype={"zip_code": str}).set_index(
        "rec_id_a"
    )
    party_b = pd.read_csv(DATA_DIR / "party_b.csv", dtype={"zip_code": str}).set_index(
        "rec_id_b"
    )
    pairs = pd.read_csv(DATA_DIR / "pairs.csv", usecols=["rec_id_a", "rec_id_b"])
    pairs["is_match"] = pairs["rec_id_a"] == pairs["rec_id_b"]
    return pairs.merge(
        party_a.add_suffix("_a"), left_on="rec_id_a", right_index=True
    ).merge(party_b.add_suffix("_b"), left_on="rec_id_b", right_index=True)


# Normalisation functions for exact match
def normalize_digits(series):
    return series.astype(str).str.replace(r"\D", "", regex=True)


def normalize_state(series):
    return series.astype(str).str.strip().str.upper()


def build_feature_matrix(candidates):
    """Plaintext 8-column similarity matrix used by tiers 1-4."""
    def jw(name):
        return jw_text_similarity(candidates[f"{name}_a"], candidates[f"{name}_b"])

    def exact(name, norm):
        return exact_match(norm(candidates[f"{name}_a"]), norm(candidates[f"{name}_b"]))

    return pd.DataFrame({
        "first_name":     jw("first_name"),
        "last_name":      jw("last_name"),
        "street_address": jw("street_address"),
        "date_of_birth":  jw("date_of_birth"),
        "email":          jw("email"),
        "zip_code":       exact("zip_code", normalize_digits),
        "state":          exact("state",    normalize_state),
        "phone":          exact("phone",    normalize_digits),
    })[FEATURE_NAMES]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def train_locally(X, y, weights, intercept, epochs, batch_size, learning_rate, rng):
    """One party's mini-batch pass on the binary cross-entropy loss."""
    weights = weights.copy()
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            batch = order[start : start + batch_size]
            Xb, yb = X[batch], y[batch]
            errors = sigmoid(Xb @ weights + intercept) - yb
            weights -= learning_rate * (Xb.T @ errors) / len(batch)
            intercept -= learning_rate * errors.mean()
    return weights, intercept


def fedavg(local_results):
    """Unweighted average of each party's weights & intercept."""
    weights = np.mean([w for w, _ in local_results], axis=0)
    intercept = float(np.mean([i for _, i in local_results]))
    return weights, intercept


def compute_metrics(
    tier_name, true_labels, predicted_labels, scores, threshold, n_pairs
):
    """Calculate precision, recall, F1, and AUC-ROC for a given tier."""
    return {
        "tier": tier_name,
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1": f1_score(true_labels, predicted_labels, zero_division=0),
        "auc_roc": roc_auc_score(true_labels, scores),
        "threshold": threshold,
        "n_pairs": n_pairs,
    }


def print_metrics(metrics):
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"Threshold : {metrics['threshold']}")
    print(f"N pairs   : {metrics['n_pairs']:,}")


def save_metrics(metrics, filename):
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / filename, index=False)


def save_example_pair_predictions(candidates, tier_name, score_column, filename):
    """Save each tier's predictions on EXAMPLE_PAIRS for the comparison notebook."""
    rows = []
    for label, rec_id_a, rec_id_b in EXAMPLE_PAIRS:
        mask = (candidates["rec_id_a"] == rec_id_a) & (candidates["rec_id_b"] == rec_id_b)
        row = candidates.loc[mask].iloc[0]
        rows.append(
            {
                "tier": tier_name,
                "label": label,
                "rec_id_a": rec_id_a,
                "rec_id_b": rec_id_b,
                "is_match": bool(row["is_match"]),
                "score": float(row[score_column]),
                "predicted_match": int(row["predicted_match"]),
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / filename, index=False)


def plot_score_distribution(
    match_scores,
    non_match_scores,
    threshold,
    title,
    x_label,
    filename,
    legend_loc="upper center",
):
    """Step histograms of match vs non-match scores on a log y-axis."""
    plt.figure(figsize=(10, 5))
    plt.hist(
        non_match_scores,
        bins=50,
        histtype="step",
        linewidth=2,
        color="#d62728",
        label=f"Non-matches (n={len(non_match_scores):,})",
    )
    plt.hist(
        match_scores,
        bins=50,
        histtype="step",
        linewidth=2,
        color="#2ca02c",
        label=f"Matches (n={len(match_scores):,})",
    )
    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Threshold = {threshold}",
    )

    plt.yscale("log")
    plt.yticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000", "10,000"])
    plt.xlim(0, 1)
    plt.xlabel(x_label)
    plt.ylabel("Number of pairs")
    plt.title(title)
    plt.legend(loc=legend_loc, frameon=False)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename)
    plt.show()
    plt.close()
