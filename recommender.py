"""Content-based Netflix title recommender.

Uses only fields available in the supplied dataset. The similarity score is a
relative ranking signal, not a probability that a user will like a title.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


DEFAULT_DATA_PATH = "data/Dataset.csv"

FEATURE_WEIGHTS = {
    "listed_in": 4.0,
    "type": 2.0,
    "director": 1.5,
    "country": 1.0,
    "rating": 0.75,
    "release_year": 0.5,
    "duration": 0.5,
}


def load_dataset(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and clean the Netflix dataset."""
    df = pd.read_csv(path)
    required = {"title", "type", "listed_in"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("").astype(str).str.strip()
    df = df.dropna(subset=["title"]).copy()
    df = df[df["title"].astype(str).str.strip().ne("")]
    df = df.drop_duplicates(subset=["title", "type"], keep="first").reset_index(drop=True)

    # Normalize fields that are often missing or stored in mixed formats.
    for col in ["type", "listed_in", "director", "country", "rating", "duration"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    if "release_year" not in df.columns:
        df["release_year"] = ""
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    return df


def _tokens(value: object) -> str:
    """Normalize comma-separated metadata into whitespace-separated tokens."""
    return " ".join(
        part.strip().lower().replace("&", " ")
        for part in str(value or "").replace("/", " ").split(",")
        if part.strip()
    )


def build_feature_text(df: pd.DataFrame) -> pd.Series:
    """Create a weighted text representation of each title's metadata."""
    text = pd.Series("", index=df.index, dtype="object")
    for field, weight in FEATURE_WEIGHTS.items():
        if field not in df.columns:
            continue
        if field == "release_year":
            values = df[field].fillna("").apply(
                lambda x: f"year_{int(x)}" if str(x).strip() and pd.notna(x) else ""
            )
        else:
            values = df[field].fillna("").apply(_tokens)

        # Repeating a field's tokens approximates a feature weight in TF-IDF.
        repeats = max(1, int(round(weight)))
        text = text + " " + values.apply(lambda value: (" " + value) * repeats)
    return text.str.strip()


class NetflixRecommender:
    """TF-IDF/cosine-similarity recommender with title disambiguation."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        self.feature_text = build_feature_text(self.df)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=30000,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.feature_text)
        self._title_lookup = {}
        for i, row in self.df.iterrows():
            self._title_lookup.setdefault(str(row["title"]).casefold(), []).append(i)

    @classmethod
    def from_csv(cls, path: str = DEFAULT_DATA_PATH) -> "NetflixRecommender":
        return cls(load_dataset(path))

    def find_title_indices(self, title: str) -> list[int]:
        return self._title_lookup.get(str(title).casefold().strip(), [])

    def recommend(
        self,
        title: str,
        top_n: int = 10,
        content_type: str | None = None,
        exclude_same_title: bool = True,
    ) -> pd.DataFrame:
        """Return the most similar titles, optionally filtering by Movie/TV Show."""
        indices = self.find_title_indices(title)
        if not indices:
            # Helpful partial title search for the UI.
            candidates = self.df[self.df["title"].str.contains(title, case=False, na=False)]
            if len(candidates):
                raise ValueError(
                    "Exact title not found. Similar title matches: "
                    + ", ".join(candidates["title"].head(8).tolist())
                )
            raise ValueError(f"Title '{title}' was not found in the dataset.")

        # If same title exists as both a movie and a show, use requested type;
        # otherwise select the first matching record.
        source_idx = indices[0]
        if content_type:
            type_matches = [i for i in indices if self.df.iloc[i]["type"].casefold() == content_type.casefold()]
            if type_matches:
                source_idx = type_matches[0]

        source = self.df.iloc[source_idx]
        scores = cosine_similarity(self.matrix[source_idx], self.matrix).ravel()
        ranked = np.argsort(scores)[::-1]

        rows = []
        for idx in ranked:
            row = self.df.iloc[int(idx)]
            if exclude_same_title and str(row["title"]).casefold() == str(source["title"]).casefold():
                continue
            if content_type and str(row["type"]).casefold() != content_type.casefold():
                continue
            rows.append((int(idx), float(scores[idx])))
            if len(rows) >= max(0, int(top_n)):
                break

        output = []
        source_genres = self._split_categories(source.get("listed_in", ""))
        for idx, score in rows:
            row = self.df.iloc[idx]
            candidate_genres = self._split_categories(row.get("listed_in", ""))
            shared = sorted(source_genres & candidate_genres)
            result = row.to_dict()
            result["similarity"] = score
            result["similarity_percent"] = round(score * 100, 1)
            result["shared_categories"] = ", ".join(shared) if shared else "Shared metadata"
            output.append(result)
        return pd.DataFrame(output)

    @staticmethod
    def _split_categories(value: object) -> set[str]:
        return {part.strip().casefold() for part in str(value or "").split(",") if part.strip()}

    def title_options(self) -> pd.DataFrame:
        """Title/type pairs suitable for a selection control."""
        return self.df[["title", "type"]].drop_duplicates().sort_values(
            ["title", "type"], kind="stable"
        ).reset_index(drop=True)

    def similar_by_index(self, index: int, top_n: int = 10) -> pd.DataFrame:
        """Index-based helper used by evaluation."""
        row = self.df.iloc[index]
        results = self.recommend(str(row["title"]), top_n=top_n, content_type=str(row["type"]))
        return results


def genre_overlap(a: object, b: object) -> float:
    """Jaccard overlap between comma-separated category labels."""
    left = NetflixRecommender._split_categories(a)
    right = NetflixRecommender._split_categories(b)
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def evaluate_recommender(
    recommender: NetflixRecommender,
    sample_size: int = 100,
    k: int = 5,
    random_state: int = 42,
) -> dict:
    """Offline proxy evaluation using genre overlap as weak relevance labels.

    This is a proxy rather than a ground-truth user satisfaction metric.
    """
    df = recommender.df
    eligible = np.arange(len(df))
    if len(eligible) == 0:
        return {"titles_evaluated": 0, "precision_at_k_proxy": 0.0, "mean_genre_overlap": 0.0}

    rng = np.random.default_rng(random_state)
    chosen = rng.choice(eligible, size=min(sample_size, len(eligible)), replace=False)
    precision_values, overlap_values = [], []

    for idx in chosen:
        source = df.iloc[int(idx)]
        # Avoid ambiguity by choosing within source content type.
        scores = cosine_similarity(recommender.matrix[int(idx)], recommender.matrix).ravel()
        scores[int(idx)] = -1
        same_type = df["type"].astype(str).str.casefold().eq(str(source["type"]).casefold()).to_numpy()
        scores[~same_type] = -1
        order = np.argsort(scores)[::-1]
        order = [int(j) for j in order if scores[j] >= 0][:k]
        if not order:
            continue

        overlaps = [genre_overlap(source.get("listed_in", ""), df.iloc[j].get("listed_in", "")) for j in order]
        overlap_values.extend(overlaps)
        # Proxy relevance threshold: at least half of the source's genre set overlaps.
        precision_values.append(float(np.mean([value >= 0.5 for value in overlaps])))

    return {
        "titles_evaluated": int(len(precision_values)),
        "k": int(k),
        "precision_at_k_proxy": float(np.mean(precision_values)) if precision_values else 0.0,
        "mean_genre_overlap": float(np.mean(overlap_values)) if overlap_values else 0.0,
        "note": "Proxy metrics use category overlap, not user ratings or viewing behavior.",
    }
