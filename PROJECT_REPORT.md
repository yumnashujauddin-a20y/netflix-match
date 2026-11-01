# Project Report: Netflix Content Recommendation System

## 1. Abstract

This project implements a content-based recommendation system that suggests Netflix movies and television shows with metadata similar to a selected title. The system uses TF-IDF feature extraction and cosine similarity to rank candidate titles. An interactive Streamlit application supports title selection, similarity explanations, catalogue exploration, CSV export, and a proxy evaluation workflow.

## 2. Problem statement

A large catalogue makes it difficult for viewers to discover related content. The objective is to build a reproducible system that recommends titles based on content-related attributes available in the supplied dataset.

## 3. Dataset and scope

The supplied CSV contains 8,790 rows and 10 columns: show ID, type, title, director, country, date added, release year, rating, duration, and listed categories. It does not include plot summaries, cast lists, user ratings, or viewing histories. The system therefore limits its feature engineering to available metadata.

## 4. Objectives

- Prepare clean title metadata.
- Convert text attributes to numeric representations.
- Calculate content similarity scores.
- Generate ranked recommendations for a selected title.
- Evaluate recommendation quality with a transparent proxy metric.
- Provide an interactive interface for users.

## 5. Methodology

### 5.1 Data preprocessing

The implementation fills missing text metadata, removes blank titles, converts release years to numeric values, and removes duplicate title/type pairs.

### 5.2 Feature engineering

The system combines listed categories, content type, director, country, rating, release year, and duration into a single text representation. Categories receive the strongest weight because they directly describe the content.

### 5.3 TF-IDF vectorization

Term Frequency–Inverse Document Frequency transforms text into sparse vectors. Terms that distinguish titles receive more influence than terms that appear broadly across many titles. Unigrams and bigrams are used.

### 5.4 Cosine similarity

Cosine similarity compares the vector of a selected title with the vectors of other titles. The highest-scoring candidates are returned, with the source title excluded. The interface can optionally restrict recommendations to the same content type.

### 5.5 User interface

The Streamlit app offers:
- title-based recommendation search;
- metadata and shared-category explanations;
- configurable top-K output;
- catalogue search and filters;
- CSV downloads;
- an offline evaluation panel.

## 6. Evaluation

The project computes a genre-overlap proxy over a reproducible random sample. For each sampled title, the top-K same-type recommendations are checked for category overlap. A candidate is considered proxy-relevant when its Jaccard overlap with the source categories is at least 0.5.

This is not a user-grounded metric. It measures alignment with a category heuristic, not whether viewers actually like the results. Report the generated metric values from `python evaluate.py` only after running the evaluation on the included dataset. Do not claim a score before executing the evaluation.

For stronger evaluation, use a human-judged relevance set and report Precision@K, Recall@K, and NDCG@K. If user interaction data becomes available, evaluate offline against held-out interactions.

## 7. Expected outcome

Given a selected Netflix title, the system returns a ranked table of similar titles, similarity scores, metadata, and shared categories. Recommendations should generally emphasize category matches, while other metadata can help break ties.

## 8. Limitations

- No plot descriptions or cast attributes are available in the provided data.
- No user-specific history or explicit preferences are available, so personalization is title-conditioned rather than individual-user learned.
- Similarity scores are not probabilities.
- The evaluation uses a proxy rather than ground-truth user judgments.
- Metadata quality and missing values affect ranking.

## 9. Future improvements

1. Add a user profile containing preferred categories and disliked content.
2. Gather explicit feedback and create a human-reviewed evaluation set.
3. Tune feature weights with a validation set.
4. Add richer text metadata if a licensed, compatible dataset is available.
5. Deploy the Streamlit app and monitor user feedback.

## 10. Conclusion

The project demonstrates a complete baseline content-based recommendation workflow using Python, pandas, scikit-learn, and Streamlit. It covers data preparation, feature extraction, similarity scoring, recommendation generation, and evaluation, while clearly documenting the limits imposed by the available dataset.
