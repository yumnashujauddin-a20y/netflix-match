"""Run a reproducible offline evaluation from the command line."""
import argparse
import json
from pathlib import Path

from recommender import NetflixRecommender, evaluate_recommender, load_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Netflix content recommender.")
    parser.add_argument("--data", default="data/Dataset.csv", help="Path to Dataset.csv")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="evaluation_results.json")
    args = parser.parse_args()

    recommender = NetflixRecommender(load_dataset(args.data))
    results = evaluate_recommender(
        recommender,
        sample_size=args.sample_size,
        k=args.k,
        random_state=args.seed,
    )
    results["dataset_path"] = str(args.data)
    results["dataset_rows"] = int(len(recommender.df))
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
