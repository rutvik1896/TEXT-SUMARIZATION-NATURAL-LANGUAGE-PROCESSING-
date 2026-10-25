import argparse

from datasets import load_dataset

from src.abstractive import abstractive_summary
from src.extractive import extractive_summary
from src.evaluation import calculate_rouge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading {args.samples} CNN/DailyMail test examples using streaming...")

    dataset = load_dataset(
        "abisee/cnn_dailymail",
        "3.0.0",
        split="test",
        streaming=True
    )

    dataset = dataset.take(args.samples)

    extractive_scores = []
    abstractive_scores = []

    for index, item in enumerate(dataset, start=1):
        article = item["article"]
        reference = item["highlights"]

        print(f"\nProcessing example {index}/{args.samples}")

        ext = extractive_summary(article)
        abs_summary = abstractive_summary(article)

        ext_scores = calculate_rouge(ext, reference)
        abs_scores = calculate_rouge(abs_summary, reference)

        extractive_scores.append(ext_scores)
        abstractive_scores.append(abs_scores)

        print("Extractive:", ext_scores)
        print("Abstractive:", abs_scores)

    if not extractive_scores:
        print("No examples were processed.")
        return

    metrics = ["rouge1", "rouge2", "rougeL"]

    print("\n=== Average ROUGE Results ===")

    for method, scores in [
        ("Extractive TextRank", extractive_scores),
        ("Abstractive BART", abstractive_scores),
    ]:
        print(f"\n{method}")
        for metric in metrics:
            average = sum(x[metric] for x in scores) / len(scores)
            print(f"{metric}: {average:.4f}")


if __name__ == "__main__":
    main()
