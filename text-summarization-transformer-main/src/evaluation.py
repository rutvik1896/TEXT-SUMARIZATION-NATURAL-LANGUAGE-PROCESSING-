import evaluate


def calculate_rouge(prediction: str, reference: str) -> dict:
    """Calculate ROUGE-1, ROUGE-2 and ROUGE-L F1 scores."""
    rouge = evaluate.load("rouge")

    result = rouge.compute(
        predictions=[prediction],
        references=[reference],
        use_stemmer=True
    )

    return {
        "rouge1": round(result["rouge1"], 4),
        "rouge2": round(result["rouge2"], 4),
        "rougeL": round(result["rougeL"], 4),
    }
