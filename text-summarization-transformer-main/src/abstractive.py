from functools import lru_cache

from transformers import pipeline


@lru_cache(maxsize=1)
def get_summarizer():
    """Load BART once and reuse it during the application session."""
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )


def _split_text(text: str, max_words: int = 500):
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def abstractive_summary(text: str) -> str:
    """Generate an abstractive summary using BART.

    Long articles are split into manageable chunks because BART has
    a finite input length.
    """
    summarizer = get_summarizer()
    chunks = _split_text(text)

    summaries = []

    for chunk in chunks:
        if len(chunk.split()) < 30:
            continue

        result = summarizer(
            chunk,
            max_length=120,
            min_length=30,
            do_sample=False,
            truncation=True
        )
        summaries.append(result[0]["summary_text"])

    if not summaries:
        result = summarizer(
            text,
            max_length=120,
            min_length=20,
            do_sample=False,
            truncation=True
        )
        return result[0]["summary_text"]

    if len(summaries) == 1:
        return summaries[0]

    # Second-pass compression for multiple chunks.
    combined = " ".join(summaries)

    result = summarizer(
        combined,
        max_length=150,
        min_length=40,
        do_sample=False,
        truncation=True
    )

    return result[0]["summary_text"]
