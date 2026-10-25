import nltk
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer


def _download_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


def extractive_summary(text: str, sentences_count: int = 3) -> str:
    """Generate an extractive summary using TextRank."""
    _download_nltk_resources()

    parser = PlaintextParser.from_string(
        text,
        Tokenizer("english")
    )

    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, sentences_count)

    return " ".join(str(sentence) for sentence in sentences)
