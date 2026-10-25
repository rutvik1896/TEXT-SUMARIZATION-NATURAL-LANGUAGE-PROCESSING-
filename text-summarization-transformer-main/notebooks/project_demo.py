"""
Simple command-line demo.

Run:
    python notebooks/project_demo.py
"""

from src.abstractive import abstractive_summary
from src.extractive import extractive_summary


ARTICLE = """
Artificial intelligence is transforming many industries around the world.
Healthcare organizations use AI to analyze medical images and support clinical
decision making. Financial institutions use machine learning to detect unusual
transactions and improve risk analysis. In education, AI systems can provide
personalized learning experiences and automated feedback. Natural language
processing allows computers to work with human language, including tasks such
as translation, question answering and text summarization. Text summarization
is useful because long reports and news articles can contain a large amount of
information. Extractive methods select important sentences from the original
document, while abstractive methods generate a shorter version using newly
generated language. Transformer models such as BART have become widely used
for abstractive summarization because they can learn relationships between
different parts of a document.
"""


if __name__ == "__main__":
    print("=== Extractive Summary ===")
    print(extractive_summary(ARTICLE))

    print("\n=== Abstractive BART Summary ===")
    print(abstractive_summary(ARTICLE))
