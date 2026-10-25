# Project Report Content

## Title

Text Summarization Using Transformer Models

## Problem Statement

Long news articles and documents contain large amounts of information. Reading
the complete document can take significant time. This project develops a
summarization system that creates shorter summaries while retaining important
information.

## Objectives

1. Develop an extractive summarization system using TextRank.
2. Develop an abstractive summarization system using pre-trained BART.
3. Evaluate generated summaries using ROUGE-1, ROUGE-2 and ROUGE-L.
4. Compare extractive and abstractive summarization.
5. Provide a simple interface for users to summarize text.

## Methodology

1. Collect article and reference-summary data from CNN/DailyMail.
2. Preprocess the input article.
3. Generate an extractive summary using TextRank.
4. Generate an abstractive summary using BART.
5. Compare generated summaries with the reference summary.
6. Calculate ROUGE metrics.
7. Present the results in a comparison table.

## Extractive Summarization

Extractive summarization selects important sentences from the original text.
TextRank represents sentences as nodes and uses relationships between them to
identify important sentences.

## Abstractive Summarization

Abstractive summarization generates a new summary rather than only selecting
sentences from the source. BART is a sequence-to-sequence Transformer model
that is pre-trained for language generation tasks.

## Evaluation

ROUGE-1 measures unigram overlap.
ROUGE-2 measures bigram overlap.
ROUGE-L measures overlap based on the longest common subsequence.

## Comparison

Extractive summarization preserves original wording and is generally less
likely to introduce newly generated wording. Abstractive summarization can
produce more compact and naturally phrased summaries, but generated text
should be checked for factual accuracy.

## Software Requirements

- Python 3.11 or 3.12
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Evaluate
- ROUGE Score
- NLTK
- Sumy
- Streamlit

## Hardware Requirements

- 8 GB RAM or more recommended
- Modern multi-core CPU
- Internet connection for first-time model/dataset download
- GPU is optional

## Limitations

- BART model loading can require substantial RAM.
- CPU inference can be slow.
- ROUGE measures textual overlap and does not fully measure factuality.
- Very long documents require chunking.

## Future Scope

- T5 comparison
- Multilingual summarization
- Factuality evaluation
- Better long-document models
- Cloud deployment
- User authentication and document history

## Conclusion

The project demonstrates two approaches to automatic text summarization and
evaluates them using standard ROUGE metrics. It provides a practical example
of applying NLP and Transformer models to news articles.
