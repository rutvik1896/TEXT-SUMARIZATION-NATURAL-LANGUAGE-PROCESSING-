# Text Summarization Using Transformer Models

A college-level NLP project that compares **extractive summarization using TextRank** with **abstractive summarization using BART**.

## Features

- Pre-trained `facebook/bart-large-cnn` model
- Extractive summarization using TextRank
- CNN/DailyMail dataset support
- ROUGE-1, ROUGE-2 and ROUGE-L evaluation
- Simple Streamlit web interface
- No need to store the full dataset or model files in GitHub

## Project Architecture

```text
News Article
    |
    +--------------------+
    |                    |
TextRank              BART
Extractive          Abstractive
Summary             Summary
    |                    |
    +---------+----------+
              |
        ROUGE Evaluation
     ROUGE-1 / ROUGE-2 / ROUGE-L
```

## Requirements

Recommended Python version: **3.11 or 3.12**.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit application

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Run dataset evaluation

```bash
python evaluate_dataset.py
```

The evaluation script uses Hugging Face streaming and only evaluates a small number of examples by default, so it does not process the entire CNN/DailyMail training dataset.

You can change the number of examples:

```bash
python evaluate_dataset.py --samples 20
```

## Important

Do not upload:
- Hugging Face model cache
- CNN/DailyMail dataset files
- Python virtual environment
- API keys or passwords
- `__pycache__` folders

These are excluded by `.gitignore`.

## Dataset

The project is designed for the CNN/DailyMail dataset available through Hugging Face Datasets. The dataset contains news articles and reference summaries.

## Model

The abstractive summarizer uses:

`facebook/bart-large-cnn`

The model is downloaded automatically the first time it is used.

## Evaluation

ROUGE compares generated summaries with reference summaries:

- **ROUGE-1:** unigram overlap
- **ROUGE-2:** bigram overlap
- **ROUGE-L:** longest common subsequence based overlap

Higher ROUGE values indicate greater lexical overlap with the reference summary, but ROUGE alone does not fully measure factual accuracy or readability.

## Project Structure

```text
text_summarization_project/
│
├── app.py
├── evaluate_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── abstractive.py
│   ├── extractive.py
│   └── evaluation.py
│
└── notebooks/
    └── project_demo.py
```

## Future Scope

- Add T5 as a second abstractive model
- Add more datasets
- Add factuality evaluation
- Add multilingual summarization
- Deploy the Streamlit application
