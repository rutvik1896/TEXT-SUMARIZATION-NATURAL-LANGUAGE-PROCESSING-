import streamlit as st

from src.abstractive import abstractive_summary
from src.extractive import extractive_summary
from src.evaluation import calculate_rouge


st.set_page_config(
    page_title="Text Summarization",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Text Summarization Using Transformer Models")
st.write(
    "Compare extractive TextRank summarization with abstractive "
    "BART summarization."
)

default_text = """Artificial intelligence is becoming an important technology in many areas of modern life. 
It is used in healthcare, education, finance, transportation and entertainment. 
Machine learning allows computers to learn patterns from data and make predictions. 
Deep learning uses neural networks with many layers to solve complex problems. 
Natural language processing helps computers understand and generate human language. 
One important NLP task is text summarization, which creates a shorter version of a long document while preserving important information."""

article = st.text_area(
    "Paste a long article here:",
    value=default_text,
    height=300
)

reference = st.text_area(
    "Optional reference summary for ROUGE evaluation:",
    height=150,
    placeholder="Paste a human-written/reference summary here if you have one."
)

if st.button("Generate Summaries", type="primary"):
    if len(article.strip()) < 50:
        st.warning("Please enter a longer article.")
        st.stop()

    with st.spinner("Generating extractive summary..."):
        extractive = extractive_summary(article)

    with st.spinner("Loading BART and generating abstractive summary..."):
        abstractive = abstractive_summary(article)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Extractive Summary - TextRank")
        st.write(extractive)

    with col2:
        st.subheader("Abstractive Summary - BART")
        st.write(abstractive)

    if reference.strip():
        scores_ext = calculate_rouge(extractive, reference)
        scores_abs = calculate_rouge(abstractive, reference)

        st.subheader("ROUGE Comparison")

        st.dataframe(
            {
                "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
                "TextRank (Extractive)": [
                    scores_ext["rouge1"],
                    scores_ext["rouge2"],
                    scores_ext["rougeL"],
                ],
                "BART (Abstractive)": [
                    scores_abs["rouge1"],
                    scores_abs["rouge2"],
                    scores_abs["rougeL"],
                ],
            },
            hide_index=True,
        )
    else:
        st.info(
            "Add a reference summary to calculate ROUGE-1, ROUGE-2 and ROUGE-L."
        )

st.markdown("---")
st.caption("College NLP Project | BART + TextRank + ROUGE")
