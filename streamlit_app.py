import streamlit as st
st.set_page_config(page_title="Fake News Detector", layout="centered")

import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from topic_utils import identify_topic, get_similar_articles, extract_text_from_url

# ---------- Load Models & Data ----------
@st.cache_resource
def load_models():
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert")
    distilbert_model = DistilBertForSequenceClassification.from_pretrained("models/distilbert").to("cpu")
    news_df = pd.read_csv("data/news_with_topics.csv")  # contains 'text', 'label', 'topic'
    return tokenizer, distilbert_model, news_df

tokenizer, distilbert_model, news_df = load_models()

# ---------- Helper Functions ----------
def predict_label(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return "Real" if probs[0][1] > probs[0][0] else "Fake", probs[0].tolist()

# ---------- Streamlit App UI ----------
st.title("🧠 Fake News Detector with Topic Modeling")

user_input = st.text_area("Paste a news article *or* URL:", height=200)
is_url = user_input.strip().startswith("http://") or user_input.strip().startswith("https://")

if st.button("Analyze") and user_input.strip():
    if is_url:
        with st.spinner("Extracting text from URL..."):
            extracted_text = extract_text_from_url(user_input.strip())
        if extracted_text:
            st.success("📰 Text successfully extracted from URL.")
            processed_text = extracted_text
        else:
            st.error("Failed to extract text from URL. Please check the link.")
            processed_text = ""
    else:
        processed_text = user_input.strip()

    if processed_text:
        with st.spinner("Analyzing the input text..."):
            label, probs = predict_label(processed_text)
            topic = identify_topic(processed_text)
            similar_articles = get_similar_articles(processed_text, topic, news_df)

        st.subheader("🔍 Prediction Result")
        st.markdown(f"**🗞️ This news is predicted to be: `{label}`**")
        st.text(f"Confidence → Fake: {probs[0]:.2f}, Real: {probs[1]:.2f}")
        
        st.subheader("📰 Top 3 Similar News Articles")
        if not similar_articles.empty:
            for _, row in similar_articles.iterrows():
                st.markdown(f"**Label**: `{row['label']}`")
                st.markdown(f"**Text**: {row['text'][:500]}...")
                st.markdown("---")
        else:
            st.info("No similar articles found in the same topic.")