from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
import os
import requests
from bs4 import BeautifulSoup

# --- Dummy topic identifier (fallback mode to keep UI alive) ---
def identify_topic(text):
    return 0  # default topic ID

# --- Return top-k similar articles from full dataset ---
def get_similar_articles(text, topic, news_df, top_k=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([text] + news_df['text'].astype(str).tolist())
    sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    top_indices = sims.argsort()[::-1][:top_k]
    return news_df.iloc[top_indices][['text', 'label']]

# --- Extract text content from a URL ---
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs if len(p.get_text()) > 30)
        return text.strip() if text else None
    except Exception as e:
        print("⚠️ Failed to extract text from URL:", e)
        return None

# Skipping model loading logic to ensure Streamlit UI always starts
print("⚠️ BERTopic disabled. UI will show default topic and fallback similarity.")
