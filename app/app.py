import streamlit as st
import pickle
import re
import requests
from io import BytesIO

# Dropbox direct download linkleri
model_url = 'https://www.dropbox.com/scl/fi/4cmeox8n41z4v19iwlu2j/logistic_model.pkl?dl=1&rlkey=5j6ke6nsp5tzl2gf3ebrx9n8q'
vectorizer_url = 'https://www.dropbox.com/scl/fi/c8b2gd3pr4bngpuuzfw0r/tfidf_vectorizer.pkl?dl=1&rlkey=nr36vphaxtzn2u2c9yh4p2thm'

# Modeli indir ve bellekten yükle
def load_pickle_from_url(url):
    r = requests.get(url)
    r.raise_for_status()
    return pickle.load(BytesIO(r.content))

model = load_pickle_from_url(model_url)
vectorizer = load_pickle_from_url(vectorizer_url)

# Minimal stopwords ve basit temizleme fonksiyonu
stop_words = set([
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now"
])

def simple_stem(word):
    if word.endswith('ing') or word.endswith('ed'):
        return word[:-3]
    if word.endswith('s'):
        return word[:-1]
    return word

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [simple_stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

import streamlit as st

st.title("📰 TrueScan - Fake News Detector (In-memory Model Load)")

news_input = st.text_area("📝 Lütfen bir haber başlığı ve metni girin:")

if st.button("🔍 Analiz Et"):
    if news_input.strip() == "":
        st.warning("⚠️ Lütfen boş bırakmayın.")
    else:
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]
        fake_score = prob[0] * 100
        real_score = prob[1] * 100
        if prediction == 1:
            st.success("✅ Bu haber muhtemelen GERÇEK.")
        else:
            st.error("🚨 Bu haber muhtemelen SAHTE.")
        st.write("### 🔎 Güven Skorları")
        st.write(f"- 🚨 FAKE: {fake_score:.2f}%")
        st.write(f"- ✅ REAL: {real_score:.2f}%")
