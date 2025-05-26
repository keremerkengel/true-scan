import streamlit as st
import pickle
import re
import requests
from io import BytesIO

# Minimal stopwords listesi
stop_words = set([
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now"
])

# Basit kelime kök bulucu
def simple_stem(word):
    if word.endswith('ing') or word.endswith('ed'):
        return word[:-3]
    if word.endswith('s'):
        return word[:-1]
    return word

# Dropbox direct download linkleri (dl=1)
model_url = 'https://www.dropbox.com/scl/fi/4cmeox8n41z4v19iwlu2j/logistic_model.pkl?rlkey=5j6ke6nsp5tzl2gf3ebrx9n8q&st=paa7gi01&dl=1'
vectorizer_url = 'https://www.dropbox.com/scl/fi/c8b2gd3pr4bngpuuzfw0r/tfidf_vectorizer.pkl?rlkey=nr36vphaxtzn2u2c9yh4p2thm&st=ilcl5yn5&dl=1'


def load_pickle_from_url(url):
    r = requests.get(url)
    r.raise_for_status()
    return pickle.load(BytesIO(r.content))

# Model ve vectorizer'ı belleğe yükle
model = load_pickle_from_url(model_url)
vectorizer = load_pickle_from_url(vectorizer_url)

# Metin temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [simple_stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit arayüzü
st.title("📰 TrueScan - Fake News Detector (Dropbox Model Bellekten)")

st.write("Bu uygulama, haberin gerçek mi sahte mi olduğunu tahmin eder ve iki sınıfa ait güven skorlarını gösterir.")

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
