import streamlit as st
import pickle
import re

stop_words = set([
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now"
])

# PorterStemmer yerine basit kök bulma fonksiyonu ekleyelim (opsiyonel)
def simple_stem(word):
    # Çok basit, sadece -ing, -ed, -s sonlarını keser
    if word.endswith('ing') or word.endswith('ed'):
        return word[:-3]
    if word.endswith('s'):
        return word[:-1]
    return word

# Model ve vectorizer'ı yükle
with open('../models/logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Metin temizleme fonksiyonu (nltk yerine minimal)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [simple_stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit arayüzü
st.title("📰 Fake News Detector (Logistic Regression - Minimal Stopwords)")

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
