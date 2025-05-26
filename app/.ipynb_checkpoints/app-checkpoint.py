import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Gerekli NLTK verisi
nltk.download('stopwords')
# Model ve vectorizer'ı yükle
with open('../models/logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
# Stopwords ve stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
# Metin temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)
# Streamlit arayüzü
st.title("📰 Fake News Detector (Logistic Regression)")
st.write("Bu uygulama, haberin gerçek mi sahte mi olduğunu tahmin eder ve iki sınıfa ait güven skorlarını gösterir.")
# Kullanıcıdan haber girişi
news_input = st.text_area("📝 Lütfen bir haber başlığı ve metni girin:")
if st.button("🔍 Analiz Et"):
    if news_input.strip() == "":
        st.warning("⚠️ Lütfen boş bırakmayın.")
    else:
        # Metni temizle ve dönüştür
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned])
        # Tahmin ve olasılık
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]
        # Skorları ayrı göster
        fake_score = prob[0] * 100
        real_score = prob[1] * 100
        if prediction == 1:
            st.success("✅ Bu haber muhtemelen GERÇEK.")
        else:
            st.error("🚨 Bu haber muhtemelen SAHTE.")
        st.write("### 🔎 Güven Skorları")
        st.write(f"- 🚨 FAKE: {fake_score:.2f}%")
        st.write(f"- ✅ REAL: {real_score:.2f}%")