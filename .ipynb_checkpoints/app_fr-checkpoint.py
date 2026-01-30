import streamlit as st

# ⚠️ OBLIGATOIRE : avant tout autre st.*
st.set_page_config(
    page_title="Analyseur de Spam ISPM",
    page_icon="✉️",
    layout="centered"
)

import joblib
import string
import nltk
from nltk.corpus import stopwords

# =========================
# CONFIGURATION NLTK
# =========================
nltk.download('stopwords', quiet=True)

@st.cache_data
def get_stopwords():
    return stopwords.words('french')

stop_words_fr = get_stopwords()

# =========================
# CHARGEMENT DU MODELE
# =========================
@st.cache_resource
def load_assets():
    model = joblib.load('model_spam_fr.pkl')
    tfidf = joblib.load('vectorizer_fr.pkl')
    return model, tfidf

model, tfidf = load_assets()

# =========================
# NETTOYAGE TEXTE
# =========================
def clean_text_fr(text):
    text = str(text).lower()
    text = "".join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words_fr]
    return " ".join(tokens)

# =========================
# INTERFACE STREAMLIT
# =========================
st.title("✉️ Détecteur de Spam SMS")
st.write("Projet ISPM – Analyse de messages en Français")

# Sidebar
st.sidebar.header("Paramètres du modèle")

threshold = st.sidebar.slider(
    "Seuil de décision",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Plus le seuil est élevé, plus le modèle est strict."
)

# Zone de saisie
user_input = st.text_area(
    "Entrez le message à analyser :",
    height=150
)

# =========================
# PREDICTION
# =========================
if st.button("Analyser"):
    if user_input.strip():

        cleaned_msg = clean_text_fr(user_input)
        vectorized_msg = tfidf.transform([cleaned_msg])

        probas = model.predict_proba(vectorized_msg)
        spam_probability = float(probas[0][1])

        is_spam = spam_probability >= threshold

        st.subheader("Résultat de l'analyse")

        col1, col2 = st.columns(2)

        with col1:
            if is_spam:
                st.error("🚨 Classification : SPAM")
            else:
                st.success("✅ Classification : HAM (Légitime)")

        with col2:
            st.metric(
                "Probabilité de Spam",
                f"{spam_probability * 100:.2f}%"
            )

        # Barre de progression sécurisée
        st.progress(min(spam_probability, 1.0))

        if is_spam:
            st.warning(
                f"⚠️ Ce message dépasse le seuil de "
                f"{threshold * 100:.0f}%."
            )

    else:
        st.info("Veuillez saisir un texte pour lancer l'analyse.")

# =========================
# FOOTER
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("© ISPM – Institut Supérieur Polytechnique de Madagascar")
