import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Configuration NLTK pour le déploiement
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords')
    return stopwords.words('french')

stop_words_fr = setup_nltk()

# Chargement du modèle et du vectoriseur
@st.cache_resource
def load_assets():
    model = joblib.load('model_spam_fr.pkl')
    tfidf = joblib.load('vectorizer_fr.pkl')
    return model, tfidf

model, tfidf = load_assets()

# Fonction de nettoyage identique à l'entraînement
def clean_text_fr(text):
    text = str(text).lower()
    text = "".join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words_fr]
    return " ".join(tokens)

# Interface Streamlit
st.set_page_config(page_title="Analyseur de Spam ISPM", page_icon="✉️")

st.title("✉️ Détecteur de Spam SMS")
st.write("Projet ISPM - Analyse de messages en Français")

# Sidebar pour le Seuil de décision
st.sidebar.header("Paramètres du modèle")
threshold = st.sidebar.slider(
    "Seuil de décision (Threshold)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05,
    help="Augmenter le seuil rend le modèle plus 'prudent' avant de marquer un message comme Spam."
)

# Zone de saisie
user_input = st.text_area("Entrez le message à analyser :", height=150)

if st.button("Analyser"):
    if user_input.strip():
        # 1. Prétraitement
        cleaned_msg = clean_text_fr(user_input)
        
        # 2. Vectorisation
        vectorized_msg = tfidf.transform([cleaned_msg])
        
        # 3. Prédiction des probabilités
        # probas[0][0] = Ham, probas[0][1] = Spam
        probas = model.predict_proba(vectorized_msg)
        spam_probability = probas[0][1]
        
        # 4. Application du seuil personnalisé
        is_spam = spam_probability >= threshold
        
        # Affichage des résultats
        st.subheader("Résultat de l'analyse")
        
        col1, col2 = st.columns(2)
        with col1:
            if is_spam:
                st.error("🚨 Classification : SPAM")
            else:
                st.success("✅ Classification : HAM (Légitime)")
        
        with col2:
            st.metric("Probabilité de Spam", f"{spam_probability*100:.2f}%")

        # Barre de progression visuelle
        st.progress(spam_probability)
        
        if is_spam:
            st.warning(f"Note : Ce message dépasse le seuil de {threshold*100:.0f}% défini.")
    else:
        st.info("Veuillez saisir un texte pour lancer l'analyse.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© ISPM - Institut Supérieur Polytechnique de Madagascar")