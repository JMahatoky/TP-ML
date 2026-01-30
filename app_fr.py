import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

# --- CONFIGURATION DE LA PAGE (DOIT ÊTRE EN PREMIER) ---
st.set_page_config(page_title="Détecteur de Spam - ISPM", page_icon="🚫", layout="centered")

# --- CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    return stopwords.words('french')

@st.cache_resource
def load_model():
    model = joblib.load('model_spam_fr.pkl')
    tfidf = joblib.load('vectorizer_fr.pkl')
    return model, tfidf

stop_words_fr = load_nltk()
model, tfidf = load_model()

# --- LOGIQUE DE TRAITEMENT ---
def clean_text_fr(text):
    text = str(text).lower()
    text = "".join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words_fr]
    return " ".join(tokens)

# --- INTERFACE UTILISATEUR ---
st.title("🚫 Détecteur de Spam SMS")
st.markdown("### Projet d'Intelligence Artificielle - ISPM")
st.write("Saisissez un message pour vérifier s'il s'agit d'un message légitime (Ham) ou d'une tentative de Spam.")

# Sidebar : Seuil de décision (Bonus demandé)
st.sidebar.header("Configuration")
threshold = st.sidebar.slider(
    "Seuil de détection (Spam)", 
    min_value=0.0, max_value=1.0, value=0.5, step=0.05
)
st.sidebar.info(f"Seuil actuel : {threshold}. Un message est considéré comme Spam si sa probabilité dépasse ce score.")

# Zone de texte
message_input = st.text_area("Entrez le message :", height=150, placeholder="Ex: Gagnez 1000€ en cliquant sur ce lien...")

if st.button("Analyser le message"):
    if message_input.strip():
        # 1. Prétraitement
        cleaned = clean_text_fr(message_input)
        
        # 2. Vectorisation
        vec = tfidf.transform([cleaned])
        
        # 3. Prédiction des probabilités
        # probas[0][1] correspond à la classe Spam (1)
        probas = model.predict_proba(vec)
        spam_score = probas[0][1]
        
        # 4. Décision selon le seuil
        st.divider()
        if spam_score >= threshold:
            st.error(f"### 🚨 Résultat : SPAM")
        else:
            st.success(f"### ✅ Résultat : HAM (Légitime)")
        
        # Affichage du score de confiance
        st.write(f"**Score de suspicion :** {spam_score*100:.2f}%")
        st.progress(spam_score)
        
        # Détails techniques (Bonus)
        with st.expander("Voir les détails techniques"):
            st.write(f"Seuil appliqué : {threshold}")
            st.write(f"Probabilité exacte (Classe 1) : {spam_score}")
            st.write(f"Texte nettoyé envoyé au modèle : *{cleaned}*")
    else:
        st.warning("Veuillez entrer un texte avant d'analyser.")

# Footer
st.markdown("---")
st.caption("© 2026 - Institut Supérieur Polytechnique de Madagascar (ISPM)")
