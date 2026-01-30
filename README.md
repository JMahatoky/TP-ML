
# TP-ML
a- ISPM – Institut Supérieur Polytechnique de Madagascar: www.ispm-edu.com
b-  Josia RANERIA Mahatoky 
    RAKOTOARISOA Laingo Tsilavina
    HASINAVALONA Lovatiana 
    ANDRIAMAROAHINA Harimanantsoa
    Cois NEWMAN Noah

c- La description du stack technologique
Le projet est développé avec un stack moderne orienté Data Science et Web Rapide :
Langage : Python 3.x, le standard pour le Machine Learning.
Traitement de données : Pandas et NumPy pour la manipulation des fichiers CSV et des matrices.
NLP (Natural Language Processing) : NLTK, utilisé pour le nettoyage linguistique et la gestion des "mots vides" (Stopwords).
Machine Learning : Scikit-learn, utilisé pour la vectorisation des textes, l'entraînement du modèle et le calcul des métriques de performance.
Visualisation : Matplotlib et Seaborn pour l'analyse exploratoire et la matrice de confusion.
Persistance du modèle : Joblib, pour sauvegarder le modèle entraîné sous forme de fichier binaire .pkl.
Déploiement : Streamlit, pour l'interface utilisateur web accessible par URL.

d- La description du processus et du modèle
Le processus suit un pipeline de transformation de données rigoureux pour passer d'un texte brut à une prédiction mathématique :

Prétraitement du texte : Le texte est converti en minuscules, la ponctuation est supprimée, et les mots courants sans valeur sémantique (stopwords français) sont filtrés via la fonction clean_text_fr.

Vectorisation TF-IDF : Transformation du texte nettoyé en vecteurs numériques. Cette méthode permet de donner plus de poids aux mots rares et discriminants pour le spam.

Entraînement : Division des données en un ensemble d'entraînement (80%) et un ensemble de test (20%) pour garantir la capacité de généralisation.

Classification : Utilisation de l'algorithme de Régression Logistique pour séparer les classes "Spam" et "Ham".

Évaluation : Génération d'un rapport de classification (Précision, Recall, F1-Score) et d'une matrice de confusion.


e- Les méthodes ML
Nous avons implémenté deux méthodes majeures de Machine Learning :

TF-IDF (Term Frequency-Inverse Document Frequency) : C'est une méthode statistique de pondération. Elle évalue l'importance d'un mot dans un message par rapport à l'ensemble du dataset. Dans votre code, nous avons limité le modèle aux 2500 caractéristiques les plus pertinentes pour optimiser les performances.

Régression Logistique : Bien que son nom indique "régression", c'est un puissant algorithme de classification binaire. Il estime la probabilité qu'un message appartienne à la classe "Spam" en utilisant une fonction sigmoïde. C'est un modèle robuste, rapide et très efficace pour la classification de textes courts.

f- Les datasets utilisés
Le modèle a été entraîné sur un dataset hybride et multilingue :Source : Fichier data-en-hi-de-fr.csv.Contenu : Ce dataset est particulièrement riche car il contient des messages traduits ou rédigés en Français, ce qui répond directement à l'objectif principal du hackathon (base principale en français).Structure : * text_fr : La colonne contenant les messages en français (notre variable d'entrée $X$).labels : La colonne cible (notre variable de sortie $y$) indiquant si le message est légitime ou indésirable.Répartition : Une analyse exploratoire (EDA) a été réalisée pour visualiser la proportion entre les spams et les hams afin de s'assurer de la qualité de l'apprentissage.

g-
