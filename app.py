import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os

# Configuration de la page
st.set_page_config(page_title="Admission Predictor", layout="wide")

# ---------------------------------------------------------
# SÉCURITÉ : VÉRIFICATION DU FICHIER D'ENTRAÎNEMENT
# ---------------------------------------------------------
TRAIN_FILE = "ex2data1.txt"

if not os.path.exists(TRAIN_FILE):
    st.error(f"Erreur critique : Le fichier '{TRAIN_FILE}' est absent du dépôt GitHub.")
    st.info("Assurez-vous d'avoir envoyé le fichier .txt sur votre repository.")
    st.stop() # Arrête l'exécution ici pour éviter un crash plus bas

# ---------------------------------------------------------
# ENTRAÎNEMENT AVEC CACHE
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    try:
        # header=None car ex2data1 n'a pas de noms de colonnes
        data = pd.read_csv(TRAIN_FILE, header=None)
        X = data[[0, 1]].values # .values pour éviter les erreurs de noms de colonnes
        y = data[2].values
        
        model = LogisticRegression()
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement : {e}")
        return None

model = train_model()

# ---------------------------------------------------------
# INTERFACE UTILISATEUR
# ---------------------------------------------------------
st.title("🎓 Système de Prédiction d'Admission")
st.markdown("Cette application utilise un modèle de **Régression Logistique** entraîné sur les données Coursera.")

# Barre latérale pour les instructions
with st.sidebar:
    st.header("Instructions")
    st.write("1. Importez un fichier CSV/Excel.")
    st.write("2. Sélectionnez les colonnes des notes.")
    st.write("3. Cliquez sur Prédire.")

uploaded_file = st.file_uploader("Choisir un fichier de scores", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Lecture robuste
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("Fichier chargé avec succès !")
        
        # Choix des colonnes
        col1, col2 = st.columns(2)
        with col1:
            ex1 = st.selectbox("Note Examen 1", df.columns)
        with col2:
            ex2 = st.selectbox("Note Examen 2", df.columns)

        if st.button("Lancer l'analyse des admissions"):
            # Extraction des données
            X_input = df[[ex1, ex2]].values
            
            # Prédiction
            preds = model.predict(X_input)
            probs = model.predict_proba(X_input)[:, 1]
            
            # Formater les résultats
            df['Décision'] = ["Admis " if p == 1 else "Refusé " for p in preds]
            df['Confiance (%)'] = (probs * 100).round(2)
            
            st.divider()
            st.subheader("Résultats")
            st.dataframe(df, use_container_width=True)
            
            # Exportation
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(" Télécharger le rapport CSV", csv, "resultats.csv", "text/csv")

    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")

else:
    st.info("En attente d'un fichier pour commencer...")