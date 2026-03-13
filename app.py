import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os

# Configuration de la page
st.set_page_config(page_title="Admission Predictor", layout="wide")

# ---------------------------------------------------------
# SECURITE : VERIFICATION DU FICHIER D'ENTRAINEMENT
# ---------------------------------------------------------
TRAIN_FILE = "ex2data1.txt"

if not os.path.exists(TRAIN_FILE):
    st.error(f"Erreur : Le fichier '{TRAIN_FILE}' est introuvable.")
    st.stop() 

# ---------------------------------------------------------
# ENTRAINEMENT AVEC CACHE (Evite le ré-entraînement à chaque clic)
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    try:
        # Chargement des données d'origine (ex2data1.txt)
        data = pd.read_csv(TRAIN_FILE, header=None)
        X = data[[0, 1]].values 
        y = data[2].values
        
        # Entraînement unique
        model = LogisticRegression()
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement : {e}")
        return None

# Appel du modèle (sera récupéré du cache si déjà entraîné)
model = train_model()

# ---------------------------------------------------------
# INTERFACE UTILISATEUR
# ---------------------------------------------------------
st.title("Systeme de Prediction d'Admission")

with st.sidebar:
    st.header("Instructions")
    st.write("1. Importez un fichier (CSV, XLSX ou TXT).")
    st.write("2. Choisissez les colonnes correspondantes aux examens.")
    st.write("3. Cliquez sur Predire.")

# Support des formats .csv, .xlsx (slx) et .txt
uploaded_file = st.file_uploader("Choisir un fichier", type=["csv", "xlsx", "txt"])

if uploaded_file is not None:
    try:
        # Lecture selon l'extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            # Pour les fichiers .txt, on tente une lecture standard
            df = pd.read_csv(uploaded_file, header=None)
            # Si le fichier n'a pas d'en-têtes, on nomme les colonnes
            if isinstance(df.columns[0], int):
                df.columns = [f"Colonne {i}" for i in range(len(df.columns))]
            
        st.info("Fichier charge. Veuillez selectionner les colonnes pour l'analyse.")
        
        # Selection des colonnes par l'utilisateur
        col1, col2 = st.columns(2)
        with col1:
            ex1 = st.selectbox("Selectionner Exam 1", df.columns)
        with col2:
            ex2 = st.selectbox("Selectionner Exam 2", df.columns)

        # Bouton Predire
        if st.button("Predire"):
            # Extraction des donnees selectionnees
            X_input = df[[ex1, ex2]].values
            
            # Prediction et Probabilite
            preds = model.predict(X_input)
            probs = model.predict_proba(X_input)[:, 1]
            
            # Ajout des resultats au tableau
            df['Resultat'] = ["Admis" if p == 1 else "Non admis" for p in preds]
            df['Probabilite (%)'] = (probs * 100).round(2)
            
            st.divider()
            st.subheader("Resultats de la prediction")
            st.dataframe(df, use_container_width=True)
            
            # Option de telechargement
            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button("Telecharger les resultats", csv_output, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")

else:
    st.write("En attente d'un fichier de donnees...")