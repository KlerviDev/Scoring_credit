import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Chargement des modèles et des encodeurs
model = joblib.load('Gradient_boosting_model.pkl')
le_sex = joblib.load('sex_encoder.pkl')
le_saving = joblib.load('saving_accounts_encoder.pkl')
le_checking = joblib.load('checking_account_encoder.pkl')
housing_columns = joblib.load('housing_columns.pkl')  # Liste des colonnes one-hot pour Housing
purpose_columns = joblib.load('purpose_columns.pkl')  # Liste des colonnes one-hot pour Purpose

# Configuration du thème via CSS
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .sidebar .sidebar-content { background-color: #6c63ff; color: white; }
    .big-font { font-size:20px !important; color: #6c63ff; }
    .center-text { text-align: center; }
    /* Style pour l'image de la section Accueil */
    .img-full-width {
        width: 100%; 
        height: auto; 
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Menu")
menu = st.sidebar.radio("Navigation", ["Accueil", "Prédictions", "À propos"])

if menu == "Accueil":
    st.image("scoring_credit.jpg", use_container_width=True)
    st.markdown('<h1 class="center-text">Bienvenue dans l\'application de Scoring de Crédit</h1>', unsafe_allow_html=True)
    st.write("""
    Cette application vous permet d'évaluer les risques de crédit pour un client. 
    Entrez les informations demandées et obtenez une prédiction instantanée.
    """)

elif menu == "Prédictions":
    st.markdown('<h2 class="big-font">Prédictions de Crédit</h2>', unsafe_allow_html=True)
    
    # Entrée des données utilisateur
    age = st.number_input("Âge", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sexe", options=["male", "female"])
    job = st.selectbox("Type d'emploi (Code)", options=[0, 1, 2, 3])
    housing = st.selectbox("Type de logement", options=housing_columns)
    saving_accounts = st.selectbox("Épargne", options=le_saving.classes_)
    checking_account = st.selectbox("Compte courant", options=le_checking.classes_)
    credit_amount = st.number_input("Montant du crédit", min_value=0.0, value=1000.0, step=500.0)
    duration = st.number_input("Durée (en mois)", min_value=1, value=12)
    purpose = st.selectbox("But du crédit", options=purpose_columns)

    # Préparation des données et prédiction
    if st.button("Prédire"):
        try:
            # Transformation des données d'entrée
            data = {
                "Age": age,
                "Sex": le_sex.transform([sex])[0],
                "Job": job,
                "Saving accounts": le_saving.transform([saving_accounts])[0],
                "Checking account": le_checking.transform([checking_account])[0],
                "Credit amount": credit_amount,
                "Duration": duration,
                **{f"Housing_{housing}": 1 if h == housing else 0 for h in housing_columns},
                **{f"Purpose_{purpose}": 1 if p == purpose else 0 for p in purpose_columns},
            }
            input_data = pd.DataFrame([data])

            # Obtenir les colonnes d'entraînement utilisées pour le modèle
            train_columns = model.feature_names_in_

            # Aligner les colonnes de l'entrée avec celles utilisées lors de l'entraînement
            input_data = input_data.reindex(columns=train_columns, fill_value=0)


            # Prédiction
            prediction = model.predict(input_data)[0]
            result = "Bon" if prediction == 1 else "Mauvais"
            st.success(f"Risque de crédit : {result}")
        except Exception as e:
            st.error(f"Erreur : {str(e)}")

elif menu == "À propos":
    st.markdown('<h2 class="big-font">À propos</h2>', unsafe_allow_html=True)
    st.write("""
    Cette application a été développée pour aider les institutions financières 
    à évaluer les risques de crédit. Basée sur un modèle de Gradient Boosting, 
    elle fournit des prédictions fiables et rapides.
    """)
    st.markdown("""
    #### Contactez-nous :
    - **Email** : someclervie@gmail.com
    - **Téléphone** : +221 77 848 59 98
    """)

# Pied de page
st.markdown('<hr><p class="center-text">Développée par Sandrine SOME,Décembre 2024</p>', unsafe_allow_html=True)
