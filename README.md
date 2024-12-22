# Projet de scoring de credit

Ce projet a pour objectif de prédire la probabilité qu'un client puisse rembourser un crédit bancaire en utilisant des techniques d'apprentissage automatique (Machine Learning). L'objectif est de fournir aux établissements financiers un outil efficace pour évaluer la solvabilité des clients à partir de leurs données personnelles, financières et comportementales. En analysant ces données, le modèle permet d'optimiser l'approbation des prêts et de réduire les risques liés aux non-remboursements.

Technologies utilisées
Les principales technologies et bibliothèques utilisées dans ce projet sont :

Python : Langage principal du projet.
Google Colab : Environnement pour l'analyse exploratoire des données et le développement des modèles.
Pandas : Bibliothèque pour la manipulation des données.
NumPy : Bibliothèque pour les calculs numériques.
Matplotlib & Seaborn : Bibliothèques pour la visualisation des données.
Scikit-learn : Pour le prétraitement des données, l'entraînement et l'évaluation des modèles de machine learning.
FastAPI : Pour la création d'une API permettant d'intégrer le modèle dans des applications externes.
Streamlit : Pour la création d'une interface utilisateur interactive permettant aux utilisateurs de saisir des données et d'obtenir des prédictions en temps réel.
Docker : Pour conteneuriser l'application et faciliter son déploiement.
Modèles de Machine Learning utilisés
Les modèles de machine learning suivants ont été utilisés pour prédire la solvabilité des clients :

Régression Logistique
Forêt Aléatoire (Random Forest)
GradientBoostingClassifier (modèle final choisi)
SVM
KNN
XGBoost
LightGBM
Parmi tous ces modèles, le GradientBoostingClassifier a été retenu comme le meilleur modèle en termes de performance pour prédire les risques de crédit.

Installation
Prérequis
Avant de commencer, assurez-vous d'avoir les éléments suivants installés sur votre machine :

Python 3.7+
Docker (pour le déploiement avec conteneurisation)
Installation des dépendances
Clonez ce repository sur votre machine locale.
Copier le code ci-dessous :
git clone <URL-du-repository>
cd scoring_credit

Créez un environnement virtuel et installez les dépendances nécessaires :
Copier le code ci-dessous
python -m venv .env
source .env/bin/activate  # Sous Linux/Mac
.env\Scripts\activate     # Sous Windows
pip install -r requirements.txt

Lancer l'API avec FastAPI
Une fois les dépendances installées, vous pouvez lancer l'API FastAPI pour tester les prédictions.
Copier le code ci-dessous :
uvicorn app:app --reload
L'API sera accessible à l'adresse : http://localhost:8000.

Lancer l'interface Streamlit
Pour lancer l'interface utilisateur Streamlit, exécutez la commande suivante :
Copier le code ci-dessous:
streamlit run app_streamlit.py
Cela ouvrira une interface dans votre navigateur où vous pourrez entrer des informations client et obtenir des prédictions.

Conteneurisation avec Docker
Pour déployer l'application dans un environnement conteneurisé, vous pouvez utiliser Docker.
Construisez l'image Docker :
Copier le code
docker build -t scoring_credit .
Exécutez le conteneur :

Copier le code
docker run -p 8000:8000 scoring_credit ou docker run -p 0.0.0.0:8501:8501 scoring-credit-app(pour arriver à le lier à toutes les interfaces réseau de votre machine (adresse IP publique ou privée))

Cela exposera l'API à http://localhost:8000 et l'interface Streamlit à http://localhost:8501.


Architecture du projet
Ce projet est composé de plusieurs parties :

Entraînement du modèle : Nous avons utilisé un ensemble de données pour entraîner différents modèles de machine learning. Après évaluation, le GradientBoostingClassifier a été choisi pour sa précision et sa robustesse.

API avec FastAPI : Une API a été créée en utilisant FastAPI pour exposer le modèle entraîné sous forme de service web. Cette API permet de soumettre des données client et de recevoir une prédiction sur la solvabilité en temps réel.

Interface avec Streamlit : Une interface utilisateur interactive a été développée avec Streamlit. Cette interface permet aux utilisateurs de saisir les informations d'un client (âge, type d'emploi, montant du crédit, etc.) et d'obtenir instantanément une prédiction sur la probabilité de remboursement.Il est aussi possible de télécharger une page donnée en local

Conteneurisation avec Docker : Un fichier Dockerfile a été créé pour conteneuriser l'application, facilitant ainsi son déploiement sur n'importe quel environnement. Cette approche permet une mise en production rapide et une gestion simplifiée des dépendances.


Structure du projet
Le projet contient les répertoires et fichiers suivants :

app.py : Code de l'API FastAPI.

app_streamlit.py : Code de l'interface utilisateur Streamlit.

Dockerfile : Pour construire l'image Docker.

requirements.txt : Liste des dépendances Python.

models/ : Contient le modèle entraîné et les encodeurs (p. ex., Gradient_boosting_model.pkl, sex_encoder.pkl).

data/ : Contient les fichiers de données brutes (non inclus ici, à utiliser localement).
