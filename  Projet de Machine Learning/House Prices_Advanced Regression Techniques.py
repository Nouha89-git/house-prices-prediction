import io
from io import StringIO
#pip install streamlit pandas numpy matplotlib seaborn scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Configuration de la page
st.set_page_config(page_title="Analyse des prix de maisons", layout="wide")

# Titre de l'application
st.title("Prédiction des prix de maisons - Advanced Regression Techniques")

# Barre latérale pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisissez une page", ["Chargement des données",
                                                "Traitement des valeurs manquantes",
                                                "Nettoyage et prétraitement",
                                                "Ingénierie des caractéristiques",
                                                "Visualisation des données",
                                                "Sélection et entraînement du modèle",
                                                "Évaluation du modèle"])


# Fonction pour charger les données
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


# Fonction pour afficher les statistiques de base des données
def display_basic_stats(df):
    st.subheader("Aperçu des données")
    st.write(df.head())

    st.subheader("Informations sur les données")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Statistiques descriptives")
    st.write(df.describe())

    st.subheader("Valeurs manquantes")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    missing_data_percent = (missing_data / len(df)) * 100
    missing_stats = pd.DataFrame({
        'Valeurs manquantes': missing_data,
        'Pourcentage (%)': missing_data_percent
    })
    st.write(missing_stats.sort_values('Valeurs manquantes', ascending=False))


# Fonction pour explorer les corrélations
def plot_correlation(df, target_col):
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

    corr_matrix = df[numerical_cols + [target_col]].corr()
    corr_with_target = corr_matrix[target_col].sort_values(ascending=False)

    st.subheader("Corrélation avec le prix (SalePrice)")
    st.write(corr_with_target)

    # Top 10 corrélations
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[corr_with_target.index[:10]].corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)


# Fonction pour traiter les valeurs manquantes
def handle_missing_values(df, strategy='drop'):
    df_processed = df.copy()

    if strategy == 'drop':
        # Supprimer les colonnes avec beaucoup de valeurs manquantes
        threshold = 0.8 * len(df)
        df_processed = df_processed.dropna(thresh=threshold, axis=1)

        # Supprimer les lignes avec des valeurs manquantes
        df_processed = df_processed.dropna()

    elif strategy == 'impute':
        # Séparation des colonnes numériques et catégorielles
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(exclude=[np.number]).columns

        # Imputation des valeurs numériques avec la médiane
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())

        # Imputation des valeurs catégorielles avec le mode
        for col in categorical_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    return df_processed


# Fonction pour le prétraitement des données
def preprocess_data(df, target_col):
    # Séparation des caractéristiques et de la cible
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
    else:
        X = df
        y = None

    # Identification des colonnes numériques et catégorielles
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns

    # Création du pipeline de prétraitement
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return X, y, preprocessor


# Fonction pour l'ingénierie des caractéristiques
def feature_engineering(df):
    df_engineered = df.copy()

    # Création de caractéristiques : surface totale
    if 'TotalBsmtSF' in df.columns and '1stFlrSF' in df.columns and '2ndFlrSF' in df.columns:
        df_engineered['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # Caractéristique : âge de la maison au moment de la vente
    if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
        df_engineered['HouseAge'] = df['YrSold'] - df['YearBuilt']

    # Caractéristique : combinaison de la qualité et de la condition
    if 'OverallQual' in df.columns and 'OverallCond' in df.columns:
        df_engineered['QualCond'] = df['OverallQual'] * df['OverallCond']

    # Caractéristique : ratio surface/chambres
    if 'GrLivArea' in df.columns and 'BedroomAbvGr' in df.columns:
        df_engineered['AreaPerRoom'] = df['GrLivArea'] / (df['BedroomAbvGr'] + 1)

    # Transformation logarithmique de la variable cible (si présente)
    if 'SalePrice' in df.columns:
        df_engineered['SalePrice'] = np.log1p(df['SalePrice'])

    return df_engineered


# Fonction pour la sélection et l'entraînement du modèle
def train_model(X_train, y_train, model_name):
    if model_name == 'Linear Regression':
        model = LinearRegression()
    elif model_name == 'Ridge':
        model = Ridge(alpha=1.0)
    elif model_name == 'Lasso':
        model = Lasso(alpha=0.1)
    elif model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    return model


# Fonction pour l'évaluation du modèle
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Prédictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Métriques
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)

    return {
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R²': train_r2,
        'Test R²': test_r2
    }


# Gestion des différentes pages
if page == "Chargement des données":
    st.header("Chargement des données")

    uploaded_file = st.file_uploader("Chargez votre fichier CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            data = load_data(uploaded_file)
            st.session_state['data'] = data
            st.success(f"Données chargées avec succès ! ({data.shape[0]} lignes, {data.shape[1]} colonnes)")

            display_basic_stats(data)

        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")

elif page == "Traitement des valeurs manquantes":
    if 'data' not in st.session_state:
        st.warning("Veuillez d'abord charger les données.")
    else:
        st.header("Traitement des valeurs manquantes")

        data = st.session_state['data']

        # Affichage des valeurs manquantes
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        missing_data_percent = (missing_data / len(data)) * 100
        missing_stats = pd.DataFrame({
            'Valeurs manquantes': missing_data,
            'Pourcentage (%)': missing_data_percent
        })

        st.subheader("Résumé des valeurs manquantes")
        st.write(missing_stats.sort_values('Valeurs manquantes', ascending=False))

        # Visualisation des valeurs manquantes
        st.subheader("Visualisation des valeurs manquantes")
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        st.pyplot(plt)

        # Options pour le traitement des valeurs manquantes
        st.subheader("Options de traitement")
        missing_strategy = st.radio(
            "Choisissez une stratégie pour les valeurs manquantes",
            ["Aucune action", "Supprimer", "Imputer"]
        )

        if missing_strategy != "Aucune action":
            if missing_strategy == "Supprimer":
                processed_data = handle_missing_values(data, strategy='drop')
            else:  # Imputer
                processed_data = handle_missing_values(data, strategy='impute')

            st.session_state['processed_data'] = processed_data

            st.success(
                f"Traitement effectué ! Dimensions des données : {processed_data.shape[0]} lignes, {processed_data.shape[1]} colonnes")

            # Aperçu des données traitées
            st.subheader("Aperçu des données après traitement")
            st.write(processed_data.head())

            # Vérification des valeurs manquantes restantes
            remaining_missing = processed_data.isnull().sum().sum()
            st.write(f"Nombre de valeurs manquantes restantes : {remaining_missing}")

elif page == "Nettoyage et prétraitement":
    if 'processed_data' not in st.session_state and 'data' not in st.session_state:
        st.warning("Veuillez d'abord charger et traiter les données.")
    else:
        st.header("Nettoyage et prétraitement des données")

        # Utiliser les données traitées si disponibles, sinon les données brutes
        if 'processed_data' in st.session_state:
            data = st.session_state['processed_data']
        else:
            data = st.session_state['data']

        # Sélection de la variable cible
        target_col = st.selectbox("Sélectionnez la variable cible", data.columns)

        # Prétraitement des données
        X, y, preprocessor = preprocess_data(data, target_col)

        # Affichage des informations sur les caractéristiques
        st.subheader("Caractéristiques sélectionnées")
        st.write(f"Nombre de caractéristiques numériques : {len(X.select_dtypes(include=[np.number]).columns)}")
        st.write(f"Nombre de caractéristiques catégorielles : {len(X.select_dtypes(exclude=[np.number]).columns)}")

        # Option pour diviser les données
        st.subheader("Division des données")
        test_size = st.slider("Pourcentage des données pour le test", 0.1, 0.4, 0.2, 0.05)

        if st.button("Prétraiter les données"):
            # Division des données
            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                # Stockage des données prétraitées
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['preprocessor'] = preprocessor
                st.session_state['target_col'] = target_col

                st.success(
                    f"Données divisées en ensemble d'entraînement ({X_train.shape[0]} échantillons) et ensemble de test ({X_test.shape[0]} échantillons)")

                # Aperçu des données d'entraînement
                st.subheader("Aperçu des données d'entraînement")
                st.write(X_train.head())
            else:
                st.error("Impossible de diviser les données. La variable cible n'est pas disponible.")

elif page == "Ingénierie des caractéristiques":
    if 'data' not in st.session_state:
        st.warning("Veuillez d'abord charger les données.")
    else:
        st.header("Ingénierie des caractéristiques")

        # Utiliser les données traitées si disponibles, sinon les données brutes
        if 'processed_data' in st.session_state:
            data = st.session_state['processed_data']
        else:
            data = st.session_state['data']

        # Création de nouvelles caractéristiques
        st.subheader("Création de nouvelles caractéristiques")
        st.write("Nous allons créer de nouvelles caractéristiques pour améliorer la performance du modèle.")

        if st.button("Créer de nouvelles caractéristiques"):
            engineered_data = feature_engineering(data)

            # Stockage des données avec les nouvelles caractéristiques
            st.session_state['engineered_data'] = engineered_data

            # Affichage des nouvelles caractéristiques
            new_features = [col for col in engineered_data.columns if col not in data.columns]

            st.success(f"Nouvelles caractéristiques créées : {len(new_features)}")

            if new_features:
                st.subheader("Nouvelles caractéristiques")
                st.write(engineered_data[new_features].head())

                # Statistiques sur les nouvelles caractéristiques
                st.subheader("Statistiques des nouvelles caractéristiques")
                st.write(engineered_data[new_features].describe())

                # Visualisation de la distribution des nouvelles caractéristiques
                for feature in new_features:
                    if engineered_data[feature].dtype in [np.float64, np.int64]:
                        st.subheader(f"Distribution de {feature}")
                        plt.figure(figsize=(10, 4))
                        sns.histplot(engineered_data[feature].dropna(), kde=True)
                        st.pyplot(plt)

elif page == "Visualisation des données":
    if 'data' not in st.session_state:
        st.warning("Veuillez d'abord charger les données.")
    else:
        st.header("Visualisation des données")

        # Utiliser les données avec les caractéristiques avancées si disponibles
        if 'engineered_data' in st.session_state:
            data = st.session_state['engineered_data']
        elif 'processed_data' in st.session_state:
            data = st.session_state['processed_data']
        else:
            data = st.session_state['data']

        # Sélection des variables pour les visualisations
        st.subheader("Sélection des variables")
        target_col = st.selectbox("Sélectionnez la variable cible", data.columns,
                                  index=data.columns.get_loc('SalePrice') if 'SalePrice' in data.columns else 0)

        # Visualisation de la distribution de la variable cible
        st.subheader(f"Distribution de {target_col}")
        plt.figure(figsize=(10, 4))
        sns.histplot(data[target_col].dropna(), kde=True)
        st.pyplot(plt)

        # Matrice de corrélation
        st.subheader("Matrice de corrélation")
        plot_correlation(data, target_col)

        # Visualisations supplémentaires
        st.subheader("Visualisations supplémentaires")

        viz_type = st.selectbox("Type de visualisation", ["Pairplot", "Boxplot", "Scatterplot"])

        if viz_type == "Pairplot":
            # Sélection des variables pour le pairplot
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            selected_vars = st.multiselect("Sélectionnez les variables (max 5)", numeric_cols, default=[target_col])

            if len(selected_vars) > 1 and len(selected_vars) <= 5:
                st.subheader("Pairplot")
                plt.figure(figsize=(12, 8))
                sns.pairplot(data[selected_vars])
                st.pyplot(plt)
            else:
                st.warning("Veuillez sélectionner entre 2 et 5 variables pour le pairplot.")

        elif viz_type == "Boxplot":
            # Sélection des variables pour le boxplot
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            x_var = st.selectbox("Variable catégorielle (axe X)", data.select_dtypes(exclude=[np.number]).columns)
            y_var = st.selectbox("Variable numérique (axe Y)", numeric_cols,
                                 index=0 if target_col not in numeric_cols else list(numeric_cols).index(target_col))

            plt.figure(figsize=(12, 6))
            sns.boxplot(x=data[x_var], y=data[y_var])
            plt.xticks(rotation=45)
            st.pyplot(plt)

        elif viz_type == "Scatterplot":
            # Sélection des variables pour le scatterplot
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            x_var = st.selectbox("Variable X", numeric_cols, index=0)
            y_var = st.selectbox("Variable Y", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=data[x_var], y=data[y_var])
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            st.pyplot(plt)

elif page == "Sélection et entraînement du modèle":
    if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
        st.warning("Veuillez d'abord prétraiter les données.")
    else:
        st.header("Sélection et entraînement du modèle")

        # Sélection du modèle
        model_type = st.selectbox(
            "Sélectionnez un modèle",
            ["Linear Regression", "Ridge", "Lasso", "Random Forest", "Gradient Boosting"]
        )

        # Configuration du modèle
        st.subheader("Configuration du modèle")

        if model_type == "Ridge" or model_type == "Lasso":
            alpha = st.slider("Valeur d'alpha (régularisation)", 0.01, 10.0, 1.0, 0.01)
        elif model_type == "Random Forest" or model_type == "Gradient Boosting":
            n_estimators = st.slider("Nombre d'estimateurs", 50, 500, 100, 10)
            max_depth = st.slider("Profondeur maximale", 3, 20, 10)

        # Entraînement du modèle
        if st.button("Entraîner le modèle"):
            X_train = st.session_state['X_train']
            y_train = st.session_state['y_train']
            preprocessor = st.session_state['preprocessor']

            # Prétraitement des données
            X_train_processed = preprocessor.fit_transform(X_train)

            with st.spinner("Entraînement du modèle en cours..."):
                # Configuration du modèle selon les paramètres
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Ridge":
                    model = Ridge(alpha=alpha)
                elif model_type == "Lasso":
                    model = Lasso(alpha=alpha)
                elif model_type == "Random Forest":
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                elif model_type == "Gradient Boosting":
                    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

                # Entraînement du modèle
                model.fit(X_train_processed, y_train)

                # Stockage du modèle entraîné
                st.session_state['model'] = model
                st.session_state['model_type'] = model_type
                st.session_state['preprocessor_fitted'] = preprocessor

                st.success(f"Modèle {model_type} entraîné avec succès !")

elif page == "Évaluation du modèle":
    if 'model' not in st.session_state:
        st.warning("Veuillez d'abord entraîner un modèle.")
    else:
        st.header("Évaluation du modèle")

        model = st.session_state['model']
        model_type = st.session_state['model_type']
        preprocessor = st.session_state['preprocessor_fitted']
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']

        # Prétraitement des données
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Évaluation du modèle
        with st.spinner("Évaluation du modèle en cours..."):
            # Prédictions
            train_preds = model.predict(X_train_processed)
            test_preds = model.predict(X_test_processed)

            # Métriques
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            train_r2 = r2_score(y_train, train_preds)
            test_r2 = r2_score(y_test, test_preds)

            # Affichage des métriques
            st.subheader("Métriques de performance")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE sur l'ensemble d'entraînement", f"{train_rmse:.4f}")
                st.metric("R² sur l'ensemble d'entraînement", f"{train_r2:.4f}")
            with col2:
                st.metric("RMSE sur l'ensemble de test", f"{test_rmse:.4f}")
                st.metric("R² sur l'ensemble de test", f"{test_r2:.4f}")

            # Visualisation des prédictions
            st.subheader("Visualisation des prédictions")

            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, test_preds, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel("Valeurs réelles")
            plt.ylabel("Prédictions")
            plt.title("Prédictions vs Valeurs réelles")
            st.pyplot(plt)

            # Distribution des erreurs
            st.subheader("Distribution des erreurs")

            errors = test_preds - y_test
            plt.figure(figsize=(10, 6))
            sns.histplot(errors, kde=True)
            plt.xlabel("Erreur de prédiction")
            plt.title("Distribution des erreurs")
            st.pyplot(plt)

            # Importance des caractéristiques (pour certains modèles)
            if model_type in ["Random Forest", "Gradient Boosting"]:
                st.subheader("Importance des caractéristiques")

                try:
                    # Récupération des noms de caractéristiques après prétraitement
                    feature_names = []
                    for name, transformer, columns in preprocessor.transformers_:
                        if hasattr(transformer, 'get_feature_names_out'):
                            feature_names.extend(transformer.get_feature_names_out(columns))
                        else:
                            feature_names.extend(columns)

                    # Création d'un DataFrame des importances
                    importances = pd.DataFrame({
                        'feature': feature_names[:len(model.feature_importances_)],
                        'importance': model.feature_importances_
                    })
                    importances = importances.sort_values('importance', ascending=False).head(20)

                    # Affichage des importances
                    plt.figure(figsize=(10, 8))
                    sns.barplot(x='importance', y='feature', data=importances)
                    plt.title("Importance des caractéristiques")
                    st.pyplot(plt)
                except:
                    st.warning("Impossible d'afficher l'importance des caractéristiques.")

            # Sauvegarde du modèle
            if st.button("Sauvegarder le modèle"):
                model_filename = f"{model_type.lower().replace(' ', '_')}_model.pkl"
                with open(model_filename, 'wb') as file:
                    pickle.dump(model, file)

                preprocessor_filename = "preprocessor.pkl"
                with open(preprocessor_filename, 'wb') as file:
                    pickle.dump(preprocessor, file)

                st.success(f"Modèle sauvegardé dans {model_filename} et préprocesseur dans {preprocessor_filename}")
