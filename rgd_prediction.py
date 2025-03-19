import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
import requests
import os
import joblib
import spacy
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from fuzzywuzzy import process
from database_operations import *

# Page config
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
if "final_symptoms" not in st.session_state:
    st.session_state.final_symptoms = []
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.label_encoder = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "community_input" not in st.session_state:
    st.session_state.community_input = ""
if "matched_symptoms" not in st.session_state:
    st.session_state.matched_symptoms = set()

# Download NLTK resources
@st.cache_data
def download_nltk_resources():
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')

download_nltk_resources()

# Load data
@st.cache_data
def load_data():
    df_comb = pd.read_csv("./Dataset/mydataset.csv")
    return df_comb

df_comb = load_data()
X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0]
dataset_symptoms = list(df_comb.columns[1:])

# Load model
@st.cache_resource
def load_model():
    mlp_model = joblib.load("./models/mlp_model.pkl")
    label_encoder = joblib.load("./models/label_encoder.pkl")
    return mlp_model, label_encoder

# NLP utilities
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')
nlp = spacy.load("en_core_web_md")

# Symptom matching functions
@st.cache_data
def get_synonyms(term):
    synonyms = set()
    for syn in wordnet.synsets(term):
        synonyms.update(syn.lemma_names())
    return list(synonyms)

@st.cache_data
def meaning_based_match(user_input, dataset):
    related_symptoms = []
    user_input_vec = nlp(user_input).vector
    for symptom in dataset:
        sym_vec = nlp(symptom).vector
        similarity = user_input_vec.dot(sym_vec) / (nlp(symptom).vector_norm * nlp(user_input).vector_norm)
        if similarity > 0.95:
            related_symptoms.append(symptom)
    return related_symptoms

@st.cache_data
def fuzzy_match_symptoms(user_input, dataset, threshold=80):
    matched = process.extractBests(user_input, dataset, score_cutoff=threshold)
    return [match[0] for match in matched]

def find_related_symptoms(user_input):
    user_input = user_input.lower().strip()
    related_symptoms = set()

    if user_input in dataset_symptoms:
        related_symptoms.add(user_input)

    synonyms = get_synonyms(user_input)
    for syn in synonyms:
        if syn in dataset_symptoms:
            related_symptoms.add(syn)

    related_symptoms.update(fuzzy_match_symptoms(user_input, dataset_symptoms))
    related_symptoms.update(meaning_based_match(user_input, dataset_symptoms))

    return list(related_symptoms)

# Main application UI
def main():
    if not st.session_state.logged_in:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.title("🏥 Disease Prediction System")
            st.write("Welcome to our advanced disease prediction system. Please login or register to continue.")

        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Login")
                if submit_button:
                    login(username, password)

        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Choose Username")
                new_password = st.text_input("Choose Password", type="password")
                email = st.text_input("Email")
                register_button = st.form_submit_button("Register")
                if register_button:
                    register(new_username, new_password, email)

    else:
        # Sidebar navigation
        st.sidebar.title(f"Welcome, {st.session_state.username}!")

        option = st.sidebar.selectbox(
            "Navigation",
            ["Disease Prediction", "Community Visualization", "History"],
            format_func=lambda x: f"📊 {x}" if x == "Community Visualization" 
                                else f"🏥 {x}" if x == "Disease Prediction"
                                else f"📜 {x}"
        )

        if st.sidebar.button("Logout"):
            logout()
            st.rerun()

        if option == "Disease Prediction":
            st.title("Disease Prediction")

            # Model training section
            with st.expander("Step 1: Train Model", expanded=True):
                if st.button("Train Model"):
                    with st.spinner("Training model..."):
                        if st.session_state.model is None:
                            st.session_state.model, st.session_state.label_encoder = load_model()
                            st.success("✅ Model trained successfully!")
                        else:
                            st.info("Model is already trained.")

            # Symptom input section
            with st.expander("Step 2: Input Symptoms", expanded=True):
                st.text_input(
                    "Enter symptoms separated by commas:",
                    value=st.session_state.user_input,
                    key="temp_user_input",
                    help="Example: fever, cough, headache"
                )

                if st.session_state.user_input:
                    processed_symptoms = [s.strip().lower() for s in st.session_state.user_input.split(",")]
                    matched_symptoms = set()

                    with st.spinner("Finding related symptoms..."):
                        for symptom in processed_symptoms:
                            matched_symptoms.update(find_related_symptoms(symptom))

                    st.session_state.matched_symptoms = list(matched_symptoms)

                    selected_symptoms = st.multiselect(
                        "Select matching symptoms:",
                        options=st.session_state.matched_symptoms,
                        default=st.session_state.selected_symptoms
                    )

                    if selected_symptoms != st.session_state.selected_symptoms:
                        st.session_state.selected_symptoms = selected_symptoms

            # Prediction section
            with st.expander("Step 3: Get Prediction", expanded=True):
                if st.button("Predict Disease"):
                    if not st.session_state.model:
                        st.error("Please train the model first.")
                    else:
                        with st.spinner("Analyzing symptoms..."):
                            sample_x = [0] * len(dataset_symptoms)
                            for symptom in st.session_state.selected_symptoms:
                                if symptom in dataset_symptoms:
                                    sample_x[dataset_symptoms.index(symptom)] = 1

                            probabilities = st.session_state.model.predict_proba([sample_x])[0]
                            top_indices = probabilities.argsort()[-5:][::-1]
                            predictions = st.session_state.label_encoder.inverse_transform(top_indices)

                            st.write("### Top 5 Predicted Diseases")
                            for i, (disease, prob) in enumerate(zip(predictions, probabilities[top_indices])):
                                st.metric(f"#{i+1}", disease, f"{prob*100:.1f}%")

                            log_action(st.session_state.username, "Prediction", 
                                     f"Symptoms: {', '.join(st.session_state.selected_symptoms)}")

        elif option == "Community Visualization":
            st.title("Symptom Communities")
            st.text_input("Search for a symptom:", key="community_search")

            if st.session_state.community_search:
                # Implementation for community visualization
                pass

        elif option == "History":
            st.title("Your History")
            display_history(st.session_state.username)

if __name__ == "__main__":
    main()