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
from database_operations import init_db, login, logout, register, log_action, display_history

# Add these imports at the top of the file
from Neighbors import (buildGraph, buildGraphWithWeights, find_directly_related_symptoms,
                      display_most_freq_symptoms, display_strongly_connected_symptoms,
                      visualize_3d_symptom_graph)
from display_communities import display_clusters

# Page config with improved styling
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

# Download NLTK resources with error handling
@st.cache_data
def download_nltk_resources():
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('omw-1.4')
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {str(e)}")
        return False
    return True

# Load data with error handling
@st.cache_data
def load_data():
    try:
        data_path = "./Dataset/mydataset.csv"
        if not os.path.exists(data_path):
            st.error(f"Dataset not found at {data_path}")
            return None
        df_comb = pd.read_csv(data_path)
        return df_comb
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        model_path = "./models/mlp_model.pkl"
        encoder_path = "./models/label_encoder.pkl"

        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            st.error("Model or encoder files not found")
            return None, None

        mlp_model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        return mlp_model, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Initialize resources
if download_nltk_resources():
    df_comb = load_data()
    if df_comb is not None:
        X = df_comb.iloc[:, 1:]
        Y = df_comb.iloc[:, 0]
        dataset_symptoms = list(df_comb.columns[1:])
    else:
        st.error("Could not initialize the application due to missing dataset")
        st.stop()

# NLP utilities
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')
try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    st.error(f"Error loading spaCy model: {str(e)}")
    st.stop()

# Enhanced symptom matching functions
@st.cache_data
def get_synonyms(term):
    """Get synonyms for a term using WordNet."""
    try:
        synonyms = set()
        for syn in wordnet.synsets(term, pos=wordnet.NOUN):
            synonyms.update(syn.lemma_names())
            # Include hypernyms (more general terms)
            for hypernym in syn.hypernyms():
                synonyms.update(hypernym.lemma_names())
        return list(synonyms)
    except Exception as e:
        st.warning(f"Error getting synonyms: {str(e)}")
        return []

@st.cache_data
def meaning_based_match(user_input, dataset, threshold=0.85):
    """Match symptoms based on semantic similarity using spaCy."""
    try:
        related_symptoms = []
        user_input_doc = nlp(user_input)

        for symptom in dataset:
            symptom_doc = nlp(symptom)
            similarity = user_input_doc.similarity(symptom_doc)
            if similarity > threshold:
                related_symptoms.append((symptom, similarity))

        # Sort by similarity score
        related_symptoms.sort(key=lambda x: x[1], reverse=True)
        return [symptom for symptom, _ in related_symptoms[:5]]
    except Exception as e:
        st.warning(f"Error in meaning based matching: {str(e)}")
        return []

@st.cache_data
def fuzzy_match_symptoms(user_input, dataset, threshold=75):
    """Match symptoms using fuzzy string matching."""
    try:
        matched = process.extractBests(user_input, dataset, score_cutoff=threshold)
        return [match[0] for match in matched[:5]]  # Return top 5 matches
    except Exception as e:
        st.warning(f"Error in fuzzy matching: {str(e)}")
        return []

def get_body_part_matches(user_input, dataset):
    """Match symptoms based on body parts mentioned."""
    body_parts = {
        'head': ['head', 'brain', 'skull'],
        'chest': ['chest', 'breast', 'thorax'],
        'stomach': ['stomach', 'abdomen', 'belly'],
        'throat': ['throat', 'neck', 'pharynx'],
        'skin': ['skin', 'dermal', 'cutaneous'],
        'eye': ['eye', 'vision', 'ocular'],
        'ear': ['ear', 'hearing', 'auditory'],
        'nose': ['nose', 'nasal', 'sinus'],
        'mouth': ['mouth', 'oral', 'tongue'],
        'arm': ['arm', 'hand', 'wrist'],
        'leg': ['leg', 'foot', 'ankle'],
        'joint': ['joint', 'bone', 'muscle']
    }

    matches = []
    user_tokens = set(user_input.lower().split())

    for part, related_terms in body_parts.items():
        if any(term in user_input.lower() for term in related_terms):
            # Find symptoms that mention this body part or related terms
            for symptom in dataset:
                if any(term in symptom.lower() for term in related_terms):
                    matches.append(symptom)

    return list(set(matches))  # Remove duplicates

def find_related_symptoms(user_input):
    """Find related symptoms using multiple matching strategies."""
    user_input = user_input.lower().strip()
    related_symptoms = set()

    # Direct match
    if user_input in dataset_symptoms:
        related_symptoms.add(user_input)

    # Synonym-based matching
    synonyms = get_synonyms(user_input)
    for syn in synonyms:
        if syn.lower() in [s.lower() for s in dataset_symptoms]:
            related_symptoms.add(syn)

    # Fuzzy matching
    fuzzy_matches = fuzzy_match_symptoms(user_input, dataset_symptoms)
    related_symptoms.update(fuzzy_matches)

    # Semantic similarity matching
    semantic_matches = meaning_based_match(user_input, dataset_symptoms)
    related_symptoms.update(semantic_matches)

    # Body part matching
    body_part_matches = get_body_part_matches(user_input, dataset_symptoms)
    related_symptoms.update(body_part_matches)

    # Filter out any non-existing symptoms
    final_symptoms = [s for s in related_symptoms if s in dataset_symptoms]
    return list(set(final_symptoms))  # Remove duplicates

# Main application UI
def main():
    if not st.session_state.logged_in:
        st.markdown("""
            <style>
            .main-title {
                font-size: 3rem !important;
                color: #384B70;
                text-align: center;
                margin-bottom: 2rem;
            }
            .auth-container {
                max-width: 800px;
                margin: 0 auto;
                padding: 2rem;
                background-color: rgba(198, 231, 255, 0.2);
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<h1 class="main-title">🏥 Disease Prediction System</h1>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
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

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Sidebar styling
        st.markdown("""
            <style>
            .sidebar-content {
                padding: 1rem;
                background-color: rgba(72, 166, 167, 0.1);
                border-radius: 10px;
                margin: 1rem 0;
            }
            .prediction-card {
                background-color: rgba(198, 231, 255, 0.5);
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                border: 1px solid #48A6A7;
            }
            </style>
        """, unsafe_allow_html=True)

        # Sidebar navigation
        st.sidebar.title(f"👋 Welcome, {st.session_state.username}!")

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
                            if st.session_state.model is not None:
                                st.success("✅ Model trained successfully!")
                            else:
                                st.error("Failed to load the model. Please check the error messages above.")

            # Symptom input section
            with st.expander("Step 2: Input Symptoms", expanded=True):
                user_input = st.text_input(
                    "Enter symptoms:",
                    value=st.session_state.user_input,
                    key="temp_user_input",
                    help="Example: fever, cough, headache"
                )

                if user_input:
                    st.session_state.user_input = user_input
                    processed_symptoms = [s.strip().lower() for s in user_input.split(",")]

                    with st.spinner("Finding related symptoms..."):
                        all_matched_symptoms = set()

                        # Add custom CSS for match display
                        st.markdown("""
                        <style>
                        .match-container {
                            background-color: rgba(198, 231, 255, 0.2);
                            padding: 15px;
                            border-radius: 10px;
                            margin: 10px 0;
                            border: 1px solid #48A6A7;
                        }
                        .match-category { 
                            color: #384B70; 
                            font-weight: bold;
                            margin-top: 10px;
                        }
                        .match-item { 
                            margin-left: 20px; 
                            color: #48A6A7;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        for symptom in processed_symptoms:
                            if symptom:  # Skip empty strings
                                matches = find_related_symptoms(symptom)
                                if matches:
                                    all_matched_symptoms.update(matches)

                                    # Display matching details using custom container
                                    st.markdown(f"""
                                    <div class="match-container">
                                    <h4>Matches for '{symptom}'</h4>
                                    """, unsafe_allow_html=True)

                                    # Group matches by type
                                    exact = [m for m in matches if m.lower() == symptom.lower()]
                                    fuzzy = fuzzy_match_symptoms(symptom, dataset_symptoms)
                                    semantic = meaning_based_match(symptom, dataset_symptoms)
                                    body_part = get_body_part_matches(symptom, dataset_symptoms)

                                    if exact:
                                        st.markdown("<p class='match-category'>Exact Matches:</p>", unsafe_allow_html=True)
                                        for m in exact:
                                            st.markdown(f"<p class='match-item'>• {m}</p>", unsafe_allow_html=True)

                                    if fuzzy:
                                        st.markdown("<p class='match-category'>Similar Symptoms:</p>", unsafe_allow_html=True)
                                        for m in fuzzy:
                                            st.markdown(f"<p class='match-item'>• {m}</p>", unsafe_allow_html=True)

                                    if semantic:
                                        st.markdown("<p class='match-category'>Related Symptoms:</p>", unsafe_allow_html=True)
                                        for m in semantic:
                                            st.markdown(f"<p class='match-item'>• {m}</p>", unsafe_allow_html=True)

                                    if body_part:
                                        st.markdown("<p class='match-category'>Body Part Related:</p>", unsafe_allow_html=True)
                                        for m in body_part:
                                            st.markdown(f"<p class='match-item'>• {m}</p>", unsafe_allow_html=True)

                                    st.markdown("</div>", unsafe_allow_html=True)

                        st.session_state.matched_symptoms = list(all_matched_symptoms)

                        # Selection of symptoms
                        if st.session_state.matched_symptoms:
                            st.markdown("### Select Symptoms for Prediction")
                            selected_symptoms = st.multiselect(
                                "Choose the relevant symptoms:",
                                options=sorted(st.session_state.matched_symptoms),
                                default=st.session_state.selected_symptoms,
                                help="Select all symptoms that apply"
                            )

                            if selected_symptoms != st.session_state.selected_symptoms:
                                st.session_state.selected_symptoms = selected_symptoms
                        else:
                            st.warning("No matching symptoms found. Please try different terms.")

            # Prediction section
            with st.expander("Step 3: Get Prediction", expanded=True):
                if st.button("Predict Disease"):
                    if not st.session_state.model:
                        st.error("Please train the model first.")
                    elif not st.session_state.selected_symptoms:
                        st.warning("Please select at least one symptom.")
                    else:
                        with st.spinner("Analyzing symptoms..."):
                            try:
                                sample_x = [0] * len(dataset_symptoms)
                                for symptom in st.session_state.selected_symptoms:
                                    if symptom in dataset_symptoms:
                                        sample_x[dataset_symptoms.index(symptom)] = 1

                                probabilities = st.session_state.model.predict_proba([sample_x])[0]
                                top_indices = probabilities.argsort()[-5:][::-1]
                                predictions = st.session_state.label_encoder.inverse_transform(top_indices)

                                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                                st.write("### 🎯 Top 5 Predicted Diseases")
                                cols = st.columns(5)
                                for i, (disease, prob) in enumerate(zip(predictions, probabilities[top_indices])):
                                    with cols[i]:
                                        st.metric(f"#{i+1}", disease, f"{prob*100:.1f}%")
                                st.markdown('</div>', unsafe_allow_html=True)

                                log_action(st.session_state.username, "Prediction",
                                         f"Symptoms: {', '.join(st.session_state.selected_symptoms)}")
                            except Exception as e:
                                st.error(f"Error during prediction: {str(e)}")

        elif option == "Community Visualization":
            st.title("Symptom Communities")

            # Add tabs for different visualizations
            viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                "🔍 Symptom Search",
                "🌐 Network Analysis",
                "👥 Communities"
            ])

            with viz_tab1:
                search_symptom = st.text_input(
                    "Search for a symptom:",
                    key="community_search",
                    help="Enter a symptom to see its relationships"
                )

                if search_symptom:
                    try:
                        # Build weighted graph
                        G = buildGraphWithWeights()

                        # Display options
                        analysis_type = st.radio(
                            "Choose analysis type:",
                            ["Most Frequent Connections", "Strongest Relationships"],
                            horizontal=True
                        )

                        count = st.slider("Number of connections to show", 5, 20, 10)

                        # Show visualization based on selection
                        with st.spinner("Generating visualization..."):
                            if analysis_type == "Most Frequent Connections":
                                fig = display_most_freq_symptoms(G, search_symptom, count)
                            else:
                                fig = display_strongly_connected_symptoms(G, search_symptom, count)

                            # Display the plotly figure
                            st.plotly_chart(fig, use_container_width=True)

                            # Show directly related symptoms
                            related = find_directly_related_symptoms(G, search_symptom)
                            if related:
                                st.markdown("### 🔗 Directly Related Symptoms")
                                for symptom in related[:10]:  # Show top 10
                                    st.markdown(f"- {symptom}")

                                if len(related) > 10:
                                    with st.expander("See more..."):
                                        for symptom in related[10:]:
                                            st.markdown(f"- {symptom}")

                    except KeyError:
                        st.warning("Symptom not found in the database. Please try another term.")
                    except Exception as e:
                        st.error(f"Error generating visualization: {str(e)}")

            with viz_tab2:
                st.markdown("""
                ### 🌐 Network Analysis
                Explore the relationships between symptoms through network analysis.
                This visualization shows how different symptoms are connected based on
                their co-occurrence in diagnoses.
                """)

                if st.button("Generate Network Overview"):
                    with st.spinner("Building network visualization..."):
                        try:
                            G = buildGraph()
                            # Generate and display a sample subgraph
                            sample_symptoms = list(G.nodes())[:20]  # Take first 20 symptoms
                            sub_G = G.subgraph(sample_symptoms)
                            fig = visualize_3d_symptom_graph(sub_G, sample_symptoms[0])
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating network overview: {str(e)}")

            with viz_tab3:
                st.markdown("""
                ### 👥 Symptom Communities
                Discover groups of symptoms that commonly occur together.
                These communities can help understand patterns in symptom relationships.
                """)

                if st.button("Show Communities"):
                    with st.spinner("Analyzing symptom communities..."):
                        try:
                            display_clusters()
                        except Exception as e:
                            st.error(f"Error displaying communities: {str(e)}")

        elif option == "History":
            st.title("Your History")
            display_history(st.session_state.username)

if __name__ == "__main__":
    main()