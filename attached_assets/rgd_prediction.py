import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
import nltk
import requests
import os
import pickle
from Neighbors import buildGraph,sub_graph,visualize_3d_symptom_graph,display_strongly_connected_symptoms,display_most_freq_symptoms,buildGraphWithWeights
from difflib import SequenceMatcher
import joblib
import spacy
import pandas as pd
from fuzzywuzzy import process
from nltk.corpus import wordnet
from gensim.models import KeyedVectors
from display_communities import display_clusters
from database_operations import *
# Download necessary NLTK resources
@st.cache_data
def download_nltk_resources():
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')

download_nltk_resources()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
# Load datasets
@st.cache_data
def load_data():
    df_comb = pd.read_csv("./Dataset/mydataset.csv")
    return df_comb
df_comb = load_data()
X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0]
dataset_symptoms = list(df_comb.columns[1:])
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')
# Cache model training
@st.cache_resource
def load_model():
    mlp_model = joblib.load("./models/mlp_model.pkl")  # Use your pretrained MLP model path
    label_encoder = joblib.load("./models/label_encoder.pkl")    # Load the label encoder
    return mlp_model, label_encoder

@st.cache_data
def load_communities():
    return {0: ['anxiety and nervousness', 'depression', 'depressive or psychotic symptoms', 'insomnia', 'abnormal involuntary movements', 'breathing fast', 'difficulty speaking', 'lack of growth', 'emotional symptoms', 'elbow weakness', 'back weakness', 'abusing alcohol', 'hostile behavior', 'drug abuse', 'restlessness', 'smoking problems', 'excessive appetite', 'excessive anger', 'focal weakness', 'slurring words', 'disturbance of memory', 'seizures', 'delusions or hallucinations', 'temper problems', 'wrist weakness', 'sleepiness', 'fears and phobias', 'low self-esteem', 'obsessions and compulsions', 'antisocial behavior', 'hysterical behavior', 'nightmares', 'muscle weakness'], 1: ['shortness of breath', 'sharp chest pain', 'dizziness', 'chest tightness', 'palpitations', 'irregular heartbeat', 'fainting', 'feeling ill', 'peripheral edema', 'weight gain', 'recent weight loss', 'weakness', 'decreased heart rate', 'increased heart rate', 'feeling cold', 'difficulty breathing', 'rib pain', 'fatigue', 'symptoms of the kidneys', 'hemoptysis', 'burning chest pain', 'sweating', 'hurts to breath', 'feeling hot', 'thirst'], 2: ['hoarse voice', 'sore throat', 'cough', 'nasal congestion', 'throat swelling', 'diminished hearing', 'lump in throat', 'throat feels tight', 'difficulty in swallowing', 'irritable infant', 'headache', 'toothache', 'dry lips', 'facial pain', 'mouth ulcer', 'swollen lymph nodes', 'infant spitting up', 'symptoms of infants', 'wheezing', 'neck mass', 'ear pain', 'jaw swelling', 'neck swelling', 'difficulty eating', 'ringing in ear', 'plugged feeling in ear', 'itchy ear(s)', 'frontal headache', 'fluid in ear', 'neck stiffness or tightness', 'decreased appetite', 'fever', 'tongue lesions', 'chills', 'coughing up sputum', 'coryza', 'allergic reaction', 'congestion in chest', 'apnea', 'abnormal breathing sounds', 'pulling at ears', 'gum pain', 'redness in ear', 'flu-like syndrome', 'sinus congestion', 'painful sinuses', 'nosebleed', 'bleeding from ear', 'swollen or red tonsils', 'mouth pain', 'neck cramps or spasms', 'sneezing', 'bleeding gums', 'pain in gums', 'throat redness'], 10: ['skin swelling', 'vaginal dryness', 'lip swelling', 'abnormal appearing skin', 'skin lesion', 'acne or pimples', 'skin growth', 'irregular appearing scalp', 'mouth dryness', 'skin moles', 'problems with shape or size of breast', 'bleeding or discharge from nipple', 'symptoms of the face', 'hand or finger lump or mass', 'pain or soreness of breast', 'lymphedema', 'skin on leg or foot looks infected', 'excessive growth', 'fluid retention', 'back mass or lump', 'unwanted hair', 'irregular appearing nails', 'itching of skin', 'skin dryness, peeling, scaliness, or roughness', 'skin irritation', 'itchy scalp', 'incontinence of stool', 'warts', 'bumps on penis', 'too little hair', 'foot or toe lump or mass', 'skin rash', 'dry or flaky scalp', 'lip sore', 'leg lump or mass', 'shoulder lump or mass', 'arm lump or mass', 'lump or mass of breast', 'postpartum problems of the breast', 'wrinkles on skin', 'skin pain', 'sore in nose'], 4: ['retention of urine', 'groin mass', 'suprapubic pain', 'blood in stool', 'symptoms of the scrotum and testes', 'swelling of scrotum', 'pain in testicles', 'flatulence', 'jaundice', 'mass in scrotum', 'sharp abdominal pain', 'vomiting', 'nausea', 'diarrhea', 'vaginal itching', 'painful urination', 'involuntary urination', 'pain during intercourse', 'frequent urination', 'lower abdominal pain', 'vaginal discharge', 'blood in urine', 'hot flashes', 'intermenstrual bleeding', 'pain of the anus', 'pain during pregnancy', 'pelvic pain', 'impotence', 'vomiting blood', 'regurgitation', 'burning abdominal pain', 'heartburn', 'scanty menstrual flow', 'vaginal pain', 'vaginal redness', 'vulvar irritation', 'side pain', 'problems during pregnancy', 'spotting or bleeding during pregnancy', 'cramps and spasms', 'upper abdominal pain', 'stomach bloating', 'changes in stool appearance', 'kidney mass', 'swollen abdomen', 'symptoms of prostate', 'groin pain', 'abdominal distention', 'regurgitation.1', 'melena', 'flushing', 'excessive urination at night', 'rectal bleeding', 'constipation', 'blood clots during menstrual periods', 'absence of menstruation', 'recent pregnancy', 'uterine contractions', 'long menstrual periods', 'heavy menstrual flow', 'unpredictable menstruation', 'painful menstruation', 'infertility', 'frequent menstruation', 'symptoms of bladder', 'mass or swelling around the anus', 'drainage in throat', 'premenstrual tension or irritability', 'pelvic pressure', 'itching of the anus', 'irregular belly button', 'vulvar sore', 'penis pain', 'loss of sex drive', 'bladder mass', 'premature ejaculation', 'penis redness', 'penile discharge', 'bedwetting', 'diaper rash', 'vaginal bleeding after menopause', 'hesitancy', 'low urine output'], 6: ['leg pain', 'hip pain', 'hand or finger pain', 'wrist pain', 'hand or finger swelling', 'arm pain', 'wrist swelling', 'arm stiffness or tightness', 'arm swelling', 'hand or finger stiffness or tightness', 'wrist stiffness or tightness', 'back pain', 'neck pain', 'low back pain', 'knee pain', 'foot or toe pain', 'bowlegged or knock-kneed', 'ankle pain', 'bones are painful', 'knee weakness', 'elbow pain', 'knee swelling', 'knee lump or mass', 'problems with movement', 'knee stiffness or tightness', 'leg swelling', 'foot or toe swelling', 'muscle pain', 'infant feeding problem', 'loss of sensation', 'paresthesia', 'shoulder pain', 'shoulder stiffness or tightness', 'shoulder weakness', 'shoulder swelling', 'leg cramps or spasms', 'ache all over', 'lower body pain', 'unusual color or odor to urine', 'leg stiffness or tightness', 'joint pain', 'muscle stiffness or tightness', 'back cramps or spasms', 'stiffness all over', 'muscle cramps, contractures, or spasms', 'low back cramps or spasms', 'skin on arm or hand looks infected', 'ankle swelling', 'foot or toe stiffness or tightness', 'elbow swelling', 'early or late onset of menopause', 'hand or finger weakness', 'hip stiffness or tightness', 'arm weakness', 'poor circulation', 'leg weakness', 'joint swelling', 'foot or toe weakness', 'hand or finger cramps or spasms', 'back stiffness or tightness', 'wrist lump or mass', 'ankle weakness'], 8: ['pus in sputum'], 9: ['pus draining from ear', 'white discharge from eye', 'eye deviation', 'diminished vision', 'double vision', 'cross-eyed', 'symptoms of eye', 'pain in eye', 'eye moves abnormally', 'abnormal movement of eyelid', 'foreign body sensation in eye', 'spots or clouds in vision', 'eye redness', 'lacrimation', 'itchiness of eye', 'blindness', 'eye burns or stings', 'itchy eyelid', 'bleeding from eye', 'muscle swelling', 'low back weakness', 'mass on eyelid', 'swollen eye', 'eyelid swelling', 'eyelid lesion or rash', 'cloudy eye', 'itching of scrotum', 'redness in or around nose'], 11: ['underweight'], 12: ['arm cramps or spasms'], 13: ['abnormal appearing tongue'], 14: ['pallor'], 15: ['shoulder cramps or spasms'], 16: ['joint stiffness or tightness'], 17: ['eye strain'], 18: ['pus in urine'], 19: ['abnormal size or shape of ear'], 20: ['elbow cramps or spasms'], 21: ['feeling hot and cold'], 22: ['nailbiting'], 23: ['hip swelling'], 24: ['foot or toe cramps or spasms'], 25: ['low back swelling'], 26: ['hip lump or mass'], 27: ['feet turned in'], 28: ['elbow stiffness or tightness'], 29: ['mass on ear'], 30: ['throat irritation'], 31: ['swollen tongue'], 32: ['disturbance of smell or taste'], 33: ['discharge in stools'], 34: ['pupils unequal'], 35: ['sleepwalking'], 36: ['skin oiliness'], 37: ['knee cramps or spasms'], 38: ['posture problems'], 39: ['bleeding in mouth'], 40: ['tongue bleeding'], 41: ['change in skin mole size or color'], 42: ['polyuria'], 43: ['infrequent menstruation'], 44: ['mass on vulva'], 45: ['jaw pain'], 46: ['eyelid retracted'], 47: ['elbow lump or mass'], 48: ['tongue pain'], 49: ['low back stiffness or tightness'], 50: ['skin on head or neck looks infected'], 51: ['stuttering or stammering'], 52: ['problems with orgasm'], 53: ['nose deformity'], 54: ['lump over jaw'], 55: ['hip weakness'], 3: ['back swelling'], 5: ['ankle stiffness or tightness'], 7: ['neck weakness']}
communities = load_communities()

@st.cache_data
def apply_light_theme():
    light_base_color = "2edef5"  # Light gray as the base
    highlight_color = "#507687"
    text_color = "#333333"  # Dark gray text
    button_color = "#4CAF50"
    a="#384B70"
    b="#C6E7FF"
    bg="#48A6A7"
    box_shadow = "0px 4px 8px rgba(0, 0, 0, 0.2)"  # Shadow styling
    st.markdown(
        f"""
        <style>
        
        /* Applying background color to the body */
        body {{
            background-color:'#48A6A7';  /* Correct background color syntax */
            color: {text_color};
            font-family: 'Book Antiqua', 'Candara', sans-serif;
        }}
        
         .block-container {{
            padding-left: 5rem !important;
            padding-right: 5rem !important;
            max-width: 100% !important;
        }}
        .stApp {{
        background-color: #E8F9FF !important;  /* Apply to the whole app */
    }}
        header {{
            background-color: #E8F9FF !important;
            color: #384B70 !important;
        }}
        .stButton > button {{
            background-color: {b};
            color: {highlight_color};
            border-radius: 8px;
            box-shadow: {box_shadow};
            font-family: 'Book Antiqua', 'Candara', sans-serif;
            transition: transform 0.2s ease;
        }}
        .stButton > button:hover {{
            transform: scale(1.05); /* Slight zoom effect */
        }}
        h1{{
        color:{a}
        }}
        .title-container{{
            position: fixed;
            top: 0;
            left: 0;
            z-index: 1000;

        }}
        .plotly-container {{
            background-color: #E8F9FF; /* Light blue background */
            padding: 15px;  /* Space around the chart */
            border-radius: 10px;  /* Rounded corners */
            box-shadow: 2px 4px 10px rgba(0, 0, 0, 0.1); /* Soft shadow */
            border: 2px solid #48A6A7; /* Border color */
        }}
        h3 {{
            font-family: 'Book Antiqua', 'Candara', sans-serif;
            font-size: 24px;
            color: {highlight_color};
            font-weight: bold;
        }}
        /* Styling for the text input field */
        .stTextInput input {{
            font-family: 'Book Antiqua', 'Candara', sans-serif;
            font-size: 18px;
            color: {text_color};
            padding: 10px;
            border-radius: 5px;
            border: 2px solid {highlight_color};
        }}
        
        h2{{
        color:{highlight_color};
        }}
        </style>
#     <div class="plotly-container">

        """,
        unsafe_allow_html=True
    )
st.markdown('<div class="main-container"></div>', unsafe_allow_html=True)

nlp = spacy.load("en_core_web_md")  # Use medium/large model for better accuracy
# if not st.session_state.logged_in:
#     option = st.radio("Choose an option:", ["Login", "Register"])
#     if option == "Login":
#         login()
#     elif option == "Register":
#         register()
# else:
#     st.sidebar.info(f"Logged in as {st.session_state.username}")
#     logout()
# main1()
local_css("style.css")


if not st.session_state.logged_in:
    # Tabs for Login and Registration
    st.title("🔐 Login & Registration")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.header("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                login(username, password)

    with tab2:
        st.header("Register")
        with st.form("register_form"):
            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")
            email = st.text_input("Email")
            register_button = st.form_submit_button("Register")

            if register_button:
                register(new_username, new_password, email)
else:
    st.sidebar.info(f"Logged in as {st.session_state.username}")
    logout()
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
        st.session_state.matched_symptoms =set()
    st.title("🩺Disease Prediction and Symptom Analysis")
    option = st.sidebar.selectbox("Choose a feature:", ["Disease Prediction", "Community Visualization","View History"])   
    if option == "Disease Prediction":
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
                
                if similarity > 0.95:  # Adjust threshold as needed
                    related_symptoms.append(symptom)
            return related_symptoms

        @st.cache_data
        def fuzzy_match_symptoms(user_input, dataset, threshold=80):
            matched = process.extractBests(user_input, dataset, score_cutoff=threshold)
            return [match[0] for match in matched]
        @st.cache_data
        def extract_body_part(symptom):
            doc = nlp(symptom)
            for token in doc:
                if token.ent_type_ == "BODY_PART" or token.pos_ in ["NOUN"]:  # Adjust based on dataset
                    return token.text.lower()
            return None
        @st.cache_data
        def find_related_symptoms(user_input):
            user_input = user_input.lower().strip()   
            related_symptoms = set()
            if user_input in dataset_symptoms:
                related_symptoms.add(user_input)# 1️⃣ Exact Match 
            synonyms = get_synonyms(user_input)
            for syn in synonyms:
                if syn in dataset_symptoms:
                    related_symptoms.add(syn)# 2️⃣ Synonym Match  
            related_symptoms.update(fuzzy_match_symptoms(user_input, dataset_symptoms))# 3️⃣ Fuzzy Matching (Fast & Accurate)  
            related_symptoms.update(meaning_based_match(user_input, dataset_symptoms))# 4️⃣ Meaning-Based Similarity Matching
            body_part = extract_body_part(user_input)# 5️⃣ Body Part Extraction & Matching
            if body_part:
                for symptom in dataset_symptoms:
                    if body_part in symptom:
                        related_symptoms.add(symptom)

            return list(related_symptoms)
        # Preprocess symptoms
        @st.cache_data
        def preprocess_symptoms(user_input):
            user_symptoms = user_input.lower().split(',')
            processed_user_symptoms = []
            for sym in user_symptoms:
                sym = sym.strip().replace('-', ' ').replace("'", '')
                sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
                processed_user_symptoms.append(sym)

            return processed_user_symptoms
        @st.cache_data
        def update_selected_symptoms():
            st.session_state.selected_symptoms = st.session_state.temp_selected_symptoms
        @st.cache_data
        def update_selected_nodes(symptom):
            key = f"selected_node_{symptom}"
            st.session_state.final_symptoms.append({
                "matched_symptom": symptom,
                "selected_node": st.session_state[key].split(",")
            })

        @st.cache_data
        def match_symptoms2(processed_symptoms):
            found_symptoms = set()
            for idx, data_sym in enumerate(dataset_symptoms):
                data_sym_split = data_sym.split()
                for user_sym in processed_symptoms:
                    count = 0
                    for symp in data_sym_split:
                        if symp in user_sym.split():
                            count += 1
                    if count / len(data_sym_split) > 0.5:
                        found_symptoms.add(data_sym)
            return list(found_symptoms)

        # Match symptoms based on similarity
        @st.cache_data
        def match_symptoms(processed_symptoms):
            found_symptoms = set()
            threshold = 0.8  # Set a similarity threshold
            for user_sym in processed_symptoms:
                for data_sym in dataset_symptoms:
                    similarity = SequenceMatcher(None, user_sym, data_sym).ratio()
                    if similarity >= threshold:
                        found_symptoms.add(data_sym)
            return list(found_symptoms)
        #Streamlit UI
        # Initialize session state if not already set
        if "model" not in st.session_state:
            st.session_state.model = None
            st.session_state.label_encoder = None
        # Apply the light theme with box shadow
        apply_light_theme()
        # Train model button
        st.header("1️⃣ Train Model")
        if st.button("Train Model"):
            if st.session_state.model is None:
                st.session_state.model, st.session_state.label_encoder = load_model()
                st.success("Model trained successfully!")
            else:
                st.warning("Model is already trained.")

        # Final symptoms list
        final_symptoms = []
        @st.cache_data
        def get_graph():
            G = buildGraph()
#             G = buildGraphWithWeights()
            return G
        @st.cache_data
        def get_weighted():
            G = buildGraphWithWeights()
            return G

        # Process user input for symptoms
        st.header("2️⃣ Input Symptoms")
        # 🔹 Ensure session state variables are initialized
        if "temp_user_input" not in st.session_state:
            st.session_state.temp_user_input = st.session_state.user_input

        # 🔹 Function to update `st.session_state.user_input`
        def update_user_input():
            st.session_state.user_input = st.session_state.temp_user_input

        # 🔹 Text input field with a temporary variable
        st.text_input(
            "Enter symptoms separated by commas:",
            value=st.session_state.temp_user_input,  # ✅ Use temp variable to prevent reset
            key="temp_user_input",
            on_change=update_user_input
        )

        if st.session_state.user_input:
            # Process the user input into a list of symptoms
            processed_user_symptoms = [
                symptom.strip().lower() for symptom in st.session_state.user_input.split(",") if symptom.strip()
            ]
            log_action(st.session_state.username, "Input Symptoms", st.session_state.user_input)
        # Find related symptoms for all processed symptoms
            matched_symptoms = set()
            for symptom in processed_user_symptoms:
                matched_symptoms.update(find_related_symptoms(symptom))  # Combine results for all input symptoms
          #     matched_symptoms = list(set(match_symptoms(processed_user_symptoms)) | set(match_symptoms2(processed_user_symptoms)))
            st.session_state.matched_symptoms = list(matched_symptoms)  # Store in session state

            if "temp_selected_symptoms" not in st.session_state:
                st.session_state.temp_selected_symptoms = st.session_state.selected_symptoms

            selected_symptoms = st.multiselect(
                "Select matching symptoms:",
                options=st.session_state.matched_symptoms,
                default=st.session_state.temp_selected_symptoms,
                key="temp_selected_symptoms",
                on_change=update_selected_symptoms
            )

            if selected_symptoms != st.session_state.selected_symptoms:
                st.session_state.selected_symptoms = selected_symptoms
                # ✅ Only update session state if selection changed

            # Build graph and visualize symptoms
            if st.session_state.selected_symptoms:
                G = get_graph()
                WG=get_weighted()# Only build once, cache graph if possible
                for symptom in st.session_state.selected_symptoms:
                    st.title(f"Exploring: {symptom}")
                    selected_nodes = []
                    col1,col2=st.columns(2)
                    sub=sub_graph(G,symptom)
                    with col1:
                        st.subheader(f" Most frequent occured symptoms of {symptom}")
                        fig1=display_most_freq_symptoms(G,symptom,12)
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        st.subheader(f" Strongly connected symptoms to {symptom}")
                        fig2=display_most_freq_symptoms(G,symptom,12)
                        st.plotly_chart(fig2, use_container_width=True)

            #         html_file, selected_nodes = visualize_symptom_graph(symptom, G)
                    key = f"selected_node_{symptom}"

                    if key not in st.session_state:
                        st.session_state[key] = ""
                    sn=[]
                    sn=st.text_input(
                        "Enter the selected symptom nodes separated by commas:",
                        key=key,
                        on_change=update_selected_nodes,
                        args=(symptom,)
                    ).split(',')
                    st.write(f"Added {sn} for symptom {symptom}.")

            st.write("### Final Symptoms and Selected Nodes:")
            st.write(st.session_state.final_symptoms)

            # Disease prediction button
            st.header("3️⃣ Predict Disease")
            # Disease prediction button
            if st.button("Predict Disease"):
                if st.session_state.model is None or st.session_state.label_encoder is None:
                    st.error("Please train the model first using the Train Model button.")
                else:
                    log_action(st.session_state.username, "Prediction", st.session_state.final_symptoms)

                    # Initialize sample_x with zeros
                    sample_x = [0] * len(dataset_symptoms)
                    
                    # Populate sample_x based on symptoms
                    for sym in st.session_state.final_symptoms:
                        matched_symptom = sym["matched_symptom"]
                        if matched_symptom in dataset_symptoms:
                            sample_x[dataset_symptoms.index(matched_symptom)] = 1
                        else:
                            st.warning(f"Symptom '{matched_symptom}' not found in dataset.")
                        
                        selected_node = sym.get("selected_node", [])
                        for sn in selected_node:
                            if sn in dataset_symptoms:
                                sample_x[dataset_symptoms.index(sn)] = 1
                            else:
                                st.warning(f"Node '{sn}' not found in dataset.")
                    # Ensure sample_x is a 2D array
                    sample_x = [sample_x]
                    # Validate feature size
                    if len(sample_x[0]) != st.session_state.model.n_features_in_:
                        st.error("Feature size mismatch. Ensure the symptoms match the training dataset.")
                    else:
                        # Predict probabilities and find top predictions
                        probabilities = st.session_state.model.predict_proba(sample_x)[0]
                        topk_indices = probabilities.argsort()[-10:][::-1]
                        predicted_labels = st.session_state.label_encoder.inverse_transform(topk_indices)
                        # Display top predictions with probabilities
                        st.write("### Predicted Diseases")
                        for i, index in enumerate(topk_indices):
                            disease = predicted_labels[i]
                            prob = probabilities[index]
                            st.write(f"{i+1}. {disease}")
        # Restart process button
        if st.button("Restart Process"):
            st.session_state.model = None
            st.session_state.label_encoder = None
            st.experimental_rerun()

    elif option == "Community Visualization":
        st.title("Community Graph Visualizations")
        st.session_state.community_input = st.text_input("Enter a symptom to find its community:", st.session_state.community_input)
        if st.session_state.community_input:
            input_symptom = st.session_state.community_input.lower().strip()
            matched_community = None

            for community_id, symptoms in communities.items():
                if input_symptom in symptoms:
                    matched_community = community_id
                    break
            if matched_community is not None:
                st.success(f"Your symptom belongs to Cluster {matched_community + 1}.")
                HTML_DIR = "community_htmls"
                html_path = os.path.join(HTML_DIR, f"community_{matched_community}.html")

                if os.path.exists(html_path):
                    with open(html_path, "r", encoding="utf-8") as html_file:
                        html_content = html_file.read()
                    st.components.v1.html(html_content, height=600, scrolling=True)
                else:
                    st.error(f"HTML file for Cluster {matched_community + 1} not found.")
            else:
                st.error("No matching community found for the entered symptom.")

        if st.button("Clear Input"):
            st.session_state.community_input = ""
            st.experimental_rerun()
    elif option=="View History":
        if st.button("View History"):
            display_history(st.session_state.username)
#             st.write("### Your History")
#             history = get_user_history(st.session_state.username)
#             for action, data, timestamp in history:
#                 st.write(f"{timestamp}: {action} - {data}")
#         st.dataframe(dat.style.applymap(lambda x: 'background-color: #f0f2f5'), height=300)


