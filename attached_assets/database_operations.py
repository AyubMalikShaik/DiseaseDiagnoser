import sqlite3
import streamlit as st
st.set_page_config(page_title="Login & Registration", page_icon="🔐", layout="centered")
import pandas as pd
# Custom CSS for styling

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
local_css("style.css")

# Database Initialization
def init_db():
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()

    # Create Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Create History table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS History (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            data TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

# Call database initialization at the start
init_db()

# User Management Functions
@st.cache_data
def register_user(username, password):
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO Users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

@st.cache_data
def verify_user(username, password):
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

# History Management Functions
import streamlit as st
@st.cache_data
def log_action(username, action, data):
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO History (username, action, data) VALUES (?, ?, ?)", (username, action, str(data)))
    conn.commit()
    conn.close()

import sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime

# Function to fetch user history
def get_user_history(username):
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT action, data, timestamp FROM History WHERE username = ? ORDER BY timestamp DESC", (username,))
    history = cursor.fetchall()
    conn.close()
    return history

# Custom CSS for card-based history display

# Function to display history in a card-based layout
def display_history(username):
    history = get_user_history(username)
    if history:
        st.write("### Your History")
        for action, data, timestamp in history:
            # Format the timestamp for better readability
            formatted_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").strftime("%b %d, %Y %I:%M %p")
            # Create a card for each history entry using HTML
            b="#C6E7FF"
            light_base_color = "2edef5"  # Light gray as the base
            highlight_color = "#507687"
            text_color = "#333333"  # Dark gray text
            button_color = "#4CAF50"
            a="#384B70"
            b="#C6E7FF"
            bg="#48A6A7"
            box_shadow = "0px 4px 8px rgba(0, 0, 0, 0.2)" 
            st.markdown(
                f"""
                    
                <div class="history-card">
                    <h4>{action}</h4>
                    <p><strong>Data:</strong> {data}</p>
                    <p class="timestamp"><strong>Timestamp:</strong> {formatted_timestamp}</p>
                </div>
                <style>
                .history-card {{
        background-color: rgba(74, 144, 226,0.5);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    h1 {{
    color: #4a90e2;
    text-align: center;
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
    .history-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.6);
    }}
    .history-card h4 {{
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
        margin-bottom: 10px;
    }}
    .history-card p {{
        color: #555;
        font-family: 'Arial', sans-serif;
        margin: 5px 0;
    }}
    .history-card .timestamp {{
        color: #888;
        font-size: 0.9em;
    }}
                </style>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No history found.")

# Example usage in your Streamlit app

# Example usage in your Streamlit app

# @st.cache_data
# def get_user_history(username):
#     conn = sqlite3.connect("app_data.db")
#     cursor = conn.cursor()
#     cursor.execute("SELECT action, data, timestamp FROM History WHERE username = ? ORDER BY timestamp DESC", (username,))
#     history = cursor.fetchall()
#     conn.close()
#     return history
#     query = "SELECT * FROM users"
#     df = pd.read_sql_query(query, conn)
#     conn.close()
#     return df

# Login, Logout, and Registration
# @st.cache_data
def main1():
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
def login(username,password):
#     st.sidebar.header("Login")
# #     username = st.sidebar.text_input("Username")
# #     password = st.sidebar.text_input("Password", type="password")
#     if st.sidebar.button("Login"):
    if verify_user(username, password):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.sidebar.success("Logged in successfully!")
    else:
        st.sidebar.error("Invalid username or password.")

# @st.cache_data
def logout():
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.sidebar.info("Logged out successfully.")
#         if "final_symptoms" not in st.session_state:
        st.session_state.final_symptoms = []
#         if "selected_symptoms" not in st.session_state:
        st.session_state.selected_symptoms = []
#         if "model" not in st.session_state:
        st.session_state.model = None
        st.session_state.label_encoder = None
#         if "user_input" not in st.session_state:
        st.session_state.user_input = ""
#         if "community_input" not in st.session_state:
        st.session_state.community_input = ""
#         if "matched_symptoms" not in st.session_state:
        st.session_state.matched_symptoms =set()
#         option = st.sidebar.radio("Choose an option:", ["Login", "Register"])
#         if option == "Login":
#             login()
#         elif option == "Register":
#             register()

def register(username,password):
#     st.sidebar.header("Register")
#     username = st.sidebar.text_input("New Username")
#     password = st.sidebar.text_input("New Password", type="password")
#     if st.sidebar.button("Register"):
    if register_user(username, password):
        st.sidebar.success("User registered successfully!")
    else:
        st.sidebar.error("Username already exists.")

# Initialize Session State
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
#     st.session_state.username = None
# 
# # Login, Logout, and Registration Options
# if not st.session_state.logged_in:
#     option = st.sidebar.radio("Choose an option:", ["Login", "Register"])
#     if option == "Login":
#         login()
#     elif option == "Register":
#         register()
# else:
#     st.sidebar.info(f"Logged in as {st.session_state.username}")
#     logout()
# 
# # Main App Logic for Logged-in Users
# if st.session_state.logged_in:
#     st.title("🩺 Disease Prediction and Symptom Analysis")
#     
#     # Example: Log user actions (replace with actual app functionality)
#     user_input = st.text_input("Enter your input (e.g., symptoms):")
#     if user_input:
#         log_action(st.session_state.username, "Input Symptoms", user_input)
#         st.write(f"Logged input: {user_input}")
#     
#     if st.button("View History"):
#         st.write("### Your History")
#         history = get_user_history(st.session_state.username)
#         for action, data, timestamp in history:
#             st.write(f"{timestamp}: {action} - {data}")
# else:
#     st.write("Please log in to access the app.")
