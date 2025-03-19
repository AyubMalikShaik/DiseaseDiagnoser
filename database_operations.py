import sqlite3
import streamlit as st
from datetime import datetime
import pandas as pd

# Database initialization
def init_db():
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT
        )
    """)

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

# Initialize database
init_db()

# User management functions
def register(username, password, email=""):
    """Register a new user in the database."""
    if not username or not password:
        st.error("Username and password are required.")
        return False

    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO Users (username, password, email) VALUES (?, ?, ?)",
            (username, password, email)
        )
        conn.commit()
        st.success("Registration successful! Please login.")
        return True
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
        return False
    finally:
        conn.close()

def verify_user(username, password):
    """Verify user credentials."""
    if not username or not password:
        st.error("Please enter both username and password.")
        return False

    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM Users WHERE username = ? AND password = ?",
        (username, password)
    )
    user = cursor.fetchone()
    conn.close()

    return user is not None

# History management
def log_action(username, action, data):
    """Log a user action in the history table."""
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO History (username, action, data) VALUES (?, ?, ?)",
        (username, action, str(data))
    )
    conn.commit()
    conn.close()

def get_user_history(username):
    """Get the history of user actions."""
    conn = sqlite3.connect("app_data.db")
    query = """
        SELECT action, data, timestamp 
        FROM History 
        WHERE username = ? 
        ORDER BY timestamp DESC
    """
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df

def display_history(username):
    """Display user history with improved styling."""
    history_df = get_user_history(username)

    if history_df.empty:
        st.info("No history found.")
        return

    st.markdown("""
    <style>
    .history-card {
        background-color: rgba(198, 231, 255, 0.5);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #48A6A7;
    }
    .history-timestamp {
        color: #666;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

    for _, row in history_df.iterrows():
        timestamp = datetime.strptime(row['timestamp'], 
                                    '%Y-%m-%d %H:%M:%S').strftime('%B %d, %Y %I:%M %p')

        st.markdown(f"""
        <div class="history-card">
            <h4>{row['action']}</h4>
            <p>{row['data']}</p>
            <p class="history-timestamp">{timestamp}</p>
        </div>
        """, unsafe_allow_html=True)

# Session management
def login(username, password):
    """Log in a user."""
    if verify_user(username, password):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.success("Login successful!")
        st.rerun()
    else:
        st.error("Invalid username or password.")

def logout():
    """Log out a user and clear session state."""
    st.session_state.logged_in = False
    st.session_state.username = None
    # Clear all session state variables
    for key in ['final_symptoms', 'selected_symptoms', 'model', 
                'label_encoder', 'user_input', 'community_input', 
                'matched_symptoms']:
        if key in st.session_state:
            if isinstance(st.session_state[key], set):
                st.session_state[key] = set()
            elif isinstance(st.session_state[key], list):
                st.session_state[key] = []
            else:
                st.session_state[key] = None