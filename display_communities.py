
import os
import streamlit as st

def display_clusters():
    HTML_DIR = "community_htmls"
    
    st.write("### 🔍 Symptom Communities")
    
    # Style for the community display
    st.markdown("""
        <style>
        .community-selector {
            background-color: rgba(198,231,255,0.1);
            padding: 20px;
            border-radius: 15px;
            margin: 15px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="community-selector">', unsafe_allow_html=True)
    
    # Get all community files
    community_files = sorted([f for f in os.listdir(HTML_DIR) if f.startswith("community_") and f.endswith(".html")])
    
    if not community_files:
        st.error("No community visualizations found.")
        return
        
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Browse Communities", "🔎 Search Communities"])
    
    with tab1:
        # Dropdown to select community
        selected_file = st.selectbox(
            "Select a community to visualize:",
            community_files,
            format_func=lambda x: f"Community {x.split('_')[1].split('.')[0]}"
        )
        
        if selected_file:
            html_path = os.path.join(HTML_DIR, selected_file)
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)
            
    with tab2:
        # Search functionality
        search_term = st.text_input("Search for a symptom in communities:")
        if search_term:
            found = False
            for file in community_files:
                html_path = os.path.join(HTML_DIR, file)
                with open(html_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if search_term.lower() in content.lower():
                        st.success(f"Found in {file}")
                        st.components.v1.html(content, height=600, scrolling=True)
                        found = True
                        break
            if not found:
                st.warning("Symptom not found in any community.")
    
    st.markdown('</div>', unsafe_allow_html=True)
