import os
import streamlit as st

def display_clusters():
    HTML_DIR = "community_htmls"

    st.write("### Symptom Communities Visualization")

    # Get all community files
    try:
        community_files = sorted([f for f in os.listdir(HTML_DIR) if f.startswith("community_") and f.endswith(".html")])

        if not community_files:
            st.error("No community visualizations found.")
            return

        # Create a dropdown to select community
        selected_file = st.selectbox(
            "Select a community to visualize:",
            community_files,
            format_func=lambda x: f"Community {x.split('_')[1].split('.')[0]}"
        )

        if selected_file:
            html_path = os.path.join(HTML_DIR, selected_file)
            try:
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600, scrolling=True)
            except Exception as e:
                st.error(f"Error loading community visualization: {str(e)}")

    except Exception as e:
        st.error(f"Error accessing community files: {str(e)}")