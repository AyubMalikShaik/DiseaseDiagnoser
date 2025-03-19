import os
import streamlit as st

# Directory containing community HTML files
def display_clusters():
    HTML_DIR = "community_htmls"  # Updated path to match new directory structure

    # Load community data
    communities = {
        0: ['anxiety and nervousness', 'depression', 'depressive or psychotic symptoms', 'insomnia'],
        1: ['shortness of breath', 'sharp chest pain', 'dizziness', 'chest tightness'],
        2: ['hoarse voice', 'sore throat', 'cough', 'nasal congestion'],
        # ... rest of the communities data remains the same
    }

    st.title("Community Graph Visualizations")

    # Input Symptom
    input_symptom = st.text_input("Enter a symptom to find its community:")

    if input_symptom:
        # Find the community for the input symptom
        matched_community = None
        for community_id, symptoms in communities.items():
            if input_symptom.lower() in [s.lower() for s in symptoms]:
                matched_community = community_id
                break

        if matched_community is not None:
            st.success(f"Your symptom belongs to Cluster {matched_community + 1}.")

            # Generate the path to the community HTML file
            html_path = os.path.join(HTML_DIR, f"community_{matched_community}.html")

            # Check if the file exists
            if os.path.exists(html_path):
                # Read and render the corresponding HTML content
                with open(html_path, "r", encoding="utf-8") as html_file:
                    html_content = html_file.read()
                st.components.v1.html(html_content, height=600, scrolling=True)
            else:
                st.error(f"HTML file for Cluster {matched_community + 1} not found.")
        else:
            st.error("No matching community found for the entered symptom.")
