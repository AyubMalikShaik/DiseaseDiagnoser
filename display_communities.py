import os
import streamlit as st

# Directory containing community HTML files
def display_clusters():
    HTML_DIR = "community_htmls"

    st.write("### Symptom Communities Visualization")
    
    # Load all community HTML files
    community_files = [f for f in os.listdir(HTML_DIR) if f.startswith("community_") and f.endswith(".html")]
    community_numbers = sorted([int(f.split("_")[1].split(".")[0]) for f in community_files])
    
    # Create a dropdown to select community
    selected_community = st.selectbox(
        "Select a community to visualize:",
        community_numbers,
        format_func=lambda x: f"Community {x}"
    )
    
    # Display the selected community visualization
    if selected_community is not None:
        html_path = os.path.join(HTML_DIR, f"community_{selected_community}.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as html_file:
                html_content = html_file.read()
            st.components.v1.html(html_content, height=600, scrolling=True)
        else:
            st.error("Community visualization not found.")

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
            # If not found in predefined communities, try to find in HTML files directly
            for i in range(38):  # We now have communities 0-37
                html_path = os.path.join(HTML_DIR, f"community_{i}.html")
                if os.path.exists(html_path):
                    # Here we could parse the HTML to check if the symptom is present
                    # For now, we'll just offer to show this community
                    if st.button(f"View Cluster {i + 1}"):
                        with open(html_path, "r", encoding="utf-8") as html_file:
                            html_content = html_file.read()
                        st.components.v1.html(html_content, height=600, scrolling=True)

            st.info("Try exploring different clusters to find related symptoms.")