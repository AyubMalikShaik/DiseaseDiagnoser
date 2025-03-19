import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load the dataset with error handling
def load_dataset():
    try:
        df = pd.read_csv('./Dataset/mydataset.csv')
        return df
    except FileNotFoundError:
        raise FileNotFoundError("Dataset not found. Please ensure 'mydataset.csv' is in the Dataset folder.")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

# Build graph from dataset
def buildGraph():
    df = load_dataset()
    G = nx.Graph()
    symptoms = df.columns[1:]
    for symptom in symptoms:
        G.add_node(symptom)

    for _, row in df.iterrows():
        for symptom in symptoms:
            if row[symptom] == 1:
                for other_symptom in symptoms:
                    if row[other_symptom] == 1 and symptom != other_symptom:
                        G.add_edge(symptom, other_symptom)
    return G

def buildGraphWithWeights():
    df = load_dataset()
    G = nx.Graph()
    symptoms = df.columns[1:]
    for symptom in symptoms:
        G.add_node(symptom)
    for _, row in df.iterrows():
        for symptom in symptoms:
            if row[symptom] == 1:
                for other_symptom in symptoms:
                    if row[other_symptom] == 1 and symptom != other_symptom:
                        if G.has_edge(symptom, other_symptom):
                            G[symptom][other_symptom]['weight'] += 1
                        else:
                            G.add_edge(symptom, other_symptom, weight=1)
    return G

[Rest of the file remains unchanged...]
