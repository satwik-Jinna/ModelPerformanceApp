import streamlit as st
import pickle
import pandas as pd
import os

# Function to load a pickle file
def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Load all pickle files and display their information
def load_all_models(directory):
    models_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            file_path = os.path.join(directory, filename)
            model_data = load_pickle_file(file_path)
            models_data[filename] = model_data
    return models_data

# Streamlit App
st.title("Model Performance Analysis")

# Directory containing pickle files
PICKLE_DIR = "pickle_files"

# Load models
if os.path.exists(PICKLE_DIR):
    models_data = load_all_models(PICKLE_DIR)
    st.sidebar.header("Available Models")
    selected_model = st.sidebar.selectbox(
        "Select a model to view details",
        list(models_data.keys())
    )

    # Display details of the selected model
    if selected_model:
        model_info = models_data[selected_model]
        st.subheader(f"Details for: {model_info['Model']}")
        st.write("**Training Accuracy:**", model_info["Training Accuracy"])
        st.write("**Testing Accuracy:**", model_info["Testing Accuracy"])
else:
    st.error(f"Directory '{PICKLE_DIR}' not found. Please upload the pickle files.")

# Allow users to upload their dataset
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for analysis", type=["csv"])

if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset")
    st.write(user_data.head())

    # Sample feedback for uploaded dataset
    st.write("You can now apply models to this dataset.")
else:
    st.info("Upload a dataset to view insights.")
