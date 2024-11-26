import streamlit as st
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt

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

        # Visualization: Bar chart for training and testing accuracy
        st.subheader("Model Performance Visualization")
        fig, ax = plt.subplots()
        ax.bar(
            ["Training Accuracy", "Testing Accuracy"],
            [model_info["Training Accuracy"], model_info["Testing Accuracy"]],
            color=["blue", "green"]
        )
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Performance of {model_info['Model']}")
        st.pyplot(fig)
else:
    st.error(f"Directory '{PICKLE_DIR}' not found. Please upload the pickle files.")

# Allow users to upload their dataset
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for analysis", type=["csv"])

if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset")
    st.write(user_data.head())

    # Add mock predictions for demonstration
    st.subheader("Mock Predictions")
    if len(user_data) > 0:
        st.write("Here is an example of applying a model to the dataset.")
        mock_predictions = [f"Class {i % 3}" for i in range(len(user_data))]
        user_data["Predictions"] = mock_predictions
        st.write(user_data.head())  # Display dataset with mock predictions
    else:
        st.warning("The uploaded dataset is empty!")
else:
    st.info("Upload a dataset to view insights.")
