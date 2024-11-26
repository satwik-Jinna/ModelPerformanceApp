import streamlit as st
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt

# Caching function to load pickle files
@st.cache_data
def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Caching function to load all models
@st.cache_data
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

        # Visualization: Enhanced Bar chart for training and testing accuracy
        st.subheader("Model Performance Visualization")
        training_acc = model_info["Training Accuracy"]
        testing_acc = model_info["Testing Accuracy"]
        accuracy_diff = training_acc - testing_acc

        # Create the bar plot
        fig, ax = plt.subplots()
        bars = ax.bar(
            ["Training Accuracy", "Testing Accuracy"],
            [training_acc, testing_acc],
            color=["red", "green"]
        )
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Performance of {model_info['Model']}")

        # Annotate the bars with their values
        for bar, value in zip(bars, [training_acc, testing_acc]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",  # Format the value to 4 decimal places
                ha='center',
                va='bottom'
            )

        # Show the difference between training and testing
        ax.text(
            0.5,  # Midpoint between the two bars
            max(training_acc, testing_acc) + 0.01,  # Slightly above the taller bar
            f"Difference: {accuracy_diff:.4f}",
            ha='center',
            va='bottom',
            fontsize=10,
            color='red',
            fontweight='bold'
        )

        # Display the plot
        st.pyplot(fig)
else:
    st.error(f"Directory '{PICKLE_DIR}' not found. Please upload the pickle files.")

# Allow users to upload their dataset
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for analysis", type=["csv"])

if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file).head(100)  # Load only the first 100 rows for optimization
    st.subheader("Uploaded Dataset (First 100 Rows)")
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
