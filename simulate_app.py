import streamlit as st
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎯", layout="centered")

# --- Sidebar ---
with st.sidebar:
    st.title("🎯 Performance Predictor")
    st.markdown("Upload a **CSV file** to predict students' performance levels based on their quiz results.")
    st.markdown("---")
    st.info("Make sure your file contains the required columns!")
    st.caption("Created with ❤️ using Streamlit.")

# --- Main Header ---
st.title("🚀 Student Performance Prediction App")
st.subheader("Easily predict student categories: Advanced, Intermediate, or Needs Reinforcement")
st.markdown("---")

# --- File Uploader ---
uploaded_file = st.file_uploader("📤 Upload your CSV file here", type=["csv"])

# --- Process File ---
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### 🔎 Preview of uploaded data:")
    st.dataframe(data.head(), use_container_width=True)

    required_features = [
        'correct_percentage',
        'average_correct_percentage',
        'answer_type_Knowledge-Based',
        'answer_type_Problem-Solving',
        'answer_type_Strategic Reasoning'
    ]

    if not all(feature in data.columns for feature in required_features):
        st.error(f"🚨 Uploaded file must contain exactly these columns:\n\n{required_features}")
    else:
        X = data[required_features]

        with st.spinner('🔄 Predicting performance...'):
            model = joblib.load('Model Training/Resulted Model/rf_model.joblib')  # Be careful with slashes!

            predictions = model.predict(X)

            mapping = {
                0: "Advanced",
                1: "Intermediate",
                2: "Needs Reinforcement"
            }
            predicted_labels = [mapping[pred] for pred in predictions]

            results = data.copy()
            results['Predicted Performance'] = predicted_labels

        st.success('🎯 Predictions completed successfully!')
        st.write("### 📈 Prediction Results:")
        st.dataframe(results.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

        # --- Download Section ---
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(results)

        st.markdown("---")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
            help="Click to download the file containing predictions.",
        )

# --- Footer ---
st.markdown("---")
st.caption("Made by Ahmed Fayad | Data Scientist 👨‍💻")