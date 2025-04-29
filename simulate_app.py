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

# --- Manual Input ---
st.sidebar.markdown("### 📝 Manual Entry for Quick Prediction")

manual_user_id = st.sidebar.text_input("User ID (optional)", "")
manual_answer_type = st.sidebar.selectbox("Answer Type", [
    "Knowledge-Based", "Problem-Solving", "Strategic Reasoning"
])
manual_correct_sum = st.sidebar.number_input("Correct Answers", min_value=0, step=1)
manual_total_questions = st.sidebar.number_input("Total Questions", min_value=1, step=1, value=1)

manual_prediction_result = None
if st.sidebar.button("🔍 Predict for Manual Entry"):
    correct_percentage = (manual_correct_sum / manual_total_questions) * 100
    manual_input_df = pd.DataFrame({
        "correct_percentage": [correct_percentage],
        "answer_type": [manual_answer_type]
    })

    # Load model and predict
    model = joblib.load('Model Training/Resulted Model/model_pipe.joblib')
    pred = model.predict(manual_input_df)[0]

    mapping = {
        0: "Advanced",
        1: "Intermediate",
        2: "Needs Reinforcement"
    }
    manual_prediction_result = mapping[pred]

# --- Manual Prediction Output Display ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎯 Manual Prediction Result")
    if manual_prediction_result:
        color = {
            "Advanced": "#2E7D32",
            "Intermediate": "#1976D2",
            "Needs Reinforcement": "#D32F2F"
        }[manual_prediction_result]
        
        st.markdown(
            f"<div style='background-color:{color};padding:10px;border-radius:10px;color:white;text-align:center;'>"
            f"<strong>{manual_prediction_result}</strong>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.caption("Prediction will appear here after submitting the form.")

# --- Process File ---
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### 🔎 Preview of uploaded data:")
    st.dataframe(data.head(), use_container_width=True)

    required_features = ['user_id', 'answer_type', 'correct_sum', 'total_questions']

    if not all(feature in data.columns for feature in required_features):
        st.error(f"🚨 Uploaded file must contain exactly these columns:\n\n{required_features}")
    else:
        # Preprocessing
        data['correct_percentage'] = (data['correct_sum'] / data['total_questions']) * 100

        features = ['correct_percentage', 'answer_type']
        X = data[features]

        # Load model
        with st.spinner('🔄 Predicting performance...'):
            model = joblib.load('Model Training/Resulted Model/model_pipe.joblib')

            predictions = model.predict(X)

            # Mapping numerical predictions to categories
            mapping = {
                0: "Advanced",
                1: "Intermediate",
                2: "Needs Reinforcement"
            }
            predicted_labels = [mapping[pred] for pred in predictions]

            data['Predicted Performance'] = predicted_labels

            # Assigning **Predicted Category** for each user_id
            user_performance = data.groupby('user_id')['Predicted Performance'].agg(lambda x: x.mode()[0]).reset_index()

        st.success('🎯 Predictions completed successfully!')
        st.write("### 📈 Prediction Results:")

        # Styling: Highlight only the 'Predicted Performance' column
        # --- Define Styling Function ---
        def highlight_advanced(val):
            if val == 'Advanced':
                return 'background-color: #2E7D32; color: white;'  # Dark green background + white text (good for dark mode)
            else:
                return ''

        # --- Apply Styling ---
        styled_df = user_performance.style.applymap(
            highlight_advanced,
            subset=['Predicted Performance']
        )

        st.dataframe(styled_df, use_container_width=True)

        # --- Download Section ---
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(user_performance)

        st.markdown("---")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name='user_performance_predictions.csv',
            mime='text/csv',
            help="Click to download the file containing predictions.",
        )

# --- Footer ---
st.markdown("---")
st.caption("Made by Ahmed M. Fayad | Data Scientist 👨‍💻")