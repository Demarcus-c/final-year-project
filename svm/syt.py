import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Set page layout
st.set_page_config(page_title="NeuroScope - Seizure Risk", layout="wide")

st.title("🧠 NeuroScope - Epilepsy Detection Dashboard")

# Sidebar: User Info
st.sidebar.header("Patient Info")
patient_name = st.sidebar.text_input("Name", "Patient A")
patient_age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=30)
patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

# Sidebar: File Upload
st.sidebar.header("📤 Upload EEG CSV")
uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")

# Random prediction function (for demo/testing)
def mock_predict(df):
    # Generate random probabilities between 30% and 100%
    df["Epilepsy Probability (%)"] = np.random.uniform(30, 100, size=len(df)).round(2)

    # Predict based on threshold
    df["Prediction"] = ["Seizure Risk" if x > 80 else "No Risk" for x in df["Epilepsy Probability (%)"]]

    # Personalized prevention tips
    def generate_tips(prob, age, gender):
        tips = []
        if prob > 80:
            tips.append("Avoid bright or flashing lights.")
            tips.append("Maintain strict sleep schedule.")
            tips.append("Take prescribed medications regularly.")
        if age < 20:
            tips.append("Avoid video games with intense visuals.")
        elif age > 60:
            tips.append("Avoid stress and overexertion.")
        if gender == "Female":
            tips.append("Track hormonal cycles, consult neurologist if patterns arise.")
        return " ".join(tips)

    df["Tips"] = df["Epilepsy Probability (%)"].apply(
        lambda prob: generate_tips(prob, patient_age, patient_gender)
    )
    return df

# Load and predict
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df = mock_predict(df)

        for idx, row in df.iterrows():
            st.markdown("----")
            col1, col2 = st.columns([1, 2])

            # Gauge Plot
            with col1:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=row['Epilepsy Probability (%)'],
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "crimson" if row['Prediction'] == "Seizure Risk" else "green"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 80], 'color': "orange"},
                            {'range': [80, 100], 'color': "red"}]
                    },
                    title={'text': f"Patient {idx} Risk (%)"}
                ))
                st.plotly_chart(gauge, use_container_width=True)

            # Info Card
            with col2:
                st.markdown(f"### 👤 Patient {idx+1}: {patient_name}")
                st.markdown(f"🧬 **Age:** {patient_age} &nbsp;&nbsp;&nbsp;&nbsp; 🚻 **Gender:** {patient_gender}")
                st.markdown(f"🔬 **Prediction:** {'🔴' if row['Prediction']=='Seizure Risk' else '🟢'} {row['Prediction']}")
                st.markdown(f"💡 **Prevention Tips:**\n- {row['Tips'].replace('. ', '.\n- ')}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a CSV file to begin epilepsy risk prediction.")
