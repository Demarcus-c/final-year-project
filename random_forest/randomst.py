import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
import base64

# Streamlit Page Config
st.set_page_config(page_title="NeuroPulse - EEG Analysis", layout="wide")

# Sidebar - Patient Details
st.sidebar.header("🧠 Patient Information")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

# Sidebar - Upload
st.sidebar.header("📁 Upload EEG Data")
uploaded_file = st.sidebar.file_uploader("Upload EEG CSV", type=["csv"])

# Main App Logic
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.title("📊 EEG Prediction Dashboard")

    # Layout for visualizations
    col1, col2 = st.columns(2)

    # EEG Signal Line Plot
    with col1:
        st.subheader("📈 EEG Signal Over Time")
        fig1, ax1 = plt.subplots()
        sns.lineplot(data=df.iloc[:, :1], ax=ax1, color='dodgerblue')
        ax1.set_title("EEG Time Series")
        st.pyplot(fig1)

    # Histogram
    with col2:
        st.subheader("📊 EEG Signal Histogram")
        fig2, ax2 = plt.subplots()
        sns.histplot(df.iloc[:, 0], bins=30, kde=True, color='mediumseagreen')
        ax2.set_title("Signal Distribution")
        st.pyplot(fig2)

    # Correlation Heatmap
    st.subheader("📌 Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="viridis", ax=ax3)
        ax3.set_title("Correlation Heatmap")
        st.pyplot(fig3)
    else:
        st.warning("⚠️ Not enough numeric features to plot a correlation heatmap.")

    # PDF Report Generation
    if st.button("📄 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "EEG Data Visualization Report", ln=True, align="C")

        def save_fig(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            return buf

        for fig in [fig1, fig2, fig3]:
            buf = save_fig(fig)
            image_path = "temp_image.png"
            with open(image_path, "wb") as f:
                f.write(buf.getbuffer())
            pdf.image(image_path, x=10, w=180)
            pdf.ln(10)

        # Save and serve the PDF
        pdf.output("EEG_Report.pdf")
        with open("EEG_Report.pdf", "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            download_link = f'<a href="data:application/pdf;base64,{base64_pdf}" download="EEG_Report.pdf">📥 Download PDF Report</a>'
            st.markdown(download_link, unsafe_allow_html=True)

else:
    st.info("📎 Upload a CSV file to begin EEG analysis.")
