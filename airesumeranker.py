import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""  # Handle None values
    return text


# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes  # Fix incorrect appending
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities


# Streamlit App UI
st.set_page_config(page_title="AI Resume Ranker", layout="wide")
st.title("📄 AI Resume Screening & Candidate Ranking")
st.markdown("### Easily rank resumes based on job descriptions using AI!")

# Sidebar for inputs
with st.sidebar:
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

    st.header("Enter Job Description")
    job_description = st.text_area("Paste the job description here")

    process_button = st.button("Rank Resumes")

if process_button and uploaded_files and job_description:
    st.subheader("Ranking Resumes...")
    progress_bar = st.progress(0)

    resumes = []
    for i, file in enumerate(uploaded_files):
        resumes.append(extract_text_from_pdf(file))
        progress_bar.progress((i + 1) / len(uploaded_files))  # Update progress

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Create results DataFrame
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    # Display results with better UI
    st.subheader("📊 Resume Ranking Results")
    st.dataframe(results.style.format({"Score": "{:.2f}"}).background_gradient(cmap="Blues"))

    # CSV Download
    csv_buffer = BytesIO()
    results.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    st.download_button("📥 Download Results as CSV", data=csv_buffer, file_name="resume_ranking_results.csv",
                       mime="text/csv")

    st.success("✅ Ranking Completed Successfully!")
