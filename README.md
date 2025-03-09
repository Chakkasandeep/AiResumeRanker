# AI Resume Screening & Candidate Ranking System

This is a Streamlit-based web application designed for AI-driven resume screening and ranking. The system allows users to upload resumes in PDF format and a job description, and ranks the resumes based on how closely they match the job description.

## Features:
- **Job Description Input**: Users can input a job description to be used for resume screening.
- **Resume Upload**: Users can upload multiple resumes in PDF format.
- **Ranking System**: The system ranks the uploaded resumes based on how closely they match the job description using cosine similarity between the job description and each resume.
- **High Contrast Mode**: The app supports forced colors for accessibility (high contrast mode) on supported devices.

## Requirements

To run this app, you'll need the following Python libraries:

- `streamlit`: For building the web interface.
- `PyPDF2`: For extracting text from PDF files.
- `pandas`: For data handling and displaying results.
- `scikit-learn`: For the TF-IDF vectorization and cosine similarity calculations.

You can install the required libraries using `pip`:

```bash
pip install streamlit PyPDF2 pandas scikit-learn
