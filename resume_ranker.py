import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Get job description from user input
job_description = input("Enter the job description: ")

# List of resume PDF file paths (this will be populated dynamically)
resume_paths = []  # This will be populated by user uploads

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

# Extract emails and names using spaCy NER
def extract_entities(text):
    emails = re.findall(r'\S+@\S+', text)  # Extract emails using regex
    names = re.findall(r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)', text)  # Extract names
    if names:
        names = [" ".join(names[0])]
    return emails, names

# Store feedback
def store_feedback(feedback_text):
    with open("feedback.csv", "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([feedback_text])

# Main function to analyze resumes
@app.route("/", methods=["POST"])
def analyze_resumes():
    global resume_paths
    job_description = request.form["job_description"]
    resume_files = request.files.getlist("resume_files")

    # Save uploaded resume files
    resume_paths = []
    for resume_file in resume_files:
        resume_path = os.path.join("uploads", resume_file.filename)
        resume_file.save(resume_path)
        resume_paths.append(resume_path)

    # Extract job description features using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

    # Rank resumes based on similarity
    ranked_resumes = []
    for resume_path in resume_paths:
        resume_text = extract_text_from_pdf(resume_path)
        emails, names = extract_entities(resume_text)
        resume_vector = tfidf_vectorizer.transform([resume_text])
        similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0]
        ranked_resumes.append((names, emails, similarity))

    # Sort resumes by similarity score
    ranked_resumes.sort(key=lambda x: x[2], reverse=True)

    # Save results to CSV
    csv_filename = "ranked_resumes.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Rank", "Name", "Email", "Similarity"])
        
        for rank, (names, emails, similarity) in enumerate(ranked_resumes, start=1):
            name = names[0] if names else "N/A"
            email = emails[0] if emails else "N/A"
            csv_writer.writerow([rank, name, email, similarity])

    return ranked_resumes  # Return ranked results for rendering

# API endpoint to receive feedback
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    feedback_text = data.get("feedback")
    if feedback_text:
        store_feedback(feedback_text)
        return jsonify({"message": "Feedback submitted successfully!"}), 200
    return jsonify({"message": "Feedback cannot be empty!"}), 400

if __name__ == "__main__":
    app.run(debug=True)
