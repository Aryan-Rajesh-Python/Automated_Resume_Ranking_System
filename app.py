from flask import Flask, render_template, request, send_file
import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import csv

app = Flask(__name__)

# Load spaCy NER model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError as e:
    print("Error loading spaCy model: ", e)

# Initialize results and feedback variables
results = []
feedbacks = []

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text

# Extract entities using regex
def extract_entities(text):
    emails = re.findall(r'\S+@\S+', text)
    names = re.findall(r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)', text)
    if names:
        names = [" ".join(names[0])]
    return emails, names

@app.route('/', methods=['GET', 'POST'])
def index():
    global results  # Ensure results is accessible in the download route
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resume_files')

        # Create a directory for uploads if it doesn't exist
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        # Process uploaded resumes
        processed_resumes = []
        for resume_file in resume_files:
            # Save the uploaded file
            resume_path = os.path.join("uploads", resume_file.filename)
            resume_file.save(resume_path)

            # Process the saved file
            resume_text = extract_text_from_pdf(resume_path)
            emails, names = extract_entities(resume_text)
            processed_resumes.append((names, emails, resume_text))

        # TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()
        job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

        # Rank resumes based on similarity
        ranked_resumes = []
        for (names, emails, resume_text) in processed_resumes:
            resume_vector = tfidf_vectorizer.transform([resume_text])
            similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100 
            ranked_resumes.append((names, emails, similarity))

        # Sort resumes by similarity score
        ranked_resumes.sort(key=lambda x: x[2], reverse=True)

        # Append the newly ranked resumes to the results
        results.extend(ranked_resumes)

    return render_template('index.html', results=results)

@app.route('/feedback', methods=['POST'])
def feedback():
    global feedbacks  # Use the global feedback variable
    feedback_text = request.form['feedback']
    feedbacks.append(feedback_text)  # Store feedback
    return "Feedback submitted successfully!"

@app.route('/download_feedback')
def download_feedback():
    global feedbacks  # Use the global feedback variable
    if not feedbacks:
        return "No feedback available to download."

    # Generate the CSV content
    feedback_csv_content = "Feedback\n"
    for feedback in feedbacks:
        feedback_csv_content += f"{feedback}\n"

    # Create a temporary file to store the feedback content
    feedback_filename = "feedback.csv"
    with open(feedback_filename, "w", encoding='utf-8') as feedback_file:
        feedback_file.write(feedback_csv_content)

    # Send the file for download
    feedback_full_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), feedback_filename)
    return send_file(feedback_full_path, as_attachment=True, download_name="feedback.csv")

@app.route('/download_csv')
def download_csv():
    global results  # Use the global results variable
    if not results:
        return "No results available to download."

    # Generate the CSV content
    csv_content = "Rank,Name,Email,Similarity\n"
    for rank, (names, emails, similarity) in enumerate(results, start=1):
        name = names[0] if names else "N/A"
        email = emails[0] if emails else "N/A"
        csv_content += f"{rank},{name},{email},{similarity:.2f}\n"

    # Create a temporary file to store the CSV content
    csv_filename = "ranked_resumes.csv"
    with open(csv_filename, "w", encoding='utf-8') as csv_file:
        csv_file.write(csv_content)

    # Send the file for download
    csv_full_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), csv_filename)
    return send_file(csv_full_path, as_attachment=True, download_name="ranked_resumes.csv")

if __name__ == '__main__':
    app.run(debug=True)