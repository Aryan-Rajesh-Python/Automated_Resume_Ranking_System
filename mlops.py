import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import optuna
import os
import PyPDF2

class MLOps:
    def __init__(self):
        # Set up MLflow tracking
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Resume_Ranking_Experiment")

    def objective(self, trial):
        # Define hyperparameters to optimize
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 1, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

        # Load your resume and job description data
        data = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(data['resume_text'], data['target'], test_size=0.2, random_state=42)

        # Create a pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', RandomForestClassifier(n_estimators=n_estimators,
                                                  max_depth=max_depth,
                                                  min_samples_split=min_samples_split,
                                                  random_state=42))
        ])

        # Train and evaluate the model
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        return accuracy

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        text = ""
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
        return text.strip()

    def load_data(self):
        """Load resumes and job descriptions, extracting text from PDF files."""
        # Directory where your PDF resumes are stored
        resume_directory = "C:\\Users\\aryan\\OneDrive\\Desktop\\Aryan_Rajesh_Resume.pdf"  # Update this path

        # Create a list to hold resume texts and their corresponding targets
        resumes = []
        targets = []

        # Load each resume PDF and extract text
        for filename in os.listdir(resume_directory):
            if filename.endswith(".pdf"):
                resume_path = os.path.join(resume_directory, filename)
                resume_text = self.extract_text_from_pdf(resume_path)
                resumes.append(resume_text)

                # You need to assign a target value based on your specific logic or labeling
                # For demonstration purposes, let's say all resumes are a match (1)
                targets.append(1)  # Change this logic as needed

        # Create a DataFrame
        df = pd.DataFrame({
            'resume_text': resumes,
            'target': targets
        })
        return df

    def train_model(self):
        # Optimize hyperparameters using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=100)

        best_params = study.best_params
        best_accuracy = study.best_value

        with mlflow.start_run():
            # Log best parameters and accuracy
            mlflow.log_params(best_params)
            mlflow.log_metric("accuracy", best_accuracy)

            # Log the best model
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('classifier', RandomForestClassifier(**best_params, random_state=42))
            ])
            data = self.load_data()
            X = data['resume_text']
            y = data['target']
            pipeline.fit(X, y)
            mlflow.sklearn.log_model(pipeline, "model")

            print(f"Model trained with best parameters: {best_params}")
            print(f"Model accuracy: {best_accuracy:.2f}")

if __name__ == "__main__":
    mlops = MLOps()
    mlops.train_model()
