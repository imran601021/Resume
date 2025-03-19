import re
import spacy

# Load the SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Preprocess the parsed resume text
cleaned_resume_text = preprocess_text(resume_text)
print(cleaned_resume_text[:])
