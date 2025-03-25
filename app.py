import streamlit as st
import pickle
from docx import Document
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained SVM model
with open('resume.pkl', 'rb') as file:
    SVM = pickle.load(file)

# Load the CountVectorizer and Bag of Words DataFrame
with open('countvectorizer.pkl', 'rb') as cv_file:
    cv = pickle.load(cv_file)

with open('bow_df.pkl', 'rb') as bow_file:
    bow_df = pickle.load(bow_file)

# Define a function to extract text from DOCX file
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ' '.join(full_text)

# Streamlit app layout
st.title("Resume Classifier using SVM")
st.write("Upload a DOCX file to classify the resume.")

# File upload section
uploaded_file = st.file_uploader("Choose a DOCX file", type="docx")

if uploaded_file is not None:
    # Extract text from the uploaded DOCX file
    text = extract_text_from_docx(uploaded_file)

    # Display extracted text (optional)
    st.subheader("Extracted Text from DOCX:")
    st.write(text)

    # Transform the text using the vectorizer (Bag of Words)
    text_vectorized = cv.transform([text])

    # Convert sparse matrix to dense (using .toarray() if not already done)
    text_vectorized_dense = text_vectorized.toarray()

    # Make a prediction using the loaded SVM model
    prediction = SVM.predict(text_vectorized_dense)

    # Show the prediction result
    st.subheader("Prediction:")
    st.write(f"The predicted class is: {prediction[0]}")
