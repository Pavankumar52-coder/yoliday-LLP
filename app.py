# Import necessary libraries
import streamlit as st
from pdfminer.high_level import extract_text
from docx import Document
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient

# Set up MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["docquery"]
collection = db["documents"]

# Function to extract text from PDF
def extract_text_from_pdf(file):
    return extract_text(file)

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

# Function to tokenize text
def tokenize_text(text):
    return word_tokenize(text)

# Function to remove stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

# Function to calculate TF-IDF
def calculate_tfidf(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix

# Function to calculate cosine similarity
def calculate_cosine_similarity(tfidf_matrix, query):
    query_tfidf = calculate_tfidf([query])
    similarity_matrix = cosine_similarity(tfidf_matrix, query_tfidf)
    return similarity_matrix

# Function to retrieve relevant documents
def retrieve_relevant_documents(query, documents):
    tfidf_matrix = calculate_tfidf(documents)
    similarity_matrix = calculate_cosine_similarity(tfidf_matrix, query)
    relevant_documents = []
    for i, similarity in enumerate(similarity_matrix):
        if similarity > 0.5:
            relevant_documents.append(documents[i])
    return relevant_documents

# Streamlit app
st.title("DocQuery")

# Upload document
file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Display uploaded document
if file:
    if file.type == "application/pdf":
        text = extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(file)
    else:
        text = file.read()
    st.write(text)

# Query input
query = st.text_input("Enter your query")

# Retrieve relevant documents
if query:
    documents = [doc["text"] for doc in collection.find()]
    relevant_documents = retrieve_relevant_documents(query, documents)
    st.write("Relevant Documents:")
    for doc in relevant_documents:
        st.write(doc)

# User history
user_history = st.text_area("User History")

# Download chat history
if st.button("Download Chat History"):
    with open("chat_history.txt", "w") as f:
        f.write(user_history)
    st.success("Chat history downloaded successfully!")

# Add document to database
if st.button("Add Document to Database"):
    if file:
        collection.insert_one({"text": text})
        st.success("Document added to database successfully!")
