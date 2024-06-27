import os
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Initialize the stemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = word_tokenize(text)
    # Stopword removal and stemming
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    # Join tokens back into a single string
    return ' '.join(processed_tokens)

def read_documents_from_directory(directory_path):
    documents = []
    filenames = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                processed_text = preprocess_text(text)
                documents.append(processed_text)
                filenames.append(filename)
    return documents, filenames

def vectorize_documents(documents, query):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents + [query])
    return tfidf_matrix

def rank_documents(tfidf_matrix):
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    ranked_indices = np.argsort(cosine_similarities)[::-1]
    return ranked_indices, cosine_similarities

def main():
    # Path to the directory containing text documents
    directory_path = 'C:\\Users\\hp\\Desktop\\IR Project'  # Update this to the path where your documents are stored
    
    # Read documents from the specified directory
    documents, filenames = read_documents_from_directory(directory_path)
    
    # Define your query and preprocess it
    query = input("Enter query: ") # Update this to your actual query
    query = preprocess_text(query)
    
    # Vectorize the documents and the query
    tfidf_matrix = vectorize_documents(documents, query)
    
    # Rank the documents based on their similarity to the query
    ranked_indices, similarities = rank_documents(tfidf_matrix)
    
    # Print the ranked documents with their similarity scores
    print("Ranked documents:")
    for idx in ranked_indices:
        
        print(f"Doc.: {filenames[idx]}, Similarity: {similarities[idx]:.4f}")

if __name__ == "__main__":
    main()
