import os
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import tkinter as tk
from tkinter import filedialog, messagebox

# Initialize the stemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    
    text = text.lower()
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = word_tokenize(text)
    
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
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

def process_query(query, directory_path):
   
    documents, filenames = read_documents_from_directory(directory_path)

    query = preprocess_text(query)

    tfidf_matrix = vectorize_documents(documents, query)


    ranked_indices, similarities = rank_documents(tfidf_matrix)

    ranked_documents = [(filenames[idx], similarities[idx]) for idx in ranked_indices]
    return ranked_documents

def browse_directory():
    directory_path = filedialog.askdirectory()
    if directory_path:
        directory_entry.delete(0, tk.END)
        directory_entry.insert(0, directory_path)

def process_query_and_display():
    query = query_entry.get().strip()
    directory_path = directory_entry.get().strip()
    if not query or not directory_path:
        messagebox.showerror("Error", "Please enter a query and select a directory.")
        return
    try:
        ranked_documents = process_query(query, directory_path)
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        for idx, (filename, similarity) in enumerate(ranked_documents, start=1):
            result_text.insert(tk.END, f"{idx}. Document: {filename}, Similarity: {similarity:.4f}\n")
        result_text.config(state=tk.DISABLED)
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("ASTU IR-project")


query_label = tk.Label(root, text="Enter query:")
query_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

query_entry = tk.Entry(root, width=50)
query_entry.grid(row=0, column=1, padx=5, pady=5)

directory_label = tk.Label(root, text="Select directory:")
directory_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)

directory_entry = tk.Entry(root, width=50)
directory_entry.grid(row=1, column=1, padx=5, pady=5)

browse_button = tk.Button(root, text="Browse", command=browse_directory)
browse_button.grid(row=1, column=2, padx=5, pady=5)

process_button = tk.Button(root, text="Process Query", command=process_query_and_display)
process_button.grid(row=2, column=0, columnspan=3, pady=5)

result_label = tk.Label(root, text="Ranked Documents:")
result_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)

result_text = tk.Text(root, width=80, height=20, wrap="word", state=tk.DISABLED)
result_text.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

root.mainloop()
