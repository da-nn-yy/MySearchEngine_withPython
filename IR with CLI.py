from collections import defaultdict
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import numpy as np
import os


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def read_documents_from_directory(dir):
    documents = {}
    for filename in os.listdir(dir):
        if filename.endswith(".txt"):
            with open(os.path.join(dir, filename), 'r', encoding='utf-8') as file:
                doc_id = filename 
                documents[doc_id] = file.read()
    return documents


def preprocess(text):
    terms = re.findall(r'\w+', text.lower())
    processed_terms = [stemmer.stem(term) for term in terms if term not in stop_words]
    return processed_terms


def create_tf_index(documents):
    tf_index = defaultdict(lambda: defaultdict(int))
    for doc_id, text in documents.items():
        terms = preprocess(text)
        for term in terms:
            tf_index[doc_id][term] += 1
    return tf_index


def calculate_idf(tf_index, total_docs):
    df = defaultdict(int)
    idf = {}
    
    for doc_id, term_freqs in tf_index.items():
        for term in term_freqs:
            df[term] += 1
    
    for term, doc_count in df.items():
        idf[term] = math.log(total_docs / float(doc_count))
    
    return idf


def calculate_tfidf(tf_index, idf):
    tfidf_index = defaultdict(lambda: defaultdict(float))
    
    for doc_id, term_freqs in tf_index.items():
        for term, freq in term_freqs.items():
            tfidf_index[doc_id][term] = freq * idf[term]
    
    return tfidf_index


def create_document_vectors(tfidf_index, idf):
    vocab = list(idf.keys())
    doc_vectors = defaultdict(lambda: np.zeros(len(vocab)))
    
    for doc_id, term_scores in tfidf_index.items():
        for term, score in term_scores.items():
            term_index = vocab.index(term)
            doc_vectors[doc_id][term_index] = score
    
    return doc_vectors, vocab


def preprocess_query(query, idf):
    terms = preprocess(query)
    query_vector = np.zeros(len(idf))
    
    for term in terms:
        if term in idf:
            term_index = list(idf.keys()).index(term)
            query_vector[term_index] += 1
    
    return query_vector


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)


def rank_documents(query_vector, doc_vectors):
    similarities = []
    
    for doc_id, doc_vector in doc_vectors.items():
        sim = cosine_similarity(query_vector, doc_vector)
        similarities.append((doc_id, sim))
    
    ranked_docs = sorted(similarities, key=lambda x: x[1], reverse=True)
    return ranked_docs


dir = 'C:\\Users\\hp\\Desktop\\IR Project\\Doc-Corpus\\'  
documents = read_documents_from_directory(dir)

# Full text retrieval process
tf_index = create_tf_index(documents)
idf = calculate_idf(tf_index, len(documents))
tfidf_index = calculate_tfidf(tf_index, idf)
doc_vectors, vocab = create_document_vectors(tfidf_index, idf)


query = input("Enter your query: ")
query_vector = preprocess_query(query, idf)
ranked_docs = rank_documents(query_vector, doc_vectors)


print(f"Ranked documents for your '{query}':")
for doc_id, sim in ranked_docs:
    print(f"Doc: {doc_id}, Similarity: {sim:.4f}")

