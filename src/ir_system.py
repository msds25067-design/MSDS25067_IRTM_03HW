# src/ir_system.py

import pandas as pd
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Load dataset
# -----------------------------
DATA_DIR = "../data"
DATA_FILE = "articles.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

try:
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
    print(f"Dataset loaded! Shape: {df.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Use the 'Article' column for text
TEXT_COL = 'Article'
documents = df[TEXT_COL].astype(str).tolist()

# -----------------------------
# 2. Preprocessing
# -----------------------------
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return tokens

processed_docs = [preprocess(doc) for doc in documents]
print("Preprocessing done!")

processed_texts = [" ".join(doc) for doc in processed_docs]

# -----------------------------
# 3. Boolean Index
# -----------------------------
def build_boolean_index(docs):
    index = {}
    for i, doc in enumerate(docs):
        for term in set(doc):
            if term not in index:
                index[term] = set()
            index[term].add(i)
    return index

boolean_index = build_boolean_index(processed_docs)
print("Boolean index created!")

# -----------------------------
# 4. TF-IDF Index
# -----------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)
print("TF-IDF index created!")

# -----------------------------
# 5. Search functions
# -----------------------------
def boolean_search(query, index):
    query_tokens = preprocess(query)
    if not query_tokens:
        return []
    result_sets = [index.get(token, set()) for token in query_tokens]
    result_docs = set.intersection(*result_sets) if result_sets else set()
    return list(result_docs)

def tfidf_search(query, vectorizer, matrix, top_k=5):
    query_processed = " ".join(preprocess(query))
    query_vec = vectorizer.transform([query_processed])
    cosine_sim = cosine_similarity(query_vec, matrix).flatten()
    ranked_idx = cosine_sim.argsort()[::-1][:top_k]
    return ranked_idx, cosine_sim[ranked_idx]

# -----------------------------
# 6. Example queries
# -----------------------------
query = "machine learning"
print("\nBoolean search results (doc indices):")
bool_results = boolean_search(query, boolean_index)
for i in bool_results[:5]:  # show top 5
    print(f"Doc {i}: {documents[i][:150]}...")  # show snippet

print("\nTF-IDF search results (top 5 doc indices and scores):")
idxs, scores = tfidf_search(query, tfidf_vectorizer, tfidf_matrix)
for i, score in zip(idxs, scores):
    print(f"Doc {i}: score {score:.4f}, snippet: {documents[i][:150]}...")
