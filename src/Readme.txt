IR System – Boolean & TF-IDF
=====================================
This project implements a basic information retrieval system using two methods: Boolean retrieval and TF-IDF ranked retrieval. The system loads a news dataset, preprocesses the text, builds indexes, and returns search results in the terminal.

Project Features

Reads dataset from /data folder

Preprocessing: tokenization, lowercasing, stopword removal, lemmatization

Inverted Index for Boolean search

TF-IDF Matrix for ranked search

Returns document IDs + top results with scores

How to Run
python -m venv venv
venv\Scripts\activate
pip install numpy pandas scikit-learn nltk rank-bm25 tqdm
cd src
python ir_system.py

Methods

Boolean Retrieval: exact-match search using inverted index

TF-IDF Retrieval: cosine similarity ranking using scikit-learn

Output

Boolean: list of documents that match all query terms

TF-IDF: top ranked documents with similarity scores and snippets