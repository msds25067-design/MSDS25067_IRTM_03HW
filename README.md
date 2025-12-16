# Information Retrieval System

A Python-based information retrieval system that implements both Boolean and TF-IDF search methods for document retrieval from a CSV dataset.

## Features

- **Boolean Search**: Fast exact-match retrieval using inverted index
- **TF-IDF Search**: Ranked retrieval based on term frequency-inverse document frequency with cosine similarity
- **Text Preprocessing**: Includes tokenization, lemmatization, stopword removal, and punctuation cleaning
- **Flexible Query Processing**: Supports natural language queries

## Requirements

```
pandas
nltk
scikit-learn
```

## Installation

1. Clone the repository or download the source code

2. Install required dependencies:
```bash
pip install pandas nltk scikit-learn
```

3. Ensure the dataset is placed in the correct location:
```
project/
├── src/
│   └── ir_system.py
└── data/
    └── articles.csv
```

## Dataset Format

The system expects a CSV file (`articles.csv`) with at least an `Article` column containing the text documents. The file should use ISO-8859-1 encoding.

## Usage

Run the information retrieval system:

```bash
python ir_system.py
```

### Modifying the Query

To search for different terms, modify the `query` variable in the script:

```python
query = "machine learning"  # Change this to your search term
```

### Search Methods

**Boolean Search:**
- Returns documents containing ALL query terms
- Results are unranked (order may vary)
- Fast for exact matching

**TF-IDF Search:**
- Returns top-k most relevant documents
- Results are ranked by relevance score
- Better for semantic similarity

## Code Structure

### 1. Data Loading
Loads the CSV dataset and extracts text from the 'Article' column.

### 2. Text Preprocessing
- Converts text to lowercase
- Removes punctuation
- Tokenizes text into words
- Removes English stopwords
- Lemmatizes tokens to their base form

### 3. Boolean Index
Builds an inverted index mapping each unique term to the set of documents containing it.

### 4. TF-IDF Index
Creates a TF-IDF vector representation of all documents using scikit-learn's TfidfVectorizer.

### 5. Search Functions

**`boolean_search(query, index)`**
- Performs AND operation on query terms
- Returns list of document indices

**`tfidf_search(query, vectorizer, matrix, top_k=5)`**
- Computes cosine similarity between query and documents
- Returns top-k document indices and their scores

## Example Output

```
Dataset loaded! Shape: (1000, 5)
Preprocessing done!
Boolean index created!
TF-IDF index created!

Boolean search results (doc indices):
Doc 42: Machine learning is a subset of artificial intelligence that focuses...
Doc 157: Recent advances in machine learning have enabled...

TF-IDF search results (top 5 doc indices and scores):
Doc 42: score 0.8521, snippet: Machine learning is a subset of artificial intelligence...
Doc 157: score 0.7834, snippet: Recent advances in machine learning have enabled...
```

## Customization

### Adjusting Top-K Results
Change the `top_k` parameter in the `tfidf_search` function:

```python
idxs, scores = tfidf_search(query, tfidf_vectorizer, tfidf_matrix, top_k=10)
```

### Modifying Preprocessing
Edit the `preprocess()` function to add or remove preprocessing steps.

### Using Different Columns
Change the `TEXT_COL` variable to use a different column from your CSV:

```python
TEXT_COL = 'Content'  # Or any other column name
```

## Notes

- NLTK downloads (punkt, stopwords, wordnet) happen automatically on first run
- The system stores processed documents in memory for fast retrieval
- Boolean search may return no results if any query term is missing from all documents
- TF-IDF search always returns the top-k documents, even with low scores

## Future Enhancements

- Interactive command-line interface
- Support for OR and NOT boolean operations
- Query expansion and spell correction
- Evaluation metrics (precision, recall, F1)
- Web interface for easier querying
- Support for multiple document formats (PDF, TXT, JSON)

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
