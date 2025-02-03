# search-lab

Indexing and Searching Tools !!!

Search Models:
- Full Text Search,
- Semantic Search,
- ...

## Tools

> ...

## TODO

FAISS (Facebook AI Similarity Search) is a powerful library for efficient similarity search and clustering of dense vectors. You can use FAISS to index files and perform semantic search by following these steps:

### **Steps to Index Files and Perform Semantic Search with FAISS**
#### **1. Install Dependencies**
You'll need `faiss` and `sentence-transformers` for vectorizing text.

```bash
pip install faiss-cpu sentence-transformers numpy
```

#### **2. Load and Vectorize the Data**
You'll convert your text data into embeddings using `sentence-transformers`.

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load pre-trained model for text embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example documents (You can replace this with file content)
documents = [
    "AI is transforming the world.",
    "Machine learning allows computers to learn from data.",
    "FAISS is an efficient similarity search library.",
    "Natural Language Processing (NLP) helps computers understand human language."
]

# Convert text documents into embeddings
embeddings = model.encode(documents, convert_to_numpy=True)

# Normalize embeddings for better performance
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
```

#### **3. Create a FAISS Index and Store Embeddings**
```python
# Define the dimension of embeddings
dimension = embeddings.shape[1]  

# Create a FAISS index (L2 distance is used here)
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
index.add(embeddings)

# Save the index to a file (optional)
faiss.write_index(index, "faiss_index.bin")
```

#### **4. Perform Semantic Search**
Now, let's search for a query in our FAISS index.

```python
def search(query, top_k=2):
    # Convert query to embedding
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Search in FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Return results
    return [(documents[i], distances[0][j]) for j, i in enumerate(indices[0])]

# Example search query
query = "What is FAISS?"
results = search(query, top_k=2)

# Print search results
for text, score in results:
    print(f"Document: {text}, Score: {score}")
```

#### **5. Load an Existing FAISS Index (Optional)**
```python
index = faiss.read_index("faiss_index.bin")
```

## References
- https://github.com/facebookresearch/faiss
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://github.com/vespa-engine/vespa
- https://github.com/milvus-io/milvus
- https://github.com/meilisearch/meilisearch
- https://righteous-guardian-68f.notion.site/Retrieval-System-Search-System-314ec15b96be4cf19365e5cd0bf73a96?pvs=4
