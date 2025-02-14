import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pdf_processor import segment_text, extract_text

# Function to create embeddings from text segments
def create_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return model, embeddings

# Function to build a FAISS index given embeddings
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance index
    index.add(embeddings)  # Add embeddings to the index
    return index

# Function to search the index with a query
def search_index(query, model, index, chapters, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)

    results = [(chapters[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results

# if __name__ == "__main__":
#     pdf_path = "documents/Designing_Machine_Learning_Systems.pdf"
    
#     # Step 1: Extract and segment text by chapters
#     full_text = extract_text(pdf_path)
#     segments = segment_text(full_text)
#     print(f"Extracted {len(segments)} chapters from the textbook.")
    
#     # Step 2: Create embeddings for each chapter
#     model, embeddings = create_embeddings(segments)
#     print("Embeddings created for each chapter.")
    
#     # Step 3: Build the FAISS index using the embeddings
#     index = build_faiss_index(embeddings)
#     print(f"FAISS index built with {index.ntotal} items.")
    
#     # Step 4: Test the search functionality
#     test_query = "How do I avoid human biases in selecting models?"
#     results = search_index(test_query, model, index, segments, k=3)
    
#     print("\nSearch Results:")
#     for idx, (segment_text, distance) in enumerate(results, start=1):
#         print(f"\n--- Result {idx} (Distance: {distance:.4f}) ---")
#         print(segment_text[:300])