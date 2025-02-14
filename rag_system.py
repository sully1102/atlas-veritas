import os
import cohere
from dotenv import load_dotenv
from indexer import search_index, create_embeddings, build_faiss_index
from pdf_processor import extract_text, segment_text

load_dotenv()

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

def generate_answer(query, context):
    prompt = f"""
    You are a helpful educational assistant. Given the following context from a textbook:
    
    {context}
    
    Please answer this question clearly and concisely:
    
    {query}
    """
    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=500,
        temperature=0.2,
    )
    return response.generations[0].text.strip()

def answer_query(query, index, model, segments, k=3):
    results = search_index(query, model, index, segments, k=k)
    combined_context = "\n\n".join([segment for segment, _ in results])
    return generate_answer(query, combined_context)

if __name__ == "__main__":
    pdf_path = "documents/Designing_Machine_Learning_Systems.pdf"
    
    print("Begin processing the textbook...")
    full_text = extract_text(pdf_path)
    segments = segment_text(full_text, max_words=150)
    print(f"Extracted {len(segments)} segments from the textbook.")
    
    # Step 3: Create embeddings for each segment
    model, embeddings = create_embeddings(segments)
    
    # Step 4: Build the FAISS index using the embeddings
    index = build_faiss_index(embeddings)
    print(f"FAISS index built with {index.ntotal} items.")
    
    # Step 5: Test a query and generate an answer
    test_query = "How do I avoid human bias in selecting a model?"
    answer = answer_query(test_query, index, model, segments, k=3)
    
    print("\nGenerated Answer:")
    print(answer)
