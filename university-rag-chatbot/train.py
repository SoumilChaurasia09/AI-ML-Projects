import os
import json
import math
import re
import pickle

print("==================================================")
print("VIT Bhopal University - RAG Model Training Pipeline")
print("==================================================")

# Stop words to filter out common English terms
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
    'before', 'after', 'above', 'below', 'from', 'up', 'down', 'out', 'off', 'over', 'under', 
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its',
    'they', 'them', 'their', 'am', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'as', 'until', 'while', 'because', 'what', 'which', 'who', 'whom', 'where',
    'why', 'how', 'are', 'is', 'am'
}

def tokenize(text):
    """Tokenize, clean, and lowercase words, filtering out stop words."""
    cleaned = re.sub(r'[^\w\s-]', '', text.lower())
    tokens = cleaned.split()
    return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]

def chunk_document(doc_text, sentence_count=2, overlap=1):
    """Parse prospectus document into sections and split them into overlapping sentence chunks."""
    sections = doc_text.split('[Section:')
    chunks = []
    
    for section in sections:
        if not section.strip():
            continue
        parts = section.split(']', 1)
        if len(parts) < 2:
            continue
        section_title = parts[0].strip()
        section_body = parts[1].strip()
        
        # Split by sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', section_body)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        i = 0
        while i < len(sentences):
            chunk_sentences = sentences[i : i + sentence_count]
            if not chunk_sentences:
                break
            
            chunk_text = " ".join(chunk_sentences).strip()
            if len(chunk_text) >= 15:
                # Prepend section tag to boost relevance
                enriched_text = f"[{section_title}] {chunk_text}"
                chunks.append({
                    "id": f"chunk_{section_title.lower().replace(' ', '_')}_{len(chunks)}",
                    "category": section_title,
                    "content": enriched_text,
                    "raw_content": chunk_text
                })
            
            if i + sentence_count >= len(sentences):
                break
            i += (sentence_count - overlap)
            
    return chunks

def train_and_index():
    # 1. Load document
    doc_path = os.path.join(os.path.dirname(__file__), 'dataset', 'document.txt')
    if not os.path.exists(doc_path):
        print(f"Error: Prospectus file {doc_path} not found.")
        return
        
    with open(doc_path, 'r', encoding='utf-8') as f:
        doc_text = f.read()
        
    print("[OK] Loaded prospectus text document.")
    
    # 2. Chunk document
    chunks = chunk_document(doc_text, sentence_count=2, overlap=1)
    print(f"[OK] Document parsed into {len(chunks)} overlapping text chunks.")
    
    # 3. Build TF-IDF weights and vocabulary index
    print("Indexing terms and training TF-IDF vectors...")
    
    vocabulary = set()
    doc_frequency = {}
    tokenized_chunks = []
    
    for chunk in chunks:
        tokens = tokenize(chunk['content'])
        tokenized_chunks.append((chunk, tokens))
        for t in set(tokens):
            vocabulary.add(t)
            doc_frequency[t] = doc_frequency.get(t, 0) + 1
            
    # Calculate IDFs
    N = len(chunks)
    idf = {}
    for term in vocabulary:
        idf[term] = math.log(1 + N / doc_frequency[term]) + 1
        
    # Calculate TF-IDF vectors
    chunk_vectors = []
    for chunk, tokens in tokenized_chunks:
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
            
        vector = {}
        total_tokens = len(tokens) if tokens else 1
        for term in vocabulary:
            if term in tf:
                vector[term] = (tf[term] / total_tokens) * idf[term]
            else:
                vector[term] = 0.0
                
        # Compute L2 Norm (length)
        sum_sq = sum(v * v for v in vector.values())
        norm = math.sqrt(sum_sq) if sum_sq > 0 else 1.0
        
        chunk_vectors.append({
            "chunk": chunk,
            "vector": vector,
            "norm": norm
        })
        
    print(f"[OK] Model trained: Vocabulary size = {len(vocabulary)} terms.")
    
    # 4. Serialize index to model.pkl
    model_data = {
        "vocabulary": vocabulary,
        "idf": idf,
        "doc_frequency": doc_frequency,
        "chunk_vectors": chunk_vectors,
        "total_chunks": N
    }
    
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"[OK] Model serialized and saved successfully to: {model_path}")
    print("RAG model index is compiled and active!")
    print("==================================================")

if __name__ == "__main__":
    train_and_index()
