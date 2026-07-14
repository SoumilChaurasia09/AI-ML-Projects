import os
import sys
import pickle
import math
import re
import urllib.request
import json

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
    cleaned = re.sub(r'[^\w\s-]', '', text.lower())
    tokens = cleaned.split()
    return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]

def retrieve(query_text, model_data, top_k=3):
    vocabulary = model_data["vocabulary"]
    idf = model_data["idf"]
    chunk_vectors = model_data["chunk_vectors"]
    
    query_tokens = tokenize(query_text)
    if not query_tokens:
        return []
        
    query_tf = {}
    for t in query_tokens:
        query_tf[t] = query_tf.get(t, 0) + 1
        
    query_vector = {}
    total_query_tokens = len(query_tokens)
    
    for term in vocabulary:
        if term in query_tf:
            query_vector[term] = (query_tf[term] / total_query_tokens) * idf[term]
        else:
            query_vector[term] = 0.0
            
    sum_sq = sum(v * v for v in query_vector.values())
    query_norm = math.sqrt(sum_sq) if sum_sq > 0 else 1.0
    
    results = []
    for item in chunk_vectors:
        doc_vector = item["vector"]
        doc_norm = item["norm"]
        
        dot_product = 0.0
        for term in query_tokens:
            if term in doc_vector:
                dot_product += query_vector[term] * doc_vector[term]
                
        similarity = dot_product / (query_norm * doc_norm)
        results.append({
            "chunk": item["chunk"],
            "score": similarity
        })
        
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def generate_local_response(query, results):
    """Fallback offline local answer generation."""
    top_result = results[0]
    
    # Check for exact dataset QA patterns
    exact_match = None
    # We can load dataset.json locally if it exists in same folder
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'dataset.json')
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            for item in dataset:
                q_words = tokenize(item["question"])
                user_words = tokenize(query)
                overlap = len([w for w in q_words if w in user_words])
                if overlap >= len(q_words) * 0.75:
                    exact_match = item
                    break
                    
    if exact_match:
        return f"[Source: {exact_match['category']} Archives]\n{exact_match['answer']}"
        
    ans = f"[Retrieved Admissions Knowledge (Cosine Score: {top_result['score']:.3f})]\n"
    ans += f"Based on the official {top_result['chunk']['category']} prospectus logs:\n"
    ans += f"\"{top_result['chunk']['raw_content']}\""
    return ans

def generate_gemini_llm_response(api_key, system_prompt, query_text):
    """Queries standard Google Gemini 1.5 Flash REST API endpoint using Python standard urllib library."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "systemInstruction": {
            "parts": [
                {"text": "You are the VIT Bhopal ChatBox admissions counselor assistant. Keep responses helpful, structured, clear, and based ONLY on the provided context logs."}
            ]
        },
        "contents": [
            {
                "parts": [
                    {"text": system_prompt + "\n\nStudent Query: " + query_text}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.15,
            "maxOutputTokens": 600
        }
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            text = res_data['candidates'][0]['content']['parts'][0]['text']
            return text
    except Exception as e:
        raise Exception(f"Failed to query Gemini API: {str(e)}")

def main():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    
    if not os.path.exists(model_path):
        print("[Error] Trained model file 'model.pkl' not found.")
        print("Please run model training first: python train.py")
        sys.exit(1)
        
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    print("[OK] Loaded VIT Bhopal RAG Vector Index database.")
    
    # API key setup check
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("\nNote: GEMINI_API_KEY environment variable not detected.")
        print("Paste your Google Gemini API Key to enable live LLM answering (or press Enter to run in offline local simulation mode):")
        input_key = input("API Key > ").strip()
        if input_key:
            api_key = input_key
            print("✓ Gemini API key loaded. Online live Counselor mode enabled.")
        else:
            print("Running in offline local simulation mode.")
            
    print("--------------------------------------------------")
    print("Enter a query below (or type 'exit' to quit):")
    print("--------------------------------------------------")
    
    while True:
        try:
            query = input("\nStudent Query > ").strip()
            if not query:
                continue
            if query.lower() in ('exit', 'quit', 'q'):
                print("Exiting search console.")
                break
                
            results = retrieve(query, model_data, top_k=3)
            
            if not results or results[0]["score"] <= 0.05:
                print("[Error] No relevant document chunks found in index.")
                continue
                
            print("\n[RAG Pipeline: Context retrieved. Generating answer...]")
            
            # Format context segments
            context_blocks = "\n\n".join(
                f"[Category: {r['chunk']['category']}]\nContent: {r['chunk']['raw_content']}\n---"
                for r in results
            )
            
            system_prompt = f"You are a helpful and detailed university admissions counselor AI assistant.\n" \
                            f"Answer the student's question based ONLY on the provided context document segments below.\n" \
                            f"Provide accurate details about pricing (in INR), credit metrics, and placement rates.\n" \
                            f"Always cite the category source logs when stating facts.\n" \
                            f"If the context segments do not contain sufficient information to answer, politely state that you cannot find this in the official admissions database and offer to direct them to the admissions office.\n\n" \
                            f"CONTEXT DOCUMENTS:\n{context_blocks}"
            
            if api_key:
                try:
                    answer = generate_gemini_llm_response(api_key, system_prompt, query)
                    print(f"\n[VIT Bhopal ChatBox - Gemini LLM Answer]")
                    print(answer)
                except Exception as err:
                    print(f"\n[Error querying LLM: {err}]")
                    print("Falling back to local simulated response...")
                    answer = generate_local_response(query, results)
                    print(f"\n[VIT Bhopal ChatBox - Offline Heuristic Answer]")
                    print(answer)
            else:
                answer = generate_local_response(query, results)
                print(f"\n[VIT Bhopal ChatBox - Offline Heuristic Answer]")
                print(answer)
                
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting search console.")
            break

if __name__ == "__main__":
    main()
