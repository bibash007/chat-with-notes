import os
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (for GEMINI_API_KEY)
load_dotenv()

# 1. Setup Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your_api_key_here":
    print("Error: Please set your GEMINI_API_KEY in the .env file.")
    exit()

genai.configure(api_key=api_key)
# We use gemini-3.5-flash as it's fast and cost-effective
model = genai.GenerativeModel('gemini-3.5-flash')

# 2. Load Vector DB and Embedding Model
print("Loading Database and Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="my_notes")

def chat_with_notes(query):
    # A. EMBED THE QUERY
    query_embedding = embedder.encode(query).tolist()
    
    # B. RETRIEVE RELEVANT CHUNKS
    # We ask ChromaDB for the top 3 most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    retrieved_chunks = results['documents'][0]
    sources = results['metadatas'][0]
    
    if not retrieved_chunks:
        return "I couldn't find any relevant information in your notes."
    
    # Combine retrieved chunks into a single context string
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        source_file = sources[i]['source']
        context += f"\n--- From {source_file} ---\n{chunk}\n"
    
    # C. GENERATE ANSWER USING LLM
    prompt = f"""
    You are a helpful assistant. Use the following context retrieved from my personal notes to answer my question. 
    If the answer is not in the context, just say "I don't know based on your notes."
    
    Context from Notes:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    response = model.generate_content(prompt)
    
    print("\n" + "="*50)
    print(f"Q: {query}")
    print("="*50)
    print(response.text)
    
    # Extract unique source filenames
    unique_sources = set([s['source'] for s in sources])
    print(f"\n(Sources used: {', '.join(unique_sources)})")
    print("="*50 + "\n")

# Simple CLI loop
print("Welcome to Chat with Notes!")
print("Type 'quit' or 'exit' to stop.")
while True:
    try:
        user_query = input("\nAsk something about your notes: ")
        if user_query.lower() in ['quit', 'exit']:
            break
        if user_query.strip():
            chat_with_notes(user_query)
    except KeyboardInterrupt:
        break
