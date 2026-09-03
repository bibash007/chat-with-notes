import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# --- PAGE CONFIG ---
st.set_page_config(page_title="Chat with Notes", page_icon="📚")
st.title("📚 Chat with your Notes")

# --- CACHING HEAVY MODELS ---
# We cache these so Streamlit doesn't reload them every time you ask a question
@st.cache_resource(show_spinner="Loading AI Models (this takes a few seconds on startup)...")
def load_models():
    # Moving heavy imports here so they don't block the initial page load!
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="my_notes")
    return embedder, collection

embedder, collection = load_models()

# --- SETUP GEMINI ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your_api_key_here":
    st.error("Please set your GEMINI_API_KEY in the .env file!")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-3.5-flash')

# --- SIDEBAR (Database Management) ---
with st.sidebar:
    st.header("Database Management")
    st.write("Upload new `.txt` notes to your database directly from the web app.")
    
    uploaded_files = st.file_uploader("Upload .txt files", type=["txt"], accept_multiple_files=True)
    
    if uploaded_files:
        st.info("⚠️ Don't forget to click 'Ingest Uploaded Files' below to add these to your database!")
        
    if st.button("Ingest Uploaded Files"):
        if uploaded_files:
            documents = []
            metadatas = []
            ids = []
            
            for file in uploaded_files:
                content = file.read().decode("utf-8")
                
                # Basic chunking (500 chars)
                chunk_size, overlap = 500, 50
                chunks = []
                start = 0
                while start < len(content):
                    end = start + chunk_size
                    chunks.append(content[start:end])
                    start += chunk_size - overlap
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({"source": file.name, "chunk_index": i})
                    ids.append(f"{file.name}_chunk_{i}")
            
            # Embed and store
            with st.spinner("Generating embeddings and saving to database..."):
                embeddings = embedder.encode(documents).tolist()
                collection.upsert(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            st.success(f"Successfully added {len(documents)} chunks to the database!")
        else:
            st.warning("Please upload files first.")
            
    st.write(f"**Total chunks in database:** {collection.count()}")

# --- CHAT INTERFACE ---
# Store conversation history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your notes..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 1. Embed the query
    query_embedding = embedder.encode(prompt).tolist()
    
    # 2. Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    retrieved_chunks = results['documents'][0]
    sources = results['metadatas'][0]
    
    if not retrieved_chunks:
        response_text = "I couldn't find any relevant information in your notes."
    else:
        # 3. Combine context
        context = ""
        unique_sources = set()
        for i, chunk in enumerate(retrieved_chunks):
            source_file = sources[i]['source']
            unique_sources.add(source_file)
            context += f"\n--- From {source_file} ---\n{chunk}\n"
        
        # 4. Generate answer with Gemini
        llm_prompt = f"""
        You are a helpful assistant. Use the following context retrieved from my personal notes to answer my question. 
        If the answer is not in the context, just say "I don't know based on your notes."
        
        Context from Notes:
        {context}
        
        Question: {prompt}
        
        Answer:
        """
        
        response = model.generate_content(llm_prompt)
        # Append the sources dynamically
        response_text = response.text + f"\n\n*(Sources: {', '.join(unique_sources)})*"
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
