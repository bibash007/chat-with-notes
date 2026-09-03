import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize the Embedding Model
print("Loading embedding model...")
# all-MiniLM-L6-v2 is a great, lightweight embedding model that runs locally
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Initialize ChromaDB
print("Initializing ChromaDB...")
# This will save the database to a local folder named 'chroma_db'
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get a collection (like a table in a relational DB)
collection = client.get_or_create_collection(name="my_notes")

# 3. Read and Chunk Notes
notes_dir = "./notes"
if not os.path.exists(notes_dir):
    os.makedirs(notes_dir)
    print(f"Created {notes_dir} directory. Please add some .txt files there!")
    exit()

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Very simple chunking function.
    Splits text into chunks of `chunk_size` characters with some `overlap`.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

documents = []
metadatas = []
ids = []
doc_id_counter = 1

print("Reading and chunking notes...")
for filepath in glob.glob(os.path.join(notes_dir, "*.txt")):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        filename = os.path.basename(filepath)
        
        # Chunk the content
        chunks = chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            # Store metadata so we know where this chunk came from
            metadatas.append({"source": filename, "chunk_index": i})
            ids.append(f"{filename}_chunk_{i}")
            doc_id_counter += 1

if not documents:
    print("No notes found. Please add .txt files to the 'notes' directory.")
    exit()

# 4. Embed and Store
print(f"Generating embeddings for {len(documents)} chunks...")
# Convert text chunks into numerical vectors
embeddings = embedder.encode(documents).tolist()

print("Storing in ChromaDB...")
# Add to our vector database
collection.upsert( # upsert instead of add so we can run it multiple times without crashing
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print("Ingestion complete! Database is ready.")
