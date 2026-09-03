# Retrieval-Augmented Generation (RAG) Architecture

Understanding the high-level architecture is the most valuable part of RAG. You can always look up the specific code syntax later, but knowing *how the data flows* is what makes you an AI engineer.

A RAG system always operates in two distinct phases: **Phase 1: Ingestion** (saving the knowledge) and **Phase 2: Retrieval & Generation** (answering the question).

```mermaid
flowchart TD
    %% Styling
    classDef user input fill:#f9f2f4,stroke:#d9534f,stroke-width:2px,color:#333;
    classDef process fill:#e1f5fe,stroke:#039be5,stroke-width:2px,color:#333;
    classDef model fill:#fff3e0,stroke:#fb8c00,stroke-width:2px,color:#333;
    classDef database fill:#e8f5e9,stroke:#43a047,stroke-width:2px,color:#333;

    %% Phase 1: Ingestion
    subgraph Phase 1: Ingestion Pipeline
        direction TB
        A1[Raw Documents / Notes]:::user
        A2[Text Splitter / Chunker]:::process
        A3[Embedding Model]:::model
        A4[(Vector Database)]:::database

        A1 -- "1. Read" --> A2
        A2 -- "2. Chunk into paragraphs" --> A3
        A3 -- "3. Convert chunks to Vectors (Numbers)" --> A4
    end

    %% Phase 2: Retrieval & Generation
    subgraph Phase 2: Retrieval & Generation
        direction TB
        B1[User Question]:::user
        B2[Embedding Model]:::model
        B3[(Vector Database)]:::database
        B4[Prompt Assembly]:::process
        B5[LLM Generator / Gemini]:::model
        B6[Final Answer]:::user

        B1 -- "1. Ask" --> B2
        B2 -- "2. Convert Question to Vector" --> B3
        B3 -- "3. Search for Similar Vectors" --> B4
        B4 -- "4. Combine Question + Top Chunks" --> B5
        B5 -- "5. Read Context & Generate" --> B6
    end

    %% Connecting the two phases logically
    A4 -. "Provides Knowledge Base" .-> B3
    A3 -. "Must be the SAME model" .-> B2
```

### Key Architectural Concepts

1. **Chunking**: LLMs have a "context window" (a memory limit). You can't feed a whole database into it at once. We break documents into "chunks" so we only send the highly relevant pieces.
2. **Embeddings**: An embedding is an array of floating-point numbers (e.g., `[0.12, -0.45, 0.89...]`). It represents the *semantic meaning* of a text. Words with similar meanings will have similar numbers.
3. **Vector Database**: A specialized database (like ChromaDB or Pinecone) designed purely to calculate the mathematical distance between arrays of numbers extremely fast.
4. **Prompt Assembly (The Magic Trick)**: RAG is essentially an illusion. The LLM isn't actually querying the database itself. *We* query the database, grab the text, and then build a prompt behind the scenes that looks like this:
   > *"Answer the question using this context: [Insert Database Results here]. Question: [User's Question]"*
