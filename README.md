# Campus Placement RAG Assistant

A Retrieval-Augmented Generation (RAG) system that lets you upload a PDF — like a company brochure or job description — and ask questions about it in plain English, getting answers grounded in that specific document.

## Why this exists

Manually searching through placement PDFs for specific details (eligibility criteria, CGPA cutoffs, role descriptions) is time-consuming. A general AI model has no knowledge of a specific company's brochure since it was never part of its training data. This project connects an LLM to your own documents so it can answer questions using only that content.

## How it works

**Ingestion pipeline:**
1. Extract raw text from an uploaded PDF using `pypdf`
2. Split the text into ~150-word chunks with 30-word overlap (overlap prevents important sentences from being lost at chunk boundaries)
3. Convert each chunk into an embedding using `sentence-transformers` (`all-MiniLM-L6-v2`), running locally
4. Store chunks and embeddings in **ChromaDB**, a vector database — each chunk ID is prefixed with its source filename, so multiple PDFs can coexist without collisions

**Retrieval pipeline:**
1. Convert the user's question into an embedding using the same model
2. Retrieve the top 3 most relevant chunks from ChromaDB based on semantic similarity, not keyword matching
3. Combine the retrieved chunks with the question into a single prompt
4. Send it to the **Gemini API**, instructed to answer only from the provided context — reducing hallucination

## Tech stack

- **Backend:** FastAPI
- **Vector DB:** ChromaDB
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **LLM:** Google Gemini API
- **PDF parsing:** pypdf

## Project structure  
campus-placement-rag/
├── app/
│ └── main.py # FastAPI app — /upload and /ask endpoints
├── ingestion/
│ ├── ingest.py # PDF extraction, chunking, embedding, storage
│ └── generate.py # Prompt construction and Gemini API call
├── data/ # Sample PDFs used for testing
├── chroma_db/ # Auto-created vector store (gitignored)
├── .env # API keys (gitignored)
└── .gitignore 

## API Endpoints

- `POST /upload` — accepts a PDF, runs the full ingestion pipeline, stores it in ChromaDB
- `POST /ask` — accepts a question, retrieves relevant chunks, returns a Gemini-generated answer grounded in the document

## Setup

1. Clone the repo and install dependencies:
```bash
   pip install -r requirements.txt
```
2. Create a `.env` file with your Gemini API key:
3. Run the FastAPI server:
```bash
   uvicorn app.main:app --reload
```
4. Upload a PDF via `/upload`, then ask questions via `/ask`.

## Known limitations

- Currently supports one question per `/ask` call, not batched queries
- Image-heavy or design-based PDFs with no extractable text (e.g., scanned/graphic PDFs) aren't supported — this would require OCR, which isn't implemented yet   
